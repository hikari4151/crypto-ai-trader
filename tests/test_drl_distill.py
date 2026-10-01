"""P2 蒸馏导出（Pine 体积门 × DRL 网络容量的长期解）专项测试。

背景（2026-10）：Pine 体积门（64KB，实测仅约 [16,16] 网络可装下）与 DRL
网络容量矛盾——大网络学习能力强但权重整段内联进 Pine 必然超限。方案：
训练大网络（teacher，软件内实盘）+ 离线行为蒸馏出小网络（student，[16,16]，
供 TradingView Pine 导出），模型文件同时保存两份，导出统一从蒸馏版走。

覆盖：
- 蒸馏保真度：student 与 teacher 动作一致率高、KL 收敛
- 蒸馏后 student 可导出 Pine（体积门内、含买卖点标记）
- teacher 直接导出失败（大网络被体积门拦截）→ 蒸馏后成功
- train_drl 自动蒸馏：distill_used=True、result 元数据、pine_code 含蒸馏标记
- 模型文件往返：to_dict/save/load 恢复 distill；export_pine_from_model 从蒸馏版导出
- 向后兼容：teacher 本可导出时不蒸馏（distill_used=False，行为不变）
- evolve 路径：save_agent_best 保留 distill，flat 文件导出从蒸馏版走
"""



def _demo_df(timeframe="5m", n=700, seed=7):
    from backtest.data_loader import generate_demo
    return generate_demo(timeframe=timeframe, n=n, seed=seed)


def _teacher_df_cfg(hidden=(64, 64), state_window=2, seed=1):
    """构造大网络 teacher + 演示数据 + 训练配置。"""
    from drl.agent import ACAgent
    df = _demo_df(n=700, seed=seed)
    sd = 13 * state_window + 2
    teacher = ACAgent(sd, 5, hidden=tuple(hidden), state_window=state_window, seed=seed)
    cfg = {"start_cash": 10000, "fee_rate": 0.001, "vol_penalty": 20,
           "slippage": 0.0005, "min_trade_zone": 0.05, "state_window": state_window}
    return teacher, df, cfg


# ============ 1. 蒸馏保真度 ============

def test_distill_fidelity_high():
    """student 与 teacher 在采样状态上的动作一致率应显著高于随机（20%）。"""
    from drl.distill import distill_agent, _sample_states, _softmax_proba
    teacher, df, cfg = _teacher_df_cfg()
    student = distill_agent(teacher, df, cfg, student_hidden=[16, 16],
                            n_rollouts=2, samples_per_rollout=200,
                            epochs=150, seed=42)
    states = _sample_states(teacher, df, cfg, n_rollouts=2,
                            samples_per_rollout=150, seed=7)
    tp = _softmax_proba(teacher, states)
    sp = student.actor.predict_proba(states, cache_for_backward=False)
    agreement = float((tp.argmax(axis=1) == sp.argmax(axis=1)).mean())
    assert agreement > 0.5, f"动作一致率 {agreement:.2%} 应显著高于随机 20%"
    assert student.state_dim == teacher.state_dim
    assert student.n_actions == teacher.n_actions


def test_distilled_student_exports_pine():
    """蒸馏 student 必须能导出 Pine（体积门内 + 买卖点标记 + 蒸馏标记）。"""
    from drl.distill import distill_agent
    from drl.pine_export import try_build_drl_pine
    teacher, df, cfg = _teacher_df_cfg()
    student = distill_agent(teacher, df, cfg, student_hidden=[16, 16],
                            n_rollouts=2, samples_per_rollout=150,
                            epochs=100, seed=1)
    code, note = try_build_drl_pine(student, model_name="rl_distilled")
    assert code and not note, f"蒸馏 student 应可导出 Pine, note={note!r}"
    assert "plotshape" in code, "必须含买卖点标记"
    assert "蒸馏版" in code, "导出头部应注明蒸馏来源"


# ============ 2. train_drl 自动蒸馏 ============

def test_train_drl_auto_distill_large_net():
    """大网络 [64,64]：直接导出必超体积门 → 自动蒸馏并成功导出 Pine。"""
    from drl.agent import train_drl
    df = _demo_df(n=900)
    cfg = {"episodes": 2, "hidden": [64, 64], "n_episodes": 2, "ppo_epochs": 2,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 5,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
           "distill": True, "distill_epochs": 150}
    res = train_drl(df, cfg)
    assert res.get("distill_used") is True, "大网络应自动蒸馏"
    assert res.get("pine_code"), "蒸馏后应能导出 Pine"
    assert not res.get("pine_note")
    assert "蒸馏版" in res["pine_code"]
    assert res["agent"].distill is not None, "蒸馏 student 应挂到 agent 上"
    assert res.get("distill_meta", {}).get("student_hidden") == [16, 16]


def test_train_drl_no_distill_when_exportable():
    """teacher [16,16] 本可导出：不蒸馏（distill_used=False），行为不变。"""
    from drl.agent import train_drl
    df = _demo_df(n=900)
    cfg = {"episodes": 2, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 5,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
           "distill": True}
    res = train_drl(df, cfg)
    assert res.get("distill_used") is False, "可导出的 teacher 不应蒸馏"
    assert res.get("pine_code") and not res.get("pine_note")
    assert "蒸馏版" not in res["pine_code"]
    assert res["agent"].distill is None


def test_train_drl_distill_disabled_keeps_note():
    """关闭蒸馏（distill=False）：大网络保持"装不进 Pine"的明确提示（不静默）。"""
    from drl.agent import train_drl
    df = _demo_df(n=900)
    cfg = {"episodes": 1, "hidden": [64, 64], "n_episodes": 1, "ppo_epochs": 1,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 5,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 1, "oos_min_bars": 80, "time_budget": 0,
           "distill": False}
    res = train_drl(df, cfg)
    assert res.get("distill_used") is False
    assert not res.get("pine_code")
    assert res.get("pine_note"), "关闭蒸馏时大网络应给明确原因"


# ============ 3. 模型文件往返与导出 ============

def test_model_file_roundtrip_with_distill(tmp_path):
    """to_dict/save/load 往返保留 distill；export_pine_from_model 从蒸馏版导出。"""
    from drl.agent import train_drl, ACAgent
    from drl.pine_export import export_pine_from_model
    df = _demo_df(n=900)
    cfg = {"episodes": 2, "hidden": [64, 64], "n_episodes": 2, "ppo_epochs": 2,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 5,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
           "distill": True, "distill_epochs": 150}
    res = train_drl(df, cfg)
    assert res.get("distill_used")

    mp = tmp_path / "model.json"
    res["agent"].save(str(mp))
    data = mp.read_text(encoding="utf-8")
    assert '"distill_agent"' in data, "模型文件应含蒸馏权重"
    assert '"distill_meta"' in data

    code, note = export_pine_from_model(mp, "rl_big")
    assert code and not note, "从模型文件导出应成功（走蒸馏版）"
    assert "蒸馏版" in code, "导出来源应标注蒸馏版"

    loaded = ACAgent.load(str(mp))
    assert loaded.distill is not None, "加载往返应恢复蒸馏 student"
    assert getattr(loaded.distill, "is_distilled", False)
    assert loaded.distill.actor.dims[1] <= 16, "student 应是小网络"


# ============ 4. evolve 路径 ============

def test_evolve_flat_file_exports_distilled(tmp_path):
    """save_agent_best 保留 distill；evolve flat 模型文件导出从蒸馏版走。"""
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from drl.distill import distill_agent
    from drl.pine_export import export_pine_from_model
    from tests.test_drl_optimizations import _MetaDB

    teacher, df, cfg = _teacher_df_cfg()
    student = distill_agent(teacher, df, cfg, student_hidden=[16, 16],
                            n_rollouts=2, samples_per_rollout=150,
                            epochs=100, seed=1)
    teacher.distill = student
    teacher.distill_meta = getattr(student, "distill_meta", None)

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng.zoo.save_agent_best(teacher, "strategy_drl", meta={"fitness": 0.1})
    flat = eng.zoo._flat_path("strategy_drl")
    assert flat.exists()
    data = flat.read_text(encoding="utf-8")
    assert '"distill_agent"' in data, "evolve flat 文件应含蒸馏权重"

    code, note = export_pine_from_model(flat, "rl_evolve")
    assert code and not note, "evolve flat 应从蒸馏版导出成功"
    assert "蒸馏版" in code
