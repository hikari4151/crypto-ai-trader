"""P1 时序网络升级（GRU 骨干）专项测试。

背景（2026-10）：MLP actor 只看"展平窗口向量"，各帧孤立；GRU 把 state_window
内的 K 线序列建模为时序，突破能力天花板。本套件验证：
- GRU 前向/反向（BPTT）数值正确性（梯度检查）
- GRUActor 接口与 MLP 兼容（ACAgent/train_batch 无感知）
- GRU 训练跑通且收敛
- GRU teacher → 蒸馏 MLP student → Pine 导出（Pine 无法重建 GRU 隐藏态）
- 序列化往返（to_dict/from_dict 恢复 GRU teacher + MLP student）
- rl_adaptive 部署端加载 GRU 模型并决策
- 向后兼容：默认 backbone="mlp" 完全走旧路径（行为不变）
"""

import numpy as np


def _demo_df(timeframe="5m", n=900, seed=7):
    from backtest.data_loader import generate_demo
    return generate_demo(timeframe=timeframe, n=n, seed=seed)


def _train_cfg(backbone="gru", episodes=2, state_window=4, distill=True):
    return {"episodes": episodes, "hidden": [16, 16], "n_episodes": 4,
            "ppo_epochs": 3, "mini_batch_size": 64, "vol_penalty": 20,
            "seed": 5, "start_cash": 10000, "fee_rate": 0.001,
            "slippage": 0.0005, "state_window": state_window,
            "oos_min_bars": 80, "time_budget": 0, "distill": distill,
            "distill_epochs": 150, "backbone": backbone, "gru_hidden": 16}


# ============ 1. GRU 数值正确性 ============

def test_gru_cell_bptt_gradient():
    """GRUCell BPTT 梯度与数值梯度一致（rel < 1e-4）。"""
    from drl.nnet_temporal import GRUCell, _GRU_W_NAMES, _GRU_B_NAMES
    cell = GRUCell(3, 4, seed=1, lr=0.0)
    x_seq = np.random.RandomState(0).randn(2, 5, 3).astype(float)
    cache, h = cell.forward(x_seq)
    dh = 2.0 * h
    gW, gb = cell.backward(cache, dh)
    eps = 1e-6
    for gi, name in enumerate(_GRU_W_NAMES):
        flat = getattr(cell, name).ravel()
        for k in range(min(2, flat.size)):
            orig = flat[k]
            flat[k] = orig + eps
            _, hp = cell.forward(x_seq)
            lp = (hp ** 2).sum()
            flat[k] = orig - eps
            _, hm = cell.forward(x_seq)
            lm = (hm ** 2).sum()
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = gW[gi].ravel()[k]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
            assert rel < 1e-4, f"{name}[{k}] num={num} ana={ana} rel={rel:.2e}"
    for gi, name in enumerate(_GRU_B_NAMES):
        flat = getattr(cell, name).ravel()
        for k in range(min(2, flat.size)):
            orig = flat[k]
            flat[k] = orig + eps
            _, hp = cell.forward(x_seq)
            lp = (hp ** 2).sum()
            flat[k] = orig - eps
            _, hm = cell.forward(x_seq)
            lm = (hm ** 2).sum()
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = gb[gi].ravel()[k]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
            assert rel < 1e-4, f"{name}[{k}] num={num} ana={ana} rel={rel:.2e}"


def test_gru_actor_full_chain_gradient():
    """GRUActor（GRU + MLP head）全链路梯度与数值梯度一致。"""
    from drl.nnet_temporal import GRUActor, _GRU_W_NAMES
    actor = GRUActor(54, 5, state_window=4, gru_hidden=8, head_hidden=(8, 8),
                     seed=1, lr=0.0)
    x = np.random.RandomState(0).randn(3, 54).astype(float)

    def loss_at(a, xx):
        return (a.forward(xx, cache_for_backward=False)[-1] ** 2).sum()

    acts = actor.forward(x, cache_for_backward=True)
    dout = 2.0 * acts[-1]
    gW, gb = actor.backward(acts, dout)
    eps = 1e-6
    for gi, name in enumerate(_GRU_W_NAMES):
        flat = getattr(actor.gru, name).ravel()
        for k in range(min(2, flat.size)):
            orig = flat[k]
            flat[k] = orig + eps
            lp = loss_at(actor, x)
            flat[k] = orig - eps
            lm = loss_at(actor, x)
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = gW[gi].ravel()[k]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
            assert rel < 1e-4, f"GRU {name}[{k}] rel={rel:.2e}"
    # head 梯度
    flat = actor.head.W[0].ravel()
    for k in range(min(3, flat.size)):
        orig = flat[k]
        flat[k] = orig + eps
        lp = loss_at(actor, x)
        flat[k] = orig - eps
        lm = loss_at(actor, x)
        flat[k] = orig
        num = (lp - lm) / (2 * eps)
        ana = gW[6].ravel()[k]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        assert rel < 1e-4, f"head W0[{k}] rel={rel:.2e}"


def test_gru_actor_interfaces_compatible():
    """GRUActor 的 predict_proba/predict_log_proba 与 MLP 同语义。"""
    from drl.nnet_temporal import GRUActor
    actor = GRUActor(54, 5, state_window=4, gru_hidden=8, head_hidden=(8, 8),
                     seed=1, lr=1e-3)
    x = np.random.RandomState(1).randn(1, 54).astype(float)
    p = actor.predict_proba(x, cache_for_backward=False)
    lp = actor.predict_log_proba(x, cache_for_backward=False)
    assert p.shape == (1, 5)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert np.allclose(lp, np.log(p), atol=1e-5)


def test_gru_training_converges():
    """GRU 能拟合固定 softmax 目标（训练收敛）。"""
    from drl.nnet_temporal import GRUActor
    actor = GRUActor(54, 5, state_window=4, gru_hidden=8, head_hidden=(8, 8),
                     seed=1, lr=5e-3)
    x = np.random.RandomState(2).randn(4, 54).astype(float)
    target = np.array([[0.7, 0.1, 0.1, 0.05, 0.05],
                       [0.05, 0.7, 0.1, 0.1, 0.05],
                       [0.05, 0.05, 0.7, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.6, 0.1]], dtype=float)
    for _ in range(200):
        acts = actor.forward(x, cache_for_backward=True)
        proba = actor.predict_proba(x, cache_for_backward=False)
        dout = (proba - target) / 4.0
        gW, gb = actor.backward(acts, dout)
        actor.apply_grad(gW, gb)
    final = actor.predict_proba(x, cache_for_backward=False)
    kl = float(np.mean(np.sum(target * np.log((target + 1e-12) / (final + 1e-12)), axis=-1)))
    assert kl < 0.05, f"GRU 应收敛，KL={kl:.4f}"


# ============ 2. GRU 训练端到端 ============

def test_train_drl_gru_backbone():
    """train_drl 用 backbone='gru' 训练：跑通 + actor 是 GRUActor + 自动蒸馏导出。"""
    from drl.agent import train_drl
    df = _demo_df()
    res = train_drl(df, _train_cfg())
    assert res["agent"].backbone == "gru"
    assert type(res["agent"].actor).__name__ == "GRUActor"
    assert res.get("distill_used") is True, "GRU 无法直接导出 Pine，应自动蒸馏"
    assert res.get("pine_code"), "蒸馏后应有 Pine 自动交易代码"
    assert "蒸馏版" in res["pine_code"]
    assert res["agent"].distill is not None
    assert res["agent"].distill.backbone == "mlp", "student 必须是 MLP（可导出）"
    assert res.get("distill_meta", {}).get("teacher_arch", {}).get("type") == "gru"


def test_gru_direct_pine_export_friendly_error():
    """GRU 模型直接导出 Pine：给出明确原因（蒸馏路径）而非 AttributeError。"""
    from drl.agent import ACAgent
    from drl.pine_export import try_build_drl_pine
    teacher = ACAgent(54, 5, hidden=(16, 16), state_window=4, seed=1,
                      backbone="gru", gru_hidden=16)
    code, note = try_build_drl_pine(teacher, model_name="g")
    assert code == ""
    assert "GRU" in note and "蒸馏" in note, f"应说明 GRU 需走蒸馏，note={note!r}"


def test_gru_model_file_roundtrip(tmp_path):
    """GRU 模型文件往返：to_dict/save/load 恢复 GRU teacher + MLP student。"""
    from drl.agent import train_drl, ACAgent
    df = _demo_df()
    res = train_drl(df, _train_cfg())
    mp = tmp_path / "gru_model.json"
    res["agent"].save(str(mp))
    data = mp.read_text(encoding="utf-8")
    assert '"backbone": "gru"' in data or "gru" in data
    loaded = ACAgent.load(str(mp))
    assert loaded.backbone == "gru"
    assert type(loaded.actor).__name__ == "GRUActor"
    assert loaded.distill is not None and loaded.distill.backbone == "mlp"


def test_rl_adaptive_loads_gru_model(tmp_path):
    """rl_adaptive 部署端加载 GRU 模型并产出决策（predict_proba 兼容）。"""
    from drl.agent import train_drl
    from strategies.rl_adaptive import make_rl_strategy
    df = _demo_df()
    res = train_drl(df, _train_cfg())
    mp = tmp_path / "gru_deploy.json"
    res["agent"].save(str(mp))
    st = make_rl_strategy("rl_gru_deploy", str(mp))
    assert st.reload_model() is True
    state = np.random.RandomState(3).randn(54).astype(float)
    proba = st._agent.actor.predict_proba(state.reshape(1, -1))[0]
    assert proba.shape == (5,)
    assert abs(float(proba.sum()) - 1.0) < 1e-5


# ============ 3. 向后兼容 ============

def test_default_backbone_mlp_unchanged():
    """默认 backbone='mlp'：actor 是 MLP，完全走旧路径（向后兼容）。"""
    from drl.agent import ACAgent
    agent = ACAgent(15, 5, hidden=(16, 16), state_window=1, seed=1)
    assert agent.backbone == "mlp"
    assert type(agent.actor).__name__ == "MLP"
    assert type(agent.critic).__name__ == "MLP"
    # to_dict 不带 gru 键
    d = agent.to_dict()
    assert d["backbone"] == "mlp"


def test_mlp_train_unchanged_with_distill():
    """MLP 训练（默认路径）行为不变：可导出时跳过蒸馏，Pine 无蒸馏标记。"""
    from drl.agent import train_drl
    df = _demo_df()
    cfg = _train_cfg(backbone="mlp", state_window=2, distill=True)
    cfg["hidden"] = [16, 16]
    res = train_drl(df, cfg)
    assert res["agent"].backbone == "mlp"
    assert res.get("distill_used") is False
    assert res.get("pine_code") and not res.get("pine_note")
    assert "蒸馏版" not in res["pine_code"]


# ============ 4. evolve 持续进化路径 ============

def test_evolve_train_cfg_has_backbone():
    """evolve 训练配置带 backbone（默认 mlp），默认不改变行为。"""
    from drl.evolve_engine import _strategy_drl_train_cfg
    cfg = _strategy_drl_train_cfg(episodes=8)
    assert cfg.get("backbone", "mlp") == "mlp"
    assert int(cfg.get("gru_hidden", 32)) >= 4


def test_training_identity_includes_backbone():
    """evolve 训练身份指纹把 backbone 纳入——GRU 与 MLP 产出不可互比回退。"""
    from drl.evolve_engine import _training_identity
    cfg_base = {"episodes": 8, "vol_penalty": 20.0, "oos_min_bars": 250,
                "multi_symbol": False, "multi_extra_names": []}
    id_mlp = _training_identity("strategy_drl", "BTC/USDT", "1h",
                                state_window=4, config={**cfg_base, "backbone": "mlp"})
    id_gru = _training_identity("strategy_drl", "BTC/USDT", "1h",
                                state_window=4, config={**cfg_base, "backbone": "gru"})
    assert id_mlp["config_fingerprint"] != id_gru["config_fingerprint"], \
        "GRU 与 MLP 的训练身份应不同（不可互比回退）"
    id_mlp2 = _training_identity("strategy_drl", "BTC/USDT", "1h",
                                 state_window=4, config={**cfg_base, "backbone": "mlp"})
    assert id_mlp["config_fingerprint"] == id_mlp2["config_fingerprint"]


def test_evolve_flat_exports_gru_distilled(tmp_path):
    """evolve 保存 GRU teacher（含蒸馏 student）到 flat → 导出走蒸馏版。"""
    from core.bus import EventBus
    from drl.agent import ACAgent
    from drl.distill import distill_agent
    from drl.evolve_engine import EvolveEngine
    from drl.pine_export import export_pine_from_model
    from tests.test_drl_optimizations import _MetaDB

    teacher = ACAgent(54, 5, hidden=(16, 16), state_window=4, seed=1,
                      backbone="gru", gru_hidden=16)
    cfg = {"start_cash": 10000, "fee_rate": 0.001, "vol_penalty": 20,
           "slippage": 0.0005, "min_trade_zone": 0.05, "state_window": 4}
    student = distill_agent(teacher, _demo_df(), cfg, n_rollouts=2,
                            samples_per_rollout=100, epochs=80, seed=1)
    teacher.distill = student
    teacher.distill_meta = student.distill_meta

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng.zoo.save_agent_best(teacher, "strategy_drl", meta={"fitness": 0.1})
    flat = eng.zoo._flat_path("strategy_drl")
    assert flat.exists()
    code, note = export_pine_from_model(flat, "rl_evolve")
    assert code and not note, "GRU 模型 flat 导出应走蒸馏版成功"
    assert "蒸馏版" in code
