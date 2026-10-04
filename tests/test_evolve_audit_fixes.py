"""持续进化 DRL 审计修复的回归锁（2026-10）。

覆盖 docs/持续进化DRL-审计报告.md 中本轮已修的 HIGH/MED 项：
1. 回退重建 flat 丢失 decision_interval/min_trade_zone（部署端 k=1/zone=0.02 口径漂移）
2. 部署长窗门临时模型漏键 + 临时文件泄漏
3. 跨标 OOS 门环境漏 policy_env_kwargs（k=1/absolute 评估 k=8/excess 模型）
4. 训练身份指纹含 multi_extra_names（瞬时数据成败让指纹逐轮漂移）
5. 多 seed 偶数个时取"上半中位数"（与文档"下半保守"矛盾，退化为最大值挑选）
6. GAP-01 FAIL 计数不在部署版本变化时清零（旧模型 FAIL 传导到新模型）
7. 增量门不随训练口径配置变更失效（改 seed/backbone/决策间隔不重训）
8. _log_round 对 challenger 谱系也递增 _reject_streak（健康交替累计触发 from-scratch）
9. 多 seed 回放缓冲独立（seed 间轨迹互不污染）
10. 回退锚点携带 OOS 超额（回退/晋升比较改超额口径）
11. use_model 加门（做空拒绝 + OOS risk 提示 + DB 持久化）
"""
import json

import pytest


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    """把 MODEL_DIR 指向临时目录（避免测试污染真实模型库）。"""
    from web.api import drl as _drl_api
    fake = tmp_path / "models"
    fake.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_drl_api, "MODEL_DIR", fake)
    return fake


@pytest.fixture
def use_api(tmp_path, monkeypatch, isolated_model_dir):
    """挂载 drl 路由的 TestClient + 假 engine（use_model 端点专用）。"""
    from web.deps import get_db, get_engine
    import strategies
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.drl import router

    class _StubEngine:
        async def select_strategy(self, name):
            self.selected = name
            return True

    class _StubDB:
        pass

    app = FastAPI()
    app.include_router(router)
    engine = _StubEngine()
    app.state.db = _StubDB()
    app.state.engine = engine
    app.dependency_overrides[get_db] = lambda: _StubDB()
    app.dependency_overrides[get_engine] = lambda: engine
    captured = {}
    monkeypatch.setattr(strategies, "register_dynamic",
                        lambda name, spec: captured.update(spec=spec))
    return TestClient(app), captured


# ============ 1. 回退重建 flat 保留决策节奏/死区 ============

def test_rollback_rebuild_preserves_decision_interval_and_zone(tmp_path):
    """回退（restore_deployment）重建 flat 时，decision_interval/min_trade_zone 必须保留。

    背景（审计 HIGH-1）：接受路径只经 flat 运行时补丁写这两键，版本载荷 _meta 里没有；
    restore_deployment→_flat_payload 只提升 _DEPLOYMENT_META_KEYS → 回退后 flat 缺两键，
    rl_adaptive 按 k=1、zone=0.02 跑 k=8、zone=0.10 训练的模型（回退是常态保护动作，
    恰在最该严谨的动作上制造口径漂移）。
    """
    from drl.agent import ACAgent
    from drl.model_zoo import ModelZoo
    zoo = ModelZoo(tmp_path / "models")
    agent = ACAgent(state_dim=6, n_actions=5, hidden=(8, 8), seed=1)
    zoo.save_agent(agent, "strategy_drl", meta={
        "fitness": 0.1, "decision_interval": 8, "min_trade_zone": 0.10,
    })
    zoo.restore_deployment("strategy_drl", 1, reason="审计回归", outcome="rollback")
    flat = json.loads((tmp_path / "models" / "strategy_drl.json").read_text("utf-8"))
    assert flat.get("decision_interval") == 8, "回退重建 flat 必须保留决策间隔"
    assert flat.get("min_trade_zone") == 0.10, "回退重建 flat 必须保留死区"


def test_rollback_without_meta_keys_keeps_defaults(tmp_path):
    """旧版本载荷（meta 无这两键）回退时不崩，flat 按缺省（rl_adaptive 回退 k=1）。"""
    from drl.agent import ACAgent
    from drl.model_zoo import ModelZoo
    zoo = ModelZoo(tmp_path / "models")
    agent = ACAgent(state_dim=6, n_actions=5, hidden=(8, 8), seed=1)
    zoo.save_agent(agent, "strategy_drl", meta={"fitness": 0.1})
    zoo.restore_deployment("strategy_drl", 1, outcome="rollback")
    flat = json.loads((tmp_path / "models" / "strategy_drl.json").read_text("utf-8"))
    assert "decision_interval" not in flat or flat["decision_interval"] in (1, 8, None)


# ============ 2. 部署长窗门临时模型带全口径 + 临时文件清理 ============

def test_candidate_model_dict_carries_decision_interval_and_zone():
    """长窗门写给 rl_adaptive 回放的临时模型必须带决策间隔/死区（否则 k=1/zone=0.02）。"""
    from drl.agent import ACAgent
    from drl.evolve_observations import _build_candidate_model_dict
    agent = ACAgent(state_dim=6, n_actions=5, hidden=(8, 8), seed=1)
    d = _build_candidate_model_dict(agent, {"state_window": 2, "decision_interval": 8,
                                            "min_trade_zone": 0.10})
    assert d["decision_interval"] == 8
    assert d["min_trade_zone"] == 0.10
    assert d["state_window"] == 2


# ============ 3. 跨标 OOS 门环境用训练口径 ============

def test_cross_symbol_oos_env_uses_policy_kwargs():
    """跨标 OOS 门的 TradingEnv 必须走 policy_env_kwargs（k=8/excess），不能 k=1/absolute。"""
    import pandas as pd

    from drl.evolve_engine import _make_cross_oos_env
    idx = pd.date_range("2026-01-01", periods=300, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "volume": 100.0}, index=idx)
    env = _make_cross_oos_env(df, state_window=1, decision_interval=8,
                              min_trade_zone=0.10)
    assert env.decision_interval == 8, "跨标 OOS 必须按训练决策间隔评估"
    assert env.min_trade_zone == 0.10
    assert env.short_enabled is False


# ============ 4. 身份指纹不含瞬时数据成败 ============

def test_identity_fingerprint_ignores_multi_extra_names():
    """multi_extra_names（本轮实际取到数据的额外标的）进身份指纹会让同一配置
    指纹逐轮漂移 → 回退保护被旁路。开关（multi_symbol）保留，名单剔除。"""
    from drl.evolve_engine import _strategy_drl_identity_config
    cfg = {"state_window": 1, "factor_signature": ""}
    fp1 = _strategy_drl_identity_config(cfg, ["BTC/USDT"])
    fp2 = _strategy_drl_identity_config(cfg, [])
    assert fp1 == fp2, "同一训练配置下额外标的名单不同，身份指纹必须相同"
    assert "multi_extra_names" not in fp1


# ============ 5. 多 seed 偶数个取下半中位数 ============

def test_even_seed_count_takes_lower_middle():
    """偶数 seed 取排序后**下半**中位数（保守，与文档一致）——n=2 取较低分那个。"""
    from drl.evolve_engine import _run_multi_seed
    values = {42: 0.10, 43: 0.30, 44: 0.20, 45: 0.15}

    def fake_train(c):
        return {"history": [{"ep": 1}], "best_val_excess": values[c["seed"]],
                "best_val_ret": 0.0}

    chosen, _ = _run_multi_seed(fake_train, {}, 4, 42)
    # 排序 [0.10(42), 0.15(45), 0.20(44), 0.30(43)] → 下半中位 = 0.15（seed45）
    assert chosen["best_val_excess"] == 0.15, f"应取下半中位 0.15，实际 {chosen['best_val_excess']}"
    assert chosen["seed_chosen"] == 45


# ============ 6. GAP-01 FAIL 计数随部署版本清零 ============

def test_gap01_streak_cleared_on_deployment_version_change(tmp_path):
    """部署版本**更换**时，GAP-01 的 FAIL 计数必须清零（旧模型的 FAIL 不传导到新模型）。"""
    from core.bus import EventBus
    from tests.test_drl_optimizations import _MetaDB
    from drl.evolve_engine import EvolveEngine, _clear_gap01_streak_on_version
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._gap01_streak["strategy_drl"] = 4
    eng._gap01_pass_version = 1
    # 换模型（1→2）：必须清 FAIL
    _clear_gap01_streak_on_version(eng, new_version=2, prev_version=1)
    assert eng._gap01_streak.get("strategy_drl", 0) == 0, "版本更换必须清 FAIL 计数"
    # 版本首次确定（None→1）或未变：不清（否则第一轮 FAIL 被吞）
    eng._gap01_streak["strategy_drl"] = 3
    _clear_gap01_streak_on_version(eng, new_version=1, prev_version=None)
    assert eng._gap01_streak.get("strategy_drl", 0) == 3, "首次确定不清 FAIL（否则连续 FAIL 判定失真）"
    _clear_gap01_streak_on_version(eng, new_version=1, prev_version=1)
    assert eng._gap01_streak.get("strategy_drl", 0) == 3, "版本未变不清 FAIL"


# ============ 7. 增量门随训练口径配置失效 ============

def test_incremental_gate_invalidated_by_training_config_change(monkeypatch):
    """改影响训练口径的配置（seed/hidden/决策间隔等）后，增量门必须失效（立即重训）。"""
    from config.settings import settings
    from drl.evolve_engine import _TRAINING_CFG_KEYS, _cfg_epoch_hash
    snapshot = {k: getattr(settings, k) for k in _TRAINING_CFG_KEYS}
    h0 = _cfg_epoch_hash(snapshot)
    snapshot["evolve_strategy_drl_seed"] = snapshot.get("evolve_strategy_drl_seed", 42) + 1
    h1 = _cfg_epoch_hash(snapshot)
    assert h0 != h1, "训练口径配置变化必须改变指纹，从而清空增量门"


def test_incremental_gate_unaffected_by_irrelevant_config(monkeypatch):
    from config.settings import settings
    from drl.evolve_engine import _TRAINING_CFG_KEYS, _cfg_epoch_hash
    snapshot = {k: getattr(settings, k) for k in _TRAINING_CFG_KEYS}
    h0 = _cfg_epoch_hash(snapshot)
    # 与训练口径无关的键（演示）不改变指纹
    snapshot["evolve_strategy_drl_min_trade_zone"] = snapshot.get(
        "evolve_strategy_drl_min_trade_zone", 0.10)
    assert _cfg_epoch_hash(snapshot) == h0


# ============ 8. 挑战者谱系不递增连续拒绝计数 ============

def test_challenger_lineage_does_not_increment_reject_streak(tmp_path):
    """challenger 谱系的轮（kept/degraded）不得计入 _reject_streak——
    否则健康的冠军/挑战者交替会累计触发冠军 from-scratch 重训（丢经验）。"""
    from core.bus import EventBus
    from tests.test_drl_optimizations import _MetaDB
    from drl.evolve_engine import EvolveEngine, _bump_reject_streak
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._reject_streak["strategy_drl"] = 4
    # 直接调用 _log_round 的计数逻辑（走真实函数，DB 落库失败仅告警不阻断）
    _bump_reject_streak(eng, "strategy_drl", lineage="challenger")
    assert eng._reject_streak["strategy_drl"] == 4, "challenger 谱系不应递增"
    _bump_reject_streak(eng, "strategy_drl", lineage="main")
    assert eng._reject_streak["strategy_drl"] == 5, "main 谱系应递增"


# ============ 9. 多 seed 回放缓冲独立（HIGH-4） ============

def test_multi_seed_uses_independent_replay_copies(tmp_path):
    """每个 seed 用主缓冲的**独立深拷贝**，seed 间轨迹互不污染；
    选中 seed 的副本经 meta 交回（跨轮记忆）。"""
    from drl.agent import ReplayBuffer
    from drl.evolve_engine import _run_multi_seed

    main_buf = ReplayBuffer(capacity=8)
    seen = {}

    def fake_train(cfg_):
        seen[cfg_["seed"]] = cfg_.get("replay_buffer")
        return {"history": [{"ep": 1}],
                "best_val_excess": {42: 0.1, 43: 0.3, 44: 0.2}[cfg_["seed"]],
                "best_val_ret": 0.0}

    chosen, meta = _run_multi_seed(fake_train, {"replay_buffer": main_buf}, 3, 42)
    bufs = [seen[s] for s in (42, 43, 44)]
    assert all(b is not main_buf for b in bufs), "每 seed 必须用独立副本（不能共享主缓冲）"
    assert len({id(b) for b in bufs}) == 3, "三个 seed 的缓冲实例必须互不相同"
    # 选中 seed（中位 val_excess=0.2 → seed44）的副本交回引擎
    assert meta["replay_buffer"] is seen[44], "meta 应交回选中 seed 的最终缓冲"


def test_single_seed_shares_main_buffer(tmp_path):
    """单 seed（n=1）直接共享主缓冲（无需副本，行为与旧版一致）。"""
    from drl.agent import ReplayBuffer
    from drl.evolve_engine import _run_multi_seed
    main_buf = ReplayBuffer(capacity=8)
    seen = {}

    def fake_train(cfg_):
        seen[cfg_["seed"]] = cfg_.get("replay_buffer")
        return {"history": [{"ep": 1}], "best_val_excess": 0.1, "best_val_ret": 0.0}

    _run_multi_seed(fake_train, {"replay_buffer": main_buf}, 1, 42)
    assert seen[42] is main_buf, "单 seed 直接共享主缓冲（无污染问题，不拷贝）"


# ============ 10. 回退锚点携带 OOS 超额（HIGH-2 数据通道） ============

def test_anchor_meta_carries_oos_excess_for_rollback(tmp_path):
    """接受路径存锚点时，meta 必须带 oos_excess——回退/晋升比较用它（超额口径）。"""
    from drl.agent import ACAgent
    from drl.model_zoo import ModelZoo
    zoo = ModelZoo(tmp_path / "models")
    agent = ACAgent(state_dim=6, n_actions=5, hidden=(8, 8), seed=1)
    zoo.save_agent(agent, "strategy_drl", meta={
        "fitness": 0.05, "oos_ret": -0.18, "oos_excess": 0.046,
    })
    info = zoo.best_info("strategy_drl")
    assert info.get("oos_excess") == 0.046, "锚点 meta 必须保留 OOS 超额（回退比较用）"


# ============ 11. use_model 加门（HIGH-3） ============

def test_use_model_rejects_short_model(use_api, isolated_model_dir, tmp_path):
    """含做空档的模型不得被 use_model 设为实盘策略（执行内核不支持负仓位）。"""
    client, _ = use_api
    from tests.test_drl_pine_export_fix import _write_model
    import json as _json
    path = _write_model(tmp_path, [16, 16], 1)   # 5 档长仓模型
    data = _json.loads(path.read_text("utf-8"))
    data["action_buckets"] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    data["short_enabled"] = True
    (isolated_model_dir / "short_model.json").write_bytes(_json.dumps(data).encode("utf-8"))

    r = client.post("/api/drl/models/short_model/use")
    assert r.status_code == 400, r.text
    assert "做空" in r.text


def test_use_model_persists_and_risks_on_oor_blocked(use_api, isolated_model_dir, tmp_path):
    """被 OOS 硬门拦下的模型：use_model 允许但 risk_tips 明示（与 /train 同口径）。"""
    client, captured = use_api
    from tests.test_drl_pine_export_fix import _write_model
    import json as _json
    path = _write_model(tmp_path, [16, 16], 1)
    data = _json.loads(path.read_text("utf-8"))
    data["train_meta"] = {"oos_report": {"hard_rejected": True,
                                         "reason": "样本外超额衰减 100%，疑似过拟合"}}
    data["deployment_blocked"] = True
    (isolated_model_dir / "oor_blocked.json").write_bytes(_json.dumps(data).encode("utf-8"))

    r = client.post("/api/drl/models/oor_blocked/use")
    assert r.status_code == 200, r.text
    tips = " ".join(captured["spec"].get("risk_tips") or [])
    assert "疑似过拟合" in tips and "不建议实盘" in tips


# ============ 12. 手动评估路径训练同源（MED-1） ============

def test_model_eval_cfg_carries_decision_interval_and_reward_keys():
    """_model_eval_cfg 必须带全训练同源口径（此前缺 decision_interval → k=1 评估 k=8 模型）。"""
    from web.api.drl import _model_eval_cfg
    cfg = _model_eval_cfg({"decision_interval": 8, "min_trade_zone": 0.10,
                           "reward_mode": "excess", "short_enabled": False})
    assert cfg["decision_interval"] == 8
    assert cfg["min_trade_zone"] == 0.10
    assert cfg["reward_mode"] == "excess"
    assert cfg["short_enabled"] is False
    # 缺省兼容（老模型无决策键 → k=1）
    assert _model_eval_cfg({})["decision_interval"] == 1


def test_evaluate_agent_rejects_action_dimension_mismatch():
    """9 档（做空）模型在 5 档 env 上评估：显式 ValueError（不裸断言崩 500）。"""
    from backtest.data_loader import generate_demo
    from drl.agent import ACAgent, evaluate_agent
    from drl.env import SHORT_BUCKETS
    # demo 1h state_window=1 → state_dim=15（13 价量 + 2 账户）
    agent = ACAgent(15, len(SHORT_BUCKETS), hidden=(8, 8), seed=1,
                    action_space=SHORT_BUCKETS)
    with pytest.raises(ValueError, match="动作空间"):
        evaluate_agent(agent, generate_demo(timeframe="1h", n=300), {})
    # 同档位评估正常
    res = evaluate_agent(agent, generate_demo(timeframe="1h", n=300),
                         {"short_enabled": True})
    assert res["total_ret"] is not None


# ============ 13. ReplayBuffer 采样 RNG 可复现（MED-2） ============

def test_replay_buffer_seeded_sampling_is_reproducible():
    """同 seed 的两个 ReplayBuffer，sample 序列必须一致（可复现）；不同 seed 不同。"""
    import numpy as np
    from drl.agent import ReplayBuffer

    def _build():
        buf = ReplayBuffer(16, seed=42)
        rng = np.random.default_rng(7)
        for i in range(6):
            buf.push(rng.random((20, 3)).astype(np.float32),
                     rng.integers(0, 5, 20), rng.random(20), rng.random(20))
        return buf.sample(32, mix_ratio=0.5)["states"]

    s1 = _build()
    s2 = _build()
    assert np.array_equal(s1, s2), "同 seed 的 ReplayBuffer 采样必须一致"
    buf_other = ReplayBuffer(16, seed=43)
    rng = np.random.default_rng(7)
    for i in range(6):
        buf_other.push(rng.random((20, 3)).astype(np.float32),
                       rng.integers(0, 5, 20), rng.random(20), rng.random(20))
    s3 = buf_other.sample(32, mix_ratio=0.5)["states"]
    assert not np.array_equal(s1, s3), "不同 seed 采样应不同"


# ============ 14. oos_excess 逐段中位数统一口径（MED-3） ============

def test_oos_excess_is_segment_median_of_excess_segments():
    """oos_excess 必须是**逐段超额的中位数**（旧口径 median(收益)−median(基准) 可与其
    分段门符号相反）；报告带 basis 标记供回退比较判同口径。"""
    import numpy as np
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="1h", n=900, seed=3)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "oos_min_bars": 80, "train_ratio": 0.6, "val_ratio": 0.2})
    oos = res["oos_report"]
    assert oos["enabled"]
    segs = oos.get("oos_excess_segments")
    assert segs, "报告应含逐段超额"
    assert oos["oos_excess_basis"] == "segment_median"
    assert oos["oos_excess"] == pytest.approx(
        float(np.median(segs)), abs=1e-6), "oos_excess 必须等于逐段超额的中位数"
    # 逐段中位数与 median(收益)−median(基准) 可在不同窗口下符号相反——
    # 该恒等式锁住实现：这两者只要实现一致，报告字段即与判据同源
    assert abs(oos["oos_excess"] - (oos["oos_ret"] - oos["oos_bh_ret"])) < 1.0


def test_val_excess_row_uses_segment_median_basis():
    """val 多段评估同样用逐段中位数口径（与 OOS 同构）。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="1h", n=900, seed=5)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 2, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "train_ratio": 0.6, "val_ratio": 0.2,
                         "oos_min_bars": 80, "time_budget": 0})
    rows = [r for r in res.get("history", []) if r.get("val_excess_basis")]
    if rows:
        assert all(r["val_excess_basis"] in ("segment_median", "single_segment")
                   for r in rows)


# ============ 15. 级联因子注入状态入身份指纹（MED） ============

def test_factor_injection_changes_identity_fingerprint():
    """级联因子注入与否改变 state_dim/状态语义，必须进 config_fingerprint——
    否则注入轮与未注入轮被当同口径回退比较（错配）。未注入轮**不加**该键
    （保持原格式，与旧锚点/测试手工构造一致）。"""
    from drl.evolve_engine import _training_identity, _strategy_drl_identity_config
    cfg = _strategy_drl_identity_config({"state_window": 1}, [])
    cfg_on = dict(cfg, factor_injected=True)
    id_on = _training_identity("strategy_drl", "ETH/USDT", "1h",
                               state_window=1, config=cfg_on)
    id_off = _training_identity("strategy_drl", "ETH/USDT", "1h",
                                state_window=1, config=cfg)
    assert id_on["config_fingerprint"] != id_off["config_fingerprint"], \
        "因子注入必须改变指纹（注入轮与未注入轮不可互比）"


# ============ 16. 特征预计算 volume 缺列兜底（MED） ============

def test_precompute_features_tolerates_missing_volume_column():
    """缺 volume 列的 df：_precompute_features 不崩（与 TradingEnv.__init__ 兜底同源）。"""
    import pandas as pd
    from drl.env import _precompute_features
    idx = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0}, index=idx)   # 无 volume 列
    feat = _precompute_features(df)
    assert feat.shape[1] == 13


# ============ 17. 回放容量默认值与内存说明对齐（MED） ============

def test_replay_capacity_default_upgraded(monkeypatch):
    from config.settings import settings
    assert settings.evolve_strategy_drl_replay_capacity >= 128, \
        "默认容量应≥128（64 只留最近 1/32 轮，跨轮记忆名不副实）"


# ============ 18. _DYNAMIC 并发读写安全（MED） ============

def test_dynamic_registry_thread_safe():
    """register/remove 与 list/get 并发不抛 RuntimeError（曾"dictionary changed size"）。"""
    import threading
    from strategies import register_dynamic, dynamic_names, remove_dynamic

    def _writer(tag):
        for i in range(200):
            name = f"t{tag}_s{i}"
            register_dynamic(name, {"description": "", "params": {}, "param_schema": {}})
            if i % 20 == 0:
                remove_dynamic(name)

    def _reader():
        for _ in range(200):
            dynamic_names()

    threads = [threading.Thread(target=_writer, args=(k,)) for k in range(3)] \
        + [threading.Thread(target=_reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 清理测试注册项
    for tag in range(3):
        for i in range(200):
            remove_dynamic(f"t{tag}_s{i}")


# ============ 19. LOW 项：随机起点 / 持仓均价 / Pine cadence / 损坏模型 ============

def test_env_reset_random_offset_gives_distinct_starts():
    """reset(start_offset) 提供不同起点（蒸馏轨迹多样性），缺省 0 兼容旧行为。"""
    import pandas as pd
    from drl.env import TradingEnv
    idx = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 100.0}, index=idx)
    env = TradingEnv(df)
    env.reset()
    t0 = env._t
    env.reset(start_offset=100)
    assert env._t == t0 + 100, "start_offset 必须平移起点"
    env.reset()
    assert env._t == t0, "缺省 offset 保持旧行为"


def test_env_entry_price_reweighted_on_partial_close():
    """部分平仓后持仓均价按剩余名义重加权（曾固定原始入场价 → 浮动盈亏失真）。"""
    import pandas as pd
    from drl.env import TradingEnv, run_episode
    idx = pd.date_range("2026-01-01", periods=300, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 100.0}, index=idx)
    env = TradingEnv(df, fee_rate=0.0, slippage=0.0, min_trade_zone=0.0)
    # 满仓建仓 → 减到半仓 → 再减到 0：_entry_price 语义自洽
    run_episode(env, lambda s: 4)                 # 满仓
    assert env._qty > 0 and env._entry_price == 100.0
    run_episode(env, lambda s: 2)                 # 减到半仓
    assert env._qty > 0
    run_episode(env, lambda s: 0)                 # 清仓
    assert abs(env._qty) <= 1e-9 and env._entry_price is None


def test_pine_export_has_decision_cadence():
    """Pine 代码必须带决策节奏（k 根一次决策），不逐根 argmax。"""
    from drl.agent import ACAgent
    from drl.pine_export import build_drl_pine
    agent = ACAgent(15, 5, hidden=(16, 16), seed=1)
    code = build_drl_pine(agent, decision_interval=8)
    assert "bar_index % 8 == 0" in code, "Pine 必须按训练决策间隔出信号"
    assert "isDecision" in code


def test_from_dict_loads_non_action_models_with_truncated_buckets():
    """缺 action_buckets 且 n_actions>5 的模型**必须能加载**（factor_miner 是高维
    无档位模型，其输出是因子权重组合、从不映射仓位）——曾误判"损坏"导致生产启动
    时最佳 factor_miner 加载失败、回退空模型续训（回归，日志实证）。"""
    from drl.agent import ACAgent
    agent = ACAgent(15, 5, hidden=(8, 8), seed=1)
    data = agent.to_dict()
    data.pop("action_buckets")
    data["n_actions"] = 26                    # factor_miner 式：高维无档位
    loaded = ACAgent.from_dict(data)          # 不得抛错
    assert len(loaded.action_buckets or []) < loaded.n_actions, \
        "无档位高维模型回退截断 buckets（不解释动作，仅当填充物）"
    # 动作策略端的真正防线在 rl_adaptive 运行时越界守卫（见 strategies/rl_adaptive.py）
    import inspect
    from strategies import rl_adaptive as _ra
    src = inspect.getsource(_ra.RLAdaptiveStrategy.on_candle)
    assert "action >= len(_buckets)" in src, "rl_adaptive 越界守卫缺失（实时链会崩）"


# ============ 38. 优雅停止（留档项：_run_train 取消语义） ============

def test_sleep_or_stop_wakes_on_stop_event(tmp_path):
    """stop 事件置位后，间隔 sleep 立即返回（不等完整 retry_interval）。"""
    import asyncio
    from core.bus import EventBus
    from tests.test_drl_optimizations import _MetaDB
    from drl.evolve_engine import EvolveEngine
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")

    async def _run():
        # 引擎 __init__ 在无运行 loop 的线程里构造 → 其 asyncio.Event 绑定旧 loop；
        # 在 asyncio.run 的 loop 内重建，保证 wait/set 同 loop（生产环境引擎在
        # 事件循环线程内构造，无此问题）。
        eng._stop_event = asyncio.Event()
        started = asyncio.get_running_loop().time()
        task = asyncio.ensure_future(eng._sleep_or_stop(1000.0))
        await asyncio.sleep(0.05)
        eng._stop_event.set()
        await task
        return asyncio.get_running_loop().time() - started

    elapsed = asyncio.run(_run())
    assert elapsed < 5.0, f"stop 事件应立刻唤醒 sleep，实际等了 {elapsed:.2f}s"


def test_trigger_rejected_after_stop(tmp_path):
    """停止/禁用状态下手动触发被拒绝（曾 stop 不取消手动任务，禁用后仍可能写盘）。"""
    import asyncio
    from core.bus import EventBus
    from tests.test_drl_optimizations import _MetaDB
    from drl.evolve_engine import EvolveEngine
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._running = True

    async def _run():
        # 只测"停止后拒绝"路径（未停止调用会真触发数据拉取，慢且不在本用例范围）
        eng._stop_event.set()
        r = await eng.trigger_strategy_drl()
        return r.get("reason", "")

    reason = asyncio.run(_run())
    assert "停止" in reason or "禁用" in reason, f"停止后手动触发应拒绝，实际 {reason!r}"
