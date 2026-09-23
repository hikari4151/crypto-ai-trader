"""TDD：strategy_drl / meta_controller 连续拒绝 from-scratch 重置（对齐 factor_miner）。

背景（诊断报告 §7 建议第 2 条）：factor_miner 已有
evolve_factor_miner_reset_after_rejects（连续 OOS 被拒达阈值→下一轮强制从零重训，
打破"永久连拒"）；strategy_drl / meta_controller 缺失同款机制，锚点锁死时只能
无限续训。本改动补上，并保证：
- 策略 DRL 只在冠军轮触发（挑战者轮有独立训练基线，不清除）；
- 触发后 base_agent=None（from-scratch），未触发保持续训起点。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine, _should_from_scratch_reset
from tests.test_drl_optimizations import _MetaDB


def _agent():
    return ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)


def test_should_from_scratch_reset_pure():
    """阈值语义：0=关闭；未达阈值不触发；达到/超过触发。"""
    assert _should_from_scratch_reset(50, 0) is False
    assert _should_from_scratch_reset(4, 5) is False
    assert _should_from_scratch_reset(5, 5) is True
    assert _should_from_scratch_reset(10, 5) is True


def _fake_result(oos_enabled=False):
    return {
        "agent": _agent(),
        "history": [{"best_ret": 0.01}],
        "best_ret": 0.01,
        "oos_report": {"enabled": oos_enabled, "oos_ret": 0.01} if oos_enabled else {"enabled": False},
        "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
        "factor_composite": None, "pine_code": "", "pine_note": "",
    }


def _make_strategy_engine(tmp_path, monkeypatch):
    from backtest.data_loader import generate_demo
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    fake_best = _agent()
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_best)
    return eng, fake_best


@pytest.mark.asyncio
async def test_strategy_champion_round_resets_to_scratch_on_reject_streak(tmp_path, monkeypatch):
    """冠军轮连续拒绝达阈值 → from-scratch（base_agent=None）。"""
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 10
    eng._strategy_drl_status["_pop_round"] = 0  # 冠军轮
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is None, "冠军轮达阈值应 from-scratch 重训"


@pytest.mark.asyncio
async def test_strategy_champion_round_keeps_best_below_threshold(tmp_path, monkeypatch):
    """未达阈值 → 继续从 best 续训（base_agent 非 None）。"""
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 2
    eng._strategy_drl_status["_pop_round"] = 0
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is fake_best, "未达阈值应续训 best"


@pytest.mark.asyncio
async def test_strategy_challenger_round_ignores_reset(tmp_path, monkeypatch):
    """挑战者轮即使达阈值也不清除基线（它有独立训练基线机制）。"""
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 10
    eng._strategy_drl_status["_pop_round"] = 1  # 挑战者轮
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is fake_best, "挑战者轮不应被 from-scratch 重置"


@pytest.mark.asyncio
async def test_meta_round_resets_to_scratch_on_reject_streak(tmp_path, monkeypatch):
    """meta_controller 连续拒绝达阈值 → from-scratch（base_agent=None）。"""
    from config.settings import settings
    from strategies import register_dynamic, remove_dynamic
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    fake_best = _agent()
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_best)
    register_dynamic("meta_controller", {
        "name": "meta_controller", "executor": "meta_controller",
        "params": {"sub_strategies": "dual_ma,grid", "mode": "drl"},
    })
    monkeypatch.setattr(settings, "evolve_meta_reset_after_rejects", 5, raising=False)
    eng._reject_streak["meta_controller"] = 10
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return {"history": [{"best_ret": 0.01}], "agent": _agent()}
    monkeypatch.setattr(ev, "train_meta_controller", _fake_train)
    try:
        await eng._train_meta_controller_once()
    finally:
        remove_dynamic("meta_controller")
    assert captured.get("base_agent") is None, "meta 达阈值应 from-scratch 重训"


@pytest.mark.asyncio
async def test_meta_round_keeps_best_below_threshold(tmp_path, monkeypatch):
    """meta 未达阈值 → 从 best 续训。"""
    from config.settings import settings
    from strategies import register_dynamic, remove_dynamic
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    fake_best = _agent()
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_best)
    register_dynamic("meta_controller", {
        "name": "meta_controller", "executor": "meta_controller",
        "params": {"sub_strategies": "dual_ma,grid", "mode": "drl"},
    })
    monkeypatch.setattr(settings, "evolve_meta_reset_after_rejects", 5, raising=False)
    eng._reject_streak["meta_controller"] = 2
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return {"history": [{"best_ret": 0.01}], "agent": _agent()}
    monkeypatch.setattr(ev, "train_meta_controller", _fake_train)
    try:
        await eng._train_meta_controller_once()
    finally:
        remove_dynamic("meta_controller")
    assert captured.get("base_agent") is fake_best, "meta 未达阈值应续训 best"