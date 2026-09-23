"""TDD：长窗回放观测（30/90 天）纳入进化引擎（§16.5 遗留升级）。

背景（2026-09-21）：v44 短窗 stable PASS/滑点鲁棒，但 30 天 -26.6% / 90 天
-67.8%——"近期窗口效应"混进候选。升级：每轮训练后对部署模型做 30/90 天
真实数据引擎撮合回放，观测 verdict（两窗收益均正才 PASS），写
models/longwindow_obs.json + status 字段，杜绝长窗亏损模型冒充候选。
"""
import pytest

from drl.evolve_engine import _longwindow_verdict


def test_longwindow_verdict_pass_when_both_positive():
    out = _longwindow_verdict(ret_30d=0.025, ret_90d=0.041, data_ok=True)
    assert out["verdict"] == "PASS"


def test_longwindow_verdict_fail_when_either_negative():
    out = _longwindow_verdict(ret_30d=-0.266, ret_90d=0.041, data_ok=True)
    assert out["verdict"] == "FAIL"
    out2 = _longwindow_verdict(ret_30d=0.025, ret_90d=-0.678, data_ok=True)
    assert out2["verdict"] == "FAIL"


def test_longwindow_verdict_inconclusive_when_data_insufficient():
    out = _longwindow_verdict(ret_30d=None, ret_90d=None, data_ok=False)
    assert out["verdict"] == "INCONCLUSIVE"


def test_longwindow_verdict_rejects_zero_or_negative():
    out = _longwindow_verdict(ret_30d=0.0, ret_90d=0.001, data_ok=True)
    assert out["verdict"] == "FAIL"


def test_longwindow_every_n_default_five():
    """长窗观测默认每 5 轮一次（避免 30/90 天回放拖慢每轮节奏——实测训练仅 9.5s/轮，
    瓶颈在长窗回放，降频可大幅提升进化节奏）。"""
    from config.settings import settings
    assert settings.evolve_strategy_drl_longwindow_every_n == 5


def test_longwindow_every_n_can_be_one():
    """覆盖为 1（每轮观测）兼容旧行为。"""
    from config.settings import settings
    settings.evolve_strategy_drl_longwindow_every_n = 1
    try:
        assert settings.evolve_strategy_drl_longwindow_every_n == 1
    finally:
        settings.evolve_strategy_drl_longwindow_every_n = 5


def test_deploy_longwindow_requires_default_on():
    """部署长窗门槛默认开启（长窗 FAIL 不部署）。"""
    from config.settings import settings
    assert settings.evolve_strategy_drl_deploy_requires_longwindow is True


@pytest.mark.asyncio
async def test_candidate_longwindow_gate_fail_blocks(tmp_path, monkeypatch):
    """候选 30 天长窗回放 FAIL → 门槛 False（不部署）。"""
    from config.settings import settings
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_longwindow",
                        True, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2026-06-01", periods=6000, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": np.linspace(2500, 2600, 6000),
                           "high": np.linspace(2500, 2600, 6000) + 5,
                           "low": np.linspace(2500, 2600, 6000) - 5,
                           "close": np.linspace(2500, 2600, 6000),
                           "volume": np.ones(6000)}, index=idx)
        return df
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)
    monkeypatch.setattr("backtest.engine.run_backtest",
                        lambda *a, **k: {"metrics": {"total_return": -0.1234}})

    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    ok, reason = await eng._candidate_longwindow_gate(
        agent, "ETH/USDT", {"state_window": 4}, {}, 0.0, 1.0)
    assert ok is False
    assert "长窗30d=-0.1234" in reason


@pytest.mark.asyncio
async def test_candidate_longwindow_gate_pass_allows(tmp_path, monkeypatch):
    """候选 30 天长窗回放 PASS → 门槛 True（允许部署）。"""
    from config.settings import settings
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_longwindow",
                        True, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2026-06-01", periods=6000, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": np.linspace(2500, 2600, 6000),
                           "high": np.linspace(2500, 2600, 6000) + 5,
                           "low": np.linspace(2500, 2600, 6000) - 5,
                           "close": np.linspace(2500, 2600, 6000),
                           "volume": np.ones(6000)}, index=idx)
        return df
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)
    monkeypatch.setattr("backtest.engine.run_backtest",
                        lambda *a, **k: {"metrics": {"total_return": 0.0726}})

    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    ok, reason = await eng._candidate_longwindow_gate(
        agent, "ETH/USDT", {"state_window": 4}, {}, 0.0, 1.0)
    assert ok is True
    assert "PASS" in reason


@pytest.mark.asyncio
async def test_candidate_longwindow_gate_disabled_allows(tmp_path, monkeypatch):
    """门槛关闭（逃生）→ 直接放行，不跑回放。"""
    from config.settings import settings
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_longwindow",
                        False, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    ok, reason = await eng._candidate_longwindow_gate(
        agent, "ETH/USDT", {"state_window": 4}, {}, 0.0, 1.0)
    assert ok is True and reason == ""


def test_deploy_longwindow_90d_default_off():
    """90 天部署门默认关闭（仅 30d 门槛，控制冠军轮成本）。"""
    from config.settings import settings
    assert settings.evolve_strategy_drl_deploy_longwindow_90d is False


@pytest.mark.asyncio
async def test_candidate_gate_90d_fail_blocks(tmp_path, monkeypatch):
    """开启 90d 门：30d PASS 但 90d FAIL → 拦截部署。"""
    from config.settings import settings
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_longwindow",
                        True, raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_longwindow_90d",
                        True, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2026-06-01", periods=27000, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": np.linspace(2500, 2600, 27000),
                           "high": np.linspace(2500, 2600, 27000) + 5,
                           "low": np.linspace(2500, 2600, 27000) - 5,
                           "close": np.linspace(2500, 2600, 27000),
                           "volume": np.ones(27000)}, index=idx)
        return df
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)
    calls = {"n": 0}

    def _fake_backtest(df, *a, **k):
        calls["n"] += 1
        # 按窗口长度区分：30d(≤8640) PASS / 90d(>8640) FAIL——并行时序无关
        ret = 0.0726 if len(df) <= 8640 else -0.18
        return {"metrics": {"total_return": ret}}
    monkeypatch.setattr("backtest.engine.run_backtest", _fake_backtest)

    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    ok, reason = await eng._candidate_longwindow_gate(
        agent, "ETH/USDT", {"state_window": 4}, {}, 0.0, 1.0)
    assert ok is False
    assert "90d" in reason
    assert calls["n"] == 2, "90d 门开启时应跑两次回放（30d+90d）"


@pytest.mark.asyncio
async def test_candidate_gate_90d_off_skips(tmp_path, monkeypatch):
    """90d 门关闭：30d PASS 即放行，只跑一次回放。"""
    from config.settings import settings
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_longwindow",
                        True, raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_longwindow_90d",
                        False, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2026-06-01", periods=6000, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": np.linspace(2500, 2600, 6000),
                           "high": np.linspace(2500, 2600, 6000) + 5,
                           "low": np.linspace(2500, 2600, 6000) - 5,
                           "close": np.linspace(2500, 2600, 6000),
                           "volume": np.ones(6000)}, index=idx)
        return df
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)
    calls = {"n": 0}

    def _fake_backtest(*a, **k):
        calls["n"] += 1
        return {"metrics": {"total_return": 0.0726}}
    monkeypatch.setattr("backtest.engine.run_backtest", _fake_backtest)

    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    ok, reason = await eng._candidate_longwindow_gate(
        agent, "ETH/USDT", {"state_window": 4}, {}, 0.0, 1.0)
    assert ok is True
    assert calls["n"] == 1, "90d 门关闭时只跑 30d 一次"