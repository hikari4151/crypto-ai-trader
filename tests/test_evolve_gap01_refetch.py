"""TDD：GAP-01 / walk-forward 观测与训练窗口解耦（2026-09-21）。

背景：evolve_rolling_window=5000 同时约束训练与观测取数；gap01_bars 已提到
15000 但 _observe_gap01 的 df 来自 _fetch_latest_data（5000）被截断——考试
覆盖不到 8 月上旬。直接把 rolling_window 调 15000 会让训练每轮 12 倍计算量。
解耦：观测方法在 df 不足 bars 时按 bars 独立重拉（load_klines_cached），
训练窗口保持 5000 不动。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


@pytest.mark.asyncio
async def test_gap01_refetches_when_df_shorter_than_bars(tmp_path, monkeypatch):
    """df 不足 gap01_bars 时，观测按 bars 独立重拉（不依赖训练窗口）。"""
    from config.settings import settings
    from backtest.data_loader import generate_demo
    monkeypatch.setattr(settings, "evolve_gap01_obs", True, raising=False)
    monkeypatch.setattr(settings, "evolve_gap01_folds", 4, raising=False)
    monkeypatch.setattr(settings, "evolve_gap01_bars", 15000, raising=False)

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    small = generate_demo(timeframe="5m", n=3000, seed=7)

    async def _fake_fetch(symbol: str = ""):
        return small
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)

    fake_agent = ACAgent(state_dim=4, n_actions=5, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)

    calls = {}

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        calls["limit"] = limit
        return None  # 返回值不影响断言（重拉是否发生才是被测行为）
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)

    await eng._observe_gap01(df=small)
    assert calls.get("limit") == 15000, \
        f"df 不足时应按 bars=15000 重拉，实际 limit={calls.get('limit')}"


@pytest.mark.asyncio
async def test_walkforward_refetches_when_df_shorter_than_bars(tmp_path, monkeypatch):
    """walk-forward 观测同样解耦（bars>训练窗口时独立重拉）。"""
    from config.settings import settings
    from backtest.data_loader import generate_demo
    monkeypatch.setattr(settings, "evolve_strategy_drl_walkforward_obs", True,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_walkforward_bars", 15000,
                        raising=False)

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    small = generate_demo(timeframe="5m", n=3000, seed=7)

    async def _fake_fetch(symbol: str = ""):
        return small
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)

    fake_agent = ACAgent(state_dim=4, n_actions=5, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)

    calls = {}

    async def _fake_klines(exchange, symbol, timeframe, limit=0, max_candles=0):
        calls["limit"] = limit
        return None
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_klines)

    await eng._observe_strategy_drl_walkforward(df=small)
    assert calls.get("limit") == 15000, \
        f"walk-forward 观测 df 不足时应按 bars 重拉，实际 limit={calls.get('limit')}"