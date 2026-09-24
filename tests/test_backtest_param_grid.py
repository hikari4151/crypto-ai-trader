"""F-④ 通用参数网格扫描（2026-09-23）：任意策略参数笛卡尔积批量回测，
返回每组合 metrics（参数敏感性/热力图数据源）。
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.grid_scan import run_param_grid


def _fixed_df(n=512):
    ts = 1700000000000 + np.arange(n) * 300000
    rng = np.random.default_rng(42)
    rets = np.concatenate([np.full(256, 0.0004), np.full(256, -0.0002)]) \
        + rng.normal(0, 0.0015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": np.roll(close, 1), "high": close * 1.003,
        "low": close * 0.997, "close": close,
        "volume": 10.0 + rng.random(n) * 5.0,
    }, index=pd.to_datetime(ts, unit="ms", utc=True))


def test_param_grid_cartesian():
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005)
    grid = {"fast_period": [5, 20], "slow_period": [30]}
    out = run_param_grid(df, cfg, grid)
    # 2×1 = 2 组合
    assert len(out) == 2
    params = [tuple(sorted(item["params"].items())) for item in out]
    assert len(set(params)) == 2
    for item in out:
        assert "total_return" in item["metrics"]
        assert "total_trades" in item["metrics"]
        assert item["metrics"]["total_trades"] >= 0


def test_param_grid_empty_returns_single():
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005)
    out = run_param_grid(df, cfg, {})
    assert len(out) == 1
    assert "total_return" in out[0]["metrics"]