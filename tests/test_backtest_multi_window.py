"""F-③ 多窗口回测对比（2026-09-23）：同策略多历史区间并行回测，
判断窗口敏感性（防 v44 式"近期窗口效应"冒充实盘候选）。
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_multi_window


def _fixed_df(n=512):
    ts = 1700000000000 + np.arange(n) * 300000
    rng = np.random.default_rng(42)
    rets = np.concatenate([np.full(256, 0.0004), np.full(256, -0.0002)]) \
        + rng.normal(0, 0.0015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open": np.roll(close, 1), "high": close * 1.003,
        "low": close * 0.997, "close": close,
        "volume": 10.0 + rng.random(n) * 5.0,
    }, index=pd.to_datetime(ts, unit="ms", utc=True))
    return df


def test_multi_window_returns_each_window():
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005)
    windows = [(0, 256), (256, 512), (0, 512)]
    out = run_multi_window(df, cfg, windows)
    assert len(out) == 3
    for item, (s, e) in zip(out, windows):
        assert item["window"] == [s, e]
        assert item["rows"] == e - s
        assert "total_return" in item["metrics"]
        assert "total_trades" in item["metrics"]
    # 三段窗口结果应不同（数据分段不同）——防止"同结果"假对比
    rets = [item["metrics"]["total_return"] for item in out]
    assert len(set(round(float(r), 6) for r in rets)) >= 2


def test_multi_window_empty_windows():
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005)
    assert run_multi_window(df, cfg, []) == []