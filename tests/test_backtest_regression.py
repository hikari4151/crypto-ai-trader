"""E-① 撮合回归对照门禁（快照测试，2026-09-23）。

固定确定性输入（自造 512 根 5m OHLCV，seed=42、时间戳固定）跑 dual_ma 回测，
断言指标与逐笔交易哈希与**硬编码基准**一致——任何撮合/特征/因子/口径改动
改变输出都会让测试变红（防止行为漂移静默引入，本会话已两次踩到）。

基准更新（有意改动撮合后）：重新截获 metrics/trades_md5 并更新本文件。
"""
import hashlib
import json

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest

# 硬编码基准（2026-09-23 截获；改动撮合后有意更新）
_METRICS_SNAPSHOT = {
    "total_return": -0.00484,
    "sharpe": -5.6863,
    "max_drawdown": 0.02819,
    "total_trades": 7,
    "win_rate": 0.1429,
    "calmar_ratio": -22.3731,
    "sortino": -7.3086,
    "avg_slippage": 0.0005,
}
_TRADES_MD5 = "0f4db7b439b9dae55ea5318ae4805345"


def _fixed_df(n: int = 512) -> pd.DataFrame:
    """确定性固定数据：趋势段+回落段+噪声，时间戳固定（不受当前时间影响）。"""
    ts = 1700000000000 + np.arange(n) * 300000
    rng = np.random.default_rng(42)
    rets = np.concatenate([np.full(256, 0.0004), np.full(256, -0.0002)]) \
        + rng.normal(0, 0.0015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + 0.003)
    low = close * (1 - 0.003)
    volume = 10.0 + rng.random(n) * 5.0
    return pd.DataFrame({
        "open": np.roll(close, 1), "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=pd.to_datetime(ts, unit="ms", utc=True))


def _run() -> dict:
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005)
    return run_backtest(df, cfg)


def test_backtest_metrics_snapshot():
    """撮合输出指标与基准完全一致（门禁）。"""
    r = _run()
    m = r.get("metrics", {})
    got = {k: round(float(m.get(k)), 6) for k in _METRICS_SNAPSHOT}
    assert got == _METRICS_SNAPSHOT, f"指标漂移: {got} != {_METRICS_SNAPSHOT}"


def test_backtest_trades_snapshot():
    """逐笔交易哈希与基准一致（任一成交变化即红）。"""
    r = _run()
    h = hashlib.md5(
        json.dumps(r.get("trades", []), default=str, sort_keys=True).encode()
    ).hexdigest()
    assert h == _TRADES_MD5, f"交易序列漂移: {h} != {_TRADES_MD5}"


# C-①（GAP-04）：限价/部分成交执行口径回归覆盖（非乐观口径）
_PARTIAL_SNAPSHOT = {
    "total_return": -0.000185,
    "total_trades": 7,
    "total_costs": 1.8757,
    "avg_slippage": 0.0005,
}


def test_backtest_partial_limit_mode_snapshot():
    """limit_order_model=partial + participation=0.1 输出稳定（C-① 执行口径门禁）。"""
    df = _fixed_df()
    cfg = BacktestConfig(symbol="ETH/USDT", timeframe="5m",
                         strategy_name="dual_ma", start_cash=10000.0,
                         fee_rate=0.001, slippage=0.0005,
                         limit_order_model="partial", participation_rate=0.1)
    m = run_backtest(df, cfg).get("metrics", {})
    got = {k: round(float(m.get(k)), 6 if k != "total_trades" else 0)
           for k in _PARTIAL_SNAPSHOT}
    assert got == _PARTIAL_SNAPSHOT, f"partial 口径漂移: {got} != {_PARTIAL_SNAPSHOT}"


# F-⑤（2026-09-24）：成交明细增强——卖出含平均建仓价 + 持仓周期
def test_backtest_trade_detail_fields():
    r = _run()
    sells = [t for t in r.get("trades", []) if t.get("side") == "sell"]
    assert sells, "应存在卖出成交"
    for t in sells:
        ep = t.get("entry_price")
        hb = t.get("hold_bars")
        assert ep is not None and float(ep) > 0, f"entry_price 缺失/非法: {t}"
        assert hb is not None and int(hb) >= 0, f"hold_bars 缺失/非法: {t}"
