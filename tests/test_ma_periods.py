"""双均线周期参数生效测试（本次优化第②点）。

- T1 回测：dual_ma 的 fast_period/slow_period 改变回测结果（此前被写死 10/30 忽略）
- T2 实时中枢：MarketDataHub 按传入 MA 周期计算 ma_fast/ma_slow
- T3 实时中枢：set_ma_periods 周期变更后重建计算器
- T4 辅助函数：strategy_ma_periods 推导规则
"""
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")


def _sine_df(n: int = 400) -> pd.DataFrame:
    t = np.arange(n)
    close = 100.0 + 8.0 * np.sin(t / 25.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.003,
        "low": close * 0.997, "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


# ============ T1. 回测：双均线周期参数生效 ============

def test_dual_ma_params_change_backtest():
    from backtest.engine import BacktestConfig, run_backtest

    df = _sine_df(400)
    r_fast = run_backtest(df, BacktestConfig(
        strategy_name="dual_ma",
        strategy_params={"fast_period": 5, "slow_period": 15}))
    r_slow = run_backtest(df, BacktestConfig(
        strategy_name="dual_ma",
        strategy_params={"fast_period": 40, "slow_period": 120}))

    trades_fast = r_fast["metrics"]["total_trades"]
    trades_slow = r_slow["metrics"]["total_trades"]
    ret_fast = r_fast["metrics"]["total_return"]
    ret_slow = r_slow["metrics"]["total_return"]
    # 快/慢周期差距大 → 成交时点/收益应明显不同（交易数可能巧合相同，故以收益判定）
    assert abs(ret_fast - ret_slow) > 1e-3, \
        f"dual_ma 周期参数应改变回测结果（ret fast={ret_fast}, slow={ret_slow}）"
    assert trades_fast >= 1 and trades_slow >= 1, "两种周期都应有成交"


# ============ T2. 实时中枢：按传入周期计算 ============

def test_hub_uses_ma_periods():
    from core.bus import EventBus
    from exchange.ws_market import MarketDataHub

    bus = EventBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus,
                        ma_fast_period=5, ma_slow_period=10)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    candles = [[1700000000000 + i * 3600000, c, c + 1, c - 1, c, 100.0]
               for i, c in enumerate(closes)]
    hub.apply_ohlcv("BTC/USDT", "1h", candles)
    ind = hub.snapshot("BTC/USDT", "1h")["indicators"]
    assert abs(ind["ma_fast"] - sum(closes[-5:]) / 5) < 1e-6, "ma_fast 应按 5 周期计算"
    assert abs(ind["ma_slow"] - sum(closes) / len(closes)) < 1e-6, "ma_slow 应按 10 周期计算"


# ============ T3. set_ma_periods 重建计算器 ============

def test_set_ma_periods_rebuilds():
    from core.bus import EventBus
    from exchange.ws_market import MarketDataHub

    bus = EventBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus,
                        ma_fast_period=5, ma_slow_period=10)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    candles = [[1700000000000 + i * 3600000, c, c + 1, c - 1, c, 100.0]
               for i, c in enumerate(closes)]
    hub.apply_ohlcv("BTC/USDT", "1h", candles)
    hub.set_ma_periods(3, 6)
    ind = hub.snapshot("BTC/USDT", "1h")["indicators"]
    assert abs(ind["ma_fast"] - sum(closes[-3:]) / 3) < 1e-6, "set_ma_periods 后 ma_fast 应按新周期 3 计算"
    assert abs(ind["ma_slow"] - sum(closes[-6:]) / 6) < 1e-6, "set_ma_periods 后 ma_slow 应按新周期 6 计算"


# ============ T4. strategy_ma_periods 推导规则 ============

def test_strategy_ma_periods_helper():
    from strategies.base import strategy_ma_periods
    from strategies.dual_ma import DualMAStrategy
    from strategies.grid import GridStrategy

    s = DualMAStrategy()
    s.update_params({"fast_period": 5, "slow_period": 20})
    assert strategy_ma_periods(s) == (5, 20), "dual_ma 应返回其 fast/slow 周期"

    default = DualMAStrategy()
    assert strategy_ma_periods(default) == (10, 30), "默认 dual_ma 应为 10/30"

    # 无 fast_period/slow_period 的策略保持默认 10/30（与 factor 库 bias_10/30 口径一致）
    assert strategy_ma_periods(GridStrategy()) == (10, 30)
