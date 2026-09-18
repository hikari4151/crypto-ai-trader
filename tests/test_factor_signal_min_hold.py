"""min_hold_bars（最短持仓期）回归测试：压换手离场门控。

P8 成本感知执行优化需求：止损永远放行，止盈/因子离场必须等到持仓满
min_hold_bars 根 K 线（默认 0 = 旧行为不限）。
"""
import pytest

from strategies.factor_signal import FactorSignalStrategy


def _ctx(price=100.0, position=0.0, macd_hist=0.5, candles=50):
    return {
        "symbol": "BTC/USDT", "price": price, "position": position,
        "indicators": {"candles_count": candles, "close": price, "open": price,
                       "high": price, "low": price, "volume": 100.0,
                       "macd_hist": macd_hist},
    }


def _make_strategy(min_hold, stop=1.0, tp=1.0):
    st = FactorSignalStrategy()
    st.update_params({
        "factor": "macd_hist", "mode": "trend",
        "buy_threshold": 0.0, "sell_threshold": 0.0,
        "stop_loss_pct": stop, "take_profit_pct": tp,
        "size_pct": 0.5, "min_hold_bars": min_hold,
    })
    st.reset()
    return st


def test_sell_held_below_min_hold_is_blocked():
    """因子转弱但持仓未满 min_hold_bars → 不产生离场信号。"""
    st = _make_strategy(min_hold=3)
    st._entry = 100.0  # 模拟已持仓
    ctx = _ctx(position=0.1, macd_hist=-0.5)  # 因子 < sell_threshold → 应离场
    assert st.on_candle(ctx) is None, "未满 min_hold 不得因子离场"


def test_sell_after_min_hold_reached():
    """held>=min_hold_bars 后因子转弱 → 正常离场。"""
    st = _make_strategy(min_hold=3)
    st._entry = 100.0
    ctx = _ctx(position=0.1, macd_hist=-0.5)
    st.on_candle(ctx)  # held=1
    st.on_candle(ctx)  # held=2
    sig = st.on_candle(ctx)  # held=3
    assert sig is not None and sig.side == "sell"


def test_stop_loss_always_allowed_during_min_hold():
    """止损不受最短持仓期约束（资金安全第一）。"""
    st = _make_strategy(min_hold=100, stop=0.05)
    st._entry = 100.0
    ctx = _ctx(price=90.0, position=0.1)  # -10% < -5% → 止损
    sig = st.on_candle(ctx)
    assert sig is not None and sig.side == "sell" and "止损" in sig.reason


def test_min_hold_zero_preserves_old_behavior():
    """min_hold_bars=0（默认）：因子转弱立即离场（旧行为）。"""
    st = _make_strategy(min_hold=0)
    st._entry = 100.0
    ctx = _ctx(position=0.1, macd_hist=-0.5)
    sig = st.on_candle(ctx)
    assert sig is not None and sig.side == "sell"