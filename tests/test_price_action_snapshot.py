"""price_action 快照含 sr/pa 测试（P0-3）。

覆盖 2026-08-14 修复：ws_market.snapshot() 补算关键位(sr)与价格行为(pa)，
窗口 buf[-120:] 与 fast_engine.py:85 同口径（window=10, min_touches=2）；
消费方 PriceActionStrategy 构造含 sr/pa 的 ctx 可产出突破买入信号。
"""
import math

from exchange.ws_market import MarketDataHub, _compute_sr_pa
from strategies.price_action import PriceActionStrategy


def _make_candles(n: int = 120, base: float = 100.0) -> list:
    """合成 K 线：正弦波动价格，含足够摆动高低点形成 S/R。格式 [ts,o,h,l,c,v]。"""
    out = []
    ts = 1700000000000
    for i in range(n):
        swing = math.sin(i / 6.0) * 3.0
        close = base + swing + i * 0.01
        open_ = base + math.sin((i - 1) / 6.0) * 3.0 + (i - 1) * 0.01
        high = max(open_, close) + 0.5
        low = min(open_, close) - 0.5
        out.append([ts + i * 3600000, round(open_, 2), round(high, 2), round(low, 2),
                    round(close, 2), 1000.0 + i])
    return out


def test_compute_sr_pa_keys():
    """120 根合成 K 线 → sr 含 support/resistance/broken_resistance，pa 含 volume_ratio。"""
    candles = _make_candles(120)
    sr, pa = _compute_sr_pa(candles)
    assert "support" in sr and "resistance" in sr and "broken_resistance" in sr
    assert "volume_ratio" in pa


def test_compute_sr_pa_short_window():
    """少于 21 根：pa 返回 {}（合法空态），sr 全 None（与 fast_engine 短数据一致）。"""
    sr, pa = _compute_sr_pa(_make_candles(10))
    assert pa == {}
    assert sr["support"] is None and sr["resistance"] is None and sr["broken_resistance"] is None


def test_breakout_signal_from_sr_pa_ctx():
    """消费方 PriceActionStrategy：构造含 sr/pa 的 ctx（broken_resistance 刚被突破）→ 买入信号。"""
    st = PriceActionStrategy()
    ind = {
        "candles_count": 120,
        "close": 105.0,
        "rsi": 50.0,
        "sr": {"support": 95.0, "resistance": 108.0, "broken_resistance": 100.0},
        "pa": {"volume_ratio": 1.5},
    }
    st._prev_close = 99.0  # 前一根收盘仍在阻力下方（"刚刚上穿"确认）
    sig = st.on_candle({"symbol": "BTC/USDT", "price": 105.0, "position": 0.0,
                        "cash": 10000.0, "indicators": ind, "timeframe": "1h"})
    assert sig is not None and sig.side == "buy", "突破场景应产出买入信号"
    assert "突破" in sig.reason


def test_snapshot_contains_sr_pa_and_stable():
    """hub.snapshot() 的 indicators 含 sr/pa 新增键；同 ts 重复调用结果稳定（ts 守卫不重算）。"""
    import asyncio
    from core.bus import EventBus

    bus = EventBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    candles = _make_candles(120)
    hub._candles[("BTC/USDT", "1h")] = [list(c) for c in candles]

    async def run():
        snap1 = hub.snapshot("BTC/USDT", "1h")
        # 同 ts 重复读取：snapshot 结果与首次一致（_sr_pa_ts 守卫，不重算）
        snap2 = hub.snapshot("BTC/USDT", "1h")
        return snap1, snap2

    snap1, snap2 = asyncio.run(run())
    # snapshot 返回键不变
    assert {"symbol", "timeframe", "candles", "closes", "indicators"} <= set(snap1)
    assert len(snap1["candles"]) == 120  # 恒为 buf[-120:]
    ind = snap1["indicators"]
    assert "sr" in ind and "pa" in ind, "snapshot 的 indicators 必须含 sr/pa"
    assert "support" in ind["sr"] and "volume_ratio" in ind["pa"]
    assert snap1["indicators"]["sr"] == snap2["indicators"]["sr"]
    assert snap1["indicators"]["pa"] == snap2["indicators"]["pa"]
    assert hub._sr_pa_ts[("BTC/USDT", "1h")] == int(candles[-1][0]), "ts 守卫应记录最后一根K线 ts"


def test_snapshot_recompute_on_new_candle():
    """新 K 线（ts 变化）到达后 sr/pa 重算，_sr_pa_ts 更新到新 ts。"""
    import asyncio
    from core.bus import EventBus

    bus = EventBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    candles = _make_candles(120)
    hub._candles[("BTC/USDT", "1h")] = [list(c) for c in candles]

    async def run():
        before = hub.snapshot("BTC/USDT", "1h")["indicators"]
        old_ts = hub._sr_pa_ts[("BTC/USDT", "1h")]
        # 新 K 线（下一根 ts，价格上移）
        last = candles[-1]
        nxt = [last[0] + 3600000, last[4], last[4] + 1.0, last[4] - 0.5, last[4] + 0.8, 5000.0]
        hub._candles[("BTC/USDT", "1h")].append(nxt)
        after = hub.snapshot("BTC/USDT", "1h")["indicators"]
        return before, after, old_ts

    before, after, old_ts = asyncio.run(run())
    assert hub._sr_pa_ts[("BTC/USDT", "1h")] != old_ts, "新 K 线应触发重算并更新 ts 守卫"
    assert before["sr"] != after["sr"] or before["pa"] != after["pa"], "新 K 线后 sr/pa 应重算更新"
    assert "sr" in after and "pa" in after

