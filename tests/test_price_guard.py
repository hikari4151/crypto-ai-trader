"""实时链路价格有限性守卫测试（bug 回归）。

覆盖修复：交易所脏数据（close=NaN/0/Inf）不得进入交易路径——
NaN 价格曾把纸面账户现金/持仓污染成 NaN 且 _record_trade 抛 IntegrityError；
0 价格曾导致 _resolve_qty 除零被宽泛 except 静默吞掉（信号丢单无告警）。
"""
import asyncio
import math

import pytest

from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from engine.trading_engine import TradingEngine
from strategies.base import Signal, Strategy


class AlwaysBuy(Strategy):
    name = "always_buy_guard"
    default_params = {"size_pct": 0.5}

    def on_candle(self, ctx):
        return Signal(ctx["symbol"], "buy", size_pct=0.5, strategy=self.name, reason="probe")

    def on_fill(self, symbol, side, price):
        pass


def _make_engine() -> tuple[TradingEngine, Database]:
    db = Database("sqlite+aiosqlite:///:memory:")
    bus = EventBus()
    eng = TradingEngine(db, bus)
    eng.running = True
    eng.paper = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = AlwaysBuy()
    eng.trade_on_open = False
    return eng, db


async def _feed(eng: TradingEngine, close_val: float) -> None:
    snap = {"symbol": eng.symbol, "timeframe": "1h",
            "candles": [[1700000000000, 0.0, 0.0, 0.0, close_val, 100.0]],
            "closes": [close_val],
            "indicators": {"close": close_val, "vol_ratio": 1.0}}
    await eng._on_candle(Event(EventType.MARKET_CANDLE,
                               {"symbol": eng.symbol, "snapshot": snap}, source="test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf"), -1.0])
async def test_invalid_price_skipped_no_pollution(bad):
    """close 非法（0/NaN/Inf/负）→ 本根K线跳过，账户零污染、无异常。"""
    eng, db = _make_engine()
    await db.init()
    try:
        await _feed(eng, bad)
        pos = eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0)
        assert pos == 0.0, "非法价格不得成交"
        assert math.isfinite(eng.paper_account.cash), "现金不得被污染"
        assert eng.paper_account.cash == 10000.0, "现金应保持初始值"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_valid_price_still_trades():
    """正常价格不受守卫影响：仍按原路径成交。"""
    eng, db = _make_engine()
    await db.init()
    try:
        await _feed(eng, 50000.0)
        pos = eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0)
        assert pos > 0.0, "正常价格应照常成交"
        assert math.isfinite(eng.paper_account.cash)
    finally:
        await db.close()
