"""protective 触发回灌后引擎侧安全行为测试（exp10 回归）。

覆盖（探针 .optim/probe_protective_after.py 验证）：
① 触发清仓后 _resync_protective 不得重挂（持仓已 0）
② FIFO 队列在回灌路径正确摊销（protective 卖出后 _live_lots 清空）
③ protective._stops 本地记录随 forget 清理
④ 普通卖出（策略主动）release→成交→resync 对口：清仓后不重挂
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from core.events import Event, EventType
from engine.order_manager import OrderManager
from engine.portfolio import PortfolioManager
from engine.trading_engine import TradingEngine
from exchange.simulated import SimulatedExchange
from strategies.base import Signal, Strategy
from sqlalchemy import select


class BuyOnce(Strategy):
    name = "sim_once"
    default_params = {"size_pct": 0.5}

    def __init__(self):
        super().__init__()
        self._n = 0
        self._entry = None

    def on_candle(self, ctx):
        self._n += 1
        if self._n == 1:
            return Signal(ctx["symbol"], "buy", size_pct=0.5, strategy=self.name, reason="sim")
        return None

    def on_fill(self, symbol, side, price):
        if side == "buy":
            self._entry = price


def _snap(i, price):
    return {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1700000000000 + i * 3600000, price, price, price, price, 100.0]],
            "closes": [price],
            "indicators": {"close": price, "vol_ratio": 1.0}}


async def _engine(db):
    bus = EventBus()
    eng = TradingEngine(db, bus)
    ex = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    ex.set_last_price(50000.0)
    eng.exchange = ex
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, bus, False, None)
    eng.order_manager.attach_exchange(ex)
    eng.order_manager.set_stop_pct_source(
        lambda: getattr(eng.strategy, "params", {}).get("stop_loss_pct"))
    eng.portfolio = PortfolioManager(db, bus, False, None, ex)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = BuyOnce()
    eng.trade_on_open = False
    return eng, ex


def _proto_ids(ex):
    return [oid for oid in ex._open if "triggerPrice" in ex._open[oid]]


async def _buy(eng, ex):
    await eng._on_candle(Event(EventType.MARKET_CANDLE,
                               {"symbol": eng.symbol, "snapshot": _snap(1, 50000.0)},
                               source="test"))
    assert ex._positions.get("BTC", 0.0) > 0, "买入应建仓"
    return _proto_ids(ex)[0]


@pytest.mark.asyncio
async def test_protective_trigger_fill_cleans_fifo_and_stops():
    """protective 触发→对账回灌：FIFO 摊销清空、_stops 清理、清仓后不重挂。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng, ex = await _engine(db)
    om = eng.order_manager
    try:
        await _buy(eng, ex)
        assert om._live_lots.get("BTC/USDT") == [(0.1, 50025.0)], "买入应入 FIFO 队列"
        assert "BTC/USDT" in om.protective._stops, "买入成交应登记 protective._stops"

        p_id = _proto_ids(ex)[0]
        trigger = ex._open[p_id]["triggerPrice"]
        ex.inject_trigger_fill(p_id, exec_price=trigger)
        await om._reconcile_once()

        # ① 回灌后清仓，不得重挂
        assert _proto_ids(ex) == [], "触发清仓后不得重挂 protective 单"
        # ② FIFO 正确摊销
        assert om._live_lots.get("BTC/USDT", []) == [], "protective 卖出应摊销清空 FIFO"
        # ③ 本地记录随 forget 清理
        assert "BTC/USDT" not in om.protective._stops, "对账回灌应 forget 已了结的 protective"
        # 回灌的卖出成交号策略标记
        async with db.session() as s:
            sells = [t for t in (await s.execute(select(Trade))).scalars().all() if t.side == "sell"]
        assert len(sells) == 1 and sells[0].strategy == "protective_stop"
        assert sells[0].qty == pytest.approx(0.1)
    finally:
        await om.close()
        await db.close()


@pytest.mark.asyncio
async def test_normal_sell_release_then_no_rehang():
    """普通卖出：先 release 服务端止损，成交后清仓不重挂（无残留保护单）。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng, ex = await _engine(db)
    om = eng.order_manager
    try:
        await _buy(eng, ex)
        assert len(_proto_ids(ex)) == 1

        await om.protective.release("BTC/USDT")
        assert len(ex._open) == 0, "卖出前应撤掉服务端兜底止损（避免余额锁定/双卖）"

        sig = Signal("BTC/USDT", "sell", size_pct=1.0, strategy="sim_once", reason="take_profit")
        fill = await om.place(sig, 50000.0, "sim_once")
        assert fill is not None
        assert ex._positions.get("BTC", 0.0) == 0.0, "卖出应清仓"
        assert om._live_lots.get("BTC/USDT", []) == [], "清仓后 FIFO 应清空"
        assert _proto_ids(ex) == [], "清仓后卖出路径不得重挂 protective"
    finally:
        await om.close()
        await db.close()