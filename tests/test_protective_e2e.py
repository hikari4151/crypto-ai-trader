"""protective stop 触发→对账回灌端到端测试（exp9 回归）。

链路演练（探针 .optim/probe_protective_e2e.py 验证）：
1. simulated 市价买入 → _resync_protective 自动挂 protective 触发单（挂单簿）
2. inject_trigger_fill 模拟交易所侧触发成交（价格触及，订单 closed）
3. 对账循环 _reconcile_once：注册表有该单、交易所 open 已无 → 复核 → 回灌
   Trade 行（strategy=protective_stop）+ 清理注册表
4. 引擎 _on_order_fill(late) → 卖出公共记账（_trade_times、FIFO 摊销、风控盈亏）
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
    return eng, ex, bus


@pytest.mark.asyncio
async def test_protective_stop_trigger_backfills_sell():
    """protective 触发→对账回灌 Trade(protective_stop) + 引擎卖出记账。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng, ex, _bus = await _engine(db)
    try:
        # 1) 买入成交 → protective 挂单
        await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                   {"symbol": eng.symbol, "snapshot": _snap(1, 50000.0)},
                                   source="test"))
        assert ex._positions.get("BTC", 0.0) > 0, "买入应建仓"
        proto_ids = [oid for oid in ex._open if "triggerPrice" in ex._open[oid]]
        assert len(proto_ids) == 1, "买入后应自动挂 protective 触发单"

        # 2) 价格触及 → 触发单成交
        proto_id = proto_ids[0]
        trigger = ex._open[proto_id]["triggerPrice"]
        ex.inject_trigger_fill(proto_id, exec_price=trigger)
        assert ex._positions.get("BTC", 0.0) == 0.0, "触发成交应清仓"
        assert proto_id in eng.order_manager._open_orders, "触发单应从挂单簿转入注册表待对账"

        # 3) 对账回灌
        await eng.order_manager._reconcile_once()
        async with db.session() as s:
            trades = (await s.execute(select(Trade).order_by(Trade.id))).scalars().all()
        assert len(trades) == 2, "回灌后应有 buy+sell 两笔 Trade"
        sells = [t for t in trades if t.side == "sell"]
        assert len(sells) == 1 and sells[0].strategy == "protective_stop", "回灌 sell 应标记 protective_stop"
        assert proto_id not in eng.order_manager._open_orders, "回灌后应清理注册表"

        # 4) 引擎公共记账（late 回灌路径）
        await eng._on_order_fill(Event(EventType.ORDER_FILL, {
            "symbol": "BTC/USDT", "side": "sell", "price": trigger,
            "qty": sells[0].qty, "trade_id": sells[0].id, "fee": sells[0].fee,
            "late": True, "strategy_name": "protective_stop"}, source="order_manager"))
        assert len(eng._trade_times) == 2, "买卖各记一次频率计数"
        assert eng.risk._recent_pnls, "卖出回灌应记录风控盈亏（亏损）"
        assert eng.risk._recent_pnls[-1]["pnl"] < 0, "触发成交价低于买入价应为亏损"
        assert eng.order_manager._live_lots.get("BTC/USDT", []) == [], "清仓后 FIFO 队列应清空"
    finally:
        await eng.order_manager.close()
        await db.close()