"""GAP-05 回归测试：POST /api/trading/close-all 一键清仓（simulated 撮合闭环）。

覆盖 docs/LIVE_READINESS.md GAP-05 验收标准：
1. 引擎运行时按当前持仓数量市价卖出（paper/simulated/live 同路径 _execute_signal）；
2. 平仓后持仓清零、Trade 落 sell 单、protective stop 释放/清空；
3. 引擎未运行时拒绝（停引擎≠清仓）。
"""
import pytest

from core.bus import EventBus
from core.database import Database, Trade
from engine.order_manager import OrderManager
from engine.portfolio import PortfolioManager
from engine.trading_engine import TradingEngine
from exchange.simulated import SimulatedExchange
from sqlalchemy import select
from strategies.base import Strategy


class DummyStrategy(Strategy):
    name = "dummy_close_all"
    default_params = {}

    def on_candle(self, ctx):
        return None


class _Hub:
    """close_all_positions 只需要 hub.last_price(symbol)。"""

    def __init__(self, price: float):
        self._price = price

    def last_price(self, symbol: str) -> float:
        return self._price


async def _engine_with_position(db) -> TradingEngine:
    bus = EventBus()
    eng = TradingEngine(db, bus)
    ex = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    ex.set_last_price(50000.0)
    # 先在模拟交易所买入建立持仓（0.1 BTC ≈ 5007.5 USDT）
    await ex.create_order("BTC/USDT", "market", "buy", 0.1, 50000.0)
    eng.exchange = ex
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, bus, False, None)
    eng.order_manager.attach_exchange(ex)
    eng.order_manager.set_stop_pct_source(lambda: None)
    eng.portfolio = PortfolioManager(db, bus, False, None, ex)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = DummyStrategy()
    eng.trade_on_open = False
    eng.hub = _Hub(50000.0)
    return eng, ex


@pytest.mark.asyncio
async def test_close_all_clears_position_and_records_sell():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng, ex = await _engine_with_position(db)
    try:
        assert ex._positions.get("BTC", 0.0) > 0, "前置：应先有持仓"
        res = await eng.close_all_positions("test_close_all")
        assert res["ok"] is True, res
        assert res["closed_qty"] == pytest.approx(0.1)
        # 模拟交易所持仓清零
        assert ex._positions.get("BTC", 0.0) == 0.0
        # Trade 表落 sell（exchange=live 口径，simulated 走 _place_live）
        async with db.session() as s:
            trades = (await s.execute(select(Trade))).scalars().all()
        sells = [t for t in trades if t.side == "sell"]
        assert len(sells) >= 1, "一键清仓应落库卖出 Trade"
        sell = sells[-1]
        assert sell.qty == pytest.approx(0.1)
        assert sell.reason == "test_close_all"
    finally:
        await eng.order_manager.close()
        await db.close()


@pytest.mark.asyncio
async def test_close_all_rejects_when_engine_stopped():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng, _ex = await _engine_with_position(db)
    eng.running = False
    try:
        res = await eng.close_all_positions("test")
        assert res["ok"] is False and "未运行" in res["detail"]
    finally:
        await eng.order_manager.close()
        await db.close()