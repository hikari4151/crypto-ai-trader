"""simulated 余额缓存滞后回归测试（exp13 修复）。

覆盖 bug：simulated 模式走 _live_balance_cached 的 30s TTL 缓存，本地撮合
成交后 <30s 内 _current_state 读到陈旧持仓（position/cash 恒为初始值），
策略误判持续开仓直到余额耗尽。修复：SimulatedExchange 标记 is_local_matching，
引擎对本地撮合跳过缓存实时取余额。
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database
from engine.order_manager import OrderManager
from engine.portfolio import PortfolioManager
from engine.trading_engine import TradingEngine
from exchange.simulated import SimulatedExchange
from strategies.base import Signal


@pytest.mark.asyncio
async def test_simulated_balance_reads_immediately_after_fill():
    """买入成交后立即读 _current_state → 实时反映（无 30s 缓存滞后）。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = TradingEngine(db, EventBus())
    ex = SimulatedExchange(start_cash=10000.0)
    ex.set_last_price(50000.0)
    eng.exchange = ex
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, eng.bus, False, None)
    eng.order_manager.attach_exchange(ex)
    eng.portfolio = PortfolioManager(db, eng.bus, False, None, ex)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.trade_on_open = False
    try:
        sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="t")
        fill = await eng.order_manager.place(sig, 50000.0, "t")
        assert fill is not None and fill["qty"] > 0
        pos, cash, _pv, _ep = await eng._current_state("BTC/USDT", 50000.0)
        assert pos > 0, "买入后 position 应立即反映（曾因 30s 缓存滞后读 0）"
        assert cash < 10000.0, "买入后现金应立即扣减"
    finally:
        await eng.order_manager.close()
        await db.close()


@pytest.mark.asyncio
async def test_simulated_flip_sell_after_buy():
    """买入→卖出交替在 simulated 下应正常成交（买卖计数平衡）。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = TradingEngine(db, EventBus())
    ex = SimulatedExchange(start_cash=10000.0)
    ex.set_last_price(50000.0)
    eng.exchange = ex
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, eng.bus, False, None)
    eng.order_manager.attach_exchange(ex)
    eng.portfolio = PortfolioManager(db, eng.bus, False, None, ex)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.trade_on_open = False
    try:
        buys = sells = 0
        for _ in range(4):
            sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="t")
            f1 = await eng.order_manager.place(sig, 50000.0, "t")
            if f1:
                buys += 1
            pos, _c, _pv, _ep = await eng._current_state("BTC/USDT", 50000.0)
            if pos > 0:
                sig2 = Signal("BTC/USDT", "sell", size_pct=1.0, strategy="t")
                f2 = await eng.order_manager.place(sig2, 50000.0, "t")
                if f2:
                    sells += 1
        assert buys == sells > 0, "买卖应交替平衡（曾只买不卖直到余额耗尽）"
    finally:
        await eng.order_manager.close()
        await db.close()