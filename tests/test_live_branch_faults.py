"""_place_live 剩余分支故障注入测试（exp8 回归固化）。

覆盖（在 SimulatedExchange 下演练，探针 .optim/probe_live_branches.py 验证过）：
① 部分成交且仍挂单（partial_open）：落库部分量 + 注册表 recorded_filled=M2 口径
② create_order 抛异常：place 返回 None、不落库、无注册表（安全）
③ 撤单失败（cancel_order 抛异常）：注册表保留（防双卖），对账兜底
④ simulated 引擎级：_current_state 实盘分支读 SimulatedExchange 余额（双向结构）
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from engine.order_manager import OrderManager
from engine.trading_engine import TradingEngine
from exchange.simulated import SimulatedExchange
from strategies.base import Signal
from sqlalchemy import select


async def _rows(db):
    async with db.session() as s:
        return (await s.execute(select(Trade))).scalars().all()


@pytest.mark.asyncio
async def test_partial_fill_registers_recorded_filled():
    """① 部分成交：fetch 返回 open+filled=0.06 → 落库部分量、注册表 recorded_filled=0.06。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    ex = SimulatedExchange(start_cash=10000.0)
    ex.set_last_price(50000.0)
    om = OrderManager(db, EventBus(), paper=False)
    om.attach_exchange(ex)

    real_fetch = ex.fetch_order

    async def partial_fetch(order_id, symbol):
        rec = ex._open.get(order_id)
        if rec:
            out = dict(rec)
            out["status"] = "open"
            out["filled"] = 0.06
            return out
        return await real_fetch(order_id, symbol)
    ex.fetch_order = partial_fetch

    try:
        sig = Signal("BTC/USDT", "buy", qty=0.1, order_type="limit",
                     limit_price=50000.0, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is not None and fill["qty"] == pytest.approx(0.06), "部分成交应返回已成交部分"
        rows = await _rows(db)
        assert len(rows) == 1 and rows[0].qty == pytest.approx(0.06), "应落库部分成交量"
        # M2：注册表记录已落库的成交部分（对账回灌按 recorded_filled 补差）
        reg = om._open_orders.get("sim1")
        assert reg is not None and reg.get("recorded_filled") == pytest.approx(0.06), \
            "部分成交订单须登记注册表且 recorded_filled=已落库量"
    finally:
        await om.close()
        await db.close()


@pytest.mark.asyncio
async def test_create_order_failure_is_safe():
    """② create_order 抛异常：place 返回 None、0 落库、0 注册表、余额不变。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    ex = SimulatedExchange(start_cash=10000.0)
    ex.set_last_price(50000.0)
    ex.fail_create_order_once = True
    om = OrderManager(db, EventBus(), paper=False)
    om.attach_exchange(ex)
    try:
        sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is None
        assert await _rows(db) == [], "create 失败不得落库"
        assert om._open_orders == {}, "create 失败不得登记注册表"
        assert ex._cash == pytest.approx(10000.0), "余额不得变动"
    finally:
        await om.close()
        await db.close()


@pytest.mark.asyncio
async def test_cancel_failure_keeps_registry():
    """③ 撤单失败：cancel_order 抛异常 → 注册表保留（防双卖），不落库。"""
    import ccxt

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    ex = SimulatedExchange(start_cash=10000.0)
    ex.set_last_price(50000.0)

    async def failing_cancel(order_id, symbol):
        raise ccxt.ExchangeError("simulated cancel failure")
    ex.cancel_order = failing_cancel

    om = OrderManager(db, EventBus(), paper=False)
    om.attach_exchange(ex)
    try:
        sig = Signal("BTC/USDT", "buy", qty=0.1, order_type="limit",
                     limit_price=49000.0, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is None
        assert "sim1" in om._open_orders, "撤单失败必须保留注册表（防止重复卖出同一持仓）"
        assert await _rows(db) == [], "撤单失败不得落库"
    finally:
        await om.close()
        await db.close()


@pytest.mark.asyncio
async def test_current_state_reads_simulated_balance():
    """④ simulated 引擎级：_current_state 实盘分支读 SimulatedExchange 余额（双向结构）。"""
    from engine.portfolio import PortfolioManager
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    eng = TradingEngine(db, bus)
    ex = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    ex.set_last_price(50000.0)
    eng.exchange = ex
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, bus, False, None)
    eng.order_manager.attach_exchange(ex)
    eng.portfolio = PortfolioManager(db, bus, False, None, ex)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.trade_on_open = False
    try:
        pos, cash, pv, _ep = await eng._current_state("BTC/USDT", 50000.0)
        assert pos == 0.0, "无持仓时 position 应为 0"
        assert cash == pytest.approx(10000.0), "实盘分支现金应读 SimulatedExchange 余额"
        assert pv == 0.0
    finally:
        await db.close()