"""实盘订单状态机故障注入测试（exp6 回归）。

覆盖修复：_place_live_inner 轮询期间 fetch_order 抛网络异常时，
订单必须登记 _open_orders 注册表（交给对账循环兜底），而不是被外层
宽泛 except 吞掉后静默丢失（订单在交易所侧仍挂单/可能已成交，但本地
无 Trade 无注册表记录 = 隐性敞口）。
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database
from core.database import Trade
from engine.order_manager import OrderManager
from strategies.base import Signal


def make_om(db, ex):
    bus = EventBus()
    om = OrderManager(db, bus, paper=False)
    om.attach_exchange(ex)
    return om, bus


@pytest.mark.asyncio
async def test_fetch_error_during_poll_registers_open_order():
    """轮询时 fetch_order 抛异常 → 订单进注册表（对账兜底），不落库、不撤单。"""
    import ccxt

    class FlakyFetchExchange:
        def __init__(self):
            self.cancel_calls = []

        async def create_order(self, symbol, otype, side, amount, price=None):
            return {"id": "flaky1", "status": "open", "filled": None, "amount": 0.5}

        async def fetch_order(self, order_id, symbol):
            raise ccxt.NetworkError("simulated network timeout")

        async def cancel_order(self, order_id, symbol):
            self.cancel_calls.append(order_id)
            return {}

        async def fetch_open_orders(self, symbol=None):
            return [{"id": "flaky1"}]

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    ex = FlakyFetchExchange()
    om, _bus = make_om(db, ex)
    try:
        sig = Signal("BTC/USDT", "buy", qty=0.5, order_type="limit",
                     limit_price=50000.0, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is None, "状态未知不得落库为成交"
        assert "flaky1" in om._open_orders, "轮询失败必须登记注册表交给对账兜底"
        assert om._open_orders["flaky1"]["symbol"] == "BTC/USDT"
        from sqlalchemy import select
        async with db.session() as s:
            rows = (await s.execute(select(Trade))).scalars().all()
        assert rows == [], "状态未知不得写 Trade 行"
        assert ex.cancel_calls == [], "网络异常时不得盲目撤单（撤单也可能失败，交给对账）"
    finally:
        await om.close()
        await db.close()


@pytest.mark.asyncio
async def test_reconcile_backfills_registered_flaky_order():
    """登记后的订单对账兜底：交易所已成交 → 回灌 Trade 行 + 清理注册表。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    om = OrderManager(db, bus, paper=False)
    ex = _ReconcileExchange()
    om.attach_exchange(ex)
    try:
        om._open_orders["flaky9"] = {
            "symbol": "BTC/USDT", "side": "buy", "qty": 0.5,
            "strategy_name": "dual_ma", "placed_ts": 0.0, "limit_price": 50000.0,
        }
        task = asyncio.create_task(om._reconcile_loop())
        await asyncio.sleep(0.2)
        from sqlalchemy import select
        async with db.session() as s:
            rows = (await s.execute(select(Trade))).scalars().all()
        assert len(rows) == 1 and rows[0].qty == 0.5
        assert "flaky9" not in om._open_orders
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        await om.close()
        await db.close()


class _ReconcileExchange:
    """对账兜底用：交易所无 open 单，fetch_order 返回已成交。"""

    async def create_order(self, *a, **k):
        raise AssertionError("不应下单")

    async def fetch_open_orders(self, symbol=None):
        return []

    async def fetch_order(self, order_id, symbol):
        return {"id": order_id, "status": "closed", "filled": 0.5,
                "average": 51000.0, "fee": {"cost": 0.25}}