"""实盘订单生命周期测试（P1-2）：filled 口径 + 轮询超时撤单 + 对账回灌迟到成交。

覆盖 2026-08-14 修复：
- filled 口径：仅 status=="closed" 允许 amount 兜底；open 且 filled=None/0 → 0（不误判全量成交）
- 3×1s 轮询仍 open 且未成交 → 主动撤单 + 复核 + open-order 注册表
- 对账循环：注册表订单交易所侧已成交 → 回灌 Trade 行 + ORDER_FILL(late) 事件
"""
import asyncio

from core.bus import EventBus
from core.database import Database, Trade
from engine.order_manager import OrderManager
from strategies.base import Signal


class FakeExchange:
    """模拟 ccxt 交易所接口（fetch_order 响应按队列弹出，耗尽后重复最后一个）。"""

    def __init__(self):
        self.create_resp = {}
        self.fetch_responses = []
        self.cancel_calls = []
        self.open_ids = []
        self.cancelled_ids = []

    async def create_order(self, symbol, otype, side, amount, price=None):
        return self.create_resp

    async def fetch_order(self, order_id, symbol):
        if self.fetch_responses:
            return self.fetch_responses.pop(0)
        return {"id": order_id, "status": "open", "filled": None, "amount": 0.0}

    async def cancel_order(self, order_id, symbol):
        self.cancel_calls.append(order_id)
        self.cancelled_ids.append(order_id)
        return {}

    async def fetch_open_orders(self, symbol=None):
        return [{"id": oid} for oid in self.open_ids]


def make_om(db, ex):
    bus = EventBus()
    om = OrderManager(db, bus, paper=False)
    om.attach_exchange(ex)
    return om, bus


async def _setup():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    return db


async def _trade_rows(db):
    from sqlalchemy import select
    async with db.session() as s:
        return (await s.execute(select(Trade))).scalars().all()


def test_open_order_not_filled_cancelled_and_registered():
    """open 且 filled=None：保持 0（不 amount 兜底）→ 撤单 + 复核 → 注册表登记 → 返回 None。"""
    ex = FakeExchange()
    ex.create_resp = {"id": "oid1", "status": "open", "filled": None, "amount": 0.5}
    # 3 次轮询 open + 撤单后复核 open
    ex.fetch_responses = [
        {"id": "oid1", "status": "open", "filled": None, "amount": 0.5}] * 3 + [
        {"id": "oid1", "status": "open", "filled": None, "amount": 0.5}]

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            sig = Signal("BTC/USDT", "buy", qty=0.5, order_type="limit",
                         limit_price=50000.0, strategy="dual_ma")
            fill = await om.place(sig, 50000.0, "dual_ma")
            assert fill is None, "未成交应返回 None"
            assert ex.cancel_calls == ["oid1"], "轮询超时应主动撤单"
            assert "oid1" in om._open_orders, "撤单后应登记 open-order 注册表"
            assert om._open_orders["oid1"]["symbol"] == "BTC/USDT"
            assert om._open_orders["oid1"]["limit_price"] == 50000.0
            assert await _trade_rows(db) == [], "未成交不得落库"
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_closed_filled_zero_uses_amount():
    """closed 且 filled=0 → amount 兜底（仅 closed 允许）；正常落库 + ORDER_FILL。"""
    ex = FakeExchange()
    ex.create_resp = {"id": "oid2", "status": "closed", "filled": None,
                      "amount": 1.5, "average": 50000.0, "fee": {"cost": 0.75}}

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            sig = Signal("BTC/USDT", "buy", qty=1.5, order_type="limit",
                         limit_price=50000.0, strategy="dual_ma")
            fill = await om.place(sig, 50000.0, "dual_ma")
            assert fill is not None and fill["qty"] == 1.5, "closed 应 amount 兜底全量成交"
            rows = await _trade_rows(db)
            assert len(rows) == 1 and rows[0].qty == 1.5 and rows[0].exchange == "live"
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_cancel_race_recheck_sees_fill():
    """撤单竞态：撤单后复核发现已成交（closed filled>0）→ 正常落库，不进注册表。"""
    ex = FakeExchange()
    ex.create_resp = {"id": "oid3", "status": "open", "filled": None, "amount": 0.4}
    # 3 次轮询 open，撤单后复核返回 closed 已成交
    ex.fetch_responses = [
        {"id": "oid3", "status": "open", "filled": None, "amount": 0.4}] * 3 + [
        {"id": "oid3", "status": "closed", "filled": 0.4, "average": 50100.0, "fee": {"cost": 0.2}}]

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            sig = Signal("BTC/USDT", "buy", qty=0.4, order_type="limit",
                         limit_price=50000.0, strategy="dual_ma")
            fill = await om.place(sig, 50000.0, "dual_ma")
            assert fill is not None and fill["qty"] == 0.4
            assert "oid3" not in om._open_orders, "竞态成交不应进注册表"
            assert len(await _trade_rows(db)) == 1
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_backfill_late_fill():
    """对账回灌：注册表订单交易所侧已成交 → Trade 行 + ORDER_FILL(late) 事件。"""
    ex = FakeExchange()
    ex.open_ids = []

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            om._open_orders["oid9"] = {
                "symbol": "BTC/USDT", "side": "buy", "qty": 0.4,
                "strategy_name": "dual_ma", "placed_ts": 0.0, "limit_price": 50000.0,
            }
            await om._backfill_fill("oid9", om._open_orders["oid9"], {
                "id": "oid9", "status": "closed", "filled": 0.4,
                "average": 51000.0, "fee": {"cost": 0.2}})
            rows = await _trade_rows(db)
            assert len(rows) == 1 and rows[0].qty == 0.4 and rows[0].order_id == "oid9"
            assert "oid9" not in om._open_orders, "回灌后应清理注册表"
            # ORDER_FILL 事件入队（late 键新增，原键不变）
            from core.events import EventType
            q = bus._queues[EventType.ORDER_FILL]
            ev = q.get_nowait()
            assert ev.type == EventType.ORDER_FILL
            assert ev.payload["late"] is True
            assert ev.payload["symbol"] == "BTC/USDT"
            assert ev.payload["qty"] == 0.4
            assert ev.payload["trade_id"] == rows[0].id
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_reconcile_loop_backfills_disappeared_order():
    """对账循环：注册表有、交易所 open 列表已无 → 复核成交 → 回灌。"""
    ex = FakeExchange()
    ex.open_ids = []  # 交易所已无该订单
    ex.fetch_responses = [
        {"id": "oid7", "status": "closed", "filled": 0.3, "average": 52000.0, "fee": {"cost": 0.15}}]

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            om._open_orders["oid7"] = {
                "symbol": "BTC/USDT", "side": "sell", "qty": 0.3,
                "strategy_name": "dual_ma", "placed_ts": 0.0, "limit_price": 52000.0,
            }
            task = asyncio.create_task(om._reconcile_loop())
            await asyncio.sleep(0.2)  # 等第一轮对账完成
            rows = await _trade_rows(db)
            assert len(rows) == 1 and rows[0].qty == 0.3
            assert "oid7" not in om._open_orders
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_reconcile_loop_keeps_still_open_order():
    """对账循环：交易所 open 列表仍有该订单 → 保留注册表，不落库。"""
    ex = FakeExchange()
    ex.open_ids = ["oid8"]  # 交易所仍挂单

    async def run():
        db = await _setup()
        om, bus = make_om(db, ex)
        try:
            om._open_orders["oid8"] = {
                "symbol": "BTC/USDT", "side": "buy", "qty": 0.3,
                "strategy_name": "dual_ma", "placed_ts": 0.0, "limit_price": 52000.0,
            }
            task = asyncio.create_task(om._reconcile_loop())
            await asyncio.sleep(0.2)
            assert "oid8" in om._open_orders, "仍 open 应保留注册表"
            assert await _trade_rows(db) == [], "不得重复落库"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        finally:
            await om.close()

    assert asyncio.run(run())


def test_engine_backfill_record_fill():
    """引擎侧 _on_order_fill：late 事件 → 公共记账（_trade_times + TRADE 事件）；
    非 late 事件忽略（避免与即时路径重复记账）；sell 回灌同口径累计风控每日盈亏。"""
    from core.events import Event, EventType
    from engine.trading_engine import TradingEngine

    async def run():
        db = await _setup()
        bus = EventBus()
        eng = TradingEngine(db, bus)
        # buy 回灌：公共记账
        await eng._on_order_fill(Event(EventType.ORDER_FILL, {
            "symbol": "BTC/USDT", "side": "buy", "price": 50000.0, "qty": 0.5,
            "trade_id": 1, "late": True}, source="order_manager"))
        assert len(eng._trade_times) == 1
        q = bus._queues[EventType.TRADE]
        ev = q.get_nowait()
        assert ev.payload["fill"]["qty"] == 0.5
        # sell 回灌：pnl 同口径更新（策略入口价 49000，卖出 50000*0.5 → +500）
        eng.strategy._entry_price = 49000.0
        await eng._on_order_fill(Event(EventType.ORDER_FILL, {
            "symbol": "BTC/USDT", "side": "sell", "price": 50000.0, "qty": 0.5,
            "trade_id": 2, "fee": 0.0, "late": True}, source="order_manager"))
        assert len(eng._trade_times) == 2
        daily = await eng.risk.get_daily_pnl()
        assert abs(daily - 500.0) < 1e-6, f"回灌 sell 应更新每日盈亏，实际 {daily}"
        assert len(eng.risk._recent_pnls) == 1, "回灌 sell 应记录风控盈亏"
        # 非 late 事件忽略
        await eng._on_order_fill(Event(EventType.ORDER_FILL, {
            "symbol": "BTC/USDT", "side": "buy", "price": 51000.0, "qty": 0.2,
            "trade_id": 3}, source="order_manager"))
        assert len(eng._trade_times) == 2, "非 late 事件不得重复记账"
        return True

    assert asyncio.run(run())
