"""P32 故障注入演练：实盘异常场景的端到端行为验证。

目标：验证「异常发生 → 告警产生 → 系统安全降级/自愈」这条链路，
而非只验证"某函数不抛错"。全部用本地 SimulatedExchange + 假行情，
**不依赖外部网络**（没开 VPN 也能跑）。

场景：
  S1 行情断连（hub 长时间无新K线）→ market_stale CRITICAL 告警 + status 可见
  S2 下单响应丢失（create_order 抛 NetworkError）→ intent 登记 + order_create_lost 告警
  S3 对账差异（本地注册表有、交易所 open 列表未见且仍 open）→ order_reconcile_missing 告警
  S4 部分成交（撤单竞态下实际成交一部分）→ 迟到成交回灌 + order_reconcile_late_fill 告警

判定：每个场景断言「告警 kind 出现」+「系统状态正确」。
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from engine.monitor import AlertMonitor
from engine.order_manager import OrderManager
from engine.trading_engine import TradingEngine
from exchange.simulated import SimulatedExchange
from strategies.base import Signal
from sqlalchemy import select


def _fresh_engine(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'drill.db'}")
    return db


# ---------------- S1 行情断连 ----------------

class _StaleHub:
    def __init__(self, *a, **k):
        self._age = k.get("age", float("inf"))
        self.stopped = False

    async def run(self):
        while not self.stopped:
            await asyncio.sleep(0.05)

    def stop(self):
        self.stopped = True

    def last_kline_age(self, symbol, timeframe):
        return self._age

    def last_price(self, symbol):
        return None

    def snapshot(self, symbol, timeframe):
        return {}


@pytest.mark.asyncio
async def test_s1_market_stale_alert(tmp_path, monkeypatch):
    """S1：行情断流 → CRITICAL 告警 + 引擎 status 暴露 market_stale。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's1.db'}")
    await db.init()
    try:
        import engine.trading_engine as te
        monkeypatch.setattr(te, "MarketDataHub", _StaleHub)
        eng = TradingEngine(db, EventBus())
        # 测试隔离：引擎 monitor 默认写生产 data/logs/alerts.log，必须重定向到 tmp
        eng.monitor.repoint_log_dir(str(tmp_path / "alerts"))
        await eng.start()
        try:
            # 注入断流行情（inf = 从未收到K线）
            eng.hub._age = float("inf")
            await eng.monitor.health_check(eng)
            c = eng.monitor.summary()["counts"]
            # portfolio_loop 每 15s 自动巡检也会触发，故 >=1
            assert c.get("market_stale", 0) >= 1, "断流必须产生 market_stale 告警"
            assert eng.status()["market_stale"] is True, "status 应暴露断流状态"
        finally:
            await eng.stop()
    finally:
        await db.close()


# ---------------- S2 下单响应丢失 ----------------

class _LostCreateExchange(SimulatedExchange):
    """create_order 连续 3 次抛超时（引擎最多重试 2 次 → 必失败），之后恢复。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fail_times = 3

    async def create_order(self, symbol, otype, side, amount, price=None, params=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise asyncio.TimeoutError("simulated create timeout (network)")
        return await super().create_order(symbol, otype, side, amount, price, params)


@pytest.mark.asyncio
async def test_s2_create_lost_intent_and_alert(tmp_path):
    """S2：create 响应丢失 → intent 登记 + order_create_lost 告警，恢复后可认领。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's2.db'}")
    await db.init()
    try:
        ex = _LostCreateExchange(start_cash=10000.0)
        ex.set_last_price(50000.0)
        mon = AlertMonitor(None, log_dir=str(tmp_path / "alerts"))
        om = OrderManager(db, EventBus(), paper=False)
        om.attach_exchange(ex)
        om.set_monitor(mon)

        sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is None, "首次 create 超时不应成交"
        intents = [k for k in om._open_orders if k.startswith("intent:")]
        assert len(intents) == 1, "响应丢失必须登记 intent 待对账认领"
        assert mon.summary()["counts"].get("order_create_lost") == 1, "必须产生下单响应丢失告警"

        # 恢复后：重下单走正常路径（intent 由对账认领）
        fill2 = await om.place(sig, 50000.0, "dual_ma")
        assert fill2 is not None and fill2["qty"] == pytest.approx(0.1)
    finally:
        await db.close()


# ---------------- S3 对账差异 ----------------

class _MissingOrderExchange(SimulatedExchange):
    """fetch_open_orders 不返回某个订单（模拟交易所列表过滤/不一致），
    但该订单本人 fetch_order 仍 open（filled=0）→ 对账"仍 open"分支。"""

    def __init__(self, *a, hidden_ids=None, **k):
        super().__init__(*a, **k)
        self._hidden = set(hidden_ids or [])

    async def fetch_open_orders(self, symbol=None, **k):
        all_o = await super().fetch_open_orders(symbol, **k)
        return [o for o in all_o if o.get("id") not in self._hidden]

    async def fetch_order(self, order_id, symbol=None, **k):
        out = await super().fetch_order(order_id, symbol, **k)
        if order_id in self._hidden:
            out = dict(out)
            out["status"] = "open"
            out["filled"] = 0.0
        return out


@pytest.mark.asyncio
async def test_s3_reconcile_missing_alert(tmp_path):
    """S3：对账差异（本地有、交易所列表未见且仍 open）→ CRITICAL 告警。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's3.db'}")
    await db.init()
    try:
        ex = _MissingOrderExchange(start_cash=10000.0)
        ex.set_last_price(50000.0)
        mon = AlertMonitor(None, log_dir=str(tmp_path / "alerts"))
        om = OrderManager(db, EventBus(), paper=False)
        om.attach_exchange(ex)
        om.set_monitor(mon)

        sig = Signal("BTC/USDT", "buy", qty=0.1, order_type="limit",
                     limit_price=50000.0, strategy="dual_ma")
        fill = await om.place(sig, 50000.0, "dual_ma")
        assert fill is None, "限价单未到价不成交"
        oid = next((k for k in om._open_orders if not k.startswith("intent:")), None)
        assert oid, "限价挂单应登记注册表"

        # 让交易所"看不见"这个订单，但仍 open
        ex._hidden.add(oid)
        await om._reconcile_once()
        c = mon.summary()["counts"]
        assert c.get("order_reconcile_missing") == 1, "对账差异必须产生 CRITICAL 告警"
        assert oid in om._open_orders, "仍 open 的差异订单应保留注册表下轮复查"
    finally:
        await db.close()


# ---------------- S4 迟到成交回灌 ----------------

class _LateFillExchange(SimulatedExchange):
    """fetch_open_orders 隐藏订单；fetch_order 返回 open+filled>0（撤单竞态下已成交）。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._hidden = set()

    async def fetch_open_orders(self, symbol=None, **k):
        all_o = await super().fetch_open_orders(symbol, **k)
        return [o for o in all_o if o.get("id") not in self._hidden]

    async def fetch_order(self, order_id, symbol=None, **k):
        out = await super().fetch_order(order_id, symbol, **k)
        if order_id in self._hidden:
            out = dict(out)
            out["status"] = "open"
            out["filled"] = 0.06
        return out


@pytest.mark.asyncio
async def test_s4_late_fill_backfill_and_alert(tmp_path):
    """S4：撤单竞态下实际已成交 → 迟到成交回灌 + late_fill 告警。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's4.db'}")
    await db.init()
    try:
        ex = _LateFillExchange(start_cash=10000.0)
        ex.set_last_price(50000.0)
        mon = AlertMonitor(None, log_dir=str(tmp_path / "alerts"))
        om = OrderManager(db, EventBus(), paper=False)
        om.attach_exchange(ex)
        om.set_monitor(mon)

        sig = Signal("BTC/USDT", "buy", qty=0.1, order_type="limit",
                     limit_price=50000.0, strategy="dual_ma")
        await om.place(sig, 50000.0, "dual_ma")
        oid = next(k for k in om._open_orders if not k.startswith("intent:"))
        ex._hidden.add(oid)  # 交易所列表不可见 → 对账走复核分支

        await om._reconcile_once()
        c = mon.summary()["counts"]
        assert c.get("order_reconcile_late_fill") == 1, "迟到成交回灌应产生告警"
        async with db.session() as s:
            rows = (await s.execute(select(Trade))).scalars().all()
        assert len(rows) == 1 and rows[0].qty == pytest.approx(0.06), "迟到成交应回灌 Trade 行"
        assert oid not in om._open_orders, "回灌后注册表应清理"
    finally:
        await db.close()