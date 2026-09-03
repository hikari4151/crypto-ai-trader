"""P0-5 服务端兜底止损测试：成交后挂触发式市价卖单 + 撤单/收养/降级。

动机（2026-08-28 修复）：引擎止损判定全跑在本机进程里，进程崩溃/断网/休眠
期间交易所侧零保护。现在成交后在交易所挂 STOP_MARKET（比软止损更深），
并靠对账循环把它的成交回灌回本地账本。

覆盖：
- 止损距离 = 策略 stop_loss_pct + buffer，缺配置走默认值，且被钳制在合理区间
- 挂单参数（market/sell/triggerPrice/clientOrderId 前缀，合约加 reduceOnly）
- 触发价不低于现价时不挂（否则等价于瞬时市价平仓）
- 持仓变化才撤旧挂新；已对齐不打交易所
- 撤单失败保留记录且不重挂（否则交易所侧躺两笔止损 → 同一笔持仓卖两次）
- 重启收养只认自己的 clientOrderId，绝不动用户手工挂单
- 挂单失败只降级告警 + SYSTEM 事件，不带崩下单链路
- 兜底单在交易所侧触发 → 对账 forget + 按剩余量重挂
"""
import pytest

from core.bus import EventBus
from core.events import EventType
from engine.order_manager import OrderManager
from engine.protective_stop import STOP_TAG, ProtectiveStopManager
from strategies.base import Signal
from tests.test_reconcile import FakeDB


class StopExchange:
    """支持触发式委托的假交易所。"""

    def __init__(self, balance=None, open_orders=None, fail_create=False, fail_cancel=False):
        self.bal = balance or {}
        self.open = open_orders or []
        self.fail_create = fail_create
        self.fail_cancel = fail_cancel
        self.placed: list[dict] = []
        self.cancelled: list[str] = []
        self._seq = 0

    async def fetch_balance(self):
        return self.bal

    def market(self, symbol: str) -> dict:
        return {"symbol": symbol, "precision": {"amount": 8, "price": 2},
                "limits": {}, "contract": ":" in symbol}

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.8f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    async def create_order(self, symbol, otype, side, amount, price=None, params=None):
        if self.fail_create:
            raise RuntimeError("交易所拒绝条件单")
        self._seq += 1
        order = {"id": f"sid{self._seq}", "symbol": symbol, "type": otype, "side": side,
                 "amount": amount, "price": price, "params": params or {},
                 "status": "open", "filled": 0.0}
        self.placed.append(order)
        return dict(order)

    async def cancel_order(self, order_id, symbol):
        if self.fail_cancel:
            raise RuntimeError("撤单网络超时")
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "canceled"}

    async def fetch_open_orders(self, symbol=None):
        return list(self.open)


def make_ps(exchange, **kw) -> ProtectiveStopManager:
    kw.setdefault("default_stop_pct", 0.07)
    kw.setdefault("buffer_pct", 0.02)
    return ProtectiveStopManager(exchange, EventBus(), **kw)


def make_om(exchange) -> OrderManager:
    om = OrderManager(FakeDB(), EventBus(), paper=False)
    om.attach_exchange(exchange)
    return om


# ---------- 止损距离 ----------

def test_stop_distance_follows_strategy_param():
    ps = make_ps(StopExchange(), stop_pct_source=lambda: 0.03)
    assert ps.stop_distance() == pytest.approx(0.05)


def test_stop_distance_falls_back_to_default():
    assert make_ps(StopExchange()).stop_distance() == pytest.approx(0.09)

    def _boom():
        raise RuntimeError("策略已切换")
    # 来源抛异常不得影响兜底计算
    assert make_ps(StopExchange(), stop_pct_source=_boom).stop_distance() == pytest.approx(0.09)


def test_stop_distance_clamped():
    ps = make_ps(StopExchange(), stop_pct_source=lambda: 9.9)
    assert ps.stop_distance() == pytest.approx(0.52)


# ---------- 挂单参数 ----------

@pytest.mark.asyncio
async def test_sync_places_spot_trigger_order():
    ex = StopExchange()
    ps = make_ps(ex)
    rec = await ps.sync("BTC/USDT", 0.5, 100000.0, 100000.0)
    assert rec and rec["trigger"] == pytest.approx(91000.0)
    order = ex.placed[0]
    assert (order["type"], order["side"], order["amount"]) == ("market", "sell", 0.5)
    assert order["params"]["triggerPrice"] == pytest.approx(91000.0)
    assert order["params"]["clientOrderId"].startswith(STOP_TAG)
    # 现货不能带 reduceOnly（交易所会拒）
    assert "reduceOnly" not in order["params"]


@pytest.mark.asyncio
async def test_sync_contract_adds_reduce_only():
    ex = StopExchange()
    ps = make_ps(ex)
    await ps.sync("BTC/USDT:USDT", 1.0, 50000.0, 50000.0)
    assert ex.placed[0]["params"]["reduceOnly"] is True


@pytest.mark.asyncio
async def test_sync_skips_when_trigger_not_below_price():
    """持仓已深跌到触发价之下：挂出去立刻成交等于市价平仓，本轮交给引擎处理。"""
    ex = StopExchange()
    ps = make_ps(ex)
    assert await ps.sync("BTC/USDT", 1.0, 100000.0, 90000.0) is None
    assert ex.placed == []


@pytest.mark.asyncio
async def test_sync_noop_when_aligned():
    ex = StopExchange()
    ps = make_ps(ex)
    await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0)
    again = await ps.sync("BTC/USDT", 1.0, 100000.0, 99999.0)
    assert len(ex.placed) == 1 and again["id"] == "sid1"


@pytest.mark.asyncio
async def test_sync_replaces_when_position_changes():
    ex = StopExchange()
    ps = make_ps(ex)
    await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0)
    await ps.sync("BTC/USDT", 1.5, 100000.0, 100000.0)   # 加仓 → 撤旧挂新
    assert ex.cancelled == ["sid1"]
    assert len(ex.placed) == 2
    assert ex.placed[1]["amount"] == 1.5


@pytest.mark.asyncio
async def test_sync_zero_qty_releases():
    ex = StopExchange()
    ps = make_ps(ex)
    await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0)
    assert await ps.sync("BTC/USDT", 0.0, 0.0, 100000.0) is None
    assert ex.cancelled == ["sid1"]
    assert ps.status()["stops"] == {}


@pytest.mark.asyncio
async def test_release_keeps_record_when_cancel_fails():
    """撤单失败（断网）必须保留记录且不重挂，否则交易所侧两笔止损会重复卖出。"""
    ex = StopExchange()
    ps = make_ps(ex)
    await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0)
    ex.fail_cancel = True
    assert await ps.release("BTC/USDT") is False
    assert "BTC/USDT" in ps.status()["stops"]
    # 后续对齐尝试撤不掉旧单就不会挂第二笔
    assert await ps.sync("BTC/USDT", 0.4, 100000.0, 100000.0)
    assert len(ex.placed) == 1


@pytest.mark.asyncio
async def test_create_failure_degrades_with_event():
    """挂单失败只降级：软止损仍在，主链路不抛，前端靠 SYSTEM 事件看到"兜底失效"。"""
    ex = StopExchange(fail_create=True)
    bus = EventBus()
    ps = ProtectiveStopManager(ex, bus, default_stop_pct=0.07, buffer_pct=0.02)
    assert await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0) is None
    assert ps.status()["stops"] == {}
    q = bus._queues.get(EventType.SYSTEM)
    assert q is not None, "挂单失败必须发布 SYSTEM 事件"
    kinds = [q.get_nowait().payload.get("kind") for _ in range(q.qsize())]
    assert "protective_stop_failed" in kinds


@pytest.mark.asyncio
async def test_disabled_is_noop():
    ex = StopExchange()
    ps = make_ps(ex, enabled=False)
    assert await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0) is None
    assert await ps.adopt("BTC/USDT") is None
    assert ex.placed == []


# ---------- 重启收养 ----------

@pytest.mark.asyncio
async def test_adopt_only_takes_our_tagged_order():
    ex = StopExchange(open_orders=[
        {"id": "user1", "side": "sell", "amount": 3.0, "remaining": 3.0,
         "triggerPrice": 80000.0, "clientOrderId": "my-manual-stop"},
        {"id": "ours1", "side": "sell", "amount": 1.0, "remaining": 1.0,
         "triggerPrice": 91000.0, "clientOrderId": f"{STOP_TAG}1700000000"},
    ])
    ps = make_ps(ex)
    rec = await ps.adopt("BTC/USDT")
    assert rec == {"id": "ours1", "trigger": 91000.0, "qty": 1.0}
    # 收养后同一持仓再对齐：不打交易所（无裸奔窗口）
    await ps.sync("BTC/USDT", 1.0, 100000.0, 100000.0)
    assert ex.placed == [] and ex.cancelled == []


@pytest.mark.asyncio
async def test_adopt_falls_back_to_info_stop_price():
    ex = StopExchange(open_orders=[
        {"id": "ours2", "side": "sell", "amount": 2.0, "remaining": 2.0,
         "info": {"clientOrderId": f"{STOP_TAG}2", "stopPrice": "88000"}},
    ])
    assert await make_ps(ex).adopt("BTC/USDT") == {"id": "ours2", "trigger": 88000.0, "qty": 2.0}


@pytest.mark.asyncio
async def test_adopt_none_when_query_fails():
    class Boom(StopExchange):
        async def fetch_open_orders(self, symbol=None):
            raise RuntimeError("行情不可达")
    assert await make_ps(Boom()).adopt("BTC/USDT") is None


# ---------- 订单管理器接线 ----------

@pytest.mark.asyncio
async def test_paper_mode_has_no_protective_stop():
    om = OrderManager(FakeDB(), EventBus(), paper=True)
    assert om.protective is None
    om.attach_exchange(StopExchange())
    assert om.protective is None


@pytest.mark.asyncio
async def test_pending_resting_sell_not_double_covered():
    """已挂着的卖单锁住基础币，兜底止损只覆盖剩余未锁定部分。"""
    om = make_om(StopExchange())
    om._live_lots["BTC/USDT"] = [(1.0, 100000.0)]
    om._register_open_order("lim1", Signal("BTC/USDT", "sell", qty=0.4), 0.4, 101000.0)
    await om._resync_protective("BTC/USDT", 100000.0)
    assert om.protective.status()["stops"]["BTC/USDT"]["qty"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_protective_fill_backfill_forgot_and_resyncs():
    """兜底单在交易所侧触发：回灌成交 + 清记录 + 剩余量重挂（这里剩余 0 → 无单）。"""
    ex = StopExchange()
    om = make_om(ex)
    om._live_lots["BTC/USDT"] = [(1.0, 100000.0)]
    await om._resync_protective("BTC/USDT", 100000.0)
    oid = om.protective.status()["stops"]["BTC/USDT"]["id"]
    assert om._open_orders[oid]["protective"] is True

    async def _fake_record(*a, **k):
        return 1
    om._record_trade = _fake_record
    await om._backfill_fill(oid, om._open_orders[oid],
                            {"id": oid, "status": "closed", "filled": 1.0,
                             "average": 90900.0, "fee": {"cost": 0.0}})
    assert om.protective.status()["stops"] == {}
    assert om._live_lots.get("BTC/USDT") is None
    # 平仓后不再挂新单（只有最初那一笔）
    assert len(ex.placed) == 1


@pytest.mark.asyncio
async def test_resync_does_not_cancel_its_own_stop():
    """兜底止损自身也是一条卖单，若被算进"已锁定量"，反复对齐会自己撤掉自己。"""
    ex = StopExchange()
    om = make_om(ex)
    om._live_lots["BTC/USDT"] = [(1.0, 100000.0)]
    for _ in range(3):
        await om._resync_protective("BTC/USDT", 100000.0)
    assert len(ex.placed) == 1
    assert ex.cancelled == []
    assert om.protective.status()["stops"]["BTC/USDT"]["qty"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_restart_adopts_existing_stop_without_replacing():
    """重启后交易所侧仍挂着我们的兜底单：收养即可，撤单重挂会开出裸奔窗口。"""
    from tests.test_restart_state import _make_db, _t
    db = await _make_db([_t("buy", 1.0, 100000.0)])
    ex = StopExchange(open_orders=[{
        "id": "old1", "side": "sell", "amount": 1.0, "remaining": 1.0,
        "triggerPrice": 91000.0, "clientOrderId": f"{STOP_TAG}1"}])
    om = OrderManager(db, EventBus(), paper=False)
    om.attach_exchange(ex)
    state = await om.restore_live_state("BTC/USDT")
    assert state["qty"] == pytest.approx(1.0)
    assert om.protective.status()["stops"]["BTC/USDT"]["id"] == "old1"
    assert ex.placed == [] and ex.cancelled == []
    # 收养的单同样登记进对账注册表：停机期间触发也能回灌成交
    assert om._open_orders["old1"]["protective"] is True
    await om.close()
    await db.close()


@pytest.mark.asyncio
async def test_restart_replaces_lost_stop_using_fifo_avg():
    """单子丢了（被外部撤销）：按回放出的持仓新挂，触发价基于 FIFO 加权均价。"""
    from tests.test_restart_state import _make_db, _t
    db = await _make_db([_t("buy", 1.0, 100000.0), _t("buy", 1.0, 120000.0)])
    om = OrderManager(db, EventBus(), paper=False)
    ex = StopExchange()
    om.attach_exchange(ex)
    await om.restore_live_state("BTC/USDT")
    stop = om.protective.status()["stops"]["BTC/USDT"]
    assert stop["qty"] == pytest.approx(2.0)
    # 均价 110000 ×(1 − 0.09) = 100100（默认止损距离，未注入策略来源）
    assert stop["trigger"] == pytest.approx(110000.0 * 0.91)
    await om.close()
    await db.close()


@pytest.mark.asyncio
async def test_restart_flat_position_places_nothing():
    from tests.test_restart_state import _make_db, _t
    db = await _make_db([_t("buy", 1.0, 100000.0), _t("sell", 1.0, 110000.0)])
    om = OrderManager(db, EventBus(), paper=False)
    ex = StopExchange()
    om.attach_exchange(ex)
    await om.restore_live_state("BTC/USDT")
    assert om.protective.status()["stops"] == {} and ex.placed == []
    await om.close()
    await db.close()


@pytest.mark.asyncio
async def test_reconcile_closed_stop_reestablishes_protection():
    """兜底单被外部撤销（交易所侧已关闭未成交）：清记录并立即重挂，保护不能静默消失。"""
    ex = StopExchange()
    om = make_om(ex)
    om._live_lots["BTC/USDT"] = [(1.0, 100000.0)]
    await om._resync_protective("BTC/USDT", 100000.0)
    oid = om.protective.status()["stops"]["BTC/USDT"]["id"]

    async def _no_open(symbol=None):
        return []

    async def _canceled(order_id, symbol):
        return {"id": order_id, "status": "canceled", "filled": 0.0}

    ex.fetch_open_orders = _no_open
    ex.fetch_order = _canceled
    await om._reconcile_once()
    assert om.protective.status()["stops"]["BTC/USDT"]["id"] != oid
    assert len(ex.placed) == 2
    await om.close()
