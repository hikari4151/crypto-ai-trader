"""GAP-03 回归测试：下单幂等（clientOrderId）+ create 网络异常重试/意图登记。

覆盖 docs/LIVE_READINESS.md GAP-03 验收标准：
1. 普通订单（市价/限价）发送时带稳定 clientOrderId；
2. create_order 网络异常时有限次数重试（复用同一 clientOrderId）；
3. 重试仍失败 → 登记 intent 注册表（对账按 clientOrderId 认领），不静默丢单；
4. 对账能把 intent 认领为交易所真实 order id。
"""
import ccxt
import pytest

from core.bus import EventBus
from engine.order_manager import OrderManager
from exchange.simulated import SimulatedExchange
from strategies.base import Signal
from tests.test_reconcile import FakeDB


class CaptureExchange:
    """记录 create_order 参数并返回已成交市的场单（无 protective 依赖）。"""

    def __init__(self):
        self.calls = []
        self.trigger_created = []

    async def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        self.calls.append((symbol, otype, side, amount, price, dict(params)))
        return {"id": "o1", "symbol": symbol, "type": otype, "side": side,
                "status": "closed", "filled": amount, "amount": amount,
                "average": price or 50000.0, "price": price or 50000.0,
                "fee": {"cost": 0.0}}

    def market(self, symbol):  # 规格查询：给足精度/限额，避免被跳过
        return {"symbol": symbol, "contract": False,
                "precision": {"amount": 1e-8, "price": 1e-6},
                "limits": {"amount": {"min": 1e-6}, "cost": {"min": 0.0}}}

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.8f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.6f}"


class FlakyCreateExchange:
    """前 fail_count 次 create_order 抛网络异常，之后成功；记录每次 clientOrderId。"""

    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.cids: list[str] = []
        self.attempts = 0

    async def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        self.cids.append(params.get("clientOrderId"))
        self.attempts += 1
        if self.fail_count > 0:
            self.fail_count -= 1
            raise ccxt.NetworkError("simulated create network failure")
        return {"id": "o2", "symbol": symbol, "type": otype, "side": side,
                "status": "closed", "filled": amount, "amount": amount,
                "average": price or 50000.0, "price": price or 50000.0,
                "fee": {"cost": 0.0}}

    def market(self, symbol):
        return {"symbol": symbol, "contract": False,
                "precision": {"amount": 1e-8, "price": 1e-6},
                "limits": {"amount": {"min": 1e-6}, "cost": {"min": 0.0}}}

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.8f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.6f}"


def _make_om(exchange) -> OrderManager:
    om = OrderManager(FakeDB(), EventBus(), paper=False)
    om.exchange = exchange  # 不 attach：protective 保持 None，测试聚焦下单路径
    return om


@pytest.mark.asyncio
async def test_place_live_sends_client_order_id():
    """普通订单必须携带幂等 clientOrderId（GAP-03 验收 1）。"""
    ex = CaptureExchange()
    om = _make_om(ex)
    om._record_trade = _fake_record_trade
    sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
    await om._place_live_inner(sig, 50000.0, "t")
    assert len(ex.calls) == 1
    _sym, _type, _side, _qty, _price, params = ex.calls[0]
    cid = params.get("clientOrderId")
    assert cid and cid.startswith("tr"), f"缺 clientOrderId: {params}"
    # OKX clOrdId 约束：纯字母数字 ≤32 位（下划线会被 OKX 拒：51000）
    assert cid.isalnum() and len(cid) <= 32, f"clientOrderId 不兼容 OKX: {cid!r}"


@pytest.mark.asyncio
async def test_create_network_error_retries_same_client_order_id():
    """create 网络异常：有限次重试且复用同一 clientOrderId（GAP-03 验收 2）。"""
    ex = FlakyCreateExchange(fail_count=2)
    om = _make_om(ex)
    om._record_trade = _fake_record_trade
    sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
    result = await om._place_live_inner(sig, 50000.0, "t")
    assert result and result["qty"] == pytest.approx(0.1)
    assert ex.attempts == 3, "2 次失败 + 1 次成功共 3 次尝试"
    assert len(set(ex.cids)) == 1, "重试必须复用同一 clientOrderId（幂等）"
    assert ex.cids[0].startswith("tr") and ex.cids[0].isalnum()


@pytest.mark.asyncio
async def test_create_network_error_registers_intent():
    """create 持续网络失败：登记 intent 供对账认领，不静默丢单（GAP-03 验收 3）。"""
    ex = FlakyCreateExchange(fail_count=99)
    om = _make_om(ex)
    sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
    result = await om._place_live_inner(sig, 50000.0, "t")
    assert result is None
    assert ex.attempts == 3, "重试次数受上限约束"
    intents = [k for k in om._open_orders if k.startswith("intent:")]
    assert len(intents) == 1
    rec = om._open_orders[intents[0]]
    assert rec["pending_create"] is True
    assert rec["client_oid"] == ex.cids[0]
    assert rec["symbol"] == "BTC/USDT" and rec["side"] == "buy"


@pytest.mark.asyncio
async def test_reconcile_adopts_intent_by_client_order_id():
    """对账按 clientOrderId 把 intent 认领为交易所真实订单（GAP-03 验收 4）。"""
    ex = SimulatedExchange(start_cash=10000.0)
    om = OrderManager(FakeDB(), EventBus(), paper=False)
    om.exchange = ex
    sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
    cid = "tr_reconcile_test_0001"
    om._register_create_intent(cid, sig, 0.1, 50000.0)
    try:
        # 交易所侧实际已挂出同 clientOrderId 的限价单（响应丢失场景）
        await ex.create_order("BTC/USDT", "limit", "buy", 0.1, 49000.0,
                              params={"clientOrderId": cid})
        real_oid = next(iter(ex._open.keys()))
        await om._reconcile_once()
        assert not any(k.startswith("intent:") for k in om._open_orders), "intent 应被认领"
        assert real_oid in om._open_orders, "认领后应使用交易所真实 order id 为键"
        assert om._open_orders[real_oid].get("client_oid") == cid
        assert om._open_orders[real_oid].get("pending_create") is not True
    finally:
        await om.close()


async def _fake_record_trade(*a, **k) -> int:
    return 1
