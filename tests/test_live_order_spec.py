"""P0-4 实盘下单口径测试：统一 symbol 解析 + 交易所精度截断 + minNotional。

覆盖 2026-08-28 修复：
- `symbol.split("/")[1]` 对合约统一代码（BTC/USDT:USDT）拆出 "USDT:USDT"，
  余额查不到 → 现金恒 0 → 下单数量算成 0 被静默跳过；对账还会把计价币
  当成一笔未知持仓误报
- 实盘数量/限价原样发给交易所：小数位过多 → 精度拒单，小额 → minNotional 拒单
"""
import math

import pytest

from core.bus import EventBus
from exchange.symbols import base_currency, parse_symbol, quote_currency
from engine.order_manager import OrderManager
from strategies.base import Signal
from tests.test_reconcile import FakeDB


class SpecExchange:
    """带 ccxt 市场规格的假交易所（precision 用 DECIMAL_PLACES 口径）。"""

    def __init__(self, balance: dict = None, precision: dict = None, limits: dict = None,
                 spec_available: bool = True):
        self.bal = balance or {}
        self.precision = precision or {}
        self.limits = limits or {}
        self.spec_available = spec_available
        self.created: list[tuple] = []
        # 触发式委托（服务端兜底止损）单独记账，避免污染普通下单断言
        self.trigger_created: list[tuple] = []

    async def fetch_balance(self):
        return self.bal

    def market(self, symbol: str) -> dict:
        if not self.spec_available:
            raise RuntimeError("markets 未加载")
        return {"symbol": symbol, "precision": self.precision, "limits": self.limits,
                "contract": ":" in symbol}

    @staticmethod
    def _floor(value: float, places) -> float:
        if places is None:
            return value
        factor = 10 ** int(places)
        return math.floor(float(value) * factor) / factor

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{self._floor(amount, self.precision.get('amount')):.8f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{self._floor(price, self.precision.get('price')):.8f}"

    async def create_order(self, symbol, otype, side, amount, price=None, params=None):
        if params:
            self.trigger_created.append((symbol, otype, side, amount, price, params))
            return {"id": "stop1", "status": "open", "filled": 0.0, "amount": amount,
                    "price": price or 0.0, "average": None, "fee": {"cost": 0.0}}
        self.created.append((symbol, otype, side, amount, price))
        return {"id": "oid1", "status": "closed", "filled": amount, "amount": amount,
                "price": price or 0.0, "average": price or 100.0, "fee": {"cost": 0.0}}


def make_om(exchange) -> OrderManager:
    om = OrderManager(FakeDB(), EventBus(), paper=False)
    om.attach_exchange(exchange)
    return om


# ---------- 统一 symbol 解析 ----------

@pytest.mark.parametrize("symbol,expected", [
    ("BTC/USDT", ("BTC", "USDT", "USDT")),
    ("BTC/USDT:USDT", ("BTC", "USDT", "USDT")),
    ("ETH/USDT:USDT", ("ETH", "USDT", "USDT")),
    ("BTC/USD:BTC", ("BTC", "USD", "BTC")),
    ("BTC", ("BTC", "", "")),
])
def test_parse_symbol(symbol, expected):
    assert parse_symbol(symbol) == expected


def test_derivative_quote_is_not_usdt_usdt():
    """合约统一代码的报价段必须是 USDT，而不是 "USDT:USDT"。"""
    assert quote_currency("BTC/USDT:USDT") == "USDT"
    assert base_currency("BTC/USDT:USDT") == "BTC"


@pytest.mark.asyncio
async def test_resolve_live_qty_for_derivative_symbol():
    """比例买单在 BTC/USDT:USDT 上按 USDT 现金换算（曾恒为 0 → 静默不下单）。"""
    ex = SpecExchange(balance={"USDT": {"free": 1000.0, "total": 1000.0}})
    om = make_om(ex)
    qty = await om._resolve_live_qty(Signal("BTC/USDT:USDT", "buy", size_pct=0.5), 50000.0)
    assert qty == pytest.approx(1000.0 * 0.5 / 50000.0)


@pytest.mark.asyncio
async def test_resolve_live_qty_sell_uses_base_free():
    ex = SpecExchange(balance={"BTC": {"free": 0.4, "total": 0.9}})
    om = make_om(ex)
    sig = Signal("BTC/USDT", "sell", size_pct=1.0)
    qty = await om._resolve_live_qty(sig, 50000.0)
    assert qty == pytest.approx(0.4), "全量卖出应按可用（free）量，非 total"


@pytest.mark.asyncio
async def test_resolve_live_qty_sell_partial_by_size_pct():
    """size_pct 决定平仓比例（Signal 默认 0.5）；策略离场信号应显式传 1.0。"""
    ex = SpecExchange(balance={"BTC": {"free": 0.4, "total": 0.9}})
    om = make_om(ex)
    assert await om._resolve_live_qty(Signal("BTC/USDT", "sell"), 50000.0) == pytest.approx(0.2)
    assert await om._resolve_live_qty(
        Signal("BTC/USDT", "sell", size_pct=0.25), 50000.0) == pytest.approx(0.1)


# ---------- 精度截断 / minNotional ----------

def test_fit_live_qty_truncates_to_step():
    om = make_om(SpecExchange(precision={"amount": 3}))
    assert om._fit_live_qty("BTC/USDT", 0.123456789, 50000.0, "buy") == pytest.approx(0.123)


def test_fit_live_qty_rejects_below_min_notional():
    """买单名义额低于 minNotional 直接跳过（交易所必拒单）。"""
    om = make_om(SpecExchange(precision={"amount": 6},
                              limits={"cost": {"min": 10.0}}))
    assert om._fit_live_qty("ETH/USDT", 0.0001, 3000.0, "buy") == 0.0


def test_fit_live_qty_never_zeroes_a_sell():
    """卖单截断为 0 时原样返回：把止损单截成 0 等于不平仓。"""
    om = make_om(SpecExchange(precision={"amount": 2}, limits={"cost": {"min": 100.0}}))
    assert om._fit_live_qty("BTC/USDT", 0.0015, 50000.0, "sell") == pytest.approx(0.0015)


def test_fit_live_qty_degrades_without_specs():
    """规格不可用（未 load_markets/未知 symbol/假交易所）→ 原样返回，不下假单。"""
    om = make_om(SpecExchange(spec_available=False))
    assert om._fit_live_qty("BTC/USDT", 0.123456789, 50000.0, "buy") == pytest.approx(0.123456789)
    assert om._fit_live_price("BTC/USDT", 100.123456) == pytest.approx(100.123456)


def test_fit_live_price_rounds_to_tick():
    om = make_om(SpecExchange(precision={"price": 1}))
    assert om._fit_live_price("BTC/USDT", 50000.987) == pytest.approx(50000.9)
    assert om._fit_live_price("BTC/USDT", None) is None


@pytest.mark.asyncio
async def test_place_live_sends_fitted_amount_and_price():
    """端到端：create_order 收到的数量/限价都已按规格修正。"""
    ex = SpecExchange(precision={"amount": 3, "price": 1},
                      limits={"cost": {"min": 5.0}})
    om = make_om(ex)

    async def _fake_record(*a, **k):
        return 1
    om._record_trade = _fake_record
    sig = Signal("BTC/USDT", "buy", qty=0.1234567, limit_price=50000.987)
    result = await om._place_live(sig, 50001.0, "dual_ma")
    assert len(ex.created) == 1
    _sym, _type, _side, amount, price = ex.created[0]
    assert amount == pytest.approx(0.123)
    assert price == pytest.approx(50000.9)
    assert result and result["qty"] == pytest.approx(0.123)
    # P0-5：买入成交后立即挂服务端兜底止损（触发价 = 均价 ×(1 − 0.07 − 0.02)）
    assert len(ex.trigger_created) == 1
    _s, otype, side, s_qty, s_price, s_params = ex.trigger_created[0]
    assert (otype, side, s_price) == ("market", "sell", None)
    assert s_qty == pytest.approx(0.123)
    assert s_params["triggerPrice"] == pytest.approx(50000.9 * 0.91, rel=1e-3)
    assert str(s_params["clientOrderId"]).startswith("ps_")
    assert om._open_orders["stop1"]["protective"] is True
