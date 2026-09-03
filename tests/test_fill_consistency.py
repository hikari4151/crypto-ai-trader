"""成交口径一致性 + AI 参数验证门 测试。

覆盖（本次优化 ① ②）：
- T1 纸面市价单叠加滑点（与回测方向一致）
- T2 纸面限价单按限价成交、不叠加滑点
- T3 trade_on_open：信号延迟到下一根K线开盘成交（与回测同口径）
- T4 AI 参数热更新验证门：数据足够且建议==当前 → 通过
- T5 AI 参数热更新验证门：数据不足 → 保守不应用
"""
import asyncio
import sys

import numpy as np
import pytest

sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")

from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from engine.order_manager import OrderManager
from engine.trading_engine import TradingEngine
from exchange.paper import PaperAccount
from strategies.base import Signal, Strategy


def _sqlite_db() -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    return db


# ============ T1. 纸面市价单叠加滑点 ============

@pytest.mark.asyncio
async def test_paper_market_buy_applies_slippage():
    db = _sqlite_db()
    await db.init()
    acc = PaperAccount(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    om = OrderManager(db, EventBus(), paper=True, paper_account=acc)
    sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="dual_ma")
    fill = await om.place(sig, 50000.0, "dual_ma")
    assert fill and fill["qty"] > 0
    # 市价买入：成交价 = 50000 * (1+0.0005)
    assert abs(fill["price"] - 50000.0 * 1.0005) < 1e-6, f"市价买入应叠加正滑点, got {fill['price']}"
    assert acc.positions["BTC/USDT"]["qty"] > 0


@pytest.mark.asyncio
async def test_paper_market_sell_applies_slippage():
    db = _sqlite_db()
    await db.init()
    acc = PaperAccount(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    # 先建仓
    acc.apply_fill("BTC/USDT", "buy", 0.1, 50000.0, 0.0)
    om = OrderManager(db, EventBus(), paper=True, paper_account=acc)
    sig = Signal("BTC/USDT", "sell", size_pct=1.0, strategy="dual_ma")
    fill = await om.place(sig, 51000.0, "dual_ma")
    assert fill and fill["qty"] > 0
    # 市价卖出：成交价 = 51000 * (1-0.0005)
    assert abs(fill["price"] - 51000.0 * 0.9995) < 1e-6, f"市价卖出应叠加负滑点, got {fill['price']}"


# ============ T2. 限价单不叠加滑点 ============

@pytest.mark.asyncio
async def test_paper_limit_no_slippage():
    db = _sqlite_db()
    await db.init()
    acc = PaperAccount(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    om = OrderManager(db, EventBus(), paper=True, paper_account=acc)
    sig = Signal("BTC/USDT", "buy", qty=0.1, order_type="limit", limit_price=49000.0, strategy="dual_ma")
    fill = await om.place(sig, 50000.0, "dual_ma")
    assert fill and fill["qty"] > 0
    # 限价单按限价成交，不叠加滑点
    assert abs(fill["price"] - 49000.0) < 1e-6, f"限价单应按限价成交, got {fill['price']}"


# ============ T3. trade_on_open 延迟到下一根开盘 ============

class StubStrategy(Strategy):
    name = "stub"
    default_params = {}

    def __init__(self):
        super().__init__()
        self._n = 0
        self._entry = None

    def on_candle(self, ctx):
        self._n += 1
        if self._n == 1:
            return Signal(ctx["symbol"], "buy", size_pct=0.5, strategy="stub")
        return None

    def on_fill(self, symbol, side, price):
        if side == "buy":
            self._entry = price


@pytest.mark.asyncio
async def test_trade_on_open_defers_to_next_open():
    from tests.test_reconcile import CollectingBus

    db = _sqlite_db()
    await db.init()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.running = True
    eng.paper = True
    eng.trade_on_open = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = StubStrategy()
    eng._pending_signal = None
    eng.paper_account.slippage = 0.0005

    # 第 1 根K线：生成买入信号，但 trade_on_open → 延迟，不应立即成交
    snap1 = {"symbol": "BTC/USDT", "timeframe": "1h",
             "candles": [[1000, 49900.0, 50100.0, 49800.0, 50000.0, 100.0]],
             "indicators": {"close": 50000.0, "vol_ratio": 1.0}}
    await eng._on_candle(Event(EventType.MARKET_CANDLE, {"symbol": "BTC/USDT", "snapshot": snap1}, source="test"))
    assert eng._pending_signal is not None, "trade_on_open 下信号应延迟到下一根"
    assert eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0) == 0.0, "不应立即成交"
    assert not [e for e in bus.events if e.type == EventType.SIGNAL], "不应发布 SIGNAL"

    # 第 2 根K线：在开盘价成交（open=50100，买入价 = 50100*(1+0.0005)）
    snap2 = {"symbol": "BTC/USDT", "timeframe": "1h",
             "candles": [[2000, 50100.0, 50200.0, 50050.0, 50150.0, 100.0]],
             "indicators": {"close": 50150.0, "vol_ratio": 1.0}}
    await eng._on_candle(Event(EventType.MARKET_CANDLE, {"symbol": "BTC/USDT", "snapshot": snap2}, source="test"))
    qty = eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0)
    assert qty > 0, "应在第 2 根K线开盘成交"
    assert eng._pending_signal is None, "成交后不应再残留待成交信号"
    assert any(e.type == EventType.SIGNAL for e in bus.events), "应发布 SIGNAL 事件"


@pytest.mark.asyncio
async def test_trade_on_open_off_immediate():
    from tests.test_reconcile import CollectingBus

    db = _sqlite_db()
    await db.init()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.running = True
    eng.paper = True
    eng.trade_on_open = False
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = StubStrategy()
    eng._pending_signal = None
    eng.paper_account.slippage = 0.0005

    snap = {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1000, 49900.0, 50100.0, 49800.0, 50000.0, 100.0]],
            "indicators": {"close": 50000.0, "vol_ratio": 1.0}}
    await eng._on_candle(Event(EventType.MARKET_CANDLE, {"symbol": "BTC/USDT", "snapshot": snap}, source="test"))
    # 默认即时成交：信号在当前收盘即成交
    assert eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0) > 0, "默认应即时成交"
    assert eng._pending_signal is None


# ============ T4/T5. AI 参数热更新验证门 ============

def _rows(n: int) -> list:
    t = np.arange(n)
    close = 100.0 + 5.0 * np.sin(t / 20.0)
    return [[1700000000000 + i * 3600000,
             float(close[i] * 0.999), float(close[i] * 1.002),
             float(close[i] * 0.998), float(close[i]), 1000.0]
            for i in range(n)]


class _ValStub(TradingEngine):
    """轻量桩：继承 TradingEngine 以便复用 _validate_param_update/_validation_df，
    但不调用 __init__，手动装配所需属性（避免完整引擎构造）。"""
    def __init__(self, rows):
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.start_cash = 10000.0
        self.paper_fee_rate = 0.001
        self.paper_slippage = 0.0005
        self.hub = None
        self.rows = rows

    async def _fetch_vision_ohlcv(self, limit=800):
        return self.rows[:limit]


_DUAL_MA_PARAMS = {"fast_period": 10, "slow_period": 30, "size_pct": 0.5,
                   "stop_loss_pct": 0.03, "take_profit_pct": 0.06}


@pytest.mark.asyncio
async def test_validate_param_update_identical_passes():
    # 300 根（<400 → 不做过拟合，仅回测对比）；建议==当前 → 通过
    stub = _ValStub(_rows(300))
    ok, info = await TradingEngine._validate_param_update(stub, "dual_ma",
                                                          dict(_DUAL_MA_PARAMS), dict(_DUAL_MA_PARAMS))
    assert isinstance(ok, bool) and isinstance(info, dict)
    assert ok is True, f"建议==当前应通过验证, got {info.get('reason')}"


@pytest.mark.asyncio
async def test_validate_param_update_insufficient_data():
    # 数据不足（<200）→ 保守不应用
    stub = _ValStub(_rows(50))
    ok, info = await TradingEngine._validate_param_update(stub, "dual_ma",
                                                          dict(_DUAL_MA_PARAMS), dict(_DUAL_MA_PARAMS))
    assert ok is False
    assert "数据不足" in info.get("reason", "")
