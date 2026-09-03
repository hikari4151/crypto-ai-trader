"""T1-T2 每日对账测试：fetch_live_balance 抽取 + 对账差异三元组 SYSTEM 事件。

覆盖：
- PortfolioManager.fetch_live_balance() 存在且返回正常数据结构
- fetch_live_balance 30s TTL 缓存行为
- _reconcile_positions 对 MISSING_POSITION / UNKNOWN_POSITION / QTY_MISMATCH 发布 SYSTEM 事件
- simulated 模式跳过对账
- 密钥缺失跳过对账
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.portfolio import PortfolioManager
from engine.trading_engine import TradingEngine
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType


class FakeDB:
    """内存 KV，模拟 Database 的 kv 接口（无需真 SQLite）。"""
    def __init__(self):
        self.kv = {}
        self.secrets = {}

    async def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    async def kv_set(self, key, value, is_secret=False):
        self.kv[key] = value

    async def kv_json_get(self, key, default=None):
        import json
        raw = self.kv.get(key)
        if raw is None:
            return default
        return json.loads(raw)

    async def kv_json_set(self, key, value):
        import json
        self.kv[key] = json.dumps(value, ensure_ascii=False)

    async def kv_get_secret(self, key):
        return self.secrets.get(key)

    async def kv_set_secret(self, key, value):
        self.secrets[key] = value

    async def session(self):
        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def execute(self, *a, **k):
                return EmptyResult()

            def add(self, *a, **k):
                pass

            async def commit(self, *a, **k):
                pass

        return FakeSession()

    async def update_trade_pnl(self, *a, **k):
        pass


class EmptyResult:
    def scalars(self):
        return self

    def all(self):
        return []


class FakeExchange:
    """模拟 ExchangeManager：仅提供 fetch_balance。"""
    def __init__(self, balance: dict, has_private: bool = True):
        self._balance = balance
        self.has_private = has_private

    async def fetch_balance(self):
        return self._balance


class FakePaperAccount:
    def __init__(self, positions: dict = None, cash: float = 10000.0):
        self.positions = positions or {}
        self.cash = cash

    def positions_value(self, prices: dict) -> float:
        """与 PaperAccount.positions_value 同口径：按 {symbol: price} 汇总持仓市值。"""
        total = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym) or pos.get("avg_price", 0.0)
            total += float(pos.get("qty", 0.0)) * float(price or 0.0)
        return total


class CollectingBus(EventBus):
    """事件总线，收集所有发布事件供断言。"""
    def __init__(self):
        super().__init__()
        self.events: list[Event] = []

    async def publish(self, event: Event) -> bool:
        self.events.append(event)
        return await super().publish(event)


# ----- T1: fetch_live_balance -----

@pytest.mark.asyncio
async def test_fetch_live_balance_exists():
    """PortfolioManager 有 fetch_live_balance 方法。"""
    db = FakeDB()
    bus = CollectingBus()
    pm = PortfolioManager(db, bus, True, None, None)
    assert hasattr(pm, "fetch_live_balance"), "PortfolioManager 缺少 fetch_live_balance"
    assert callable(pm.fetch_live_balance)


@pytest.mark.asyncio
async def test_fetch_live_balance_no_exchange():
    """无 exchange 时返回空 total。"""
    db = FakeDB()
    bus = CollectingBus()
    pm = PortfolioManager(db, bus, True, None, None)
    result = await pm.fetch_live_balance("k", "s", "")
    assert result == {"total": {}}


@pytest.mark.asyncio
async def test_fetch_live_balance_caches():
    """fetch_live_balance 有 30s TTL 缓存，短时间内重复调用不触发 fetch_balance。"""
    db = FakeDB()
    bus = CollectingBus()
    ex = FakeExchange({"total": {"USDT": 100, "BTC": 0.5}})
    pm = PortfolioManager(db, bus, False, None, ex)
    # 第一次调用：走 fetch_balance
    r1 = await pm.fetch_live_balance("k", "s", "")
    assert r1["total"]["BTC"] == 0.5
    # 修改 exchange 的余额（模拟余额变化，但缓存应返回旧值）
    ex._balance = {"total": {"USDT": 200, "BTC": 1.0}}
    # 第二次调用：应在 30s 缓存内，返回旧值
    r2 = await pm.fetch_live_balance("k", "s", "")
    assert r2["total"]["BTC"] == 0.5, "缓存未命中，应返回缓存旧值"
    # 清空缓存 ts，第三次调用应走新值
    pm._live_balance_ts = 0.0
    r3 = await pm.fetch_live_balance("k", "s", "")
    assert r3["total"]["BTC"] == 1.0, "缓存过期后应返回新值"


# ----- T2: _daily_reconcile_if_due / _reconcile_positions -----

@pytest.mark.asyncio
async def test_reconcile_missing_position():
    """本地有持仓、交易所无 → MISSING_POSITION 事件。"""
    bus = CollectingBus()
    db = FakeDB()
    await db.kv_set_secret("exchange_binance_api_key", "key")
    await db.kv_set_secret("exchange_binance_secret", "secret")
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.exchange_id = "binance"
    eng.symbol = "BTC/USDT"
    # 模拟本地持仓
    eng.paper = False
    eng.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)
    eng.paper_account = None
    # 模拟 _live_balance_cached 返回本地有持仓
    async def fake_live_balance():
        return {"total": {"BTC": 0.5, "USDT": 5000.0}}
    eng._live_balance_cached = fake_live_balance
    # 模拟 fetch_live_balance 返回交易所无持仓
    eng.portfolio.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)

    async def fake_fetch_live_balance(api_key, secret, password):
        return {"total": {"USDT": 10000.0}}  # 交易所无 BTC
    eng.portfolio.fetch_live_balance = fake_fetch_live_balance

    await eng._reconcile_positions()
    # 应发布 MISSING_POSITION
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
    assert len(sys_events) >= 1
    reconcile_events = [e for e in sys_events if e.payload.get("kind") == "reconcile"]
    assert len(reconcile_events) >= 1
    ev = reconcile_events[0]
    assert ev.payload["type"] == "MISSING_POSITION"
    assert "BTC" in ev.payload["symbol"] or "BTC" in str(ev.payload)
    assert ev.payload["local_qty"] > 0
    assert ev.payload["exchange_qty"] == 0.0


@pytest.mark.asyncio
async def test_reconcile_unknown_position():
    """交易所持币、本地无 → UNKNOWN_POSITION 事件。"""
    bus = CollectingBus()
    db = FakeDB()
    await db.kv_set_secret("exchange_binance_api_key", "key")
    await db.kv_set_secret("exchange_binance_secret", "secret")
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.exchange_id = "binance"
    eng.symbol = "BTC/USDT"
    eng.paper = False
    eng.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)
    eng.paper_account = None

    async def fake_live_balance():
        return {"total": {"USDT": 10000.0}}  # 本地无 ETH
    eng._live_balance_cached = fake_live_balance
    eng.portfolio.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)

    async def fake_fetch_live_balance(api_key, secret, password):
        return {"total": {"ETH": 0.5, "USDT": 10000.0}}  # 交易所多出 ETH
    eng.portfolio.fetch_live_balance = fake_fetch_live_balance

    await eng._reconcile_positions()
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM and e.payload.get("kind") == "reconcile"]
    assert len(sys_events) >= 1
    ev = sys_events[0]
    assert ev.payload["type"] == "UNKNOWN_POSITION"
    assert "ETH" in ev.payload["symbol"]
    assert ev.payload["local_qty"] == 0.0
    assert ev.payload["exchange_qty"] > 0


@pytest.mark.asyncio
async def test_reconcile_qty_mismatch():
    """双方都持仓但数量差异 >5% → QTY_MISMATCH 事件。"""
    bus = CollectingBus()
    db = FakeDB()
    await db.kv_set_secret("exchange_binance_api_key", "key")
    await db.kv_set_secret("exchange_binance_secret", "secret")
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.exchange_id = "binance"
    eng.symbol = "BTC/USDT"
    eng.paper = False
    eng.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)
    eng.paper_account = None

    async def fake_live_balance():
        return {"total": {"BTC": 1.0, "USDT": 5000.0}}  # 本地 1 BTC
    eng._live_balance_cached = fake_live_balance
    eng.portfolio.exchange = FakeExchange({"total": {"USDT": 10000.0}}, has_private=True)

    async def fake_fetch_live_balance(api_key, secret, password):
        return {"total": {"BTC": 0.8, "USDT": 10000.0}}  # 交易所 0.8 BTC
    eng.portfolio.fetch_live_balance = fake_fetch_live_balance

    await eng._reconcile_positions()
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM and e.payload.get("kind") == "reconcile"]
    assert len(sys_events) >= 1
    ev = sys_events[0]
    assert ev.payload["type"] == "QTY_MISMATCH"
    assert "BTC" in ev.payload["symbol"]
    assert ev.payload["diff_pct"] > 0.05


@pytest.mark.asyncio
async def test_reconcile_skips_when_key_missing():
    """密钥缺失时跳过对账，无事件。"""
    bus = CollectingBus()
    db = FakeDB()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.exchange_id = "binance"
    await eng._reconcile_positions()
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
    assert len(sys_events) == 0


@pytest.mark.asyncio
async def test_reconcile_skips_non_live():
    """simulated/paper 模式跳过对账。"""
    bus = CollectingBus()
    db = FakeDB()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "simulated"
    # 即使有密钥也不执行
    await db.kv_set_secret("exchange_binance_api_key", "key")
    result = await eng._daily_reconcile_if_due()
    assert result is None  # 跳过
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
    assert len(sys_events) == 0


@pytest.mark.asyncio
async def test_daily_reconcile_if_due_window():
    """_daily_reconcile_if_due 在非 0 点窗口不执行。"""
    bus = CollectingBus()
    db = FakeDB()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng._daily_reconciled_today = False
    eng._last_reconcile_date = ""
    # 不修改时间，直接调用应被窗口条件拦截
    await eng._daily_reconcile_if_due()
    # 没有密钥，但应先通过窗口检查才到密钥检查——由于不在 0 点窗口，不执行
    sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
    assert len(sys_events) == 0
    assert eng._daily_reconciled_today is False