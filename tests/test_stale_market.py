"""T4-T5 断链自动平仓测试：last_klines_ts + 断流阈值 + 引擎检查。

覆盖：
- MarketDataHub 有 last_klines_ts / max_stale_seconds / last_kline_age
- _upsert 新K线后 last_klines_ts 更新
- Settings.max_stale_seconds 默认 None（按周期自动取 2 根 K 线），0=显式关闭
- _stale_threshold 按周期推导；<=0 时 _check_market_stale 跳过
- live 断链 + 有持仓 → 真正下市价平仓单（订单必须到达 order_manager，
  曾只 publish 无人订阅的 SIGNAL 事件却对外宣称"已触发自动平仓"）
- 平仓失败要如实上报（SYSTEM stale_close_failed），不得假称已平仓
- simulated 模式只告警不平仓
- 恢复后 market_recovered 事件
"""
import asyncio
import time

import pytest

from config.settings import settings
from core.bus import EventBus
from core.events import Event, EventType
from exchange.ws_market import MarketDataHub


class CollectingBus(EventBus):
    """事件总线，收集所有发布事件供断言。"""
    def __init__(self):
        super().__init__()
        self.events: list[Event] = []

    async def publish(self, event: Event) -> bool:
        self.events.append(event)
        return await super().publish(event)


# ----- T4: last_klines_ts -----

@pytest.mark.asyncio
async def test_hub_has_last_klines_ts():
    """MarketDataHub 初始化后有 last_klines_ts 和 max_stale_seconds。"""
    bus = CollectingBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    assert hasattr(hub, "last_klines_ts")
    assert isinstance(hub.last_klines_ts, dict)
    assert hub.max_stale_seconds == settings.max_stale_seconds


@pytest.mark.asyncio
async def test_hub_has_last_kline_age():
    """MarketDataHub 有 last_kline_age 方法。"""
    bus = CollectingBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    assert hasattr(hub, "last_kline_age")
    age = hub.last_kline_age("BTC/USDT", "1h")
    assert age == float("inf"), "无数据时应返回 inf"


@pytest.mark.asyncio
async def test_upsert_updates_last_klines_ts():
    """_upsert 新K线后 last_klines_ts 更新。"""
    bus = CollectingBus()
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    key = ("BTC/USDT", "1h")
    assert hub.last_klines_ts.get(key) is None

    # 模拟新K线
    ts = int(time.time() * 1000)
    changed = hub.apply_ohlcv("BTC/USDT", "1h", [[ts, 100.0, 110.0, 90.0, 105.0, 1000.0]])
    # _upsert_locked 中 changed=True 时更新 last_klines_ts
    assert changed is True


# ----- T5: Settings.max_stale_seconds -----

def test_settings_max_stale_seconds_default():
    """默认 None=按周期自动判定断流（0 才是显式关闭）。"""
    assert hasattr(settings, "max_stale_seconds")
    assert settings.max_stale_seconds is None


def test_stale_threshold_derived_from_timeframe():
    """阈值未配置时按周期取 2 根 K 线；显式配置值优先；0 表示关闭。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus

    eng = TradingEngine(FakeDB(), CollectingBus())
    original = settings.max_stale_seconds
    try:
        settings.max_stale_seconds = None
        eng.timeframe = "1h"
        assert eng._stale_threshold() == 7200
        eng.timeframe = "1m"
        assert eng._stale_threshold() == 120
        settings.max_stale_seconds = 300
        assert eng._stale_threshold() == 300
        settings.max_stale_seconds = 0
        assert eng._stale_threshold() == 0
    finally:
        settings.max_stale_seconds = original


# ----- T5: _check_market_stale (engine) -----

@pytest.mark.asyncio
async def test_check_market_stale_disabled_by_default():
    """max_stale_seconds=0 时 _check_market_stale 跳过。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    # 确保 max_stale_seconds=0
    original = settings.max_stale_seconds
    settings.max_stale_seconds = 0
    try:
        # 没有 hub 时应直接返回
        eng.hub = None
        await eng._check_market_stale()
        # 无事件
        sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
        assert len(sys_events) == 0
    finally:
        settings.max_stale_seconds = original


@pytest.mark.asyncio
async def test_check_market_stale_skips_without_hub():
    """无 hub 时 _check_market_stale 跳过。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.hub = None
    original = settings.max_stale_seconds
    settings.max_stale_seconds = 300
    try:
        await eng._check_market_stale()
        sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
        assert len(sys_events) == 0
    finally:
        settings.max_stale_seconds = original


@pytest.mark.asyncio
async def test_check_market_stale_alerts_in_simulated():
    """simulated 模式断链时只告警（SYSTEM 事件）不平仓（无 SIGNAL 事件）。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "simulated"
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    eng.hub = hub
    eng._stale_alerted = False

    original = settings.max_stale_seconds
    settings.max_stale_seconds = 1  # 1 秒超时
    try:
        # last_klines_ts 为空 → last_kline_age 返回 inf → 超时
        await eng._check_market_stale()
        # 应有 SYSTEM(market_stale) 事件
        sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
        stale_events = [e for e in sys_events if e.payload.get("kind") == "market_stale"]
        assert len(stale_events) >= 1, "simulated 模式也应有 market_stale 告警"
        # 不应有 SIGNAL 事件（不平仓）
        sig_events = [e for e in bus.events if e.type == EventType.SIGNAL]
        assert len(sig_events) == 0, "simulated 模式不应发平仓 SIGNAL"
    finally:
        settings.max_stale_seconds = original


@pytest.mark.asyncio
async def test_check_market_stale_flatterns_in_live(monkeypatch):
    """live 模式断链 + 有持仓 → 真正下市价平仓单（订单必须到达 order_manager）。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus, FakePaperAccount

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    # 模拟有持仓
    eng.paper = True
    eng.paper_account = FakePaperAccount(positions={"BTC/USDT": {"qty": 0.5, "avg_price": 50000.0}})
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    eng.hub = hub
    eng._stale_alerted = False

    # 拦截下单：断言"订单真的发出"，而不是旧口径的"事件真的发布"
    placed: list = []

    async def _fake_place(signal, last_price, strategy_name):
        placed.append(signal)
        return {"price": 50000.0, "qty": signal.qty, "fee": 0.5, "trade_id": 1,
                "cost_price": 49000.0}
    monkeypatch.setattr(eng.order_manager, "place", _fake_place)
    monkeypatch.setattr(eng, "_record_fill", lambda *a, **k: asyncio.sleep(0))

    original = settings.max_stale_seconds
    settings.max_stale_seconds = 1
    try:
        await eng._check_market_stale()
        sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
        stale_events = [e for e in sys_events if e.payload.get("kind") == "market_stale"]
        assert len(stale_events) >= 1, "live 模式应有 market_stale 告警"
        # 核心：平仓单真的发出，数量=全部持仓
        assert len(placed) == 1, "断链时未真正下平仓单（旧缺陷：只发事件不下单）"
        assert placed[0].side == "sell"
        assert placed[0].qty == 0.5
        assert "交易所断连自动平仓" in placed[0].reason
        assert not [e for e in sys_events if e.payload.get("kind") == "stale_close_failed"]
    finally:
        settings.max_stale_seconds = original


@pytest.mark.asyncio
async def test_stale_close_failure_is_reported(monkeypatch):
    """平仓下单失败时如实上报，不得假称"已触发自动平仓"。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus, FakePaperAccount

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.paper = True
    eng.paper_account = FakePaperAccount(positions={"BTC/USDT": {"qty": 0.5}})
    eng.hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    eng._stale_alerted = False

    async def _boom(signal, last_price, strategy_name):
        raise RuntimeError("交易所拒单：余额不足")
    monkeypatch.setattr(eng.order_manager, "place", _boom)

    original = settings.max_stale_seconds
    settings.max_stale_seconds = 1
    try:
        await eng._check_market_stale()
        failed = [e for e in bus.events if e.payload.get("kind") == "stale_close_failed"]
        assert len(failed) == 1, "下单失败必须留下 stale_close_failed 事件"
        assert "余额不足" in failed[0].payload["detail"]
        assert eng._stale_alerted is True, "失败后仍处告警态，下轮不重复刷屏"
    finally:
        settings.max_stale_seconds = original


@pytest.mark.asyncio
async def test_check_market_stale_recovered():
    """断链恢复后发布 market_recovered 事件。"""
    from engine.trading_engine import TradingEngine
    from tests.test_reconcile import FakeDB, CollectingBus

    db = FakeDB()
    bus = CollectingBus()
    eng = TradingEngine(db, bus)
    eng.trading_mode = "live"
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    hub = MarketDataHub("binance", ["BTC/USDT"], ["1h"], bus)
    eng.hub = hub

    original = settings.max_stale_seconds
    settings.max_stale_seconds = 3600  # 1h，确保不超时
    try:
        # 先设置 _stale_alerted=True 模拟已告警
        eng._stale_alerted = True
        # 给 hub 一个最近的时间戳，确保不超时
        hub.last_klines_ts[("BTC/USDT", "1h")] = time.time()
        await eng._check_market_stale()
        # 应有 SYSTEM(market_recovered) 事件
        sys_events = [e for e in bus.events if e.type == EventType.SYSTEM]
        recovered = [e for e in sys_events if e.payload.get("kind") == "market_recovered"]
        assert len(recovered) >= 1, "恢复后应有 market_recovered 事件"
        assert eng._stale_alerted is False, "恢复后 _stale_alerted 应重置"
    finally:
        settings.max_stale_seconds = original