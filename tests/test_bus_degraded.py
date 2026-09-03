"""bus 连续失败计数 + degraded 健康信号测试（P1-3）。

覆盖 2026-08-14 修复：
- 同一 handler 连续失败 ≥5 次 → 发布 SYSTEM 事件（bus_handler_failed），
  health()["degraded"]=True；成功一次清零
- SYSTEM 事件无人订阅时 publish 必须正常返回（红线）
"""
import asyncio

from core.bus import EventBus
from core.events import Event, EventType


def test_failures_degrade_then_reset():
    """5 次连续失败 → degraded + SYSTEM 事件；第 6 次成功 → 计数清零恢复。"""
    bus = EventBus()
    state = {"fail": True}

    async def flaky(ev):
        if state["fail"]:
            raise ValueError("boom")

    sys_events = []

    async def sys_handler(ev):
        sys_events.append(ev)

    bus.subscribe(EventType.MARKET_CANDLE, flaky)
    bus.subscribe(EventType.SYSTEM, sys_handler)

    async def run():
        # SYSTEM 事件经 publish 入队，由 run() 分发循环投递给订阅者
        runner = asyncio.create_task(bus.run())
        try:
            for i in range(5):
                await bus._run_handler(flaky, Event(EventType.MARKET_CANDLE, {"i": i}, source="test"))
            await asyncio.sleep(0.05)  # 等待 run() 分发 SYSTEM 事件
            first = bus.health()
            # 第 6 次成功：计数清零，degraded 恢复 False
            state["fail"] = False
            await bus._run_handler(flaky, Event(EventType.MARKET_CANDLE, {"i": 5}, source="test"))
            second = bus.health()
            return first, second
        finally:
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    first, second = asyncio.run(run())
    assert first["degraded"] is True
    assert len(sys_events) == 1, "第 5 次失败应产生 SYSTEM 事件"
    payload = sys_events[0].payload
    assert payload["kind"] == "bus_handler_failed"
    assert payload["etype"] == EventType.MARKET_CANDLE.value
    assert payload["failures"] == 5
    assert payload["last_error"] == "boom"
    assert "flaky" in payload["handler"]
    assert second["degraded"] is False, "成功一次应清零计数"
    assert first["handlers"], "health 应含 handler 失败明细"


def test_system_unsubscribed_publish_ok():
    """SYSTEM 事件无人订阅时发布必须正常返回（红线：publish 不得抛异常）。"""
    bus = EventBus()

    async def boom(ev):
        raise ValueError("boom")

    bus.subscribe(EventType.ORDER_FILL, boom)

    async def run():
        for i in range(5):
            await bus._run_handler(boom, Event(EventType.ORDER_FILL, {"i": i}, source="test"))
        return bus.health()

    h = asyncio.run(run())
    assert h["degraded"] is True
    assert h["handlers"], "应有失败统计"
    assert any(rec["failures"] >= 5 for rec in h["handlers"].values())


def test_engine_status_has_degraded():
    """status() 追加 degraded/bus 键，现有 6 键保持（/api/health 零改动接线）。"""
    from core.database import Database
    from engine.trading_engine import TradingEngine

    db = Database("sqlite+aiosqlite:///:memory:")
    bus = EventBus()

    async def run():
        await db.init()
        eng = TradingEngine(db, bus)
        return eng.status()

    st = asyncio.run(run())
    keys = ["running", "paper", "mode", "exchange", "symbol", "timeframe", "strategy"]
    for k in keys:
        assert k in st, f"status() 缺少既有键 {k}"
    assert "degraded" in st and st["degraded"] is False
    assert "bus" in st and st["bus"] == {}


def test_engine_status_degraded_reflects_bus_failures():
    """端到端：真实 handler 连续失败 ≥5 次 → engine.status()['degraded']=True。"""
    from core.database import Database
    from engine.trading_engine import TradingEngine

    db = Database("sqlite+aiosqlite:///:memory:")
    bus = EventBus()

    async def run():
        await db.init()
        eng = TradingEngine(db, bus)

        async def bad(ev):
            raise ValueError("boom")

        bus.subscribe(EventType.MARKET_CANDLE, bad)
        runner = asyncio.create_task(bus.run())
        try:
            for i in range(5):
                await bus._run_handler(bad, Event(EventType.MARKET_CANDLE, {"i": i}, source="test"))
            await asyncio.sleep(0.05)
            st = eng.status()
            assert st["degraded"] is True, f"status() 应反映 bus 降级: {st}"
            assert any(r["failures"] >= 5 for r in st["bus"].values()), st["bus"]
            return True
        finally:
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    assert asyncio.run(run())

