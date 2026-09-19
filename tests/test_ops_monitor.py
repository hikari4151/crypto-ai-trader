"""运维监控回归：引擎心跳活性判定 + /api/ops/overview 聚合。

覆盖：
① 引擎 status 暴露 heartbeat_age_sec / engine_loop_alive；
   heartbeat 老化（>60s）→ loop_alive=False（抓"REST 活着但引擎假死"）
② overview 聚合字段齐全，health 聚合判定（CRITICAL/断流/假死 → health=false）
"""
import time

import pytest

from core.bus import EventBus
from core.database import Database
from engine.trading_engine import TradingEngine


def _engine(tmp_path, heartbeat_age=0.0, running=False):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ops.db'}")
    eng = TradingEngine(db, EventBus())
    eng._heartbeat_ts = time.time() - heartbeat_age
    eng.running = running  # 测"假死"须 running=True（未运行不算假死）
    return eng, db


def test_heartbeat_alive_when_recent(tmp_path):
    """心跳新鲜 → 引擎主循环视为活着。"""
    eng, db = _engine(tmp_path, heartbeat_age=5.0, running=True)
    try:
        s = eng.status()
        assert s["heartbeat_age_sec"] == 5
        assert s["engine_loop_alive"] is True
    finally:
        import asyncio
        asyncio.run(db.close())


def test_heartbeat_dead_when_stale(tmp_path):
    """心跳老化（>60s）→ loop_alive=False（抓假死）。"""
    eng, db = _engine(tmp_path, heartbeat_age=120.0, running=True)
    try:
        s = eng.status()
        assert s["heartbeat_age_sec"] >= 120
        assert s["engine_loop_alive"] is False, "心跳停滞必须判为假死"
    finally:
        import asyncio
        asyncio.run(db.close())


def test_heartbeat_always_alive_when_stopped(tmp_path):
    """引擎未运行时不应误报假死（running=False → loop_alive=True）。"""
    eng, db = _engine(tmp_path, heartbeat_age=999.0)
    try:
        eng.running = False
        s = eng.status()
        assert s["engine_loop_alive"] is True, "停引擎不是假死"
    finally:
        import asyncio
        asyncio.run(db.close())


@pytest.mark.asyncio
async def test_ops_overview_aggregates_and_health(tmp_path):
    """overview 聚合：字段齐全 + health 聚合判定。"""
    from web.api.ops import ops_overview

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ops2.db'}")
    await db.init()
    try:
        eng = TradingEngine(db, EventBus())
        eng._heartbeat_ts = time.time() - 10  # 心跳新鲜
        eng.running = True
        out = await ops_overview(engine=eng, db=db)
        assert out["engine"]["loop_alive"] is True
        assert out["engine"]["heartbeat_age_sec"] <= 10
        assert "critical_recent" in out["alerts"] and "counts" in out["alerts"]
        assert "logs" in out and "db" in out
        # 健康 = 无 CRITICAL、无断流、引擎活着
        assert out["health"] is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ops_overview_health_false_on_critical(tmp_path):
    """存在 CRITICAL 告警 → health=false（聚合判定生效）。"""
    from web.api.ops import ops_overview

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ops3.db'}")
    await db.init()
    try:
        eng = TradingEngine(db, EventBus())
        eng.running = True
        # 注入一条 CRITICAL 告警
        eng.monitor.alert("CRITICAL", "test_critical", "注入测试告警", key="t")
        out = await ops_overview(engine=eng, db=db)
        assert out["alerts"]["critical_recent"] >= 1
        assert out["health"] is False, "CRITICAL 告警必须使 health=false"
    finally:
        await db.close()