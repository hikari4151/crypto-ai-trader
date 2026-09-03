"""P0-3 重启状态恢复测试：风控连亏/冷却跨重启 + 实盘 FIFO 队列与入口价回填。

覆盖 2026-08-28 修复：
- 连亏/冷却/人工确认状态曾纯内存 → 重启一次就把"连续亏损暂停交易"清零
- 实盘 _live_lots 重启即空 → 卖出成本走兜底、Trade.pnl 失真
- 策略 _entry 重启即 None → 止损判据 `position > 0 and self._entry` 恒不成立，
  交易所里的真实持仓再无止损
"""
import time

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from core.events import EventType
from engine.order_manager import OrderManager
from engine.risk import RiskManager
from engine.trading_engine import TradingEngine
from strategies.base import Signal
from tests.test_reconcile import CollectingBus, FakeDB


# ---------- 风控冷却状态跨重启 ----------

@pytest.mark.asyncio
async def test_cooldown_survives_restart():
    """连亏冷却落库后，新进程（新 RiskManager 实例）应仍然拦截新开仓。"""
    db = FakeDB()
    rm1 = RiskManager(db)
    await rm1.update_rules({"max_consecutive_losses": 2, "cooldown_minutes": 30})
    for _ in range(2):
        await rm1.record_trade(-10.0, "止损")
    # 第一次 check 才真正进入冷却（与 _check 连亏分支同口径）
    ok1, why1 = await rm1.check(Signal("BTC/USDT", "buy", qty=0.1),
                                100.0, 10000.0, 0.0, 0.0, [], None)
    assert not ok1 and "触发冷却" in why1

    # 模拟重启：同一 DB，全新 RiskManager 实例
    rm2 = RiskManager(db)
    await rm2.update_rules({"max_consecutive_losses": 2, "cooldown_minutes": 30})
    await rm2.restore_state()
    assert rm2._cooldown_until > time.time(), "冷却到期时间应跨重启保留"
    assert len(rm2._recent_pnls) == 2, "连亏记录应跨重启保留"
    ok2, why2 = await rm2.check(Signal("BTC/USDT", "buy", qty=0.1),
                                100.0, 10000.0, 0.0, 0.0, [], None)
    assert not ok2 and "冷却中" in why2, f"重启后应仍在冷却，实际: {why2}"


@pytest.mark.asyncio
async def test_clear_cooldown_survives_restart():
    """人工确认解除并落库后，重启不应把熔断又挂回来。"""
    db = FakeDB()
    rm1 = RiskManager(db)
    await rm1.update_rules({"max_consecutive_losses": 1, "cooldown_minutes": 30,
                            "risk_manual_recovery": 1})
    await rm1.record_trade(-10.0, "止损")
    ok_first, _ = await rm1.check(Signal("BTC/USDT", "buy", qty=0.1),
                                  100.0, 10000.0, 0.0, 0.0, [], None)
    assert not ok_first
    rm1.clear_cooldown()
    await rm1.persist_state()

    rm2 = RiskManager(db)
    await rm2.restore_state()
    assert rm2._recent_pnls == [], "已人工解除的连亏记录不得复活"
    ok2, why2 = await rm2.check(Signal("BTC/USDT", "buy", qty=0.1),
                                100.0, 10000.0, 0.0, 0.0, [], None)
    assert ok2, f"重启后应放行新开仓，实际: {why2}"


@pytest.mark.asyncio
async def test_restore_state_is_idempotent_and_tolerant():
    """无存档/坏存档都不应抛异常（风控状态恢复失败不得阻止引擎启动）。"""
    import json
    db = FakeDB()
    rm = RiskManager(db)
    await rm.restore_state()
    await rm.restore_state()  # 幂等：只生效一次
    assert rm._recent_pnls == []
    db.kv["risk_state"] = json.dumps({
        "recent_pnls": [{"nope": 1}, "junk", {"ts": 1.0, "pnl": -2.0}],
        "cooldown_until": "bad"})
    rm2 = RiskManager(db)
    await rm2.restore_state()
    assert len(rm2._recent_pnls) == 1, "脏记录应被逐条丢弃"
    assert rm2._cooldown_until == 0.0


# ---------- 实盘 FIFO 队列回放 ----------

async def _make_db(rows: list[Trade]) -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    async with db.session() as s:
        for r in rows:
            s.add(r)
        await s.commit()
    return db


def _t(side: str, qty: float, price: float, symbol: str = "BTC/USDT",
       exchange: str = "live") -> Trade:
    return Trade(exchange=exchange, symbol=symbol, side=side, price=price, qty=qty,
                 value=qty * price, fee=0.0, strategy="dual_ma")


@pytest.mark.asyncio
async def test_restore_live_state_rebuilds_fifo():
    """buy 1@100 + buy 1@200 - sell 1@150 → 剩余 1 lot@200，入口价回填 200。"""
    db = await _make_db([_t("buy", 1.0, 100.0), _t("buy", 1.0, 200.0), _t("sell", 1.0, 150.0)])
    om = OrderManager(db, EventBus(), paper=False)
    state = await om.restore_live_state("BTC/USDT")
    assert state["qty"] == pytest.approx(1.0)
    assert state["entry"] == pytest.approx(200.0)
    assert om._live_lots["BTC/USDT"] == [(1.0, 200.0)]
    # 重启后第一笔卖出的 FIFO 成本应来自回放队列（曾恒为 None → 走兜底）
    cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 300.0)
    assert cost == pytest.approx(200.0)
    assert "BTC/USDT" not in om._live_lots, "清仓后应回收空 key"
    await om.close()


@pytest.mark.asyncio
async def test_restore_live_state_ignores_other_symbol_and_paper():
    """回放只取本 symbol 的 live 成交：纸面成交/其它 symbol 不得污染队列。"""
    db = await _make_db([
        _t("buy", 1.0, 100.0, symbol="ETH/USDT"),
        _t("buy", 2.0, 50.0, exchange="paper"),
        _t("buy", 1.0, 700.0),
    ])
    om = OrderManager(db, EventBus(), paper=False)
    state = await om.restore_live_state("BTC/USDT")
    assert state["qty"] == pytest.approx(1.0)
    assert state["entry"] == pytest.approx(700.0)
    assert om._live_lots["BTC/USDT"] == [(1.0, 700.0)]
    await om.close()


@pytest.mark.asyncio
async def test_restore_live_state_flat_position_leaves_no_queue():
    """全部平仓后回放：队列为空（不留空 key），入口价仍报末笔买入。"""
    db = await _make_db([_t("buy", 1.0, 100.0), _t("sell", 1.0, 120.0)])
    om = OrderManager(db, EventBus(), paper=False)
    state = await om.restore_live_state("BTC/USDT")
    assert state["qty"] == 0.0
    assert "BTC/USDT" not in om._live_lots
    await om.close()


# ---------- 引擎回填策略入口价 ----------

class _StubStrategy:
    name = "stub"

    def __init__(self) -> None:
        self._entry = None
        self._entry_price = None


class _StubOrderManager:
    def __init__(self, state: dict) -> None:
        self._state = state

    async def restore_live_state(self, symbol: str) -> dict:
        return self._state


@pytest.mark.asyncio
async def test_engine_backfills_entry_and_publishes_event():
    """有未平仓时：回填两个入口价属性 + 发 position_state_restored 事件。"""
    bus = CollectingBus()
    eng = TradingEngine(FakeDB(), bus)
    eng.symbol = "BTC/USDT"
    eng.strategy = _StubStrategy()
    eng.order_manager = _StubOrderManager({"qty": 0.5, "entry": 48000.0, "trades": 3})
    await eng._restore_live_position_state()
    assert eng.strategy._entry == 48000.0
    assert eng.strategy._entry_price == 48000.0
    events = [e for e in bus.events
              if e.type == EventType.SYSTEM and e.payload.get("kind") == "position_state_restored"]
    assert len(events) == 1
    assert events[0].payload["qty"] == 0.5


@pytest.mark.asyncio
async def test_engine_skips_backfill_when_flat():
    """无持仓时不回填（入口价保持 None，避免把历史买入价当成本）。"""
    bus = CollectingBus()
    eng = TradingEngine(FakeDB(), bus)
    eng.symbol = "BTC/USDT"
    eng.strategy = _StubStrategy()
    eng.order_manager = _StubOrderManager({"qty": 0.0, "entry": 48000.0, "trades": 2})
    await eng._restore_live_position_state()
    assert eng.strategy._entry is None
    assert not [e for e in bus.events
                if e.payload.get("kind") == "position_state_restored"]


@pytest.mark.asyncio
async def test_engine_survives_restore_failure():
    """回放异常不得阻断引擎启动（降级为兜底成本，只记日志）。"""
    class _Boom:
        async def restore_live_state(self, symbol: str) -> dict:
            raise RuntimeError("DB 不可用")

    bus = CollectingBus()
    eng = TradingEngine(FakeDB(), bus)
    eng.symbol = "BTC/USDT"
    eng.strategy = _StubStrategy()
    eng.order_manager = _Boom()
    await eng._restore_live_position_state()
    assert eng.strategy._entry is None
