"""引擎启动失败复位测试（P1-5）。

覆盖 2026-08-14 修复：_start_locked 失败时清理新建资源（exchange.close +
恢复 paper/paper_account/order_manager/portfolio/exchange/hub 快照 +
running=False + 取消已建任务）并 re-raise——main.py 降级逻辑与
/api/trading/start 的 500 行为保持不变；失败后状态干净可重试。
"""
import asyncio

import pytest

import engine.trading_engine as te
from config.settings import settings
from core.bus import EventBus
from core.database import Database


class FakeHub:
    """替代 MarketDataHub：run 为可取消的空循环（不连网），snapshot 返回真值。"""

    def __init__(self, *a, **k):
        self.stopped = False

    async def run(self):
        while not self.stopped:
            await asyncio.sleep(0.05)

    def stop(self):
        self.stopped = True

    def snapshot(self, symbol, timeframe):
        return {"symbol": symbol, "timeframe": timeframe,
                "candles": [], "closes": [], "indicators": {}}


def _quiet_ai_intervals():
    """调大 AI 定时任务间隔，避免测试期间触发真实网络调用；返回恢复函数。"""
    saved = {}
    for k in ("ai_market_analysis_interval", "ai_optimize_interval", "ai_review_interval"):
        saved[k] = getattr(settings, k)
        setattr(settings, k, 10 ** 12)  # 当前 epoch 秒 ~1.78e9，须大于它才不触发

    def restore():
        for k, v in saved.items():
            setattr(settings, k, v)
    return restore


def test_start_failure_resets_state_and_reraise():
    """hub 创建抛错 → 状态复位（对象恢复启动前快照、running False、任务清空）且 re-raise。"""

    def boom_hub(*a, **k):
        raise RuntimeError("simulated hub failure")

    async def run():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init()
        bus = EventBus()
        eng = te.TradingEngine(db, bus)
        # 快照：启动前的对象与 paper 标志
        paper_before = eng.paper
        om_before, pa_before = eng.order_manager, eng.paper_account
        orig = te.MarketDataHub
        te.MarketDataHub = boom_hub
        try:
            with pytest.raises(RuntimeError, match="simulated hub failure"):
                await eng.start()
        finally:
            te.MarketDataHub = orig
        assert eng.running is False, "失败后 running 必须为 False"
        assert eng.hub is None, "失败后 hub 必须恢复为 None"
        assert eng._tasks == [], "失败后任务列表必须清空"
        assert eng.order_manager is om_before, "order_manager 应恢复启动前快照"
        assert eng.paper_account is pa_before, "paper_account 应恢复启动前快照"
        assert eng.paper == paper_before, "paper 应恢复启动前值"
        return True

    assert asyncio.run(run())


def test_start_success_after_failure():
    """失败后可再次启动成功（幂等重试），stop 正常收敛。"""
    restore = _quiet_ai_intervals()

    async def run():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init()
        bus = EventBus()
        eng = te.TradingEngine(db, bus)
        orig = te.MarketDataHub
        te.MarketDataHub = FakeHub
        try:
            await eng.start()
            assert eng.running is True
            assert eng.hub is not None
            assert len(eng._tasks) == 4, "成功态应含 4 个任务（hub/portfolio/ai_scheduler/bus）"
            await eng.stop()
            assert eng.running is False
            await eng.start()  # 再次启动（幂等重试）
            assert eng.running is True
            await eng.stop()
            return True
        finally:
            te.MarketDataHub = orig
            restore()

    assert asyncio.run(run())
