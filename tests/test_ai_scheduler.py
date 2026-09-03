"""AI scheduler 锁外调用测试（P1-6）。

覆盖 2026-08-14 修复：_ai_scheduler 曾 `async with self._strategy_lock:
optimize_price_action(...)` 持锁执行 AI 网络调用（最坏数分钟），阻塞全部
K 线处理（_on_candle_locked 需要同一把锁）。修复后锁内仅快照策略名与参数，
AI 调用移出锁，锁内 apply_strategy_params 应用。
"""
import asyncio
import time

from config.settings import settings
from core.bus import EventBus
from core.database import Database
from engine.trading_engine import TradingEngine


def test_ai_scheduler_optimize_outside_lock():
    """优化（慢 AI 调用）进行中，_strategy_lock 可被立即获取（K 线处理不被阻塞）。"""
    restore = []
    for k, v in (("ai_market_analysis_interval", 10 ** 12),
                 ("ai_review_interval", 10 ** 12),
                 ("ai_optimize_interval", 0),
                 ("ai_optimize_validate", False)):  # 本测试聚焦锁行为，关闭验证门
        restore.append((k, getattr(settings, k)))
        setattr(settings, k, v)

    async def run():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init()
        bus = EventBus()
        eng = TradingEngine(db, bus)
        eng.running = True
        # 快照与近期表现：本地构造，不连网
        snap = {"symbol": "BTC/USDT", "timeframe": "1h",
                "candles": [], "closes": [], "indicators": {"close": 50000.0, "sr": {}, "pa": {}}}
        eng._current_snapshot = make_async(snap)

        # 慢 AI 调用：sleep 0.4s 后返回参数（模拟真实网络耗时）
        async def slow_opt(*a, **k):
            await asyncio.sleep(0.4)
            return {"params": {"fast_period": 12}, "reason": "test", "focus": "价格行为与关键位"}

        eng.optimizer.optimize_price_action = slow_opt
        task = asyncio.create_task(eng._ai_scheduler())
        try:
            await asyncio.sleep(0.15)  # 让 scheduler 进入慢优化
            t0 = time.time()
            async with eng._strategy_lock:
                pass  # 模拟 _on_candle 需要取锁
            elapsed = time.time() - t0
            assert elapsed < 0.2, f"优化应持锁外执行，取锁被阻塞 {elapsed:.3f}s"
            await asyncio.sleep(0.4)  # 等优化完成 + apply_strategy_params
            assert eng.strategy.params.get("fast_period") == 12, "优化结果应已应用"
            return True
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            for k, v in restore:
                setattr(settings, k, v)

    assert asyncio.run(run())


def make_async(value):
    async def _fn(*a, **k):
        return value
    return _fn
