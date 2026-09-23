"""GAP-07② 断电恢复演练：trading_auto_start=true 下，服务重启后自动拉起引擎并恢复状态。

覆盖 docs/LIVE_READINESS.md GAP-07 验收 2：
- lifespan 在 trading_auto_start=true 时自动启动交易引擎（无需人工点“启动”）；
- 重启（新进程/新 app 实例、同一 DB）后：
  ① 引擎再次自动运行；
  ② FIFO 成本队列/策略入口价从 Trade 表回放（否则重启后无止损）；
  ③ 风控连亏/冷却状态从 KV 恢复（否则熔断被清零）。

用 simulated 模式 + FakeHub，避免真实行情/交易所；不触碰真实资金。
"""
import time
from types import SimpleNamespace

import pytest

import engine.trading_engine as te
from config.settings import settings
from core.database import Database, Trade
from strategies.base import Signal
from sqlalchemy import select


class FakeHub:
    def __init__(self, *a, **k):
        self.stopped = False

    def snapshot(self, symbol, timeframe):
        return {}

    def last_kline_age(self, symbol, timeframe):
        return 1.0

    def last_price(self, symbol):
        return 50000.0

    def set_ma_periods(self, f, s):
        pass

    async def run(self):
        while not self.stopped:
            await __import__("asyncio").sleep(0.01)

    def stop(self):
        self.stopped = True


async def _noop_async(*a, **k):
    return None


@pytest.mark.asyncio
async def test_auto_start_and_state_restore_drill(tmp_path, monkeypatch):
    db_path = tmp_path / "drill.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr(settings, "trading_auto_start", True)
    # prewarm_proxy 是方法不是字段：pydantic Settings 禁止实例属性覆盖，改 patch 类
    monkeypatch.setattr(type(settings), "prewarm_proxy", _noop_async)
    monkeypatch.setattr(te, "MarketDataHub", FakeHub)
    from drl.evolve_engine import EvolveEngine
    monkeypatch.setattr(EvolveEngine, "start", _noop_async, raising=False)

    # 预置：simulated 模式 + 实盘参数
    seed = Database(f"sqlite+aiosqlite:///{db_path}")
    await seed.init()
    await seed.kv_set("trading_mode", "simulated")
    await seed.kv_json_set("live_trading_config", {
        "exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1h",
        "start_cash": 10000.0, "strategy": "dual_ma"})
    await seed.close()

    import web.main as wm

    # ---------- 第一段：自动启动 → 建仓 + 触发风控冷却 ----------
    app1 = SimpleNamespace(state=SimpleNamespace())
    async with wm.lifespan(app1):
        eng1 = app1.state.engine
        assert eng1.running is True, "trading_auto_start=true 应自动拉起引擎"

        fill = await eng1.order_manager.place(
            Signal("BTC/USDT", "buy", qty=0.1, strategy="dual_ma"), 50000.0, "dual_ma")
        assert fill is not None, "模拟买入应成交"
        assert eng1.order_manager._live_lots.get("BTC/USDT"), "FIFO 应记录持仓"

        await eng1.risk.update_rules({"max_consecutive_losses": 2, "cooldown_minutes": 30})
        await eng1.risk.record_trade(-10.0, "止损")
        await eng1.risk.record_trade(-10.0, "止损")
        ok, why = await eng1.risk.check(Signal("BTC/USDT", "buy", qty=0.1),
                                        100.0, 10000.0, 0.0, 0.0, [], None)
        assert not ok and "冷却" in why, f"应触发连亏冷却，实际: {why}"
        await eng1.risk.persist_state()

    # ---------- 第二段：重启（新 app/引擎，同一 DB） ----------
    app2 = SimpleNamespace(state=SimpleNamespace())
    async with wm.lifespan(app2):
        eng2 = app2.state.engine
        assert eng2.running is True, "重启后 trading_auto_start 应再次自动拉起引擎"
        assert eng2.order_manager._live_lots.get("BTC/USDT"), \
            "重启后 FIFO 成本队列应从 Trade 表回放（否则无止损基准）"
        assert eng2.risk._cooldown_until > time.time(), \
            "重启后连亏冷却应仍生效（风控状态跨重启恢复）"
        async with eng2.db.session() as s:
            n = len((await s.execute(select(Trade))).scalars().all())
        assert n >= 1, "成交应落库"

        # 冷却中的人工/风控拦截仍然生效（重启不清零）
        ok2, why2 = await eng2.risk.check(Signal("BTC/USDT", "buy", qty=0.1),
                                          100.0, 10000.0, 0.0, 0.0, [], None)
        assert not ok2, f"重启后仍应处于冷却，实际: {why2}"