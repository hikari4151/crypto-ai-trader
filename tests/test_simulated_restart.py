"""simulated 重启状态保留回归测试（exp17 修复）。

覆盖 bug：simulated 模式 _start_locked 每次新建 SimulatedExchange → stop→start
重启即丢现金/持仓/成交历史（等价实盘重启不应丢交易所持仓）。修复：重启复用
既有实例 + 回放 FIFO/入口价。
"""
import asyncio

import pytest

import engine.trading_engine as te
from core.bus import EventBus
from core.database import Database, Trade
from core.events import Event, EventType
from exchange.simulated import SimulatedExchange
from sqlalchemy import select, func


class FakeHub:
    def __init__(self, *a, **k):
        self.stopped = False

    def snapshot(self, symbol, timeframe):
        return {}

    def last_kline_age(self, symbol, timeframe):
        return 1.0

    def set_ma_periods(self, f, s):
        pass

    def last_price(self, symbol):
        return None

    async def run(self):
        while not self.stopped:
            await asyncio.sleep(0.01)

    def stop(self):
        self.stopped = True


def _ind(close: float) -> dict:
    return {"close": close, "vol_ratio": 1.0,
            "ma_fast": close * 1.001, "ma_slow": close * 0.999}


async def _setup() -> tuple[Database, te.TradingEngine]:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = te.TradingEngine(db, EventBus())
    await db.kv_set("trading_mode", "simulated")
    await db.kv_json_set("live_trading_config", {"exchange": "binance", "symbol": "BTC/USDT",
                                                 "timeframe": "1h", "start_cash": 10000.0,
                                                 "strategy": "dual_ma", "fee_rate": 0.001,
                                                 "slippage": 0.0005, "trade_on_open": False})
    await db.kv_json_set("paper_trading_config", {"start_cash": 10000.0, "fee_rate": 0.001,
                                                  "slippage": 0.0005, "trade_on_open": False})
    return db, eng


def _kline(ts: int, close: float):
    return {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[ts, close, close + 1, close - 1, close, 100.0]],
            "closes": [close], "indicators": _ind(close)}


@pytest.mark.asyncio
async def test_simulated_restart_preserves_account_state():
    """stop→start 重启后现金/持仓/成交历史应保留（曾重置回 10000）。"""
    db, eng = await _setup()
    orig = te.MarketDataHub
    te.MarketDataHub = FakeHub
    try:
        await eng.start()
        ex0 = eng.exchange
        assert isinstance(ex0, SimulatedExchange)
        # 直接注入成交（模拟第一段运行产生账户变化）
        from strategies.base import Signal
        sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
        fill = await eng.order_manager.place(sig, 50000.0, "t")
        assert fill is not None
        cash_before = ex0._cash
        assert cash_before < 10000.0, "买入后现金应扣减"

        # 重启
        await eng.stop()
        await eng.start()
        # 关键断言：同一 exchange 实例被复用（账户真相保留）或状态等价
        assert eng.exchange is ex0 or eng.exchange._cash < 10000.0, \
            "重启不应把现金重置回 10000（exp17）"
        async with db.session() as s:
            nt = (await s.execute(select(func.count()).select_from(Trade))).scalar_one()
        assert nt >= 1, "成交应落库"
    finally:
        await eng.stop()
        te.MarketDataHub = orig
        await db.close()


@pytest.mark.asyncio
async def test_simulated_restart_fifo_restored():
    """重启后 FIFO 成本队列可从 Trade 回放（_restore_live_position_state 对 simulated 生效）。"""
    db, eng = await _setup()
    orig = te.MarketDataHub
    te.MarketDataHub = FakeHub
    try:
        await eng.start()
        from strategies.base import Signal
        sig = Signal("BTC/USDT", "buy", qty=0.1, strategy="t")
        fill = await eng.order_manager.place(sig, 50000.0, "t")
        assert fill is not None
        lots_before = eng.order_manager._live_lots.get("BTC/USDT", [])
        assert len(lots_before) >= 1

        await eng.stop()
        await eng.start()
        # FIFO 应在重启后从 Trade 回放重建
        lots_after = eng.order_manager._live_lots.get("BTC/USDT", [])
        assert len(lots_after) >= 1, "重启后 FIFO 队列应回放保留（exp17）"
    finally:
        await eng.stop()
        te.MarketDataHub = orig
        await db.close()