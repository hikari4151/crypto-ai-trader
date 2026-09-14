"""simulated 模式真实 _start_locked 启动路径测试（exp11 回归）。

覆盖（探针 .optim/probe_sim_start.py 验证）：
① db.kv(trading_mode=simulated) + FakeHub → eng.start() 真实启动：
   paper=False、SimulatedExchange、OrderManager(paper=False)、portfolio 装配
② 注入完整指标快照 → 成交端到端（_place_live + protective 挂单）
③ stop 后状态干净（tasks=0），可再次 start（幂等重启）
"""
import asyncio

import pytest

import engine.trading_engine as te
from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.simulated import SimulatedExchange


class FakeHub:
    """替代 MarketDataHub：run 可取消空循环，snapshot 由外部注入。"""

    def __init__(self, *a, **k):
        self.stopped = False
        self._snap = {}

    def set_snapshot(self, snap):
        self._snap = snap

    def snapshot(self, symbol, timeframe):
        return self._snap

    def last_kline_age(self, symbol, timeframe):
        return 1.0

    def set_ma_periods(self, fast, slow):
        pass

    def last_price(self, symbol):
        return None

    async def run(self):
        while not self.stopped:
            await asyncio.sleep(0.02)

    def stop(self):
        self.stopped = True


def _quiet_ai_intervals():
    saved = {}
    for k in ("ai_market_analysis_interval", "ai_optimize_interval", "ai_review_interval"):
        saved[k] = getattr(settings, k)
        setattr(settings, k, 10 ** 12)

    def restore():
        for k, v in saved.items():
            setattr(settings, k, v)
    return restore


def _snap(i, price, ma_fast=50000.0, ma_slow=49900.0):
    return {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1700000000000 + i * 3600000, price, price, price, price, 100.0]],
            "closes": [price],
            "indicators": {"close": price, "vol_ratio": 1.0,
                           "ma_fast": ma_fast, "ma_slow": ma_slow}}


@pytest.mark.asyncio
async def test_simulated_real_start_assembly():
    """simulated 真实启动：装配正确（SimulatedExchange/paper=False/OrderManager live）。"""
    restore = _quiet_ai_intervals()
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    eng = te.TradingEngine(db, bus)
    await db.kv_set("trading_mode", "simulated")
    await db.kv_json_set("live_trading_config", {"exchange": "binance", "symbol": "BTC/USDT",
                                                 "timeframe": "1h", "start_cash": 20000.0,
                                                 "strategy": "dual_ma", "fee_rate": 0.001,
                                                 "slippage": 0.0005, "trade_on_open": False})
    await db.kv_json_set("paper_trading_config", {"start_cash": 10000.0, "fee_rate": 0.001,
                                                  "slippage": 0.0005, "trade_on_open": False})
    orig = te.MarketDataHub
    te.MarketDataHub = FakeHub
    try:
        await eng.start()
        assert eng.running is True, "simulated 启动应成功"
        assert eng.paper is False, "simulated 必须 paper=False（走 _place_live）"
        assert eng.paper_account is None
        assert isinstance(eng.exchange, SimulatedExchange), "simulated 应装配 SimulatedExchange"
        assert eng.order_manager.paper is False, "OrderManager 应为 live 口径"
        assert eng.order_manager.exchange is eng.exchange, "OrderManager 应绑定 SimulatedExchange"
        # 启动资金来自 live_trading_config（simulated 用实盘参数）
        assert eng.exchange._cash == pytest.approx(20000.0), "simulated 应使用实盘配置资金"
        await eng.stop()
        assert eng.running is False and eng._tasks == [], "stop 后状态干净"
        # 幂等重启
        await eng.start()
        assert eng.running is True
        await eng.stop()
        assert eng.running is False
    finally:
        te.MarketDataHub = orig
        restore()
        await db.close()


@pytest.mark.asyncio
async def test_simulated_real_start_end_to_end_fill():
    """simulated 真实启动 + 完整指标快照 → 双均线交叉成交 + protective 挂单。"""
    restore = _quiet_ai_intervals()
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    eng = te.TradingEngine(db, bus)
    await db.kv_set("trading_mode", "simulated")
    await db.kv_json_set("live_trading_config", {"exchange": "binance", "symbol": "BTC/USDT",
                                                 "timeframe": "1h", "start_cash": 20000.0,
                                                 "strategy": "dual_ma", "fee_rate": 0.001,
                                                 "slippage": 0.0005, "trade_on_open": False})
    await db.kv_json_set("paper_trading_config", {"start_cash": 10000.0, "fee_rate": 0.001,
                                                  "slippage": 0.0005, "trade_on_open": False})
    orig = te.MarketDataHub
    te.MarketDataHub = FakeHub
    try:
        await eng.start()
        assert isinstance(eng.exchange, SimulatedExchange)
        # 第1根：ma_fast<ma_slow（无交叉）→ 预热 prev 值
        await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                   {"symbol": eng.symbol, "snapshot": _snap(1, 50000.0,
                                                                            ma_fast=49900.0,
                                                                            ma_slow=50000.0)},
                                   source="ws"))
        # 第2根：ma_fast>ma_slow（上穿）→ 买入信号 → _place_live 成交
        await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                   {"symbol": eng.symbol, "snapshot": _snap(2, 50500.0,
                                                                            ma_fast=50200.0,
                                                                            ma_slow=50100.0)},
                                   source="ws"))
        pos = eng.exchange._positions.get("BTC", 0.0)
        assert pos > 0, "双均线交叉应触发 simulated 买入成交"
        assert eng.order_manager._live_lots.get("BTC/USDT"), "成交应入 FIFO 队列"
        assert eng.order_manager.protective is not None, "simulated 应启用 protective"
        proto = {k: v for k, v in eng.order_manager.protective._stops.items()}
        assert "BTC/USDT" in proto, "买入成交应挂 protective 兜底止损"
        # 余额按实盘配置资金变动
        assert eng.exchange._cash < 20000.0, "买入应扣减余额"
        await eng.stop()
    finally:
        te.MarketDataHub = orig
        restore()
        await db.close()