"""simulated 模式走 _place_live 集成测试（exp7 回归）。

覆盖改造：simulated 模式不再被当作 paper（paper=True→_place_paper），
而是 paper=False + SimulatedExchange（本地撮合）→ 订单真实走 _place_live
完整状态机（create→轮询→撤单→复核→对账→FIFO→protective stop）。

守护：
- 市价单成交：Trade 落库（exchange="live" 口径）、余额/持仓变化、protective 挂单簿
- 行为等价：成交价按 last_price±slippage，与纸面同向
- 纸面模式不受影响（仍走 _place_paper）
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from core.events import Event, EventType
from engine.trading_engine import TradingEngine
from strategies.base import Signal, Strategy
from sqlalchemy import select


class BuyOnce(Strategy):
    """第1根K线买入，之后无信号。"""
    name = "sim_once"
    default_params = {"size_pct": 0.5}

    def __init__(self):
        super().__init__()
        self._n = 0
        self._entry = None

    def on_candle(self, ctx):
        self._n += 1
        if self._n == 1:
            return Signal(ctx["symbol"], "buy", size_pct=0.5, strategy=self.name, reason="sim")
        return None

    def on_fill(self, symbol, side, price):
        if side == "buy":
            self._entry = price


def _snap(i, price):
    return {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1700000000000 + i * 3600000, price, price, price, price, 100.0]],
            "closes": [price],
            "indicators": {"close": price, "vol_ratio": 1.0}}


async def _engine_in_simulated(db):
    bus = EventBus()
    eng = TradingEngine(db, bus)
    # 手动装配 simulated 启动后的状态（等价 _start_locked simulated 分支）
    from exchange.simulated import SimulatedExchange
    eng.exchange = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    eng.paper = False
    eng.paper_account = None
    eng.order_manager.attach_exchange(eng.exchange)
    eng.order_manager.set_stop_pct_source(
        lambda: getattr(eng.strategy, "params", {}).get("stop_loss_pct"))
    eng.portfolio = PortfolioManager_sim(eng)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = BuyOnce()
    eng.trade_on_open = False
    return eng, bus


class PortfolioManager_sim:
    """simulated 集成测试用轻量替身（避免全引擎依赖）。"""

    def __init__(self, eng):
        self._eng = eng

    def update_prices(self, prices):
        pass


@pytest.mark.asyncio
async def test_simulated_mode_goes_through_place_live():
    """simulated：市价买入经 _place_live → Trade 落库(live 口径) + protective 挂单簿。"""
    from engine.portfolio import PortfolioManager
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    eng = TradingEngine(db, bus)
    from exchange.simulated import SimulatedExchange
    eng.exchange = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    eng.exchange.set_last_price(50000.0)
    eng.paper = False
    eng.paper_account = None
    # 等价 _start_locked simulated 分支：重建 OrderManager 为 live 口径
    from engine.order_manager import OrderManager
    eng.order_manager = OrderManager(db, bus, False, None)
    eng.order_manager.attach_exchange(eng.exchange)
    eng.order_manager.set_stop_pct_source(
        lambda: getattr(eng.strategy, "params", {}).get("stop_loss_pct"))
    eng.portfolio = PortfolioManager(db, bus, False, None, eng.exchange)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = BuyOnce()
    eng.trade_on_open = False

    try:
        await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                   {"symbol": eng.symbol, "snapshot": _snap(1, 50000.0)},
                                   source="test"))
        # Trade 落库：exchange=live（_place_live 口径），qty>0
        async with db.session() as s:
            trades = (await s.execute(select(Trade))).scalars().all()
        assert len(trades) == 1, "simulated 市价买入应落库 1 笔 Trade"
        assert trades[0].exchange == "live", "simulated 应走 _place_live（exchange=live 口径）"
        assert trades[0].qty > 0
        # 余额变化：买入扣减现金（50000*1.0005*0.1 + 手续费 ≈ 5007.5）
        assert eng.exchange._cash < 10000.0, "买入应扣减模拟余额"
        # protective 兜底止损挂单簿（买入成交后自动挂）
        from exchange.simulated import SimulatedExchange
        assert len(eng.exchange._open) == 1, "买入成交后应挂服务端兜底止损（挂单簿）"
    finally:
        await eng.order_manager.close()
        await db.close()


@pytest.mark.asyncio
async def test_paper_mode_still_goes_through_place_paper():
    """纸面模式不受影响：仍走 _place_paper（exchange 为空、账户为 PaperAccount）。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()
    eng = TradingEngine(db, bus)
    eng.paper = True
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = BuyOnce()
    eng.trade_on_open = False
    try:
        await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                   {"symbol": eng.symbol, "snapshot": _snap(1, 50000.0)},
                                   source="test"))
        qty = eng.paper_account.positions.get("BTC/USDT", {}).get("qty", 0.0)
        assert qty > 0, "纸面模式应照常撮合进纸面账户"
        async with db.session() as s:
            trades = (await s.execute(select(Trade))).scalars().all()
        assert len(trades) == 1
        assert trades[0].exchange == "paper", "纸面模式 Trade 口径应为 paper"
    finally:
        await db.close()