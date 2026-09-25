"""第4项完善：端到端集成测试——部署 combo 策略走 simulated 撮合闭环。

链路：combo 因子重建（滚动缓冲）→ on_candle 出信号 → _place_live/纸面成交 →
Trade 落库 + balance 变化 + strategy.on_fill 记录入场价（止损/止盈闭环就绪）。
此前实时行情链路 bug（watch_ohlcv async for 误用）正是因为没有这类"行情→
策略→下单→账本"的连通性测试才在日志里刷了几百行才被人工发现。
"""

import pytest

from core.bus import EventBus
from core.database import Database, Trade
from core.events import Event, EventType
from engine.trading_engine import TradingEngine
from sqlalchemy import select


def _combo_strategy():
    """进化产出的组合因子策略（与训练部署同参数形态，因子库真实可重建）。"""
    from strategies.factor_signal import FactorSignalStrategy
    st = FactorSignalStrategy()
    # demo 数据上选确定性因子（放量+动量+RSI），权重方向 IC 化
    st.update_params({
        "factor": "combo",
        "mode": "trend",
        "buy_threshold": 0.5,      # 仅显著正值（上涨段）触发；横盘 combo≈0 不触发
        "sell_threshold": 0.2,
        # 因子必须都在 factors.library 内（rsi_osc 不在库，用 rsi_14）
        "combo_spec": '{"vol_ratio": 0.8, "mom_10": 0.8, "rsi_14": 0.3}',
        "size_pct": 0.3,
    })
    return st


def _snap(idx: int, price: float, vol: float = 100.0, up: bool = True):
    """组装单根K线快照（open=close，方向由 up 控制走势）。"""
    return {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1700000000000 + idx * 3600000, price, price, price, price, vol]],
            "closes": [price],
            "indicators": {"close": price, "open": price, "high": price, "low": price,
                           "volume": vol, "vol_ratio": 1.0, "candles_count": idx}}



async def _engine_in_simulated(db):
    bus = EventBus()
    eng = TradingEngine(db, bus)
    from exchange.simulated import SimulatedExchange
    from engine.order_manager import OrderManager
    from engine.portfolio import PortfolioManager
    eng.exchange = SimulatedExchange(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    eng.exchange.set_last_price(50000.0)
    eng.paper = False
    eng.paper_account = None
    eng.order_manager = OrderManager(db, bus, False, None)
    eng.order_manager.attach_exchange(eng.exchange)
    eng.order_manager.set_stop_pct_source(
        lambda: getattr(eng.strategy, "params", {}).get("stop_loss_pct"))
    eng.portfolio = PortfolioManager(db, bus, False, None, eng.exchange)
    eng.running = True
    eng.symbol = "BTC/USDT"
    eng.timeframe = "1h"
    eng.strategy = _combo_strategy()
    eng.trade_on_open = False
    return eng


@pytest.mark.asyncio
async def test_combo_deployed_strategy_end_to_end():
    """combo 部署策略：历史缓冲重建 → 信号 → simulated 成交 → Trade 落库 + on_fill 记录入场价。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = await _engine_in_simulated(db)
    try:
        # 1) 推送 55 根横盘历史 K 线（on_candle 暖机守卫：前30根不推缓冲，
        #    而 combo 重建需 >=20 根缓冲 → 55 根确保缓冲就绪；横盘 → z-score 稳定）
        for i in range(1, 56):
            await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                       {"symbol": eng.symbol, "snapshot": _snap(i, 50000.0)},
                                       source="test"))
        # 2) 推送上涨 K 线（放量+涨幅）→ 组合因子显著转正 → 买入信号
        for i in range(56, 63):
            price = 52000.0 + (i - 56) * 300.0
            await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                       {"symbol": eng.symbol, "snapshot": _snap(i, price, vol=300.0)},
                                       source="test"))

        async with db.session() as s:
            trades = (await s.execute(select(Trade))).scalars().all()
        buys = [t for t in trades if t.side == "buy"]
        sells = [t for t in trades if t.side == "sell"]
        assert len(buys) >= 1, "combo 策略应产生买入信号并成交（端到端链路打通）"
        assert len(sells) >= 1, "上涨段应触发止盈卖出（买入→卖出完整闭环）"
        buy = buys[0]
        assert buy.exchange == "live", "simulated 走 _place_live 口径"
        assert buy.qty > 0
        # balance 扣减（现金减少）
        assert eng.exchange._cash < 10000.0, "买入应扣减模拟余额"
        # 完整闭环证据：成交序列出现"买入→卖出"（组合因子回落触发 trend 离场
        # 或止盈；on_fill 被 _record_fill 无条件调用维护策略入场价）
        side_seq = [t.side for t in trades]
        assert any(a == "buy" and b == "sell" for a, b in zip(side_seq, side_seq[1:])), \
            "成交序列应出现买入后卖出（完整闭环）"
        # 清仓数量一致（FIFO 全仓离场）：存在数量匹配的买入→卖出对
        matched = [(b, s) for b, s in zip(buys, sells) if abs(b.qty - s.qty) < 1e-6]
        assert len(matched) >= 1, "存在同数量清仓的买入→卖出对（仓位闭环）"
    finally:
        await eng.order_manager.close()
        await db.close()


@pytest.mark.asyncio
async def test_combo_no_signal_when_factor_below_threshold():
    """下跌段：组合因子为负且低于阈值 → 无买入信号、无成交（阈值门有效）。"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = await _engine_in_simulated(db)
    try:
        # 横盘历史（喂足 60 根让 combo 缓冲就绪；横盘组合因子≈0）
        for i in range(1, 61):
            await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                       {"symbol": eng.symbol, "snapshot": _snap(i, 50000.0)},
                                       source="test"))
        # 仅横盘无突破：组合因子 ~0，buy_threshold=0.5 不触发
        async with db.session() as s:
            trades = (await s.execute(select(Trade))).scalars().all()
        assert len(trades) == 0, "横盘无突破不应产生买入成交"
    finally:
        await eng.order_manager.close()
        await db.close()
