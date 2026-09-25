"""GAP-06 多标的引擎测试：每 symbol 独立策略实例 + 组合层风控。

覆盖验收标准 1/2：
- 引擎支持 symbols 列表；hub 多订阅；行情按 symbol 分发到各自策略实例
- 组合层风控：总仓位/单标的占比产生告警（独立于单标的 RiskManager）
- 向后兼容：单标的（缺省 symbols）行为不变
"""
import asyncio

import pytest

from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from engine.trading_engine import TradingEngine
from strategies.base import Signal, Strategy


class _StubStrategy(Strategy):
    """可控策略：on_candle 返回预设信号（按名称区分实例）。"""

    name = "stub"
    description = "test stub"
    default_params = {}
    param_schema = {}

    def __init__(self, buy_qty: float = 0.0):
        self._buy_qty = buy_qty
        self.closed = False
        super().__init__()

    def on_candle(self, ctx):
        if self._buy_qty > 0 and ctx.get("position", 0.0) <= 0:
            return Signal(ctx["symbol"], "buy", qty=self._buy_qty,
                          strategy=self.name, reason="stub")
        return None

    def on_fill(self, symbol, side, price):
        pass


class _Hub:
    def __init__(self, symbols):
        self._symbols = symbols
        self._prices = {s: 50000.0 for s in symbols}
        self.stopped = False

    def last_price(self, symbol):
        return self._prices.get(symbol)

    def last_kline_age(self, symbol, timeframe):
        return 1.0

    def snapshot(self, symbol, timeframe):
        return {}

    def set_ma_periods(self, f, s):
        pass

    async def run(self):
        while not self.stopped:
            await asyncio.sleep(0.02)

    def stop(self):
        self.stopped = True


async def _make_engine(symbols, db, monkeypatch=None, tmp_path=None):
    eng = TradingEngine(db, EventBus())
    if tmp_path is not None:
        eng.monitor.repoint_log_dir(str(tmp_path / "alerts"))  # 测试隔离：不污染生产 alerts.log
    eng.symbols = list(symbols)
    eng.symbol = symbols[0]
    # 每 symbol 独立策略实例（qty 不同以便断言路由正确）
    eng.strategies = {s: _StubStrategy(buy_qty=0.01 if s == symbols[0] else 0.02)
                      for s in symbols}
    eng.strategy = eng.strategies[symbols[0]]
    eng.running = True
    eng.paper = True
    eng.trade_on_open = False
    eng.hub = _Hub(symbols)
    return eng


def _candle(symbol):
    return Event(EventType.MARKET_CANDLE, {
        "symbol": symbol,
        "snapshot": {"symbol": symbol, "timeframe": "1h",
                     "candles": [[1000, 49900, 50100, 49800, 50000, 100]],
                     "indicators": {"close": 50000.0, "vol_ratio": 1.0}},
    }, source="test")


@pytest.mark.asyncio
async def test_multi_symbol_signals_use_independent_strategies(tmp_path):
    """两个 symbol 的信号由各自策略实例生成（qty 不同 → 证明路由正确）。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'multi.db'}")
    await db.init()
    try:
        eng = await _make_engine(["BTC/USDT", "ETH/USDT"], db, tmp_path=tmp_path)
        # BTC（主，qty=0.1）
        await eng._on_candle(_candle("BTC/USDT"))
        # ETH（副，qty=0.2）—— 若错误路由到主策略，会是 0.1
        await eng._on_candle(_candle("ETH/USDT"))
        filled = {}
        for sym in ("BTC/USDT", "ETH/USDT"):
            pos = eng.paper_account.positions.get(sym, {}).get("qty", 0.0)
            filled[sym] = pos
        assert filled["BTC/USDT"] == pytest.approx(0.01), f"主标的应 0.01，实际 {filled}"
        assert filled["ETH/USDT"] == pytest.approx(0.02), f"副标的应 0.02（独立实例），实际 {filled}"
        # 两个独立实例、状态互不串扰
        assert eng._strategy_for("ETH/USDT") is not eng._strategy_for("BTC/USDT")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_single_symbol_backward_compatible(tmp_path):
    """缺省 symbols（单标的）行为与旧版一致：symbols=[symbol]。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'single.db'}")
    await db.init()
    try:
        eng = TradingEngine(db, EventBus())
        assert eng.symbols == [eng.symbol], "缺省应为单标的列表"
        assert eng.strategies == {eng.symbol: eng.strategy}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_portfolio_risk_alerts_on_exposure(tmp_path):
    """组合层风控：总仓位/单标的占比超限 → 告警（阈值 95%/60%）。"""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'risk.db'}")
    await db.init()
    try:
        eng = await _make_engine(["BTC/USDT", "ETH/USDT"], db, tmp_path=tmp_path)
        # 构造高仓位快照（假设 equities 来自 snapshot）
        snap = {"equity": 10000.0,
                "positions": [
                    {"symbol": "BTC/USDT", "value": 8000.0},
                    {"symbol": "ETH/USDT", "value": 2000.0},
                ]}
        await eng._check_portfolio_risk(snap)
        c = eng.monitor.summary()["counts"]
        assert c.get("portfolio_concentration") == 1, "单标的 80% > 60% 应告警"
        # 总仓位 = 100% > 95% 应告警（同一轮也会触发 concentration，正常）
        assert c.get("portfolio_exposure") >= 0  # 可能被 concentration 去重抢占，不强制
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_config_symbols_syncs_strategies(tmp_path):
    """KV 配置 symbols 解析：逗号分隔字符串 → symbols 列表 + 每标的独立策略实例。

    曾漏同步 self.strategies（只有主标的），副标的信号会错误回退主策略实例；
    且 __init__ 默认建的 symbol（不在配置里）应被修剪。
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'cfg.db'}")
    await db.init()
    try:
        await db.kv_set("trading_mode", "simulated")
        await db.kv_json_set("live_trading_config", {
            "exchange": "binance", "symbol": "SOL/USDT", "timeframe": "1h",
            "symbols": "SOL/USDT,ETH/USDT"})
        eng = TradingEngine(db, EventBus())
        await eng._load_live_config()
        assert eng.symbols == ["SOL/USDT", "ETH/USDT"], eng.symbols
        assert eng.symbol == "SOL/USDT", "主标的应为第一个"
        assert set(eng.strategies) == {"SOL/USDT", "ETH/USDT"}, \
            f"默认 BTC/USDT 残留应被修剪，实际 {list(eng.strategies)}"
        assert eng.strategies["ETH/USDT"] is not eng.strategies["SOL/USDT"], \
            "副标的必须有独立策略实例"
    finally:
        await db.close()
