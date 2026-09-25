"""TDD：MarketDataHub REST/本地回退（2026-09-23）。

根因：simulated 引擎依赖 ccxt.pro WebSocket 行情，初始化需 binance
exchangeInfo/klines 端点；本环境 binance 不可达 → hub 无行情 → market_stale=-1
→ 模拟实盘不交易。本地 kline_store 缓存有 26414 根历史 K 线可读。
修复：hub 加 `rest_fallback`（默认开）——WS 之外并行 REST/本地轮询
（load_klines_cached 本地优先），无外网时仍有用本地行情驱动模拟盘。
"""
import pandas as pd
import pytest

from core.bus import EventBus
from core.events import EventType
from exchange.ws_market import MarketDataHub


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


def _df(n=5):
    idx = pd.date_range("2026-09-23 00:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": [2500.0 + i for i in range(n)],
        "high": [2510.0 + i for i in range(n)],
        "low": [2490.0 + i for i in range(n)],
        "close": [2505.0 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    }, index=idx)


@pytest.mark.asyncio
async def test_fallback_poll_fills_hub_and_publishes(monkeypatch):
    hub = MarketDataHub("binance", ["ETH/USDT"], ["5m"], EventBus(), rest_fallback=True)
    bus = _FakeBus()
    hub.bus = bus

    async def _fake_load(exchange, symbol, timeframe, limit=0, max_candles=0):
        return _df(5)
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_load)

    changed = await hub._fallback_poll_once("ETH/USDT", "5m")
    assert changed is True
    snap = hub.snapshot("ETH/USDT", "5m")
    assert len(snap.get("candles", [])) == 5
    # 发布了 MARKET_CANDLE 事件
    types = [getattr(e, "type", None) for e in bus.events]
    assert EventType.MARKET_CANDLE in types


@pytest.mark.asyncio
async def test_fallback_poll_idempotent_no_duplicate(monkeypatch):
    hub = MarketDataHub("binance", ["ETH/USDT"], ["5m"], EventBus(), rest_fallback=True)
    bus = _FakeBus()
    hub.bus = bus

    async def _fake_load(exchange, symbol, timeframe, limit=0, max_candles=0):
        return _df(5)
    monkeypatch.setattr("backtest.data_loader.load_klines_cached", _fake_load)

    await hub._fallback_poll_once("ETH/USDT", "5m")
    n_after_first = len(hub.snapshot("ETH/USDT", "5m").get("candles", []))
    changed2 = await hub._fallback_poll_once("ETH/USDT", "5m")
    # 同一批数据幂等：无新K线 → changed=False、缓冲不重复增长
    assert changed2 is False
    assert len(hub.snapshot("ETH/USDT", "5m").get("candles", [])) == n_after_first


@pytest.mark.asyncio
async def test_fallback_disabled_does_nothing(monkeypatch):
    hub = MarketDataHub("binance", ["ETH/USDT"], ["5m"], EventBus(), rest_fallback=False)
    changed = await hub._fallback_poll_once("ETH/USDT", "5m")
    assert changed is False
    assert hub.snapshot("ETH/USDT", "5m") == {}
