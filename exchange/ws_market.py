"""ccxt.pro WebSocket 行情中枢：订阅 ticker / K线，维护滚动缓冲区并计算指标。

优化：
1. 使用 IncrIndicators 增量计算指标，每根新K线 O(1) 而非 O(n) 全量重算
2. 指数退避重连机制，避免频繁重连
3. 支持多交易对并行订阅（asyncio.gather）
4. 指标缓存：多订阅者共享同一份指标快照
"""
import asyncio
import logging
from typing import Any, Optional

from config.settings import settings
from core.bus import EventBus
from core.events import Event, EventType
from indicators.vectorized import IncrIndicators

log = logging.getLogger(__name__)


class MarketDataHub:
    def __init__(self, exchange_id: str, symbols: list[str], timeframes: list[str], bus: EventBus, max_candles: int = 300) -> None:
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.timeframes = timeframes
        self.bus = bus
        self.max_candles = max_candles
        self._proxy = settings.resolved_proxy or None
        self._running = False
        self._candles: dict[tuple[str, str], list] = {}
        self._last_price: dict[str, float] = {}
        self._lock = asyncio.Lock()
        # 增量指标计算器（每个 symbol+timeframe 一组）
        self._indicators: dict[tuple[str, str], IncrIndicators] = {}
        # 已更新指标的最后一根K线 ts（同一根未收盘K线重复推送不重复计入）
        self._indicator_ts: dict[tuple[str, str], int] = {}

    def last_price(self, symbol: str) -> Optional[float]:
        return self._last_price.get(symbol)

    def snapshot(self, symbol: str, timeframe: str) -> dict:
        """最近 N 根K线 + 最新指标快照（供 AI 解读 / 前端 / 策略）。

        只读不写：指标只在新K线到达时由 _refresh_indicators 增量更新。
        曾在此处反复 update 同一根未收盘K线，导致 SMA/布林/量比缓冲被
        "最后一根收盘价重复"填满，实盘 MA/BB/vol_ratio 信号失真。
        """
        buf = self._candles.get((symbol, timeframe), [])
        if not buf:
            return {}
        closes = [c[4] for c in buf]
        ind = self._snapshot_indicators(symbol, timeframe, buf)
        return {"symbol": symbol, "timeframe": timeframe, "candles": buf[-120:], "closes": closes, "indicators": ind}

    def _refresh_indicators(self, symbol: str, timeframe: str, buf: list) -> dict:
        """增量更新指标（仅新K线）：同一 ts 的重复推送/快照不会重复计入。"""
        key = (symbol, timeframe)
        if not buf:
            return {}
        last_ts = int(buf[-1][0])
        if self._indicator_ts.get(key) == last_ts:
            # 同一根K线（含未收盘的重复推送）已更新过
            return self._indicators[key].snapshot() if key in self._indicators else {}
        incr = self._indicators.get(key)
        if incr is None:
            # 首次：用已有K线预热（最后一根在下面统一处理）
            incr = IncrIndicators()
            self._indicators[key] = incr
            for c in buf[:-1]:
                incr.update(float(c[4]), float(c[5]))
        snap = incr.update(float(buf[-1][4]), float(buf[-1][5]))
        self._indicator_ts[key] = last_ts
        return snap

    def _snapshot_indicators(self, symbol: str, timeframe: str, buf: list) -> dict:
        """读取指标快照（只读）：首次访问时预热增量计算器。"""
        key = (symbol, timeframe)
        incr = self._indicators.get(key)
        if incr is None:
            self._refresh_indicators(symbol, timeframe, buf)
            incr = self._indicators[key]
        snap = incr.snapshot()
        # 暖机守卫依赖 candles_count（回测引擎的快照有此键，实盘路径此前缺失）
        snap["candles_count"] = len(buf)
        return snap

    def _upsert(self, symbol: str, timeframe: str, candles: list) -> bool:
        key = (symbol, timeframe)
        buf = self._candles.setdefault(key, [])
        changed = False
        for c in candles:
            ts = int(c[0])
            if buf and buf[-1][0] == ts:
                buf[-1] = c
            elif (not buf) or buf[-1][0] < ts:
                buf.append(c)
                changed = True
        if len(buf) > self.max_candles:
            del buf[: len(buf) - self.max_candles]
        return changed

    async def run(self) -> None:
        import ccxt.pro as ccxtpro
        if not hasattr(ccxtpro, self.exchange_id):
            raise ValueError(f"ccxt.pro 不支持交易所: {self.exchange_id}")
        cfg: dict[str, Any] = {"enableRateLimit": True}
        if self._proxy:
            # REST 代理：aiohttp_proxy（仅对 REST 生效）
            cfg["aiohttp_proxy"] = self._proxy
            # WebSocket 代理：ccxt.pro 的 watch_* 只认 wsProxy（曾误设
            # httpProxy/httpsProxy——对 WS 无效，导致"REST 测试通过但行情连不通"）
            cfg["wsProxy"] = self._proxy
        ex = getattr(ccxtpro, self.exchange_id)(cfg)
        try:
            await ex.load_markets()
        except Exception:  # noqa: BLE001
            log.warning("[ws] load_markets 失败（网络受限时仍可继续）", exc_info=True)
        self._running = True
        log.info("[ws] 行情中枢启动: %s %s %s", self.exchange_id, self.symbols, self.timeframes)

        async def watch_symbols():
            while self._running:
                for symbol in self.symbols:
                    try:
                        t = await ex.watch_ticker(symbol)
                        self._last_price[symbol] = float(t["last"])
                        await self.bus.publish(Event(EventType.MARKET_TICKER, {"symbol": symbol, "ticker": t}, source="ws"))
                    except Exception as e:  # noqa: BLE001
                        log.warning("[ws] ticker 异常 %s: %s", symbol, e)
                        await asyncio.sleep(1)

        async def watch_ohlcv_for(symbol: str, tf: str) -> None:
            """为单个 symbol+timeframe 订阅K线，带指数退避重连。"""
            backoff = 1
            while self._running:
                try:
                    async for candles in ex.watch_ohlcv(symbol, tf):
                        backoff = 1  # 成功重置延迟
                        changed = self._upsert(symbol, tf, candles)
                        if changed:
                            # 新K线到达：增量更新指标（同 ts 重复推送自动去重），发布事件
                            self._refresh_indicators(symbol, tf, self._candles.get((symbol, tf), []))
                            snap = self.snapshot(symbol, tf)
                            await self.bus.publish(Event(
                                EventType.MARKET_CANDLE,
                                {"symbol": symbol, "timeframe": tf, "candle": candles[-1], "snapshot": snap},
                                source="ws",
                            ))
                except Exception as e:  # noqa: BLE001
                    log.warning("[ws] ohlcv %s/%s 异常: %s，%ds 后重连", symbol, tf, e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)  # 指数退避，最多 30s

        try:
            # 多交易对并行订阅
            tasks = [watch_symbols()]
            for symbol in self.symbols:
                for tf in self.timeframes:
                    tasks.append(watch_ohlcv_for(symbol, tf))
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await ex.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._running = False
