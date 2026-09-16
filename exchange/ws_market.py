"""ccxt.pro WebSocket 行情中枢：订阅 ticker / K线，维护滚动缓冲区并计算指标。

优化：
1. 使用 IncrIndicators 增量计算指标，每根新K线 O(1) 而非 O(n) 全量重算
2. 指数退避重连机制，避免频繁重连
3. 支持多交易对并行订阅（asyncio.gather）
4. 指标缓存：多订阅者共享同一份指标快照
"""
import asyncio
import logging
import threading
import time
from typing import Any, Optional

from config.settings import settings
from core.bus import EventBus
from core.events import Event, EventType
from indicators.vectorized import IncrIndicators

log = logging.getLogger(__name__)


def _compute_sr_pa(candles: list) -> tuple[dict, dict]:
    """由 K 线窗口计算关键位(sr) + 价格行为(pa)（snapshot 补算用，纯函数便于测试）。

    窗口口径与 backtest/fast_engine.py:85 一致（window=10, min_touches=2）；
    len<21 时 pa 返回 {}（合法空态），sr 在短窗下全 None——与 fast_engine 短数据行为一致。
    暖机守卫（candles_count<30）由消费方 price_action.py:48 负责，不在此拦截。
    脏数据兜底：交易所返回 None/NaN 字段时返回空 sr/pa + warning，
    异常不得外溢到行情循环（曾类型错误中断整条 K 线流，引擎"看似运行实则已死"）。
    """
    empty_sr = {"support": None, "resistance": None, "broken_resistance": None,
                "at_support": False, "at_resistance": False,
                "support_levels": [], "resistance_levels": [],
                "distance_to_support": None, "distance_to_resistance": None}
    if len(candles) < 21:
        return empty_sr, {}
    try:
        import numpy as np
        from indicators.technical import price_action_features, support_resistance
        highs = np.asarray([c[2] for c in candles], dtype=float)
        lows = np.asarray([c[3] for c in candles], dtype=float)
        closes = np.asarray([c[4] for c in candles], dtype=float)
        if not (np.isfinite(highs).all() and np.isfinite(lows).all() and np.isfinite(closes).all()):
            raise ValueError("K 线含非有限数值")
        return (support_resistance(highs, lows, closes, window=10, min_touches=2),
                price_action_features([list(c) for c in candles]))
    except Exception as e:  # noqa: BLE001
        log.warning("[hub] sr/pa 补算失败（返回空态）: %s", e)
        return empty_sr, {}


class MarketDataHub:
    def __init__(self, exchange_id: str, symbols: list[str], timeframes: list[str], bus: EventBus,
                 max_candles: int = 300, ma_fast_period: int = 10, ma_slow_period: int = 30) -> None:
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.timeframes = timeframes
        self.bus = bus
        self.max_candles = max_candles
        # MA 快/慢线周期（跟随当前策略；dual_ma 的 fast_period/slow_period 生效）
        self.ma_fast_period = int(ma_fast_period)
        self.ma_slow_period = int(ma_slow_period)
        self._proxy = settings.resolved_proxy or None
        self._running = False
        self._candles: dict[tuple[str, str], list] = {}
        self._last_price: dict[str, float] = {}
        # 线程锁：保护 _candles/_last_price/_sr_pa_cache 等共享结构的并发读写
        # （async 单线程下同步方法内无 await 天然原子，但外部线程/多循环场景仍需互斥）
        self._lock = threading.Lock()
        # 增量指标计算器（每个 symbol+timeframe 一组）
        self._indicators: dict[tuple[str, str], IncrIndicators] = {}
        # 已更新指标的最后一根K线 ts（同一根未收盘K线重复推送不重复计入）
        self._indicator_ts: dict[tuple[str, str], int] = {}
        # 已补算 sr/pa 的最后一根K线 ts（同一根未收盘K线重复读取不重算）
        self._sr_pa_ts: dict[tuple[str, str], int] = {}
        # sr/pa 补算结果缓存：snapshot 每次返回新建 ind dict，sr/pa 必须
        # 恒在——"同 ts 不重算"仅指跳过重算，结果仍要缓存后每次放入
        self._sr_pa_cache: dict[tuple[str, str], tuple[dict, dict]] = {}
        # 每 (symbol, timeframe) 最后一根新K线的到达时间（断链检测用，T4）
        self.last_klines_ts: dict[tuple[str, str], float] = {}
        self.max_stale_seconds = settings.max_stale_seconds  # 0=禁用

    def last_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._last_price.get(symbol)

    def last_kline_age(self, symbol: str, timeframe: str) -> float:
        """当前 symbol/tf 最近一根K线距今秒数（无数据返回无限大）。"""
        with self._lock:
            ts = self.last_klines_ts.get((symbol, timeframe))
            return float("inf") if ts is None else time.time() - ts

    def snapshot(self, symbol: str, timeframe: str) -> dict:
        """最近 N 根K线 + 最新指标快照（供 AI 解读 / 前端 / 策略）。

        线程安全：持锁读取 _candles、_sr_pa_cache 等共享结构。
        指标只在新K线到达时由 _refresh_indicators 增量更新；sr/pa 补算结果
        缓存到 _sr_pa_cache（仅随新K线 ts 重算）。
        曾在此处反复 update 同一根未收盘K线，导致 SMA/布林/量比缓冲被
        "最后一根收盘价重复"填满，实盘 MA/BB/vol_ratio 信号失真。
        """
        with self._lock:
            return self._snapshot_locked(symbol, timeframe)

    def _snapshot_locked(self, symbol: str, timeframe: str) -> dict:
        """snapshot 的锁内实现（调用方已持锁）。"""
        buf = self._candles.get((symbol, timeframe), [])
        if not buf:
            return {}
        closes = [c[4] for c in buf]
        ind = self._snapshot_indicators(symbol, timeframe, buf)
        # 补算关键位(sr) + 价格行为(pa)：窗口 buf[-120:]（含当前未收盘K线）
        key = (symbol, timeframe)
        last_ts = int(buf[-1][0])
        if self._sr_pa_ts.get(key) != last_ts:
            self._sr_pa_cache[key] = _compute_sr_pa(buf[-120:])
            self._sr_pa_ts[key] = last_ts
        sr, pa = self._sr_pa_cache.get(key, ({}, {}))
        ind["sr"] = sr
        ind["pa"] = pa
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
            incr = IncrIndicators(period_ma_fast=self.ma_fast_period, period_ma_slow=self.ma_slow_period)
            self._indicators[key] = incr
            for c in buf[:-1]:
                incr.update(float(c[4]), float(c[5]))
        snap = incr.update(float(buf[-1][4]), float(buf[-1][5]))
        self._indicator_ts[key] = last_ts
        return snap

    def set_ma_periods(self, ma_fast_period: int, ma_slow_period: int) -> None:
        """更新 MA 快/慢线周期（策略切换 / 参数热更新时调用）。

        周期变更后重建已订阅的增量计算器，使其下次快照按新周期重热身；
        否则 dual_ma 的 fast_period/slow_period 在实时路径不生效。
        """
        ma_fast_period = int(ma_fast_period)
        ma_slow_period = int(ma_slow_period)
        if ma_fast_period == self.ma_fast_period and ma_slow_period == self.ma_slow_period:
            return
        with self._lock:
            self.ma_fast_period = ma_fast_period
            self.ma_slow_period = ma_slow_period
            # 重置全部已订阅增量计算器（缓冲按 buf 自动重热身，暖机损失极小）
            self._indicators.clear()
            self._indicator_ts.clear()

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

    def apply_ohlcv(self, symbol: str, timeframe: str, candles: list) -> bool:
        """应用交易所推送的K线：原子更新缓冲 + 指标 + 快照缓存。

        线程安全：单个调用内持锁完成，防止 _upsert 与 snapshot 之间插入并发读取。
        返回是否有新K线（changed）。
        """
        with self._lock:
            changed = self._upsert_locked(symbol, timeframe, candles)
            if changed:
                # 新K线到达：增量更新指标（同 ts 重复推送自动去重）
                self._refresh_indicators(symbol, timeframe, self._candles.get((symbol, timeframe), []))
            return changed

    def _upsert_locked(self, symbol: str, timeframe: str, candles: list) -> bool:
        """追加/更新K线缓冲（调用方已持锁）。"""
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
        if changed:
            self.last_klines_ts[key] = time.time()
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
            """为单个 symbol+timeframe 订阅K线，带指数退避重连。

            bug-fix（自实验发现）：ccxt.pro 的 watch_ohlcv 是 **coroutine**（await
            后返回K线数组、阻塞至有新数据），不是 async-generator——原 `async for
            candles in ex.watch_ohlcv(...)` 每次循环都抛 "'async for' requires an
            object with __aiter__ method, got coroutine" 且泄漏未 await 的协程，
            导致实时K线**从未流入交易引擎**（5m 信号链路静默断裂）。改为标准
            while + await 用法。
            """
            backoff = 1
            while self._running:
                try:
                    candles = await ex.watch_ohlcv(symbol, tf)
                    backoff = 1  # 成功重置延迟
                    changed = self.apply_ohlcv(symbol, tf, candles)
                    if changed:
                        # 新K线到达：取快照并发布事件（快照已在 apply_ohlcv 内持锁更新）
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
