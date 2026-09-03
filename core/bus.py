"""轻量异步事件总线：发布/订阅，处理器为协程。

优化：
- 背压机制：每类事件维护独立队列，队列满时丢弃旧事件（防内存爆炸）
- 处理器并发执行：多个 handler 同时 await，不再串行阻塞
"""
import asyncio
import logging
import time
from collections import defaultdict
from typing import Awaitable, Callable

from .events import Event, EventType

log = logging.getLogger(__name__)
Handler = Callable[[Event], Awaitable[None]]

# 每类事件的最大队列深度，超出时丢弃旧事件
_DEFAULT_QUEUE_SIZE = 1000

# 同一 handler 连续失败达到该次数 → 发布 SYSTEM 事件（健康降级信号）
_HANDLER_FAIL_THRESHOLD = 5
# 这些类型的事件处理失败属于"交易热路径"，即使未达阈值也要单独 log.error 告警
_CRITICAL_EVENT_TYPES = (EventType.MARKET_CANDLE, EventType.ORDER_FILL)


class EventBus:
    def __init__(self, max_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._subs: dict[EventType, list[Handler]] = defaultdict(list)
        self._max_queue_size = max_queue_size
        # 每类事件独立队列，实现背压
        self._queues: dict[EventType, asyncio.Queue] = {}
        # 各 handler 连续失败统计（成功一次清零）：{handler: {"count", "last_ts", "last_error"}}
        self._handler_failures: dict[Handler, dict] = {}
        # 唤醒事件：publish 时 set，run 循环等待，避免空转 sleep 轮询
        self._wake = asyncio.Event()

    def _get_queue(self, etype: EventType) -> asyncio.Queue:
        if etype not in self._queues:
            self._queues[etype] = asyncio.Queue(maxsize=self._max_queue_size)
        return self._queues[etype]

    def subscribe(self, etype: EventType, handler: Handler) -> None:
        self._subs[etype].append(handler)

    def unsubscribe(self, etype: EventType, handler: Handler) -> None:
        if handler in self._subs[etype]:
            self._subs[etype].remove(handler)

    async def publish(self, event: Event) -> bool:
        """发布事件到队列。队列满时丢弃最旧事件并记录警告。

        Returns:
            True 表示事件已入队，False 表示事件被丢弃（队列满）。
        """
        q = self._get_queue(event.type)
        try:
            q.put_nowait(event)
            self._wake.set()
            return True
        except asyncio.QueueFull:
            log.warning("[bus] 队列满，丢弃旧事件 type=%s", event.type)
            try:
                q.get_nowait()  # 丢弃最旧的
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(event)
            self._wake.set()
            return False

    async def run(self) -> None:
        """启动事件分发循环：并发处理所有类型的事件队列。"""
        running = True
        while running:
            tasks = []
            for etype, q in list(self._queues.items()):
                if not q.empty():
                    try:
                        event = q.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    # 为该事件类型的所有 handler 创建并发任务
                    for handler in list(self._subs.get(etype, [])):
                        tasks.append(self._run_handler(handler, event))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # 没有事件时挂起等待 publish 唤醒，避免空转轮询
                self._wake.clear()
                await self._wake.wait()

    async def _run_handler(self, handler: Handler, event: Event) -> None:
        try:
            await handler(event)
            # 成功一次清零连续失败计数
            self._handler_failures.pop(handler, None)
        except Exception as e:  # noqa: BLE001
            log.exception("事件处理器异常: type=%s handler=%s", event.type, handler)
            rec = self._handler_failures.setdefault(
                handler, {"count": 0, "last_ts": 0.0, "last_error": ""})
            rec["count"] += 1
            rec["last_ts"] = time.time()
            rec["last_error"] = str(e) or e.__class__.__name__
            hname = getattr(handler, "__qualname__", str(handler))
            if event.type in _CRITICAL_EVENT_TYPES:
                # 交易热路径（K线处理/成交回灌）失败：即使未达阈值也单独告警
                log.error("[bus] 关键事件处理器失败: type=%s handler=%s error=%s",
                          event.type.value, hname, rec["last_error"])
            if rec["count"] >= _HANDLER_FAIL_THRESHOLD:
                # 连续失败达阈值：发布 SYSTEM 事件（无人订阅时 publish 正常返回）
                await self.publish(Event(EventType.SYSTEM, {
                    "kind": "bus_handler_failed",
                    "etype": event.type.value,
                    "handler": hname,
                    "failures": rec["count"],
                    "last_error": rec["last_error"],
                }, source="bus"))

    def health(self) -> dict:
        """bus 健康状态（只读）：degraded = 任一 handler 连续失败达阈值。

        仅供同步读：状态只在主循环内写，单循环无锁安全。
        """
        handlers = {
            getattr(h, "__qualname__", str(h)): {"failures": rec["count"], "last_error": rec["last_error"]}
            for h, rec in self._handler_failures.items()
        }
        return {
            "degraded": any(rec["count"] >= _HANDLER_FAIL_THRESHOLD
                            for rec in self._handler_failures.values()),
            "handlers": handlers,
        }
