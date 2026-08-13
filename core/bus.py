"""轻量异步事件总线：发布/订阅，处理器为协程。

优化：
- 背压机制：每类事件维护独立队列，队列满时丢弃旧事件（防内存爆炸）
- 处理器并发执行：多个 handler 同时 await，不再串行阻塞
"""
import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from .events import Event, EventType

log = logging.getLogger(__name__)
Handler = Callable[[Event], Awaitable[None]]

# 每类事件的最大队列深度，超出时丢弃旧事件
_DEFAULT_QUEUE_SIZE = 1000


class EventBus:
    def __init__(self, max_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._subs: dict[EventType, list[Handler]] = defaultdict(list)
        self._max_queue_size = max_queue_size
        # 每类事件独立队列，实现背压
        self._queues: dict[EventType, asyncio.Queue] = {}

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
            return True
        except asyncio.QueueFull:
            log.warning("[bus] 队列满，丢弃旧事件 type=%s", event.type)
            try:
                q.get_nowait()  # 丢弃最旧的
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(event)
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
                await asyncio.sleep(0.01)

    async def _run_handler(self, handler: Handler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:  # noqa: BLE001
            log.exception("事件处理器异常: type=%s handler=%s", event.type, handler)
