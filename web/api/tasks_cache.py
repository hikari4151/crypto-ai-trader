"""任务进度缓存的统一清理工具：防内存无限增长（长跑服务上回测/AI/训练任务会永久累积）。"""
import time
from threading import Lock
from typing import Any


def prune_task_cache(cache: dict[int, dict], lock: Lock,
                     max_items: int = 200, ttl_hours: float = 12.0) -> None:
    """任务数超上限时，惰性清理"已完成且超过 TTL"的条目。

    - 完成写入处需给条目打上 _done_ts 时间戳（本模块提供 mark_done）
    - 正在运行的任务永不清理
    """
    if len(cache) <= max_items:
        return
    now = time.time()
    cutoff = now - ttl_hours * 3600
    with lock:
        expired = [
            k for k, v in cache.items()
            if not v.get("running") and v.get("_done_ts", 0) < cutoff
        ]
        for k in expired:
            cache.pop(k, None)
        # TTL 窗口内（12h 内完成的）也可能超过 max_items：按完成时间保留最近 N 个，
        # 其余清理——曾只清过期条目，短时间大量任务时缓存可无限超限
        if len(cache) > max_items:
            done_keys = sorted(
                (k for k, v in cache.items() if not v.get("running")),
                key=lambda k: cache[k].get("_done_ts", 0),
            )
            for k in done_keys[: len(cache) - max_items]:
                cache.pop(k, None)


def mark_done(entry: dict[str, Any]) -> dict[str, Any]:
    """给任务条目打完成时间戳（prune 判定依据）。"""
    entry["_done_ts"] = time.time()
    return entry
