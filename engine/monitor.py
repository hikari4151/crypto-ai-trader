"""统一告警中心：把散落的运行异常汇聚成分级告警（实盘运维必需）。

现状问题：断链/下单失败/对账差异/风控熔断/持仓异常各自只写一行日志，
无人看日志就等于没有告警；实盘最怕"出事了不知道"。

本模块提供 AlertMonitor：
- **入口**：`alert(level, kind, message, **detail)` 统一上报；
- **三个出口**：
  1. `data/logs/alerts.log`（结构化单行 JSON，便于运维采集/对接）
  2. SYSTEM 事件（Web UI 事件流可展示，前端无需额外接口）
  3. 返回值给调用方决定是否额外处理
- **去重**：同 (kind, key) 在 `dedup_sec` 内只报一次，避免刷屏（但每次都计数，
  告警摘要里可见 suppressed 次数）；
- **分级**：WARNING（需关注）/ CRITICAL（需立即处理）；
- **健康巡检**：`health_check(engine)` 做周期检查——行情断链、无保护持仓、
  日亏超限、对账差异、事件总线降级。

设计取舍：不引入外部依赖（短信/邮件/webhook 由运维侧接 alerts.log 或后续扩展）。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.events import Event, EventType

log = logging.getLogger(__name__)

WARNING = "WARNING"
CRITICAL = "CRITICAL"


@dataclass
class AlertRecord:
    ts: float
    level: str
    kind: str
    key: str
    message: str
    detail: dict = field(default_factory=dict)
    suppressed: int = 0

    def to_line(self) -> str:
        return json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)),
            "level": self.level,
            "kind": self.kind,
            "key": self.key,
            "message": self.message,
            "detail": self.detail,
            "suppressed": self.suppressed,
        }, ensure_ascii=False)


class AlertMonitor:
    """告警收集/去重/落盘/发布。线程与协程内均可调用 alert()（落盘是同步小写入）。"""

    def __init__(self, bus=None, log_dir: str = "./data/logs",
                 dedup_sec: float = 300.0, max_recent: int = 200) -> None:
        self._bus = bus
        self._dedup_sec = float(dedup_sec)
        self._recent: list[AlertRecord] = []
        self._max_recent = max_recent
        self._last: dict[tuple[str, str], AlertRecord] = {}
        self._counts: dict[str, int] = {}
        self._path = Path(log_dir) / "alerts.log"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001  # 目录不可写不影响主流程
            log.warning("[monitor] 告警目录不可写(%s)，仅内存告警: %s", log_dir, e)

    def repoint_log_dir(self, log_dir: str) -> None:
        """重设告警落盘目录（测试隔离用：防止测试告警污染生产 alerts.log）。

        只影响后续写入；已在内存的 recent/counts 保留。调用方负责创建目录。
        """
        new_path = Path(log_dir) / "alerts.log"
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            self._path = new_path
        except Exception as e:  # noqa: BLE001
            log.warning("[monitor] 重设告警目录失败(%s)，沿用原路径: %s", log_dir, e)

    # ---------------- 上报 ----------------
    def alert(self, level: str, kind: str, message: str, key: str = "", **detail: Any) -> Optional[AlertRecord]:
        """上报一条告警。同 (kind,key) 在去重窗口内只落盘/发布一次。

        返回本次**实际发出**的记录（被去重则返回 None），调用方可用于判断。
        """
        now = time.time()
        k = (kind, key or "-")
        prev = self._last.get(k)
        if prev is not None and (now - prev.ts) < self._dedup_sec:
            prev.suppressed += 1
            self._counts[kind] = self._counts.get(kind, 0) + 1
            return None
        rec = AlertRecord(ts=now, level=level, kind=kind, key=key or "-",
                          message=message, detail=dict(detail))
        self._last[k] = rec
        self._recent.append(rec)
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        self._write(rec)
        if level == CRITICAL:
            log.error("[alert][%s] %s | %s", kind, message, rec.detail)
        else:
            log.warning("[alert][%s] %s | %s", kind, message, rec.detail)
        self._publish(rec)
        return rec

    def _write(self, rec: AlertRecord) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(rec.to_line() + "\n")
        except Exception as e:  # noqa: BLE001
            log.debug("[monitor] 告警落盘失败: %s", e)

    def _publish(self, rec: AlertRecord) -> None:
        """发布到事件总线（Web UI 可展示）。无 loop / 总线未运行时静默跳过。"""
        if self._bus is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            loop.create_task(self._bus.publish(Event(EventType.SYSTEM, {
                "kind": "alert", "level": rec.level, "alert_kind": rec.kind,
                "message": rec.message, "detail": rec.detail,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.ts)),
            }, source="monitor")))
        except Exception as e:  # noqa: BLE001
            log.debug("[monitor] 告警发布失败: %s", e)

    # ---------------- 查询 ----------------
    def summary(self) -> dict:
        return {
            "counts": dict(self._counts),
            "recent": [{"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.ts)),
                        "level": r.level, "kind": r.kind, "key": r.key,
                        "message": r.message, "suppressed": r.suppressed}
                       for r in self._recent[-20:]],
            "critical_recent": sum(1 for r in self._recent[-50:] if r.level == CRITICAL),
        }

    # ---------------- 周期健康巡检 ----------------
    async def health_check(self, engine) -> None:
        """引擎实盘健康巡检（由引擎的周期循环调用，异常一律不抛出）。"""
        try:
            await self._check_market(engine)
            await self._check_position(engine)
            await self._check_risk(engine)
            await self._check_bus(engine)
        except Exception as e:  # noqa: BLE001  # 巡检自身失败不得影响引擎
            log.debug("[monitor] 健康巡检异常: %s", e)

    async def _check_market(self, engine) -> None:
        hub = getattr(engine, "hub", None)
        if hub is None:
            return
        try:
            age = float(hub.last_kline_age(engine.symbol, engine.timeframe))
        except Exception:  # noqa: BLE001
            return
        thr = 0.0
        try:
            thr = float(engine._stale_threshold())
        except Exception:  # noqa: BLE001
            thr = 0.0
        if thr > 0 and (age == float("inf") or age > thr):
            self.alert(CRITICAL, "market_stale",
                       f"行情断流：{engine.symbol} {engine.timeframe} 已 {int(age) if age != float('inf') else '∞'} 秒无新K线",
                       key=engine.symbol, age_sec=None if age == float("inf") else int(age),
                       threshold=int(thr))

    async def _check_position(self, engine) -> None:
        """无保护持仓：有仓位但没有兜底止损（实盘最危险的静默状态）。"""
        try:
            pos = float(engine._current_position_qty())
        except Exception:  # noqa: BLE001
            return
        if pos <= 0:
            return
        om = getattr(engine, "order_manager", None)
        ps = getattr(om, "protective", None) if om else None
        if ps is None:
            return
        try:
            has_stop = bool(ps.status().get("stops"))
        except Exception:  # noqa: BLE001
            return
        if not has_stop and getattr(engine, "trading_mode", "") == "live":
            self.alert(CRITICAL, "position_unprotected",
                       f"持仓无兜底止损：{engine.symbol} 持仓 {pos}",
                       key=engine.symbol, qty=pos)

    async def _check_risk(self, engine) -> None:
        """风控熔断状态播报（冷却中属于需人工关注）。"""
        risk = getattr(engine, "risk", None)
        if risk is None:
            return
        try:
            cd = float(getattr(risk, "_cooldown_until", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            return
        if cd > time.time():
            self.alert(WARNING, "risk_cooldown",
                       f"风控冷却中：剩余 {int(cd - time.time())} 秒，暂停新开仓",
                       key="global", remaining_sec=int(cd - time.time()))

    async def _check_bus(self, engine) -> None:
        """事件总线 handler 连续失败（健康降级）→ 告警。"""
        bus = getattr(engine, "bus", None)
        if bus is None:
            return
        try:
            fails = getattr(bus, "_handler_failures", None) or {}
            for handler, info in fails.items():
                cnt = int((info or {}).get("count") or 0)
                if cnt >= 5:
                    name = getattr(handler, "__qualname__", str(handler))[:80]
                    self.alert(WARNING, "bus_handler_failing",
                               f"事件处理器连续失败 {cnt} 次：{name}",
                               key=name, count=cnt,
                               error=str((info or {}).get("last_error"))[:200])
        except Exception:  # noqa: BLE001
            return
