"""程序日志查看 API（/api/logs/*）。

背景：程序告警此前只落在 ``data/logs/*.log`` 里，Web UI 只能看到「优化日志」
（数据库里的参数优化记录）与「AI 行情解读」，用户**看不到程序自身的 WARNING/ERROR**，
排障只能去翻文件。本模块把日志文件与内存告警汇总成可浏览的接口。

三个出口：
- ``GET /api/logs/sources`` 可选日志源清单（大小 / 修改时间，用于提示"有没有在写"）
- ``GET /api/logs``         按级别过滤读取某个源的尾部（新→旧，支持翻页）
- ``GET /api/logs/alerts``  引擎 AlertMonitor 的**内存**分级告警（含去重次数）

设计要点：
- **只读文件尾部**（默认最多回看 512KB），不把 3MB+ 的日志整读进内存；
  长跑服务下 trader.log 可达 5MB×5 份，全读会拖慢事件循环。
- 路由声明为**同步 def**（非 async def）⇒ FastAPI 自动丢线程池执行，
  文件 IO 不阻塞事件循环（这是本模块最容易被忽略的性能点）。
- 只允许读取白名单源，路径不接受外部输入 ⇒ 无路径遍历面。
"""
import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from config.settings import settings

router = APIRouter(prefix="/api/logs", tags=["logs"])

# 可选日志源：key -> (文件名, 展示名, 说明)
_SOURCES: dict[str, tuple[str, str, str]] = {
    "trader":  ("trader.log", "主程序日志", "应用全量日志（含训练/回测/因子，INFO 起）"),
    "trading": ("trading.log", "交易事件", "下单/成交/撤单/对账/风控熔断（运维只盯这个）"),
    "server":  ("server.log", "HTTP 服务", "uvicorn 访问与错误日志"),
    "alerts":  ("alerts.log", "分级告警", "AlertMonitor 结构化告警（JSON 行，CRITICAL/WARNING）"),
}

# 级别从低到高：level 参数按「不低于」过滤（默认 WARNING ⇒ 只看警告与错误）
_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_LEVEL_RANK = {lv: i for i, lv in enumerate(_LEVELS)}
# 日志行格式：%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(?P<level>[A-Z]+)\s*\|\s*"
    r"(?P<logger>\S+)\s*\|\s*(?P<msg>.*)$"
)
# 尾部读取上限：够装下约 5000 行，单次 ~1ms
_TAIL_BYTES = 512 * 1024


def _log_path(source: str) -> Path:
    if source not in _SOURCES:
        raise HTTPException(status_code=400,
                            detail=f"未知日志源 {source!r}，可选：{', '.join(_SOURCES)}")
    return Path(settings.log_dir) / _SOURCES[source][0]


def _read_tail(path: Path, max_bytes: int = _TAIL_BYTES) -> tuple[list[str], bool]:
    """读文件尾部，返回 (文本行列表, 是否截断过)。

    截断时必须丢掉**第一行**——它多半是被切断的半行，直接解析会得到假条目。
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return [], False
    except OSError:
        return [], False
    if size == 0:
        return [], False
    truncated = size > max_bytes
    start = size - max_bytes if truncated else 0
    with path.open("rb") as fh:
        fh.seek(start)
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if truncated and lines:
        lines = lines[1:]
    return lines, truncated


def _parse_text_log(lines: list[str]) -> list[dict]:
    """把 `ts | LEVEL | logger | msg` 格式的行解析成条目。

    多行条目（异常堆栈等）不以时间戳开头 ⇒ 追加到上一条的 detail，
    否则一次 Traceback 会被拆成十几条无级别的"垃圾"行。
    """
    out: list[dict] = []
    for ln in lines:
        m = _LINE_RE.match(ln)
        if m:
            out.append({
                "ts": m.group("ts"),
                "level": m.group("level"),
                "logger": m.group("logger"),
                "msg": m.group("msg"),
                "detail": "",
            })
        elif out and ln.strip():
            prev = out[-1]
            prev["detail"] = (prev["detail"] + "\n" + ln) if prev["detail"] else ln
    return out


def _parse_alert_log(lines: list[str]) -> list[dict]:
    """alerts.log 是单行 JSON；解析失败的行降级为普通文本条目而不是丢弃。"""
    out: list[dict] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            d = json.loads(s)
        except (ValueError, TypeError):
            out.append({"ts": "", "level": "WARNING", "logger": "alerts",
                        "msg": s, "detail": "", "kind": "", "key": "", "suppressed": 0})
            continue
        detail = d.get("detail")
        out.append({
            "ts": str(d.get("ts") or ""),
            "level": str(d.get("level") or "WARNING").upper(),
            "logger": f"alert.{d.get('kind') or '-'}",
            "msg": str(d.get("message") or ""),
            "detail": json.dumps(detail, ensure_ascii=False) if detail else "",
            "kind": str(d.get("kind") or ""),
            "key": str(d.get("key") or ""),
            "suppressed": int(d.get("suppressed") or 0),
        })
    return out


def _source_stat(source: str) -> dict:
    name, label, desc = _SOURCES[source]
    p = Path(settings.log_dir) / name
    try:
        st = p.stat()
        size_kb = round(st.st_size / 1024, 1)
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        exists = True
    except OSError:
        size_kb, mtime, exists = 0, "", False
    # 同源轮转文件（trader.log.1、alerts.log.bak…）只计数，供"是否被轮转过"参考
    rotated = 0
    try:
        rotated = len(list(Path(settings.log_dir).glob(name + ".*")))
    except OSError:
        pass
    return {"key": source, "label": label, "desc": desc, "file": name,
            "size_kb": size_kb, "mtime": mtime, "exists": exists, "rotated": rotated}


@router.get("/sources")
def log_sources():
    """可选日志源清单（大小 / 修改时间 / 轮转份数）。"""
    return {"sources": [_source_stat(k) for k in _SOURCES]}


@router.get("")
def read_logs(source: str = Query("trader"),
              level: str = Query("WARNING"),
              limit: int = Query(300, ge=1, le=2000),
              offset: int = Query(0, ge=0)):
    """读取某个日志源的尾部条目（新→旧）。

    level 为**下限**：`WARNING` 返回 WARNING/ERROR/CRITICAL。
    传 `INFO` 或 `DEBUG` 可看全量；alerts.log 天然只有 CRITICAL/WARNING。
    """
    level = (level or "WARNING").upper()
    if level not in _LEVEL_RANK:
        raise HTTPException(status_code=400,
                            detail=f"未知级别 {level!r}，可选：{', '.join(_LEVELS)}")
    path = _log_path(source)
    lines, truncated = _read_tail(path)
    entries = _parse_alert_log(lines) if source == "alerts" else _parse_text_log(lines)

    floor = _LEVEL_RANK[level]
    # 级别未知的行（如解析降级条目）一律放行，避免"看不到内容"比"看不清级别"更糟
    kept = [e for e in entries if _LEVEL_RANK.get(e["level"], 99) >= floor]
    kept.reverse()  # 新 → 旧
    page = kept[offset:offset + limit]
    counts: dict[str, int] = {}
    for e in kept:
        counts[e["level"]] = counts.get(e["level"], 0) + 1
    return {
        "source": source,
        "file": _SOURCES[source][0],
        "level": level,
        "offset": offset,
        "limit": limit,
        "matched": len(kept),
        "returned": len(page),
        "truncated": truncated,
        "counts": counts,
        "entries": page,
    }


@router.get("/alerts")
def memory_alerts(request: Request):
    """引擎 AlertMonitor 的**内存**分级告警汇总（含去重抑制次数）。

    与 alerts.log 的区别：这里是最新一轮的内存视图（按 (kind,key) 去重，
    同一条告警在窗口内重复只计 suppressed），文件是历史流水。
    """
    engine = getattr(request.app.state, "engine", None)
    monitor = getattr(engine, "monitor", None)
    if monitor is None:
        return {"available": False, "counts": {}, "recent": [],
                "critical_recent": 0, "warning_recent": 0,
                "reason": "引擎未初始化或未挂载告警中心"}
    try:
        s = monitor.summary()
    except Exception as e:  # noqa: BLE001  读告警失败不应让页面整体 500
        return {"available": False, "counts": {}, "recent": [],
                "critical_recent": 0, "warning_recent": 0, "reason": f"读取失败: {e}"}
    recent = s.get("recent") or []
    return {
        "available": True,
        "counts": s.get("counts") or {},
        "recent": recent,
        "critical_recent": s.get("critical_recent") or 0,
        "warning_recent": sum(1 for r in recent if r.get("level") == "WARNING"),
        "total_kinds": len(s.get("counts") or {}),
    }
