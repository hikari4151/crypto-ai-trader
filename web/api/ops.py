"""运维监控总览 API（/api/ops/overview）。

一次调用拿全实盘运行健康状态，供外部监控（看门狗脚本/前端运维页）轮询：

- engine：引擎状态 + **主循环活性**（heartbeat 判定，抓"REST 活着但引擎假死"）
- alerts：告警分类计数 + 最近告警（CRITICAL 数）
- market：行情新鲜度（断流秒数）
- logs：trader/trading/alerts 日志文件大小（磁盘观测）
- db：SQLite 文件大小
- frontend：前端构建版本（build_manifest.json 的 sha256 前 8 位）
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends

from core.database import Database
from web.deps import get_db, get_engine

router = APIRouter(prefix="/api/ops", tags=["ops"])

_DATA_DIR = Path("data")
_LOGS_DIR = _DATA_DIR / "logs"
_MANIFEST = Path("web") / "static" / "build_manifest.json"


def _frontend_version() -> dict:
    """读取前端构建清单版本（sha256 前 8 位 + 构建时间）。清单缺失返回空。"""
    try:
        m = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        sha = str(m.get("sha256") or "")
        return {"version": sha[:8] if sha else None,
                "built_at": m.get("built_at"),
                "manifest_exists": True}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": None, "built_at": None, "manifest_exists": False}
    except OSError:
        return {"version": None, "built_at": None, "manifest_exists": False}


def _file_kb(p: Path) -> int:
    try:
        return round(p.stat().st_size / 1024, 1)
    except FileNotFoundError:
        return 0
    except OSError:
        return -1


@router.get("/overview")
async def ops_overview(engine=Depends(get_engine), db: Database = Depends(get_db)):
    """运维总览：引擎活性 / 告警 / 行情 / 日志与库大小。"""
    st = {}
    try:
        st = engine.status()
    except Exception:  # noqa: BLE001
        st = {"running": False, "error": "engine.status 不可用"}

    alerts = st.get("alerts") or {}
    out = {
        "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": {
            "running": st.get("running", False),
            "mode": st.get("mode"),
            "strategy": st.get("strategy"),
            "symbol": st.get("symbol"),
            "symbols": st.get("symbols"),
            # 看门狗判据：进程活着 + 引擎主循环活着
            "loop_alive": st.get("engine_loop_alive", True),
            "heartbeat_age_sec": st.get("heartbeat_age_sec"),
            "degraded": st.get("degraded", False),
        },
        "alerts": {
            "critical_recent": alerts.get("critical_recent", 0),
            "counts": alerts.get("counts", {}),
            "recent": alerts.get("recent", [])[:10],
        },
        "market": {
            "stale": st.get("market_stale", False),
            "stale_sec": st.get("market_stale_sec"),
            "threshold": st.get("market_stale_threshold"),
        },
        "logs": {
            "trader_log_kb": _file_kb(_LOGS_DIR / "trader.log"),
            "trading_log_kb": _file_kb(_LOGS_DIR / "trading.log"),
            "alerts_log_kb": _file_kb(_LOGS_DIR / "alerts.log"),
            "server_log_kb": _file_kb(_LOGS_DIR / "server.log"),
        },
        "db": {
            "trader_db_kb": _file_kb(_DATA_DIR / "trader.db"),
        },
        # 前端构建版本：build_manifest.json 的 sha256 前 8 位（运维确认部署版本）
        "frontend": _frontend_version(),
    }
    # 汇总判定：CRITICAL 告警 OR 断流 OR 引擎假死 → health=false
    out["health"] = bool(
        not (out["alerts"]["critical_recent"] > 0
             or out["market"]["stale"]
             or not out["engine"]["loop_alive"])
    )
    return out
