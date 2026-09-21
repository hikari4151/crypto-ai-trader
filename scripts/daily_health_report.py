"""P2-11 每日健康报告：把散落的运行状态汇成一份结构化日报，供运维巡检。

复用 web/api/ops.py 的聚合口径，但独立可跑（不依赖 Web 服务）：
直接读日志/DB/磁盘，输出到 data/logs/health_report.log（JSON 行）。

用法：
  python scripts/daily_health_report.py            # 生成一次日报（stdout + 落盘）
  python scripts/daily_health_report.py --check    # 只输出退出码：0=健康 1=需关注
  定时：Windows 计划任务每日 08:00 跑一次（脚本幂等）
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / "data" / "logs"
DATA_DIR = REPO / "data"
REPORT_PATH = LOGS_DIR / "health_report.log"


def _dir_kb(p: Path) -> int:
    try:
        return round(sum(f.stat().st_size for f in p.glob("*") if f.is_file()) / 1024, 1)
    except (FileNotFoundError, OSError):
        return 0


def _file_kb(p: Path) -> int:
    try:
        return round(p.stat().st_size / 1024, 1)
    except (FileNotFoundError, OSError):
        return 0


def _load_alerts() -> dict:
    """读取最近告警（alerts.log 尾部，最多 50 条）并统计。"""
    path = LOGS_DIR / "alerts.log"
    if not path.exists():
        return {"counts": {}, "critical_recent": 0, "recent": []}
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    recs, counts = [], {}
    for ln in lines[-50:]:
        try:
            rec = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        counts[rec.get("kind", "?")] = counts.get(rec.get("kind", "?"), 0) + 1
        recs.append(rec)
    critical = sum(1 for r in recs if r.get("level") == "CRITICAL")
    return {"counts": counts, "critical_recent": critical,
            "recent": [{"ts": r.get("ts"), "level": r.get("level"), "kind": r.get("kind"),
                        "message": r.get("message")} for r in recs[-10:]]}


def build_report() -> dict:
    alerts = _load_alerts()
    server_log = LOGS_DIR / "server.log"
    server_age = -1
    if server_log.exists():
        server_age = int(time.time() - server_log.stat().st_mtime)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "engine": {"server_log_age_sec": server_age},
        "alerts": alerts,
        "disk": {
            "logs_dir_kb": _dir_kb(LOGS_DIR),
            "data_dir_kb": _dir_kb(DATA_DIR),
            "trader_db_kb": _file_kb(DATA_DIR / "trader.db"),
            "klines_db_kb": _file_kb(DATA_DIR / "klines.db"),
        },
    }
    reasons = []
    if server_age > 600:
        reasons.append(f"server.log 超过 10 分钟未更新（{server_age}s）")
    if alerts["critical_recent"] > 0:
        reasons.append(f"近 50 条告警含 {alerts['critical_recent']} 条 CRITICAL")
    if report["disk"]["data_dir_kb"] > 500 * 1024:
        reasons.append(f"data 目录 {report['disk']['data_dir_kb'] / 1024:.0f}MB 超 500MB")
    report["verdict"] = {
        "health": not reasons,
        "needs_attention": bool(reasons),
        "reasons": reasons,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只输出退出码：0=健康 1=需关注")
    ap.add_argument("--quiet", action="store_true", help="不打印详情")
    args = ap.parse_args()

    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")

    if args.check:
        code = 0 if report["verdict"]["health"] else 1
        print(f"[health] {'健康' if code == 0 else '需关注'}: {report['verdict']['reasons']}")
        raise SystemExit(code)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[health] 已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()
