"""运维一体化 maintenance：一次调用完成日常运维（备份 + 保留策略 + 健康报告）。

整合已完成的基础建设组件，避免手动逐个执行：
  1. db_backup：在线备份两库（含 WAL 增量）+ integrity 验证 + 清理旧备份
  2. prune_equity_snapshots：资金快照保留策略（删超期行，防表无限增长）
  3. daily_health_report：聚合日志/告警/磁盘/DB 输出健康报告

用法：
  python scripts/maintenance.py                    # 全套（备份+prune+健康报告）
  python scripts/maintenance.py --skip-backup      # 只做保留策略+健康报告
  python scripts/maintenance.py --skip-health      # 只做备份+prune
  python scripts/maintenance.py --keep-days 7 --prune-days 90

退出码：0=健康；1=备份失败或健康需关注（供计划任务/supervisor 联动）
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import db_backup  # noqa: E402
import daily_health_report as health  # noqa: E402

REPORT_PATH = ROOT / "data" / "logs" / "maintenance_report.log"


def run_backup(keep_days: int, prune_days: int) -> tuple[bool, dict]:
    """备份两库 + 清理旧备份 + 资金快照保留。返回 (ok, 明细)。"""
    ok = True
    detail: dict = {"backups": {}, "prune": {}}
    for name, src in db_backup.DBS:
        if not src.exists():
            detail["backups"][name] = {"skipped": True}
            continue
        path, size = db_backup.backup_one(name, src)
        if not path:
            ok = False
            detail["backups"][name] = {"error": "backup failed"}
            continue
        vok, info = db_backup.verify_backup(Path(path), name)
        detail["backups"][name] = {"path": Path(path).name, "size_kb": round(size / 1024, 1),
                                   "integrity": info.get("integrity")}
        if not vok:
            Path(path).unlink(missing_ok=True)
            ok = False
            detail["backups"][name]["error"] = "integrity failed"
    removed = db_backup.clean_old(keep_days)
    detail["removed_old_backups"] = removed
    if prune_days > 0:
        detail["prune"] = db_backup.prune_equity_snapshots(
            db_backup.DATA_DIR / "trader.db", keep_days=prune_days)
    return ok, detail


def run_health() -> dict:
    """生成健康报告（复用 daily_health_report 的判定逻辑）。"""
    report = health.build_report()
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-days", type=int, default=7, help="旧备份保留天数")
    ap.add_argument("--prune-days", type=int, default=90, help="资金快照保留天数（0=跳过）")
    ap.add_argument("--skip-backup", action="store_true")
    ap.add_argument("--skip-health", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    result: dict = {"ts": __import__("datetime").datetime.now().isoformat(),
                    "steps": {}, "ok": True, "reasons": []}

    if not args.skip_backup:
        ok, detail = run_backup(args.keep_days, args.prune_days)
        result["steps"]["backup"] = detail
        result["ok"] = result["ok"] and ok

    if not args.skip_health:
        report = run_health()
        result["steps"]["health"] = report
        verdict = report.get("verdict", {})
        if verdict.get("needs_attention"):
            result["ok"] = False
            result["reasons"].extend(verdict.get("reasons", []))

    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["exit_code"] = 0 if result["ok"] else 1
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[maintenance] 完成：{'全部正常' if result['ok'] else '需关注'} "
          f"({result['elapsed_sec']}s) -> {REPORT_PATH}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())