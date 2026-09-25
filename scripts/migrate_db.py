"""版本化迁移 CLI：独立于应用运行，检查/应用 SQLite 迁移。

用法：
  python scripts/migrate_db.py --check    # 只读：当前版本 + 待应用迁移
  python scripts/migrate_db.py --list     # 列出全部已定义迁移
  python scripts/migrate_db.py --apply    # 应用未执行的迁移（写库）
  python scripts/migrate_db.py --apply --dry-run   # 只显示计划不写库

退出码：0=成功/无待应用；1=有待应用（--check）或应用失败。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.migrations import MIGRATIONS, apply, current_version, pending  # noqa: E402

DB_PATH = ROOT / "data" / "trader.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只读：当前版本 + 待应用")
    ap.add_argument("--list", action="store_true", help="列出全部已定义迁移")
    ap.add_argument("--apply", action="store_true", help="应用未执行的迁移（写库）")
    ap.add_argument("--dry-run", action="store_true", help="apply 时只显示计划不写库")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite 库路径")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print("[migrate] 库不存在: %s" % db)
        return 1

    if args.list:
        for m in MIGRATIONS:
            print("  v%d %s: %s" % (m.version, m.name, m.notes))
        return 0

    ver = current_version(db)
    todo = pending(db)
    print("[migrate] %s: 当前版本 v%d，已定义 %d 个迁移，待应用 %d 个"
          % (db.name, ver, len(MIGRATIONS), len(todo)))

    if args.apply:
        records = apply(db, dry_run=args.dry_run)
        failed = False
        for r in records:
            if r.get("dry_run"):
                tag = "DRY-RUN"
            elif r.get("applied"):
                tag = "OK"
            else:
                tag = "FAIL"
                failed = True
            line = "  [%s] v%d %s" % (tag, r["version"], r["name"])
            if r.get("error"):
                line += " error=%s" % r["error"]
            print(line)
        return 1 if failed else 0

    if todo:
        for m in todo:
            print("  [PENDING] v%d %s" % (m.version, m.name))
        return 1
    print("[migrate] 已是最新版本，无需迁移")
    return 0


if __name__ == "__main__":
    sys.exit(main())
