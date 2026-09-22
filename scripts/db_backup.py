"""P0-1 数据库备份 + SQLite 运维（备用一致性、恢复演练、保留策略）。

核心原则：
- **必须用 SQLite 在线备份 API（source.backup()）而非文件复制**——WAL 模式
  下文件复制会丢未 checkpoint 的增量，恢复出来是残缺库。backup() 走引擎级
  快照，保证一致性。
- 备份后立即 `PRAGMA integrity_check` 验证可读，坏的备份不留。
- 保留策略：默认保留 7 天，超期自动清理（防止 data/backups 无限增长）。
- 恢复演练：`--verify` 把最新备份恢复到临时库，跑 integrity_check +
  关键表行数对比，证明"备份真的能恢复"，而不是只生成文件。

用法：
  python scripts/db_backup.py                          # 备份两个库 + 验证 + 清理旧
  python scripts/db_backup.py --verify                 # 只做恢复演练（读到哪备份到哪）
  python scripts/db_backup.py --vacuum                 # 先 VACUUM（整理碎片）再备份
  python scripts/db_backup.py --keep-days 14           # 自定义保留天数
"""
import argparse
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "backups"

DBS = [
    ("trader", DATA_DIR / "trader.db"),   # 交易/密钥/配置（主库）
    ("klines", DATA_DIR / "klines.db"),   # 行情缓存（大库，恢复优先级低）
]

# 恢复演练用关键表（存在则校验行数）
KEY_TABLES = {
    "trader": {"users", "kv", "trades", "analysis", "equity_snapshots"},
    "klines": {"klines"},
}


def ts_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_one(name: str, src: Path) -> tuple[str, int]:
    """在线备份单个库到 data/backups/<name>_<ts>.db，返回 (备份路径, 字节数)。"""
    if not src.exists():
        return "", 0
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / f"{name}_{ts_str()}.db"
    src_con = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    dst_con = sqlite3.connect(str(dst))
    try:
        src_con.backup(dst_con)  # 在线备份：源可继续被写入，结果一致
        dst_con.commit()
    finally:
        dst_con.close()
        src_con.close()
    return str(dst), dst.stat().st_size if dst.exists() else 0


def verify_backup(path: Path, name: str) -> tuple[bool, dict]:
    """校验备份可读：integrity_check + 关键表行数。返回 (ok, info)。"""
    info: dict = {"integrity": None, "tables": {}}
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
            info["integrity"] = row[0] if row else "unknown"
            for t in KEY_TABLES.get(name, set()):
                try:
                    n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    info["tables"][t] = int(n)
                except sqlite3.Error:
                    pass  # 表不存在（旧版 schema），不视为失败
        finally:
            con.close()
        return info["integrity"] == "ok", info
    except sqlite3.Error as e:
        info["error"] = str(e)
        return False, info


def clean_old(keep_days: int) -> list[str]:
    """删除超过保留期的备份，返回清理的文件名。"""
    removed = []
    cutoff = time.time() - keep_days * 86400
    if not BACKUP_DIR.exists():
        return removed
    for f in BACKUP_DIR.glob("*.db"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            removed.append(f.name)
    return removed


def prune_equity_snapshots(db_path: Path, keep_days: int = 90) -> dict:
    """删除 trader.db 中超过 keep_days 的 EquitySnapshot 行（保留策略，防无限增长）。

    引擎每 15s 落一条资金快照 → 日增约 5760 行，长期运行表会无限膨胀。
    默认保留 90 天（约 52 万行，足够资金曲线展示与复盘），更早的删除。
    返回 {"deleted": n, "remaining": m}；表不存在/库不存在时返回空结果。
    """
    if not db_path.exists():
        return {}
    import time as _t
    try:
        con = sqlite3.connect(str(db_path))
        try:
            # 校验表存在，避免旧版 schema 报错
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='equity_snapshots'"
            ).fetchone()
            if not row:
                return {"table_exists": False}
            cutoff = _t.time() - keep_days * 86400
            cur = con.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
            con.execute("DELETE FROM equity_snapshots WHERE ts < ?", (cutoff,))
            con.commit()
            after = con.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
            return {"table_exists": True, "deleted": int(cur - after),
                    "remaining": int(after), "keep_days": keep_days}
        finally:
            con.close()
    except sqlite3.Error as e:
        return {"error": str(e)}


def vacuum_main() -> dict:
    """对主库/行情库 VACUUM（整理碎片，缩库）。仅在库空闲时建议定期执行。"""
    out = {}
    for name, src in DBS:
        if not src.exists():
            continue
        try:
            con = sqlite3.connect(str(src))
            try:
                con.execute("VACUUM")
                out[name] = {"ok": True, "size_kb": round(src.stat().st_size / 1024, 1)}
            finally:
                con.close()
        except sqlite3.Error as e:
            out[name] = {"ok": False, "error": str(e)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-days", type=int, default=7)
    ap.add_argument("--verify", action="store_true", help="恢复演练：验证最新备份可恢复")
    ap.add_argument("--vacuum", action="store_true", help="备份前先 VACUUM")
    ap.add_argument("--prune-equity-days", type=int, default=0,
                    help="删除 trader.db 中超过 N 天的 EquitySnapshot（0=跳过；建议 90）")
    args = ap.parse_args()

    if args.verify:
        return run_verify()

    if args.vacuum:
        v = vacuum_main()
        print(f"[backup] VACUUM: {v}")

    if args.prune_equity_days > 0:
        p = prune_equity_snapshots(DATA_DIR / "trader.db", keep_days=args.prune_equity_days)
        print(f"[backup] 资金快照保留策略: {p}")

    print(f"[backup] 开始备份 -> {BACKUP_DIR}")
    all_ok = True
    for name, src in DBS:
        if not src.exists():
            print(f"[backup] {name}: 跳过（{src.name} 不存在）")
            continue
        path, size = backup_one(name, src)
        if not path:
            print(f"[backup] {name}: 备份失败")
            all_ok = False
            continue
        ok, info = verify_backup(Path(path), name)
        print(f"[backup] {name}: {size / 1024:.0f}KB -> {Path(path).name} "
              f"integrity={info.get('integrity')} ok={ok}")
        if not ok:
            Path(path).unlink(missing_ok=True)  # 坏备份不留
            print(f"[backup] {name}: 校验失败，已删除坏备份")
            all_ok = False
    removed = clean_old(args.keep_days)
    if removed:
        print(f"[backup] 清理旧备份（> {args.keep_days} 天）: {', '.join(removed)}")
    print(f"[backup] 完成：{'全部成功' if all_ok else '存在失败'}")
    return 0 if all_ok else 1


def run_verify() -> int:
    """恢复演练：对每个库找最新备份，恢复到临时库并验证。"""
    print("[backup] 恢复演练开始")
    all_ok = True
    for name, _ in DBS:
        cands = sorted(BACKUP_DIR.glob(f"{name}_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print(f"[backup] {name}: 无备份可验证")
            all_ok = False
            continue
        latest = cands[0]
        tmp = BACKUP_DIR / f"_verify_{name}.db"
        try:
            shutil.copy2(latest, tmp)
            ok, info = verify_backup(tmp, name)
            src_row = {}
            src = dict(DBS)[name]
            if src.exists():
                try:
                    con = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
                    for t in KEY_TABLES.get(name, set()):
                        try:
                            src_row[t] = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                        except sqlite3.Error:
                            pass
                    con.close()
                except sqlite3.Error:
                    pass
            print(f"[backup] {name}: 最新备份 {latest.name} integrity={info.get('integrity')} "
                  f"表行数={info.get('tables')} 源={src_row}")
            if not ok:
                all_ok = False
        finally:
            tmp.unlink(missing_ok=True)
    print(f"[backup] 恢复演练：{'通过' if all_ok else '有失败'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())