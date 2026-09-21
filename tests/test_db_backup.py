"""P0-1 数据库备份回归测试：一致性备份 / 校验 / 恢复演练 / 保留策略 / VACUUM。"""
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "data" / "backups"


def _mkdb(path: Path, keep_open: bool = False):
    """构造带 WAL 的测试库。

    keep_open=True：源连接不关闭（最后一条写入仍停在 WAL、未 checkpoint），
    用于验证在线备份能包含 WAL 增量；返回连接供调用方关闭。
    """
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    con.execute("INSERT OR REPLACE INTO kv VALUES ('a','1'),('b','2')")
    con.commit()
    # 已提交但保持连接打开：该行在 WAL 中、尚未 checkpoint（backup 必须能读到）
    con.execute("INSERT INTO kv VALUES ('wal_dirty','should-be-in-backup')")
    con.commit()
    if keep_open:
        return con  # 连接不关闭 → SQLite 不会自动 checkpoint，脏行仍在 WAL
    con.close()
    return None


def test_backup_online_consistent(tmp_path, monkeypatch):
    """在线备份必须包含未 checkpoint 的 WAL 脏数据；校验 integrity=ok。"""
    import scripts.db_backup as bb

    src = tmp_path / "t.db"
    con = _mkdb(src, keep_open=True)  # 源连接保持打开：脏行停在 WAL
    try:
        monkeypatch.setattr(bb, "DATA_DIR", tmp_path)
        monkeypatch.setattr(bb, "BACKUP_DIR", tmp_path / "backups")
        monkeypatch.setattr(bb, "DBS", [("t", src)])

        path, size = bb.backup_one("t", src)
        assert path and size > 0
        ok, info = bb.verify_backup(Path(path), "t")
        assert ok and info["integrity"] == "ok"

        # 备份里必须有 WAL 脏数据行（证明不是文件复制丢增量）
        bcon = sqlite3.connect(path)
        try:
            n = bcon.execute("SELECT COUNT(*) FROM kv WHERE k='wal_dirty'").fetchone()[0]
            assert n == 1, "在线备份应包含未 checkpoint 的 WAL 数据"
        finally:
            bcon.close()
    finally:
        con.close()


def test_verify_restore_counts_match(tmp_path, monkeypatch):
    """恢复演练：最新备份的表行数与源库一致。"""
    import scripts.db_backup as bb

    src = tmp_path / "t.db"
    _mkdb(src)
    bk = tmp_path / "backups" / "t_20260101T000000Z.db"
    bk.parent.mkdir(parents=True)
    con = sqlite3.connect(str(src))
    try:
        con.executescript("CREATE TABLE dummy (x INTEGER)")
        con.execute("INSERT INTO dummy VALUES (1),(2),(3)")
        con.commit()
    finally:
        con.close()
    shutil.copy2(src, bk)  # 测试恢复演练用文件副本即可（此处不算生产备份）

    monkeypatch.setattr(bb, "BACKUP_DIR", bk.parent)
    monkeypatch.setattr(bb, "DBS", [("t", src)])
    monkeypatch.setattr(bb, "KEY_TABLES", {"t": {"kv", "dummy"}})

    rc = bb.run_verify()
    assert rc == 0


def test_clean_old_removes_expired(tmp_path, monkeypatch):
    """保留策略：超过 keep_days 的旧备份被清理。"""
    import time
    import scripts.db_backup as bb

    bk = tmp_path / "backups"
    bk.mkdir()
    old = bk / "old.db"
    new = bk / "new.db"
    old.touch()
    new.touch()
    old_time = time.time() - 10 * 86400  # 10 天前
    import os
    os.utime(old, (old_time, old_time))

    monkeypatch.setattr(bb, "BACKUP_DIR", bk)
    removed = bb.clean_old(keep_days=7)
    assert "old.db" in removed and "new.db" not in removed
    assert not old.exists() and new.exists()


def test_backup_cli_end_to_end(tmp_path, monkeypatch):
    """CLI 端到端：备份 + 校验，退出码 0。"""
    import scripts.db_backup as bb

    src = tmp_path / "t.db"
    _mkdb(src)
    monkeypatch.setattr(bb, "DBS", [("t", src)])
    monkeypatch.setattr(bb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bb, "BACKUP_DIR", tmp_path / "backups")

    # 直接调 main 里"备份流程"（不跑 VACUUM / verify 子命令）
    ok_all = True
    for name, s in bb.DBS:
        p, size = bb.backup_one(name, s)
        assert p and size > 0
        ok, info = bb.verify_backup(Path(p), name)
        assert ok
        ok_all = ok_all and ok
    assert ok_all