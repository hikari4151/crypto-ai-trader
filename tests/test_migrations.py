"""版本化迁移回归：版本递增 / 幂等 / 失败停止 / CLI 检查。"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mkdb(path: Path) -> None:
    """构造带 evolve_rounds 表的库（迁移的目标表）。"""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE IF NOT EXISTS evolve_rounds ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, model VARCHAR(64), "
                "round_no INTEGER, fitness REAL, params_json TEXT)")
    con.commit()
    con.close()


def test_current_version_zero_and_apply(tmp_path):
    """新库版本 0 → 应用迁移 → 版本递增到最新。"""
    from core.migrations import MIGRATIONS, apply, current_version

    db = tmp_path / "t.db"
    _mkdb(db)
    assert current_version(db) == 0
    records = apply(db)
    assert all(r["applied"] for r in records), records
    assert current_version(db) == MIGRATIONS[-1].version

    # 列确实存在
    con = sqlite3.connect(str(db))
    cols = {c[1] for c in con.execute("PRAGMA table_info(evolve_rounds)").fetchall()}
    con.close()
    assert "audit_json" in cols and "lineage" in cols


def test_apply_is_idempotent(tmp_path):
    """重复 apply：第二次无待执行迁移（幂等）。"""
    from core.migrations import MIGRATIONS, apply, current_version

    db = tmp_path / "t.db"
    _mkdb(db)
    r1 = apply(db)
    r2 = apply(db)
    assert r1 and not r2, "第二次应无迁移执行"
    assert current_version(db) == MIGRATIONS[-1].version


def test_pending_reports_only_new(tmp_path):
    """pending 只返回版本 > 当前的迁移。"""
    from core.migrations import MIGRATIONS, apply, pending

    db = tmp_path / "t.db"
    _mkdb(db)
    apply(db)
    # 手动把版本往后拨一位 → 应有 1 个 pending
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA user_version = %d" % (MIGRATIONS[-1].version - 1,))
    con.commit()
    con.close()
    todo = pending(db)
    assert len(todo) == 1 and todo[0].version == MIGRATIONS[-1].version


def test_dry_run_does_not_write(tmp_path):
    """dry_run 不写库：版本不变、无列添加。"""
    from core.migrations import apply, current_version

    db = tmp_path / "t.db"
    _mkdb(db)
    records = apply(db, dry_run=True)
    assert records and all(r.get("dry_run") for r in records)
    assert current_version(db) == 0, "dry_run 不应改版本"

    con = sqlite3.connect(str(db))
    cols = {c[1] for c in con.execute("PRAGMA table_info(evolve_rounds)").fetchall()}
    con.close()
    assert "audit_json" not in cols, "dry_run 不应加列"


def test_cli_check_reports_pending(tmp_path, monkeypatch):
    """CLI --check：有待应用迁移时退出码 1。"""
    import sys
    import scripts.migrate_db as m

    db = tmp_path / "t.db"
    _mkdb(db)
    monkeypatch.setattr(sys, "argv", ["migrate_db.py", "--check", "--db", str(db)])
    assert m.main() == 1


def test_cli_apply_then_check_zero(tmp_path, monkeypatch):
    """CLI --apply 后 --check 退出码 0（已最新）。"""
    import sys
    import scripts.migrate_db as m

    db = tmp_path / "t.db"
    _mkdb(db)
    monkeypatch.setattr(sys, "argv", ["migrate_db.py", "--apply", "--db", str(db)])
    assert m.main() == 0
    monkeypatch.setattr(sys, "argv", ["migrate_db.py", "--check", "--db", str(db)])
    assert m.main() == 0


def test_apply_guard_skips_existing_columns(tmp_path):
    """列已存在（旧 init 手工 ALTER 过）但版本为 0：guard 跳过，不报重复列。"""
    from core.migrations import MIGRATIONS, apply, current_version

    db = tmp_path / "t.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE evolve_rounds (id INTEGER PRIMARY KEY, model VARCHAR(64), "
                "round_no INTEGER, fitness REAL, params_json TEXT, "
                "audit_json TEXT NOT NULL DEFAULT '{}', "
                "lineage VARCHAR(16) NOT NULL DEFAULT '')")  # 列已存在
    con.commit()
    con.close()
    assert current_version(db) == 0

    records = apply(db)
    assert all(r["applied"] for r in records), records
    # 前两个迁移被 guard 跳过（skipped_by_guard），v3 索引真正执行
    skipped = [r for r in records if r.get("skipped_by_guard")]
    assert len(skipped) == 2, f"audit_json/lineage 应被 guard 跳过: {records}"
    assert current_version(db) == MIGRATIONS[-1].version