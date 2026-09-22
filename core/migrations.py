"""版本化数据库迁移（SQLite PRAGMA user_version 驱动）。

现状：core/database.py 的 init() 里散落幂等 DDL / 手动 ALTER（audit_json、
lineage 等），无版本记录——新列靠"PRAGMA table_info 检查后 ALTER"兜底，
迁移顺序与历史不可见。

本模块：把迁移收敛为**版本化列表**，用 SQLite 原生 `PRAGMA user_version`
记录当前 schema 版本（无需额外版本表），顺序执行未应用的迁移。
幂等：已应用的版本自动跳过；每个迁移应在 SQLite 幂等语义下可重入。

用法（代码）：
  from core.migrations import migrate_sqlite
  await migrate_sqlite(db)              # db: core.database.Database

用法（CLI，独立入口不依赖 app）：
  python scripts/migrate_db.py --check    # 只报告当前版本与待应用迁移
  python scripts/migrate_db.py --apply    # 应用未执行的迁移（写库）
  python scripts/migrate_db.py --list     # 列出全部迁移

注意：本模块独立于 database.py（该文件可能被并行工作流编辑）；
接线到 init() 的时机由调用方决定。
"""
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    # guard_sql：可选，执行前先跑；若返回非空行则视为"已应用"跳过（幂等，
    # 兼容旧 init 已手工 ALTER 过列的库，避免 duplicate column 报错）
    guard_sql: str = ""


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="evolve_rounds_audit_json",
        statements=(
            "ALTER TABLE evolve_rounds ADD COLUMN audit_json TEXT NOT NULL DEFAULT '{}'",
        ),
        notes="evolve_rounds 审计信息列（原散落在 database.py init 的手动 ALTER）",
        guard_sql="SELECT 1 FROM pragma_table_info('evolve_rounds') WHERE name='audit_json'",
    ),
    Migration(
        version=2,
        name="evolve_rounds_lineage",
        statements=(
            "ALTER TABLE evolve_rounds ADD COLUMN lineage VARCHAR(16) NOT NULL DEFAULT ''",
        ),
        notes="evolve_rounds 族群谱系列（champion/challenger 谱系分离）",
        guard_sql="SELECT 1 FROM pragma_table_info('evolve_rounds') WHERE name='lineage'",
    ),
    Migration(
        version=3,
        name="evolve_rounds_model_indexes",
        statements=(
            "CREATE INDEX IF NOT EXISTS ix_evolve_rounds_model_id "
            "ON evolve_rounds (model, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_evolve_rounds_model_round_no "
            "ON evolve_rounds (model, round_no DESC)",
        ),
        notes="evolve_rounds 查询热路径复合索引（原 init 幂等 DDL）",
    ),
)


def current_version(db_path: Path) -> int:
    """读取库的当前 schema 版本（PRAGMA user_version）。"""
    con = sqlite3.connect(str(db_path))
    try:
        return int(con.execute("PRAGMA user_version").fetchone()[0] or 0)
    finally:
        con.close()


def pending(db_path: Path) -> list[Migration]:
    """返回待应用的迁移（version > 当前版本，且 guard 未满足）。

    guard 已满足的迁移视为已被旧 init 手工执行过（列已存在等），
    不列入待办——避免对"已 ALTER 过列"的真实库重复 ADD COLUMN 报错。
    """
    ver = current_version(db_path)
    if ver >= max((m.version for m in MIGRATIONS), default=0):
        return []
    con = sqlite3.connect(str(db_path))
    try:
        todo = []
        for m in MIGRATIONS:
            if m.version <= ver:
                continue
            if m.guard_sql and _guard_satisfied(con, m):
                continue  # 已被手工/旧 init 应用，跳过
            todo.append(m)
        return todo
    finally:
        con.close()


def _guard_satisfied(con: sqlite3.Connection, m: Migration) -> bool:
    """guard_sql 有返回行 = 已应用。guard 执行出错（如表不存在）视为未满足。"""
    try:
        row = con.execute(m.guard_sql).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def apply(db_path: Path, dry_run: bool = False) -> list[dict]:
    """应用所有待执行迁移。dry_run=True 只返回计划不写库。返回执行记录。"""
    todo = pending(db_path)
    if dry_run:
        return [{"version": m.version, "name": m.name, "applied": False,
                 "dry_run": True, "notes": m.notes} for m in todo]
    records = []
    con = sqlite3.connect(str(db_path))
    try:
        ver = current_version(db_path)
        for m in MIGRATIONS:
            if m.version <= ver:
                continue
            try:
                if m.guard_sql and _guard_satisfied(con, m):
                    # 已被旧 init 手工应用（列/索引已存在）：只推进版本不重复执行
                    con.execute(f"PRAGMA user_version = {m.version}")
                    con.commit()
                    records.append({"version": m.version, "name": m.name,
                                    "applied": True, "skipped_by_guard": True,
                                    "dry_run": False})
                    continue
                for stmt in m.statements:
                    con.execute(stmt)
                con.execute(f"PRAGMA user_version = {m.version}")
                con.commit()
                records.append({"version": m.version, "name": m.name,
                                "applied": True, "dry_run": False})
            except sqlite3.Error as e:
                con.rollback()
                records.append({"version": m.version, "name": m.name,
                                "applied": False, "error": str(e)})
                break  # 迁移失败即停：避免部分应用造成版本错乱
    finally:
        con.close()
    return records