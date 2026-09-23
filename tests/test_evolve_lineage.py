"""TDD：evolve_rounds 记录 lineage（冠军/挑战者谱系），供曲线区分显示。

背景（2026-09-19 用户反馈）：strategy_drl 训练轮次曲线"一个有盈亏数据一个为0"
交替——根因是冠军/挑战者族群交替轮次：挑战者谱系长期 OOS 门失败（fitness=0.0）
且与冠军轮混在同一条 strategy_drl 曲线里。修复：落库 lineage 字段，
前端曲线 fitness 线只画主谱系、挑战者轮统一标为"挑战者未晋升"点。
"""
import pytest
from sqlalchemy import text

from core.bus import EventBus
from core.database import Database, EvolveRound
from drl.evolve_engine import EvolveEngine

_OLD_SCHEMA = """
CREATE TABLE evolve_rounds (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  model VARCHAR(32), symbol VARCHAR(32) DEFAULT '',
  timeframe VARCHAR(16) DEFAULT '', data_source VARCHAR(16) DEFAULT 'exchange',
  round_no INTEGER DEFAULT 0, fitness FLOAT DEFAULT 0.0, oos_ret FLOAT DEFAULT 0.0,
  decay FLOAT DEFAULT 0.0, position_ratio FLOAT DEFAULT 0.0,
  selected_factors TEXT DEFAULT '[]', status VARCHAR(32) DEFAULT 'ok',
  audit_json TEXT DEFAULT '{}'
)
"""


def test_evolve_round_model_has_lineage():
    """模型字段存在（构造时为 None，INSERT 时按 default 写 ''，兼容旧行）。"""
    r = EvolveRound()
    assert r.lineage in (None, "")


@pytest.mark.asyncio
async def test_db_migration_adds_lineage_column(tmp_path):
    """既有库（无 lineage 列）init() 时幂等补列。"""
    import aiosqlite
    db_path = tmp_path / "old.db"
    async with aiosqlite.connect(str(db_path)) as con:
        await con.execute(_OLD_SCHEMA)
        await con.commit()
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    await db.init()
    try:
        async with db.engine.connect() as conn:
            cols = [c[1] for c in (await conn.execute(
                text("PRAGMA table_info(evolve_rounds)"))).fetchall()]
        assert "lineage" in cols
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_log_round_persists_lineage(tmp_path):
    """_log_round 的 lineage 参数必须落库。"""
    db_path = tmp_path / "evolve.db"
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    await db.init()
    try:
        eng = EvolveEngine(db, EventBus(), tmp_path / "models")
        await eng._log_round("strategy_drl", "ETH/USDT", "5m", "exchange",
                             0.01, status="rollback", lineage="challenger")
        async with db.session() as s:
            rows = (await s.execute(text("select lineage from evolve_rounds"))).fetchall()
        assert rows and rows[0][0] == "challenger"
    finally:
        await db.close()
