"""运维一体化 maintenance.py 回归：备份+保留策略+健康报告一次完成。"""
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_maintenance_run_backup_uses_submodules(tmp_path, monkeypatch):
    """run_backup 复用 db_backup：备份 + prune 资金快照 + 清理旧备份。"""
    import scripts.maintenance as mt

    src = tmp_path / "trader.db"
    con = sqlite3.connect(str(src))
    try:
        con.execute("CREATE TABLE equity_snapshots (equity REAL, cash REAL, "
                    "positions_value REAL, ts REAL)")
        con.execute("INSERT INTO equity_snapshots VALUES (100,50,50,?)",
                    (time.time() - 200 * 86400,))  # 超期
        con.execute("INSERT INTO equity_snapshots VALUES (110,60,50,?)",
                    (time.time(),))  # 近期
        con.commit()
    finally:
        con.close()

    # maintenance 内部 import 的是顶层 db_backup 模块，须 patch 同一对象
    monkeypatch.setattr(mt.db_backup, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mt.db_backup, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(mt.db_backup, "DBS", [("trader", src)])

    ok, detail = mt.run_backup(keep_days=7, prune_days=90)
    assert ok, f"备份应成功: {detail}"
    assert detail["backups"]["trader"]["integrity"] == "ok"
    assert detail["prune"]["deleted"] == 1, "应删 1 条超期快照"
    assert detail["prune"]["remaining"] == 1


def test_maintenance_health_uses_health_report(tmp_path, monkeypatch):
    """run_health 复用 daily_health_report 的判定（含 verdict 结构）。"""
    import scripts.maintenance as mt

    # maintenance 内部 import 的是顶层 daily_health_report（mt.health），
    # patch 同一对象（曾误 patch scripts.daily_health_report 导致失效）
    monkeypatch.setattr(mt.health, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(mt.health, "DATA_DIR", tmp_path)
    report = mt.run_health()
    assert "verdict" in report
    assert "health" in report["verdict"] and "reasons" in report["verdict"]
    # 空目录（无告警/日志）→ 健康
    assert report["verdict"]["health"] is True


def test_maintenance_writes_report_json(tmp_path, monkeypatch):
    """maintenance 把结果写 JSON 行到 maintenance_report.log。"""
    import scripts.maintenance as mt

    log = tmp_path / "maintenance_report.log"
    monkeypatch.setattr(mt, "REPORT_PATH", log)
    result = {"ts": "2026-01-01T00:00:00", "ok": True, "steps": {}, "exit_code": 0}
    mt.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with mt.REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    assert log.exists()
    parsed = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[0])
    assert parsed["ok"] is True and parsed["exit_code"] == 0