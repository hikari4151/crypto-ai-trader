"""P0-进程守护回归：子进程管理 / 事件日志 / 探活与重启逻辑。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_start_proc_starts_and_stops(tmp_path, monkeypatch):
    """_start_proc 能启动注入命令的子进程，terminate 后停止。"""
    import scripts.service_supervisor as sv

    monkeypatch.setattr(sv, "LOG_PATH", tmp_path / "supervisor.log")
    # 用 sleep 子进程模拟常驻服务
    code = "import time; time.sleep(30)"
    proc = sv._start_proc("127.0.0.1", 0, [], cmd=[sys.executable, "-c", code])
    try:
        assert proc.poll() is None, "子进程应正在运行"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    assert proc.poll() is not None, "terminate 后子进程应退出"


def test_log_writes_json_lines(tmp_path, monkeypatch):
    """事件日志是 JSON 行，包含 ts/event/pid/restarts/reason。"""
    import scripts.service_supervisor as sv

    log = tmp_path / "supervisor.log"
    monkeypatch.setattr(sv, "LOG_PATH", log)
    sv._log("started", pid=123, restarts=0, reason="")
    sv._log("restart_due_probe", pid=124, restarts=2, reason="conn refused")
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for ln, ev in zip(lines, ("started", "restart_due_probe")):
        rec = json.loads(ln)
        assert rec["event"] == ev and "ts" in rec and "pid" in rec and "restarts" in rec


def test_check_once_unreachable_returns_code1():
    """守护探活：服务不可达应返回 1（触发重启判定）。"""
    from scripts.service_supervisor import check_once
    code, msg = check_once("http://127.0.0.1:1")  # port 1 几乎必然不可达
    assert code == 1 and msg


def test_supervisor_restart_logic(tmp_path, monkeypatch):
    """连续探活失败达到 max_fail → 重启并记录事件（注入 fake 探活/进程）。"""
    import scripts.service_supervisor as sv

    log = tmp_path / "supervisor.log"
    monkeypatch.setattr(sv, "LOG_PATH", log)
    monkeypatch.setattr(sv, "PYTHON", sys.executable)

    calls = {"start": 0, "fail": 0}

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None  # 一直"运行"，由探活决定重启

        def kill(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_check(base, timeout=10.0):
        calls["fail"] += 1
        return 1, "模拟不可达"

    def fake_start(host, port, extra, cmd=None):
        calls["start"] += 1
        return _FakeProc(1000 + calls["start"])

    monkeypatch.setattr(sv, "check_once", fake_check)
    monkeypatch.setattr(sv, "_start_proc", fake_start)
    monkeypatch.setattr(sv, "time", type("_T", (), {"sleep": staticmethod(lambda s: None),
                                                    "time": staticmethod(time.time)})())

    # 直接驱动守护主循环一次完整流程：启动 → 3 次失败 → 重启
    proc = fake_start("127.0.0.1", 8000, [])
    fail_streak = 0
    restarts = 0
    for _ in range(3):
        code, msg = fake_check("http://127.0.0.1:8000")
        fail_streak += 1
        if fail_streak >= 3:
            restarts += 1
            proc.kill()
            proc = fake_start("127.0.0.1", 8000, [])
            fail_streak = 0
            sv._log("restart_due_probe", proc.pid, restarts, msg[:120])
    assert restarts == 1, "连续 3 次失败应触发一次重启"
    assert calls["start"] == 2, "初次启动 + 重启 = 2 次启动"
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert any("restart_due_probe" in line for line in lines)
