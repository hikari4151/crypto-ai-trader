"""P0-进程守护（service supervisor）：拉起并守护 Web 服务进程。

与 scripts/watchdog.py 的分工：
- watchdog.py：只读探针（外部 cron/supervisor 用），不拉起
- 本脚本：常驻守护——负责启动子进程、周期探活、失败自动重启

能力：
- 启动子进程 `run.py`（web 服务）
- 每 interval 秒探活 `/api/ops/overview`（复用 watchdog.check_once）
- 连续失败 >= max_fail 次 → 杀掉并重启子进程（记录到 supervisor.log）
- 子进程自行崩溃退出 → 自动拉起
- 事件日志：data/logs/supervisor.log（JSON 行：ts/event/pid/restarts/reason）

用法：
  python scripts/service_supervisor.py [--port 8000] [--interval 30] [--max-fail 3]
  （Ctrl+C 优雅退出并停止子进程）
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "logs" / "supervisor.log"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# 复用 watchdog 探活逻辑（避免重复实现健康判定）
sys.path.insert(0, str(ROOT / "scripts"))
from watchdog import check_once, _load_token  # noqa: E402


def _log(event: str, pid: int, restarts: int, reason: str = "") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
           "pid": pid, "restarts": restarts, "reason": reason}
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[supervisor] 写日志失败: {e}", file=sys.stderr)
    print(f"[supervisor] {event} pid={pid} restarts={restarts} {reason}", flush=True)


def _start_proc(host: str, port: int, extra: list[str],
                cmd: list[str] | None = None) -> subprocess.Popen:
    cmd = cmd or [str(PYTHON), str(ROOT / "run.py"), "web",
                  "--host", host, "--port", str(port), *extra]
    # P0-3：uvicorn 日志已由 core.logging_config.uvicorn_log_config 写入
    # server.log（带轮转），这里 stdout 丢弃即可——若仍重定向到 server.log
    # 会和 file handler 双写同一文件（且绕过轮转）。
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=int, default=30, help="探活间隔秒")
    ap.add_argument("--max-fail", type=int, default=3, help="连续失败多少次触发重启")
    ap.add_argument("--base", default="", help="探活 URL（缺省 http://host:port）")
    args = ap.parse_args()
    base = args.base or f"http://{args.host}:{args.port}"
    if not PYTHON.exists():
        print(f"[supervisor] 未找到解释器 {PYTHON}", file=sys.stderr)
        return 1

    proc: subprocess.Popen | None = None
    fail_streak = 0
    restarts = 0
    stop = False

    def _handle_sig(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print(f"[supervisor] 守护 {base}（interval={args.interval}s max_fail={args.max_fail}）", flush=True)
    while not stop:
        # 子进程不在/已退出 → 拉起
        if proc is None or proc.poll() is not None:
            code = proc.poll() if proc is not None else None
            if proc is not None:
                restarts += 1
                _log("restart_due_exit", proc.pid, restarts, f"exit_code={code}")
            proc = _start_proc(args.host, args.port, [])
            _log("started", proc.pid, restarts)
            time.sleep(max(2.0, args.interval * 0.5))  # 启动宽限
            continue

        # 探活
        code, msg = check_once(base)
        if code == 0:
            if fail_streak:
                _log("recovered", proc.pid, restarts, f"after {fail_streak} fails")
            fail_streak = 0
        else:
            fail_streak += 1
            _log("probe_fail", proc.pid, restarts, f"{code}: {msg[:120]} (streak={fail_streak})")
            if fail_streak >= args.max_fail:
                restarts += 1
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
                _log("restart_due_probe", proc.pid, restarts, msg[:120])
                proc = None
                continue
        time.sleep(args.interval)

    # 退出：停掉子进程
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        _log("stopped", proc.pid, restarts)
    print(f"[supervisor] 退出（共重启 {restarts} 次）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())