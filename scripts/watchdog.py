"""运维看门狗探针（watchdog）：独立进程检查实盘服务健康。

用途：供外部 supervisor / systemd / cron 定时调用（每 30~60s 一次）。
退出码语义（脚本化监控友好）：
  0 = 健康
  1 = 服务不可达（进程死了 / 端口不通）
  2 = 引擎假死（REST 活着但引擎主循环 heartbeat 停滞 / 行情断流 / CRITICAL 告警）

用法：
  .venv/Scripts/python scripts/watchdog.py [--base http://127.0.0.1:8000]
  .venv/Scripts/python scripts/watchdog.py --loop 5 --interval 10   # 持续巡检模式
"""
import argparse
import sys
import time
import urllib.request
import json
from pathlib import Path

import yaml  # type: ignore


def _load_token(base: str) -> str:
    """从 settings 或 data/api_token.txt 读取 API token。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config.settings import settings
        if getattr(settings, "api_token", ""):
            return settings.api_token
    except Exception:  # noqa: BLE001
        pass
    try:
        p = Path(__file__).resolve().parent.parent / "data" / "api_token.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        pass
    # 兜底：从应用生成的 token 配置里找
    try:
        cfg = Path(__file__).resolve().parent.parent / "data" / "api_token.yaml"
        if cfg.exists():
            return yaml.safe_load(cfg.read_text(encoding="utf-8")).get("api_token", "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def check_once(base: str, timeout: float = 10.0) -> tuple[int, str]:
    url = f"{base.rstrip('/')}/api/ops/overview"
    req = urllib.request.Request(url, headers={"X-API-Token": _load_token(base)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return 1, f"服务不可达 {base}: {e}"
    health = data.get("health", False)
    eng = data.get("engine") or {}
    if not eng.get("loop_alive", True):
        return 2, (f"引擎假死：heartbeat_age={eng.get('heartbeat_age_sec')}s "
                   f"(阈值60s)，REST 活着但主循环停滞")
    if data.get("market", {}).get("stale"):
        return 2, f"行情断流 {data['market'].get('stale_sec')}s"
    if (data.get("alerts") or {}).get("critical_recent", 0) > 0:
        return 2, f"存在 {data['alerts']['critical_recent']} 条 CRITICAL 告警"
    if not health:
        return 2, "health=false（聚合判定异常）"
    return 0, (f"健康: mode={eng.get('mode')} loop_alive={eng.get('loop_alive')} "
               f"hb_age={eng.get('heartbeat_age_sec')}s "
               f"critical={data['alerts']['critical_recent']} stale={data['market'].get('stale')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="实盘服务看门狗探针")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--loop", type=int, default=0, help="持续巡检次数（0=单次）")
    ap.add_argument("--interval", type=float, default=30.0, help="巡检间隔秒")
    args = ap.parse_args()

    if args.loop <= 0:
        code, msg = check_once(args.base)
        print(f"[watchdog] {msg}", file=sys.stderr if code else sys.stdout)
        sys.exit(code)

    bad = 0
    for i in range(args.loop):
        code, msg = check_once(args.base)
        print(f"[watchdog][{i + 1}/{args.loop}] {msg}")
        if code != 0:
            bad += 1
            if bad >= 3:
                print(f"[watchdog] 连续 {bad} 次异常，提前退出 code={code}", file=sys.stderr)
                sys.exit(code)
        time.sleep(args.interval)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
