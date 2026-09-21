"""P2-9 部署助手：生成 Windows 服务/计划任务安装命令（不实际执行系统级操作）。

当前是"骨架只读"版本：打印可复制执行的命令，不自动安装——
避免脚本带系统级副作用（用户确认后自己执行）。

场景：
  A. Windows 计划任务（推荐，无需额外软件）：每日 08:00 健康报告 + 开机自启
  B. NSSM 服务（需先安装 nssm.exe）：把 run.py 注册为常驻服务
  C. 手动启动（无守护的兜底方式）

用法：
  python scripts/deploy_helper.py --print-commands
  python scripts/deploy_helper.py --check-prereqs   # 探测 nssm/计划任务可用性（只读）
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
RUN_PY = REPO / "run.py"
HEALTH = REPO / "scripts" / "daily_health_report.py"


def _win() -> bool:
    return sys.platform.startswith("win")


def check_prereqs() -> dict:
    nssm = shutil.which("nssm")
    return {
        "os": sys.platform,
        "python_exists": PYTHON.exists(),
        "run_py_exists": RUN_PY.exists(),
        "nssm_available": nssm,
    }


def print_commands() -> None:
    py = str(PYTHON)
    run = str(RUN_PY)
    health = str(HEALTH)
    print(f"# ===== 部署命令（{sys.platform}）=====")
    print(f"# 仓库: {REPO}")
    if not PYTHON.exists():
        print(f"# ⚠️ 未找到 venv 解释器: {PYTHON}")
    if not _win():
        print("# 非 Windows：以下为参考，实际按 systemd 单元文件部署（见注释）")
        print("# [Unit]/[Service] 示例：ExecStart={py} {run} --host 0.0.0.0 --port 8000")
        return

    print("""
# ---------- A. Windows 计划任务（推荐，无需额外软件）----------
# 1) 开机自启 Web 服务（最简：任务计划程序 GUI/命令行二选一）
schtasks /Create /TN "crypto_ai_trader_web" /TR "\"{py}\" \"{run}\" --host 0.0.0.0 --port 8000" /SC ONSTART /RL HIGHEST /RU SYSTEM /F
# 2) 每日 08:00 健康报告（--check 退出码供告警使用）
schtasks /Create /TN "crypto_ai_trader_health" /TR "\"{py}\" \"{health}\" --check" /SC DAILY /ST 08:00 /F
# 3) 每日 06:00 数据库备份（待 P0-1 备份脚本就绪后挂）

# ---------- B. NSSM 服务（需要先安装 nssm；运行中崩溃自动拉起）----------
# 安装 nssm 后执行：
REM set PATH=C:\\tools\\nssm;%PATH%
REM nssm install crypto_ai_trader "{py}" "{run}" --host 0.0.0.0 --port 8000
REM nssm set crypto_ai_trader AppDirectory "{repo}"
REM nssm set crypto_ai_trader AppStdout "{logs}\\server.log"
REM nssm set crypto_ai_trader AppStderr "{logs}\\server.err.log"
REM nssm set crypto_ai_trader Start SERVICE_AUTO_START
REM nssm start crypto_ai_trader
# 常用运维：
REM nssm restart crypto_ai_trader
REM nssm status crypto_ai_trader

# ---------- C. 手动启动（无守护兜底）----------
# 前台（终端保持打开）：
# "{py}" "{run}" --host 0.0.0.0 --port 8000
""".format(py=py, run=run, repo=REPO, logs=REPO / "data" / "logs", health=health))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-commands", action="store_true", default=True, help="打印部署命令（只读）")
    ap.add_argument("--check-prereqs", action="store_true", help="探测前置条件（只读）")
    args = ap.parse_args()
    if args.check_prereqs:
        import json
        print(json.dumps(check_prereqs(), ensure_ascii=False, indent=1))
    else:
        print_commands()


if __name__ == "__main__":
    main()