"""统一验证链（P1-B1）：一条命令跑完整质量门禁，CI / pre-commit 可复用。

默认顺序：
  check_lock → ruff → mypy_pilot → validate_config → check_doc_sync → check_js → build_frontend → compileall
  → pytest(--cov --cov-fail-under=65) → coverage_json → coverage_gate(模块级) → 双引擎基线回测

用法：
  python scripts/verify_all.py                    # 完整链
  python scripts/verify_all.py --skip-backtest    # 跳过双引擎基线
  python scripts/verify_all.py --skip-pytest --skip-backtest   # fast 门禁（pre-commit 用）
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

RUFF_DIRS = ("core/", "engine/", "ai/", "web/", "scripts/", "backtest/",
             "strategies/", "indicators/", "config/", "drl/", "exchange/", "factors/",
             "net/", "tests/")
COMPILE_DIRS = ("core", "engine", "ai", "web", "scripts", "backtest",
                "strategies", "indicators", "config", "drl", "exchange", "factors",
                "net", "tests", "run.py")


def run(cmd: list[str], label: str) -> bool:
    print(f"[verify] {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[verify][FAIL] {label} exit={r.returncode}", flush=True)
        return False
    print(f"[verify][OK] {label}", flush=True)
    return True


def build_steps(skip_pytest: bool = False, skip_backtest: bool = False) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("check_lock", [PY, str(ROOT / "scripts" / "check_lock.py")]),
        ("ruff", [PY, "-m", "ruff", "check", *RUFF_DIRS]),
        # mypy 试点范围由 pyproject.toml [tool.mypy].files 声明（单一事实来源，
        # 避免此处与 CI 各写一份文件清单发生漂移）
        ("mypy_pilot", [PY, "-m", "mypy"]),
        ("validate_config", [PY, str(ROOT / "scripts" / "validate_config.py")]),
        ("check_doc_sync", [PY, str(ROOT / "scripts" / "check_doc_sync.py")]),
        ("check_js", [PY, str(ROOT / "scripts" / "check_js.py")]),
        # F1/F2 编码红线：入库文本文件 UTF-8 无 BOM、.bat 纯 ASCII
        ("check_encoding", [PY, str(ROOT / "scripts" / "check_encoding.py")]),
        # grep 型红线：B8 禁交易所 symbol 手工拆分、E2 禁 CDN、E7 !important 基线
        ("check_redlines", [PY, str(ROOT / "scripts" / "check_redlines.py")]),
        # 本地/pre-commit 用宽松模式：正在编辑 index.html 时不拦人（漂移会打印 WARN）。
        # CI 用 `--check-only --require-manifest`（干净检出 ⇒ 等价于"清单必须描述已提交的 index.html"）
        ("build_frontend", [PY, str(ROOT / "scripts" / "build_frontend.py"), "--check-only"]),
        ("compileall", [PY, "-m", "compileall", "-q", *COMPILE_DIRS]),
    ]
    if not skip_pytest:
        steps.append(("pytest_cov", [PY, "-m", "pytest", "-q", "--cov", "--cov-fail-under=65"]))
        cov_json = str(ROOT / ".optim" / "coverage_gate.json")
        steps.append(("coverage_json", [PY, "-m", "coverage", "json", "-o", cov_json]))
        steps.append(("coverage_gate", [PY, str(ROOT / "scripts" / "coverage_gate.py"), "--json", cov_json]))
    if not skip_backtest:
        steps.append(("backtest_baseline", [PY, str(ROOT / "run.py"), "backtest",
                                            "--source", "demo", "--strategy", "dual_ma"]))
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="统一验证链（REDEV_GUIDE §4 自动化）")
    ap.add_argument("--skip-pytest", action="store_true", help="跳过全量 pytest（--cov）")
    ap.add_argument("--skip-backtest", action="store_true", help="跳过双引擎基线回测")
    args = ap.parse_args(argv)

    steps = build_steps(skip_pytest=args.skip_pytest, skip_backtest=args.skip_backtest)
    failed: list[str] = []
    for label, cmd in steps:
        if not run(cmd, label):
            failed.append(label)
    if failed:
        print(f"[verify] FAILED: {', '.join(failed)}")
        return 1
    print("[verify] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
