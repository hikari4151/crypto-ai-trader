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
import os
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
    # Windows GBK 控制台：precompress/tailwind/echarts 构建脚本输出 ✅/❌ emoji，
    # Python 默认按 GBK 编码 stdout 会抛 UnicodeEncodeError 并 exit 1（门禁误报）。
    # 强制子进程 stdout 用 UTF-8——只影响这三个前端产物校验脚本，其余步骤行为不变。
    env = dict(os.environ)
    if label in ("check_precompressed", "check_tailwind_artifact",
                 "check_echarts_artifact"):
        env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[verify][FAIL] {label} exit={r.returncode}", flush=True)
        return False
    print(f"[verify][OK] {label}", flush=True)
    return True


def get_staged_files() -> list[str]:
    """返回本次暂存（待提交）的文件列表（pre-commit 场景）。"""
    r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def build_steps(skip_pytest: bool = False, skip_backtest: bool = False,
                staged: bool = False) -> list[tuple[str, list[str]]]:
    # 性能基线门禁：fast 链（本地/pre-commit）用 --quick——跳过 100k/批量两条
    # 重量级路径（合计 ~2.3s，曾占门禁链耗时 66%）；完整链/CI 全量兜底复杂度
    # 回归（O(n²) 化必须靠 100k 档才能暴露）。
    perf_cmd = [PY, str(ROOT / "scripts" / "perf_baseline.py")]
    if skip_pytest:
        perf_cmd.append("--quick")
    steps: list[tuple[str, list[str]]] = [
        ("check_lock", [PY, str(ROOT / "scripts" / "check_lock.py")]),
    ]
    if staged:
        # 第十六轮：pre-commit 用 staged 模式——只校验本次暂存文件。
        # 背景：hook 全量校验会被工作树里任何未完成工作（他人 WIP / 进行中编辑）
        # 阻塞，导致团队被迫 --no-verify 绕钩子。staged 模式只查待提交内容：
        #   - ruff 只查暂存文件（不查全目录）
        #   - check_redlines 的 E7（index.html 全量扫描）仅当 index.html 被本次
        #     暂存时才执行（未碰 index.html 的提交不受他人 WIP 影响）
        #   - 其余快步骤（validate_config/check_js/check_encoding/compileall）
        #     保持全量（它们秒级且属全局性契约）
        staged_files = get_staged_files()
        staged_py = [f for f in staged_files if f.endswith(".py")]
        if staged_py:
            steps.append(("ruff(staged)", [PY, "-m", "ruff", "check", *staged_py]))
        else:
            steps.append(("ruff(staged)", [PY, "-m", "ruff", "check", "--exit-zero"]))
        index_staged = any(f == "web/static/index.html" for f in staged_files)
        if index_staged:
            steps.append(("check_redlines", [PY, str(ROOT / "scripts" / "check_redlines.py")]))
        else:
            # 本次未碰 index.html：跳过全量 E7 基线（他人 WIP 的红线不阻塞本提交）
            steps.append(("check_redlines(skip-e7)", [PY, "-c", "import sys; sys.exit(0)"]))
    else:
        steps.append(("ruff", [PY, "-m", "ruff", "check", *RUFF_DIRS]))
        steps.append(("check_redlines", [PY, str(ROOT / "scripts" / "check_redlines.py")]))
    steps.extend([
        # mypy 试点范围由 pyproject.toml [tool.mypy].files 声明（单一事实来源，
        # 避免此处与 CI 各写一份文件清单发生漂移）
        ("mypy_pilot", [PY, "-m", "mypy"]),
        ("validate_config", [PY, str(ROOT / "scripts" / "validate_config.py")]),
        ("check_doc_sync", [PY, str(ROOT / "scripts" / "check_doc_sync.py")]),
        ("check_js", [PY, str(ROOT / "scripts" / "check_js.py")]),
        # F1/F2 编码红线：入库文本文件 UTF-8 无 BOM、.bat 纯 ASCII
        ("check_encoding", [PY, str(ROOT / "scripts" / "check_encoding.py")]),
        # 本地/pre-commit 用宽松模式：正在编辑 index.html 时不拦人（漂移会打印 WARN）。
        # CI 用 `--check-only --require-manifest`（干净检出 ⇒ 等价于"清单必须描述已提交的 index.html"）
        ("build_frontend", [PY, str(ROOT / "scripts" / "build_frontend.py"), "--check-only"]),
        ("compileall", [PY, "-m", "compileall", "-q", *COMPILE_DIRS]),
        # 第十六轮：前端构建产物校验——precompress .gz 最新、tailwind/echarts 产物
        # SRI 与 index.html 一致（防"改了产物/源忘了重建或更新 SRI"；scripts 见
        # precompress_static.py --check / build_tailwind_css.py --check / build_echarts_custom.py --check）。
        ("check_precompressed", [PY, str(ROOT / "scripts" / "precompress_static.py"), "--check"]),
        ("check_tailwind_artifact", [PY, str(ROOT / "scripts" / "build_tailwind_css.py"), "--check"]),
        ("check_echarts_artifact", [PY, str(ROOT / "scripts" / "build_echarts_custom.py"), "--check"]),
        ("perf_baseline", perf_cmd),
    ])
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
    ap.add_argument("--staged", action="store_true",
                    help="只校验本次暂存文件（pre-commit 用；他人未完成工作不阻塞本提交）")
    args = ap.parse_args(argv)

    steps = build_steps(skip_pytest=args.skip_pytest, skip_backtest=args.skip_backtest,
                        staged=args.staged)
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
