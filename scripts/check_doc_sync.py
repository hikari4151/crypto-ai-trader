"""C5 文档同步机制：核对 REDEV_GUIDE §4 验证链与 CI 门禁目录清单不漂移。

背景：docs/ 与 .github 若各写一套目录清单，迟早会像 §2.8 那样漂移
（如 CI 漏掉 exchange/factors）。本脚本把"核心目录集合"作为契约，
校验：
  1. ci.yml 的 compileall 覆盖全部核心目录 + run.py + tests
  2. ci.yml 的 ruff 覆盖全部核心目录（除 tests，lint 单独推进）
  3. REDEV_GUIDE.md §4 的 compileall 至少覆盖核心目录（本地链含 run.py）

用法：
  python scripts/check_doc_sync.py    # 0=一致 1=有漂移（并打印差异）
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
REDEV = ROOT / "docs" / "REDEV_GUIDE.md"

CORE_DIRS = {"core", "engine", "ai", "web", "scripts", "backtest", "strategies",
             "indicators", "config", "drl", "exchange", "factors", "tests"}


def extract_dirs(text: str) -> set[str]:
    """从一条命令文本中提取出现过的核心目录/ run.py token（含 .in 免误匹配）。"""
    found: set[str] = set()
    for tok in CORE_DIRS | {"run.py"}:
        if tok == "run.py":
            pattern = re.escape("run.py")
        else:
            # 目录 token 后可能跟 /（ruff 命令用 core/ engine/ 形式）
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(tok) + r"(?![A-Za-z0-9_])"
        if re.search(pattern, text):
            found.add(tok)
    return found


def _capture(text: str, needle: str) -> set[str]:
    m = re.search(re.escape(needle) + r" (.+?)(?:\n|$)", text)
    return extract_dirs(m.group(1)) if m else set()


def check() -> list[str]:
    issues: list[str] = []
    ci = CI.read_text(encoding="utf-8")
    if not REDEV.exists():
        issues.append(f"缺少 {REDEV.name}")
        return issues

    ci_compile = _capture(ci, "python -m compileall -q")
    ci_ruff = _capture(ci, "python -m ruff check")
    redev_compile = extract_dirs(REDEV.read_text(encoding="utf-8"))

    req_compile = CORE_DIRS | {"run.py"}
    req_ruff = CORE_DIRS - {"tests"}
    req_redev = CORE_DIRS - {"tests"} | {"run.py"}

    miss = req_compile - ci_compile
    if miss:
        issues.append(f"ci.yml compileall 缺目录: {sorted(miss)}")
    miss = req_ruff - ci_ruff
    if miss:
        issues.append(f"ci.yml ruff 缺目录: {sorted(miss)}")
    miss = req_redev - redev_compile
    if miss:
        issues.append(f"REDEV_GUIDE §4 编译链缺目录: {sorted(miss)}")

    if "ruff" in ci_ruff and "tests" in ci_ruff:
        issues.append("ci.yml ruff 不应包含 tests（tests 的 F8xx 另走 B6 专项）")
    return issues


def main() -> int:
    if not CI.exists():
        print(f"[docsync] 缺少 {CI}")
        return 1
    issues = check()
    if issues:
        for i in issues:
            print(f"[docsync] [FAIL] {i}")
        print("[docsync] FAILED")
        return 1
    print("[docsync] 通过：REDEV_GUIDE §4 与 ci.yml 门禁目录集一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
