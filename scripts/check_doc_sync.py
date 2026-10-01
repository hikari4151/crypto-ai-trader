"""C5/A5 门禁一致性契约：核对 REDEV_GUIDE §4 / ci.yml / verify_all / 覆盖率配置的目录清单不漂移。

背景：docs/ 与 .github 若各写一套目录清单，迟早会漂移（§2.8 曾出现 CI 漏掉 exchange/factors）。
2026-09-27 又出现第二种漂移：新增 `net/` 包被 ruff/compileall/覆盖率三处同时漏掉——
门禁"全绿"但新模块实际上从没被检查过。因此本脚本把**目录清单本身**当契约，校验：
  1. 契约集合 CORE_DIRS 覆盖工作区真实存在的全部顶层 Python 包（自检，防新包漏登记）
  2. ci.yml 的 compileall 覆盖 CORE_DIRS + run.py
  3. ci.yml 的 ruff 覆盖 CORE_DIRS（含 tests，B6 已纳入）
  4. verify_all.py 的 RUFF_DIRS / COMPILE_DIRS 与契约一致
  5. .coveragerc 的 source 与 coverage_gate.py 的 THRESHOLDS 覆盖全部被测模块
  6. REDEV_GUIDE.md §4 的编译链至少覆盖 CORE_DIRS + run.py

用法：
  python scripts/check_doc_sync.py    # 0=一致 1=有漂移（并打印差异）
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
REDEV = ROOT / "docs" / "REDEV_GUIDE.md"
VERIFY_ALL = ROOT / "scripts" / "verify_all.py"
COVERAGERC = ROOT / ".coveragerc"
COVERAGE_GATE = ROOT / "scripts" / "coverage_gate.py"

# 门禁契约：所有需要被 lint / 编译 / 覆盖率覆盖的顶层包。
CORE_DIRS = {"core", "engine", "ai", "web", "scripts", "backtest", "strategies",
             "indicators", "config", "drl", "exchange", "factors", "net"}
# 不参与门禁的顶层目录（数据/文档/虚拟环境/工具产物），显式登记避免误报。
NON_GATE_DIRS = {"data", "docs", "vendor", "bin"}
# tests 参与 ruff/compileall，但不参与覆盖率 source（测试自身不计覆盖率）。
COVERED_DIRS = CORE_DIRS - {"scripts"}
GATE_DIRS = CORE_DIRS | {"tests"}


def discover_packages() -> set[str]:
    """工作区真实存在的顶层 Python 包（含 .py 的目录）。"""
    found: set[str] = set()
    for p in ROOT.iterdir():
        if not p.is_dir() or p.name.startswith(".") or p.name.endswith("__pycache__"):
            continue
        if p.name in NON_GATE_DIRS:
            continue
        try:
            has_py = any(f.suffix == ".py" for f in p.iterdir() if f.is_file())
        except OSError:
            continue
        if has_py:
            found.add(p.name)
    return found


def extract_dirs(text: str) -> set[str]:
    """从一条命令文本中提取出现过的核心目录/ run.py token（含 .in 免误匹配）。"""
    found: set[str] = set()
    for tok in GATE_DIRS | {"run.py"}:
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


def _verify_all_dirs() -> tuple[set[str], set[str]]:
    """从 verify_all.py 读取 RUFF_DIRS / COMPILE_DIRS 字面量（去掉尾部 /）。"""
    src = VERIFY_ALL.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("RUFF_DIRS", "COMPILE_DIRS") and isinstance(node.value, ast.Tuple):
                vals = set()
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        vals.add(elt.value.rstrip("/"))
                out[name] = vals
    return out.get("RUFF_DIRS", set()), out.get("COMPILE_DIRS", set())


def _coveragerc_source() -> set[str]:
    m = re.search(r"^source\s*=\s*(.+)$", COVERAGERC.read_text(encoding="utf-8"), re.M)
    if not m:
        return set()
    return {x.strip() for x in m.group(1).split(",") if x.strip()}


def _coverage_gate_modules() -> set[str]:
    src = COVERAGE_GATE.read_text(encoding="utf-8")
    m = re.search(r"THRESHOLDS[^=]*=\s*\{(.+?)\n\}", src, re.S)
    if not m:
        return set()
    return set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', m.group(1)))


def check() -> list[str]:
    issues: list[str] = []
    if not REDEV.exists():
        issues.append(f"缺少 {REDEV.name}")
        return issues

    missing_gate = discover_packages() - GATE_DIRS
    if missing_gate:
        issues.append(f"新增顶层包未纳入门禁契约 CORE_DIRS: {sorted(missing_gate)}")

    ci = CI.read_text(encoding="utf-8")
    ci_compile = _capture(ci, "python -m compileall -q")
    ci_ruff = _capture(ci, "python -m ruff check")
    redev_compile = extract_dirs(REDEV.read_text(encoding="utf-8"))
    va_ruff, va_compile = _verify_all_dirs()

    req_compile = CORE_DIRS | {"tests", "run.py"}
    req_ruff = CORE_DIRS | {"tests"}
    req_redev = CORE_DIRS | {"run.py"}

    miss = req_compile - ci_compile
    if miss:
        issues.append(f"ci.yml compileall 缺目录: {sorted(miss)}")
    miss = req_ruff - ci_ruff
    if miss:
        issues.append(f"ci.yml ruff 缺目录: {sorted(miss)}")
    miss = req_redev - redev_compile
    if miss:
        issues.append(f"REDEV_GUIDE §4 编译链缺目录: {sorted(miss)}")

    miss = (CORE_DIRS | {"tests"}) - va_ruff
    if miss:
        issues.append(f"verify_all.py RUFF_DIRS 缺目录: {sorted(miss)}")
    miss = (CORE_DIRS | {"tests", "run.py"}) - va_compile
    if miss:
        issues.append(f"verify_all.py COMPILE_DIRS 缺目录: {sorted(miss)}")

    miss = COVERED_DIRS - _coveragerc_source()
    if miss:
        issues.append(f".coveragerc source 缺模块: {sorted(miss)}")
    miss = COVERED_DIRS - _coverage_gate_modules()
    if miss:
        issues.append(f"coverage_gate.py 模块级门槛缺模块: {sorted(miss)}")

    # F1/F2 编码门禁（check_encoding.py）必须同时在 ci.yml 与 verify_all.py 注册，
    # 否则会像 net/ 一样"脚本写了但没接进任何验证链"（A5 教训的延伸）。
    if "check_encoding.py" not in ci:
        issues.append("ci.yml 未注册 check_encoding.py（F1/F2 编码红线门禁）")
    va_src = VERIFY_ALL.read_text(encoding="utf-8")
    if "check_encoding.py" not in va_src:
        issues.append("verify_all.py 未注册 check_encoding.py（F1/F2 编码红线门禁）")

    # grep 型红线门禁（check_redlines.py：B8/E2/E7）同样必须注册在 ci.yml 与 verify_all。
    if "check_redlines.py" not in ci:
        issues.append("ci.yml 未注册 check_redlines.py（B8/E2/E7 grep 型红线门禁）")
    if "check_redlines.py" not in va_src:
        issues.append("verify_all.py 未注册 check_redlines.py（B8/E2/E7 grep 型红线门禁）")

    # 性能基线门禁（perf_baseline.py）同样必须注册（否则又是"脚本写了没接链"）。
    if "perf_baseline.py" not in ci:
        issues.append("ci.yml 未注册 perf_baseline.py（性能基线门禁）")
    if "perf_baseline.py" not in va_src:
        issues.append("verify_all.py 未注册 perf_baseline.py（性能基线门禁）")

    # 前端构建产物脚本（第十六轮）：必须至少注册进 verify_all / ci.yml / tests 之一，
    # 否则属于"脚本写了没接链"（A5 教训延伸：构建脚本脱链=产物无法在门禁中重建/校验）。
    _frontend_build_scripts = {
        "precompress_static.py": "预压缩 .gz（verify_all 已注册 --check）",
        "build_tailwind_css.py": "tailwind 静态化（tests/test_tailwind_staticization.py 契约覆盖）",
        "build_echarts_custom.py": "echarts 按需打包（SRI 由 build_frontend --require-manifest 校验）",
    }
    tests_src = " ".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in (ROOT / "tests").glob("test_*.py")
                         if p.is_file())
    for script, why in _frontend_build_scripts.items():
        registered = script in ci or script in va_src or script in tests_src
        if not registered:
            issues.append(f"{script} 未注册进 ci.yml / verify_all / tests 任一（{why}）")
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
    print("[docsync] 通过：门禁契约（ci.yml / verify_all / .coveragerc / coverage_gate / REDEV §4）一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
