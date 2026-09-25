"""C1 模块级覆盖率门禁：按顶层模块设置最低线，避免全量均值掩盖盲区。

输入：coverage.json（`python -m coverage json` 产物）。
聚合：按每个文件的顶层包（core/engine/ai/web/backtest/strategies/indicators/config/drl/exchange/factors）
从 coverage.json 的 summary.percent_covered 汇总为模块级覆盖率。

用法：
  python scripts/coverage_gate.py --report                         # 只打印各模块覆盖率
  python scripts/coverage_gate.py --json coverage.json             # 执行门禁（默认）
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 模块级最低覆盖率（%），按 fast（-m "not slow"）与 full 实测取 min 后留余量校准（C1）。
# 实测（2026-09）：fast vs full → core79.9/80.0、engine80.1/80.1、exchange70.5/70.5、
# backtest78.3/82.0、strategies58.1/81.2、drl62.6/73.8、ai59.3/59.0、web38.1/42.6、
# factors60.4/60.5、config86.2/48.6、indicators92.4/92.4。
THRESHOLDS: dict[str, float] = {
    "core": 75.0,
    "engine": 75.0,
    "exchange": 64.0,
    "backtest": 70.0,
    "strategies": 50.0,
    "drl": 55.0,
    "ai": 50.0,
    "web": 32.0,
    "factors": 50.0,
    "indicators": 85.0,
    "config": 42.0,
}
MODULES = tuple(THRESHOLDS)


def parse_module_coverage(text: str) -> dict[str, dict]:
    """coverage.json -> {path: {summary: {...}}}。"""
    data = json.loads(text)
    return data.get("files") or {}


def group_by(p: str) -> str:
    norm = p.replace("\\", "/")
    head = norm.split("/", 1)[0]
    return head if head in MODULES else "_other"


def aggregate(files: dict[str, dict]) -> dict[str, float]:
    """按模块（顶层目录）聚合 percent_covered（按语句数加权）。"""
    per: dict[str, list[tuple[float, float]]] = {}
    for path, raw in files.items():
        summary = raw.get("summary") or {}
        pc = float(summary.get("percent_covered", 0.0))
        n = float(summary.get("num_statements", 0))
        mod = group_by(path)
        per.setdefault(mod, []).append((pc, n))
    out: dict[str, float] = {}
    for mod, items in per.items():
        total = sum(n for _, n in items)
        out[mod] = (sum(pc * n for pc, n in items) / total) if total else 0.0
    return out


def check(agg: dict[str, float]) -> list[str]:
    issues = []
    for mod, low in THRESHOLDS.items():
        val = agg.get(mod)
        if val is None:
            issues.append(f"{mod}: 无数据（模块未纳入 coverage.json）")
        elif val < low:
            issues.append(f"{mod}: {val:.1f}% < 门槛 {low:.0f}%")
    return issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模块级覆盖率门禁（C1）")
    ap.add_argument("--json", default=None, help="coverage.json 路径")
    ap.add_argument("--report", action="store_true", help="只打印不执行门禁")
    args = ap.parse_args(argv)
    path = Path(args.json) if args.json else ROOT / "coverage.json"
    if not path.exists():
        print(f"[covgate] 缺少 {path}（先运行: python -m pytest --cov && python -m coverage json）")
        return 1
    files = parse_module_coverage(path.read_text(encoding="utf-8"))
    agg = aggregate(files)
    if args.report:
        print("[coverage] 模块级覆盖率:")
        for mod in sorted(agg, key=lambda m: (MODULES.index(m) if m in MODULES else 99, m)):
            print(f"  {mod:14s} {agg[mod]:6.1f}%")
        return 0
    issues = check(agg)
    if issues:
        for i in issues:
            print(f"[coverage] [FAIL] {i}")
        print("[coverage] FAILED")
        return 1
    print("[coverage] 通过：所有核心模块达到最低覆盖率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
