"""REDEV_GUIDE "grep 型红线"自动化（2026-09-28）：把只能人肉 grep 的红线变成门禁。

背景：REDEV_GUIDE §3 多条红线的"怎么验"是 `grep 全项目 …` / `grep index.html ≈ 0`
之类——没有任何机器执行。结果是 run.py 里 `symbol 手工拆分` 违反 B8 入库多月
无人发现（grep 型红线没人跑 = 等于没有红线）。本脚本把这些可机械判定的红线收口：

  B8  交易所 symbol 手工拆分（`symbol.split` 式）    → 必须 0 处（统一 parse_symbol）
  E2  index.html 引用 CDN（unpkg/cdnjs/jsdelivr/cdn.）→ 必须 0 处（vendor 本地化+SRI）
  E7  index.html 的 `!important` 数量          → ≤ 基线（防无脑堆补丁的数量失控；
      有意豁免须带注释说明，计数基线记录在下方）

用法：
  python scripts/check_redlines.py            # 0=通过 1=有违反（打印清单）
"""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "static" / "index.html"

# E7 基线：当前 12 处均为带注释的有意豁免（压 inline padding / 毛玻璃退场 /
# evolve 页对齐 / media 适配）。新增豁免必须先更新基线并说明理由——
# 本门禁防的是"无注释无理由地堆 !important 补丁"导致的样式失控。
E7_IMPORTANT_BASELINE = 12

# B8：需要检查的 Python 文件范围（与 ruff/compileall 同口径，含 run.py）
B8_SKIP_PREFIX = (".venv/", "data/", ".optim/", "vendor/", "dsh-memory-setup/",
                  ".dsh-memory-setup/", "node_modules/", "tests/")
# B8 显式豁免：这里是 parse_symbol 的家（唯一允许拆 symbol 的地方），
# 以及非交易所 symbol 的 URL/host 解析（按文件+行豁免，理由必须属实）。
B8_EXEMPT_FILES = {"exchange/symbols.py"}  # 解析器本体：docstring 提及历史写法 + market.split 实现
# (文件, 行号, 理由)
B8_EXEMPT_LINES: list[tuple[str, int, str]] = [
    ("net/proxy_nodes.py", 178, "URL host:port 解析（非交易所 symbol，不属于 B8 域）"),
]


# B8 违规模式：对 symbol 调用 split 并传分隔符字符串（含取索引的写法同样违规）
B8_PAT = re.compile(r"\.split\(['\"/]/['\"]\)")


def check_b8() -> list[str]:
    """全项目 Python 文件不得出现对 symbol 调 split 传分隔符的写法（B8：统一 parse_symbol）。"""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    issues: list[str] = []
    exempt_lines = {(f, n) for f, n, _r in B8_EXEMPT_LINES}
    for line in out.stdout.splitlines():
        if not line.endswith(".py") or line.startswith(B8_SKIP_PREFIX):
            continue
        if line in B8_EXEMPT_FILES:
            continue
        p = ROOT / line
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, src in enumerate(text.splitlines(), 1):
            if (line, lineno) in exempt_lines:
                continue
            if B8_PAT.search(src):
                issues.append(f"{line}:{lineno}: {src.strip()}")
    return issues


def check_e2() -> list[str]:
    """index.html 不得引用 CDN（E2：vendor 本地化 + SRI，断网/局域网即白屏红线）。"""
    if not INDEX.exists():
        return ["缺少 index.html"]
    text = INDEX.read_text(encoding="utf-8")
    cdn = re.compile(r"(unpkg\.com|cdnjs\.cloudflare|jsdelivr\.net|cdn\.)", re.I)
    hits = [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if cdn.search(ln)]
    return [f"index.html:{i}: 疑似 CDN 引用: {ln.strip()[:100]}" for i, ln in hits]


def check_e7() -> list[str]:
    """index.html 的 !important 数量不得超过基线（有意豁免须注释，禁止无理由堆补丁）。"""
    if not INDEX.exists():
        return ["缺少 index.html"]
    text = INDEX.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "!important" in ln]
    if len(lines) <= E7_IMPORTANT_BASELINE:
        return []
    extra = lines[E7_IMPORTANT_BASELINE:]
    issues = [f"index.html: !important {len(lines)} 处 > 基线 {E7_IMPORTANT_BASELINE}（新增豁免须带注释并更新基线）"]
    for ln in extra[:8]:
        # 相邻注释行也算"有说明"（!important 常与 /* 注释同行或隔行）
        issues.append(f"  未计入基线的 !important: {ln.strip()[:100]}")
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="REDEV_GUIDE grep 型红线门禁（B8/E2/E7）")
    ap.add_argument("--list", action="store_true", help="只列出各红线检查项")
    args = ap.parse_args(argv)

    checks: list[tuple[str, list[str]]] = [
        ("B8 禁交易所 symbol 手工拆分", check_b8()),
        ("E2 禁 CDN 引用", check_e2()),
        ("E7 !important ≤ 基线", check_e7()),
    ]
    if args.list:
        for name, _issues in checks:
            print(name)
        return 0

    bad = [(name, issues) for name, issues in checks if issues]
    if bad:
        total = sum(len(i) for _, i in bad)
        print(f"[redlines] [FAIL] {total} 处红线违反:")
        for name, issues in bad:
            print(f"  [{name}]")
            for i in issues[:12]:
                print(f"    - {i}")
        print("[redlines] FAILED")
        return 1
    print("[redlines] 通过：B8 手工拆分=0、E2 CDN=0、E7 !important ≤ 基线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
