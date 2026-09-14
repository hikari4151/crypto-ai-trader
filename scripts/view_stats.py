# -*- coding: utf-8 -*-
"""按视图切分 index.html 模板，打印每个视图的结构统计。
用法: python scripts/view_stats.py [view]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "static" / "index.html"

src = SRC.read_text(encoding="utf-8")

starts = [(m.start(), m.group(1))
          for m in re.finditer(r"""<section v-if="view==='([a-zA-Z0-9_]+)'" """, src)]

segs = {}
for idx, (pos, name) in enumerate(starts):
    end = starts[idx + 1][0] if idx + 1 < len(starts) else len(src)
    segs[name] = src[pos:end]

if len(sys.argv) > 1:
    names = [sys.argv[1]]
else:
    names = [n for _, n in starts]

for name in names:
    seg = segs.get(name)
    if seg is None:
        print(f"[MISSING] {name}")
        continue
    print("=" * 70)
    print(f"[{name}]  {len(seg.splitlines())} 行")
    for token in ("stat-card", "mac-window", "mac-card", "page-head",
                  "mac-form-section", "empty-state"):
        print(f"  {token:20s} {seg.count(token)}")
