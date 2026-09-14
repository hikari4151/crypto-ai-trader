# -*- coding: utf-8 -*-
"""抽取各视图的结构骨架（只保留布局意义的行），用于快速比对页面间的信息架构差异。

关注：页头 / 玻璃容器 / 卡片 / 表格 / 栅格 / 分组表单
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = ROOT / "web" / "static" / "index.html"

lines = HTML.read_text(encoding="utf-8").split("\n")

starts = []
for i, l in enumerate(lines):
    m = re.search(r"""<section v-if="view==='([a-zA-Z0-9_]+)'""", l)
    if m:
        starts.append((i, m.group(1)))

bounds = []
for idx, (i, name) in enumerate(starts):
    end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
    if idx + 1 == len(starts):
        for j in range(i, len(lines)):
            if re.match(r"^\s*</section>", lines[j]):
                end = j
                break
    bounds.append((name, i, end))

PAT = re.compile(
    r"page-head|class=\"mac-window|mac-title\"|stat-card|class=\"mac-card|<table"
    r"|grid-cols|mac-form-section|empty-state|<!--\s*=|-->"
)

only = sys.argv[1] if len(sys.argv) > 1 else None

for name, a, b in bounds:
    if only and name != only:
        continue
    print("=" * 70)
    print(f"[{name}]  行 {a+1}-{b}  ({b-a} 行)")
    print("=" * 70)
    for j in range(a, b):
        l = lines[j]
        if PAT.search(l):
            t = l.strip()
            t = re.sub(r"\s+", " ", t)
            print(f"  {j+1:>5} | {t[:150]}")
    print()
