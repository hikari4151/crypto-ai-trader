# -*- coding: utf-8 -*-
"""量化各视图的风格基座使用情况，定位"掉队页面"。

判据（画布深色玻璃体系的既成约定）：
  win   — .mac-window 玻璃容器（所有页面都应以此承载内容）
  card  — .mac-card / .stat-card 次级卡片
  head  — .page-head 统一页头（标题 + 操作区）
  裸底  — 直接写 inline background / border 的裸 div（脱离 token 体系）
  裸表  — 未包在 .mac-window 内的 <table>
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = ROOT / "web" / "static" / "index.html"

s = HTML.read_text(encoding="utf-8")
lines = s.split("\n")

starts = []
for i, l in enumerate(lines):
    m = re.search(r"""<section v-if="view==='([a-zA-Z0-9_]+)'""", l)
    if m:
        starts.append((i, m.group(1)))

bounds = []
for idx, (i, name) in enumerate(starts):
    end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
    # 末个视图止于自身的 </section>，否则会把尾部 <script> 一并算进来
    if idx + 1 == len(starts):
        for j in range(i, len(lines)):
            if re.match(r"^\s*</section>", lines[j]):
                end = j
                break
    bounds.append((name, i, end))

hdr = f"{'view':<16}{'lines':>7}{'win':>5}{'card':>6}{'head':>6}{'裸底':>6}{'裸表':>6}{'inline':>8}"
print(hdr)
print("-" * len(hdr))

for name, a, b in bounds:
    seg = "\n".join(lines[a:b])
    n = b - a
    win = len(re.findall(r"mac-window", seg))
    card = len(re.findall(r"mac-card|stat-card", seg))
    head = len(re.findall(r"page-head", seg))
    rawbg = len(re.findall(r'style="[^"]*background', seg))
    # 裸表：统计 table 标签数，减去被 mac-window 包裹的（近似：win>0 时视为已包裹）
    tbl = len(re.findall(r"<table", seg))
    rawtbl = 0 if win >= tbl and tbl > 0 else tbl
    inline = len(re.findall(r'style="', seg))
    print(f"{name:<16}{n:>7}{win:>5}{card:>6}{head:>6}{rawbg:>6}{rawtbl:>6}{inline:>8}")

print("\n总视图数:", len(bounds))
