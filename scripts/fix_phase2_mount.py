#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补救 patch_ui_phase2.py 的 ins() 缺陷：纯插入分支把锚点行本身吃掉了。
本脚本恢复被误删的两行（.mac-form-err 已由 Edit 恢复，这里补 app.mount）。
可重复执行：锚点已被修复时安全跳过。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "web" / "static" / "index.html"
J = "\r\n"

text = TARGET.read_bytes().decode("utf-8")
if "app.mount(" in text:
    print("SKIP mount already present")
    sys.exit(0)

old = J.join(["});", "", "</script>"])
new = J.join(["});", "app.mount('#app');", "</script>"])
assert text.count(old) == 1, "tail anchor count = %d" % text.count(old)
text = text.replace(old, new)
TARGET.write_bytes(text.encode("utf-8"))
print("OK mount restored, lines %d" % len(text.split(J)))
