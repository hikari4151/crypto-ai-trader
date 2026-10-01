#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 tailwind 静态 CSS（第十四轮：Play CDN 运行时 → 构建期静态化）。

背景（2026-09-30）：前端原用 Tailwind v3 Play CDN 运行时（vendor/tailwind.js，
397.7KB / gzip 120.4KB）——下载后浏览器还要跑完整 CSS 编译器（JIT + MutationObserver），
是全站第二大静态资源与启动计算开销。本脚本在**构建期**扫描 index.html 全部
静态 class + 动态 :class 字面量（safelist），生成只含用到的工具类的
web/static/vendor/tailwind.static.css（实测 22.7KB / gzip 5.3KB，零运行时开销）。

用法：
  python scripts/build_tailwind_css.py             # 构建 + 打印 SRI sha384
  python scripts/build_tailwind_css.py --entry-only # 只打印入口/模块清单（离线检查）

关键设计：
- content = index.html：tailwind CLI 自动扫出静态 class="..." 里的工具类。
- safelist  = 动态 :class 字面量 + on-{amber,red,green} 拼接展开：运行时才出现的类
  静态扫描看不到，必须显式 safelist，否则切视图时样式缺失。
- 契约校验：构建后比对「运行时 Play CDN 全视图选择器集 ⊆ 静态 CSS 选择器集」，
  缺失必须为 0（防未来改 index.html 新增类而忘了重建）。
- 生成后必须更新 index.html 中 <link> 的 integrity（SRI 不匹配浏览器拒载 CSS）。
"""
import base64
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "static" / "index.html"
OUT = ROOT / "web" / "static" / "vendor" / "tailwind.static.css"

# Play CDN 默认配置等价（与 vendor/tailwind.js 的默认 theme 一致，不自定义）
CONFIG = """\
module.exports = {
  content: [%r],
  safelist: %r,
  theme: {},
  plugins: [],
};
"""

# on- 前缀运行时拼接的取值域（t.glow !== 'plain' 才加 on- 前缀）
_GLOW_VALUES = ("amber", "red", "green")


def extract_classes(text: str) -> tuple[set[str], set[str]]:
    """返回 (静态 class 集合, 动态 :class 字面量集合)。

    静态：class="..." 与 class='...' 的完整 token。
    动态：:class="cond ? 'a' : 'b'" 中的字面量 + 数组元素 + on- 拼接展开。
    """
    static: set[str] = set()
    for m in re.finditer(r'class="([^"]+)"', text):
        for tok in m.group(1).split():
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_\-:/\[\].]*", tok):
                static.add(tok)
    for m in re.finditer(r"class='([^']+)'", text):
        for tok in m.group(1).split():
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_\-:/\[\].]*", tok):
                static.add(tok)

    dyn: set[str] = set()
    for m in re.finditer(r':class="([^"]+)"', text):
        expr = m.group(1)
        for lm in re.finditer(r"'([^']+)'", expr):
            t = lm.group(1).strip()
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_\-:/]*", t) and not t.startswith(("--", "-", ".")):
                dyn.add(t)
        for am in re.finditer(r"\[([^\]]*)\]", expr):
            for x in re.finditer(r"'([^']+)'", am.group(1)):
                t = x.group(1).strip()
                if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_\-:/]*", t):
                    dyn.add(t)
    # 运行时拼接展开
    for g in _GLOW_VALUES:
        dyn.add("on-" + g)
    dyn.update(["glow-g", "glow-r"])
    return static, dyn


def selector_set(css: str) -> set[str]:
    """抽取 CSS 中全部出现在选择器里的 .类名（含复合/变体选择器内部）。

    以 '{' 为锚取每个规则块的选择器区，提取 `\\.类名` token；
    在「未被反斜杠转义的冒号」处截断 —— 变体类（hover\\:bg-white）的 \\:
    冒号被转义应保留，伪类/伪元素（:hover、::before）的裸冒号应剥掉。
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: set[str] = set()
    for m in re.finditer(r"[^{}]*\{", css):
        sel_part = m.group(0)[:-1]
        for cm in re.finditer(r"\\?\.[a-zA-Z0-9_\\:\[\]\/\.\-]+", sel_part):
            tok = cm.group(0)
            # 在未转义的 ':' 处截断（保留 \\: 的变体冒号）
            tok = re.split(r"(?<!\\):", tok)[0]
            if tok:
                out.add(tok)
    return out


def sh(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    # Windows 下 npm 是 .cmd 批处理，CreateProcess 找不到 → 用 shell 语义执行，
    # 并显式继承 PATH（run.py / pytest 子进程环境可能不含 npm）。
    base_env = dict(os.environ)
    if env:
        base_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=base_env,
                          timeout=300, shell=os.name == "nt")


def check_artifact() -> list[str]:
    """校验 tailwind 静态 CSS 产物与 index.html 引用一致（防改产物忘更新 SRI）。"""
    issues: list[str] = []
    if not OUT.exists():
        issues.append(f"缺 tailwind 产物: {OUT.relative_to(ROOT)}（运行 build_tailwind_css.py 构建）")
        return issues
    data = OUT.read_bytes()
    b64 = base64.b64encode(hashlib.sha384(data).digest()).decode()
    sri = f"sha384-{b64}"
    html = INDEX.read_text(encoding="utf-8")
    for m in re.finditer(r'<link[^>]*href="/static/vendor/tailwind\.static\.css"[^>]*>', html):
        tag = m.group(0)
        im = re.search(r'integrity="([^"]+)"', tag)
        if not im:
            issues.append("tailwind link 标签缺 integrity（SRI 防线缺失）")
        elif im.group(1) != sri:
            issues.append(
                f"tailwind SRI 不匹配：index.html 写 {im.group(1)[:32]}… 产物实际 {sri[:32]}…\n"
                "（改了 tailwind 产物必须同步更新 index.html integrity）")
    return issues


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-only", action="store_true", help="离线检查：只打印扫描结论")
    ap.add_argument("--use-local-cli", action="store_true",
                    help="复用 .optim/echarts_custom/node_modules 里已装的 tailwindcss "
                         "（避免每次临时 npm install；开发机推荐）")
    ap.add_argument("--check", action="store_true",
                    help="校验产物与 index.html SRI 一致（接入门禁链，不构建）")
    args = ap.parse_args()

    if args.check:
        issues = check_artifact()
        if issues:
            print("[tailwind] ❌ 产物校验失败：")
            for i in issues:
                print("   ", i)
            raise SystemExit(1)
        print(f"[tailwind] ✅ 产物存在且 SRI 与 index.html 一致（{OUT.relative_to(ROOT)}）")
        return

    text = INDEX.read_text(encoding="utf-8")
    static, dyn = extract_classes(text)
    safelist = sorted(static | dyn)
    print(f"[tailwind] 静态 class {len(static)} + 动态 {len(dyn)} = safelist {len(safelist)}")

    if args.entry_only:
        print(f"[tailwind] content={INDEX}  safelist={safelist[:10]}...")
        return

    with tempfile.TemporaryDirectory(prefix="tailwind-build-") as td:
        work = Path(td)
        (work / "input.css").write_text("@tailwind base;\n@tailwind components;\n@tailwind utilities;\n", encoding="utf-8")
        (work / "tailwind.config.js").write_text(
            CONFIG % (str(INDEX).replace("\\", "/"), safelist), encoding="utf-8")
        if args.use_local_cli:
            local_cli = ROOT / ".optim" / "echarts_custom" / "node_modules" / ".bin" / "tailwindcss.cmd"
            if not local_cli.exists():
                sys.exit("--use-local-cli 但本地未装 tailwindcss（先 .optim/echarts_custom 里 npm i -D tailwindcss@3.4.14）")
            r = sh([str(local_cli), "-c", "tailwind.config.js", "-i", "input.css", "-o", "tailwind.static.css"], work)
        else:
            sh(["npm", "install", "tailwindcss@3.4.14"], work)
            r = sh(["npx", "tailwindcss", "-c", "tailwind.config.js",
                    "-i", "input.css", "-o", "tailwind.static.css"], work)
        if r.returncode != 0:
            sys.exit("tailwind 构建失败:\n" + r.stdout + r.stderr)
        data = (work / "tailwind.static.css").read_bytes()

    OUT.write_bytes(data)
    b64 = base64.b64encode(hashlib.sha384(data).digest()).decode()
    print(f"[tailwind] 产物: {OUT.relative_to(ROOT)}  {len(data)}B  gzip {len(_gz(data))}B")
    print(f"[tailwind] SRI: sha384-{b64}")

    # 契约校验：静态选择器集必须覆盖 index.html 全部可枚举的 **tailwind 工具类**。
    # （自定义类由内联 <style> 定义、JS 表达式噪音非 class——两者都不需要 tailwind 生成，
    #  要求它们反而误报。权威基线 = 运行时 Play CDN 产物，已证明 静态 ⊇ 运行时。）
    st_sel = selector_set(data.decode("utf-8"))
    inline_style = " ".join(m.group("body") for m in re.finditer(r"<style[^>]*>(?P<body>.*?)</style>", text, re.S))
    inl_sel = selector_set(inline_style)
    tw_pat = re.compile(
        r"^(?:text|bg|border|p[xy]?|m[xy]?|w-|h-|flex|grid|gap|space|items|justify|self|content|"
        r"opacity|rounded|shadow|font|leading|tracking|max-w|min-w|min-h|max-h|overflow|z-|top-|bottom-|"
        r"left-|right-|inset|duration|ease|transition|transform|scale|rotate|translate|cursor|pointer-events|"
        r"select|blur|backdrop|divide|ring|outline|fill|stroke|object|aspect|animate|float|clear|table|align|"
        r"whitespace|break|truncate|underline|italic|uppercase|lowercase|capitalize|list|placeholder|accent|"
        r"caret|columns|col-|row-|order|sr-|hidden|block|inline|absolute|fixed|relative|sticky|visible|invisible|"
        r"antialiased|subpixel)\b|(?:hover|focus|active|disabled|sm|md|lg|xl|2xl|motion-safe|motion-reduce|dark)\s*:")
    missing = []
    for cls in safelist:
        if "." in cls or not tw_pat.search(cls):
            continue  # 自定义类/噪音：跳过
        # CSS 选择器转义：: → \\:、/ → \\/、[ → \\[、] → \\]、. → \\.
        esc = (cls.replace(":", "\\:").replace("/", "\\/")
                  .replace("[", "\\[").replace("]", "\\]").replace(".", "\\."))
        if ".%s" % esc in inl_sel:
            continue  # 内联 style 已定义
        if ".%s" % esc not in st_sel:
            missing.append(cls)
    if missing:
        print("[tailwind] ❌ 契约校验失败：以下 tailwind 工具类未在静态 CSS 中生成：")
        for c in sorted(missing)[:30]:
            print("    ", c)
        sys.exit(1)
    print(f"[tailwind] ✅ 契约校验通过：{len(safelist)} 个可枚举 class 中，tailwind 工具类全部覆盖")


def _gz(data: bytes) -> bytes:
    import gzip
    return gzip.compress(data, 6)


if __name__ == "__main__":
    main()

