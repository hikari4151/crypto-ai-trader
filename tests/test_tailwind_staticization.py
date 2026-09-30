"""第十四轮 tailwind 静态化契约测试。

守卫：web/static/vendor/tailwind.static.css（构建期预生成）必须覆盖
index.html 中全部可枚举的 tailwind 工具类——防止有人改了 index.html 的 class
却忘了重建静态 CSS（那样运行时样式静默缺失）。

判定逻辑（与 scripts/build_tailwind_css.py 的契约校验同源）：
1. 扫描 index.html：静态 class="..." + 动态 :class 字面量 + on-{amber,red,green} 展开
2. 过滤出「真 tailwind 工具类」：非内联 <style> 自定义类、非 JS 噪音
3. 断言其 CSS 选择器（转义后）全部存在于静态 CSS 中
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "static" / "index.html"
CSS = ROOT / "web" / "static" / "vendor" / "tailwind.static.css"

# 与 build 脚本一致的 tailwind 工具类模式
TW_PAT = re.compile(
    r"^(?:text|bg|border|p[xy]?|m[xy]?|w-|h-|flex|grid|gap|space|items|justify|self|content|"
    r"opacity|rounded|shadow|font|leading|tracking|max-w|min-w|min-h|max-h|overflow|z-|top-|bottom-|"
    r"left-|right-|inset|duration|ease|transition|transform|scale|rotate|translate|cursor|pointer-events|"
    r"select|blur|backdrop|divide|ring|outline|fill|stroke|object|aspect|animate|float|clear|table|align|"
    r"whitespace|break|truncate|underline|italic|uppercase|lowercase|capitalize|list|placeholder|accent|"
    r"caret|columns|col-|row-|order|sr-|hidden|block|inline|absolute|fixed|relative|sticky|visible|invisible|"
    r"antialiased|subpixel)\b|(?:hover|focus|active|disabled|sm|md|lg|xl|2xl|motion-safe|motion-reduce|dark)\s*:")

_GLOW_VALUES = ("amber", "red", "green")


def extract_classes(text: str) -> set[str]:
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
    for g in _GLOW_VALUES:
        dyn.add("on-" + g)
    dyn.update(["glow-g", "glow-r"])
    return static | dyn


def css_class_set(css: str) -> set[str]:
    """静态 CSS 里出现的类（含变体，剥伪类后缀）。"""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: set[str] = set()
    for m in re.finditer(r"[^{}]*\{", css):
        sel = m.group(0)[:-1]
        for cm in re.finditer(r"\\?\.[a-zA-Z0-9_\\:\[\]\/\.\-]+", sel):
            tok = cm.group(0)
            tok = re.split(r"(?<!\\):", tok)[0]
            if tok:
                out.add(tok)
    return out


def css_escape_class(cls: str) -> str:
    return (cls.replace(":", "\\:").replace("/", "\\/")
            .replace("[", "\\[").replace("]", "\\]").replace(".", "\\."))


def test_static_css_covers_all_tailwind_utilities():
    text = INDEX.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    inline = " ".join(m.group("body") for m in re.finditer(r"<style[^>]*>(?P<body>.*?)</style>", text, re.S))
    inl_sel = css_class_set(inline)
    css_sels = css_class_set(css)

    all_cls = extract_classes(text)
    missing = []
    for cls in sorted(all_cls):
        if "." in cls or not TW_PAT.search(cls):
            continue  # 自定义类 / JS 噪音：不要求 tailwind 生成
        esc = css_escape_class(cls)
        if ".%s" % esc in inl_sel:
            continue  # 内联 style 自定义类
        if ".%s" % esc not in css_sels:
            missing.append(cls)
    assert not missing, (
        "以下 tailwind 工具类未被静态 CSS 覆盖（改了 index.html 的 class 需重建 "
        "scripts/build_tailwind_css.py）：\n" + "\n".join(sorted(missing))
    )


def test_static_css_exists_and_reasonable_size():
    css = CSS.read_text(encoding="utf-8")
    assert len(css) > 2000, "静态 CSS 不应为空"
    assert len(css) < 200_000, "静态 CSS 不应比 Play CDN 运行时还大"


def test_index_html_uses_static_css_not_runtime():
    text = INDEX.read_text(encoding="utf-8")
    assert "tailwind.static.css" in text, "index.html 应引用静态 CSS"
    assert "vendor/tailwind.js" not in text, "Play CDN 运行时 script 应已移除"
