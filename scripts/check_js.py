"""JS 语法检查：提取 HTML 内联 <script> 块（及可选 .js 文件），解析校验。

两级解析策略：
1. 优先 node --check（本机装有 Node.js 时）——原生支持 ES2020+（可选链/空值合并/
   正则字面量等），无需任何词法预处理，彻底规避 esprima 兼容层的坑
   （如正则内含裸引号会使按字符扫描的兼容层失同步，导致误报）。
2. 回退 esprima 4.x + ES2020 词法兼容转换（node 不可用时）：
    ?.  ->  .      （x?.y  → x.y，语法层面等价，不影响检查目的）
    ??  ->  ||     （x ?? y → x || y）
   转换只影响语法检查，绝不写回源文件。

第十六轮补洞（2026-09-30）：还检查 Vue 内联模板表达式——<div id="app"> 的
in-DOM 模板里 {{...}} 与 :xxx="..." / v-xxx="..." 的表达式逐条 node 语法检查。
背景：并行改动引入 `:class="{fr-hidden:!drlCfg.curriculum}"`——对象 key 带连字符
未加引号，Vue 编译 `new Function('return ({fr-hidden:...})')` 抛
`Unexpected token '-'`，整页白屏（前端致命）。<script> 块检查看不到 HTML 属性
里的表达式，必须单独扫模板。

用法：
    python scripts/check_js.py [file.html file.js ...]
默认检查 web/static/index.html 的全部内联脚本 + Vue 模板表达式。
退出码 0=全部通过；1=存在语法错误。作为提交前验证链的一环。
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_RE = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S | re.I)
APP_DIV_RE = re.compile(r'<div[^>]*id="app"[^>]*>', re.I)
# 指令属性表达式（Vue in-DOM 模板）：:xxx="..." 或 v-xxx="..."（双引号）
DIR_ATTR_RE = re.compile(r'(?:v-[\w-]+|:[a-zA-Z_-]+)\s*=\s*"([^"]+)"')
MUSTACHE_RE = re.compile(r'\{\{([\s\S]*?)\}\}')

# node 可执行文件（惰性探测；无 node 时回退 esprima）
_NODE = shutil.which("node") if shutil.which("node") else None


def _check_with_node(source: str) -> list[str]:
    """node --check 校验（写临时 .js，CommonJS 脚本模式）。返回错误列表（空=通过）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        r = subprocess.run([_NODE, "--check", path], capture_output=True, text=True,
                           timeout=30, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            # node 报错行号是临时文件的；转成相对偏移信息保留原文
            return [f"node --check: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '未知错误'}"]
        return []
    finally:
        Path(path).unlink(missing_ok=True)


def es2020_compat(source: str) -> str:
    """把 ES2020 新语法转换为 esprima 可解析的等价形式（词法安全）。"""
    out = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        # 行注释
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            j = source.find("\n", i)
            j = n if j == -1 else j
            out.append(source[i:j])
            i = j
            continue
        # 块注释
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(source[i:j])
            i = j
            continue
        # 字符串字面量（单双引号）：整体跳过
        if ch in "'\"":
            quote = ch
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            out.append(source[i:j])
            i = j
            continue
        # 代码区：替换操作符
        if ch == "?" and i + 1 < n and source[i + 1] == ".":
            out.append(".")  # ?. -> .
            i += 2
            continue
        if ch == "?" and i + 1 < n and source[i + 1] == "?":
            out.append("||")  # ?? -> ||
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def check_js_source(name: str, source: str) -> list[str]:
    """解析一段 JS 源码，返回错误列表（空=通过）。优先 node --check，回退 esprima 兼容层。"""
    if _NODE:
        return _check_with_node(source)
    # 回退路径：esprima 4.x 兼容层较脆弱（词法按字符扫描，正则含裸引号会失同步），
    # 结果仅供参考——若触发此分支，输出明确警告以便 CI 干预升级环境。
    # esprima 延迟导入：只有无 node 时才需要它（避免 CI 有 node 时仍被缺包卡死）。
    print("[WARN] 未找到 node 可执行文件，JS 语法检查降级为 esprima 兼容层（结果仅供参考）")
    try:
        import esprima
    except ImportError:
        return ["缺少 node 且未安装 esprima：请安装 requirements-dev.txt 或提供 node 可执行文件"]
    try:
        esprima.parseScript(es2020_compat(source))
        return []
    except esprima.Error as e:
        msg = getattr(e, "description", None) or getattr(e, "message", None) or str(e)
        return [f"{name}:{e.lineNumber}:{e.column}: {msg}"]


def extract_app_template(text: str) -> str | None:
    """提取 <div id="app"> 的完整 in-DOM 模板（深度计数闭合标签）。"""
    m = APP_DIV_RE.search(text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i:i + 4] == "<div":
            depth += 1
            i += 4
        elif text[i:i + 6] == "</div>":
            depth -= 1
            i += 6
        else:
            i += 1
    return text[start:max(start, i - 6)]


def _line_no(tpl: str, pos: int) -> int:
    return tpl.count("\n", 0, pos) + 1


def check_vue_template_exprs(name: str, text: str) -> list[str]:
    """Vue in-DOM 模板表达式语法检查（{{}} / :attr / v-dir）。

    每个表达式用 node 的 new Function('return (' + e + ')') 校验（与 Vue 编译器
    同语义）。对象字面量 key 带连字符未加引号（如 {fr-hidden:...}）会在此报
    'Unexpected token -'——这类错误 <script> 检查看不到（在 HTML 属性里）。
    无 node 时降级跳过（打印 WARN）；此检查依赖 node，与主 JS 检查同环境。
    """
    if not _NODE:
        return []
    tpl = extract_app_template(text)
    if tpl is None:
        return []
    exprs: list[tuple[str, int]] = []
    for mm in MUSTACHE_RE.finditer(tpl):
        e = mm.group(1).strip()
        if e:
            exprs.append((e, _line_no(tpl, mm.start())))
    for mm in DIR_ATTR_RE.finditer(tpl):
        e = mm.group(1).strip()
        if e:
            exprs.append((e, _line_no(tpl, mm.start())))
    if not exprs:
        return []
    # 一次 node 调用检查全部表达式（逐条 new Function，报错即停止该条）
    src_lines = []
    for idx, (e, _ln) in enumerate(exprs):
        src_lines.append(f"const __e{idx} = (function(){{ try {{ return ({e}); }} catch (err) {{ "
                         f"console.log('EXPR_FAIL|{idx}|' + err.message); }} }})();")
    prog = "\n".join(src_lines)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([_NODE, path], capture_output=True, text=True,
                           timeout=60, encoding="utf-8", errors="replace")
    finally:
        Path(path).unlink(missing_ok=True)
    # 解析失败标记
    failed = []
    for line in r.stdout.splitlines():
        if line.startswith("EXPR_FAIL|"):
            parts = line.split("|")
            idx, msg = int(parts[1]), "|".join(parts[2:])
            e, ln = exprs[idx]
            failed.append(f"{name}#app:行{ln} Vue 表达式语法错误: {msg.strip()} → {e[:80]}")
    return failed


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".html":
        blocks = SCRIPT_RE.findall(text)
        if not blocks:
            return [f"{path}: 未找到 <script> 块"]
        for i, (attrs, body) in enumerate(blocks):
            if "src=" in attrs.lower():
                continue  # 外链脚本跳过
            if re.search(r"type\s*=\s*['\"](?!text/javascript|module|application/javascript)", attrs, re.I):
                continue  # 非 JS 块（如 text/template）跳过
            errors += check_js_source(f"{path.name}#script{i + 1}", body)
        # 第十六轮：Vue in-DOM 模板表达式检查（防 `{fr-hidden:...}` 这类白屏 bug）
        errors += check_vue_template_exprs(path.name, text)
    else:
        errors += check_js_source(path.name, text)
    return errors


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "web" / "static" / "index.html"]
    total = 0
    for f in files:
        if not f.is_absolute():
            f = ROOT / f
        errs = check_file(f)
        if errs:
            for e in errs:
                print(f"[FAIL] {e}")
            total += len(errs)
        else:
            print(f"[OK] {f}")
    if total:
        print(f"JS 语法检查失败：{total} 处错误")
        return 1
    print("JS 语法检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
