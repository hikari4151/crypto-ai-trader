"""JS 语法检查：提取 HTML 内联 <script> 块（及可选 .js 文件），解析校验。

两级解析策略：
1. 优先 node --check（本机装有 Node.js 时）——原生支持 ES2020+（可选链/空值合并/
   正则字面量等），无需任何词法预处理，彻底规避 esprima 兼容层的坑
   （如正则内含裸引号会使按字符扫描的兼容层失同步，导致误报）。
2. 回退 esprima 4.x + ES2020 词法兼容转换（node 不可用时）：
    ?.  ->  .      （x?.y  → x.y，语法层面等价，不影响检查目的）
    ??  ->  ||     （x ?? y → x || y）
   转换只影响语法检查，绝不写回源文件。

用法：
    python scripts/check_js.py [file.html file.js ...]
默认检查 web/static/index.html 的全部内联脚本。
退出码 0=全部通过；1=存在语法错误。作为提交前验证链的一环。
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import esprima

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_RE = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S | re.I)

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
    # 结果仅供参考——若触发此分支，输出明确警告以便 CI 干预升级环境
    print(f"[WARN] 未找到 node 可执行文件，JS 语法检查降级为 esprima 兼容层（结果仅供参考）")
    try:
        esprima.parseScript(es2020_compat(source))
        return []
    except esprima.Error as e:
        msg = getattr(e, "description", None) or getattr(e, "message", None) or str(e)
        return [f"{name}:{e.lineNumber}:{e.column}: {msg}"]


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
