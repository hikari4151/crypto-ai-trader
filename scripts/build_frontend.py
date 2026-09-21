"""前端构建链 v1（P2-⑩）：只读校验 + 产物清单。

设计约束：
- 当前 web/static/index.html 是 7200+ 行单文件且可能被并行工作流编辑，
  因此**不重写/不拆分**源文件（避免冲突）；本构建链做"构建前校验 +
  产物元数据"，为将来缓存 hash 化 / 多文件拆分打基础，零副作用。

能力：
1. JS 校验：复用 scripts/check_js.py（node --check 优先，esprima 回退）
2. CSS 校验：提取 <style> 块，做花括号配平 + 注释/字符串安全剥离
3. 资源引用校验（保守）：静态 `href="#id"` / `use href="#id"` 引用的
   id 须在文档中存在（仅静态引用；动态 :href 绑定跳过）
4. 构建清单：输出 web/static/build_manifest.json
   {file, size_bytes, sha256, mtime_utc, scripts, style_blocks, checks}

用法：
  python scripts/build_frontend.py            # 校验 + 写清单；失败退出码 1
  python scripts/build_frontend.py --check-only  # 只校验，不写清单
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "static" / "index.html"
MANIFEST = ROOT / "web" / "static" / "build_manifest.json"

# scripts/ 可能不在 sys.path（测试 import 本模块时），显式加入以便复用 check_js
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT_RE = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S | re.I)
STYLE_RE = re.compile(r"<style[^>]*>(?P<body>.*?)</style>", re.S | re.I)
STATIC_HREF_RE = re.compile(r'\b(?:href|use)\s*=\s*"#(?P<id>[A-Za-z0-9_:-]+)"')
# 排除 Vue 动态绑定（:href / :use）
DYNAMIC_ATTR_RE = re.compile(r':(href|use)\s*=')


def check_css_blocks(text: str) -> list[str]:
    """对每个 <style> 块做花括号配平（剥离注释/字符串后）。"""
    errors = []
    for i, m in enumerate(STYLE_RE.finditer(text), 1):
        css = m.group("body")
        # 剥离注释
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        # 剥离字符串
        css = re.sub(r"'[^']*'|\"[^\"]*\"", "''", css)
        depth = 0
        for ch in css:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"<style>#{i}: 花括号提前闭合（depth<0）")
                    break
        if depth != 0 and not errors:
            errors.append(f"<style>#{i}: 花括号不平衡 depth={depth}")
    return errors


def check_static_refs(text: str) -> list[str]:
    """静态 #id 引用必须存在（保守：只查字面 href/use="...#..."）。"""
    ids_in_doc = set(re.findall(r'\bid="([A-Za-z0-9_:-]+)"', text))
    warnings = []
    for m in STATIC_HREF_RE.finditer(text):
        ref_id = m.group("id")
        if ref_id not in ids_in_doc:
            # 排除合法的前端内置引用（如 <use> 指向 svg symbol、aria 等常见误报）
            if ref_id in ("i-close", "i-check"):
                continue
            warnings.append(f"静态引用 # {ref_id} 未在文档中找到 id（可能是模板条件渲染，仅警告）")
    return warnings


def build_manifest() -> dict:
    text = INDEX.read_text(encoding="utf-8")
    js_errors = _check_js(text)
    css_errors = check_css_blocks(text)
    ref_warnings = check_static_refs(text)
    scripts = sum(1 for _ in SCRIPT_RE.finditer(text))
    style_blocks = sum(1 for _ in STYLE_RE.finditer(text))
    manifest = {
        "file": str(INDEX.relative_to(ROOT)),
        "size_bytes": INDEX.stat().st_size,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(INDEX.stat().st_mtime)),
        "scripts": scripts,
        "style_blocks": style_blocks,
        "checks": {
            "js_ok": not js_errors,
            "css_balanced": not css_errors,
            "static_ref_warnings": len(ref_warnings),
        },
        "errors": js_errors + css_errors,
        "warnings": ref_warnings[:10],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return manifest


def _check_js(text: str) -> list[str]:
    from check_js import check_file
    return check_file(INDEX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只校验，不写清单")
    args = ap.parse_args()

    if not INDEX.exists():
        print(f"[build] 找不到 {INDEX}")
        return 1
    manifest = build_manifest()
    ok = manifest["checks"]["js_ok"] and manifest["checks"]["css_balanced"]
    print(f"[build] index.html: {manifest['size_bytes']}B sha256={manifest['sha256'][:16]}… "
          f"scripts={manifest['scripts']} styles={manifest['style_blocks']}")
    print(f"[build] JS {'通过' if manifest['checks']['js_ok'] else '失败'} | "
          f"CSS 配平 {'通过' if manifest['checks']['css_balanced'] else '失败'} | "
          f"静态引用警告 {manifest['checks']['static_ref_warnings']} 条")
    for e in manifest["errors"]:
        print(f"  [FAIL] {e}")
    for w in manifest["warnings"]:
        print(f"  [WARN] {w}")
    if not args.check_only and ok:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[build] 清单已写入 {MANIFEST.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
