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
  python scripts/build_frontend.py --check-only --require-manifest
                                                 # 只校验，并要求入库清单与源文件一致
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

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


def normalized_bytes() -> bytes:
    """index.html 的 LF 归一化字节（跨平台稳定基底）。

    本仓库 core.autocrlf=true 且无 .gitattributes：git blob 是 LF，Windows 工作区检出为
    CRLF（实测同一文件 601424B vs 614685B）。清单一旦按"工作区原始字节"计算，
    Ubuntu CI 与 Windows 会得出不同的 sha256 —— 而该 sha256 正是 web/api/ops.py
    暴露给运维的"前端部署版本"、也是缓存 hash，必须跨平台一致。
    因此统一以 LF 归一化字节作为 sha256 与 size_bytes 的共同基底
    （与原实现 `read_text()`（universal newlines）+ 编码 的语义一致，只是让两个字段同源）。
    """
    raw = INDEX.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _rel(path: Path) -> str:
    """项目内相对路径；不在项目内（测试/自定义路径）时退回原路径字符串。"""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_manifest() -> dict:
    text = INDEX.read_text(encoding="utf-8")
    # sha256 与 size_bytes 必须同基底：原实现 sha256 用 read_text 归一化文本、
    # size_bytes 用 stat() 原始字节，两者在 CRLF 工作区互不相符（清单不能作凭据）。
    norm = normalized_bytes()
    js_errors = _check_js(text)
    css_errors = check_css_blocks(text)
    ref_warnings = check_static_refs(text)
    scripts = sum(1 for _ in SCRIPT_RE.finditer(text))
    style_blocks = sum(1 for _ in STYLE_RE.finditer(text))
    manifest = {
        "file": _rel(INDEX),
        "size_bytes": len(norm),
        "sha256": hashlib.sha256(norm).hexdigest(),
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


def manifest_drift(manifest: dict) -> Optional[str]:
    """已提交清单与当前 index.html 的漂移说明（无漂移/清单缺失返回 None）。

    背景：build_manifest.json 是入库的生成物，被 web/api/ops.py 当作「前端部署版本」
    暴露给运维（sha256 前 8 位）。index.html 改了而清单没重生成，运维就会看到一个
    永不变化、指向旧构建的版本号——属于「显示正常但信息是假的」，必须显式拦住。
    只比较 size_bytes/sha256（mtime/built_at 天然随生成时刻变化，不参与判定）。
    """
    if not MANIFEST.exists():
        return None  # 未生成过：由调用方决定是否要求生成（check-only 不强制）
    try:
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{MANIFEST.name} 不是合法 JSON（重新生成: python scripts/build_frontend.py）"
    if old.get("sha256") == manifest["sha256"] and old.get("size_bytes") == manifest["size_bytes"]:
        return None
    return (f"{MANIFEST.name} 已过期：清单 sha256={str(old.get('sha256'))[:12]}… "
            f"size={old.get('size_bytes')}，当前 index.html(LF) sha256={manifest['sha256'][:12]}… "
            f"size={manifest['size_bytes']}（重新生成: python scripts/build_frontend.py）")


def _check_js(text: str) -> list[str]:
    from check_js import check_file
    return check_file(INDEX)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只校验，不写清单")
    ap.add_argument("--require-manifest", action="store_true",
                    help="check-only 下也要求已提交清单与 index.html 一致（CI/门禁用）")
    args = ap.parse_args(argv)

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
    # 写模式：漂移是预期中间态（本次就会重写清单），不提示；
    # check-only：源文件改了而入库清单没跟上 —— 默认 WARN（本地并行编辑 index.html 时不拦人），
    # --require-manifest 时 FAIL（CI 判定"已提交状态是否自洽"）。
    drift = manifest_drift(manifest)
    if not args.check_only:
        if ok:
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
            # relative_to 在 MANIFEST 被指向项目外时抛 ValueError（测试/自定义路径场景），
            # 打印路径不构成失败理由，降级为绝对路径
            try:
                shown = MANIFEST.relative_to(ROOT)
            except ValueError:
                shown = MANIFEST
            print(f"[build] 清单已写入 {shown}")
        return 0 if ok else 1
    if drift:
        tag = "FAIL" if args.require_manifest else "WARN"
        print(f"  [{tag}] {drift}")
    if drift and args.require_manifest:
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
