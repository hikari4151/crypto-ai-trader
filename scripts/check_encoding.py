"""F1/F2 编码红线自动化（2026-09-27）：把 REDEV_GUIDE 里两条只能人肉双击验证的
Windows/编码红线变成 CI/verify_all 可拦截的门禁。

背景（都是当年真实踩过的坑，见 docs/REDEV_GUIDE.md §3-F）：
- F1：中文 Windows 默认 locale=GBK；项目文本文件必须 **UTF-8（无 BOM）** 可解码。
  settings.py 读 config.yaml 显式 encoding='utf-8' 就是为此——若某个 .py/.yaml 是
  GBK 或带 BOM，会启动即崩或首字符被吞（双击 start.bat 才复现，开发 shell 复现不了）。
- F2：.bat 文件必须 **纯 ASCII**。cmd 按 ANSI 解析 UTF-8，每行首字符会被吞；
  start.bat 曾有非 ASCII 字符导致 `cho`/`EM` 类错误。

本脚本用 `git ls-files` 锁定**入库文本文件**（不看运行态目录），校验：
  1. 文本文件 UTF-8 可解码（errors 后无 replacement char）且无 UTF-8 BOM
  2. .bat 文件字节全部 <= 0x7F（纯 ASCII）
  3. Python 文件额外检查：源码能 compile（保护 F1 域常见的中文注释/cr 换行问题）

用法：
  python scripts/check_encoding.py            # 0=通过 1=有违规（打印清单）
  python scripts/check_encoding.py --list     # 只列出待检查文件（调试用）
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 运行态/工具目录：不入库或不参与编码门禁（git ls-files 不会列 .venv，这里兜底排除）
SKIP_PREFIX = ("data/", ".optim/", ".venv/", "vendor/", "dsh-memory-setup/",
               ".dsh-memory-setup/", "node_modules/")
# 二进制扩展名：跳过文本校验（永远不要尝试按文本解码它们）
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".zip", ".gz",
               ".exe", ".dll", ".db", ".sqlite", ".pkl", ".npz", ".pdf", ".woff",
               ".woff2", ".ttf", ".eot"}
BOM = b"\xef\xbb\xbf"


def tracked_text_files() -> list[str]:
    """git ls-files 全量，过滤运行态/二进制，返回相对路径。"""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    files: list[str] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(SKIP_PREFIX):
            continue
        ext = Path(line).suffix.lower()
        if ext in BINARY_EXTS:
            continue
        files.append(line)
    return files


def check_one(rel: str) -> list[str]:
    """单文件校验，返回违规说明（空=通过）。二进制/未知后缀按文件头嗅探跳过。"""
    p = ROOT / rel
    if not p.exists():
        return []  # 已被删除但仍在 index？git ls-files 不会含已删文件
    issues: list[str] = []
    try:
        raw = p.read_bytes()
    except OSError as e:
        return [f"{rel}: 读取失败 {e}"]
    # 二进制防御：扩展名已知或 NUL 密集（文本文件几乎不含 NUL）即跳过文本解码
    if Path(rel).suffix.lower() in BINARY_EXTS:
        return []
    if not raw:
        return []
    if b"\x00" in raw[:4096]:
        return []
    if raw.startswith(BOM):
        issues.append(f"{rel}: 含 UTF-8 BOM（应为无 BOM UTF-8，防止首字符被吞）")
    if Path(rel).suffix.lower() == ".bat":
        bad = [i for i, b in enumerate(raw) if b > 0x7F]
        if bad:
            head = ", ".join(str(i) for i in bad[:10])
            issues.append(f"{rel}: 非 ASCII 字节 {len(bad)} 处（cmd 按 ANSI 解析会吞行首字符）"
                          f"@{head}")
        return issues
    # 文本：UTF-8 可解码（无 replacement char = 解码完全）
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        issues.append(f"{rel}: 非 UTF-8 文本（第 {e.start} 字节附近解码失败：{e.reason}）")
        return issues
    if "\ufffd" in text:
        issues.append(f"{rel}: 含 replacement char（U+FFFD，可能是 GBK/非法序列）")
    # Python 源码编译（F1 域：中文注释在 GBK 下成假语法错误）
    if rel.endswith(".py"):
        try:
            compile(text, rel, "exec")
        except SyntaxError as e:
            issues.append(f"{rel}: 语法错误 {e.msg} @ line {e.lineno}（疑似编码问题）")
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1/F2 编码红线门禁（UTF-8 无 BOM / .bat 纯 ASCII）")
    ap.add_argument("--list", action="store_true", help="只列出待检查文件")
    args = ap.parse_args(argv)

    files = tracked_text_files()
    if args.list:
        for f in files:
            print(f)
        return 0

    bad: list[str] = []
    for rel in files:
        bad.extend(check_one(rel))
    if bad:
        print(f"[encoding] [FAIL] {len(bad)} 处编码违规:")
        for line in bad[:50]:
            print(f"  - {line}")
        if len(bad) > 50:
            print(f"  … 其余 {len(bad) - 50} 处未列出")
        print("[encoding] FAILED")
        return 1
    print(f"[encoding] 通过：{len(files)} 个入库文本文件均 UTF-8（无 BOM），.bat 纯 ASCII")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
