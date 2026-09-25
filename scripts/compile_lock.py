"""依赖锁文件治理（B5）：检查 / 重新生成 Windows 与 Linux 平台锁。

源清单：
  requirements.in        —— 运行依赖顶层声明
  requirements-dev.in    —— 开发/测试工具顶层声明

平台锁：
  requirements-lock.txt        —— Windows 完整锁定（含 winloop/cupy/nvidia-* 等专用包）
  requirements-lock-linux.txt  —— 平台无关/Ubuntu 锁定（过滤 Windows/CUDA 专用包）

用法：
  python scripts/compile_lock.py                  # --check（默认）：校验 Windows lock 覆盖 .in
  python scripts/compile_lock.py --check-platform  # 校验 Windows + Linux 两个锁都覆盖 .in
  python scripts/compile_lock.py --freeze         # 从当前 .venv 重新生成 Windows 锁 + 派生 Linux 锁
  python scripts/compile_lock.py --compile        # uv/pip-compile 生成 Linux 锁（有工具时）
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements-lock.txt"
LINUX_LOCK = ROOT / "requirements-lock-linux.txt"
IN_FILES = (ROOT / "requirements.in", ROOT / "requirements-dev.in")
PY = sys.executable

WINDOWS_ONLY_PREFIXES = ("winloop", "cupy", "nvidia-", "cuda-")


def top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(m.group(1))
    return names


def lock_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r"^([A-Za-z0-9_.\-]+)==", path.read_text(encoding="utf-8"), re.M))


def want_names() -> set[str]:
    want: set[str] = set()
    for p in IN_FILES:
        if p.exists():
            want |= top_level_names(p)
    return want


def filter_linux(freeze_text: str) -> str:
    """从 pip freeze 输出过滤 Windows/CUDA 专用包，得到平台无关（Linux）锁内容。"""
    lines = []
    for line in freeze_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            pkg = stripped.split("==", 1)[0].strip()
            if any(pkg.startswith(p) for p in WINDOWS_ONLY_PREFIXES):
                continue
        lines.append(line)
    return "\n".join(lines)


def check() -> tuple[list[str], set[str], set[str]]:
    """Windows 锁校验：返回 (缺失源文件, 源有而锁缺的包, 锁名集合)。"""
    missing_files = [str(p) for p in IN_FILES if not p.exists()]
    lk = lock_names(LOCK)
    want = want_names()
    missing_names = want - lk
    return missing_files, missing_names, lk


def check_platform() -> list[str]:
    """校验 Windows + Linux 两个锁都覆盖 .in 源清单。"""
    issues: list[str] = []
    want = want_names()
    for label, path in (("Windows", LOCK), ("Linux", LINUX_LOCK)):
        if not path.exists():
            issues.append(f"{label} 锁缺失: {path.name}（请运行 --freeze）")
            continue
        miss = want - lock_names(path)
        if miss:
            issues.append(f"{label} 锁缺包: {sorted(miss)}")
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="依赖锁文件治理（B5）")
    ap.add_argument("--check", action="store_true", help="校验 Windows lock 覆盖 .in（默认）")
    ap.add_argument("--check-platform", action="store_true", help="校验 Windows+Linux 两个锁覆盖 .in")
    ap.add_argument("--freeze", action="store_true", help="从当前 .venv 重新生成 Windows 锁 + 派生 Linux 锁")
    ap.add_argument("--compile", action="store_true", help="uv/pip-compile 生成 Linux 锁（有工具时）")
    args = ap.parse_args(argv)

    if args.compile:
        uv = shutil.which("uv")
        pip_compile = shutil.which("pip-compile")
        if not (uv or pip_compile):
            print("[lock] 未检测到 uv / pip-compile。安装：pip install uv 或 pip install pip-tools")
            return 1
        print("[lock] 使用平台无关 .in 生成 Linux 锁（Windows 锁保留完整专用包）...")
        cmd = ([uv, "pip", "compile", *IN_FILES, "-o", str(LINUX_LOCK)] if uv else
               [pip_compile, *IN_FILES, "-o", str(LINUX_LOCK)])
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[lock] 编译失败: {r.stderr.strip()[:400]}")
            return 1
        print(f"[lock] 已生成 {LINUX_LOCK.name}")
        return 0

    if args.freeze:
        print("[lock] 用 pip freeze 重新生成 Windows 完整锁 + 派生 Linux 锁 ...")
        r = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[lock] pip freeze 失败: {r.stderr.strip()[:200]}")
            return 1
        body = r.stdout.strip()
        win_header = (
            "# requirements-lock.txt - 由 scripts/compile_lock.py --freeze 重新生成（B5）\n"
            "# 安装: pip install -r requirements-lock.txt（Windows 完整环境）\n"
            "# 源清单: requirements.in + requirements-dev.in（check_lock.py 校验覆盖）\n"
            "# 注：含 Windows/CUDA 专用包（winloop/cupy/nvidia-*）；Linux 用 requirements-lock-linux.txt。\n"
        )
        LOCK.write_text(win_header + body + "\n", encoding="utf-8")
        linux_body = filter_linux(body)
        linux_header = (
            "# requirements-lock-linux.txt - 由 scripts/compile_lock.py --freeze 派生（B5）\n"
            "# 安装: pip install -r requirements-lock-linux.txt（平台无关/Ubuntu）\n"
            "# 源清单: requirements.in + requirements-dev.in；Windows 完整锁为 requirements-lock.txt。\n"
        )
        LINUX_LOCK.write_text(linux_header + linux_body + "\n", encoding="utf-8")
        print(f"[lock] 已重新生成 {LOCK.name}（{len(body.splitlines())} 行）+ "
              f"{LINUX_LOCK.name}（{len(linux_body.splitlines())} 行）")
        return 0

    if args.check_platform:
        issues = check_platform()
        if issues:
            for i in issues:
                print(f"[lock][FAIL] {i}")
            return 1
        print("[lock] 平台锁校验通过：Windows 与 Linux 锁均覆盖 .in 全部顶层包")
        return 0

    # 默认 --check（Windows 锁）
    missing_files, missing_names, lk = check()
    if missing_files:
        print(f"[lock] 缺失源文件: {', '.join(missing_files)}")
        return 1
    if not lk:
        print(f"[lock] 缺失 {LOCK.name}（请先运行 --freeze）")
        return 1
    if missing_names:
        print(f"[lock] 不一致：.in 有 {len(missing_names)} 个包不在 Windows 锁中:")
        for n in sorted(missing_names):
            print(f"  - {n}")
        return 1
    print(f"[lock] 校验通过：{LOCK.name} 覆盖 requirements.in + requirements-dev.in 全部顶层包")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
