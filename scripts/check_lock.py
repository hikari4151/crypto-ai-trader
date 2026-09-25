"""依赖一致性校验：requirements*.txt + requirements*.in 顶层包必须都在 requirements-lock.txt 中。

目的：防止"改了 requirements 却忘了更新 lock"——lock 是可复现环境的
唯一来源，两者不同步时 CI 应拦截（否则新机器装出不同依赖集）。
同时校验开发依赖文件（ruff/esprima/pytest-asyncio 等），避免 CI 缺工具。
.in 是生成 lock 的源清单（B5），txt 是当前安装说明，两边都必须被 lock 覆盖。

用法：
  python scripts/check_lock.py          # 校验，0=一致 1=不一致
  python scripts/check_lock.py --fix    # 把缺失项附加到 lock（自动修复，谨慎）
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
REQ_DEV = ROOT / "requirements-dev.txt"
REQ_IN = ROOT / "requirements.in"
REQ_DEV_IN = ROOT / "requirements-dev.in"
LOCK = ROOT / "requirements-lock.txt"


def top_level_names(req_text: str) -> set[str]:
    """提取 requirements.txt 的顶层包名（去注释/行内注释/extras/版本符）。"""
    names = set()
    for line in req_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line)[0].strip()  # 去行内注释
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(m.group(1))
    return names


def lock_names(lock_text: str) -> set[str]:
    return set(re.findall(r"^([A-Za-z0-9_.\-]+)==", lock_text, re.M))


def requirement_files() -> list[Path]:
    """校验源清单：requirements*.txt 与 *.in（.in 是生成 lock 的源，须与 lock 同覆盖）。"""
    return [p for p in (REQ, REQ_DEV, REQ_IN, REQ_DEV_IN) if p.exists()]


def check() -> tuple[set[str], set[str]]:
    """返回 (missing, extra)。missing=requirements* 有 lock 缺；extra=lock 有 req 缺。"""
    req_top = set()
    for p in requirement_files():
        req_top |= top_level_names(p.read_text(encoding="utf-8"))
    lk = lock_names(LOCK.read_text(encoding="utf-8"))
    missing = req_top - lk
    # lock 里可以有传递依赖（extra 是正常的），只报缺失
    return missing, set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="把缺失项附加到 lock（需人工确认版本）")
    args = ap.parse_args()
    missing, _ = check()
    if not missing:
        print(f"[lock] 依赖一致性校验通过（{LOCK.name} 覆盖 requirements*.txt / *.in 全部顶层包）")
        return 0
    print(f"[lock] 不一致：requirements 文件有 {len(missing)} 个包不在 lock 中:")
    for m in sorted(missing):
        print(f"  - {m}")
    if args.fix:
        # 从已安装环境提取缺失包的精确版本（pip freeze 输出 pkg==ver，更可靠）
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        freeze_map = {}
        for line in r.stdout.splitlines():
            mm = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line)
            if mm:
                freeze_map[mm.group(1).lower()] = f"{mm.group(1)}=={mm.group(2)}"
        adds = []
        missing_now = sorted(missing)
        for pkg in missing_now:
            hit = freeze_map.get(pkg.lower())
            if hit:
                adds.append(hit)
        if adds:
            with LOCK.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(adds) + "\n")
            print(f"[lock] 已追加到 {LOCK.name}: {adds}")
        else:
            print(f"[lock] 无法自动确定版本（pip freeze 未找到），请手动更新 {LOCK.name}")
        return 0
    print(f"[lock] 请运行: python {Path(__file__).name} --fix 或手动更新 {LOCK.name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
