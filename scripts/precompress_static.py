#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预压缩静态资源（第十五轮）：为不可变/静态大文件生成 .gz 副本。

背景：starlette 1.6.0 的 GZipMiddleware 无缓存，每次请求对 ≥100KB 响应实时压缩
（vendor 文件走线程池 + capacity limiter）。实测 20 并发 echarts(637KB) 请求：
实时压缩墙钟 522ms vs 原样 76ms——重复压缩/排队占 85% 延迟。immutable vendor
与 no-cache index.html 每次都被重复 gzip，属纯浪费。

方案：构建期生成 <file>.gz，web/main.py 的 index/vendor 路由对支持 gzip 的
客户端直接伺服预压缩文件（Content-Encoding: gzip），零实时压缩。GZipMiddleware
保留兜底（API 大响应动态、不预压缩）。

用法：
  python scripts/precompress_static.py             # 生成 web/static/*.gz 与 vendor/*.gz
  python scripts/precompress_static.py --check    # 校验 .gz 与源文件 mtime 一致
"""
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"
VENDOR = STATIC / "vendor"

# 预压缩目标：仅 vendor 不可变文件（immutable 长缓存、内容稳定，.gz 同步零风险）。
# 排除 index.html（no-cache + 会被并行编辑实时改动——预压缩副本易过期，mtime 检查在
# 并行编辑下不可靠，可能伺服陈旧页面；它保留 GZipMiddleware 兜底，单次压缩 ~14ms 可接受）。
# 排除 <100KB 小文件（GZipMiddleware 本来就不压缩，收益 <1ms）。
MIN_SIZE = 50 * 1024  # 略低于 GZipMiddleware 的 100KB，覆盖稍小但有收益的边缘


def targets() -> list[Path]:
    files = sorted(VENDOR.glob("*"))
    return [p for p in files if p.is_file() and p.suffix != ".gz" and p.stat().st_size >= MIN_SIZE]


def generate(force: bool = False) -> list[tuple[Path, int, int]]:
    out = []
    for src in targets():
        gz = src.with_suffix(src.suffix + ".gz")
        if not force and gz.exists() and gz.stat().st_mtime >= src.stat().st_mtime:
            continue
        data = gzlib(src)
        gz.write_bytes(data)
        out.append((src, src.stat().st_size, len(data)))
    return out


def gzlib(data_path: Path, level: int = 6) -> bytes:
    with data_path.open("rb") as f:
        raw = f.read()
    return gzip.compress(raw, level)


def check() -> list[str]:
    """校验：.gz 存在且不早于源文件。"""
    issues = []
    for src in targets():
        gz = src.with_suffix(src.suffix + ".gz")
        if not gz.exists():
            issues.append(f"缺 .gz: {gz.relative_to(ROOT)}")
        elif gz.stat().st_mtime < src.stat().st_mtime:
            issues.append(f"过期: {gz.relative_to(ROOT)}")
    return issues


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不生成")
    ap.add_argument("--force", action="store_true", help="强制重新生成")
    args = ap.parse_args()

    if args.check:
        issues = check()
        if issues:
            print("[precompress] ❌ 超期/缺失 .gz：")
            for i in issues[:20]:
                print("   ", i)
            sys.exit(1)
        print(f"[precompress] ✅ 全部 {len(targets())} 个目标 .gz 最新")
        return

    out = generate(force=args.force)
    if not out:
        print("[precompress] 全部已最新，无变更")
        return
    print(f"[precompress] 生成 {len(out)} 个 .gz：")
    for src, raw, comp in out:
        print(f"   {src.relative_to(ROOT)}: {raw/1024:.0f}KB -> {comp/1024:.0f}KB")


if __name__ == "__main__":
    main()
