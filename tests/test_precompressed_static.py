"""第十五轮：预压缩静态资源契约测试。

守卫：web/static/vendor 下的 .gz 预压缩副本（scripts/precompress_static.py 生成）必须
存在且不早于源文件——否则 web/main.py 的预压缩伺服路由会静默回退（仍可用，
但重复压缩成本回归）。同时锁定 main.py 的伺服逻辑契约（Content-Encoding/Vary）。
注：index.html 是 no-cache + 并行编辑频繁，**有意不做预压缩**（.gz 同步有陈旧风险），
由 GZipMiddleware 兜底；本测试只覆盖 immutable vendor。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "web" / "static" / "vendor"
MIN_SIZE = 50 * 1024  # 与 precompress_static.py 一致


def _precompress_targets():
    files = sorted(VENDOR.glob("*"))
    return [p for p in files if p.is_file() and p.suffix != ".gz" and p.stat().st_size >= MIN_SIZE]


def test_gz_files_exist_and_fresh():
    """所有 ≥50KB vendor 文件必须有最新 .gz 副本。"""
    stale = []
    for src in _precompress_targets():
        gz = src.with_suffix(src.suffix + ".gz")
        if not gz.exists():
            stale.append(f"缺 .gz: {gz.relative_to(ROOT)}")
        elif gz.stat().st_mtime < src.stat().st_mtime:
            stale.append(f"过期: {gz.relative_to(ROOT)}")
    assert not stale, "运行 scripts/precompress_static.py 重新生成 .gz：\n" + "\n".join(stale)


def test_precompressed_served_by_main():
    """main.py 必须对 gzip 客户端伺服 vendor 的 .gz（Content-Encoding:gzip + Vary）。"""
    src = (ROOT / "web" / "main.py").read_text(encoding="utf-8")
    assert "Content-Encoding" in src, "预压缩响应须带 Content-Encoding: gzip"
    assert '"gzip"' in src, "须检测 Accept-Encoding 协商"
    assert "Vary" in src, "须带 Vary: Accept-Encoding"
    assert ".gz" in src and "vendor" in src, "vendor 路由须伺服预压缩副本"


def test_precompressed_content_matches_source():
    """.gz 解压后必须与源文件逐字节一致（防压缩过程损坏）。"""
    import gzip

    checked = 0
    for src in _precompress_targets():
        gz = src.with_suffix(src.suffix + ".gz")
        if not gz.exists():
            continue
        with src.open("rb") as f:
            raw = f.read()
        with gz.open("rb") as f:
            decompressed = gzip.decompress(f.read())
        assert decompressed == raw, f"{src.name} 的 .gz 解压后与源不一致"
        checked += 1
    assert checked > 0, "应至少有一个预压缩目标"
