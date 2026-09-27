"""F1/F2 编码红线门禁（scripts/check_encoding.py）回归。

覆盖：
- 纯 ASCII .bat 通过；含非 ASCII 字节的 .bat 必须报错（F2）
- UTF-8 带 BOM 的文件必须报错（F1）
- 非 UTF-8（GBK）文本必须报错
- 二进制/运行态目录不参与文本校验
- 当前仓库真实状态通过（CI/本机同一判定）
"""
import scripts.check_encoding as ce
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_bat_with_non_ascii_detected():
    """F2：.bat 有任何 >0x7F 字节即为违规（cmd ANSI 解析吞行首字符）。"""
    issues = ce.check_one("x.bat")
    assert issues == []
    # 用真实文件验证检测逻辑：临时造一个带中文的 .bat
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.bat"
        p.write_bytes("@echo off\r\n中文字符\r\n".encode("gbk"))
        orig = ce.ROOT
        try:
            ce.ROOT = Path(d)
            issues = ce.check_one("bad.bat")
            assert any("非 ASCII" in i and "bad.bat" in i for i in issues)
            # 纯 ASCII bat 无违规
            p2 = Path(d) / "ok.bat"
            p2.write_bytes(b"@echo off\r\necho hello\r\n")
            assert ce.check_one("ok.bat") == []
        finally:
            ce.ROOT = orig


def test_bom_detected():
    """F1：UTF-8 BOM 文件必须拦截（首字符被吞/解析错位）。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "with_bom.py"
        p.write_bytes(b"\xef\xbb\xbfprint('x')\n")
        orig = ce.ROOT
        try:
            ce.ROOT = Path(d)
            issues = ce.check_one("with_bom.py")
            assert any("BOM" in i for i in issues)
        finally:
            ce.ROOT = orig


def test_gbk_text_detected():
    """F1：GBK 编码文本（中文 Windows 默认）必须拦截。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gbk.py"
        p.write_bytes("# 中文注释\nprint('ok')\n".encode("gbk"))
        orig = ce.ROOT
        try:
            ce.ROOT = Path(d)
            issues = ce.check_one("gbk.py")
            assert any("非 UTF-8" in i for i in issues), issues
        finally:
            ce.ROOT = orig


def test_valid_utf8_cn_passes():
    """合法 UTF-8 中文注释必须放行（当前仓库全是这种）。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ok.py"
        p.write_text("# 中文注释\nprint('ok')\n", encoding="utf-8")
        orig = ce.ROOT
        try:
            ce.ROOT = Path(d)
            assert ce.check_one("ok.py") == []
        finally:
            ce.ROOT = orig


def test_binary_and_runtime_skipped():
    """二进制扩展名与运行态目录不参与文本校验。"""
    assert ".png" in ce.BINARY_EXTS
    assert ce.SKIP_PREFIX[0] == "data/"
    # 一个假二进制文件不会被文本解码（无违规，即便字节非法）
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "blob.png"
        p.write_bytes(b"\x89PNG\xff\xfe\x00")
        orig = ce.ROOT
        try:
            ce.ROOT = Path(d)
            assert ce.check_one("blob.png") == []
        finally:
            ce.ROOT = orig


def test_repo_state_is_clean():
    """当前仓库必须通过编码红线（CI 与本机同一判定）。"""
    assert ce.main([]) == 0
