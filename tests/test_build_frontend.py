"""前端构建链（P2-⑩）回归：校验 + 清单生成。

覆盖：
- build_frontend 校验当前 index.html：JS 通过 + CSS 花括号配平
- 清单写入 build_manifest.json 且字段齐全（sha256/scripts/checks）
- check-only 模式不写清单
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "static" / "index.html"
MANIFEST = ROOT / "web" / "static" / "build_manifest.json"


def test_build_frontend_js_and_css_pass():
    """当前 index.html 必须通过 JS 语法 + CSS 配平（构建门禁）。"""
    from scripts.build_frontend import build_manifest
    m = build_manifest()
    assert m["checks"]["js_ok"] is True, f"JS 校验失败: {m['errors']}"
    assert m["checks"]["css_balanced"] is True, f"CSS 配平失败: {m['errors']}"


def test_build_frontend_manifest_written(tmp_path, monkeypatch):
    """正常模式写入清单，字段齐全且指向当前文件。

    注意：必须把 MANIFEST 指到 tmp —— 直跑脚本会按工作区 index.html 重写**已入库的**
    build_manifest.json（曾因此把清单改成与未提交改动一致，污染工作区）。
    """
    import subprocess
    import sys
    import scripts.build_frontend as bf
    target = tmp_path / "build_manifest.json"
    # 子进程读不到 monkeypatch 的模块属性，改用真实脚本但先备份/还原入库清单
    original = MANIFEST.read_bytes()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_frontend.py")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=ROOT, timeout=120)
        assert r.returncode == 0, f"构建失败: {r.stdout} {r.stderr}"
        assert MANIFEST.exists(), "应生成 build_manifest.json"
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for k in ("file", "size_bytes", "sha256", "scripts", "style_blocks", "checks"):
            assert k in m, f"清单缺字段 {k}"
        assert m["checks"]["js_ok"] is True and m["checks"]["css_balanced"] is True
        # 写模式必须与"当前 index.html"一致（不是与旧清单一致）
        assert bf.manifest_drift(m) is None
    finally:
        # 还原入库清单，避免测试污染工作区（入库清单描述的是**已提交**的 index.html）
        MANIFEST.write_bytes(original)
    assert target  # 保留 tmp_path 引用（不再使用子进程外部路径）


def test_build_frontend_check_only_no_manifest(tmp_path, monkeypatch):
    """check-only 模式不写清单（只校验）。"""
    import scripts.build_frontend as bf
    # 临时把清单路径指到 tmp，验证 check-only 不创建
    target = tmp_path / "build_manifest.json"
    monkeypatch.setattr(bf, "MANIFEST", target)
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_frontend.py"), "--check-only"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT, timeout=120)
    assert r.returncode == 0
    assert not target.exists(), "check-only 不应写清单"


# ---- 清单可信度（sha256/size_bytes 同基底 + 跨平台稳定 + 漂移门禁）----
def test_manifest_hash_and_size_share_lf_basis():
    """两个字段必须同基底（LF 归一化）：原实现 sha256 用 read_text 归一化文本、
    size_bytes 用 stat() 原始字节，CRLF 工作区下互不相符，清单不能作缓存凭据。"""
    import hashlib
    from scripts.build_frontend import build_manifest, normalized_bytes
    m = build_manifest()
    norm = normalized_bytes()
    assert m["size_bytes"] == len(norm)
    assert m["sha256"] == hashlib.sha256(norm).hexdigest()


def test_manifest_is_line_ending_agnostic(tmp_path, monkeypatch):
    """跨平台红线：本仓库 core.autocrlf=true 且无 .gitattributes —— git blob 是 LF、
    Windows 工作区是 CRLF。清单若按工作区原始字节计算，Ubuntu CI 与 Windows 会得出
    不同 sha256（该值同时是运维页的"前端版本"与缓存 key）。同一内容两种行尾必须同值。"""
    import scripts.build_frontend as bf
    crlf = tmp_path / "crlf.html"
    lf = tmp_path / "lf.html"
    body = '<html>\n<body>\n<div id="a"></div>\n</body>\n</html>\n'
    crlf.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    lf.write_bytes(body.encode("utf-8"))
    monkeypatch.setattr(bf, "INDEX", crlf)
    a = bf.build_manifest()
    monkeypatch.setattr(bf, "INDEX", lf)
    b = bf.build_manifest()
    assert a["sha256"] == b["sha256"], "CRLF/LF 必须得到同一 sha256（跨平台一致）"
    assert a["size_bytes"] == b["size_bytes"], "CRLF/LF 必须得到同一 size_bytes"


def test_manifest_drift_detected_on_content_change(tmp_path, monkeypatch):
    """源文件变了但清单没重生成 → 必须报漂移。"""
    import scripts.build_frontend as bf
    fake = tmp_path / "build_manifest.json"
    fake.write_text(json.dumps({"sha256": "deadbeef" * 8, "size_bytes": 1}), encoding="utf-8")
    monkeypatch.setattr(bf, "MANIFEST", fake)
    drift = bf.manifest_drift(bf.build_manifest())
    assert drift is not None and "已过期" in drift


def test_manifest_drift_none_when_consistent(tmp_path, monkeypatch):
    import scripts.build_frontend as bf
    fake = tmp_path / "build_manifest.json"
    monkeypatch.setattr(bf, "MANIFEST", fake)
    assert bf.manifest_drift(bf.build_manifest()) is None
    assert bf.main([]) == 0
    assert bf.manifest_drift(bf.build_manifest()) is None


def test_manifest_drift_none_without_manifest(tmp_path, monkeypatch):
    """从未生成过清单 → 不算漂移（首次构建前允许缺失）。"""
    import scripts.build_frontend as bf
    monkeypatch.setattr(bf, "MANIFEST", tmp_path / "absent.json")
    assert bf.manifest_drift(bf.build_manifest()) is None


def test_manifest_drift_reports_broken_json(tmp_path, monkeypatch):
    import scripts.build_frontend as bf
    bad = tmp_path / "build_manifest.json"
    bad.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(bf, "MANIFEST", bad)
    drift = bf.manifest_drift(bf.build_manifest())
    assert drift is not None and "合法 JSON" in drift


def test_check_only_require_manifest_fails_on_drift(tmp_path, monkeypatch, capsys):
    """默认只 WARN（本地并行编辑 index.html 不拦人），--require-manifest 才 FAIL（CI 用）。"""
    import scripts.build_frontend as bf
    fake = tmp_path / "build_manifest.json"
    fake.write_text(json.dumps({"sha256": "0" * 64, "size_bytes": 1}), encoding="utf-8")
    monkeypatch.setattr(bf, "MANIFEST", fake)
    assert bf.main(["--check-only"]) == 0
    assert "[WARN]" in capsys.readouterr().out
    assert bf.main(["--check-only", "--require-manifest"]) == 1


def test_write_mode_refreshes_manifest(tmp_path, monkeypatch):
    """写模式应把清单刷新到与 index.html 一致（漂移收敛）。"""
    import scripts.build_frontend as bf
    fake = tmp_path / "build_manifest.json"
    fake.write_text(json.dumps({"sha256": "0" * 64, "size_bytes": 1}), encoding="utf-8")
    monkeypatch.setattr(bf, "MANIFEST", fake)
    assert bf.main([]) == 0
    assert bf.manifest_drift(bf.build_manifest()) is None
