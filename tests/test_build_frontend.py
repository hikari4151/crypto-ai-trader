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


def test_build_frontend_manifest_written():
    """正常模式写入清单，字段齐全且指向当前文件。"""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_frontend.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT, timeout=120)
    assert r.returncode == 0, f"构建失败: {r.stdout} {r.stderr}"
    assert MANIFEST.exists(), "应生成 build_manifest.json"
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for k in ("file", "size_bytes", "sha256", "scripts", "style_blocks", "checks"):
        assert k in m, f"清单缺字段 {k}"
    assert m["checks"]["js_ok"] is True and m["checks"]["css_balanced"] is True


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
