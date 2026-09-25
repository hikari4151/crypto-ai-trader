"""依赖一致性校验回归：requirements*.txt / *.in 顶层包必须都在 lock 中。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _write_files(tmp_path, req: str, lock: str, req_dev: str = ""):
    r = tmp_path / "requirements.txt"
    d = tmp_path / "requirements-dev.txt"
    lk = tmp_path / "requirements-lock.txt"
    r.write_text(req, encoding="utf-8")
    d.write_text(req_dev, encoding="utf-8")
    lk.write_text(lock, encoding="utf-8")
    return r, lk, d


def _patch_paths(monkeypatch, cl, req, lock, dev, req_in=None, dev_in=None):
    monkeypatch.setattr(cl, "REQ", req)
    monkeypatch.setattr(cl, "REQ_DEV", dev)
    monkeypatch.setattr(cl, "LOCK", lock)
    # 默认 .in 文件不存在（不参与）；测试需要时显式创建
    monkeypatch.setattr(cl, "REQ_IN", req_in if req_in is not None else req.parent / "requirements.in")
    monkeypatch.setattr(cl, "REQ_DEV_IN", dev_in if dev_in is not None else dev.parent / "requirements-dev.in")


def test_top_level_names_parsing():
    """解析 requirements：去注释/行内注释/extras/版本符。"""
    from scripts.check_lock import top_level_names
    text = ("# 注释\n"
            "requests>=2.0 # 行内注释\n"
            "flask\n"
            "pandas[extra]>=1.0\n"
            "numpy==1.24.0\n")
    names = top_level_names(text)
    assert names == {"requests", "flask", "pandas", "numpy"}, names


def test_check_consistent(tmp_path, monkeypatch):
    """lock 覆盖全部顶层包（含 dev 文件）→ 通过。"""
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path,
                                  "flask\npandas>=1.0\n",
                                  "flask==3.0.0\npandas==2.0.0\n")
    _patch_paths(monkeypatch, cl, req, lock, dev)
    missing, _ = cl.check()
    assert not missing


def test_check_missing_reports(tmp_path, monkeypatch):
    """requirements 有而 lock 缺 → 报告缺失。"""
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path,
                                  "flask\npandas>=1.0\n",
                                  "flask==3.0.0\n")
    _patch_paths(monkeypatch, cl, req, lock, dev)
    missing, _ = cl.check()
    assert missing == {"pandas"}


def test_check_missing_from_dev_file_reports(tmp_path, monkeypatch):
    """requirements-dev.txt 有而 lock 缺 → 报告缺失。"""
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path,
                                  "flask\n",
                                  "flask==3.0.0\n",
                                  "ruff>=0.1\n")
    _patch_paths(monkeypatch, cl, req, lock, dev)
    missing, _ = cl.check()
    assert missing == {"ruff"}


def test_check_missing_from_in_file_reports(tmp_path, monkeypatch):
    """requirements.in 有而 lock 缺 → 报告缺失（B5 源清单纳入校验）。"""
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path, "flask\n", "flask==3.0.0\n")
    req_in = tmp_path / "requirements.in"
    req_in.write_text("ruff>=0.1\n", encoding="utf-8")
    _patch_paths(monkeypatch, cl, req, lock, dev, req_in=req_in)
    missing, _ = cl.check()
    assert missing == {"ruff"}


def test_main_returns_one_on_missing(tmp_path, monkeypatch, capsys):
    """CLI：不一致时退出码 1。"""
    import sys
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path, "missing_pkg\n", "other==1.0\n")
    _patch_paths(monkeypatch, cl, req, lock, dev)
    monkeypatch.setattr(sys, "argv", ["check_lock.py"])
    rc = cl.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "缺失" in out or "不一致" in out


def test_main_returns_zero_on_consistent(tmp_path, monkeypatch):
    """CLI：一致时退出码 0。"""
    import sys
    import scripts.check_lock as cl
    req, lock, dev = _write_files(tmp_path, "flask\n", "flask==3.0.0\n")
    _patch_paths(monkeypatch, cl, req, lock, dev)
    monkeypatch.setattr(sys, "argv", ["check_lock.py"])
    assert cl.main() == 0
