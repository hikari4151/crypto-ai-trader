"""B5：依赖锁文件治理（compile_lock.py）回归。"""
import subprocess

import scripts.compile_lock as cl


def _write(tmp_path, in_a: str, in_b: str, lock: str, linux_lock: str | None = None):
    ia = tmp_path / "requirements.in"
    ib = tmp_path / "requirements-dev.in"
    lk = tmp_path / "requirements-lock.txt"
    la = tmp_path / "requirements-lock-linux.txt"
    ia.write_text(in_a, encoding="utf-8")
    ib.write_text(in_b, encoding="utf-8")
    lk.write_text(lock, encoding="utf-8")
    if linux_lock is not None:
        la.write_text(linux_lock, encoding="utf-8")
    return ia, ib, lk, la


def test_check_ok(tmp_path, monkeypatch):
    ia, ib, lk, _ = _write(tmp_path, "flask>=2.0\n", "ruff==0.1\n", "flask==3.0\nruff==0.1\n")
    monkeypatch.setattr(cl, "IN_FILES", (ia, ib))
    monkeypatch.setattr(cl, "LOCK", lk)
    missing_files, missing_names, lk_names = cl.check()
    assert missing_files == []
    assert missing_names == set()
    assert "flask" in lk_names


def test_check_reports_missing_lock_pkg(tmp_path, monkeypatch):
    ia, ib, lk, _ = _write(tmp_path, "flask>=2.0\n", "ruff==0.1\n", "flask==3.0\n")
    monkeypatch.setattr(cl, "IN_FILES", (ia, ib))
    monkeypatch.setattr(cl, "LOCK", lk)
    _, missing_names, _ = cl.check()
    assert missing_names == {"ruff"}


def test_check_reports_missing_source_file(tmp_path, monkeypatch):
    ia = tmp_path / "requirements.in"
    ib = tmp_path / "requirements-dev.in"
    lk = tmp_path / "requirements-lock.txt"
    ia.write_text("flask\n", encoding="utf-8")
    lk.write_text("flask==3.0\n", encoding="utf-8")  # ib 不创建
    monkeypatch.setattr(cl, "IN_FILES", (ia, ib))
    monkeypatch.setattr(cl, "LOCK", lk)
    missing_files, _, _ = cl.check()
    assert len(missing_files) == 1 and str(ib) in missing_files


def test_filter_linux_removes_windows_only():
    out = cl.filter_linux("pkg==1.0\nwinloop==0.6.3\ncupy-cuda12x==14.1.1\nnvidia-cublas-cu12==12.9\nfoo==2.0\n")
    assert "pkg==1.0" in out and "foo==2.0" in out
    assert "winloop" not in out and "cupy" not in out and "nvidia" not in out


def test_check_platform_reports_missing_linux(tmp_path, monkeypatch):
    ia, ib, lk, la = _write(tmp_path, "flask\n", "ruff==0.1\n", "flask==3.0\nruff==0.1\n")
    monkeypatch.setattr(cl, "IN_FILES", (ia, ib))
    monkeypatch.setattr(cl, "LOCK", lk)
    monkeypatch.setattr(cl, "LINUX_LOCK", la)  # 不创建 → 缺失
    issues = cl.check_platform()
    assert any("Linux" in i and "缺失" in i for i in issues)


def test_check_platform_ok(tmp_path, monkeypatch):
    ia, ib, lk, la = _write(tmp_path, "flask\n", "ruff==0.1\n",
                            "flask==3.0\nruff==0.1\n", linux_lock="flask==3.0\nruff==0.1\n")
    monkeypatch.setattr(cl, "IN_FILES", (ia, ib))
    monkeypatch.setattr(cl, "LOCK", lk)
    monkeypatch.setattr(cl, "LINUX_LOCK", la)
    assert cl.check_platform() == []


def test_main_freeze_writes_both_locks(tmp_path, monkeypatch):
    lk = tmp_path / "requirements-lock.txt"
    la = tmp_path / "requirements-lock-linux.txt"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0,
                                           stdout="pkg==1.0\nfoo==2.0\nwinloop==0.6.3\n", stderr="")

    monkeypatch.setattr(cl, "LOCK", lk)
    monkeypatch.setattr(cl, "LINUX_LOCK", la)
    monkeypatch.setattr(cl.subprocess, "run", fake_run)
    rc = cl.main(["--freeze"])
    assert rc == 0
    win_text = lk.read_text(encoding="utf-8")
    linux_text = la.read_text(encoding="utf-8")
    assert "pkg==1.0" in win_text and "winloop==0.6.3" in win_text
    assert "compile_lock.py --freeze" in win_text
    assert "pkg==1.0" in linux_text and "winloop" not in linux_text


def test_main_compile_returns_one_without_tool():
    assert cl.main(["--compile"]) == 1
