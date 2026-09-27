"""P1-B2：pre-commit hook 安装器回归。"""
from pathlib import Path

import scripts.install_git_hooks as gh


def test_hook_script_calls_verify_all_fast():
    s = gh.hook_script()
    assert "verify_all.py" in s
    assert "--skip-pytest" in s and "--skip-backtest" in s
    assert "#!/bin/sh" in s


def test_install_uninstall_roundtrip(tmp_path):
    git_dir = Path(tmp_path) / ".git"
    assert gh.install_hook(git_dir) is True
    target = git_dir / "hooks" / "pre-commit"
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith("#!/bin/sh")

    # 幂等：已存在时不覆盖
    assert gh.install_hook(git_dir) is False

    assert gh.uninstall_hook(git_dir) is True
    assert not target.exists()
    assert gh.uninstall_hook(git_dir) is False


def test_check_hook_states(tmp_path):
    git_dir = Path(tmp_path) / ".git"
    target = git_dir / "hooks" / "pre-commit"

    # 未安装
    assert gh.check_hook(git_dir) == (False, False)

    # 已装且内容最新
    gh.install_hook(git_dir)
    assert gh.check_hook(git_dir) == (True, True)

    # 内容过期（外部改动钩子）
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert gh.check_hook(git_dir) == (True, False)


def test_install_force_overwrites(tmp_path):
    git_dir = Path(tmp_path) / ".git"
    target = git_dir / "hooks" / "pre-commit"
    gh.install_hook(git_dir)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    # 无 force 不覆盖
    assert gh.install_hook(git_dir) is False
    # force 覆盖
    assert gh.install_hook(git_dir, force=True) is True
    assert gh.check_hook(git_dir) == (True, True)


def test_main_check_exit_codes(tmp_path, monkeypatch, capsys):
    git_dir = Path(tmp_path) / ".git"
    monkeypatch.setattr(gh, "ROOT", tmp_path)
    # 未安装 → exit 1
    assert gh.main(["--check"]) == 1
    # 安装后 → exit 0
    gh.install_hook(git_dir)
    assert gh.main(["--check"]) == 0
    # 过期 → exit 1
    (git_dir / "hooks" / "pre-commit").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert gh.main(["--check"]) == 1
