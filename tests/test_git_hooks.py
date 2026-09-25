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
