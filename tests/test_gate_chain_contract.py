"""第十六轮：门禁链基建契约测试。

守卫三件事（防基建倒退）：
1. pre-commit hook 内容 = verify_all --skip-pytest --skip-backtest --staged
   （staged 模式——他人未完成工作不阻塞本提交；install_git_hooks.py 是事实来源）
2. verify_all 必须含 --staged 参数且 hook 用到它
3. 前端构建产物脚本（precompress/tailwind/echarts）必须注册进 verify_all / ci.yml
   / tests 至少其一（A5 教训：脚本写了没接链 = 白写）
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL_HOOKS = ROOT / "scripts" / "install_git_hooks.py"
VERIFY_ALL = ROOT / "scripts" / "verify_all.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def test_hook_uses_staged_mode():
    """pre-commit hook 必须用 --staged（而非全量），否则他人 WIP 阻塞提交。"""
    src = INSTALL_HOOKS.read_text(encoding="utf-8")
    assert "--staged" in src, "install_git_hooks.py 的 hook 脚本须带 --staged"
    assert "--skip-pytest" in src, "hook 保持 fast 门禁（不跑全量 pytest）"


def test_verify_all_has_staged_flag():
    """verify_all.py 必须实现 --staged 模式（get_staged_files + build_steps(staged=)）。"""
    src = VERIFY_ALL.read_text(encoding="utf-8")
    assert "add_argument(\"--staged\"" in src, "verify_all 须有 --staged 参数"
    assert "--cached" in src and "--name-only" in src, "须读取暂存文件列表"
    assert "ruff(staged)" in src, "staged 模式须只查暂存文件"
    assert "check_redlines(skip-e7)" in src, "未碰 index.html 时须跳过 E7 全量基线"


def test_frontend_build_scripts_registered():
    """前端构建产物脚本必须注册进 verify_all/ci.yml/tests 至少其一。"""
    va = VERIFY_ALL.read_text(encoding="utf-8")
    ci = CI.read_text(encoding="utf-8")
    tests_src = " ".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in (ROOT / "tests").glob("test_*.py") if p.is_file())
    for script in ("precompress_static.py", "build_tailwind_css.py", "build_echarts_custom.py"):
        assert script in va or script in ci or script in tests_src, \
            f"{script} 未注册进 verify_all / ci.yml / tests（A5：写了没接链 = 白写）"


def test_ci_runs_artifact_checks():
    """ci.yml 必须跑三个产物 --check（quality-matrix 与 nightly-full 各一）。"""
    ci = CI.read_text(encoding="utf-8")
    assert ci.count("3a) 前端产物一致性门禁") == 2, \
        "ci.yml 的 quality-matrix 与 nightly-full 都应含前端产物一致性步骤"
    assert "precompress_static.py --check" in ci
    assert "build_tailwind_css.py --check" in ci
    assert "build_echarts_custom.py --check" in ci
