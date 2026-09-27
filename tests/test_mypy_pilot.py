"""C2 类型检查试点（pyproject.toml [tool.mypy]）回归。

背景：mypy 试点范围一度在 3 处各写一份（pyproject / verify_all / ci.yml），
扩容时必然漂移。2026-09-27 收敛为 pyproject.toml 单一事实来源，
verify_all 与 CI 都只调 `python -m mypy`，由本文件锁住这个约定与范围。
"""
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _mypy_cfg() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["tool"]["mypy"]


def test_mypy_section_exists():
    cfg = _mypy_cfg()
    assert cfg["python_version"] == "3.13"
    assert cfg["follow_imports"] == "skip"
    assert cfg["ignore_missing_imports"] is True
    assert cfg["warn_unused_ignores"] is True


def test_mypy_scope_includes_core_and_net():
    """C2 扩容：core/ 与 net/ 必须整包纳入（net 曾被全门禁漏掉）。"""
    files = _mypy_cfg()["files"]
    assert "core/" in files, "core/ 应整包纳入试点"
    assert "net/" in files, "net/ 应整包纳入试点"


def test_mypy_scope_keeps_original_pilot_files():
    files = _mypy_cfg()["files"]
    for f in ("backtest/schema.py", "exchange/symbols.py"):
        assert f in files, f"{f} 是原始试点文件，不得回退"


def test_mypy_scope_avoids_duplicate_module():
    """不得同时列整包与包内文件（mypy 会报 Duplicate module named ...）。"""
    files = _mypy_cfg()["files"]
    pkgs = [f for f in files if f.endswith("/")]
    for f in files:
        if f.endswith("/"):
            continue
        assert not any(f.startswith(p) for p in pkgs), f"{f} 与整包条目重复"


def test_ci_and_verify_all_use_single_source():
    """CI 与 verify_all 都只跑裸 `mypy`，不得再写死文件清单。"""
    ci = CI.read_text(encoding="utf-8")
    assert "python -m mypy\n" in ci or "python -m mypy\r\n" in ci, "CI 应只运行裸 mypy"
    assert "python -m mypy core/events.py" not in ci, "CI 不应再写死文件清单"

    import scripts.verify_all as va
    mypy_cmd = dict(va.build_steps(skip_pytest=True, skip_backtest=True))["mypy_pilot"]
    assert mypy_cmd[-1] == "mypy"


def test_pilot_scope_is_type_clean():
    """试点范围必须零错误（本测试即门禁的本地等价物）。"""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "mypy"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    assert r.returncode == 0, f"mypy 试点范围有类型错误:\n{r.stdout}\n{r.stderr}"
    assert "Success" in r.stdout
