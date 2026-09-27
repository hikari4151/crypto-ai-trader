"""P1-B1：verify_all 统一验证链回归。"""
import subprocess

import scripts.verify_all as va


def test_build_steps_fast_has_core_gates():
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    labels = [label for label, _ in steps]
    assert labels == ["check_lock", "ruff", "mypy_pilot", "validate_config",
                      "check_doc_sync", "check_js", "build_frontend", "compileall"]


def test_build_steps_full_has_pytest_and_backtest():
    steps = va.build_steps(skip_pytest=False, skip_backtest=False)
    labels = [label for label, _ in steps]
    assert "pytest_cov" in labels
    assert "coverage_json" in labels and "coverage_gate" in labels
    assert "backtest_baseline" in labels


def test_ruff_and_compile_include_exchange_factors_tests():
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    ruff_cmd = dict(steps)["ruff"]
    assert "exchange/" in ruff_cmd and "factors/" in ruff_cmd
    compile_cmd = dict(steps)["compileall"]
    assert "exchange" in compile_cmd and "factors" in compile_cmd and "tests" in compile_cmd


def test_ruff_and_compile_include_net():
    """A5：net/ 代理包曾被 ruff/compileall/覆盖率三处同时漏掉，必须有回归。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    assert "net/" in dict(steps)["ruff"]
    assert "net" in dict(steps)["compileall"]


def test_mypy_range_comes_from_pyproject():
    """C2：mypy 试点范围只在 pyproject.toml 声明，命令不得再写死文件清单。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    mypy_cmd = dict(steps)["mypy_pilot"]
    assert mypy_cmd[-1] == "mypy", f"mypy 步骤不应传文件参数: {mypy_cmd}"


def test_build_frontend_check_only_locally():
    """本地链用宽松 check-only（正在编辑 index.html 时不拦人）；CI 才加 --require-manifest。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    fe = dict(steps)["build_frontend"]
    assert "--check-only" in fe
    assert "--require-manifest" not in fe, "本地链不应要求清单已同步（编辑中会误拦）"
    ci = (va.ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--check-only --require-manifest" in ci, "CI 必须校验清单漂移"


def test_main_returns_zero_when_all_steps_pass(monkeypatch, capsys):
    calls: list[list[str]] = []

    def fake_run(cmd):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(va.subprocess, "run", fake_run)
    rc = va.main(["--skip-pytest", "--skip-backtest"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[verify] ALL PASS" in out
    assert len(calls) == 8
    assert calls[1][0] == va.PY  # ruff 第一步首元素是 python
    assert calls[2][-1] == "mypy"
    assert any("check_doc_sync.py" in c for c in calls[4])


def test_main_returns_one_when_step_fails(monkeypatch, capsys):
    def fake_run(cmd):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(va.subprocess, "run", fake_run)
    rc = va.main(["--skip-pytest", "--skip-backtest"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[verify] FAILED" in out
