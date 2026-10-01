"""P1-B1：verify_all 统一验证链回归。"""
import subprocess

import scripts.verify_all as va


def test_build_steps_fast_has_core_gates():
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    labels = [label for label, _ in steps]
    assert labels == ["check_lock", "ruff", "check_redlines", "mypy_pilot",
                      "validate_config", "check_doc_sync", "check_js",
                      "check_encoding", "build_frontend", "compileall",
                      "check_precompressed", "check_tailwind_artifact",
                      "check_echarts_artifact", "perf_baseline"]


def test_build_steps_full_has_pytest_and_backtest():
    steps = va.build_steps(skip_pytest=False, skip_backtest=False)
    labels = [label for label, _ in steps]
    assert "pytest_cov" in labels
    assert "coverage_json" in labels and "coverage_gate" in labels
    assert "backtest_baseline" in labels


def test_encoding_gate_registered():
    """F1/F2 编码红线门禁必须注册在 fast 链里（否则又是"脚本写了没接链"）。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    enc = dict(steps)["check_encoding"]
    assert any("check_encoding.py" in c for c in enc)


def test_redlines_gate_registered():
    """grep 型红线门禁（B8/E2/E7）必须注册在 fast 链里。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True)
    red = dict(steps)["check_redlines"]
    assert any("check_redlines.py" in c for c in red)


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
    assert len(calls) == 14  # 基础步 + check_encoding/check_redlines + 3 产物校验 + perf_baseline
    assert calls[1][0] == va.PY  # ruff 第一步首元素是 python
    assert calls[3][-1] == "mypy"
    # 注意：c 是命令列表，须用 " ".join(c) 做子串（in 对 list 是相等比较，非子串）
    flattened = [" ".join(c) for c in calls]
    assert any("check_doc_sync.py" in p for p in flattened)
    assert any("check_encoding.py" in p for p in flattened)
    assert any("precompress_static.py" in p for p in flattened)  # 产物校验必须在链内


def test_staged_mode_only_checks_staged_files():
    """第十六轮：--staged 模式 ruff 只查暂存文件、未碰 index.html 时跳过 E7。"""
    steps = va.build_steps(skip_pytest=True, skip_backtest=True, staged=True)
    labels = [label for label, _ in steps]
    assert "ruff(staged)" in labels, "staged 模式应有 ruff(staged) 步骤"
    assert "check_redlines(skip-e7)" in labels, "未碰 index.html 时须跳过 E7 全量基线"


def test_main_returns_one_when_step_fails(monkeypatch, capsys):
    def fake_run(cmd):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(va.subprocess, "run", fake_run)
    rc = va.main(["--skip-pytest", "--skip-backtest"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[verify] FAILED" in out
