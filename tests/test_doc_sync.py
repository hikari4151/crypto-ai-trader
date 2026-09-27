"""C5/A5：门禁一致性契约（check_doc_sync.py）回归。

覆盖：
- 目录 token 抽取
- 顶层包自检（新增包未登记 → 报漂移）
- ci.yml compileall / ruff 缺目录 → 报漂移
- verify_all.py 的 RUFF_DIRS / COMPILE_DIRS 与契约一致
- .coveragerc source / coverage_gate THRESHOLDS 覆盖全部被测模块
- 当前仓库真实状态通过（防 CI 与本机判定分叉）
"""
import scripts.check_doc_sync as cs


def test_extract_dirs_matches_tokens():
    text = "python -m compileall -q core engine ai web scripts backtest exchange factors net tests run.py"
    got = cs.extract_dirs(text)
    assert {"core", "engine", "ai", "web", "scripts", "backtest", "exchange",
            "factors", "net", "tests", "run.py"} <= got


def test_capture_returns_set_for_line():
    text = "step: run: python -m compileall -q core ai\nnext:"
    got = cs._capture(text, "python -m compileall -q")
    assert "core" in got and "ai" in got


def test_contract_includes_net():
    """A5：net/ 曾被整包漏出全部门禁，契约集合必须含 net。"""
    assert "net" in cs.CORE_DIRS
    assert "net" in cs.COVERED_DIRS


def test_verify_all_dirs_parsed_from_source():
    """verify_all.py 的目录常量必须能被解析（契约校验依赖它）。"""
    ruff_dirs, compile_dirs = cs._verify_all_dirs()
    assert "core" in ruff_dirs and "tests" in ruff_dirs
    assert "run.py" in compile_dirs and "tests" in compile_dirs


def test_coveragerc_and_gate_modules_parsed():
    src = cs._coveragerc_source()
    mods = cs._coverage_gate_modules()
    assert "core" in src and "net" in src
    assert "core" in mods and "net" in mods


def test_check_ok_when_all_consistent(tmp_path, monkeypatch):
    ci = tmp_path / "ci.yml"
    dirs = " ".join(sorted(cs.CORE_DIRS))
    ci.write_text(
        f"run: python -m compileall -q {dirs} tests run.py\n"
        f"run: python -m ruff check {dirs} tests\n"
        "run: python scripts/check_encoding.py\n",
        encoding="utf-8")
    redev = tmp_path / "REDEV_GUIDE.md"
    redev.write_text(f"compileall -q {dirs} run.py", encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redev)
    monkeypatch.setattr(cs, "discover_packages", lambda: set(cs.CORE_DIRS))
    monkeypatch.setattr(cs, "_verify_all_dirs",
                        lambda: (cs.CORE_DIRS | {"tests"}, cs.CORE_DIRS | {"tests", "run.py"}))
    assert cs.check() == []


def test_check_reports_missing_exchange_in_ci(tmp_path, monkeypatch):
    ci = tmp_path / "ci.yml"
    dirs = " ".join(sorted(cs.CORE_DIRS - {"exchange"}))
    ci.write_text(
        f"run: python -m compileall -q {dirs} tests run.py\n"
        f"run: python -m ruff check {dirs} tests\n",
        encoding="utf-8")
    redev = tmp_path / "REDEV_GUIDE.md"
    redev.write_text("compileall -q " + " ".join(sorted(cs.CORE_DIRS)) + " run.py\n",
                     encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redev)
    monkeypatch.setattr(cs, "discover_packages", lambda: set(cs.CORE_DIRS))
    monkeypatch.setattr(cs, "_verify_all_dirs",
                        lambda: (cs.CORE_DIRS | {"tests"}, cs.CORE_DIRS | {"tests", "run.py"}))
    issues = cs.check()
    assert any("compileall 缺目录" in i and "exchange" in i for i in issues)
    assert any("ruff 缺目录" in i and "exchange" in i for i in issues)


def test_check_reports_unregistered_new_package(tmp_path, monkeypatch):
    """新增顶层包没进契约 → 报漂移（A5 让 net/ 漏门的根因）。"""
    ci = tmp_path / "ci.yml"
    dirs = " ".join(sorted(cs.CORE_DIRS))
    ci.write_text(
        f"run: python -m compileall -q {dirs} tests run.py\n"
        f"run: python -m ruff check {dirs} tests\n",
        encoding="utf-8")
    redev = tmp_path / "REDEV_GUIDE.md"
    redev.write_text(f"compileall -q {dirs} run.py", encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redev)
    monkeypatch.setattr(cs, "discover_packages", lambda: set(cs.CORE_DIRS) | {"brand_new_pkg"})
    monkeypatch.setattr(cs, "_verify_all_dirs",
                        lambda: (cs.CORE_DIRS | {"tests"}, cs.CORE_DIRS | {"tests", "run.py"}))
    issues = cs.check()
    assert any("未纳入门禁契约" in i and "brand_new_pkg" in i for i in issues)


def test_check_reports_verify_all_drift(tmp_path, monkeypatch):
    """verify_all.py 漏目录也要被抓（此前只校验 ci.yml / REDEV）。"""
    ci = tmp_path / "ci.yml"
    dirs = " ".join(sorted(cs.CORE_DIRS))
    ci.write_text(
        f"run: python -m compileall -q {dirs} tests run.py\n"
        f"run: python -m ruff check {dirs} tests\n",
        encoding="utf-8")
    redev = tmp_path / "REDEV_GUIDE.md"
    redev.write_text(f"compileall -q {dirs} run.py", encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redev)
    monkeypatch.setattr(cs, "discover_packages", lambda: set(cs.CORE_DIRS))
    monkeypatch.setattr(cs, "_verify_all_dirs",
                        lambda: (cs.CORE_DIRS - {"net"}, cs.CORE_DIRS | {"tests", "run.py"}))
    issues = cs.check()
    assert any("verify_all.py RUFF_DIRS 缺目录" in i and "net" in i for i in issues)


def test_repo_state_is_consistent():
    """当前仓库必须自洽（CI/本地同一判定，防只在 CI 挂）。"""
    assert cs.check() == []


def test_main_returns_zero_when_ok(monkeypatch):
    monkeypatch.setattr(cs, "check", lambda: [])
    assert cs.main() == 0


def test_main_returns_one_on_drift(monkeypatch, capsys):
    monkeypatch.setattr(cs, "check", lambda: ["ci.yml compileall 缺目录: ['exchange']"])
    assert cs.main() == 1
    assert "[docsync] FAILED" in capsys.readouterr().out

def test_encoding_gate_registered_in_contract():
    """F1/F2 编码门禁必须同时注册在 ci.yml 与 verify_all（A5 教训：脚本写了≠接进链）。"""
    issues = [i for i in cs.check() if "check_encoding" in i]
    assert issues == [], f"编码门禁未注册: {issues}"
    ci = cs.CI.read_text(encoding="utf-8")
    va = cs.VERIFY_ALL.read_text(encoding="utf-8")
    assert "check_encoding.py" in ci
    assert "check_encoding.py" in va
