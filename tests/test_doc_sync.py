"""C5：文档同步机制（check_doc_sync.py）回归。"""
import scripts.check_doc_sync as cs


def test_extract_dirs_matches_tokens():
    text = "python -m compileall -q core engine ai web scripts backtest exchange factors tests run.py"
    got = cs.extract_dirs(text)
    assert {"core", "engine", "ai", "web", "scripts", "backtest", "exchange",
            "factors", "tests", "run.py"} <= got


def test_capture_returns_set_for_line():
    text = "step: run: python -m compileall -q core ai\nnext:"
    got = cs._capture(text, "python -m compileall -q")
    assert "core" in got and "ai" in got


def test_check_ok_when_all_consistent(tmp_path, monkeypatch):
    ci = tmp_path / "ci.yml"
    ci.write_text(
        "run: python -m compileall -q core engine ai web scripts backtest strategies indicators config drl exchange factors tests run.py\n"
        "run: python -m ruff check core/ engine/ ai/ web/ scripts/ backtest/ strategies/ indicators/ config/ drl/ exchange/ factors/\n",
        encoding="utf-8")
    redesign = tmp_path / "REDEV_GUIDE.md"
    redesign.write_text("compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py",
                        encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redesign)
    assert cs.check() == []


def test_check_reports_missing_exchange_in_ci(tmp_path, monkeypatch):
    ci = tmp_path / "ci.yml"
    ci.write_text(
        "run: python -m compileall -q core engine ai web scripts backtest strategies indicators config drl factors tests run.py\n"
        "run: python -m ruff check core/ engine/ ai/ web/ scripts/ backtest/ strategies/ indicators/ config/ drl/ factors/\n",
        encoding="utf-8")
    redesign = tmp_path / "REDEV_GUIDE.md"
    redesign.write_text("compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py\n",
                        encoding="utf-8")
    monkeypatch.setattr(cs, "CI", ci)
    monkeypatch.setattr(cs, "REDEV", redesign)
    issues = cs.check()
    assert any("compileall 缺目录" in i and "exchange" in i for i in issues)
    assert any("ruff 缺目录" in i and "exchange" in i for i in issues)


def test_main_returns_zero_when_ok(monkeypatch):
    monkeypatch.setattr(cs, "check", lambda: [])
    assert cs.main() == 0


def test_main_returns_one_on_drift(monkeypatch, capsys):
    monkeypatch.setattr(cs, "check", lambda: ["ci.yml compileall 缺目录: ['exchange']"])
    assert cs.main() == 1
    assert "[docsync] FAILED" in capsys.readouterr().out
