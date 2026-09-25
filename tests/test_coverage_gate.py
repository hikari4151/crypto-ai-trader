"""覆盖率门禁回归：配置存在、阈值合理、CI 已接入。"""
import json
from pathlib import Path

import scripts.coverage_gate as cg

ROOT = Path(__file__).resolve().parent.parent


def test_coveragerc_has_fail_under():
    cfg = (ROOT / ".coveragerc").read_text(encoding="utf-8")
    assert "fail_under" in cfg and "65" in cfg, "覆盖率门槛应配置为 65"
    assert "source" in cfg, "应限定核心模块 source"


def test_coveragerc_omits_tests_and_optim():
    cfg = (ROOT / ".coveragerc").read_text(encoding="utf-8")
    assert "*/tests/*" in cfg and "*/.optim/*" in cfg, "测试/研究目录应排除在统计外"


def test_ci_runs_coverage_gate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--cov" in ci and "--cov-fail-under=65" in ci, "CI 应带覆盖率门禁"


def test_lock_contains_pytest_cov():
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "pytest-cov==" in lock and "coverage==" in lock, "lock 应含覆盖率依赖"


# ---- C1 模块级覆盖率门禁（scripts/coverage_gate.py）----
def _module_json(files: dict[str, tuple[float, int]]) -> str:
    payload = {"files": {}}
    for path, (pc, n) in files.items():
        payload["files"][path] = {"summary": {"percent_covered": pc, "num_statements": n}}
    return json.dumps(payload)


def test_c1_aggregate_weighted():
    text = _module_json({
        "core/a.py": (50.0, 10),
        "core/b.py": (100.0, 10),
        "web/x.py": (100.0, 5),
    })
    files = cg.parse_module_coverage(text)
    agg = cg.aggregate(files)
    assert abs(agg["core"] - 75.0) < 1e-9
    assert agg["web"] == 100.0


def test_c1_group_by_unknown_is_other():
    assert cg.group_by("vendor/foo.py") == "_other"
    assert cg.group_by("core/bus.py") == "core"


def test_c1_check_reports_below_threshold(monkeypatch):
    monkeypatch.setattr(cg, "THRESHOLDS", {"core": 80.0})
    issues = cg.check({"core": 70.0, "web": 99.0})
    assert any("core" in i and "70.0" in i for i in issues)
    assert len(issues) == 1


def test_c1_check_ok(monkeypatch):
    monkeypatch.setattr(cg, "THRESHOLDS", {"core": 80.0, "web": 40.0})
    assert cg.check({"core": 85.0, "web": 45.0}) == []


def test_c1_main_fails_below_threshold(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cg, "THRESHOLDS", {"core": 95.0})
    jf = tmp_path / "cov.json"
    jf.write_text(_module_json({"core/a.py": (90.0, 10)}), encoding="utf-8")
    assert cg.main(["--json", str(jf)]) == 1
    assert "[coverage] FAILED" in capsys.readouterr().out
