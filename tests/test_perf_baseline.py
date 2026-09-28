"""perf_baseline 性能基线门禁测试（2026-09-28，性能/延迟探索落地）。

锁死门禁自身行为：防灾难性回归的语义、阈值下限（亚毫秒路径不因抖动必红）、
基线文件入库（CI 干净检出可用）、verify_all/ci.yml 注册契约。
"""
import scripts.perf_baseline as pb


def test_floor_applies_to_submillisecond_paths():
    """亚毫秒路径（K 线引擎 p50≈0ms）：阈值 = max(基线×余量, 下限)，
    否则实测任何波动都 > 0 → 门禁必然红（CI 抖动直接挂）。"""
    assert pb.FLOOR_MS >= 10.0
    base, tol = 0.0035, 4.0
    limit = max(base * tol, pb.FLOOR_MS)
    assert limit == pb.FLOOR_MS, "0.0035×4≈0.014 必须被下限兜底"
    assert limit > 0.0


def test_baseline_file_is_versioned_for_ci():
    """基线文件必须入库（.gitignore 有 !例外），否则 CI 干净检出无基线 → 门禁必红。"""
    gi = (pb.ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!.optim/perf_baseline.json" in gi, "基线文件必须入库（CI 干净检出需要）"


def test_gate_registered_in_verify_all_and_ci():
    """A5 教训延伸：脚本写了必须接链——perf_baseline 需在 verify_all 与 ci.yml 注册。"""
    va_src = (pb.ROOT / "scripts" / "verify_all.py").read_text(encoding="utf-8")
    assert "perf_baseline.py" in va_src, "verify_all.py 未注册 perf_baseline"
    ci = (pb.ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "perf_baseline.py" in ci, "ci.yml 未注册 perf_baseline"


def test_measure_returns_all_expected_paths():
    """测量覆盖回测双引擎 / compute_latest / K 线引擎四条关键路径。"""
    m = pb.measure()
    for key in ("backtest_engine_3000", "backtest_fast_3000",
                "compute_latest_3000", "evolve_batch_10x5000", "kline_engine_100"):
        assert key in m, f"缺测量项 {key}"
        assert m[key] >= 0.0
    assert m["backtest_engine_3000"] > 0.0, "3000 根回测不可能 0ms"
    assert m["compute_latest_3000"] > 0.0
    assert m["evolve_batch_10x5000"] > 0.0, "批量候选回测不可能 0ms"


def test_main_returns_one_without_baseline(monkeypatch, capsys):
    """无基线文件 → 门禁失败并提示先 --update（不静默通过）。"""
    monkeypatch.setattr(pb, "BASELINE_FILE", pb.ROOT / ".optim" / "_nonexistent_base.json")
    rc = pb.main(["--report"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "--update" in out, "必须提示校准命令"


def test_main_passes_within_tolerance(monkeypatch, capsys):
    """耗时在基线×余量内 → 通过（防灾难性回归语义，非脆弱毫秒断言）。"""
    fake = {"paths": {"backtest_engine_3000": 64.0, "backtest_fast_3000": 64.0,
                      "compute_latest_3000": 31.0, "evolve_batch_10x5000": 1100.0,
                      "kline_engine_100": 0.0},
            "tolerance": 4.0}
    monkeypatch.setattr(pb, "load_baseline", lambda: fake)
    monkeypatch.setattr(pb, "measure", lambda: {"backtest_engine_3000": 65.0,
                                                "backtest_fast_3000": 66.0,
                                                "compute_latest_3000": 32.0,
                                                "evolve_batch_10x5000": 1200.0,
                                                "kline_engine_100": 0.01})
    rc = pb.main(["--report"])
    assert rc == 0


def test_main_fails_on_order_of_magnitude_regression(monkeypatch, capsys):
    """O(n²) 化回归（耗时 ×10）必须拦下——这是门禁存在的意义。"""
    fake = {"paths": {"backtest_engine_3000": 64.0, "backtest_fast_3000": 64.0,
                      "compute_latest_3000": 31.0, "evolve_batch_10x5000": 1100.0,
                      "kline_engine_100": 0.0},
            "tolerance": 4.0}
    monkeypatch.setattr(pb, "load_baseline", lambda: fake)
    monkeypatch.setattr(pb, "measure", lambda: {"backtest_engine_3000": 900.0,
                                                "backtest_fast_3000": 64.0,
                                                "compute_latest_3000": 31.0,
                                                "evolve_batch_10x5000": 1100.0,
                                                "kline_engine_100": 0.0})
    rc = pb.main(["--report"])
    assert rc == 1
