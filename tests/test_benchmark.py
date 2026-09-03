"""基准对比测试：compute_benchmark + 引擎集成"""
import numpy as np
import pytest
from backtest.metrics import compute_benchmark, compute_metrics
from backtest.data_loader import generate_demo
from backtest.engine import BacktestConfig, run_backtest
from backtest.fast_engine import run_backtest_fast


class TestComputeBenchmark:
    def test_basic_benchmark(self):
        """closes=[1,2,4], start_cash=10000 → buy_hold_ret=3.0, bench_equity_curve=[10000,20000,40000]"""
        closes = np.array([1., 2., 4.])
        r = compute_benchmark(closes, 10000.0)
        assert r["buy_hold_ret"] == pytest.approx(3.0, rel=1e-6)
        assert len(r["bench_equity_curve"]) == 3
        assert r["bench_equity_curve"] == [10000.0, 20000.0, 40000.0]
        assert r["excess_return"] == 0.0
        assert r["information_ratio"] == 0.0
        assert r["excess_max_drawdown"] == 0.0

    def test_short_series(self):
        """少于2个点返回占位零值"""
        closes = np.array([100.0])
        r = compute_benchmark(closes, 10000.0)
        assert r["buy_hold_ret"] == 0.0
        assert len(r["bench_equity_curve"]) == 1
        assert r["bench_equity_curve"][0] == 10000.0

    def test_empty_closes(self):
        """空数组不抛异常"""
        closes = np.array([])
        r = compute_benchmark(closes, 10000.0)
        assert r["buy_hold_ret"] == 0.0
        assert len(r["bench_equity_curve"]) == 1

    def test_zero_start_price(self):
        """首价格为0时返回占位零值"""
        closes = np.array([0., 1., 2.])
        r = compute_benchmark(closes, 10000.0)
        assert r["buy_hold_ret"] == 0.0

    def test_down_market(self):
        """下跌市场基准收益为负"""
        closes = np.array([100., 90., 80.])
        r = compute_benchmark(closes, 10000.0)
        assert r["buy_hold_ret"] == pytest.approx(-0.2, rel=1e-6)
        assert r["bench_equity_curve"][-1] == pytest.approx(8000.0, rel=1e-4)


class TestMetricsIntegration:
    """compute_metrics 新增 closes 参数"""

    def test_compute_metrics_without_closes_backward_compat(self):
        """不传 closes 时返回原结构，不含 benchmark"""
        equity = [10000.0, 10100.0, 10200.0]
        trades = []
        m = compute_metrics(equity, trades, "1h", 10000.0)
        assert "total_return" in m
        assert "benchmark" not in m

    def test_compute_metrics_with_closes(self):
        """传 closes 时返回含 benchmark 键"""
        equity = [10000.0, 10100.0, 10200.0]
        closes = np.array([100., 101., 102.])
        trades = []
        m = compute_metrics(equity, trades, "1h", 10000.0, closes=closes)
        assert "benchmark" in m
        assert "buy_hold_ret" in m["benchmark"]
        assert "excess_return" in m["benchmark"]
        assert "information_ratio" in m["benchmark"]
        assert "excess_max_drawdown" in m["benchmark"]
        assert "bench_equity_curve" in m["benchmark"]
        assert len(m["benchmark"]["bench_equity_curve"]) == 3
        # 上涨行情 benchmark 应>0
        assert m["benchmark"]["buy_hold_ret"] > 0


class TestEngineIntegration:
    """引擎集成：回测结果含 benchmark 键"""

    @pytest.fixture(scope="class")
    @classmethod
    def demo_df(cls):
        return generate_demo(timeframe="1h", n=1500)

    def test_engine_result_has_benchmark(self, demo_df):
        """run_backtest 结果含 benchmark 键"""
        cfg = BacktestConfig(strategy_name="dual_ma")
        r = run_backtest(demo_df, cfg)
        assert "benchmark" in r
        assert "benchmark" in r["metrics"]
        assert "buy_hold_ret" in r["benchmark"]
        assert "bench_equity_curve" in r["benchmark"]
        assert len(r["benchmark"]["bench_equity_curve"]) == len(r["equity_curve"])

    def test_fast_engine_result_has_benchmark(self, demo_df):
        """run_backtest_fast 结果含 benchmark 键"""
        cfg = BacktestConfig(strategy_name="dual_ma")
        r = run_backtest_fast(demo_df, cfg)
        assert "benchmark" in r
        assert "benchmark" in r["metrics"]
        assert "buy_hold_ret" in r["benchmark"]
        assert "bench_equity_curve" in r["benchmark"]
        assert len(r["benchmark"]["bench_equity_curve"]) == len(r["equity_curve"])

    def test_benchmark_curve_starts_with_start_cash(self, demo_df):
        """bench_equity_curve 首点为 start_cash"""
        cfg = BacktestConfig(strategy_name="dual_ma", start_cash=50000.0)
        r = run_backtest(demo_df, cfg)
        assert r["benchmark"]["bench_equity_curve"][0] == 50000.0

    def test_existing_metrics_keys_unchanged(self, demo_df):
        """既有 metrics 键值不受 benchmark 影响"""
        cfg = BacktestConfig(strategy_name="dual_ma")
        r = run_backtest(demo_df, cfg)
        expected_keys = {"total_return", "annual_return", "max_drawdown", "sharpe",
                         "win_rate", "profit_factor", "payoff_ratio", "total_trades",
                         "avg_win", "avg_loss", "final_equity"}
        assert expected_keys.issubset(r["metrics"].keys())

    def test_dual_engines_benchmark_consistent(self, demo_df):
        """两引擎的 benchmark 一致"""
        cfg = BacktestConfig(strategy_name="dual_ma")
        r_event = run_backtest(demo_df, cfg)
        r_fast = run_backtest_fast(demo_df, cfg)
        b1 = r_event["benchmark"]
        b2 = r_fast["benchmark"]
        assert b1["buy_hold_ret"] == pytest.approx(b2["buy_hold_ret"], rel=1e-6)
        assert b1["excess_return"] == pytest.approx(b2["excess_return"], rel=1e-6)
        assert b1["information_ratio"] == pytest.approx(b2["information_ratio"], rel=1e-4)
        assert b1["excess_max_drawdown"] == pytest.approx(b2["excess_max_drawdown"], rel=1e-6)
        assert len(b1["bench_equity_curve"]) == len(b2["bench_equity_curve"])