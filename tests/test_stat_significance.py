"""统计显著性测试：Deflated Sharpe Ratio + Bootstrap 置信区间（Bailey & López de Prado）"""
import numpy as np
import pytest
from backtest.metrics import (
    _norm_cdf,
    _norm_ppf,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    equity_returns,
)


class TestNormQuantile:
    def test_ppf_known_values(self):
        assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert _norm_ppf(0.05) == pytest.approx(-1.644854, abs=1e-5)

    def test_cdf_known_values(self):
        assert _norm_cdf(1.96) == pytest.approx(0.975002, abs=1e-5)
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_ppf_invalid_p(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.5)


class TestDeflatedSharpe:
    def test_null_returns_half_with_one_trial(self):
        """零均值收益 + 1 次试验 → DSR ≈ 0.5（等价 PSR vs 0）"""
        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 0.01, 3000)
        r = r - r.mean()
        d = deflated_sharpe_ratio(r, trials=1)
        assert d["dsr"] == pytest.approx(0.5, abs=0.02)

    def test_multi_trial_deflates(self):
        """同样收益，试验数越多 DSR 越低（多重测试惩罚单调递增）"""
        rng = np.random.default_rng(7)
        r = rng.normal(0.001, 0.01, 3000)
        d1 = deflated_sharpe_ratio(r, trials=1)
        d10 = deflated_sharpe_ratio(r, trials=10)
        d500 = deflated_sharpe_ratio(r, trials=500)
        assert d500["dsr"] < d10["dsr"] < d1["dsr"]

    def test_strong_edge_passes_even_with_many_trials(self):
        """真实强优势 + 500 次搜索后仍显著"""
        rng = np.random.default_rng(7)
        r = rng.normal(0.004, 0.008, 3000)
        d = deflated_sharpe_ratio(r, trials=500)
        assert d["dsr"] > 0.95

    def test_pure_noise_with_many_trials_fails(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 0.01, 3000)
        r = r - r.mean()
        d = deflated_sharpe_ratio(r, trials=500)
        assert d["dsr"] < 0.05

    def test_trial_sharpes_path(self):
        """提供候选 SR 列表时用跨试验方差估计 SR₀"""
        rng = np.random.default_rng(7)
        r = rng.normal(0.002, 0.01, 2000)
        ts = list(rng.normal(0.0, 0.06, 20)) + [0.25]
        d = deflated_sharpe_ratio(r, trials=21, trial_sharpes=ts)
        assert 0.0 < d["dsr"] <= 1.0
        assert d["sr0"] > 0.0

    def test_short_series_returns_none(self):
        assert deflated_sharpe_ratio([0.01] * 10, trials=5)["dsr"] is None

    def test_constant_series_returns_none(self):
        """零方差（全为同一值）→ 无法定义 SR"""
        assert deflated_sharpe_ratio([0.01] * 100)["dsr"] is None


class TestBootstrapCI:
    def test_positive_edge_ci_excludes_zero(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.003, 0.01, 2000)
        b = bootstrap_sharpe_ci(r, n_boot=800, periods_per_year=365)
        assert b["sharpe_ci"][0] > 0
        assert b["p_sharpe_pos"] > 0.95

    def test_null_ci_contains_zero(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.0, 0.01, 2000)
        r = r - r.mean()
        b = bootstrap_sharpe_ci(r, n_boot=800, periods_per_year=365)
        assert b["sharpe_ci"][0] < 0 < b["sharpe_ci"][1]
        assert 0.0 < b["p_sharpe_pos"] < 1.0

    def test_reproducible_seed(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.001, 0.01, 500)
        b1 = bootstrap_sharpe_ci(r, n_boot=300, seed=42)
        b2 = bootstrap_sharpe_ci(r, n_boot=300, seed=42)
        assert b1 == b2

    def test_ret_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.001, 0.01, 1000)
        b = bootstrap_sharpe_ci(r, n_boot=500)
        point = float(np.prod(1 + r) - 1)
        assert b["ret_ci"][0] <= point <= b["ret_ci"][1] * 1.5 + 1e-6

    def test_short_series_returns_none(self):
        b = bootstrap_sharpe_ci([0.01] * 10)
        assert b["sharpe_ci"] is None and b["p_sharpe_pos"] is None


class TestEquityReturns:
    def test_basic(self):
        er = equity_returns([100.0, 110.0, 99.0])
        assert er[0] == pytest.approx(0.1)
        assert er[1] == pytest.approx(-0.1)

    def test_too_short(self):
        assert len(equity_returns([100.0])) == 0


class TestOverfitReportSignificance:
    """detect_overfit 集成：报告必须带 DSR/Bootstrap 字段"""

    def test_report_contains_dsr_and_ci(self):
        from backtest.data_loader import generate_demo
        from backtest.overfit import OverfitConfig, detect_overfit

        df = generate_demo(timeframe="1h")
        cfg = OverfitConfig(symbol="BTC/USDT", timeframe="1h", n_folds=3)
        report = detect_overfit(df, "dual_ma", {"fast": 10, "slow": 30}, cfg)
        assert report.dsr is not None
        assert report.dsr_trials >= 1
        assert report.sharpe_ci is not None and len(report.sharpe_ci) == 2
        assert report.p_sharpe_pos is not None
        assert report.verdict in ("通过", "疑似过拟合", "严重过拟合", "无法判定")


class TestOverfitLowEvidenceVerdict:
    """回归：AI 设计策略在验证段 OOS 交易极少/无交易时，统计证据不足，
    不应被误判为“严重过拟合”而硬性拒绝注册（无法应用、不入库）。

    见 ai/strategy_designer._guard_ai_strategy：verdict == "严重过拟合" 即拒绝注册。
    证据不足以支撑该结论时应降级为“无法判定”。
    """

    def _report(self, avg_oos_trades: float, oos_ret: float = -0.01,
                pbo: float = 0.6, dsr: float = 0.3, n_folds: int = 3) -> "OverfitReport":
        from backtest.overfit import FoldResult, OverfitReport
        # 构造 n_folds 折，每折 oos_trades 平均后 ≈ avg_oos_trades
        # （此前用 int(round(avg/n_folds)) 每折取整，avg=8/n_folds=3 → 每折3笔，
        #  真实平均只有3笔，与"avg=8=交易充分"的用例意图不符，已修正为按总量分配）
        total = int(round(avg_oos_trades * n_folds))
        base, rem = divmod(total, n_folds)
        trades_per_fold = [base + 1 if i < rem else base for i in range(n_folds)]
        folds = [FoldResult(fold=i + 1, is_start="2024-01-01 00:00:00+00:00",
                            is_end="2024-06-01 00:00:00+00:00",
                            oos_start="2024-06-01 00:00:00+00:00",
                            oos_end="2024-12-01 00:00:00+00:00",
                            is_ret=0.05, oos_ret=oos_ret / n_folds,
                            oos_sharpe=-0.5, oos_drawdown=0.1,
                            oos_trades=trades_per_fold[i]) for i in range(n_folds)]
        return OverfitReport(
            verdict="未判定", score=5.0,
            flags=[], folds=folds,
            is_ret=0.05, oos_ret=oos_ret, decay=1.2, pbo=pbo,
            oos_sharpe=-0.5, oos_win_rate=0.0, stability=0.0,
            n_folds=n_folds, dsr=dsr, dsr_trials=9, sharpe_ci=[-10.0, -0.1], ret_ci=None,
            p_sharpe_pos=0.1,
        )

    def test_no_trades_is_insufficient_not_severe_overfit(self):
        """OOS 完全无交易 → 应判“无法判定”，绝不判“严重过拟合”。
        （旧逻辑仅在 oos_ret==0.0 且无交易时放行；本轮回归曾因单折产生一两笔
        微小负收益使 oos_ret 略非 0，落入“严重过拟合”被硬拒。）"""
        from backtest.overfit import OverfitReport, _score_and_verdict
        rep = self._report(avg_oos_trades=0.0, oos_ret=-1e-4)
        _score_and_verdict(rep)
        assert rep.verdict == "无法判定", f"got {rep.verdict}"

    def test_very_few_trades_is_insufficient_not_severe_overfit(self):
        """OOS 平均交易 1-2 笔（<3）→ 统计证据不足，判“无法判定”而非“严重过拟合”。"""
        from backtest.overfit import _score_and_verdict
        rep = self._report(avg_oos_trades=1.5, oos_ret=-0.008)
        _score_and_verdict(rep)
        assert rep.verdict == "无法判定", f"got {rep.verdict}"

    def test_low_evidence_negative_is_inconclusive_not_severe_overfit(self):
        """3≤avg<8 且收益为负 → “无法判定”而非“严重过拟合”。

        校准：内置默认策略（dual_ma/factor_signal/grid）在近 7 个月真实 1h
        K线上 OOS 收益为负时 avg_oos_trades 仅 3~12 笔，此前全被判“严重过拟合”，
        其中低频段（3~7 笔）的负收益是成交噪声/行情不适配，不是参数过拟合。"""
        from backtest.overfit import _score_and_verdict
        rep = self._report(avg_oos_trades=4.0, oos_ret=-0.02, pbo=0.8, dsr=0.1)
        _score_and_verdict(rep)
        assert rep.verdict == "无法判定", f"got {rep.verdict}"

    def test_enough_trades_negative_still_severe_overfit(self):
        """OOS 交易充分且收益转负、PBO 高 → 保留“严重过拟合”拦截。"""
        from backtest.overfit import OverfitReport, _score_and_verdict
        rep = self._report(avg_oos_trades=8.0, oos_ret=-0.04, pbo=0.8, dsr=0.1)
        _score_and_verdict(rep)
        assert rep.verdict == "严重过拟合", f"got {rep.verdict}"

    def test_enough_trades_positive_passes(self):
        """OOS 交易充分且收益为正 → 判“通过”（或至少不是严重过拟合）。"""
        from backtest.overfit import _score_and_verdict
        rep = self._report(avg_oos_trades=8.0, oos_ret=0.06, pbo=0.1, dsr=0.98)
        _score_and_verdict(rep)
        assert rep.verdict != "严重过拟合", f"got {rep.verdict}"
