"""TDD：ETH/5m 同品种 walk-forward 多折评估（固定 DRL 模型 × 时间窗口）。

背景（用户选"2"）：策略只做 ETH/5m，不希望跨品种门污染/误杀；
改用"换一段历史也赚"验证——把 ETH/5m 历史切成多折，用已部署的
strategy_drl 最佳模型在每折（互不重叠的时间段）上 greedy 评估，
输出每折收益/成交数/最大回撤，并按方向一致性与样本量给 PASS/FAIL/INCONCLUSIVE。
"""
import numpy as np
import pytest

from backtest.data_loader import generate_demo
from scripts.walkforward_rl_evolve import (
    build_factor_column,
    expanding_fold_bounds,
    summarize_folds,
)


def test_expanding_fold_bounds_covers_tail_without_overlap():
    """n=1000, warmup=100, 4 折 → 每折不重叠且覆盖到数据末尾。"""
    bounds = expanding_fold_bounds(n=1000, warmup=100, n_folds=4)
    assert bounds[0][0] == 100
    assert bounds[-1][1] == 1000
    for (s1, e1), (s2, e2) in zip(bounds, bounds[1:]):
        assert e1 == s2  # 相邻折首尾相接、无重叠


def test_expanding_fold_bounds_rejects_too_little_data():
    with pytest.raises(ValueError, match="数据不足"):
        expanding_fold_bounds(n=50, warmup=100, n_folds=4)


def _folds(rets, trades):
    return [{"fold": i, "oos_ret": r, "oos_trades": t}
            for i, (r, t) in enumerate(zip(rets, trades))]


def test_summarize_verdict_pass_on_consistency():
    """≥75% 折为正、中位数>0、各折成交充足 → PASS。"""
    folds = _folds([0.02, 0.01, 0.03, -0.01], [10, 12, 9, 11])
    out = summarize_folds(folds, min_trades=8)
    assert out["verdict"] == "PASS"
    assert out["positive_folds"] == 3
    assert out["n_folds"] == 4


def test_summarize_verdict_fail_on_low_win_rate():
    folds = _folds([0.02, -0.01, -0.02, 0.01], [10, 12, 9, 11])
    out = summarize_folds(folds, min_trades=8)
    assert out["verdict"] == "FAIL"


def test_summarize_verdict_inconclusive_when_trades_insufficient():
    """正收益但某折成交样本不足 → 证据不足，判 INCONCLUSIVE 而非 PASS。"""
    folds = _folds([0.02, 0.01, 0.03, -0.01], [10, 12, 3, 11])
    out = summarize_folds(folds, min_trades=8)
    assert out["verdict"] == "INCONCLUSIVE"


def test_summarize_empty_is_inconclusive():
    out = summarize_folds([], min_trades=8)
    assert out["verdict"] == "INCONCLUSIVE"


def test_build_factor_column_length_and_finite():
    """级联因子列按当轮 df 对齐、标准化后长度一致且无 NaN。"""
    df = generate_demo(n=300, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    weights = {"mom_7": -0.05, "bb_width": 0.06}
    col = build_factor_column(df, weights, mu=0.1, sd=2.0)
    assert col is not None
    assert len(col) == len(df)
    assert col.dtype == float
    assert not col[~np.isfinite(col)].any()


def test_build_factor_column_returns_none_without_weights():
    df = generate_demo(n=200, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    assert build_factor_column(df, {}, mu=0.0, sd=1.0) is None
