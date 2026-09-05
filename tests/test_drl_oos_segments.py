"""多段 OOS 评估必须按段统计，不能拼接重置资金的权益曲线。"""

import numpy as np
import pytest

from drl.agent import _aggregate_oos_segment_metrics, _equity_drawdown, _equity_sharpe


def test_segment_metrics_are_aggregated_without_boundary_jump():
    segments = [[100.0, 110.0, 105.0], [100.0, 90.0, 95.0], [100.0, 102.0, 101.0]]
    report = _aggregate_oos_segment_metrics(segments)

    assert report["sharpe"] == pytest.approx(
        float(np.median([_equity_sharpe(s) for s in segments]))
    )
    assert report["max_drawdown"] == pytest.approx(
        float(np.median([_equity_drawdown(s) for s in segments]))
    )
    assert report["sharpe"] != pytest.approx(
        _equity_sharpe([value for segment in segments for value in segment])
    )


def test_empty_and_short_segments_have_zero_metrics():
    assert _aggregate_oos_segment_metrics([[], [100.0, 101.0]]) == {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }
