"""E-② 回测结果 Schema 版本化测试（2026-09-24）。"""
import pytest
from pydantic import ValidationError

from backtest.schema import SCHEMA_VERSION, BacktestMetrics, validate_backtest_result


def _valid_result() -> dict:
    return {
        "symbol": "ETH/USDT", "timeframe": "5m", "strategy": "dual_ma",
        "metrics": {
            "total_return": -0.00484, "total_trades": 7,
            "avg_slippage": 0.0005, "calmar_ratio": -22.37, "sortino": -7.3,
        },
        "trades": [{
            "ts": "2023-11-15T18:38:20+00:00", "symbol": "ETH/USDT",
            "side": "sell", "price": 108.05, "qty": 48.17, "fee": 5.2,
            "pnl": 200.49, "reason": "x", "slippage": 0.0005,
        }],
        "equity_curve": [10000.0, 10000.0],
    }


def test_valid_result_passes_and_version():
    r = validate_backtest_result(_valid_result())
    assert r.schema_version == SCHEMA_VERSION
    assert r.metrics.avg_slippage == 0.0005
    assert r.trades[0].slippage == 0.0005


def test_missing_total_return_rejected():
    d = _valid_result()
    del d["metrics"]["total_return"]
    with pytest.raises(ValidationError):
        validate_backtest_result(d)


def test_nan_inf_rejected():
    """NaN/inf 指标必须拒绝（JSONResponse 会 500，schema 层兜底）。"""
    d = _valid_result()
    d["metrics"]["total_return"] = float("nan")
    with pytest.raises(ValidationError):
        validate_backtest_result(d)
    d2 = _valid_result()
    d2["metrics"]["sharpe"] = float("inf")
    with pytest.raises(ValidationError):
        validate_backtest_result(d2)


def test_extra_fields_allowed():
    d = _valid_result()
    d["metrics"]["new_future_field"] = 1.0
    d["custom"] = {"a": 1}
    r = validate_backtest_result(d)
    assert r.metrics.new_future_field == 1.0


def test_metrics_defaults_for_missing_optional():
    m = BacktestMetrics(total_return=-0.01)
    assert m.schema_version == SCHEMA_VERSION
    assert m.calmar_ratio == 0.0 and m.avg_slippage == 0.0