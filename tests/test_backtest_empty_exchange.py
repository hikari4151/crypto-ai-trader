"""回测数据入口必须拒绝交易所空数据或不足数据。"""

import pandas as pd
import pytest

from web.api.backtest import _require_backtest_data


def test_exchange_empty_data_is_rejected():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError, match="交易所未返回有效 K 线"):
        _require_backtest_data(df, "exchange")


def test_exchange_short_data_is_rejected():
    df = pd.DataFrame({
        "open": [1.0] * 59,
        "high": [1.0] * 59,
        "low": [1.0] * 59,
        "close": [1.0] * 59,
        "volume": [1.0] * 59,
    })
    with pytest.raises(ValueError, match="至少需要 60"):
        _require_backtest_data(df, "exchange")
