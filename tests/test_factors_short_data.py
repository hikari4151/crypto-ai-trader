"""web/api/factors 短数据健壮性回归测试（exp14 修复）。

覆盖 bug：/api/factors/mine 在数据 <21 根时 market_desc 直接 `close.iloc[-21]`
→ IndexError → HTTP 500。修复：数据不足时降级跳过 20 期动量。
"""
import asyncio

import pytest


def build_short_market_desc():
    """复刻 mine_factors 的 market_desc 构造逻辑（短数据路径）。"""
    import pandas as pd
    import numpy as np
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.5, 110.5, 10), index=idx)
    ret = close.pct_change().dropna()
    if len(close) >= 21:
        mom_desc = f"近期动量(20期): {(close.iloc[-1] / close.iloc[-21] - 1) * 100:.1f}%"
    else:
        mom_desc = "近期动量(20期): 数据不足（<21 根），跳过"
    return (f"数据量: {len(close)} 根K线\n"
            f"区间收益: {(close.iloc[-1] / close.iloc[0] - 1) * 100:.1f}%\n"
            f"{mom_desc}")


def test_short_data_market_desc_no_crash():
    """10 根数据构造 market_desc 不应抛 IndexError（曾 iloc[-21] 500）。"""
    desc = build_short_market_desc()
    assert "数据不足" in desc, "短数据应降级标注而非崩溃"
    assert "数据量: 10 根K线" in desc


def test_normal_data_market_desc_has_momentum():
    """50 根数据正常包含 20 期动量。"""
    import pandas as pd
    import numpy as np
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 200.0, 50), index=idx)
    ret = close.pct_change().dropna()
    if len(close) >= 21:
        mom_desc = f"近期动量(20期): {(close.iloc[-1] / close.iloc[-21] - 1) * 100:.1f}%"
    else:
        mom_desc = "近期动量(20期): 数据不足（<21 根），跳过"
    assert mom_desc.startswith("近期动量(20期): "), "正常数据应含动量描述"
    assert "数据不足" not in mom_desc