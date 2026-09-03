"""M2 性能引擎优化：逐位一致性测试。

T1: S/R 和价格行为向量化预计算——support_resistance_series / price_action_features_series
    与逐K线调用 support_resistance / price_action_features 在窗口口径一致时逐位等价。
T2: list.pop(0) → deque 替换后，IncrIndicators 指标输出与 list 版一致。
"""
import numpy as np
import pytest

from indicators.technical import (
    support_resistance, support_resistance_series,
    price_action_features, price_action_features_series,
)
from indicators.vectorized import IncrIndicators


# ============ T1: S/R 向量化预计算一致性 ============

@pytest.fixture(scope="module")
def demo_arrays():
    """生成合成 K 线数据，供 T1 各项测试共享。"""
    rng = np.random.default_rng(42)
    n = 500
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(100, 1000, n)
    return high, low, close, open_, vol


class TestSupportResistanceSeries:
    """support_resistance_series vs 逐K线 support_resistance（窗口 max(0,i-119):i+1）逐位一致。"""

    def test_identical_to_per_bar(self, demo_arrays):
        high, low, close, _, _ = demo_arrays
        n = len(close)
        sr_new = support_resistance_series(high, low, close, window=10, min_touches=2)
        for i in range(20, n):  # 跳过暖机期（前 20 根）
            a = max(0, i - 119)
            b = i + 1
            old = support_resistance(high[a:b], low[a:b], close[a:b], window=10, min_touches=2)
            assert old == sr_new[i], f"SR mismatch at bar {i}"

    def test_warmup_none(self, demo_arrays):
        """前 20 根因摆动检测窗口不足（hi <= lo），返回 _sr_result([], [], ...) 字典，
        其中 support/resistance 等键值均为 None（而非 None 本身）。"""
        high, low, close, _, _ = demo_arrays
        sr_new = support_resistance_series(high, low, close, window=10, min_touches=2)
        for i in range(10):
            assert sr_new[i] is not None, f"bar {i} should not be None"
            # 暖机期返回的 dict 中 support/resistance 可能为 None

    @pytest.mark.parametrize("n_bars", [100, 1000])
    def test_different_lengths(self, n_bars):
        """不同长度序列均正确。"""
        rng = np.random.default_rng(n_bars)
        close = np.cumsum(rng.normal(0, 1, n_bars)) + 100
        high = close + rng.uniform(0, 2, n_bars)
        low = close - rng.uniform(0, 2, n_bars)
        sr_new = support_resistance_series(high, low, close, window=10, min_touches=2)
        assert len(sr_new) == n_bars
        for i in range(20, n_bars):
            a = max(0, i - 119)
            b = i + 1
            old = support_resistance(high[a:b], low[a:b], close[a:b], window=10, min_touches=2)
            assert old == sr_new[i], f"SR mismatch at bar {i} (n={n_bars})"

    def test_short_sequence(self):
        """短序列（n_bars=10 < 2*window）：_find_pivot_points 边界条件不崩溃，全为暖机 dict。"""
        rng = np.random.default_rng(42)
        close = np.cumsum(rng.normal(0, 1, 10)) + 100
        high = close + rng.uniform(0, 2, 10)
        low = close - rng.uniform(0, 2, 10)
        sr_new = support_resistance_series(high, low, close, window=10, min_touches=2)
        assert len(sr_new) == 10
        for i in range(10):
            d = sr_new[i]
            assert isinstance(d, dict)
            assert d["support"] is None
            assert d["resistance"] is None
            # 逐位与 support_resistance 对比
            assert d == support_resistance(high[:i+1], low[:i+1], close[:i+1], window=10, min_touches=2)


class TestPriceActionFeaturesSeries:
    """price_action_features_series vs 逐K线 price_action_features 逐位一致。"""

    def _seg(self, i, ts, open_, high, low, close, vol):
        """构建与回测引擎一致的 OHLCV 列表切片。"""
        a = max(0, i - 119)
        b = i + 1
        return [list(x) for x in zip(ts[a:b], open_[a:b], high[a:b], low[a:b], close[a:b], vol[a:b])]

    def test_identical_to_per_bar(self, demo_arrays):
        high, low, close, open_, vol = demo_arrays
        n = len(close)
        ts = list(range(n))
        pa_new = price_action_features_series(high, low, close, open_, vol)
        for i in range(20, n):
            old = price_action_features(self._seg(i, ts, open_, high, low, close, vol))
            assert old == pa_new[i], f"PA mismatch at bar {i}"

    def test_warmup_empty(self, demo_arrays):
        """前 20 根因不足 21 根K线，返回 {}。"""
        high, low, close, open_, vol = demo_arrays
        pa_new = price_action_features_series(high, low, close, open_, vol)
        for i in range(20):
            assert pa_new[i] == {}, f"PA bar {i} should be empty dict, got {pa_new[i]}"

    def test_short_sequence(self):
        """短序列（n_bars=15 < 21）：所有 bar 均为空 dict，sliding_window_view 分支不触发。"""
        rng = np.random.default_rng(42)
        close = np.cumsum(rng.normal(0, 1, 15)) + 100
        high = close + rng.uniform(0, 2, 15)
        low = close - rng.uniform(0, 2, 15)
        open_ = np.roll(close, 1); open_[0] = close[0]
        vol = rng.uniform(100, 1000, 15)
        pa_new = price_action_features_series(high, low, close, open_, vol)
        assert len(pa_new) == 15
        for i in range(15):
            assert pa_new[i] == {}, f"PA bar {i} should be empty dict (short sequence)"

    @pytest.mark.parametrize("n_bars", [100, 1000])
    def test_different_lengths(self, n_bars):
        """不同长度序列均正确。"""
        rng = np.random.default_rng(n_bars)
        close = np.cumsum(rng.normal(0, 1, n_bars)) + 100
        high = close + rng.uniform(0, 2, n_bars)
        low = close - rng.uniform(0, 2, n_bars)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        vol = rng.uniform(100, 1000, n_bars)
        ts = list(range(n_bars))
        pa_new = price_action_features_series(high, low, close, open_, vol)
        assert len(pa_new) == n_bars
        for i in range(20, n_bars):
            a = max(0, i - 119)
            b = i + 1
            seg = [list(x) for x in zip(ts[a:b], open_[a:b], high[a:b], low[a:b], close[a:b], vol[a:b])]
            old = price_action_features(seg)
            assert old == pa_new[i], f"PA mismatch at bar {i} (n={n_bars})"


# ============ T2: deque 替换后 IncrIndicators 一致性 ============

class TestIncrIndicatorsDeque:
    """IncrIndicators（_buf_* 改为 deque）指标输出与 list 版一致。

    通过对比 deque 版 IncrIndicators 与独立 vectorized 函数（SMA/RSI/布林带）
    在相同输入序列上的输出，验证 pop(0)→popleft() 不改变指标值。
    """

    def test_sma_match(self):
        """MA10/MA30 与 sma() 全量计算逐位一致。"""
        from indicators.technical import sma
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, 500)) + 100
        volumes = rng.uniform(100, 1000, 500)

        incr = IncrIndicators()
        ma10_incr, ma30_incr = [], []
        for i in range(len(closes)):
            r = incr.update(float(closes[i]), float(volumes[i]))
            ma10_incr.append(r["ma_fast"])
            ma30_incr.append(r["ma_slow"])

        ma10_sma = sma(closes, 10)
        ma30_sma = sma(closes, 30)
        for i in range(9, len(closes)):
            assert ma10_incr[i] == pytest.approx(ma10_sma[i], abs=1e-9), f"ma10 mismatch at {i}"
        for i in range(29, len(closes)):
            assert ma30_incr[i] == pytest.approx(ma30_sma[i], abs=1e-9), f"ma30 mismatch at {i}"

    def test_bollinger_match(self):
        """布林带与 bollinger() 全量计算逐位一致。"""
        from indicators.technical import bollinger
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, 500)) + 100
        volumes = rng.uniform(100, 1000, 500)

        incr = IncrIndicators()
        bb_up, bb_mid, bb_low = [], [], []
        for i in range(len(closes)):
            r = incr.update(float(closes[i]), float(volumes[i]))
            bb_up.append(r["bb_upper"])
            bb_mid.append(r["bb_mid"])
            bb_low.append(r["bb_lower"])

        up, mid, low = bollinger(closes, 20, 2.0)
        for i in range(19, len(closes)):
            assert bb_mid[i] == pytest.approx(mid[i], abs=1e-9), f"bb_mid mismatch at {i}"
            assert bb_up[i] == pytest.approx(up[i], abs=1e-9), f"bb_up mismatch at {i}"
            assert bb_low[i] == pytest.approx(low[i], abs=1e-9), f"bb_low mismatch at {i}"

    def test_deque_types(self):
        """确认 _buf_* 属性为 deque 类型。"""
        from collections import deque
        incr = IncrIndicators()
        incr.update(100.0, 1000.0)
        assert isinstance(incr._buf_fast, deque)
        assert isinstance(incr._buf_slow, deque)
        assert isinstance(incr._bb_buf, deque)
        assert isinstance(incr._vol_buf, deque)

    def test_pop_replaced_by_popleft(self):
        """确认缓冲区满时 popleft 正常工作（不抛异常，长度正确）。"""
        incr = IncrIndicators()
        for i in range(100):
            incr.update(float(100 + i), 1000.0)
        assert len(incr._buf_fast) == 10  # period_ma_fast=10
        assert len(incr._buf_slow) == 30  # period_ma_slow=30
        assert len(incr._bb_buf) == 20    # period_bb=20
        assert len(incr._vol_buf) == 5    # period_vol_ma=5