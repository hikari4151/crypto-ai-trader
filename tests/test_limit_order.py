"""限价单模拟 + 数据清洗测试"""
import numpy as np
import pandas as pd
import pytest
from backtest.engine import BacktestConfig, run_backtest
from backtest.fast_engine import run_backtest_fast
from backtest.data_loader import generate_demo, OHLCVSanitizer
from strategies.base import Signal, Strategy


class TestLimitOrderConfig:
    """BacktestConfig / BacktestRequest / GridScanIn 新增 limit_order_model"""

    def test_backtest_config_default(self):
        """默认 limit_order_model='none'"""
        cfg = BacktestConfig()
        assert cfg.limit_order_model == "none"

    def test_backtest_config_explicit(self):
        """显式设 partial"""
        cfg = BacktestConfig(limit_order_model="partial")
        assert cfg.limit_order_model == "partial"

    def test_none_mode_keeps_original_results(self):
        """none 模式（默认）下回测结果与修改前基线一致（使用 dual_ma）"""
        df = generate_demo(timeframe="1h", n=1500)
        cfg_none = BacktestConfig(strategy_name="dual_ma", limit_order_model="none")
        cfg_default = BacktestConfig(strategy_name="dual_ma")  # 默认 none
        r1 = run_backtest(df, cfg_none)
        r2 = run_backtest(df, cfg_default)
        assert r1["metrics"]["total_return"] == pytest.approx(r2["metrics"]["total_return"], rel=1e-9)
        assert r1["metrics"]["sharpe"] == pytest.approx(r2["metrics"]["sharpe"], rel=1e-9)
        assert len(r1["trades"]) == len(r2["trades"])

    def test_signal_has_order_type_default(self):
        """Signal 类 order_type 默认 market 保持原有行为"""
        sig = Signal("BTC/USDT", "buy")
        assert sig.order_type == "market"
        assert sig.limit_price is None


class TestLimitOrderPartialFill:
    """限价单部分成交模拟"""

    @pytest.fixture(scope="module")
    def demo_df(self):
        return generate_demo(timeframe="1h", n=1500)

    def test_limit_order_model_partial_no_crash(self, demo_df):
        """partial 模式 + 现有策略（非限价信号）不报错，结果与 none 一致"""
        cfg_none = BacktestConfig(strategy_name="dual_ma", limit_order_model="none")
        cfg_partial = BacktestConfig(strategy_name="dual_ma", limit_order_model="partial")
        r_none = run_backtest(demo_df, cfg_none)
        r_partial = run_backtest(demo_df, cfg_partial)
        # 现有策略全是 market 信号，partial 模式不影响
        assert r_none["metrics"]["total_return"] == pytest.approx(r_partial["metrics"]["total_return"], rel=1e-9)
        assert len(r_none["trades"]) == len(r_partial["trades"])

    def test_two_engines_consistent_partial(self, demo_df):
        """两引擎在 partial 模式下一致"""
        cfg = BacktestConfig(strategy_name="dual_ma", limit_order_model="partial")
        r_event = run_backtest(demo_df, cfg)
        r_fast = run_backtest_fast(demo_df, cfg)
        assert r_event["metrics"]["total_return"] == pytest.approx(r_fast["metrics"]["total_return"], rel=1e-6)
        assert len(r_event["trades"]) == len(r_fast["trades"])

    def test_limit_order_skip_fill_keeps_equity_curve_length(self, demo_df):
        """未触及的限价单不跳过 equity_curve 点，长度与数据一致"""
        cfg = BacktestConfig(strategy_name="dual_ma", limit_order_model="partial")
        r_event = run_backtest(demo_df, cfg)
        assert len(r_event["equity_curve"]) == len(demo_df)
        r_fast = run_backtest_fast(demo_df, cfg)
        assert len(r_fast["equity_curve"]) == len(demo_df)

    def test_limit_order_not_touched_no_fill(self):
        """限价远低于市场且全程未触及 → 不成交"""
        rng = np.random.default_rng(42)
        n = 100
        base = 100.0
        # 波动取 ±5：人为构造的 K 线低点需落在正常波动内，
        # 否则会被 run_backtest 的 OHLCVSanitizer 当离群值替换（Z 过滤），
        # 限价单将"看似未触及"而无法成交（曾致 fills=[]）
        closes = base + rng.normal(0, 5, n)
        lows = closes - abs(rng.normal(0, 1, n))  # 全部 > 50
        df = _make_df(closes, lows)

        with _register_test_strategy() as base_cls:
            class LimitBuyStrat(base_cls):
                name = "_test_limit_low"
                def on_candle(self, ctx):
                    if not self._emitted and ctx.get("position", 0) <= 0:
                        self._emitted = True
                        return Signal(ctx["symbol"], "buy", 0.5,
                                      order_type="limit", limit_price=50.0,  # 远低于市场
                                      reason="test_limit_low")
                    return None
            import strategies
            _r = strategies._REGISTRY
            _r["_test_limit_low"] = LimitBuyStrat
            try:
                cfg = BacktestConfig(strategy_name="_test_limit_low", limit_order_model="partial")
                r = run_backtest(df, cfg)
                assert len(r["trades"]) == 0, "限价未触及不应成交"
            finally:
                _r.pop("_test_limit_low", None)

    def test_limit_order_touched_fills_at_limit(self):
        """限价触及 K 线低点 → 以限价成交（fill price = limit_price）"""
        rng = np.random.default_rng(42)
        n = 100
        base = 100.0
        # 波动取 ±5：人为构造的 K 线低点需落在正常波动内，
        # 否则会被 run_backtest 的 OHLCVSanitizer 当离群值替换（Z 过滤），
        # 限价单将"看似未触及"而无法成交（曾致 fills=[]）
        closes = base + rng.normal(0, 5, n)
        lows = closes - abs(rng.normal(0, 1, n))
        lows[1] = 90.0  # 人为压低第 2 根K线低点，确保 95 触及
        df = _make_df(closes, lows)

        fills: list[tuple] = []

        with _register_test_strategy() as base_cls:
            class LimitBuyTouch(base_cls):
                name = "_test_limit_touch"
                def on_candle(self, ctx):
                    if not self._emitted and ctx.get("position", 0) <= 0:
                        self._emitted = True
                        return Signal(ctx["symbol"], "buy", 0.5,
                                      order_type="limit", limit_price=95.0,
                                      reason="test_limit_touch")
                    return None
                def on_fill(self, symbol, side, price):
                    fills.append((side, price))
            import strategies
            _r = strategies._REGISTRY
            _r["_test_limit_touch"] = LimitBuyTouch
            try:
                cfg = BacktestConfig(strategy_name="_test_limit_touch", limit_order_model="partial")
                r = run_backtest(df, cfg)
                # 触及应以限价 95 成交（买入成交在 on_fill 中被记录）
                assert any(side == "buy" and price == pytest.approx(95.0, abs=1e-3)
                           for side, price in fills), f"应在限价 95 成交，实际 fills={fills}"
                assert len(r["trades"]) >= 1, "触及成交后期末应有强制平仓"
            finally:
                _r.pop("_test_limit_touch", None)

    def test_limit_order_three_bar_cancel(self):
        """未触及时最多挂 3 根K线后取消（超时无成交）"""
        rng = np.random.default_rng(42)
        n = 100
        base = 100.0
        # 波动取 ±5：人为构造的 K 线低点需落在正常波动内，
        # 否则会被 run_backtest 的 OHLCVSanitizer 当离群值替换（Z 过滤），
        # 限价单将"看似未触及"而无法成交（曾致 fills=[]）
        closes = base + rng.normal(0, 5, n)
        lows = closes - abs(rng.normal(0, 1, n))  # low 在 95 以上
        df = _make_df(closes, lows)

        with _register_test_strategy() as base_cls:
            class LimitBuyCancel(base_cls):
                name = "_test_limit_cancel"
                def on_candle(self, ctx):
                    if not self._emitted and ctx.get("position", 0) <= 0:
                        self._emitted = True
                        return Signal(ctx["symbol"], "buy", 0.5,
                                      order_type="limit", limit_price=10.0,  # 永远无法触及
                                      reason="test_limit_cancel")
                    return None
            import strategies
            _r = strategies._REGISTRY
            _r["_test_limit_cancel"] = LimitBuyCancel
            try:
                cfg = BacktestConfig(strategy_name="_test_limit_cancel", limit_order_model="partial")
                r = run_backtest(df, cfg)
                assert len(r["trades"]) == 0, "3 根K线未触及应取消，不成交"
            finally:
                _r.pop("_test_limit_cancel", None)


def _make_df(closes, lows, n=100):
    """构造标准 OHLCV DataFrame"""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "open": np.roll(closes, 1),
        "high": closes + abs(rng.normal(0, 1, n)),
        "low": lows,
        "close": closes,
        "volume": abs(rng.normal(100, 20, n)),
    }).set_index("timestamp")


class _TestLimitBase(Strategy):
    """测试用途策略基类：一次信号后不再发"""

    def __init__(self):
        super().__init__()
        self._emitted = False


import contextlib


@contextlib.contextmanager
def _register_test_strategy():
    """注册测试策略基类到全局注册表。"""
    import strategies
    orig = strategies._REGISTRY.copy()
    strategies._REGISTRY["_test_limit_base"] = _TestLimitBase
    try:
        yield _TestLimitBase
    finally:
        strategies._REGISTRY = orig


class TestOHLCVSanitizer:
    """数据清洗器"""

    def test_clean_no_nan(self):
        """正常数据不变"""
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0],
                           "low": [0.5, 1.5, 2.5], "close": [1.5, 2.5, 3.5],
                           "volume": [100, 200, 300]})
        s = OHLCVSanitizer()
        cleaned = s.clean(df)
        assert cleaned["close"].tolist() == [1.5, 2.5, 3.5]

    def test_clean_ffill_nan(self):
        """缺失值前向填充"""
        df = pd.DataFrame({"open": [1.0, np.nan, 3.0], "high": [2.0, 3.0, 4.0],
                           "low": [0.5, 1.5, 2.5], "close": [1.5, np.nan, 3.5],
                           "volume": [100, 200, 300]})
        s = OHLCVSanitizer()
        cleaned = s.clean(df)
        assert cleaned["close"].tolist() == [1.5, 1.5, 3.5]  # ffill
        assert cleaned["open"].tolist() == [1.0, 1.0, 3.0]

    def test_clean_outlier_mark_by_default(self):
        """P1-9 默认标记制：异常值只标记不清洗（避免全样本中位数替换的前视）"""
        values = [100.0] * 50 + [10000.0]  # 一个极端值
        df = pd.DataFrame({"close": values, "open": values, "high": values,
                           "low": values, "volume": values})
        s = OHLCVSanitizer(z_threshold=5.0)
        cleaned = s.clean(df, mode="mark")
        # 标记制：真实值保留（10000 是真实行情，不应被抹平）
        assert cleaned["close"].iloc[-1] == pytest.approx(10000.0, abs=1.0)
        flags = cleaned.attrs.get("clean_flags", {})
        assert flags.get("outliers_close", 0) == 1, f"应标记离群值, flags={flags}"

    def test_clean_outlier_replace_mode(self):
        """显式 mode='replace'：保留旧行为（极端异常值用中位数替换）"""
        values = [100.0] * 50 + [10000.0]  # 一个极端值
        df = pd.DataFrame({"close": values, "open": values, "high": values,
                           "low": values, "volume": values})
        s = OHLCVSanitizer(z_threshold=5.0)
        cleaned = s.clean(df, mode="replace")
        # 异常值被替换为中位数 100.0
        assert cleaned["close"].iloc[-1] == pytest.approx(100.0, abs=1.0)

    def test_clean_returns_copy(self):
        """返回副本，不修改原数据"""
        df = pd.DataFrame({"close": [1.0, np.nan, 3.0]})
        original = df.copy()
        s = OHLCVSanitizer()
        cleaned = s.clean(df)
        assert cleaned is not df
        assert df["close"].isna().sum() == 1  # 原数据未变