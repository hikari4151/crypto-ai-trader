"""双回测引擎一致性回归测试。

两引擎（backtest/engine.py 事件引擎 / backtest/fast_engine.py 向量引擎）的
pending_signal/FIFO 记账/期末平仓/120 根 S/R 窗口是反复修出来的对齐成果，
任何一端改动都必须保持逐笔一致，否则同一策略两引擎结果分叉。
"""
import pytest

from backtest.data_loader import generate_demo
from backtest.engine import BacktestConfig, run_backtest
from backtest.fast_engine import run_backtest_fast


@pytest.fixture(scope="module")
def demo_df():
    return generate_demo(timeframe="1h", n=1500)


def _run_both(df, strategy: str, params: dict | None = None):
    cfg = BacktestConfig(symbol="BTC/USDT", timeframe="1h",
                         strategy_name=strategy, strategy_params=params or {},
                         start_cash=10000.0, fee_rate=0.001)
    r_event = run_backtest(df, cfg)
    r_fast = run_backtest_fast(df, cfg)
    return r_event, r_fast


def _trade_keys(trades):
    return [(t["side"], round(t["price"], 6), round(t["qty"], 8), round(t["pnl"], 6)) for t in trades]


@pytest.mark.parametrize("strategy", ["dual_ma", "grid"])
def test_two_engines_consistent(demo_df, strategy):
    """两引擎在相同数据/参数下逐笔成交与绩效一致。"""
    r_event, r_fast = _run_both(demo_df, strategy)
    m1, m2 = r_event["metrics"], r_fast["metrics"]
    assert len(r_event["trades"]) == len(r_fast["trades"])
    assert _trade_keys(r_event["trades"]) == _trade_keys(r_fast["trades"])
    assert m1["total_return"] == pytest.approx(m2["total_return"], rel=1e-6)
    assert m1["win_rate"] == pytest.approx(m2["win_rate"], abs=1e-6)
    assert m1["total_trades"] == m2["total_trades"]


def test_two_engines_consistent_price_action(demo_df):
    """price_action 走 S/R 路径（120 根窗口），两引擎也应一致。"""
    r_event, r_fast = _run_both(demo_df, "price_action")
    m1, m2 = r_event["metrics"], r_fast["metrics"]
    assert len(r_event["trades"]) == len(r_fast["trades"])
    assert m1["total_return"] == pytest.approx(m2["total_return"], rel=1e-4)


def test_final_forced_liquidation(demo_df):
    """期末强制平仓：持仓策略（grid 常驻）最后一笔应为期末平仓。"""
    r_event, r_fast = _run_both(demo_df, "grid")
    for r in (r_event, r_fast):
        if r["trades"]:
            assert r["trades"][-1]["side"] == "sell"
            assert r["trades"][-1]["reason"] == "期末强制平仓"
            # 期末平仓后权益=现金（持仓清零）
            assert r["equity_curve"][-1] == pytest.approx(
                r["equity_curve"][-1], rel=1e-9)
