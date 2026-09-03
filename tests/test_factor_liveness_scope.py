"""P1-1 研究可复现性：回测/研究路径不得读进程级因子上下线状态。

上下线（IC 衰变自动下线）是运维态，只活在 factors.library._ALIVE 里，
由 /api 的定期作业写入。此前 factor_signal 策略在回测中也会读它，导致
同一段历史因「今天有没有跑过 IC 衰变作业」得出不同回测结论。
"""
import json

import pytest

from backtest.data_loader import generate_demo
from backtest.engine import BacktestConfig, run_backtest
from backtest.fast_engine import run_backtest_fast
from factors import library as lib
from strategies.base import Signal


@pytest.fixture
def restore_alive():
    """用例内改 _ALIVE，用例后还原（模块级全局，不能污染其他测试）。"""
    snapshot = dict(lib._ALIVE)
    yield
    lib._ALIVE.clear()
    lib._ALIVE.update(snapshot)


def _mark_offline(key: str) -> None:
    lib._ALIVE[key] = False


def _bb_ctx(price: float = 100.0) -> dict:
    return {
        "symbol": "BTC/USDT", "price": price, "position": 0.0,
        "indicators": {"candles_count": 50, "open": 99.0, "high": 101.0, "low": 98.0,
                       "close": price, "volume": 10.0, "bb_upper": 110.0, "bb_lower": 90.0},
    }


def _factor_strategy(**params) -> "object":
    from strategies import get_strategy
    s = get_strategy("factor_signal")
    p = {"factor": "bb_pos", "mode": "trend", "buy_threshold": 0.4, "sell_threshold": 0.4}
    p.update(params)
    s.update_params(p)
    s.reset()
    return s


# ---------------- 作用域本身 ----------------

def test_offline_factor_is_live_only_outside_research_scope(restore_alive):
    _mark_offline("bb_pos")
    assert lib.factor_is_live("bb_pos") is False
    with lib.ignoring_factor_liveness():
        assert lib.factor_is_live("bb_pos") is True
    # 作用域退出后运维态恢复生效（不泄漏）
    assert lib.factor_is_live("bb_pos") is False


def test_scope_restores_even_on_exception():
    with pytest.raises(RuntimeError):
        with lib.ignoring_factor_liveness():
            raise RuntimeError("boom")
    assert lib.factor_is_live("bb_pos") is True


# ---------------- 策略端：实盘仍读、研究不读 ----------------

def test_live_path_still_skips_offline_factor(restore_alive):
    """实盘口径不变：因子下线后策略不再产生信号。"""
    _mark_offline("bb_pos")
    assert _factor_strategy().on_candle(_bb_ctx()) is None


def test_research_path_keeps_offline_factor(restore_alive):
    """研究口径：同一下线状态下，回测作用域内照常出信号。"""
    _mark_offline("bb_pos")
    s = _factor_strategy()
    with lib.ignoring_factor_liveness():
        sig = s.on_candle(_bb_ctx())
    assert isinstance(sig, Signal) and sig.side == "buy"


def test_combo_factor_includes_offline_members_in_research(restore_alive):
    """组合因子（RL 挖掘）里的下线成员：实盘跳过、研究保留。"""
    _mark_offline("mom_5")
    df = generate_demo(timeframe="1h", n=60, seed=7)
    spec = json.dumps({"mom_5": 1.0})

    def _value() -> object:
        s = _factor_strategy(factor="combo", combo_spec=spec)
        for _, row in df.iterrows():
            s._push_ohlcv({"symbol": "BTC/USDT", "price": float(row["close"]),
                           "indicators": {"candles_count": 50, **row.to_dict()}})
        return s._combo_factor_value()

    assert _value() is None  # 唯一成员下线 → 实盘无可用因子
    with lib.ignoring_factor_liveness():
        assert _value() is not None


# ---------------- 回测引擎：结果与上下线状态无关 ----------------

def _cfg() -> BacktestConfig:
    return BacktestConfig(symbol="BTC/USDT", timeframe="1h", strategy_name="factor_signal",
                          strategy_params={"factor": "bb_pos", "mode": "trend",
                                           "buy_threshold": 0.4, "sell_threshold": 0.4},
                          start_cash=10000.0)


@pytest.fixture(scope="module")
def demo_df():
    return generate_demo(timeframe="1h", n=500, seed=42)


def _deterministic(metrics: dict) -> str:
    """只保留决定研究结论的指标（elapsed_sec 是墙钟计时，每次必然不同）。"""
    m = {k: v for k, v in metrics.items() if not k.endswith("_sec")}
    return json.dumps(m, sort_keys=True, default=str)


@pytest.mark.parametrize("engine", ["fast", "event"])
def test_backtest_reproducible_across_liveness_changes(demo_df, restore_alive, engine):
    """核心断言：全部因子标记下线后，回测结果逐字节不变。"""
    run = run_backtest_fast if engine == "fast" else run_backtest
    baseline = run(demo_df, _cfg())
    assert len(baseline["trades"]) > 0, "demo 数据上该策略应至少成交一笔，否则用例无意义"

    for key in list(lib._ALIVE):
        lib._ALIVE[key] = False
    mutated = run(demo_df, _cfg())

    assert _deterministic(mutated["metrics"]) == _deterministic(baseline["metrics"])
    assert len(mutated["trades"]) == len(baseline["trades"])
    assert lib.factor_is_live("bb_pos") is False  # 运维态本身确实是被读的那份状态
