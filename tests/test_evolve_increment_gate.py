"""P2-11 增量门必须与 DatetimeIndex 的时间戳精度无关。

pandas 3 起 `pd.to_datetime(..., unit="ms")` 产出 datetime64[ms] 而不再升频到 [ns]，
本地K线库 load_df 因此给出 ms 精度索引。门内用 `idx.asi8 // 10**9` 按纳秒换算秒，
ms 索引下 5000 根 K 线全部变成 1788，n_new 恒为 0：增量门在第一轮训练后永久关闭，
每条管线每个进程生命周期只训练一次，之后一直「无新增K线，跳过本轮」。
"""
import pandas as pd
import pytest

from config.settings import settings
from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


def _df(n_bars: int, unit: str) -> pd.DataFrame:
    idx = pd.date_range("2026-09-02T00:00", periods=n_bars, freq="5min", tz="UTC").as_unit(unit)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                        index=idx)


@pytest.fixture
def eng(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "evolve_min_new_bars", 2, raising=False)
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_gate_reopens_after_enough_new_bars(eng, unit):
    """攒够 min_new_bars 根新增 K 线后必须再次放行——任何时间戳精度下都一样。"""
    name = "factor_miner"
    assert eng._data_incremented(name, _df(60, unit)) is True      # 首轮无记录恒放行
    assert eng._data_incremented(name, _df(60, unit)) is False     # 末尾无变化 → 跳过
    assert eng._data_incremented(name, _df(61, unit)) is False     # 仅 1 根新增 → 继续等
    assert eng._data_incremented(name, _df(62, unit)) is True      # 累计 2 根 → 训练
    assert eng._data_incremented(name, _df(63, unit)) is False     # 训练后重新计数


def test_gate_opens_on_kline_store_index(eng):
    """真实数据源（kline_store.load_df）的索引补一根新 K 线后必须放行。"""
    from backtest import kline_store

    df = kline_store.load_df("binance", "ETH/USDT", "5m", 500)
    if df.empty:
        pytest.skip("本机本地K线库无 ETH/USDT 5m 数据")
    assert df.index.dtype.kind == "M"
    assert eng._data_incremented("factor_miner", df) is True
    grown = df.copy()
    grown.index = df.index + pd.Timedelta(minutes=10)   # 整段右移 = 有新根
    assert eng._data_incremented("factor_miner", grown) is True


def test_gate_not_locked_by_non_datetime_index(eng):
    """非时间戳索引（无法判定新增根数）必须放行而不是把门永久关死。"""
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.Index([1788341400000, 1788341700000]))
    assert eng._data_incremented("strategy_drl", df) is True
    assert eng._data_incremented("strategy_drl", df) is True
