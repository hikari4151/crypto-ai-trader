"""kline_store.load_rows(limit=n) 必须返回「最新」n 根，而不是「最旧」n 根。

契约（docstring 明示「limit>0 取最新 limit 根」「读取K线（升序）」）与实际实现相反：
旧实现把 DESC 拼在了子查询括号外 —— `SELECT * FROM (... ORDER BY ts) DESC LIMIT ?`，
SQLite 静默忽略这个悬空 DESC，等价于 ASC LIMIT n，于是取回窗口最早 n 根。

后果不止「回测拿错数据」：持续进化的增量门比较的是 df 末尾时间戳，而末尾被永久
钉在第 n 根上（新 K 线只追加在尾部），门永远关着 → 三管线每个进程生命周期只训一轮。
"""
import threading

import pytest

import backtest.kline_store as ks

_T0 = 1_788_000_000_000
_STEP = 300_000  # 5m


@pytest.fixture
def store(tmp_path, monkeypatch):
    """把库指到临时文件并清掉线程局部连接，避免污染真实 data/klines.db。"""
    monkeypatch.setattr(ks, "DB_PATH", tmp_path / "klines.db")
    monkeypatch.setattr(ks, "_local", threading.local())
    return ks


def _rows(n: int) -> list:
    return [[_T0 + i * _STEP, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(n)]


def test_limit_returns_the_newest_bars(store):
    store.save_rows("binance", "ETH/USDT", "5m", _rows(100))
    rows = store.load_rows("binance", "ETH/USDT", "5m", limit=10)
    assert [r[0] for r in rows] == [_T0 + i * _STEP for i in range(90, 100)]


def test_limit_result_stays_ascending(store):
    """调用方（_to_df / 指标计算 / 增量门）全部假定升序，reverse 不能丢。"""
    store.save_rows("binance", "ETH/USDT", "5m", _rows(50))
    ts = [r[0] for r in store.load_rows("binance", "ETH/USDT", "5m", limit=8)]
    assert ts == sorted(ts)


def test_limit_larger_than_available_returns_all(store):
    store.save_rows("binance", "ETH/USDT", "5m", _rows(6))
    assert len(store.load_rows("binance", "ETH/USDT", "5m", limit=1000)) == 6


def test_no_limit_returns_all_ascending(store):
    store.save_rows("binance", "ETH/USDT", "5m", _rows(6))
    rows = store.load_rows("binance", "ETH/USDT", "5m")
    assert len(rows) == 6 and rows[0][0] < rows[-1][0]


def test_load_df_tail_is_the_latest_bar(store):
    """load_df 是训练/回测的取数入口：末尾必须是全库最新一根。"""
    store.save_rows("binance", "ETH/USDT", "5m", _rows(30))
    df = store.load_df("binance", "ETH/USDT", "5m", limit=10)
    assert len(df) == 10
    assert int(df.index[-1].timestamp() * 1000) == _T0 + 29 * _STEP
