"""注册前守卫的数据量与"无法判定"语义。

背景：AI 守卫长期只有 800 根K线（且 /api/v3/klines 单次上限 1000 根，limit 给再大
也会被静默截断），足量的前推验证根本做不出来 → 结论多为"无法判定"，而"无法判定"
过去被当成静默放行：策略照注册、AI 参数热更新照接管实盘。
现在：守卫取 3000 根（本地库优先 + 分页回填）；"无法判定"仍可注册，但禁止自动接管。
"""
import numpy as np
import pytest

from ai import strategy_designer as sd
from engine.trading_engine import TradingEngine


def _candles(n: int, start_ts: int = 1700000000000) -> list:
    t = np.arange(n)
    close = 100.0 + 5.0 * np.sin(t / 20.0)
    return [[start_ts + i * 3600000,
             float(close[i] * 0.999), float(close[i] * 1.002),
             float(close[i] * 0.998), float(close[i]), 1000.0] for i in range(n)]


# ---------------- 1. 历史取数：本地优先 + 分页回填 ----------------

class _HistStub(TradingEngine):
    """只装配 _fetch_history_ohlcv 需要的属性，不构造完整引擎。"""

    def __init__(self, rows=None):
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.rows = rows or []


@pytest.mark.asyncio
async def test_local_store_is_used_without_network(monkeypatch):
    monkeypatch.setattr("backtest.kline_store.load_rows",
                        lambda *a, **k: _candles(4000))

    async def _no_network(*a, **k):
        raise AssertionError("本地够用时不该再打 REST")

    monkeypatch.setattr(TradingEngine, "_fetch_klines_pages", _no_network)
    rows = await _HistStub()._fetch_history_ohlcv(3000)
    assert len(rows) == 3000
    assert rows[0][0] == _candles(1)[0][0] + 1000 * 3600000  # 取的是最新 3000 根


@pytest.mark.asyncio
async def test_gap_is_backfilled_into_store_then_reread(monkeypatch):
    store: list = []

    def _load(*a, **k):
        return list(store)

    def _save(exchange, symbol, tf, rows):
        store.extend(rows)
        return len(rows)

    monkeypatch.setattr("backtest.kline_store.load_rows", _load)
    monkeypatch.setattr("backtest.kline_store.save_rows", _save)

    async def _pages(self, target, sym, tf):
        return _candles(target)

    monkeypatch.setattr(TradingEngine, "_fetch_klines_pages", _pages)
    rows = await _HistStub()._fetch_history_ohlcv(3000)
    assert len(rows) == 3000
    assert len(store) == 3000  # 回填进本地库，下次直接命中


@pytest.mark.asyncio
async def test_history_is_capped_at_target(monkeypatch):
    monkeypatch.setattr("backtest.kline_store.load_rows",
                        lambda *a, **k: _candles(9000))
    rows = await _HistStub()._fetch_history_ohlcv(3000)
    assert len(rows) == 3000
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)  # 保持升序


class _FakeResp:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return self._rows


class _FakeHttp:
    """模拟币安 /klines：给定 endTime + limit，返回不晚于 endTime 的最多 limit 根K线。

    total_rows 小于 limit 的倍数时，末页必然"给不满"，那就是真实 API
    触到该品种历史起点的信号，分页循环必须据此停止，否则会无限往前要页。
    """

    def __init__(self, total_rows, page_limit=None):
        self.master = _candles(total_rows)
        self.page_limit = page_limit or 1000
        self.seen_endTimes: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        end = int(params["endTime"])
        limit = int(params["limit"])
        assert limit <= 1000, "币安单次上限 1000，请求更大 limit 会被静默截断"
        self.seen_endTimes.append(end)
        eligible = [r for r in self.master if r[0] <= end]
        count = min(limit, len(eligible), self.page_limit)
        # 真实 REST 字段是字符串，引擎侧靠 int()/float() 转换
        return _FakeResp([[str(int(c[0])), str(c[1]), str(c[2]), str(c[3]), str(c[4]), str(c[5])]
                          for c in eligible[-count:]])


@pytest.mark.asyncio
async def test_pages_step_backwards_until_target_reached(monkeypatch):
    fake = _FakeHttp(5000)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake)

    rows = await _HistStub()._fetch_klines_pages(3000, "BTC/USDT", "1h")
    assert len(rows) == 3000
    assert len(fake.seen_endTimes) == 3          # 3 页 × 1000
    assert fake.seen_endTimes == sorted(fake.seen_endTimes, reverse=True)
    ts = [r[0] for r in rows]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)  # 升序且无重叠


@pytest.mark.asyncio
async def test_pages_stop_at_series_start_without_looping_forever(monkeypatch):
    fake = _FakeHttp(1400)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake)

    rows = await _HistStub()._fetch_klines_pages(6000, "BTC/USDT", "1h")
    # 第 2 页只回 400 根 → 判定已触到历史起点，不再继续要第 3/4/5/6 页
    assert len(rows) == 1400
    assert len(fake.seen_endTimes) == 2


# ---------------- 2. "无法判定"不再等于放行 ----------------

class _ValStub(TradingEngine):
    """只装配参数热更新验证门需要的属性，不构造完整引擎。"""

    def __init__(self, rows):
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.start_cash = 10000.0
        self.paper_fee_rate = 0.001
        self.paper_slippage = 0.0005
        self.hub = None
        self.rows = rows

    async def _fetch_vision_ohlcv(self, limit=800):
        return self.rows[:limit]


_PARAMS = {"fast_period": 10, "slow_period": 30, "size_pct": 0.5,
           "stop_loss_pct": 0.03, "take_profit_pct": 0.06}


class _Rep:
    def __init__(self, verdict):
        self.verdict = verdict
        self.score = 50.0
        self.oos_ret = 0.01
        self.pbo = 0.2
        self.decay = 0.1
        self.n_folds = 4


def _fake_verdict(monkeypatch, verdict):
    import backtest.overfit as ov
    monkeypatch.setattr(ov, "detect_overfit", lambda df, name, params, cfg: _Rep(verdict))


@pytest.mark.asyncio
async def test_takeover_gate_refuses_inconclusive(monkeypatch):
    """守卫拿不出结论时，AI 参数不许无人值守地把实盘/纸面仓位接管过去。"""
    _fake_verdict(monkeypatch, "无法判定")

    ok, info = await _ValStub(_candles(500))._validate_param_update(
        "dual_ma", dict(_PARAMS), dict(_PARAMS))

    assert ok is False
    assert "无法判定" in info["reason"]


@pytest.mark.asyncio
async def test_takeover_gate_still_applies_when_guard_clears(monkeypatch):
    """拦截只针对"证据不足/严重过拟合"，不能顺手把正常通道也堵死。"""
    _fake_verdict(monkeypatch, "通过")

    ok, info = await _ValStub(_candles(500))._validate_param_update(
        "dual_ma", dict(_PARAMS), dict(_PARAMS))

    assert ok is True, info.get("reason")
