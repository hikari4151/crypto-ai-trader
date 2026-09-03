"""P1-2 限价挂单超时计数跟着挂单走，不再挂在对象地址上。

撮合内核早期用 `_limit_pending[id(sig)]` 记挂单已尝试的K线数。Signal 被回收后
CPython 会把它的内存地址复用给新建的 Signal（同大小类的连续分配极易撞），于是
新挂单继承旧挂单的超时次数，第一根没摸到就被判「超时取消」；同时这个旁路字典
只增不减，长回测里越积越多。
"""
import numpy as np
import pandas as pd

from backtest._matching import run_matching_loop
from backtest.engine import BacktestConfig
from indicators.vectorized import precompute_indicator_series
from strategies.base import Signal, Strategy

N = 60
LIMIT_PRICE = 90.0    # 远低于市价：只有插针那根K线会触及
CHURN_UNTIL = 45      # 此前每根收盘都挂一张摸不到的限价单（制造大量已废弃挂单）
EMIT_BAR = 50         # 最后一张需要活满尝试次数的单
# 该单从 51 根起尝试：51/52 未触及 → 53 是第 3 次尝试，插针放这里才成交


class _LimitTickStrategy(Strategy):
    """前段每根挂一张摸不到的限价单，之后只在 EMIT_BAR 挂一张需要活 3 根的单。"""
    name = "limit_tick_test"
    description = "撮合内核限价挂单超时计数测试用策略"
    default_params = {}
    param_schema = {}

    def reset(self) -> None:
        self._i = 0
        self.fills: list[tuple] = []

    def on_candle(self, ctx: dict):
        i = self._i
        self._i += 1
        if i < CHURN_UNTIL or i == EMIT_BAR:
            return Signal(ctx["symbol"], "buy", 0.5, order_type="limit",
                          limit_price=LIMIT_PRICE, strategy=self.name, reason="测试挂单")
        return None

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        self.fills.append((self._i, side, round(price, 6)))


def _make_inputs(pin_bar):
    closes = np.full(N, 100.0)
    opens = closes.copy()
    highs = closes.copy()
    lows = closes.copy()
    vols = np.full(N, 10.0)
    if pin_bar is not None:
        lows[pin_bar] = LIMIT_PRICE - 0.5   # 插针向下触及限价买单
    idx = pd.date_range("2024-01-01", periods=N, freq="1h", tz="UTC")
    data = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vols}, index=idx)
    series = precompute_indicator_series(closes, opens, highs, lows, vols)
    return data, series, closes, opens, highs, lows


def _run(pin_bar=EMIT_BAR + 3):
    data, series, closes, opens, highs, lows = _make_inputs(pin_bar)
    cfg = BacktestConfig(symbol="BTC/USDT", timeframe="1h", strategy_name="limit_tick_test",
                         limit_order_model="partial")
    strategy = _LimitTickStrategy()
    res = run_matching_loop(data=data, cfg=cfg, strategy=strategy, closes=closes,
                            opens=opens, highs=highs, lows=lows, series=series,
                            need_sr=False)
    return res, strategy.fills


def test_limit_order_gets_full_try_count_after_many_expired_orders():
    """挂单必须活满 3 次尝试：第 3 次触及才成交。

    旧实现下这张单的 id 大概率撞上前面 45 根里某张已耗尽尝试的老单，第一根没触及
    就被丢弃，第 53 根的插针也就无人认领。
    """
    res, fills = _run()
    assert fills == [(EMIT_BAR + 3, "buy", round(LIMIT_PRICE, 6))], (
        f"限价挂单应在第 3 次尝试触及成交，实际成交记录 {fills}"
        "（挂单超时计数疑似被其他挂单污染）")
    assert res["position"] > 0
    assert res["lots"][0][1] == LIMIT_PRICE


def test_no_fill_when_limit_never_touched():
    """没有任何一根触及挂单价时，所有单安静超时，不留持仓也不花钱。"""
    res, fills = _run(pin_bar=None)
    assert fills == []
    assert res["position"] == 0.0
    assert res["cash"] == BacktestConfig().start_cash


def test_fill_on_first_attempt_not_blocked():
    """第一次尝试就触及 → 立即成交（超时计数不得干扰正常撮合）。"""
    res, fills = _run(pin_bar=EMIT_BAR + 1)
    assert fills == [(EMIT_BAR + 1, "buy", round(LIMIT_PRICE, 6))]


def test_matching_is_repeatable():
    """两轮回测逐笔一致：超时计数既不跨挂单泄漏，也不跨运行泄漏。"""
    a, af = _run()
    b, bf = _run()
    assert af == bf
    assert a["cash"] == b["cash"]
    assert a["position"] == b["position"]
    assert a["equity_curve"] == b["equity_curve"]
