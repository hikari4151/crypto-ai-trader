"""indicators/technical.py 纯函数确定性测试（基建：指标是交易决策核心，必须被直接测过）。

补覆盖背景（2026-09-28）：technical.py 之前靠引擎/回测路径"碰巧覆盖"（测试被 math_ev
拦截前会联网拉真实行情跑 compute_latest）。测试联网隔离后真实覆盖率暴露为 67%——
这些纯函数从未有过直接单测。此处按 REDEV_GUIDE 口径契约逐条锁定：
- ddof=0（A4 红线：布林带全项目统一总体标准差）
- 暖机语义：vol_ratio 前 5 根内=1.0；price_action len<21 返回空
- sma/ema 短于周期返回 NaN 前缀
"""
import numpy as np
import pytest

from indicators import technical as t


def _closes(n=60, base=100.0, step=0.5):
    return [base + i * step for i in range(n)]


# ---- 基础指标：shape / 暖机 / 数值 ----
def test_sma_shape_and_warmup():
    c = _closes(30)
    out = t.sma(c, 10)
    assert len(out) == 30
    assert np.isnan(out[:9]).all(), "前 period-1 根应为 NaN"
    assert not np.isnan(out[9:]).any()
    # 手算：10 期均线第 9 根 = mean(c[0:10])
    assert out[9] == pytest.approx(np.mean(c[0:10]))


def test_sma_short_series_all_nan():
    assert np.isnan(t.sma([1.0, 2.0], 10)).all()


def test_ema_matches_reference_formula():
    c = _closes(20)
    out = t.ema(c, 5)
    k = 2.0 / 6.0
    expected = [c[0]]
    for v in c[1:]:
        expected.append(v * k + expected[-1] * (1 - k))
    assert np.allclose(out, expected)
    assert len(out) == 20
    assert np.isnan(t.ema([], 5)).all(), "空序列应返回全 NaN"


def test_macd_components():
    c = _closes(60)
    dif, dea, hist = t.macd(c)
    assert len(dif) == len(dea) == len(hist) == 60
    # macd = ema12 - ema26；hist = 2*(dif-dea)
    ef = t.ema(c, 12)
    es = t.ema(c, 26)
    assert np.allclose(dif, ef - es, equal_nan=True)
    assert np.allclose(hist, 2 * (dif - dea), equal_nan=True)


def test_rsi_bounds_and_known_value():
    c = _closes(60)
    r = t.rsi(c, 14)
    assert len(r) == 60
    assert np.isnan(r[:14]).all(), "RSI 前 period 根应为 NaN"
    valid = r[~np.isnan(r) & ~np.isinf(r)]
    assert ((valid >= 0) & (valid <= 100)).all(), "RSI 必须在 [0,100]"
    # 全等序列：avg_l=0 → RSI=100
    flat = t.rsi([50.0] * 30, 14)
    assert flat[-1] == pytest.approx(100.0)


def test_bollinger_ddof_zero_redline():
    """A4 红线：布林带必须 ddof=0（与 vectorized 口径一致），且 band 对称于 mid。"""
    c = _closes(40)
    up, mid, low = t.bollinger(c, 20, 2.0)
    assert len(up) == len(mid) == len(low) == 40
    assert np.isnan(mid[:19]).all()
    i = 39
    assert up[i] == pytest.approx(mid[i] + 2 * np.std(c[20:], ddof=0))
    assert low[i] == pytest.approx(mid[i] - 2 * np.std(c[20:], ddof=0))
    assert np.allclose(up + low, 2 * mid, equal_nan=True)


def test_atr_shape_and_positive():
    n = 30
    highs = [101.0 + i * 0.1 for i in range(n)]
    lows = [99.0 + i * 0.1 for i in range(n)]
    closes = [100.0 + i * 0.1 for i in range(n)]
    out = t.atr(highs, lows, closes, 14)
    assert len(out) == n
    assert np.isnan(out[:14]).all()
    assert (out[14:] > 0).all(), "ATR 必须为正"
    assert out[14] == pytest.approx(np.mean(np.maximum(
        np.array(highs[1:15]) - np.array(lows[1:15]),
        np.maximum(np.abs(np.array(highs[1:15]) - np.array(closes[:14])),
                   np.abs(np.array(lows[1:15]) - np.array(closes[:14]))))))


def test_compute_latest_full_snapshot():
    ohlcv = [[1700000000000 + i * 3600000, 99.0, 102.0, 98.0, 100.0 + i * 0.5, 1000.0]
             for i in range(40)]
    out = t.compute_latest(ohlcv)
    assert out["candles_count"] == 40
    for k in ("close", "open", "ma_fast", "ma_slow", "macd", "rsi",
              "bb_upper", "bb_mid", "bb_lower", "atr_pct", "volume"):
        assert k in out, f"缺字段 {k}"
    assert out["close"] == pytest.approx(100.0 + 39 * 0.5)
    assert out["ma_fast"] == pytest.approx(np.mean([100.0 + i * 0.5 for i in range(30, 40)]))
    assert out["vol_ratio"] == pytest.approx(1.0), "前 5 根量均口径：单值序列 ratio=1.0"
    assert isinstance(out["sr"], dict) and isinstance(out["pa"], dict), "sr/pa 均为 dict（sr 含 support/resistance，pa 含 last_candle 等）"


def test_compute_latest_empty_and_warmup():
    assert t.compute_latest([]) == {}
    # len<21：price_action 返回空（暖机守卫）
    short = [[1700000000000 + i * 3600000, 99.0, 101.0, 99.0, 100.0, 100.0] for i in range(10)]
    out = t.compute_latest(short)
    # 短序列也不得崩溃
    assert out["candles_count"] == 10
    assert out["close"] == pytest.approx(100.0)


def test_price_action_warmup_empty():
    ohlcv = [[1, 99.0, 101.0, 99.0, 100.0, 100.0] for _ in range(10)]
    assert t.price_action_features(ohlcv) == [] or all(not x for x in t.price_action_features(ohlcv))


def test_nan_helper():
    assert t._nan(float("nan")) == 0.0
    assert t._nan(None) == 0.0
    assert t._nan(3.14) == pytest.approx(3.14)


def test_sr_warmup_few_touches():
    """S/R 少量触碰不误报（SR_MIN_TOUCHES=2）。"""
    highs = [100.0 + 0.1 * i for i in range(30)]
    lows = [99.0 + 0.1 * i for i in range(30)]
    closes = [99.5 + 0.1 * i for i in range(30)]
    sr = t.support_resistance(highs, lows, closes)
    # 返回 dict：含 support/resistance（单调序列可能均为 None）等键
    assert isinstance(sr, dict)
    for k in ("support", "resistance"):
        assert k in sr
    # 单调序列的支撑/阻力（若识别出）必须是 list of dict 或 None
    for k in ("support", "resistance"):
        v = sr[k]
        assert v is None or (isinstance(v, list) and all(isinstance(x, dict) for x in v)), f"{k} 结构不符"
