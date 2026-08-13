"""常用技术指标 + 关键位(S/R) + 价格行为特征（纯 numpy 实现）。"""
from typing import Any
import numpy as np


def sma(values: Any, period: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < period:
        return out
    c = np.cumsum(np.insert(v, 0, 0.0))
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def ema(values: Any, period: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = v[i] * k + out[i - 1] * (1 - k)
    return out


def macd(values: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ef = ema(values, fast)
    es = ema(values, slow)
    dif = ef - es
    dea = ema(np.nan_to_num(dif, nan=0.0), signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def rsi(values: Any, period: int = 14) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < period + 1:
        return out
    diff = np.diff(v)
    gains = np.clip(diff, 0, None)
    losses = np.clip(-diff, 0, None)
    avg_g = gains[:period].mean()
    avg_l = losses[:period].mean()
    out[period] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)
    for i in range(period + 1, len(v)):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)
    return out


def bollinger(values: Any, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(values, period)
    v = np.asarray(values, dtype=float)
    std = np.full(len(v), np.nan)
    for i in range(period - 1, len(v)):
        std[i] = v[i - period + 1:i + 1].std()
    return mid + num_std * std, mid, mid - num_std * std


def support_resistance(highs, lows, closes, window: int = 10, min_touches: int = 2) -> dict:
    """基于摆动高低点聚类识别关键支撑/阻力位（价格行为学核心）。"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    piv_high, piv_low = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window:i + window + 1]
        seg_l = lows[i - window:i + window + 1]
        if highs[i] >= seg_h.max() and highs[i] > seg_h.mean():
            piv_high.append((i, float(highs[i])))
        if lows[i] <= seg_l.min() and lows[i] < seg_l.mean():
            piv_low.append((i, float(lows[i])))

    def _cluster(points):
        levels = []
        for idx, price in points:
            hit = False
            for lvl in levels:
                if abs(lvl["price"] - price) / price < 0.003:
                    lvl["touches"] += 1
                    lvl["last_idx"] = max(lvl["last_idx"], idx)
                    hit = True
                    break
            if not hit:
                levels.append({"price": price, "touches": 1, "last_idx": idx})
        return [l for l in levels if l["touches"] >= min_touches]

    sup = _cluster(piv_low)
    res = _cluster(piv_high)
    last = float(closes[-1])
    near_sup = max([l["price"] for l in sup if l["price"] < last], default=None)
    near_res = min([l["price"] for l in res if l["price"] > last], default=None)
    # 刚被突破的阻力位：低于当前价的最高阻力（此前是阻力、现被上穿）。
    # 突破入场判据必须用它——"上方最近的阻力"在数学上永远无法被突破。
    broken_res = max([l["price"] for l in res if l["price"] < last], default=None)
    return {
        "support": None if near_sup is None else round(near_sup, 4),
        "resistance": None if near_res is None else round(near_res, 4),
        "broken_resistance": None if broken_res is None else round(broken_res, 4),
        "at_support": bool(near_sup and (last - near_sup) / last < 0.005),
        "at_resistance": bool(near_res and (near_res - last) / last < 0.005),
        "support_levels": [round(l["price"], 4) for l in sorted(sup, key=lambda x: -x["price"]) if l["price"] <= last][:5],
        "resistance_levels": [round(l["price"], 4) for l in sorted(res, key=lambda x: x["price"]) if l["price"] >= last][:5],
        "distance_to_support": None if near_sup is None else round((last - near_sup) / last, 5),
        "distance_to_resistance": None if near_res is None else round((near_res - last) / last, 5),
    }


def price_action_features(ohlcv) -> dict:
    """蜡烛形态、量价关系、突破与连续趋势等价格行为特征。"""
    if len(ohlcv) < 21:
        return {}
    closes = np.array([c[4] for c in ohlcv], dtype=float)
    opens = np.array([c[1] for c in ohlcv], dtype=float)
    highs = np.array([c[2] for c in ohlcv], dtype=float)
    lows = np.array([c[3] for c in ohlcv], dtype=float)
    vols = np.array([c[5] for c in ohlcv], dtype=float)
    body = closes - opens
    rng = (highs - lows) + 1e-12
    last = -1
    is_bull = body[last] > 0
    vol_ratio = float(vols[-1] / (vols[-20:-1].mean() + 1e-12))
    streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if (closes[i] >= closes[i - 1]) == (closes[-1] >= closes[-2]):
            streak += 1
        else:
            break
    n = min(20, len(closes) - 1)
    return {
        "last_candle": "bullish" if is_bull else "bearish",
        "body_ratio": round(float(abs(body[last]) / rng[last]), 3),
        "upper_wick": round(float((highs[last] - max(closes[last], opens[last])) / rng[last]), 3),
        "lower_wick": round(float((min(closes[last], opens[last]) - lows[last]) / rng[last]), 3),
        "volume_ratio": round(vol_ratio, 3),
        "streak": int(streak),
        "breakout_high": bool(closes[-1] >= max(closes[-n - 1:-1])),
        "breakout_low": bool(closes[-1] <= min(closes[-n - 1:-1])),
        "range_pct": round(float((highs[-1] - lows[-1]) / closes[-1] * 100), 4),
    }


def compute_latest(ohlcv: list) -> dict[str, Any]:
    """从 K 线列表计算最新一期指标 + 关键位 + 价格行为特征。"""
    if not ohlcv:
        return {}
    closes = np.array([c[4] for c in ohlcv], dtype=float)
    vols = np.array([c[5] for c in ohlcv], dtype=float)
    highs = np.array([c[2] for c in ohlcv], dtype=float)
    lows = np.array([c[3] for c in ohlcv], dtype=float)
    opens = np.array([c[1] for c in ohlcv], dtype=float)
    ma_fast = sma(closes, 10)
    ma_slow = sma(closes, 30)
    dif, dea, hist = macd(closes)
    r = rsi(closes, 14)
    up, mid, low = bollinger(closes, 20, 2.0)
    last = len(closes) - 1
    sr = support_resistance(highs, lows, closes, window=10, min_touches=2)
    pa = price_action_features(ohlcv)
    # 量比（与 fast_engine 的 vol_ratio 口径一致：当前量/5期均量，暖机期=1.0）
    vol_ma5 = vols[-5:].mean() if len(vols) >= 5 else 0.0
    vol_ratio = float(vols[-1] / vol_ma5) if vol_ma5 > 0 else 1.0
    return {
        "close": float(closes[last]),
        "open": float(opens[last]),
        "ma_fast": _nan(ma_fast[last]),
        "ma_slow": _nan(ma_slow[last]),
        "ma10_series": ma_fast.tolist(),
        "ma30_series": ma_slow.tolist(),
        "macd": _nan(dif[last]),
        "macd_signal": _nan(dea[last]),
        "macd_hist": _nan(hist[last]),
        "rsi": _nan(r[last]),
        "bb_upper": _nan(up[last]),
        "bb_mid": _nan(mid[last]),
        "bb_lower": _nan(low[last]),
        "volume": float(vols[-5:].sum()),
        "vol_ratio": vol_ratio,
        "high": float(highs[-1]),
        "low": float(lows[-1]),
        "sr": sr,
        "pa": pa,
        "candles_count": len(ohlcv),
    }


def _nan(x: Any) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)