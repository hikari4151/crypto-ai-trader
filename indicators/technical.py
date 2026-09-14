"""常用技术指标 + 关键位(S/R) + 价格行为特征（纯 numpy 实现）。"""
from typing import Any
import numpy as np

# 共享口径常量：S/R 与价格行为序列的回测滑动窗口口径（原散落硬编码 120）。
# 两个回测引擎（engine/fast_engine）与 indicators 序列必须一致，否则分叉。
SR_LOOKBACK = 120          # 关键位/价格行为的滑动窗口（a=max(0,i-119), b=i+1）
SR_WINDOW = 10             # 摆动高低点检测窗口（默认）
SR_MIN_TOUCHES = 2         # 关键位最少触碰次数（默认）
PA_MIN_BARS = 21           # price_action_features 最少K线数




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
    # P2-10：总体标准差（ddof=0），与 vectorized/IncrIndicators 口径一致——
    # 曾 np.std 默认 ddof=1，BB 上下轨与两套引擎系统性偏差 ~2.5%（period=20），
    # factor_signal 的 bb_pos/vol_break 信号在事件引擎与向量引擎间漂移
    for i in range(period - 1, len(v)):
        std[i] = v[i - period + 1:i + 1].std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def atr(highs, lows, closes, period: int = 14) -> np.ndarray:
    """真实波幅均值（Wilder 平滑）——波动率基准，供 ATR 归一化与风险提示。"""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out[period] = tr[1:period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _find_pivot_points(highs, lows, window: int = 10) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """全量摆动高低点检测：返回 (piv_high, piv_low)，每项为 [(idx, price), ...]。

    摆动高低点的判定只用局部窗口（[i-window, i+window]），与窗口起点无关，
    因此可对全序列一次预计算，供 support_resistance_series 复用（避免每根K线
    在 120 根滑动窗口上重复检测 O(m×window)）。
    """
    piv_high: list[tuple[int, float]] = []
    piv_low: list[tuple[int, float]] = []
    for i in range(window, len(highs) - window):
        seg_h = highs[i - window:i + window + 1]
        seg_l = lows[i - window:i + window + 1]
        if highs[i] >= seg_h.max() and highs[i] > seg_h.mean():
            piv_high.append((i, float(highs[i])))
        if lows[i] <= seg_l.min() and lows[i] < seg_l.mean():
            piv_low.append((i, float(lows[i])))
    return piv_high, piv_low


def _cluster_pivots(points, min_touches: int) -> list[dict]:
    """把摆动点按其价格（±0.3%）聚合成支撑/阻力位。"""
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


def _sr_result(sup, res, last: float) -> dict:
    """由已聚类的支撑/阻力位构造 S/R 结果结构（两函数共用，保证口径一致）。"""
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


def support_resistance(highs, lows, closes, window: int = 10, min_touches: int = 2) -> dict:
    """基于摆动高低点聚类识别关键支撑/阻力位（价格行为学核心）。"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    piv_high, piv_low = _find_pivot_points(highs, lows, window)
    sup = _cluster_pivots(piv_low, min_touches)
    res = _cluster_pivots(piv_high, min_touches)
    return _sr_result(sup, res, float(closes[-1]))


def support_resistance_series(highs, lows, closes, window: int = 10, min_touches: int = 2) -> list[dict]:
    """向量化预计算全量 S/R 序列（每根K线位置的结果），与输入等长。

    第 i 项 = support_resistance(highs[max(0,i-119):i+1], ...) 的窗口口径逐位一致
    （修 engine 之前的逐K线调用语义）。优化思路：摆动高低点判定只依赖局部窗口，
    与窗口起点无关——对全序列预计算一次 (O(n×window)，n=5000 时 ~5e4 次比较，
    而逐K线重算是 O(n×lookback×window) ≈ 6e6)，此后每根K线仅需：
      1) 从全量摆动点中过滤出落在 [a+window, b-window) 的（O(1) 二分/线性）
      2) 对窗口内摆动点做聚类（与 support_resistance 完全一致）

    返回 _sr_result([], [], ...) 项：摆动点在窗口内的数量不足以启用时
    （hi <= lo，即 i < 2*window 的暖机期），返回的 dict 中 support/resistance
    等键值均为 None（与 support_resistance 的 len<2*window 行为逐位一致）。
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    lookback = SR_LOOKBACK  # 与两个回测引擎的滑动窗口口径一致（a=max(0,i-119), b=i+1）
    piv_high, piv_low = _find_pivot_points(highs, lows, window)
    # 排好序的摆动点下标数组（_find_pivot_points 天然递增），供窗口过滤用
    piv_high_idx = np.array([p[0] for p in piv_high], dtype=np.int64)
    piv_low_idx = np.array([p[0] for p in piv_low], dtype=np.int64)

    result = [None] * n
    # P2-13：窗口 [lo, hi) 随 i 单调右移（lo、hi 均单调不减），用双指针推进
    # 替代每根全量 flatnonzero + 元组重建（原实现 n=5000 时 ~128ms → 切片 ~15ms）。
    # lo_h/hi_h 指向 piv_high_idx 中 [lo, hi) 区间的边界，随 i 只前进不回退，
    # 总体 O(n + 摆动点总数)；lo_l/hi_l 同作用于 piv_low_idx。
    lo_h = hi_h = 0
    lo_l = hi_l = 0
    for i in range(n):
        a = max(0, i - (lookback - 1))
        b = i + 1
        # 落在窗口内的摆动点：[a+window, b-window)（与 support_resistance 的摆动检测区间一致，
        # 该函数对切片 highs[a:b] 在 [window, L-window) 内检测摆动点，映射回全序列即此区间）
        lo, hi = a + window, b - window
        if hi <= lo:
            result[i] = _sr_result([], [], float(closes[i]))
            continue
        # 双指针单调推进（lo/hi 单调不减，指针只前进）
        while lo_h < len(piv_high_idx) and piv_high_idx[lo_h] < lo:
            lo_h += 1
        while hi_h < len(piv_high_idx) and piv_high_idx[hi_h] < hi:
            hi_h += 1
        while lo_l < len(piv_low_idx) and piv_low_idx[lo_l] < lo:
            lo_l += 1
        while hi_l < len(piv_low_idx) and piv_low_idx[hi_l] < hi:
            hi_l += 1
        # 直接切片元组列表（复用 _find_pivot_points 结果，免去重建）
        piv_high_win = piv_high[lo_h:hi_h]
        piv_low_win = piv_low[lo_l:hi_l]
        sup = _cluster_pivots(piv_low_win, min_touches)
        res = _cluster_pivots(piv_high_win, min_touches)
        result[i] = _sr_result(sup, res, float(closes[i]))
    return result


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


def price_action_features_series(highs, lows, closes, opens, volumes) -> list[dict]:
    """预计算全量价格行为特征序列（每根K线位置的结果），与输入等长。

    第 i 项 = price_action_features(ohlcv[max(0,i-119):i+1]) 逐位一致
    （修 engine 之前的逐K线调用语义；o/h/l/c/v 与回测引擎的数组口径一致）。
    优化点：streak（连续同向收盘）用方向差分一次递推得到 O(n)，而原始逐K线
    每次从窗口尾部回扫 O(window)；breakout 与 vol_ratio 只依赖最近 20/19 根
    （窗口恒 ≥21 根），用滚动窗口预计算后每根 O(1) 取值。

    暖机期（不足 21 根K线，与 price_action_features 的 len<21 返回 {} 一致）
    的项为 {}。
    """
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)
    lookback = SR_LOOKBACK  # 与两个回测引擎的滑动窗口口径一致（a=max(0,i-119), b=i+1）

    body = closes - opens
    rng = (highs - lows) + 1e-12
    # streak：连续同向收盘递推（direction[i] = closes[i]>=closes[i-1]，与
    # price_action_features 的 (closes[i]>=closes[i-1]) 逐位一致，含等号）
    direction = np.zeros(n, dtype=np.int8)
    direction[1:] = (closes[1:] >= closes[:-1]).astype(np.int8)
    streak_arr = np.ones(n, dtype=int)
    for i in range(1, n):
        streak_arr[i] = streak_arr[i - 1] + 1 if direction[i] == direction[i - 1] else 1
    # breakout 窗口极值：closes[i-n:i], n=min(20,i-a)——对有效 bar（i>=20）恒为 20；
    # 用 20 根滑动窗口预计算（sliding_window_view(w)[i-20] = closes[i-20:i]）
    if n >= 21:
        from numpy.lib.stride_tricks import sliding_window_view
        _sw = sliding_window_view(closes, 20)  # 形状 (n-19, 20)，_sw[j] = closes[j:j+20]
        win_max = np.max(_sw, axis=1)          # win_max[i-20] = max(closes[i-20:i])
        win_min = np.min(_sw, axis=1)
    else:
        win_max = win_min = None

    result: list[dict] = []
    # P2-13：vol_ratio 的 19 根滚动均量用 cumsum 一次预计算（原逐根切片 mean O(19n)），
    # 口径不变：volumes[max(0,i-19):i].mean()，i>=20 恒为前 19 根。
    if n >= 21:
        _cs = np.concatenate(([0.0], np.cumsum(volumes)))  # _cs[k] = sum(volumes[:k])
        vol_mean19 = np.full(n, np.nan)
        # rolling_sum[i] = sum(volumes[max(0,i-19):i])；i>=20 时 = _cs[i]-_cs[i-19]
        # 对齐：vol_mean19[20:] ↔ i=20..n-1 ↔ _cs[20:n]-_cs[1:n-19]
        vol_mean19[20:] = (_cs[20:n] - _cs[1:n - 19]) / 19.0
    for i in range(n):
        if i < 20:  # 需要 21 根K线（price_action_features 的 len<21 判据）
            result.append({})
            continue
        a = max(0, i - (lookback - 1))
        # streak 只统计窗口内 [a+1..i]（原始回扫在窗口相对索引 1 停，不查窗口首根），
        # 故上限为 i-a
        streak = min(int(streak_arr[i]), i - a)
        vm = vol_mean19[i]
        vol_ratio = float(volumes[i] / (vm if vm > 0 else 1e-12))
        prev_max = float(win_max[i - 20])
        prev_min = float(win_min[i - 20])
        result.append({
            "last_candle": "bullish" if body[i] > 0 else "bearish",
            "body_ratio": round(float(abs(body[i]) / rng[i]), 3),
            "upper_wick": round(float((highs[i] - max(closes[i], opens[i])) / rng[i]), 3),
            "lower_wick": round(float((min(closes[i], opens[i]) - lows[i]) / rng[i]), 3),
            "volume_ratio": round(vol_ratio, 3),
            "streak": int(streak),
            "breakout_high": bool(closes[i] >= prev_max),
            "breakout_low": bool(closes[i] <= prev_min),
            "range_pct": round(float((highs[i] - lows[i]) / closes[i] * 100), 4),
        })
    return result


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
    sr = support_resistance(highs, lows, closes, window=SR_WINDOW, min_touches=SR_MIN_TOUCHES)
    pa = price_action_features(ohlcv)
    # 量比（与 fast_engine 的 vol_ratio 口径一致：当前量/5期均量，暖机期=1.0）
    vol_ma5 = vols[-5:].mean() if len(vols) >= 5 else 0.0
    vol_ratio = float(vols[-1] / vol_ma5) if vol_ma5 > 0 else 1.0
    # ---- 增强指标（ATR/%B/MA 斜率/带宽）：供 AI 上下文与风险提示 ----
    atr_arr = atr(highs, lows, closes, 14)
    atr_now = _nan(atr_arr[last])
    atr_pct = round(atr_now / closes[last] * 100, 3) if atr_now > 0 else 0.0
    # %B：价格在布林带内位置（0=下轨 1=上轨）；带宽（收口/扩张）
    bb_u, bb_m, bb_l = _nan(up[last]), _nan(mid[last]), _nan(low[last])
    bb_position = float(np.clip((closes[last] - bb_l) / (bb_u - bb_l + 1e-12), 0, 1)) if bb_u > bb_l else 0.5
    bb_width = round((bb_u - bb_l) / bb_m * 100, 3) if bb_m > 0 else 0.0
    # MA 斜率（ATR 归一化）：近 8 根 MA10 变化 / ATR——趋势强度证据
    ma_slope = 0.0
    if last >= 8 and not np.isnan(ma_fast[last]) and not np.isnan(ma_fast[last - 8]) and atr_now > 0:
        ma_slope = round((ma_fast[last] - ma_fast[last - 8]) / atr_now, 3)
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
        "atr_pct": atr_pct,          # 波动率（ATR/close %）
        "bb_position": bb_position,  # %B：布林带内位置 0~1
        "bb_width": bb_width,        # 带宽 %（收口/扩张）
        "ma_slope": ma_slope,        # MA10 斜率（ATR 归一化）
        "volume": float(vols[-1]),
        "vol_ratio": vol_ratio,
        "high": float(highs[-1]),
        "low": float(lows[-1]),
        "sr": sr,
        "pa": pa,
        "candles_count": len(ohlcv),
    }


def _nan(x: Any) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)