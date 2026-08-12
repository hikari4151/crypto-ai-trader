"""确定性方向复算引擎：从 K 线序列程序化计算市场方向（五信号投票）。

设计动机（借鉴 PA Agent 的"AI 提出、程序裁决"思路）：
AI 解读的方向(bias)是可被 K 线结构客观验证的——本模块从原始 K 线独立
复算方向，与 AI 输出对比。方向矛盾时程序强制对齐或要求 AI 解释，
把 AI 行情解读从"格式校验"升级为"内容校验"，防止 AI 凭感觉给方向。

五个客观信号（各 +1/0/-1）：
  S1 EMA 斜率：EMA(10) 近端斜率，超过 ATR 死区才给信号（抗噪声）
  S2 收盘重心：近半窗口加权收盘均值 vs 远半（近端权重线性递减）
  S3 波段结构：摆动点 HH+HL / LL+LH（双K枢轴，抗噪）
  S4 趋势棒占优：近 N 根中实体大且收向极点的趋势棒方向占比
  S5 重叠率：K线重叠少→趋势确认（与 S1 同向加成）；重叠多→区间（中性）

总分阈值：score>=2 看多，<=-2 看空，否则中性。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# 方向阈值：|score| 达到该值才判为有方向
_BULL_THRESHOLD = 2
_BEAR_THRESHOLD = -2

# 各信号配置
_EMA_PERIOD = 10
_EMA_SLOPE_LOOKBACK = 8
_EMA_SLOPE_DEAD_ATR = 0.5        # 斜率小于 0.5×ATR 视为平（死区）
_WEIGHT_DECAY = 0.85             # 收盘重心近端权重衰减
_SWING_LOOKBACK = 40             # 波段结构窗口
_TREND_BAR_WINDOW = 12           # 趋势棒统计窗口
_TREND_BAR_BODY_MIN = 0.55       # 实体占比下限
_TREND_BAR_DOMINANCE = 1.5       # 多头/空头比下限
_OVERLAP_LOW = 0.40              # 重叠低于此=趋势确认
_OVERLAP_HIGH = 0.62             # 重叠高于此=区间


def _ema(v: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = v[i] * k + out[i - 1] * (1 - k)
    return out


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    n = len(closes)
    if n < period + 1:
        return 0.0
    prev_c = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - prev_c), np.abs(lows[1:] - prev_c)))
    return float(tr[-period:].mean())


def _swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = _SWING_LOOKBACK) -> list[dict]:
    """双K枢轴摆动点（左右各 2 根严格更高/更低），返回 {kind, price, idx}。"""
    pts: list[dict] = []
    n = len(highs)
    start = max(2, n - lookback)
    for i in range(start, n - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
           highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            pts.append({"kind": "high", "price": float(highs[i]), "idx": i})
        elif lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
                lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            pts.append({"kind": "low", "price": float(lows[i]), "idx": i})
    return pts


def _swing_structure_score(pts: list[dict]) -> int:
    """HH+HL(看多) / LL+LH(看空) 结构打分。

    高点序列持续抬高（看多+1），低点序列持续抬高（看多+1），
    反之各 -1。总评分 clip 到 [-1, 1]。
    """
    highs = [p["price"] for p in pts if p["kind"] == "high"]
    lows = [p["price"] for p in pts if p["kind"] == "low"]
    score = 0
    if len(highs) >= 2:
        rising = sum(1 for a, b in zip(highs, highs[1:]) if b > a)
        falling = sum(1 for a, b in zip(highs, highs[1:]) if b < a)
        if rising >= len(highs) - 1 and rising >= 1:
            score += 1
        elif falling >= len(highs) - 1 and falling >= 1:
            score -= 1
    if len(lows) >= 2:
        rising = sum(1 for a, b in zip(lows, lows[1:]) if b > a)
        falling = sum(1 for a, b in zip(lows, lows[1:]) if b < a)
        if rising >= len(lows) - 1 and rising >= 1:
            score += 1
        elif falling >= len(lows) - 1 and falling >= 1:
            score -= 1
    return max(-1, min(1, score))


def _trend_bar_score(closes: np.ndarray, opens: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, window: int = _TREND_BAR_WINDOW) -> int:
    """趋势棒占优：实体大且收向极点的棒方向占比。"""
    n = len(closes)
    if n < 5:
        return 0
    seg_c = closes[-window:]
    seg_o = opens[-window:]
    seg_h = highs[-window:]
    seg_l = lows[-window:]
    body = np.abs(seg_c - seg_o)
    rng = (seg_h - seg_l) + 1e-12
    body_ratio = body / rng
    bull = (seg_c > seg_o) & (body_ratio >= _TREND_BAR_BODY_MIN)
    bear = (seg_c < seg_o) & (body_ratio >= _TREND_BAR_BODY_MIN)
    nb, ns = int(bull.sum()), int(bear.sum())
    total = nb + ns
    if total == 0:
        return 0
    if nb / total >= _TREND_BAR_DOMINANCE / (1 + _TREND_BAR_DOMINANCE) and nb >= 2:
        return 1
    if ns / total >= _TREND_BAR_DOMINANCE / (1 + _TREND_BAR_DOMINANCE) and ns >= 2:
        return -1
    return 0


def _overlap_score(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                   window: int = 10) -> float:
    """近 window 根平均 K 线重叠率（0=无重叠，1=完全重叠）。"""
    n = len(closes)
    if n < 3:
        return 0.0
    seg_h, seg_l = highs[-window:], lows[-window:]
    ratios = []
    for i in range(1, len(seg_h)):
        hi = min(seg_h[i], seg_h[i - 1])
        lo = max(seg_l[i], seg_l[i - 1])
        overlap = max(0.0, hi - lo)
        denom = max(seg_h[i], seg_h[i - 1]) - min(seg_l[i], seg_l[i - 1])
        if denom > 1e-12:
            ratios.append(overlap / denom)
    return float(np.mean(ratios)) if ratios else 0.0


def compute_direction(candles: list, lookback: int = 80) -> dict[str, Any]:
    """从 K 线序列程序化计算方向。

    candles: [[ts, open, high, low, close, volume], ...]（旧→新顺序）
    返回 {direction, score, confidence, signals, reasons}
      direction: 'long' | 'short' | 'neutral'
      score: 五信号总分
      confidence: 0~1（|score| 归一化 + 信号一致性）
      signals: 每个信号的明细 [{name, sign, reason}]
    """
    if not candles or len(candles) < 8:
        return {"direction": "neutral", "score": 0, "confidence": 0.0,
                "signals": [], "reasons": [], "enough_data": False}

    closes = np.array([c[4] for c in candles], dtype=float)
    opens = np.array([c[1] for c in candles], dtype=float)
    highs = np.array([c[2] for c in candles], dtype=float)
    lows = np.array([c[3] for c in candles], dtype=float)

    # 取近端窗口（默认 80 根），老数据用于指标预热
    start = max(0, len(closes) - lookback - 30)
    closes_w = closes[start:]
    opens_w = opens[start:]
    highs_w = highs[start:]
    lows_w = lows[start:]

    atr = _atr(highs_w, lows_w, closes_w)
    signals: list[dict] = []
    reasons: list[str] = []

    # S1: EMA 斜率
    ema10 = _ema(closes_w, _EMA_PERIOD)
    if atr > 0 and len(ema10) > _EMA_SLOPE_LOOKBACK + 1:
        slope = (ema10[-1] - ema10[-1 - _EMA_SLOPE_LOOKBACK])
        if slope > atr * _EMA_SLOPE_DEAD_ATR:
            signals.append({"name": "ema_slope", "sign": 1, "reason": f"EMA10 近{_EMA_SLOPE_LOOKBACK}根上行 {slope:.2g}"})
        elif slope < -atr * _EMA_SLOPE_DEAD_ATR:
            signals.append({"name": "ema_slope", "sign": -1, "reason": f"EMA10 近{_EMA_SLOPE_LOOKBACK}根下行 {slope:.2g}"})
        else:
            signals.append({"name": "ema_slope", "sign": 0, "reason": "EMA10 走平（死区内）"})

    # S2: 收盘重心（近半加权 vs 远半）
    n = len(closes_w)
    if n >= 12:
        half = max(6, n // 2)
        weights = np.array([_WEIGHT_DECAY ** i for i in range(half)])
        near_c = closes_w[-half:]
        far_c = closes_w[-2 * half:-half] if len(closes_w) >= 2 * half else closes_w[:half]
        if len(far_c) >= half:
            wsum = weights.sum() + 1e-12
            near_w = float((near_c * weights).sum() / wsum)
            far_w = float((far_c * weights).sum() / wsum)
            if atr > 0:
                diff = near_w - far_w
                if diff > atr * 0.15:
                    signals.append({"name": "close_center", "sign": 1, "reason": "收盘重心上移（近端强于远端）"})
                elif diff < -atr * 0.15:
                    signals.append({"name": "close_center", "sign": -1, "reason": "收盘重心下移（近端弱于远端）"})
                else:
                    signals.append({"name": "close_center", "sign": 0, "reason": "收盘重心平移"})

    # S3: 波段结构
    pts = _swing_points(highs_w, lows_w)
    s3 = _swing_structure_score(pts)
    signals.append({"name": "swing_structure", "sign": s3,
                    "reason": f"摆动点结构评分{s3:+d}（{len(pts)}个枢轴）"})

    # S4: 趋势棒占优
    s4 = _trend_bar_score(closes_w, opens_w, highs_w, lows_w)
    signals.append({"name": "trend_bar", "sign": s4, "reason": f"趋势棒占优评分{s4:+d}"})

    # S5: 重叠率（与 S1 联动，增强趋势确认或压向中性）
    overlap = _overlap_score(closes_w, highs_w, lows_w)
    s1 = next((s["sign"] for s in signals if s["name"] == "ema_slope"), 0)
    if overlap <= _OVERLAP_LOW and s1 != 0:
        s5 = s1  # 低重叠确认趋势
        signals.append({"name": "overlap", "sign": s5, "reason": f"K线重叠率{overlap:.2f}低，确认趋势"})
    elif overlap >= _OVERLAP_HIGH:
        s5 = -s1 if s1 != 0 else 0  # 高重叠压向中性
        signals.append({"name": "overlap", "sign": s5, "reason": f"K线重叠率{overlap:.2f}高，倾向区间"})
    else:
        signals.append({"name": "overlap", "sign": 0, "reason": f"K线重叠率{overlap:.2f}中性"})

    score = sum(s["sign"] for s in signals)
    if score >= _BULL_THRESHOLD:
        direction = "long"
    elif score <= _BEAR_THRESHOLD:
        direction = "short"
    else:
        direction = "neutral"

    # 置信度：|score| 映射 + 无反对票加成
    magnitude = abs(score)
    n_signals = max(1, len(signals))
    if direction == "neutral":
        confidence = round(min(0.5, 0.3 + magnitude * 0.05), 3)
    else:
        agrees = sum(1 for s in signals if s["sign"] == (1 if direction == "long" else -1))
        opposing = sum(1 for s in signals if s["sign"] == (-1 if direction == "long" else 1))
        base = min(0.9, 0.45 + magnitude * 0.1)
        if opposing == 0:
            base = min(0.95, base + 0.1)
        confidence = round(base, 3)

    for s in signals:
        reasons.append(s["reason"])

    return {
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "signals": signals,
        "reasons": reasons,
        "enough_data": True,
    }
