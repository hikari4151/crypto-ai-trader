"""流式特征引擎（2026-09-23，P1-② 核心实施，风险隔离阶段）。

背景：rl_adaptive.on_candle 每根对最近 400 根调用 pandas _precompute_features
全量重算 13 特征（实测 ~4.1ms/根，500 根 2.06s）——瓶颈。本引擎改为**增量**：
每根只更新滚动/EWM 状态并输出当前特征行（O(1~w)），目标与 _precompute_features
(np) 输出**逐位一致**（训练端口径不变式，部署行为不漂移）。

等价性难点（pandas 语义逐一对齐）：
- rolling(k).std() 默认 ddof=1、窗口不足 NaN → 特征表 fillna(0.0)
- ewm(alpha, adjust=False) 的 NaN 起始（首个非 NaN 值作为初值）
- rolling(40).rank(pct=True) 的 average-tie 百分位语义
"""
from collections import deque

import numpy as np

_TANH_RET = 5.0
_TANH_VOL = 3.0
_RSI_N = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_BB_N = 20
_RISK_WIN = 40


def _ema_next(old: float, x: float, alpha: float) -> float:
    """pandas ewm(alpha, adjust=False).mean() 语义（NaN 起始：首个非 NaN 作为初值）。"""
    if old is None:
        return x
    return (1.0 - alpha) * old + alpha * x


def _percentile_last(vals: list, cur: float) -> float:
    """rolling(40).rank(pct=True) 最后一行（average-tie 口径；窗口不足→NaN→0）。"""
    n = len(vals)
    if n < _RISK_WIN:
        return 0.0  # pandas min_periods=window：不足 40 输出 NaN → 特征表 fillna(0)
    less = sum(1 for x in vals if x < cur)
    eq = sum(1 for x in vals if x == cur)
    rank = less + 1 + (eq - 1) / 2.0  # average tie
    return rank / n


class StreamingFeatureEngine:
    """增量 13 特征（与 drl.env._precompute_features numpy 口径一致）。"""

    def __init__(self) -> None:
        # 状态窗口：trend_regime 需 60 期 pct_change → 保留 61+；滚动 40/30/20；加余量
        self._nbuf = 96
        self._closes: deque = deque(maxlen=self._nbuf)
        self._volumes: deque = deque(maxlen=self._nbuf)
        self._rets: deque = deque(maxlen=self._nbuf)   # close.pct_change()*100
        self._std20: deque = deque(maxlen=_RISK_WIN)   # ret.rolling(20).std(ddof=1) 历史值
        # EWM 状态（pandas adjust=False，NaN→首个非 NaN 初值）
        self._rsi_gain = None
        self._rsi_loss = None
        self._ema_fast = None
        self._ema_slow = None
        self._dea = None

    def update(self, close: float, volume: float) -> np.ndarray:
        """喂入一根 K 线（close/volume；open/high/low 不被特征使用），返回 13 特征行。"""
        closes = list(self._closes)
        ret = (close / closes[-1] - 1.0) * 100.0 if closes else float("nan")
        self._closes.append(close)
        self._volumes.append(volume)
        self._rets.append(ret)
        c_arr = list(self._closes)
        v_arr = list(self._volumes)
        r_arr = list(self._rets)
        n = len(c_arr)

        # ---- 损失函数侧：增量更新 EWM ----
        d = (c_arr[-1] - c_arr[-2]) if n >= 2 else float("nan")
        gain = d if (d == d and d > 0) else 0.0
        loss = -d if (d == d and d < 0) else 0.0
        if d == d:  # 非 NaN 才更新（pandas ewm 忽略 NaN 输入）
            self._rsi_gain = _ema_next(self._rsi_gain, gain, 1.0 / _RSI_N)
            self._rsi_loss = _ema_next(self._rsi_loss, loss, 1.0 / _RSI_N)
        rsi = 100.0 - 100.0 / (1.0 + (self._rsi_gain or 0.0) / ((self._rsi_loss or 0.0) + 1e-12))

        # ---- MACD：EMA12/26/9（span→alpha=2/(span+1)）----
        a_f = 2.0 / (_MACD_FAST + 1)
        a_s = 2.0 / (_MACD_SLOW + 1)
        self._ema_fast = _ema_next(self._ema_fast, close, a_f)
        self._ema_slow = _ema_next(self._ema_slow, close, a_s)
        dif = (self._ema_fast or 0.0) - (self._ema_slow or 0.0)
        self._dea = _ema_next(self._dea, dif, 2.0 / (_MACD_SIGNAL + 1))
        macd_hist = (dif - (self._dea or 0.0)) * 2.0

        # ---- 滚动窗口各特征（pandas ddof=1 std / 窗口不足 NaN→0）----
        def _roll_mean(vals: list, k: int) -> float:
            if len(vals) < k:
                return float("nan")
            return sum(vals[-k:]) / k

        def _roll_std(vals: list, k: int) -> float:
            if len(vals) < k:
                return float("nan")
            seg = vals[-k:]
            m = sum(seg) / k
            var = sum((x - m) ** 2 for x in seg) / (k - 1)  # ddof=1
            return var ** 0.5

        vol_5 = _roll_std(r_arr, 5)
        vol_ratio = v_arr[-1] / (_roll_mean(v_arr, 5) + 1e-12)
        ma10 = _roll_mean(c_arr, 10)
        ma30 = _roll_mean(c_arr, 30)
        bb_mid = _roll_mean(c_arr, _BB_N)
        bb_sd = _roll_std(c_arr, _BB_N)

        # risk_regime：ret.rolling(20).std().rolling(40).rank(pct=True)
        std20 = _roll_std(r_arr, 20)
        if std20 == std20:
            self._std20.append(std20)
        risk = _percentile_last(list(self._std20), std20) if std20 == std20 else 0.0

        # ---- 组装（与 _precompute_features 同表、同 tanh/clip/fillna(0)）----
        def pct(k: int) -> float:
            return (c_arr[-1] / c_arr[-1 - k] - 1.0) * 100.0 if n > k else float("nan")
        feat = np.array([
            np.tanh(ret / _TANH_RET) if ret == ret else 0.0,
            np.tanh(pct(3) / _TANH_RET) if n > 3 and pct(3) == pct(3) else 0.0,
            np.tanh(pct(5) / _TANH_RET) if n > 5 and pct(5) == pct(5) else 0.0,
            np.tanh(pct(10) / _TANH_RET) if n > 10 and pct(10) == pct(10) else 0.0,
            np.tanh(vol_5 / _TANH_VOL) if vol_5 == vol_5 else 0.0,
            (rsi / 100.0) if rsi == rsi else 0.0,
            np.clip(macd_hist / close * 100.0, -10, 10) / 10.0 if n >= 2 else 0.0,
            np.clip(vol_ratio, 0, 5) / 5.0 if vol_ratio == vol_ratio else 0.0,
            ((c_arr[-1] / ma10 - 1.0) * 100.0 / 10.0) if ma10 == ma10 else 0.0,
            ((c_arr[-1] / ma30 - 1.0) * 100.0 / 10.0) if ma30 == ma30 else 0.0,
            np.clip((c_arr[-1] - (bb_mid - 2 * bb_sd)) / (4 * bb_sd + 1e-12), 0, 1)
            if (bb_mid == bb_mid and bb_sd == bb_sd) else 0.0,
            np.sign(pct(60)) if n > 60 and pct(60) == pct(60) else 0.0,
            risk,
        ], dtype=float)
        return feat
