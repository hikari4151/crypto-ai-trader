"""流式组合因子引擎（2026-09-23，P1-② 因子侧专项）。

目标：与 factors.mining.CompositeFactorEvaluator.eval（每根对 400 根全量
compute_factor_matrix + expanding z）**逐位一致**，但增量计算（O(1~w) 因子值
+ O(窗口) numpy 前缀统计），替代 eval 的 pandas 全量重算（实测 4.4ms/根）。

对齐语义（pandas 口径）：
- 因子值：与 factors/library.py 各 _f 公式一致（rolling/ewm）
- expanding z：值序列（≤400 缓冲）前缀 mean/std（跳过 NaN，ddof=1），
  当前行 z=(v-cummean)/(cumstd+1e-12)，NaN→0（fillna0）
- cols 筛选：缓冲内非 NaN 计数 > 5 的因子参与合成；权重按参与列绝对值归一
- 二次标准化：模型内 mu/sd
"""
from typing import Optional

import numpy as np


def _ema_next(old: Optional[float], x: float, alpha: float) -> float:
    """pandas ewm(alpha, adjust=False).mean()（首个非 None 为初值）。"""
    if old is None:
        return x
    return (1.0 - alpha) * old + alpha * x


def _rolling_sum(vals, k: int) -> Optional[float]:
    if len(vals) < k:
        return None
    return sum(vals[-k:])


def _rolling_mean(vals, k: int) -> Optional[float]:
    s = _rolling_sum(vals, k)
    return None if s is None else s / k


class StreamingCompositeFactor:
    """增量复算 factor_composite 组合因子（部署/回放端）。"""

    def __init__(self, weights: dict, mu: Optional[float] = None,
                 sd: Optional[float] = None, maxbuf: int = 400) -> None:
        self.weights = {str(k): float(w) for k, w in (weights or {}).items() if w}
        self.mu = float(mu) if mu is not None else None
        self.sd = float(sd) if sd else None
        self._maxbuf = maxbuf
        self._close: list = []
        self._high: list = []
        self._low: list = []
        self._vol: list = []
        self._vals: dict = {}  # key -> 值序列 list（≤maxbuf）
        # MACD ewm 状态
        self._ema_f = None
        self._ema_s = None
        self._dea = None
        # ADX Wilder 状态（span=14, adjust=False）
        self._atr = None
        self._pdm = None
        self._mdm = None
        self._adx = None

    # ---------- 因子值（增量，O(1~w)）----------
    def _factor_values(self, close: float, high: float, low: float,
                       volume: float) -> dict:
        c, h, l, v = self._close, self._high, self._low, self._vol
        n = len(c)
        out: dict = {}
        # mom_10 / mom_5 / mom_accel（pct_change(k)*100，窗口不足 None=NaN）
        k10 = len(c) >= 11
        if k10:
            mom10 = (c[-1] / c[-11] - 1.0) * 100.0
            out["mom_10"] = mom10
            # mom_accel = roc10 - roc5：roc10 就绪（len>=11）后才有值（pandas 前 10 根 NaN）
            mom5 = (c[-1] / c[-6] - 1.0) * 100.0
            out["mom_accel"] = mom10 - mom5
        # bias_20（rolling20 mean）
        ma20 = _rolling_mean(c, 20)
        if ma20 is not None:
            out["bias_20"] = (c[-1] - ma20) / ma20 * 100.0
        # vwap_dist（rolling20 sum(c*v)/sum(v)）
        cv = [a * b for a, b in zip(c, v)]
        scv = _rolling_sum(cv, 20)
        sv = _rolling_sum(v, 20)
        if scv is not None and sv is not None:
            out["vwap_dist"] = ((c[-1] - scv / (sv + 1e-12)) / c[-1]) * 100.0
        # macd_hist_pct（EWM12/26/9，adjust=False）
        a_f, a_s = 2.0 / 13.0, 2.0 / 27.0
        self._ema_f = _ema_next(self._ema_f, close, a_f)
        self._ema_s = _ema_next(self._ema_s, close, a_s)
        dif = (self._ema_f or 0.0) - (self._ema_s or 0.0)
        self._dea = _ema_next(self._dea, dif, 2.0 / 10.0)
        out["macd_hist_pct"] = (dif - (self._dea or 0.0)) * 2.0 / close * 100.0
        # adx_14（Wilder：TR/+DM/-DM 的 EWM span14 + DX 的 EWM14）
        # pandas 口径：第 1 根 up/down=NaN→+DM/-DM=0、TR=h-l、ATR/DM/ADX ewm 初值
        # 从第 1 根起（atr=tr0、pdm=mdm=0、adx=0）；up=high.diff()、down=-low.diff()。
        a = 2.0 / 15.0
        if n == 1:
            tr0 = high - low
            self._atr = tr0
            self._pdm = 0.0
            self._mdm = 0.0
            self._adx = 0.0
        else:
            prev_c = c[-2]
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
            up = high - h[-2]
            down = -(low - l[-2])
            pdm_ = up if (up > down and up > 0) else 0.0
            mdm_ = down if (down > up and down > 0) else 0.0
            self._atr = _ema_next(self._atr, tr, a)
            self._pdm = _ema_next(self._pdm, pdm_, a)
            self._mdm = _ema_next(self._mdm, mdm_, a)
            pdi = 100.0 * (self._pdm or 0.0) / ((self._atr or 0.0) + 1e-12)
            mdi = 100.0 * (self._mdm or 0.0) / ((self._atr or 0.0) + 1e-12)
            dx = abs(pdi - mdi) / (pdi + mdi + 1e-12) * 100.0
            self._adx = _ema_next(self._adx, dx, a)
        out["adx_14"] = self._adx if self._adx is not None else float("nan")
        return out

    # ---------- expanding z 合成 ----------
    def _composite(self) -> Optional[float]:
        if not self.weights:
            return None
        cols = []
        for k in self.weights:
            seq = self._vals.get(k)
            if seq is None:
                continue
            cnt = sum(1 for x in seq if x == x)
            if cnt > 5:
                cols.append(k)
        if not cols:
            return None
        total = sum(abs(self.weights[c]) for c in cols)
        if total <= 0:
            return None
        out = 0.0
        for c in cols:
            arr = np.asarray(self._vals[c], dtype=float)
            m = ~np.isnan(arr)
            cc = np.cumsum(m.astype(float))
            cs = np.cumsum(np.where(m, arr, 0.0))
            cs2 = np.cumsum(np.where(m, arr * arr, 0.0))
            cummean = cs / np.maximum(cc, 1e-12)
            var = (cs2 - cs * cs / np.maximum(cc, 1e-12)) / np.maximum(cc - 1.0, 1e-12)
            z_last = (arr[-1] - cummean[-1]) / (np.sqrt(var[-1]) + 1e-12)
            if not (z_last == z_last):  # NaN
                z_last = 0.0
            out += float(z_last) * (self.weights[c] / total)
        if self.mu is not None and self.sd:
            out = (out - self.mu) / self.sd
        return float(out) if out == out else None

    # ---------- 主入口 ----------
    def update(self, close: float, high: float, low: float,
               volume: float) -> Optional[float]:
        self._close.append(float(close))
        self._high.append(float(high))
        self._low.append(float(low))
        self._vol.append(float(volume))
        if len(self._close) > self._maxbuf:
            self._close = self._close[-self._maxbuf:]
            self._high = self._high[-self._maxbuf:]
            self._low = self._low[-self._maxbuf:]
            self._vol = self._vol[-self._maxbuf:]
        vals = self._factor_values(close, high, low, volume)
        for k in self.weights:
            self._vals.setdefault(k, []).append(vals.get(k, float("nan")))
            if len(self._vals[k]) > self._maxbuf:
                self._vals[k] = self._vals[k][-self._maxbuf:]
        return self._composite()