"""向量化指标预计算：一次性算出全量指标数组，供快速回测循环按索引取值。

相比逐K线调用 compute_latest（每次 O(n) 重建），本模块把所有指标序列一次算好（O(n)），
配合 numpy 向量化运算，可大幅提升回测速度；支持可选 GPU（cuPy）加速指标批量计算。

此外提供 IncrIndicators 增量计算类，用于实时行情流：每根新 K 线 O(1) 增量更新指标，
避免全量重算的 O(n²) 开销。

性能量级：
- 原引擎：每根K线重建 ohlcv + 重算全部指标 → O(n²)
- 本引擎（预计算）：指标数组一次预计算 → O(n)，循环内 O(1) 取索引
- 实时引擎（增量）：每根K线 O(1) 增量更新
"""
from typing import Any, Optional

import numpy as np


def _get_xp(backend: str):
    """返回数组库：numpy 或 cupy（不可用回退 numpy）。"""
    if backend == "cupy":
        try:
            import cupy  # type: ignore
            return cupy
        except ImportError:
            pass
    return np


def _arr_module(arr):
    """根据数组类型返回其所属模块（numpy 或 cupy），避免用 numpy 函数操作 cupy 数组。"""
    if hasattr(arr, "get"):  # cupy.ndarray 特有
        import cupy
        return cupy
    return np


def _ema_vec(v, period: int):
    """EMA 向量化实现。v 可以是 numpy 或 cupy 数组，返回同类型。"""
    xp = _arr_module(v)
    k = 2.0 / (period + 1)
    if len(v) == 0:
        return v
    out = xp.empty_like(v)
    out[0] = v[0]
    km1 = 1.0 - k
    for i in range(1, len(v)):
        out[i] = v[i] * k + out[i - 1] * km1
    return out


def ema_incremental(old_val: float, new_val: float, period: int) -> float:
    """增量 EMA：O(1) 更新 EMA 值（用于实时行情流）。"""
    k = 2.0 / (period + 1)
    return new_val * k + old_val * (1.0 - k)


def _sma_vec(v, period: int):
    xp = _arr_module(v)
    out = xp.full(len(v), xp.nan)
    if len(v) < period:
        return out
    # 用 concatenate 前插 0（cupy 没有 insert）
    prefix = xp.zeros(1, dtype=v.dtype)
    c = xp.cumsum(xp.concatenate((prefix, v)))
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def _rsi_vec(v, period: int = 14):
    xp = _arr_module(v)
    out = xp.full(len(v), xp.nan)
    if len(v) < period + 1:
        return out
    diff = xp.diff(v)
    gains = xp.clip(diff, 0, None)
    losses = xp.clip(-diff, 0, None)
    avg_g = xp.full(len(v), xp.nan)
    avg_l = xp.full(len(v), xp.nan)
    avg_g[period] = gains[:period].mean()
    avg_l[period] = losses[:period].mean()
    for i in range(period + 1, len(v)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_g / (avg_l + 1e-12)
    out[period:] = 100.0 - 100.0 / (1 + rs[period:])
    # 价格完全走平（无涨无跌）：avg_g=avg_l=0 时 rs=0 → RSI=0 误判极超卖；
    # 对齐 technical.py / IncrIndicators 口径（无下跌 → RSI=100）
    flat = (avg_l < 1e-12) & (avg_g < 1e-12)
    out[flat] = 100.0
    return out


def _bollinger_vec(v, period: int = 20, num_std: float = 2.0):
    xp = _arr_module(v)
    mid = _sma_vec(v, period)
    std = xp.full(len(v), xp.nan)
    if len(v) >= period:
        prefix = xp.zeros(1, dtype=v.dtype)
        c1 = xp.cumsum(xp.concatenate((prefix, v)))
        c2 = xp.cumsum(xp.concatenate((prefix, v * v)))
        n = period
        sums = c1[period:] - c1[:-period]
        sq = c2[period:] - c2[:-period]
        std[period - 1:] = xp.sqrt(xp.maximum(sq / n - (sums / n) ** 2, 0))
    return mid + num_std * std, mid, mid - num_std * std


def precompute_indicator_series(closes, opens, highs, lows, volumes, backend: str = "numpy") -> dict[str, Any]:
    """一次性预计算所有指标序列，返回各指标的完整数组（与输入等长）。

    backend: "numpy" | "cupy"（GPU，需已安装 cupy）
    """
    xp = _get_xp(backend)
    closes = xp.asarray(closes, dtype=xp.float64)
    opens = xp.asarray(opens, dtype=xp.float64)
    highs = xp.asarray(highs, dtype=xp.float64)
    lows = xp.asarray(lows, dtype=xp.float64)
    volumes = xp.asarray(volumes, dtype=xp.float64)

    ma_fast = _sma_vec(closes, 10)
    ma_slow = _sma_vec(closes, 30)
    macd_dif = _ema_vec(closes, 12) - _ema_vec(closes, 26)
    macd_dea = _ema_vec(xp.nan_to_num(macd_dif, nan=0.0), 9)
    macd_hist = 2 * (macd_dif - macd_dea)
    rsi = _rsi_vec(closes, 14)
    bb_up, bb_mid, bb_low = _bollinger_vec(closes, 20, 2.0)
    vol_ma5 = _sma_vec(volumes, 5)
    # 暖机期 vol_ma5 为 NaN：量比置 1.0（无量比信息视为正常），
    # 曾用 nan_to_num(0)+1e-12 导致前 4 根量比爆炸到 ~1e14，污染 vol_break 类信号
    vol_ratio_series = volumes / vol_ma5
    vol_ratio_series = xp.where(xp.isfinite(vol_ratio_series), vol_ratio_series, 1.0)

    return {
        "close": closes, "open": opens, "high": highs, "low": lows, "volume": volumes,
        "ma_fast": ma_fast, "ma_slow": ma_slow,
        "macd": macd_dif, "macd_signal": macd_dea, "macd_hist": macd_hist,
        "rsi": rsi, "bb_upper": bb_up, "bb_mid": bb_mid, "bb_lower": bb_low,
        "vol_ma5": vol_ma5, "vol_ratio": vol_ratio_series,
    }


def snapshot_at(series: dict[str, Any], i: int) -> dict[str, Any]:
    """从预计算序列中取第 i 根K线的指标快照（替代 compute_latest 的逐K线重算）。"""
    def _val(arr):
        # numpy / cupy 数组都支持 item() 或 [i] 取标量
        v = arr[i]
        return float(v) if hasattr(v, "__float__") else float(v)
    def _n(v):
        try:
            f = float(v)
            return 0.0 if f != f else f  # NaN -> 0
        except (TypeError, ValueError):
            return 0.0
    return {
        "close": _n(_val(series["close"])),
        "open": _n(_val(series["open"])),
        "high": _n(_val(series["high"])),
        "low": _n(_val(series["low"])),
        "ma_fast": _n(_val(series["ma_fast"])),
        "ma_slow": _n(_val(series["ma_slow"])),
        "macd": _n(_val(series["macd"])),
        "macd_signal": _n(_val(series["macd_signal"])),
        "macd_hist": _n(_val(series["macd_hist"])),
        "rsi": _n(_val(series["rsi"])),
        "bb_upper": _n(_val(series["bb_upper"])),
        "bb_mid": _n(_val(series["bb_mid"])),
        "bb_lower": _n(_val(series["bb_lower"])),
        "volume": _n(_val(series["volume"])),
        "vol_ratio": _n(_val(series["vol_ratio"])),
        "candles_count": i + 1,
    }


_GPU_STATE: Optional[bool] = None


def gpu_available() -> bool:
    """是否真正可用 GPU 后端（import 成功 + 实际能跑一次运算）。

    仅 import cupy 成功不代表可用——缺 CUDA toolkit 头文件时
    会在运行时抛错。这里做一次真实小运算验证，结果缓存。
    """
    global _GPU_STATE
    if _GPU_STATE is not None:
        return _GPU_STATE
    try:
        import cupy  # type: ignore
        a = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        _ = cupy.asnumpy(cupy.cumsum(a))  # 触发编译，缺头文件会抛错
        _GPU_STATE = True
        return True
    except Exception:  # noqa: BLE001
        _GPU_STATE = False
        return False


class IncrIndicators:
    """增量指标计算器：维护 EMA/SMA/RSI/布林带/成交量均值 的当前状态，
    每根新 K 线 O(1) 增量更新，避免全量重算。

    用于实时行情流（ws_market.py），相比 compute_latest() 的 O(n) 全量重算，
    增量更新将每根K线的指标计算降至 O(1)。
    """

    def __init__(
        self,
        period_ma_fast: int = 10,
        period_ma_slow: int = 30,
        ema_fast: int = 12,
        ema_slow: int = 26,
        ema_signal: int = 9,
        period_rsi: int = 14,
        period_bb: int = 20,
        num_std_bb: float = 2.0,
        period_vol_ma: int = 5,
    ) -> None:
        self.period_ma_fast = period_ma_fast
        self.period_ma_slow = period_ma_slow
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_signal = ema_signal
        self.period_rsi = period_rsi
        self.period_bb = period_bb
        self.num_std_bb = num_std_bb
        self.period_vol_ma = period_vol_ma

        # EMA 状态（MACD）
        self._ema_fast_val: Optional[float] = None
        self._ema_slow_val: Optional[float] = None
        self._ema_signal_val: Optional[float] = None
        self._macd_hist: Optional[float] = None

        # SMA 状态（滑动窗口缓冲）
        self._buf_fast: list[float] = []
        self._buf_slow: list[float] = []
        self._ma_fast: float = 0.0
        self._ma_slow: float = 0.0

        # RSI 状态
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi_count: int = 0
        self._rsi: float = 50.0
        self._last_close: Optional[float] = None

        # 布林带 状态（增量 mean + std）
        self._bb_buf: list[float] = []
        self._bb_sum: float = 0.0
        self._bb_sum_sq: float = 0.0
        self._bb_upper: float = 0.0
        self._bb_mid: float = 0.0
        self._bb_lower: float = 0.0

        # 成交量均线
        self._vol_buf: list[float] = []
        self._vol_ma: float = 0.0

    def update(self, close: float, volume: float) -> dict[str, Any]:
        """增量更新指标，返回最新快照。每根新K线 O(1) 复杂度。"""
        # ---- EMA 更新（MACD）----
        if self._ema_fast_val is None:
            self._ema_fast_val = close
            self._ema_slow_val = close
            # DEA 播种与回测/向量化一致（dif[0]=0）：曾用 close 播种，
            # 启动后 30-50 根 K 线内 MACD 信号与回测系统性偏离
            self._ema_signal_val = 0.0
            self._macd_hist = 0.0
        else:
            self._ema_fast_val = ema_incremental(self._ema_fast_val, close, self.ema_fast)
            self._ema_slow_val = ema_incremental(self._ema_slow_val, close, self.ema_slow)
            dif = self._ema_fast_val - self._ema_slow_val
            self._ema_signal_val = ema_incremental(self._ema_signal_val, dif, self.ema_signal)
            self._macd_hist = 2.0 * (dif - self._ema_signal_val)

        # ---- SMA 更新（MA10 / MA30）----
        self._buf_fast.append(close)
        if len(self._buf_fast) > self.period_ma_fast:
            self._buf_fast.pop(0)
        self._ma_fast = sum(self._buf_fast) / len(self._buf_fast)

        self._buf_slow.append(close)
        if len(self._buf_slow) > self.period_ma_slow:
            self._buf_slow.pop(0)
        self._ma_slow = sum(self._buf_slow) / len(self._buf_slow)

        # ---- RSI 更新 ----
        if self._last_close is not None:
            diff = close - self._last_close
            if self._rsi_count < self.period_rsi:
                # 初始期：简单平均
                self._avg_gain = (self._avg_gain * self._rsi_count + max(diff, 0.0)) / (self._rsi_count + 1)
                self._avg_loss = (self._avg_loss * self._rsi_count + max(-diff, 0.0)) / (self._rsi_count + 1)
                self._rsi_count += 1
            else:
                # Wilder 平滑
                self._avg_gain = (self._avg_gain * (self.period_rsi - 1) + max(diff, 0.0)) / self.period_rsi
                self._avg_loss = (self._avg_loss * (self.period_rsi - 1) + max(-diff, 0.0)) / self.period_rsi
        self._last_close = close

        if self._rsi_count > 0 and self._avg_loss < 1e-12:
            self._rsi = 100.0
        elif self._rsi_count > 0:
            rs = self._avg_gain / self._avg_loss
            self._rsi = 100.0 - 100.0 / (1.0 + rs)
        else:
            self._rsi = 50.0

        # ---- 布林带 更新（增量 mean + std）----
        self._bb_buf.append(close)
        if len(self._bb_buf) > self.period_bb:
            old = self._bb_buf.pop(0)
            self._bb_sum -= old          # 均值累加器减旧值本身（曾误减 old² 导致均线漂移）
            self._bb_sum_sq -= old * old
        self._bb_sum += close
        self._bb_sum_sq += close * close
        n_bb = len(self._bb_buf)
        if n_bb >= self.period_bb:
            mean_bb = self._bb_sum / n_bb
            var_bb = max(self._bb_sum_sq / n_bb - mean_bb * mean_bb, 0.0)
            std_bb = var_bb ** 0.5
        else:
            std_bb = 0.0
        self._bb_mid = self._bb_sum / n_bb
        self._bb_upper = self._bb_mid + self.num_std_bb * std_bb
        self._bb_lower = self._bb_mid - self.num_std_bb * std_bb

        # ---- 成交量均线 ----
        self._vol_buf.append(volume)
        if len(self._vol_buf) > self.period_vol_ma:
            self._vol_buf.pop(0)
        self._vol_ma = sum(self._vol_buf) / len(self._vol_buf)

        return {
            "close": close,
            "ma_fast": self._ma_fast,
            "ma_slow": self._ma_slow,
            "macd": (self._ema_fast_val or 0.0) - (self._ema_slow_val or 0.0),
            "macd_signal": self._ema_signal_val or 0.0,
            "macd_hist": self._macd_hist or 0.0,
            "rsi": self._rsi,
            "bb_upper": self._bb_upper,
            "bb_mid": self._bb_mid,
            "bb_lower": self._bb_lower,
            "volume": volume,
            "vol_ratio": volume / (self._vol_ma + 1e-12),
            "count": n_bb,
        }

    def snapshot(self) -> dict[str, Any]:
        """只读快照：返回当前指标值，不改变任何状态（与 update 返回结构一致）。

        用于高频读取场景（如行情快照/AI 调度），避免同一根K线被反复 update
        污染 SMA/布林带等滚动缓冲。
        """
        return {
            "close": self._last_close or 0.0,
            "ma_fast": self._ma_fast,
            "ma_slow": self._ma_slow,
            "macd": (self._ema_fast_val or 0.0) - (self._ema_slow_val or 0.0),
            "macd_signal": self._ema_signal_val or 0.0,
            "macd_hist": self._macd_hist or 0.0,
            "rsi": self._rsi,
            "bb_upper": self._bb_upper,
            "bb_mid": self._bb_mid,
            "bb_lower": self._bb_lower,
            "volume": self._vol_buf[-1] if self._vol_buf else 0.0,
            "vol_ratio": (self._vol_buf[-1] / self._vol_ma) if (self._vol_buf and self._vol_ma > 0) else 1.0,
            "count": len(self._bb_buf),
        }

    def reset(self) -> None:
        """重置所有状态（用于切换交易对或重新开始）。"""
        self.__init__(
            self.period_ma_fast, self.period_ma_slow,
            self.ema_fast, self.ema_slow, self.ema_signal,
            self.period_rsi, self.period_bb, self.num_std_bb,
            self.period_vol_ma,
        )
