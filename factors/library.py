"""内置因子库：动量/波动率/量价/趋势/均值回归/宏观代理因子。

全部为纯向量化实现（pandas），输入标准 OHLCV DataFrame。
窗口期内的前部返回 NaN，分析时自动剔除。
"""
import numpy as np
import pandas as pd

from .base import (CATEGORY_MACRO, CATEGORY_MEAN_REV, CATEGORY_MOMENTUM,
                   CATEGORY_TREND, CATEGORY_VOLATILITY, CATEGORY_VOLUME, Factor)

_LIB: list[Factor] = []


def _f(*args, **kwargs) -> Factor:
    _LIB.append(Factor(*args, **kwargs))
    return _LIB[-1]


def _roc(v: pd.Series, n: int) -> pd.Series:
    """n 期收益率（%）。"""
    return v.pct_change(n) * 100.0


# ============ 动量 ============
_f("mom_5", "5期动量", CATEGORY_MOMENTUM, "过去5根K线收益率（%），短期动量。",
   lambda df: _roc(df["close"], 5), {"n": 5})
_f("mom_10", "10期动量", CATEGORY_MOMENTUM, "过去10根K线收益率（%），中期动量。",
   lambda df: _roc(df["close"], 10), {"n": 10})
_f("mom_20", "20期动量", CATEGORY_MOMENTUM, "过去20根K线收益率（%），较长周期动量。",
   lambda df: _roc(df["close"], 20), {"n": 20})
_f("roc_1", "1期收益率", CATEGORY_MOMENTUM, "相邻K线收益率（%），短线价格变动速度。",
   lambda df: _roc(df["close"], 1), {"n": 1})
_f("mom_accel", "动量加速度", CATEGORY_MOMENTUM, "10期动量与5期动量之差，动量加速/减速信号。",
   lambda df: _roc(df["close"], 10) - _roc(df["close"], 5))


# ============ 波动率 ============
def _realized_vol(df, n: int) -> pd.Series:
    """n 期已实现波动率（收益std，%）。"""
    r = df["close"].pct_change() * 100.0
    return r.rolling(n).std()


def _atr(df, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


_f("vol_10", "10期波动率", CATEGORY_VOLATILITY, "10期收益标准差（%），短期波动水平。",
   lambda df: _realized_vol(df, 10), {"n": 10})
_f("vol_20", "20期波动率", CATEGORY_VOLATILITY, "20期收益标准差（%），中期波动水平。",
   lambda df: _realized_vol(df, 20), {"n": 20})
_f("atr_pct", "ATR占比", CATEGORY_VOLATILITY, "ATR(14)占收盘价比（%），真实波幅强度。",
   lambda df: _atr(df, 14) / df["close"] * 100.0, {"n": 14})
_f("bb_width", "布林带宽", CATEGORY_VOLATILITY, "20期布林带上轨与下轨间距相对中轨（%），带宽扩张=波动加大。",
   lambda df: (df["close"].rolling(20).std() * 4) / df["close"].rolling(20).mean() * 100.0, {"n": 20})


# ============ 量价 ============
def _vol_ratio(df, n: int = 5) -> pd.Series:
    """n 期量比：当前量 / n 期均量。"""
    return df["volume"] / (df["volume"].rolling(n).mean() + 1e-12)


def _obv(df) -> pd.Series:
    """OBV 能量潮，用其滚动斜率反映量能趋势。"""
    c = df["close"]
    sign = np.sign(c.diff()).fillna(0.0)
    obv = (sign * df["volume"]).cumsum()
    return obv


def _mfi(df, n: int = 14) -> pd.Series:
    """MFI 资金流量指标：量加权 RSI。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    mr = pos.rolling(n).sum() / (neg.rolling(n).sum() + 1e-12)
    return 100.0 - 100.0 / (1.0 + mr)


_f("vol_ratio", "量比", CATEGORY_VOLUME, "当前成交量与5期均量之比，放量/缩量。",
   lambda df: _vol_ratio(df, 5), {"n": 5})
_f("obv_slope", "OBV斜率", CATEGORY_VOLUME, "OBV 5期变化量，量能趋势方向。",
   lambda df: _obv(df).diff(5), {"n": 5})
_f("mfi_14", "资金流量MFI", CATEGORY_VOLUME, "14期资金流量指标（0-100），量价配合度。",
   lambda df: _mfi(df, 14), {"n": 14})
_f("vwap_dist", "VWAP偏离", CATEGORY_VOLUME, "收盘价相对成交均价(VWAP)偏离（%），正=价格高于平均成交价。",
   lambda df: (df["close"] - (df["close"] * df["volume"]).rolling(20).sum() /
               (df["volume"].rolling(20).sum() + 1e-12)) / df["close"] * 100.0, {"n": 20})


# ============ 趋势 ============
def _ma_dist(df, n: int) -> pd.Series:
    return (df["close"] / df["close"].rolling(n).mean() - 1.0) * 100.0


def _macd_hist(df, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ef = df["close"].ewm(span=fast, adjust=False).mean()
    es = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=signal, adjust=False).mean()
    return (dif - dea) * 2.0 / df["close"] * 100.0


def _adx(df, n: int = 14) -> pd.Series:
    """ADX 平均趋向指数：趋势强度（0-100），越高趋势越强。"""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=n, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100.0 * minus_dm.ewm(span=n, adjust=False).mean() / (atr + 1e-12)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) * 100.0
    return dx.ewm(span=n, adjust=False).mean()


_f("ma_dist_10", "均线偏离10", CATEGORY_TREND, "收盘价偏离MA10（%），短期趋势方向。",
   lambda df: _ma_dist(df, 10), {"n": 10})
_f("ma_dist_30", "均线偏离30", CATEGORY_TREND, "收盘价偏离MA30（%），中期趋势方向。",
   lambda df: _ma_dist(df, 30), {"n": 30})
_f("macd_hist_pct", "MACD柱占比", CATEGORY_TREND, "MACD柱相对价格（%），动量强弱与拐点。",
   lambda df: _macd_hist(df), {"fast": 12, "slow": 26, "signal": 9})
_f("adx_14", "ADX趋势强度", CATEGORY_TREND, "14期ADX（0-100），趋势确立/震荡判别。",
   lambda df: _adx(df, 14), {"n": 14})


# ============ 均值回归 ============
def _rsi(df, n: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _bias(df, n: int) -> pd.Series:
    """乖离率：价格相对均线偏离，超买超卖反转信号。"""
    return (df["close"] - df["close"].rolling(n).mean()) / df["close"].rolling(n).mean() * 100.0


def _cci(df, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


_f("rsi_14", "RSI(14)", CATEGORY_MEAN_REV, "相对强弱指标（0-100），超买超卖反转。",
   lambda df: _rsi(df, 14), {"n": 14})
_f("bias_20", "20期乖离率", CATEGORY_MEAN_REV, "价格偏离20日均线幅度（%），极端乖离回归。",
   lambda df: _bias(df, 20), {"n": 20})
_f("cci_20", "CCI(20)", CATEGORY_MEAN_REV, "顺势/摆动指标，超买超卖区间反转。",
   lambda df: _cci(df, 20), {"n": 20})
_f("bb_pos", "布林位置", CATEGORY_MEAN_REV, "价格在布林带内的位置（0-1），贴近上下轨的反转信号。",
   lambda df: (df["close"] - (df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std())) /
              (4 * df["close"].rolling(20).std() + 1e-12), {"n": 20})


# ============ 宏观/市场状态（可计算代理） ============
def _trend_regime(df, n: int = 60) -> pd.Series:
    """趋势 regime：长周期动量方向（+趋势/-趋势/0震荡）。"""
    return np.sign(_roc(df["close"], n))


def _risk_regime(df, n: int = 20) -> pd.Series:
    """风险 regime：波动率分位（0-1），1=高波动环境。"""
    v = _realized_vol(df, n)
    return v.rolling(n * 2, min_periods=n).rank(pct=True)


_f("trend_regime", "趋势环境", CATEGORY_MACRO, "60期动量方向（+1趋势上行/-1下行/0震荡）。",
   lambda df: _trend_regime(df, 60), {"n": 60})
_f("risk_regime", "风险环境", CATEGORY_MACRO, "20期波动率在40期中的分位（0-1），高位=风险环境。",
   lambda df: _risk_regime(df, 20), {"n": 20})


# ============ 注册表 ============
_BY_KEY = {f.key: f for f in _LIB}


def factor_keys() -> list[str]:
    return list(_BY_KEY)


def get_factor(key: str) -> Factor | None:
    return _BY_KEY.get(key)


def list_factors() -> list[Factor]:
    return list(_LIB)


def factors_by_category(category: str) -> list[Factor]:
    return [f for f in _LIB if f.category == category]
