"""横截面因子：同一时刻跨品种的相对强弱排名（动量/波动/量能/反转/振幅）。

与单品种时序因子互补：时序因子回答"该品种现在值不值得买"，横截面因子回答
"一堆品种里谁相对更强、谁相对更弱"——多品种轮动/多空对冲的信号基础。

经典横截面研究流程（本模块实现）：
1. 每个品种独立计算因子时序（纯历史窗口，无未来数据）
2. 按公共时间戳对齐 → 每期横截面排名
3. 截面 IC：每期 因子值排名 vs 未来 h 期收益排名 的 Spearman 相关，
   得到 IC 时间序列 → IC 均值 / ICIR / 胜率（与时序因子 IC 同构可比）
4. 多空价差：每期做多 Top 1/3、做空 Bottom 1/3（品种少时各取 1 个），
   持有 h 期复利累计 → 多空组合收益曲线与 Sharpe
5. 最新快照：各品种因子值/排名/近期收益（前端轮动表直接渲染）

统计意义提示：横截面 IC 每期只有 n_symbols 个点，品种数 <5 时 IC 序列噪声
很大，报告会带 warning 字段提示。
"""
import math

import numpy as np
import pandas as pd

# 横截面因子定义：key -> (名称, 说明, 每品种计算函数)
# 计算函数签名 fn(df, lookback) -> pd.Series（index 与 df 一致，允许 NaN）


def _xs_mom(df: pd.DataFrame, lb: int) -> pd.Series:
    return df["close"] / df["close"].shift(lb) - 1.0


def _xs_reversal(df: pd.DataFrame, lb: int) -> pd.Series:
    r = max(1, min(5, lb // 6))
    return -(df["close"] / df["close"].shift(r) - 1.0)


def _xs_vol(df: pd.DataFrame, lb: int) -> pd.Series:
    ret = np.log(df["close"]).diff()
    return ret.rolling(lb).std()


def _xs_volume(df: pd.DataFrame, lb: int) -> pd.Series:
    ma = df["volume"].rolling(lb).mean()
    return df["volume"] / (ma + 1e-12) - 1.0


def _xs_range(df: pd.DataFrame, lb: int) -> pd.Series:
    hi = df["high"].rolling(lb).max()
    lo = df["low"].rolling(lb).min()
    return (hi - lo) / df["close"]


XS_FACTORS: dict[str, dict] = {
    "xs_mom": {"name": "横截面动量", "desc": "近N期收益率（高=强势，动量延续）",
               "fn": _xs_mom},
    "xs_reversal": {"name": "横截面反转", "desc": "短期反转（高=近期跌幅大，博弈反弹）",
                    "fn": _xs_reversal},
    "xs_vol": {"name": "横截面波动", "desc": "近N期已实现波动率（低波动异象通常 IC 为负）",
               "fn": _xs_vol},
    "xs_volume": {"name": "横截面量能", "desc": "量能比（当期量 / 近N期均量 - 1）",
                  "fn": _xs_volume},
    "xs_range": {"name": "横截面振幅", "desc": "近N期高低价振幅占比（高=博弈激烈）",
                 "fn": _xs_range},
}

_MIN_SYMBOLS = 3
_MAX_CURVE_POINTS = 400


def align_frames(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """按公共时间戳（交集）对齐多品种 DataFrame，剔除无有效数据的品种。"""
    out = {}
    for sym, df in dfs.items():
        if df is None or len(df) < 10 or "close" not in df.columns:
            continue
        out[sym] = df
    if len(out) < 2:
        return out
    idx = None
    for df in out.values():
        i = df.index
        idx = i if idx is None else idx.intersection(i)
    return {sym: df.loc[idx] for sym, df in out.items()}


def xs_values(dfs: dict[str, pd.DataFrame], key: str, lookback: int) -> pd.DataFrame:
    """每品种计算因子时序并对齐 → 因子值矩阵（列=品种，行=公共时间戳）。"""
    if key not in XS_FACTORS:
        raise ValueError(f"未知横截面因子 {key}")
    fn = XS_FACTORS[key]["fn"]
    cols = {}
    for sym, df in dfs.items():
        s = fn(df, lookback)
        if isinstance(s, pd.Series) and s.notna().sum() > 5:
            cols[sym] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def _rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """逐行排名（升序：值最小=1，NaN 当期剔除）。"""
    return frame.rank(axis=1, ascending=True)


def _spearman_rows(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """逐行 Spearman 秩相关（a/b 列对齐），返回每期相关系数序列。"""
    ra = _rank_rows(a)
    rb = _rank_rows(b)
    out = {}
    for ts in a.index:
        x = ra.loc[ts]
        y = rb.loc[ts]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        xv = xv - xv.mean()
        yv = yv - yv.mean()
        denom = math.sqrt(float((xv ** 2).sum()) * float((yv ** 2).sum()))
        if denom < 1e-12:
            continue
        out[ts] = float((xv * yv).sum() / denom)
    return pd.Series(out)


def forward_return_matrix(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """未来 h 期收益矩阵（列=品种）。"""
    return closes.shift(-horizon) / closes - 1.0


def xs_ic_series(values: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.Series:
    """横截面 IC 序列：每期 因子值 vs 未来h期收益 的 Spearman 相关。"""
    fwd = forward_return_matrix(closes, horizon)
    common = values.index.intersection(fwd.index)
    return _spearman_rows(values.loc[common], fwd.loc[common])


def long_short_spread(values: pd.DataFrame, closes: pd.DataFrame, horizon: int,
                      quantile: float = 0.34) -> dict:
    """多空价差：每 horizon 期再平衡，做多因子值最高组、做空最低组。

    组大小 = max(1, floor(n_symbols * quantile))；品种不足 2 个返回空。
    spread 序列为每个再平衡期的 多头均值收益 - 空头均值收益（非重叠）。
    """
    if len(values.columns) < 2:
        return {"ts": [], "spread": [], "total_ret": None, "sharpe": None,
                "n_periods": 0, "group_size": 0}
    fwd = forward_return_matrix(closes, horizon)
    group_n = max(1, int(len(values.columns) * quantile))
    # 非重叠采样：从第一个因子可用的行开始，每 horizon 行取一次
    valid_rows = values.dropna(how="all").index
    if len(valid_rows) == 0:
        return {"ts": [], "spread": [], "total_ret": None, "sharpe": None,
                "n_periods": 0, "group_size": group_n}
    fwd = fwd.reindex(values.index)
    ts_list, spreads = [], []
    i = values.index.get_loc(valid_rows[0])
    step = max(1, horizon)
    while i < len(values):
        row = values.iloc[i]
        mask = row.notna()
        if mask.sum() >= 2:
            vals = row[mask]
            frow = fwd.iloc[i][mask]
            ok = vals.notna() & frow.notna()
            if ok.sum() >= 2:
                vals, frow = vals[ok], frow[ok]
                order = vals.sort_values(ascending=False)
                n = max(1, int(len(order) * quantile))
                long_syms = order.index[:n]
                short_syms = order.index[-n:]
                long_ret = float(frow[long_syms].mean())
                short_ret = float(frow[short_syms].mean())
                ts_list.append(values.index[i])
                spreads.append(long_ret - short_ret)
        i += step
    if not spreads:
        return {"ts": [], "spread": [], "total_ret": None, "sharpe": None,
                "n_periods": 0, "group_size": group_n}
    s = pd.Series(spreads, index=pd.DatetimeIndex(ts_list))
    total = float((1.0 + s).prod() - 1.0)
    sharpe = None
    if len(s) > 2 and s.std(ddof=1) > 1e-12:
        sharpe = float(s.mean() / s.std(ddof=1))  # 每再平衡期一档，未年化
    return {"ts": [str(t) for t in s.index], "spread": [round(float(v), 6) for v in s.values],
            "total_ret": round(total, 6), "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "n_periods": int(len(s)), "group_size": group_n}


def _downsample(ts: list, vals: list, cap: int = _MAX_CURVE_POINTS) -> tuple[list, list]:
    """曲线点数封顶（等距抽样，首尾保留）。"""
    n = len(ts)
    if n <= cap:
        return ts, vals
    idx = np.linspace(0, n - 1, cap).astype(int)
    keep = sorted(set(int(i) for i in idx))
    return [ts[i] for i in keep], [vals[i] for i in keep]


def cross_section_report(dfs: dict[str, pd.DataFrame], lookback: int = 24,
                         horizon: int = 1) -> dict:
    """横截面因子全景报告：每因子 IC 统计 + 多空价差 + 最新快照。

    dfs: {symbol: OHLCV DataFrame}（时间戳索引）
    返回可直接 JSON 化的 dict；品种 <3 抛 ValueError。
    """
    aligned = align_frames(dfs)
    symbols = sorted(aligned.keys())
    if len(symbols) < _MIN_SYMBOLS:
        raise ValueError(
            f"横截面分析至少需要 {_MIN_SYMBOLS} 个品种（对齐后 {len(symbols)} 个）。"
            "请在数据管理页下载更多品种，或用演示模式。")
    closes = pd.DataFrame({sym: aligned[sym]["close"] for sym in symbols})
    n_candles = len(closes)
    if n_candles < lookback + horizon + 10:
        raise ValueError(f"对齐后仅 {n_candles} 根K线，不足以计算 lookback={lookback} 的横截面因子")

    factors_out, curves = [], {}
    for key, defn in XS_FACTORS.items():
        vals = xs_values(aligned, key, lookback)
        if vals.empty:
            continue
        ic_series = xs_ic_series(vals, closes, horizon)
        ics = ic_series.dropna()
        ic_mean = float(ics.mean()) if len(ics) else 0.0
        ic_std = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
        ic_ir = float(ic_mean / (ic_std + 1e-12)) if len(ics) > 5 else 0.0
        ls = long_short_spread(vals, closes, horizon)

        # 最新快照：因子值/排名/近 horizon 期已实现收益
        last_row = vals.iloc[-1].dropna()
        ranks = last_row.rank(ascending=False)  # 1 = 因子值最高
        recent = closes.iloc[-1] / closes.iloc[-1 - horizon] - 1.0
        latest = []
        for sym, v in last_row.items():
            latest.append({
                "symbol": sym,
                "value": round(float(v), 6),
                "rank": int(ranks[sym]),
                "ret_recent": round(float(recent.get(sym, float("nan"))), 6)
                if pd.notna(recent.get(sym)) else None,
            })
        latest.sort(key=lambda x: x["rank"])

        # 多空累计曲线（复利）
        if ls["spread"]:
            cum = list(np.cumprod([1.0 + x for x in ls["spread"]]) - 1.0)
            ts, vals_c = _downsample(ls["ts"], [round(float(c), 6) for c in cum])
            curves[key] = {"ts": ts, "cum": vals_c}

        factors_out.append({
            "key": key, "name": defn["name"], "desc": defn["desc"],
            "ic_mean": round(ic_mean, 4), "ic_ir": round(ic_ir, 4),
            "ic_win_rate": round(float((ics > 0).mean()), 4) if len(ics) else None,
            "n_ic_obs": int(len(ics)),
            "long_short": {"total_ret": ls["total_ret"], "sharpe": ls["sharpe"],
                           "n_periods": ls["n_periods"], "group_size": ls["group_size"]},
            "latest": latest,
        })
    factors_out.sort(key=lambda x: -abs(x["ic_mean"]))

    warning = None
    if len(symbols) < 5:
        warning = (f"仅 {len(symbols)} 个品种：横截面每期只有 {len(symbols)} 个样本点，"
                   "IC 与多空统计噪声大，建议扩充到 5 个以上品种（数据管理页可批量下载）")
    return {
        "ok": True, "symbols": symbols, "n_symbols": len(symbols),
        "n_candles": n_candles, "lookback": lookback, "horizon": horizon,
        "factors": factors_out, "spread_curves": curves, "warning": warning,
    }
