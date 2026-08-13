"""因子有效性分析：IC / RankIC / ICIR / 收益分组 / 相关性 / 因子暴露 / 安检门槛。

核心指标：
- IC（信息系数）：因子值 t 期 与 未来 h 期收益的 Pearson 相关
- RankIC：Spearman 秩相关（对非线性更稳健，量化研究最常用）
- ICIR：IC 均值 / IC 标准差，衡量因子稳定性
- 分层收益：按因子分位数分组，看 top 组 - bottom 组 单调性
- 因子暴露：对合成组合因子的标准化暴露度
- 换手率 / fitness：因子信号稳定性与质量综合分（借鉴 WorldQuant BRAIN）
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def forward_returns(close: pd.Series, h: int = 1) -> pd.Series:
    """未来 h 期收益率（%），最后一个 h 期前移，末尾无标签为 NaN。"""
    return close.shift(-h) / close - 1.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（有 NaN 则剔除后计算）。"""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    a, b = a[mask], b[mask]
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def factor_ic(factor_series: pd.Series, close: pd.Series, h: int = 1,
              method: str = "rank") -> dict:
    """单个因子的 IC 分析。

    返回: {ic, rank_ic, icir, ic_mean, ic_std, ic_win_rate, n}
    """
    f = factor_series.to_numpy(dtype=float)
    fwd = forward_returns(close, h).to_numpy(dtype=float)
    ic = _pearson(f, fwd)
    rank_ic = _spearman(f, fwd)

    # 滑动 IC 序列（滚动 30 期）算 ICIR
    ic_series = _rolling_ic(factor_series, close, h=h, method=method, window=30)
    ics = ic_series[ic_series.notna()]
    icir = float(ics.mean() / (ics.std() + 1e-12)) if len(ics) > 5 else 0.0
    ic_win = float((ics > 0).mean()) if len(ics) > 0 else 0.0

    return {
        "ic": round(ic, 6), "rank_ic": round(rank_ic, 6),
        "icir": round(icir, 6), "ic_mean": round(float(ics.mean()), 6) if len(ics) else 0.0,
        "ic_std": round(float(ics.std()), 6) if len(ics) else 0.0,
        "ic_win_rate": round(ic_win, 4), "n": int(len(ics)),
    }


def _rolling_ic(factor_series: pd.Series, close: pd.Series, h: int = 1,
                method: str = "rank", window: int = 30, step: int = None) -> pd.Series:
    """滚动窗口 IC 序列（对过去 window 期求 IC，每 step 期采样一次）。

    step 默认等于 window（非重叠窗）：重叠窗会引入 IC 序列的自相关，
    std 被低估 → ICIR 系统性高估，因子选择门槛失真。非重叠窗的 IC
    序列样本独立，ICIR 才是因子稳定性的可靠度量。
    """
    if step is None:
        step = window
    f = factor_series.astype(float)
    fwd = forward_returns(close, h)
    out = pd.Series(np.nan, index=factor_series.index)
    vals = f.to_numpy(dtype=float)
    fwds = fwd.to_numpy(dtype=float)
    idx = factor_series.index
    for i in range(window, len(vals), step):
        a = vals[i - window:i]
        b = fwds[i - window:i]
        if method == "rank":
            out.iloc[i] = _spearman(a, b)
        else:
            out.iloc[i] = _pearson(a, b)
    return out


def factor_ic_table(mat: pd.DataFrame, close: pd.Series, h: int = 1,
                    method: str = "rank") -> list[dict]:
    """批量因子 IC 分析表，按 |IC| 排序。"""
    rows = []
    for col in mat.columns:
        s = mat[col]
        try:
            r = factor_ic(s, close, h=h, method=method)
            r["key"] = col
            rows.append(r)
        except Exception:  # noqa: BLE001
            log.warning("[factor] IC 分析失败 %s", col, exc_info=True)
    rows.sort(key=lambda x: abs(x.get("rank_ic", 0.0)), reverse=True)
    return rows


def factor_group_returns(mat: pd.DataFrame, close: pd.Series, h: int = 1,
                         n_groups: int = 5) -> dict[str, dict]:
    """按因子分位数分组的未来收益（组均值），观察单调性。

    返回 {factor_key: {group: 1..n_groups, ret: 均值收益, monotonic: bool}}
    """
    fwd = forward_returns(close, h)
    out = {}
    for col in mat.columns:
        s = mat[col]
        tmp = pd.DataFrame({"f": s, "ret": fwd})
        tmp = tmp.dropna(subset=["f", "ret"])
        if len(tmp) < n_groups * 10:
            continue
        try:
            q = pd.qcut(tmp["f"].rank(method="first"), n_groups, labels=False) + 1
            tmp["g"] = q
            grp = tmp.groupby("g")["ret"].mean()
            groups = {int(k): round(float(v), 6) for k, v in grp.items()}
            rets = grp.to_numpy()
            monotonic = bool((np.diff(rets) >= 0).all() or (np.diff(rets) <= 0).all())
            spread = round(float(rets[-1] - rets[0]), 6)
            out[col] = {"groups": groups, "monotonic": monotonic,
                        "spread": spread, "top_minus_bottom": spread}
        except Exception:  # noqa: BLE001
            continue
    return out


def factor_correlation(mat: pd.DataFrame) -> pd.DataFrame:
    """因子间 Pearson 相关矩阵（去除全 NaN 列）。"""
    cols = [c for c in mat.columns if mat[c].notna().sum() > 5]
    if len(cols) < 2:
        return pd.DataFrame()
    return mat[cols].corr()


def factor_exposure(weights: dict[str, float], mat: pd.DataFrame) -> dict[str, float]:
    """组合因子（线性加权）对每个单因子的暴露度 = 组合与单因子的相关。"""
    if not weights or mat.empty:
        return {}
    cols = [c for c in mat.columns if c in weights and mat[c].notna().sum() > 5]
    if not cols:
        return {}
    combo = pd.Series(0.0, index=mat.index)
    for c in cols:
        z = (mat[c] - mat[c].mean()) / (mat[c].std() + 1e-12)
        combo = combo + z * weights[c]
    out = {}
    for c in cols:
        z = (mat[c] - mat[c].mean()) / (mat[c].std() + 1e-12)
        out[c] = round(float(combo.corr(z)), 4)
    return out


def find_redundant(mat: pd.DataFrame, threshold: float = 0.85) -> list[dict]:
    """相关性去冗余：返回高相关因子对。"""
    corr = factor_correlation(mat)
    if corr.empty:
        return []
    pairs = []
    seen = set()
    for i in corr.columns:
        for j in corr.columns:
            if i >= j or (i, j) in seen:
                continue
            v = corr.loc[i, j]
            if abs(v) >= threshold:
                pairs.append({"factor_a": i, "factor_b": j,
                              "correlation": round(float(v), 4)})
                seen.add((j, i))
    pairs.sort(key=lambda x: -abs(x["correlation"]))
    return pairs


# ============ 换手率 / fitness 评分 / 因子安检门 ============

# 因子安检门槛默认值（AI 挖掘、进化变体、模型因子共用同一套质检标准）
DEFAULT_GATES = {
    "min_abs_ic": 0.01,    # |rank_ic| 最低门槛（与 ic_weighted_composite 的 min_abs_ic 同口径）
    "min_abs_icir": 0.1,   # |icir| 稳定性门槛：IC 时高时低的因子不可信
    "max_turnover": 0.5,   # 换手上限（信号方向翻转率 0-1，高换手=高交易成本）
    "min_samples": 20,     # 有效样本下限
}


def factor_turnover(factor_series: pd.Series) -> float:
    """因子换手率：信号方向翻转频率（0-1）。

    借鉴 BRAIN turnover 语义，针对单标的时序因子做适配：
    统计因子符号在相邻两期发生翻转的比例（0=方向恒定，1=每期都翻=纯噪音）。
    高换手意味着高交易成本，fitness 评分会显著惩罚。
    """
    s = factor_series.astype(float).replace(0.0, np.nan)  # 0 值不参与方向判定
    signs = np.sign(s)
    flips = signs.diff().abs().dropna()
    if flips.empty:
        return 0.0
    return float((flips / 2.0).mean())


def factor_fitness(rank_ic: float, turnover: float, min_turnover: float = 0.125) -> float:
    """fitness 式综合评分（借鉴 BRAIN：fitness = Sharpe×√(|Returns|/max(Turnover,0.125))）。

    本地化适配（单标的时序因子）：
      fitness = |rank_ic| × √(1 / max(turnover, min_turnover))
    - |rank_ic| 类比信号强度（Sharpe 的角色）
    - 1/max(turnover, 0.125) 类比换手惩罚：低换手加分，极端高换手被压制
    """
    return float(abs(rank_ic) * math.sqrt(1.0 / max(turnover, min_turnover)))


def factor_quality_gate(series: pd.Series, close: pd.Series, h: int = 1,
                        gates: Optional[dict] = None,
                        method: str = "rank") -> dict:
    """因子安检门：一次性计算 IC/RankIC/ICIR/换手/fitness/分层单调性，并给出 valid 判定。

    用途：AI 挖掘因子、进化变体、模型因子上线前的统一质检关卡
    （借鉴 WorldQuant BRAIN 的 IS 检查 + Alphalens 因子评估）。

    返回 {valid, reason, ic, rank_ic, icir, ic_win_rate, turnover, fitness,
          spread, monotonic, samples, n_ic}
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    icr = factor_ic(series, close, h=h, method=method)
    s = series.astype(float)
    samples = int(s.notna().sum())
    turnover = factor_turnover(s)
    fitness = factor_fitness(icr.get("rank_ic", 0.0), turnover)
    # 分层收益（单调性 + 多空 spread）
    grp = factor_group_returns(pd.DataFrame({"f": s}), close, h=h, n_groups=5)
    spread = monotonic = None
    row = grp.get("f")
    if row:
        spread = row.get("spread")
        monotonic = row.get("monotonic")

    reason = []
    if samples < g["min_samples"]:
        reason.append(f"样本过少({samples}<{g['min_samples']})")
    if abs(icr.get("rank_ic", 0.0)) < g["min_abs_ic"]:
        reason.append(f"|rank_ic|={abs(icr.get('rank_ic', 0.0)):.4f}<{g['min_abs_ic']}")
    if abs(icr.get("icir", 0.0)) < g["min_abs_icir"]:
        reason.append(f"|icir|={abs(icr.get('icir', 0.0)):.4f}<{g['min_abs_icir']}")
    if turnover > g["max_turnover"]:
        reason.append(f"换手过高({turnover:.2f}>{g['max_turnover']})")

    return {
        "valid": not reason,
        "reason": "；".join(reason),
        "ic": icr["ic"], "rank_ic": icr["rank_ic"], "icir": icr["icir"],
        "ic_win_rate": icr["ic_win_rate"],
        "turnover": round(turnover, 4),
        "fitness": round(fitness, 4),
        "spread": spread, "monotonic": monotonic,
        "samples": samples, "n_ic": icr["n"],
    }
