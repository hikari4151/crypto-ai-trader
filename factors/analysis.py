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


def _numpy_rank(a: np.ndarray) -> np.ndarray:
    """平均秩，与 pandas Series.rank(method='average', na_option='keep') 逐位一致。

    run4 E1：numpy 原生替代 pd.Series(a).rank()（-34%）；run4 E2：并列分组
    向量化（flatnonzero 边界 + np.repeat 赋秩，去掉 Python while 扫描）。
    .optim/verify_fm_e2_rank_vec.py 1000 组随机/并列/NaN/全同 逐位 PASS。
    NaN 保持 NaN、不参与秩分配（na_option='keep'）。
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    ranks = np.empty(n, dtype=float)
    if n == 0:
        return ranks
    nan_mask = np.isnan(a)
    finite_idx = np.flatnonzero(~nan_mask)
    nf = finite_idx.size
    ranks[:] = np.nan
    if nf == 0:
        return ranks
    vals = a[finite_idx]
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    b = np.flatnonzero(np.concatenate(([True], sv[1:] != sv[:-1], [True])))
    avg = (b[:-1] + b[1:] - 1) / 2.0 + 1.0  # 1-based 平均秩
    counts = b[1:] - b[:-1]
    pos = np.empty(nf, dtype=float)
    pos[order] = np.repeat(avg, counts)
    ranks[finite_idx] = pos
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（有 NaN 则剔除后计算）。"""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    a, b = a[mask], b[mask]
    ra = _numpy_rank(a)
    rb = _numpy_rank(b)
    # E2：std 守卫等价替换——rank 值若不全同则 std≥0.5（>1e-12），
    # 全同则 std=0；np.all(ra==ra[0]) 与之同判且省两次 _var。
    if np.all(ra == ra[0]) or np.all(rb == rb[0]):
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _newey_west_std(series: np.ndarray, max_lags: int = None) -> float:
    """Newey-West 异方差自相关一致性标准差估计（Bartlett 核）。

    HAC 估计量：Var_longrun = gamma_0 + 2 * sum_{j=1}^{L} w_j * gamma_j
    - gamma_j = j 阶自协方差（除 n），w_j = 1 - j/(L+1)（Bartlett 核）
    - L = max_lags，默认取 N^(1/4) 经验法则

    IC 序列存在正自相关时，样本 std 低估真实波动 → ICIR 被高估。
    本函数返回自相关修正后的长程标准差：白噪声退化为 ≈ np.std(ddof=1)，
    正自相关时 > np.std(ddof=1)（对应 Lo 2002 风格的 AC 修正波动率）。
    """
    n = len(series)
    if n <= 1:
        return 0.0
    if n < 10:
        return float(np.std(series, ddof=1))
    if max_lags is None:
        max_lags = max(1, int(n ** 0.25))  # N^(1/4) 经验法则
    demeaned = series - series.mean()
    # gamma_0 = 方差（除 n）
    gamma_0 = float(np.sum(demeaned ** 2) / n)
    if gamma_0 <= 0:
        return 0.0
    # 自协方差 + Bartlett 核
    var_hac = gamma_0
    n_max_lag = min(max_lags, n - 2)
    for j in range(1, n_max_lag + 1):
        gamma_j = float(np.sum(demeaned[j:] * demeaned[:-j]) / n)
        w = 1.0 - j / (max_lags + 1)  # Bartlett 核
        var_hac += 2 * w * gamma_j
    if var_hac <= 0:  # 数值兜底：HAC 方差不可能为负（理论），样本下层正保护
        var_hac = gamma_0
    # 无偏调整（n-1），等价白噪声退化为 np.std(ddof=1)
    return float(np.sqrt(var_hac * n / (n - 1)))


def factor_ic(factor_series: pd.Series, close: pd.Series, h: int = 1,
              method: str = "rank", fwd=None) -> dict:
    """单个因子的 IC 分析。

    返回: {ic, rank_ic, icir, ic_mean, ic_std, ic_win_rate, n}
    fwd: 可选预计算的 forward_returns(close, h)。批量场景（factor_ic_table /
    mining 逐因子循环）由上层算一次传入，避免每因子重复两次 O(n) shift/除法；
    默认 None 时自算，行为与旧版完全一致。
    """
    f = factor_series.to_numpy(dtype=float)
    if fwd is None:
        fwd = forward_returns(close, h).to_numpy(dtype=float)
    elif hasattr(fwd, "to_numpy"):
        fwd = fwd.to_numpy(dtype=float)
    else:
        fwd = np.asarray(fwd, dtype=float)
    ic = _pearson(f, fwd)
    rank_ic = _spearman(f, fwd)

# 滑动 IC 序列（滚动 30 期）算 ICIR
    ic_series = _rolling_ic(factor_series, close, h=h, method=method, window=30, fwd=fwd)
    ics = ic_series[ic_series.notna()]
    ic_std = float(ics.std()) if len(ics) else 0.0
    # Newey-West 修正：IC 序列自相关导致样本 std 低估波动 → ICIR 高估。
    # ICIR 改用 HAC 长程标准差；原始 ic_std 保留用于对比。
    if len(ics):
        nw_std = _newey_west_std(ics.to_numpy(dtype=float))
        ic_std_nw = float(nw_std)
    else:
        nw_std = 0.0
        ic_std_nw = 0.0
    icir = float(ics.mean() / (ic_std_nw + 1e-12)) if len(ics) > 5 else 0.0
    ic_win = float((ics > 0).mean()) if len(ics) > 0 else 0.0

    return {
        "ic": round(ic, 6), "rank_ic": round(rank_ic, 6),
        "icir": round(icir, 6), "ic_mean": round(float(ics.mean()), 6) if len(ics) else 0.0,
        "ic_std": round(ic_std, 6) if len(ics) else 0.0,
        # 新增：Newey-West 修正标准差（自相关调整，白噪声退化为 ≈ ic_std）
        "ic_std_nw": round(ic_std_nw, 6) if len(ics) else 0.0,
        "ic_win_rate": round(ic_win, 4), "n": int(len(ics)),
    }


def _rolling_ic(factor_series: pd.Series, close: pd.Series, h: int = 1,
                method: str = "rank", window: int = 30, step: int = None,
                fwd=None) -> pd.Series:
    """滚动窗口 IC 序列（对过去 window 期求 IC，每 step 期采样一次）。

    step 默认等于 window（非重叠窗）：重叠窗会引入 IC 序列的自相关，
    std 被低估 → ICIR 系统性高估，因子选择门槛失真。非重叠窗的 IC
    序列样本独立，ICIR 才是因子稳定性的可靠度量。
    fwd: 可选预计算的 forward_returns(close, h)（factor_ic 传入，避免重复计算）。
    """
    if step is None:
        step = window
    f = factor_series.astype(float)
    if fwd is None:
        fwd = forward_returns(close, h)
    fwds = fwd.to_numpy(dtype=float) if hasattr(fwd, "to_numpy") else np.asarray(fwd, dtype=float)
    out = pd.Series(np.nan, index=factor_series.index)
    vals = f.to_numpy(dtype=float)
    idx = factor_series.index
    # E3（run4）：预分配 numpy 数组承接窗口结果，最后一次性构造 Series——
    # 原实现循环内 out.iloc[i]= 逐点赋值触发 pandas iloc setitem/__finalize__
    # 调度开销（factor_miner 训练链路 26 因子 × 41 窗 × 多环境构造，profile
    # Series 构造+finalize 占 wall 大头）；numpy 数组赋值后同一 Series 构造，
    # 输出值与索引逐位一致（.optim/verify_fm_e3_rolling.py 60 组×3 窗口 PASS）。
    out_arr = np.full(len(vals), np.nan, dtype=float)
    if method == "rank":
        for i in range(window, len(vals), step):
            out_arr[i] = _spearman(vals[i - window:i], fwds[i - window:i])
    else:
        for i in range(window, len(vals), step):
            out_arr[i] = _pearson(vals[i - window:i], fwds[i - window:i])
    return pd.Series(out_arr, index=idx)


def factor_ic_table(mat: pd.DataFrame, close: pd.Series, h: int = 1,
                    method: str = "rank") -> list[dict]:
    """批量因子 IC 分析表，按 |IC| 排序。"""
    rows = []
    # forward_returns 只依赖 close/h、与因子无关：批量前算一次全部因子共享，
    # 每因子由 2 次 O(n) shift/除法降为 0 次（factor_ic 与 _rolling_ic 各省一次）
    fwd = forward_returns(close, h)
    for col in mat.columns:
        s = mat[col]
        try:
            r = factor_ic(s, close, h=h, method=method, fwd=fwd)
            r["key"] = col
            rows.append(r)
        except (ValueError, TypeError):  # IC 分析：值/类型错误属于已知异常类型
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
        except (ValueError, TypeError):  # 分组收益：值/类型错误属于已知异常类型
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
    "min_abs_ic": 0.03,    # |rank_ic| 最低门槛（与 ic_weighted_composite 的 min_abs_ic 同口径）
    "min_abs_icir": 0.1,   # |icir| 稳定性门槛：IC 时高时低的因子不可信
    "max_turnover": 0.5,   # 换手上限（信号方向翻转率 0-1，高换手=高交易成本）
    "min_samples": 20,     # 有效样本下限
}


def factor_turnover(factor_series: pd.Series) -> float:
    """因子换手率：信号方向翻转频率（0-1）。

    借鉴 BRAIN turnover 语义，针对单标的时序因子做适配：
    统计因子符号在相邻两期发生翻转的比例（0=方向恒定，1=每期都翻=纯噪音）。
    高换手意味着高交易成本，fitness 评分会显著惩罚。

    run5 E2：numpy 化（原 pandas sign/diff/abs/dropna/mean 全链路；
    profile 显示 factor_miner 训练回合内 2038 次调用 cumtime 3.3s）。
    与 pandas 版逐位一致（.optim/verify_evolve_e2_turnover.py 2000 组
    含 0/NaN/常数 PASS）：replace(0,nan) → np.where；diff 首元素 NaN 被
    drop 等价；除以 2 为精确位运算；mean 归约同序。
    """
    v = factor_series.to_numpy(dtype=float)
    v = np.where(v == 0.0, np.nan, v)  # 0 值不参与方向判定
    d = np.abs(np.diff(np.sign(v)))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float((d / 2.0).mean())


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
                        method: str = "rank",
                        cross_validate_data: Optional[dict] = None,
                        cross_factor_fn: Optional = None) -> dict:
    """因子安检门：一次性计算 IC/RankIC/ICIR/换手/fitness/分层单调性，并给出 valid 判定。

    用途：AI 挖掘因子、进化变体、模型因子上线前的统一质检关卡
    （借鉴 WorldQuant BRAIN 的 IS 检查 + Alphalens 因子评估）。

    cross_validate_data: 可选跨品种验证数据 dict[symbol -> pd.DataFrame] (OHLCV)。
         需配合 cross_factor_fn (callable(df)->pd.Series) 使用，计算各品种因子序列后
         做 IC/ICIR 方向一致性检查。
    cross_factor_fn: 可调用对象，接收一个 OHLCV DataFrame 返回因子 Series。
         当 cross_validate_data 不为 None 且 cross_factor_fn 不为 None 时启用跨品种验证。
         当 cross_validate_data 为 None 时行为与旧版完全一致（不新增返回键）。

    返回 {valid, reason, ic, rank_ic, icir, ic_win_rate, turnover, fitness,
          spread, monotonic, samples, n_ic}
          当 cross_validate_data 提供时追加 cross_validation, cross_detail。
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

    result = {
        "valid": not reason,
        "reason": "；".join(reason),
        "ic": icr["ic"], "rank_ic": icr["rank_ic"], "icir": icr["icir"],
        "ic_win_rate": icr["ic_win_rate"],
        "turnover": round(turnover, 4),
        "fitness": round(fitness, 4),
        "spread": spread, "monotonic": monotonic,
        "samples": samples, "n_ic": icr["n"],
    }

    # 跨品种验证（仅当 cross_validate_data 和 cross_factor_fn 均提供时启用）
    if cross_validate_data is not None and cross_factor_fn is not None:
        main_ic = icr.get("rank_ic", 0.0)
        sv_cross = []
        for sym, cdf in cross_validate_data.items():
            if len(cdf) < 200:
                continue
            try:
                s_c = cross_factor_fn(cdf).astype(float)
                g_c = factor_quality_gate(s_c, cdf["close"], h=h, gates=gates, method=method)
                if g_c["samples"] >= 20:
                    sv_cross.append({"symbol": sym, "rank_ic": g_c["rank_ic"],
                                     "icir": g_c["icir"], "samples": g_c["samples"]})
            except Exception:
                continue
        if not sv_cross:
            result["cross_validation"] = "not_configured"
            result["cross_detail"] = []
        else:
            aligned = [x for x in sv_cross if np.sign(x["rank_ic"]) == np.sign(main_ic) or main_ic == 0.0]
            if len(aligned) == len(sv_cross) and len(aligned) > 0 and np.mean([x["rank_ic"] for x in aligned]) >= 0.01:
                result["cross_validation"] = "passed"
            else:
                result["cross_validation"] = "failed"
            result["cross_detail"] = sv_cross

    return result


def _consecutive_same_sign(roll: list, negative: bool = True) -> int:
    """统计序列末尾连续同符号段长度（用于 IC 衰变连续判定）。"""
    count = 0
    for v in reversed(roll):
        if (negative and v < 0) or (not negative and v > 0):
            count += 1
        else:
            break
    return count


def periodic_ic_refresh(data_by_symbol: dict[str, pd.DataFrame],
                        horizon: int = 1) -> dict:
    """用各品种最新行情计算所有已注册因子的滚动 IC，按衰变规则自动上线/下线。

    data_by_symbol: {symbol: OHLCV DataFrame}，每只品种须 ≥200 根K线。
    horizon: 预测期数（默认 1）。

    返回 {"updated": [key,...], "offlined": [key,...], "onlined": [key,...]}
    """
    from .library import _BY_KEY, _ALIVE, _IC_DECAY

    offlined, onlined = [], []
    for key, f in _BY_KEY.items():
        roll = _IC_DECAY.get(key, {"rolling_ic": []})["rolling_ic"]
        valid_ics = []
        for sym, df in data_by_symbol.items():
            if len(df) < 200:
                continue
            try:
                s = f.series(df).astype(float)
                if s.notna().sum() < 30:
                    continue
                r = factor_ic(s, df["close"], h=horizon, method="rank")
                valid_ics.append(r["rank_ic"])
            except Exception as e:
                log.debug("[factor] IC 刷新 %s 跳过 %s: %s", key, sym, e)
                continue
        if not valid_ics:
            continue
        roll.append(float(np.mean(valid_ics)))
        if len(roll) > 30:
            roll.pop(0)
        mean_ic = float(np.mean(roll))
        # icir = 滚动 IC 序列 mean / std（与 factor_ic 同口径：用 Newey-West
        # 修正标准差，自相关下样本 std 低估波动 → ICIR 高估；短序列退化为
        # np.std(ddof=1)；样本不足时 icir 置 0.0 阻止无意下线）
        if len(roll) >= 3:
            icir = float(np.mean(roll) / (_newey_west_std(np.asarray(roll, dtype=float)) + 1e-12))
        else:
            icir = 0.0
        default_decay = {"rolling_ic": [], "rolling_icir": 0.0, "last_updated": None,
                         "auto_offline": False, "auto_online": False}
        decay = _IC_DECAY.get(key, dict(default_decay))
        decay["rolling_ic"] = roll
        decay["rolling_icir"] = round(icir, 4)
        decay["last_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        _IC_DECAY[key] = decay

        alive = _ALIVE.get(key, True)
        if alive and mean_ic < -0.01 and icir < -0.1 and _consecutive_same_sign(roll, negative=True) >= 3:
            _ALIVE[key] = False
            decay["auto_offline"] = True
            offlined.append(key)
            log.warning("[factor] 因子 %s IC 衰变（mean_ic=%.4f icir=%.4f）→ 自动下线", key, mean_ic, icir)
        elif not alive and mean_ic > 0.01 and _consecutive_same_sign(roll, negative=False) >= 3:
            _ALIVE[key] = True
            decay["auto_online"] = True
            onlined.append(key)
            log.info("[factor] 因子 %s IC 恢复（mean_ic=%.4f）→ 自动上线", key, mean_ic)

    return {"updated": list(_BY_KEY.keys()), "offlined": offlined, "onlined": onlined}
