"""同品种 walk-forward 多折评估的纯函数（引擎观测与 scripts 工具共用）。

背景（2026-09-19）：策略只做 ETH/5m，不引入跨品种门；改为"换一段历史也赚"
验证——固定 DRL 模型在多个互不重叠的时间窗口上 greedy 评估（TradingEnv，
与进化引擎 `_cross_symbol_oos` 同构），按方向一致性 + 成交样本量判定。

本模块只放纯函数/同步逻辑；异步封装（拉数、to_thread）在调用方
（`drl/evolve_engine.py::_observe_strategy_drl_walkforward`、
`scripts/walkforward_rl_evolve.py::main`）。
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


def expanding_fold_bounds(n: int, warmup: int, n_folds: int) -> list[tuple[int, int]]:
    """把 [0, n) 切成 n_folds 个互不重叠的评估段（跳过前 warmup 根暖机）。

    返回 [(start, end), ...]，相邻段首尾相接、最后一段覆盖到数据末尾。
    数据不足以容纳 n_folds 段时抛 ValueError。
    """
    if n <= warmup or (n - warmup) < n_folds:
        raise ValueError(f"数据不足：{n} 根K线中可评估段 {n - warmup} < {n_folds} 折")
    seg = (n - warmup) // n_folds
    bounds = []
    for i in range(n_folds - 1):
        bounds.append((warmup + i * seg, warmup + (i + 1) * seg))
    bounds.append((warmup + (n_folds - 1) * seg, n))
    return bounds


def summarize_folds(folds: list[dict], min_trades: int = 8) -> dict:
    """按方向一致性与成交样本量给出多折评估结论。"""
    n = len(folds)
    if n == 0:
        return {"verdict": "INCONCLUSIVE", "n_folds": 0, "positive_folds": 0,
                "win_rate": 0.0, "median_ret": 0.0, "mean_ret": 0.0,
                "reason": "无评估折"}
    rets = [float(f.get("oos_ret", 0.0) or 0.0) for f in folds]
    trades = [int(f.get("oos_trades", 0) or 0) for f in folds]
    positive = sum(1 for r in rets if r > 0)
    win_rate = positive / n
    median_ret = float(np.median(rets))
    mean_ret = float(np.mean(rets))
    reason = ""
    if any(t < min_trades for t in trades):
        verdict = "INCONCLUSIVE"
        reason = f"存在成交样本不足的折（min_trades={min_trades}，实际 {trades}）"
    elif win_rate >= 0.75 and median_ret > 0:
        verdict = "PASS"
        reason = f"{positive}/{n} 折为正、中位数收益 {median_ret:+.4f} > 0"
    else:
        verdict = "FAIL"
        reason = (f"方向一致性不足：{positive}/{n} 折为正（win_rate={win_rate:.2f}）"
                  f"或中位数收益 {median_ret:+.4f} ≤ 0")
    return {"verdict": verdict, "n_folds": n, "positive_folds": positive,
            "win_rate": win_rate, "median_ret": median_ret, "mean_ret": mean_ret,
            "reason": reason}


def _annual_bars(timeframe: str) -> float:
    return {
        "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2, "1h": 365 * 24, "4h": 365 * 6, "1d": 365,
    }.get(str(timeframe).strip().lower(), 365 * 24 * 12)


def metrics_from_traj(traj: dict, timeframe: str) -> dict:
    """从 run_episode 轨迹汇总单折指标。"""
    actions = np.asarray(traj.get("actions", []), dtype=int)
    trades = int(((actions == 1) | (actions == 2)).sum()) if len(actions) else 0
    infos = list(traj.get("infos") or [])
    eq = np.asarray([float(i.get("equity", 0.0) or 0.0) for i in infos], dtype=float)
    rets_pct = np.asarray([float(i.get("ret_pct", 0.0) or 0.0) for i in infos], dtype=float)
    max_dd = 0.0
    if len(eq):
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-9)
        max_dd = float(dd.min()) if len(dd) else 0.0
    sharpe = 0.0
    if len(rets_pct) > 5 and float(np.std(rets_pct)) > 0:
        sharpe = float(np.mean(rets_pct) / np.std(rets_pct) * np.sqrt(_annual_bars(timeframe)))
    return {
        "oos_ret": float(traj.get("total_ret", 0.0) or 0.0),
        "oos_trades": trades,
        "max_drawdown": max_dd,
        "sharpe_annual": sharpe,
        "final_position_ratio": float(traj.get("final_position_ratio", 0.0) or 0.0),
    }


def build_factor_column(df, weights: dict, mu: float = 0.0, sd: float = 1.0):
    """按 rl_adaptive 部署端同口径复算级联组合因子列（长度 == len(df)）。

    composite_factor(z_mode="expanding") 与 factor_miner 训练端同式（无前视），
    再按模型文件里的训练段 mu/sd 二次标准化，NaN 填 0（与 train_drl 注入一致）。
    无有效权重时返回 None。
    """
    w = {str(k): float(v) for k, v in (weights or {}).items() if v}
    if not w:
        return None
    from factors.engine import compute_factor_matrix
    from factors.mining import composite_factor
    mat = compute_factor_matrix(df, keys=list(w))
    combo = composite_factor(mat, w, method="ic", z_mode="expanding")
    vals = (combo.to_numpy(float) - float(mu)) / float(sd or 1.0)
    return np.nan_to_num(vals, nan=0.0)
