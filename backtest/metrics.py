"""回测绩效指标。"""
import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PERIODS_PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}


def compute_benchmark(closes: np.ndarray, start_cash: float, timeframe: str = "1h",
                      fee_rate: float = 0.0, slippage: float = 0.0) -> dict:
    """买入持有基准权益曲线与本段区间收益。

    closes: 原始收盘价序列（不含手续费/滑点）。
    fee_rate: 手续费率（双向，买入+卖出各扣一次），默认 0.0 保持向后兼容。
    slippage: 滑点（单向，买入时扣减），默认 0.0。
    excess_return/information_ratio/excess_max_drawdown 三个超额指标需要策略
    权益曲线，由 engine 层调用后回填；此处初始化占位并仅保证键恒在。
    """
    eq = np.asarray(closes, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {
            "buy_hold_ret": 0.0, "excess_return": 0.0,
            "information_ratio": 0.0, "excess_max_drawdown": 0.0,
            "bench_equity_curve": [round(start_cash, 4)],
        }
    # 基准权益曲线（不含交易成本，用于超额收益计算的 equity 差分）
    bench_equity = start_cash * eq / eq[0]

    # 含交易成本的买入持有收益
    buy_cost = start_cash * fee_rate
    effective_cash = start_cash - buy_cost
    initial_units = effective_cash / eq[0] * (1 - slippage)
    final_value = initial_units * eq[-1]
    sell_cost = final_value * fee_rate
    final_value -= sell_cost
    buy_hold_ret = final_value / start_cash - 1.0

    return {
        "buy_hold_ret": round(buy_hold_ret, 6),
        "excess_return": 0.0,
        "information_ratio": 0.0,
        "excess_max_drawdown": 0.0,
        "bench_equity_curve": [round(float(x), 4) for x in bench_equity],
    }


def compute_metrics(equity: list[float], trades: list[dict], timeframe: str = "1h",
                    start_cash: float = 10000.0,
                    closes: np.ndarray | None = None,
                    traded_notional: float = 0.0,
                    total_costs: float = 0.0) -> dict[str, Any]:
    eq = np.asarray(equity, dtype=float)
    n = len(eq)
    if n < 2:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    total_return = eq[-1] / start_cash - 1.0
    periods = PERIODS_PER_YEAR.get(timeframe, 8760)
    annual_return = (1 + total_return) ** (periods / n) - 1.0 if total_return > -1 else -1.0

    # 最大回撤
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_drawdown = float(drawdown.max())

    # 夏普（回测序列为总体，用 ddof=0 总体标准差，避免系统性高估）
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    std = rets.std(ddof=0)
    sharpe = float(rets.mean() / std * math.sqrt(periods)) if std > 1e-12 else 0.0

    # 交易统计
    pnls = [t.get("pnl", 0.0) for t in trades if t.get("side") == "sell"]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # ---- P1-6 换手 / 持仓时长 / 成本拖累 ----
    sells = [t for t in trades if t.get("side") == "sell"]
    # 平均持仓时长（根K线）：卖单带 entry_i/exit_i（撮合内核写入）
    holding = [t["exit_i"] - t["entry_i"] for t in sells
               if t.get("entry_i") is not None and t.get("exit_i") is not None
               and t["exit_i"] >= t["entry_i"]]
    avg_holding_bars = float(np.mean(holding)) if holding else 0.0
    # 换手率：总成交额 / 平均权益（成交额由引擎撮合内核累计传入）
    avg_equity = float(eq.mean()) if n else start_cash
    turnover = traded_notional / avg_equity if avg_equity > 0 else 0.0
    # 成本拖累：总成本占毛盈亏（反映成本对净收益的侵蚀）
    gross_abs = abs(gross_profit) + abs(gross_loss)
    cost_drag = total_costs / gross_abs if gross_abs > 0 else 0.0

    result = {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        # profit_factor 为 inf（全额盈利）时用大数哨兵，保证下游 JSON/类型一致
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else 999999.0,
        "payoff_ratio": round(payoff_ratio, 4),
        "total_trades": len(pnls),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "final_equity": round(float(eq[-1]), 4),
        # ---- P1-6 ----
        "avg_holding_bars": round(avg_holding_bars, 2),
        "turnover": round(turnover, 4),
        "total_costs": round(total_costs, 4),
        "cost_drag": round(cost_drag, 6),
    }
    # 可选基准计算：传入 closes 时自动计算并填入 benchmark 键
    if closes is not None:
        result["benchmark"] = compute_benchmark(closes, start_cash, timeframe)
    return result

def finalize_metrics(equity_curve: list[float], trades: list[dict],
                     timeframe: str, start_cash: float,
                     closes: np.ndarray,
                     engine_tag: str | None = None,
                     elapsed_sec: float | None = None,
                     traded_notional: float = 0.0,
                     total_fees: float = 0.0,
                     bootstrap: bool = True) -> dict:
    """统一计算回测指标 + 基准 + 超额指标回填。

    两个回测引擎共用：compute_metrics + compute_benchmark + 超额指标
    （excess_return / information_ratio / excess_max_drawdown）回填，
    消除此前双引擎各自实现一段超额回填导致的隐性不一致。

    engine_tag: 若提供则写入 metrics["backtest_engine"]。
    elapsed_sec: 若提供则写入 metrics["elapsed_sec"]。
    traded_notional / total_fees: P1-6 撮合内核累计的成交额与总成本，
        用于换手率 / 成本拖累指标（默认 0 向后兼容）。
    bootstrap: 是否计算 block bootstrap 置信区间（P2-13）。网格扫描/成本
        扫描只消费基础指标，跳过 1000 次重采样可省约 50ms/回测 × 组合数；
        默认 True 保持单次回测报告完整。
    """
    metrics = compute_metrics(equity_curve, trades, timeframe, start_cash, closes=closes,
                              traded_notional=traded_notional, total_costs=total_fees)
    # ---- P1-7 统计显著性：stationary block bootstrap 夏普/收益置信区间 ----
    # 在 finalize_metrics（每次回测仅调用一次）计算，避免网格扫描/过拟合等
    # 高频 compute_metrics 路径引入 bootstrap 开销。替换 UI 原先展示的 iid CI：
    # block bootstrap 保留收益自相关结构（趋势/波动聚集），对策略显著性判断更保守可信。
    if bootstrap:
        try:
            _rets = equity_returns(equity_curve)
            if len(_rets) >= 20:
                _bb = block_bootstrap_sharpe_ci(_rets, n_boot=1000, seed=7)
                if _bb.get("sharpe_ci"):
                    metrics["sharpe_ci"] = _bb["sharpe_ci"]
                    metrics["ret_ci"] = _bb["ret_ci"]
                    metrics["p_sharpe_pos"] = _bb["p_sharpe_pos"]
                    metrics["bootstrap_method"] = "block"
                    metrics["bootstrap_block_len"] = _bb.get("block_len")
                    # 区间收益展示：与年化收益并列的置信区间（由 ret_ci 提供）
                    metrics["interval_ret_low"] = _bb["ret_ci"][0]
                    metrics["interval_ret_high"] = _bb["ret_ci"][1]
        except Exception as e:  # noqa: BLE001 统计信息失败不阻断主结果
            log.warning("[metrics] block bootstrap 计算失败（忽略）: %s", e)
    if engine_tag is not None:
        metrics["backtest_engine"] = engine_tag
    if elapsed_sec is not None:
        metrics["elapsed_sec"] = round(elapsed_sec, 4)
    # 基准（含手续费/滑点口径由 compute_benchmark 内计算）
    benchmark = metrics["benchmark"]
    bench_eq = np.asarray(benchmark["bench_equity_curve"], dtype=float)
    seq = np.asarray(equity_curve, dtype=float)
    if len(bench_eq) == len(seq) and len(seq) >= 2:
        excess = seq[-1] / start_cash - 1.0 - benchmark["buy_hold_ret"]
        periods = PERIODS_PER_YEAR.get(timeframe, 8760)
        s_ret = np.diff(seq) / (seq[:-1] + 1e-9)
        b_ret = np.diff(bench_eq) / (bench_eq[:-1] + 1e-9)
        diff = s_ret - b_ret
        ir = float(diff.mean() / (diff.std() + 1e-12) * math.sqrt(periods)) if len(diff) > 5 else 0.0
        peak = np.maximum.accumulate(seq)
        dd = (peak - seq) / (peak + 1e-12)
        ex_mdd = float(dd.max())
        benchmark.update({"excess_return": round(excess, 6),
                          "information_ratio": round(ir, 4),
                          "excess_max_drawdown": round(ex_mdd, 6)})
    return metrics


# ---------- 统计显著性：Deflated Sharpe + Bootstrap 置信区间 ----------

_EULER_GAMMA = 0.5772156649015329  # 欧拉-马歇罗尼常数（DSR 极值分布用）


def _norm_cdf(x: float) -> float:
    """标准正态 CDF（math.erf，避免依赖 scipy）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态分位数（Acklam 有理近似，误差 < 1.15e-9；避免依赖 scipy）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def deflated_sharpe_ratio(returns, trials: int = 1, trial_sharpes=None) -> dict:
    """缩水夏普比率 DSR（Bailey & López de Prado 2014）。

    多重测试校正：网格扫描/AI 迭代试过 N 个配置后，即使全部真实 SR=0，
    选出的最大样本 SR 也会系统性偏高（选择偏差）。DSR 回答：
    "扣除运气后，这个夏普仍显著 > 0 的概率"。DSR ≥ 0.95 视为统计显著。

    returns: 单周期收益率序列（等权益差分，非年化）
    trials: 搜索过的配置总数（网格候选数 / AI 迭代版本数；1=单次回测）
    trial_sharpes: 各候选的样本 SR 列表（可得时用于估计 SR 跨试验方差，更准）
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20 or trials < 1:
        return {"dsr": None, "sr": 0.0, "sr0": 0.0, "trials": max(1, int(trials)), "n_obs": n}

    mu, sigma = r.mean(), r.std(ddof=0)
    if sigma <= 1e-12:
        return {"dsr": None, "sr": 0.0, "sr0": 0.0, "trials": max(1, int(trials)), "n_obs": n}
    sr = mu / sigma  # 单周期夏普（DSR 公式口径，勿年化）

    # 偏度/峰度（原点矩，非超额）
    z = (r - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))

    # SR₀ = 全部真实 SR=0 时 N 次独立试验的最大样本 SR 期望
    n_trials = max(1, int(trials))
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        v_sr = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        # 单试验近似：H0 下 SR 估计量方差 ≈ 1/(n-1)
        v_sr = 1.0 / (n - 1)
    if n_trials > 1:
        e_max = (1 - _EULER_GAMMA) * _norm_ppf(1 - 1.0 / n_trials) \
            + _EULER_GAMMA * _norm_ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(max(v_sr, 0.0)) * e_max
    else:
        sr0 = 0.0

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(max(denom, 1e-12))
    dsr = float(_norm_cdf((sr - sr0) * math.sqrt(n - 1) / denom))
    return {"dsr": round(dsr, 4), "sr": round(sr, 6), "sr0": round(sr0, 6),
            "trials": n_trials, "n_obs": n}


def bootstrap_sharpe_ci(returns, n_boot: int = 1000, alpha: float = 0.05,
                        periods_per_year: int = 8760, seed: int = 7) -> dict:
    """iid Bootstrap 夏普/总收益置信区间 + P(夏普>0)。

    对单周期收益率重采样（n_boot 次），每次计算年化夏普与总收益，
    取经验分位数做区间。区间不含 0 / P(夏普>0) 接近 1 → 表现非运气。

    P1-7 说明：iid 重采样假设收益率独立同分布；对存在自相关（趋势/均值回归）
    的收益序列，推荐用 block_bootstrap_sharpe_ci（stationary block bootstrap，
    保留自相关结构）。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return {"sharpe_ci": None, "ret_ci": None, "p_sharpe_pos": None, "n_boot": 0, "n_obs": n}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = r[idx]  # (n_boot, n)
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=0)
    sd = np.where(sd <= 1e-12, np.nan, sd)
    sharpes = mu / sd * math.sqrt(periods_per_year)
    rets = np.prod(1.0 + samples, axis=1) - 1.0

    ok = np.isfinite(sharpes)
    sharpes_f = sharpes[ok]
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    if len(sharpes_f) < 10:
        return {"sharpe_ci": None, "ret_ci": None, "p_sharpe_pos": None,
                "n_boot": int(ok.sum()), "n_obs": n}
    return {
        "sharpe_ci": [round(float(np.quantile(sharpes_f, lo)), 4),
                      round(float(np.quantile(sharpes_f, hi)), 4)],
        "ret_ci": [round(float(np.quantile(rets[ok], lo)), 6),
                   round(float(np.quantile(rets[ok], hi)), 6)],
        "p_sharpe_pos": round(float((sharpes_f > 0).mean()), 4),
        "n_boot": int(ok.sum()), "n_obs": n,
    }


def block_bootstrap_sharpe_ci(returns, n_boot: int = 1000, alpha: float = 0.05,
                              periods_per_year: int = 8760, seed: int = 7,
                              block_len: int | None = None) -> dict:
    """Stationary Block Bootstrap 夏普/总收益置信区间 + P(夏普>0)（P1-7）。

    相比 iid 重采样，block bootstrap 以长度随机的块为单位重采样，
    保留收益序列的自相关结构（趋势/波动聚集），避免 iid 假设高估显著性。
    适用于高频/趋势策略的置信区间估计。

    block_len: 平均块长（根K线）。默认取 n^(1/3)（Politis & Romano 经验值），
        上限 2*floor(sqrt(n)) 保证块长不过大。

    性能（0.5.0 优化）：原实现每样本 while 循环逐块采样（n_boot 次 × 几何
    分布逐次调用 rng），5000 根 × 1000 次 bootstrap 约 1.3s，占单次回测 90%。
    现在改为：几何块长/起点矩阵一次生成，逐样本用 numpy 批量拼接索引，
    统计量批量计算。视觉/统计口径不变（块长 ~Geo(1/b)、起点均匀、块到数组
    末尾截断、样本截断到 n 根的语义与原实现一致）。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return {"sharpe_ci": None, "ret_ci": None, "p_sharpe_pos": None, "n_boot": 0, "n_obs": n,
                "method": "block"}

    if block_len is None or block_len < 1:
        block_len = max(1, min(int(round(n ** (1.0 / 3.0))), int(2 * np.sqrt(n))))
    b = int(block_len)
    rng = np.random.default_rng(seed)

    # 每样本需要的最大块数：均值 n/b 块；几何分布长尾下 3 倍余量 + 16 保险。
    # 一次生成全部随机数（替代原实现 n_boot 次 while 循环内的逐次 rng 调用）
    k_max = int(np.ceil(n / b) * 3) + 16
    lens = rng.geometric(1.0 / b, size=(n_boot, k_max))
    starts = rng.integers(0, n, size=(n_boot, k_max))

    sharpes = np.empty(n_boot)
    rets = np.empty(n_boot)
    for k in range(n_boot):
        # 块有效长度：截断到数组末尾（end = min(n, start+len)，与原实现一致）
        eff = np.minimum(lens[k], n - starts[k])
        c = np.cumsum(eff)
        t = int(np.searchsorted(c, n, side="left"))  # 填满 n 根所需块数
        t = min(t + 1, k_max)
        # 块号索引 + 块内偏移（np.repeat 展开成逐位置数组）
        reps = eff[:t]
        # 可能出现 cum 首块即 ≥ n（块长超过剩余容量），仍只取前 n 根
        blk = np.repeat(np.arange(t), reps)
        starts_pos = np.concatenate(([0], c[: t - 1]))
        off = np.arange(len(blk)) - np.repeat(starts_pos, reps)
        idx = starts[k][blk] + off
        idx = idx[:n]
        if len(idx) < n:  # 余量不足（几何长尾极小概率）：循环补足到 n
            s_need = n - len(idx)
            while s_need > 0:
                s0 = int(rng.integers(0, n))
                seg = r[s0:min(n, s0 + s_need)]
                idx = np.concatenate((idx, np.arange(s0, s0 + len(seg))))
                s_need -= len(seg)
        sample = r[idx]
        mu = float(sample.mean())
        sd = float(sample.std(ddof=0))
        if sd > 1e-12:
            sharpes[k] = mu / sd * math.sqrt(periods_per_year)
        else:
            sharpes[k] = np.nan
        rets[k] = np.prod(1.0 + sample) - 1.0

    ok = np.isfinite(sharpes)
    sharpes_f = sharpes[ok]
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    if len(sharpes_f) < 10:
        return {"sharpe_ci": None, "ret_ci": None, "p_sharpe_pos": None,
                "n_boot": int(ok.sum()), "n_obs": n, "method": "block"}
    return {
        "sharpe_ci": [round(float(np.quantile(sharpes_f, lo)), 4),
                      round(float(np.quantile(sharpes_f, hi)), 4)],
        "ret_ci": [round(float(np.quantile(rets[ok], lo)), 6),
                   round(float(np.quantile(rets[ok], hi)), 6)],
        "p_sharpe_pos": round(float((sharpes_f > 0).mean()), 4),
        "n_boot": int(ok.sum()), "n_obs": n, "method": "block",
        "block_len": b,
    }


def equity_returns(equity: list[float]) -> np.ndarray:
    """权益曲线 → 单周期收益率（bootstrap/DSR 输入口径）。"""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return np.array([])
    return np.diff(eq) / (np.abs(eq[:-1]) + 1e-9)
