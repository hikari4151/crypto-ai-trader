"""多标的组合层回测（P2-11）：研究层组合视角，不改变实盘单标的引擎。

问题：当前回测/实盘都是单标的逐个跑，缺乏组合视角——无法回答
"两个策略或两个币种叠加后的组合夏普/回撤/相关性"。

本模块：
1. 对多个标的分别回测（可共用同一策略或按标的指定策略），资金按权重分配；
2. 按时间轴对齐各标的权益曲线，相加得组合权益曲线；
3. 计算组合指标 + 标的策略两两收益相关性矩阵（分散度检验）。
4. 权重支持：equal（等权）| vol_inv（过去N根波动率倒数）| 显式字典。

研究层定位：仅做"回测分析"输出，不接入实盘撮合，避免引入组合级
执行语义（跨币种资金再平衡成本等）到实盘路径。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .engine import BacktestConfig
from .metrics import compute_benchmark, compute_metrics, finalize_metrics

log = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """组合回测配置。

    symbols: 标的列表（与 strategies 一一对应，缺省共用 strategy_name）。
    strategies: {symbol: 策略名}，缺省全部用 strategy_name。
    weights: "equal" | "vol_inv" | {symbol: 权重}。默认 equal。
    vol_lookback: vol_inv 权重使用的波动率回看K线数。
    start_cash: 组合总资金（按权重分配至各标的）。
    fee_rate / slippage / maker_fee_rate / taker_fee_rate / funding_rate /
    participation_rate / intrabar_stops / cleaning_mode / limit_order_model:
        传给各标的 BacktestConfig（口径与单标的回测一致）。
    params: {symbol: 参数字典}，缺省用 strategy_params 对全部标的生效。
    """
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    strategy_name: str = "dual_ma"
    strategies: dict = field(default_factory=dict)     # {symbol: 策略名}
    params: dict = field(default_factory=dict)          # {symbol: 参数}
    strategy_params: dict = field(default_factory=dict)  # 全标默认参数
    weights: str | dict = "equal"                       # equal | vol_inv | {sym: w}
    vol_lookback: int = 120
    start_cash: float = 20000.0
    timeframe: str = "1h"
    fee_rate: float = 0.001
    slippage: float = 0.0005
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    funding_rate: float = 0.0
    participation_rate: float = 0.0
    intrabar_stops: bool = True
    cleaning_mode: str = "mark"
    limit_order_model: str = "none"
    run_backtest: Optional[Callable] = None            # 可注入回测函数


def _resolve_weights(cfg: PortfolioConfig, closes_by_symbol: dict[str, np.ndarray]) -> dict[str, float]:
    n = len(cfg.symbols)
    if isinstance(cfg.weights, dict):
        w = {s: float(cfg.weights.get(s, 0.0)) for s in cfg.symbols}
    elif cfg.weights == "vol_inv":
        w = {}
        for s in cfg.symbols:
            closes = closes_by_symbol[s]
            rets = np.diff(closes) / (np.abs(closes[:-1]) + 1e-9)
            vol = float(rets[-cfg.vol_lookback:].std(ddof=0)) if len(rets) >= 10 else 1e-9
            w[s] = 1.0 / (vol + 1e-9)
    else:  # equal
        w = {s: 1.0 for s in cfg.symbols}
    total = sum(w.values())
    if total <= 0:
        raise ValueError("组合权重求和必须为正")
    return {s: wt / total for s, wt in w.items()}


def run_portfolio_backtest(data_by_symbol: dict[str, pd.DataFrame],
                           cfg: PortfolioConfig,
                           on_progress: Optional[Callable[[str], None]] = None) -> dict:
    """多标的组合回测：逐标的回测 → 对齐权益曲线 → 组合指标 + 相关性矩阵。

    data_by_symbol: {symbol: OHLCV DataFrame}（各标的周期需相同）。
    返回：
    {
        "per_symbol": [{symbol, strategy, weight, metrics, equity_curve, trades}],
        "portfolio_metrics": {...},            # 组合权益曲线算出的指标
        "portfolio_equity_curve": [...],
        "correlation": {symbol: {other_symbol: pearson}},   # 标的收益相关矩阵
        "benchmark": {...},                    # 组合层面买入持有基准（等权假设）
        "weights": {symbol: w},
    }
    """
    if not data_by_symbol:
        raise ValueError("组合回测需要至少一个标的")
    if on_progress:
        on_progress("[portfolio] 加载各标的K线")

    closes_by_symbol: dict[str, np.ndarray] = {}
    for s, df in data_by_symbol.items():
        closes_by_symbol[s] = df["close"].to_numpy(float)

    weights = _resolve_weights(cfg, closes_by_symbol)

    run_fn = cfg.run_backtest or _default_run
    per_symbol: list[dict] = []
    equity_aligned: dict[str, np.ndarray] = {}

    # ---- 逐标的回测 ----
    for idx, symbol in enumerate(cfg.symbols):
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            raise ValueError(f"缺少标的数据: {symbol}")
        strat = cfg.strategies.get(symbol, cfg.strategy_name)
        params = cfg.params.get(symbol, dict(cfg.strategy_params))
        sub_cash = cfg.start_cash * weights[symbol]
        if on_progress:
            on_progress(f"[portfolio] {idx + 1}/{len(cfg.symbols)} 回测 {symbol} (w={weights[symbol]:.3f})")
        bcfg = BacktestConfig(
            symbol=symbol, timeframe=cfg.timeframe, strategy_name=strat,
            strategy_params=dict(params), start_cash=sub_cash,
            fee_rate=cfg.fee_rate, slippage=cfg.slippage,
            maker_fee_rate=cfg.maker_fee_rate, taker_fee_rate=cfg.taker_fee_rate,
            funding_rate=cfg.funding_rate, participation_rate=cfg.participation_rate,
            intrabar_stops=cfg.intrabar_stops, cleaning_mode=cfg.cleaning_mode,
            limit_order_model=cfg.limit_order_model,
        )
        res = run_fn(df, bcfg)
        metrics = res["metrics"]
        equity = res["equity_curve"]
        per_symbol.append({
            "symbol": symbol, "strategy": strat, "weight": round(weights[symbol], 6),
            "metrics": metrics, "equity_curve": equity,
            "benchmark": res.get("benchmark") or metrics.get("benchmark"),
            "trades": res.get("trades", []),
        })
        # 按行号索引对齐（各标的 K 线数量可能不同：取公共长度）
        equity_aligned[symbol] = np.asarray(equity, dtype=float)

    # ---- 对齐：按最短标的权益曲线长度截断，逐点加权求和（权重已在 start_cash 分配，
    # 这里直接用各标的权益之和 = 组合权益；权重再乘一次会双重计权，故不乘） ----
    min_len = min(len(e) for e in equity_aligned.values())
    combo = np.zeros(min_len)
    for s in cfg.symbols:
        combo += equity_aligned[s][:min_len]

    combo_metrics = compute_metrics(list(combo), [], cfg.timeframe, cfg.start_cash)
    # 组合基准：各标的基准加权（在截断窗口上）
    closes_pad = {}
    for s in cfg.symbols:
        arr = closes_by_symbol[s]
        closes_pad[s] = arr[:min_len]
    # 买入持有等权基准：各标的直接 b&h 组合（简化假设：权重代表初始资金占比）
    bench_eq = np.zeros(min_len)
    for s in cfg.symbols:
        bcfg0 = BacktestConfig(symbol=s, timeframe=cfg.timeframe, strategy_name=cfg.strategy_name,
                               start_cash=cfg.start_cash * weights[s], fee_rate=cfg.fee_rate,
                               slippage=cfg.slippage)
        bm = compute_benchmark(closes_pad[s], bcfg0.start_cash, cfg.timeframe, cfg.fee_rate, cfg.slippage)
        bench_eq += np.asarray(bm["bench_equity_curve"], dtype=float)
    bench = {"buy_hold_ret": round(float(bench_eq[-1] / cfg.start_cash - 1.0), 6),
             "bench_equity_curve": [round(float(x), 4) for x in bench_eq]}

    # ---- 标的收益相关性矩阵 ----
    rets_map: dict[str, np.ndarray] = {}
    for s in cfg.symbols:
        e = equity_aligned[s][:min_len]
        rets_map[s] = np.diff(e) / (np.abs(e[:-1]) + 1e-9)
    correlation: dict[str, dict] = {}
    for s1 in cfg.symbols:
        correlation[s1] = {}
        for s2 in cfg.symbols:
            r1, r2 = rets_map[s1], rets_map[s2]
            corr = float(np.corrcoef(r1, r2)[0, 1]) if len(r1) > 2 else 0.0
            correlation[s1][s2] = round(0.0 if np.isnan(corr) else corr, 4)

    return {
        "per_symbol": per_symbol,
        "portfolio_metrics": combo_metrics,
        "portfolio_equity_curve": [round(float(x), 4) for x in combo],
        "correlation": correlation,
        "benchmark": bench,
        "weights": {s: round(w, 6) for s, w in weights.items()},
    }


def _default_run(df: pd.DataFrame, bcfg: BacktestConfig) -> dict:
    from .fast_engine import run_backtest_fast
    # P2-13：组合层逐标的回测只消费基础指标（组合/相关性在组合层聚合）
    return run_backtest_fast(df, bcfg, bootstrap=False)