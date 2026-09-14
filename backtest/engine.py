"""事件驱动回测引擎：逐K线仿真撮合，计入手续费与滑点。

撮合循环（pending_signal / 限价单 / FIFO 记账 / 期末平仓）已抽到
backtest/_matching.py 共享内核，本引擎与 fast_engine 共用同一实现，
消除此前双引擎各维护一份相同撮合逻辑导致的分叉风险。
"""
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Optional

import pandas as pd

from factors.library import ignoring_factor_liveness
from indicators.technical import SR_MIN_TOUCHES, SR_WINDOW, price_action_features_series, support_resistance_series
from indicators.vectorized import precompute_indicator_series
from strategies.base import strategy_ma_periods

from ._matching import forced_liquidation, needs_sr, run_matching_loop
from .metrics import finalize_metrics

log = logging.getLogger(__name__)


def research_scope(fn: Callable) -> Callable:
    """把回测全程包进「忽略因子上下线状态」的研究作用域。

    IC 衰变下线是运维态，只活在进程内存里。回测/参数优化/过拟合检验若读它，
    同一段历史会因「今天有没有跑过 IC 衰变作业」得出不同结论——不可复现，
    且与研究口径和实盘口径分叉。
    """
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        with ignoring_factor_liveness():
            return fn(*args, **kwargs)

    return _wrapped


@dataclass
class BacktestConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy_name: str = "dual_ma"
    strategy_params: dict = field(default_factory=dict)
    start_cash: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    start: Optional[str] = None
    end: Optional[str] = None
    limit_order_model: str = "none"  # "none" | "partial" | "probabilistic"
    # ---- P0-1/P0-2/P0-3 优化项新增字段（默认保持旧行为，向后兼容） ----
    # P0-3 成本模型：maker/taker 分离费率（默认 None → 沿用 fee_rate 单一费率）
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    # P0-3 资金费率（每根K线按持仓价值收取的比例；0=关闭，默认）
    funding_rate: float = 0.0
    # P0-2 成交量参与率约束：买单数量上限 = 该K线成交量 × participation_rate
    # （0=不限，默认，保持旧行为）
    participation_rate: float = 0.0
    # P0-1 intrabar 止损/止盈触价建模开关（策略声明 protective_levels 时生效）
    intrabar_stops: bool = True
    # P1-9 数据清洗模式："mark"=标记不清洗（默认，保留真实行情）| "replace"=旧行为
    cleaning_mode: str = "mark"


@research_scope
def run_backtest(df: pd.DataFrame, cfg: BacktestConfig,
                 bootstrap: bool = True) -> dict:
    """执行回测，返回 {metrics, equity_curve, trades}。

    bootstrap: 是否计算 block bootstrap 置信区间（P2-13）。扫描路径
        （overfit/compare 等批量回测）传 False 跳过，只保留基础指标。
    """
    if df.empty:
        raise ValueError("回测数据为空")
    data = df.copy()
    # 数据清洗：异常值检测 + 缺失值填充，避免脏数据污染回测/参数优化。
    # cleaning_mode="mark"（默认）：只做结构完整性修复，不清洗真实行情（P1-9）
    from .data_loader import OHLCVSanitizer
    sanitizer = OHLCVSanitizer()
    data = sanitizer.clean(data, mode=getattr(cfg, "cleaning_mode", "mark"))
    if cfg.start:
        data = data[data.index >= pd.Timestamp(cfg.start, tz="UTC")]
    if cfg.end:
        data = data[data.index <= pd.Timestamp(cfg.end, tz="UTC")]
    if len(data) < 60:
        raise ValueError("回测数据不足（至少需要 60 根K线）")

    from strategies import get_strategy
    strategy = get_strategy(cfg.strategy_name)
    strategy.update_params(cfg.strategy_params)
    strategy.reset()

    # ---- 指标一次向量化预计算（O(n)），循环内 O(1) 取快照 ----
    # MA 周期跟随策略参数（dual_ma 的 fast_period/slow_period 此前被写死 10/30 忽略）
    ma_fast_p, ma_slow_p = strategy_ma_periods(strategy)
    closes = data["close"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    opens = data["open"].to_numpy(float)
    vols = data["volume"].to_numpy(float)
    series = precompute_indicator_series(closes, opens, highs, lows, vols,
                                         ma_fast_period=ma_fast_p, ma_slow_period=ma_slow_p)

    # 是否需要 S/R 与价格行为（有状态策略专用；与 fast_engine 同口径）
    need_sr = needs_sr(strategy.name, strategy)
    sr_series: Optional[list] = None
    pa_series: Optional[list] = None
    if need_sr:
        sr_series = support_resistance_series(highs, lows, closes, window=SR_WINDOW, min_touches=SR_MIN_TOUCHES)
        pa_series = price_action_features_series(highs, lows, closes, opens, vols)

    # 共享撮合内核（engine 不记录买入成交，保持原行为）
    result = run_matching_loop(
        data=data, cfg=cfg, strategy=strategy,
        closes=closes, opens=opens, highs=highs, lows=lows,
        series=series, need_sr=need_sr,
        sr_series=sr_series, pa_series=pa_series,
        volumes=vols,
        record_buy_trades=False,
    )
    equity_curve = result["equity_curve"]
    trades = result["trades"]
    cash = result["cash"]
    stats = {"traded_notional": result.get("traded_notional", 0.0),
             "total_fees": result.get("total_fees", 0.0)}

    # 末尾强制平仓（共享实现，FIFO 成本摊销）
    forced_liquidation(data=data, cfg=cfg, closes=closes,
                       cash=cash, position=result["position"], lots=result["lots"],
                       trades=trades, equity_curve=equity_curve, stats=stats)

    metrics = finalize_metrics(equity_curve, trades, cfg.timeframe, cfg.start_cash, closes,
                               traded_notional=stats["traded_notional"],
                               total_fees=stats["total_fees"],
                               bootstrap=bootstrap)
    return {"metrics": metrics, "equity_curve": equity_curve,
            "trades": trades, "symbol": cfg.symbol, "timeframe": cfg.timeframe,
            "strategy": cfg.strategy_name, "params": strategy.params,
            "benchmark": metrics["benchmark"]}
