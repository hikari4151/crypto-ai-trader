"""高性能向量化回测引擎。

相比事件驱动引擎，本引擎：
1. 指标序列一次向量化预计算（O(n)），循环内 O(1) 取值 → 消除 O(n²) 指标重算
2. 支持 GPU（cuPy）加速指标批量计算（可选）
3. 支持进度回调 on_progress(i, n, ctx, trade) → 前端实时展示K线与成交盈亏
4. 输出与 run_backtest 完全兼容（metrics/equity_curve/trades）

撮合循环（pending_signal / 限价单 / FIFO 记账 / 期末平仓）已抽到
backtest/_matching.py 共享内核，本引擎与 engine.py 共用同一实现。
"""
import logging
import time
from typing import Callable, Optional

import pandas as pd

from indicators.technical import SR_MIN_TOUCHES, SR_WINDOW, support_resistance_series, price_action_features_series
from indicators.vectorized import precompute_indicator_series
from strategies.base import strategy_ma_periods

from ._matching import forced_liquidation, needs_sr, run_matching_loop
from .engine import research_scope
from .metrics import finalize_metrics

log = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[int, int, dict, Optional[dict]], None]]


@research_scope
def prepare_backtest_fast(df: pd.DataFrame, cfg, backend: str = "numpy") -> dict:
    """批量扫描预计算（run11 E1）：清洗 + 指标预计算一次，供多组合复用。

    与 run_backtest_fast 内部 prepare 段完全同式（同一函数、同一清洗口径、
    同一 MA 周期推导）——产出可直接喂 run_backtest_fast(_prepared=...)。
    成本/网格/组合扫描对同一 df、同一策略参数连续跑多次仅成本参数不同，
    复用后省每次的清洗+预计算（~6.6ms/次，12 组合扫描总量 -32%）。
    """
    if df.empty:
        raise ValueError("回测数据为空")
    from .data_loader import OHLCVSanitizer
    data = df.copy()
    sanitizer = OHLCVSanitizer()
    data = sanitizer.clean(data, mode=getattr(cfg, "cleaning_mode", "mark"))
    if getattr(cfg, "start", None):
        data = data[data.index >= pd.Timestamp(cfg.start, tz="UTC")]
    if getattr(cfg, "end", None):
        data = data[data.index <= pd.Timestamp(cfg.end, tz="UTC")]
    if len(data) < 60:
        raise ValueError("回测数据不足（至少需要 60 根K线）")

    from strategies import get_strategy
    strategy = get_strategy(cfg.strategy_name)
    strategy.update_params(cfg.strategy_params)
    strategy.reset()
    ma_fast_p, ma_slow_p = strategy_ma_periods(strategy)
    closes = data["close"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    opens = data["open"].to_numpy(float)
    vols = data["volume"].to_numpy(float)
    series = precompute_indicator_series(closes, opens, highs, lows, vols, backend=backend,
                                         ma_fast_period=ma_fast_p, ma_slow_period=ma_slow_p)
    need_sr = needs_sr(strategy.name, strategy)
    sr_series = None
    pa_series = None
    if need_sr:
        sr_series = support_resistance_series(highs, lows, closes, window=SR_WINDOW, min_touches=SR_MIN_TOUCHES)
        pa_series = price_action_features_series(highs, lows, closes, opens, vols)
    return {"data": data, "series": series, "need_sr": need_sr,
            "sr_series": sr_series, "pa_series": pa_series}


@research_scope
def run_backtest_fast(df: pd.DataFrame, cfg, on_progress: ProgressCB = None,
                      backend: str = "numpy", use_numba: bool = False,
                      bootstrap: bool = True, _prepared: Optional[dict] = None,
                      _external_clean: Optional[pd.DataFrame] = None) -> dict:
    """向量化快速回测，进度回调 on_progress(i, n, ctx, trade)。

    参数：
        use_numba: 兼容参数。Numba JIT 加速路径已禁用（其核心循环写死为双均线
            逻辑，与真实策略不符），统一走共享撮合内核。
        bootstrap: 是否计算 block bootstrap 置信区间（P2-13）。网格/成本/
            组合扫描只消费基础指标，传 False 跳过 1000 次重采样，省约 50ms/回测。
        _prepared: 内部参数（run11 E1）。批量扫描（成本/网格/组合）对同一 df、
            同一策略参数连续跑多次，仅成本参数不同——清洗与指标预计算与成本
            无关，可 prepare 一次循环复用（省单次 ~6.6ms，扫描总量 -32%）。
            结构 = {data, series, sr_series, pa_series, need_sr}，由
            prepare_backtest_fast() 生成；指标输出与逐次独立调用逐位一致
            （.optim/probe_costscan_prep.py：12 组合 metrics 全等验证 PASS）。
        _external_clean: 内部参数（清洗复用）。调用方已用 OHLCVSanitizer().clean()
            清洗过同一 df（如 CSCV 多候选共用一段数据时只清洗一次），传清洗结果
            跳过内部重复清洗。clean 幂等（probe_sanitize_idem PASS），输出与
            独立调用逐位一致；仅当 _prepared 为 None 时生效。
    """
    if df.empty:
        raise ValueError("回测数据为空")
    if _prepared is not None:
        data = _prepared["data"]
        series = _prepared["series"]
        need_sr = _prepared["need_sr"]
        sr_series = _prepared.get("sr_series")
        pa_series = _prepared.get("pa_series")
        # _prepared 路径需要自己的策略实例（prepare 阶段与撮合阶段分离，
        # 实例化时机不同——保持与独立调用一致的"实例化→reset→撮合"顺序）
        from strategies import get_strategy
        strategy = get_strategy(cfg.strategy_name)
        strategy.update_params(cfg.strategy_params)
        strategy.reset()
    else:
        data = df.copy()
        # 数据清洗：异常值检测 + 缺失值填充，避免脏数据污染回测/参数优化。
        # 与参数无关且在调用方已清洗时可复用（_external_clean，幂等验证 PASS）。
        if _external_clean is not None:
            data = _external_clean.copy()
        else:
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

        # ---- 向量化预计算指标序列（O(n)）；MA 周期跟随策略参数（dual_ma 的
        # fast_period/slow_period 此前被写死 10/30 忽略，已修复） ----
        ma_fast_p, ma_slow_p = strategy_ma_periods(strategy)
        closes = data["close"].to_numpy(float)
        highs = data["high"].to_numpy(float)
        lows = data["low"].to_numpy(float)
        opens = data["open"].to_numpy(float)
        vols = data["volume"].to_numpy(float)
        series = precompute_indicator_series(closes, opens, highs, lows, vols, backend=backend,
                                             ma_fast_period=ma_fast_p, ma_slow_period=ma_slow_p)

        # 判断是否需要逐K线计算 S/R（有状态策略；grid 不消费 sr/pa，跳过）
        need_sr = needs_sr(strategy.name, strategy)
        sr_series = None
        pa_series = None
        if need_sr:
            sr_series = support_resistance_series(highs, lows, closes, window=SR_WINDOW, min_touches=SR_MIN_TOUCHES)
            pa_series = price_action_features_series(highs, lows, closes, opens, vols)

    t0 = time.time()
    closes = data["close"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    opens = data["open"].to_numpy(float)
    vols = data["volume"].to_numpy(float)

    if use_numba:
        # Numba JIT 加速路径已禁用：其核心循环写死为双均线逻辑，与真实策略不符，
        # 对 price_action/grid 等策略会静默产出错误回测结果（曾发生），统一走标准路径。
        log.warning("[backtest] Numba 加速路径已禁用（核心循环与真实策略逻辑不符），使用标准向量化路径")

    # 共享撮合内核（fast_engine 记录买入成交供进度展示，不参与 metrics）
    result = run_matching_loop(
        data=data, cfg=cfg, strategy=strategy,
        closes=closes, opens=opens, highs=highs, lows=lows,
        series=series, need_sr=need_sr,
        sr_series=sr_series, pa_series=pa_series,
        volumes=vols,
        record_buy_trades=True,
        on_progress=on_progress,
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

    elapsed = time.time() - t0
    metrics = finalize_metrics(equity_curve, trades, cfg.timeframe, cfg.start_cash, closes,
                               engine_tag="vectorized", elapsed_sec=elapsed,
                               traded_notional=stats["traded_notional"],
                               total_fees=stats["total_fees"],
                               bootstrap=bootstrap)
    return {"metrics": metrics, "equity_curve": equity_curve,
            "trades": trades, "symbol": cfg.symbol, "timeframe": cfg.timeframe,
            "strategy": cfg.strategy_name, "params": strategy.params,
            "benchmark": metrics["benchmark"]}
