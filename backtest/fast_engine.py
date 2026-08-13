"""高性能向量化回测引擎。

相比事件驱动引擎，本引擎：
1. 指标序列一次向量化预计算（O(n)），循环内 O(1) 取值 → 消除 O(n²) 指标重算
2. 支持 GPU（cuPy）加速指标批量计算（可选）
3. 支持进度回调 on_progress(i, n, ctx, trade) → 前端实时展示K线与成交盈亏
4. 输出与 run_backtest 完全兼容（metrics/equity_curve/trades）
5. 可选 Numba JIT 编译核心循环，大幅提升回测速度
"""
import logging
import time
from typing import Any, Callable, Optional

import pandas as pd

from backtest.metrics import compute_metrics
from indicators.technical import support_resistance, price_action_features
from indicators.vectorized import precompute_indicator_series, snapshot_at
from strategies.base import Signal, Strategy

log = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[int, int, dict, Optional[dict]], None]]


def run_backtest_fast(df: pd.DataFrame, cfg, on_progress: ProgressCB = None,
                      backend: str = "numpy", use_numba: bool = False) -> dict:
    """向量化快速回测，进度回调 on_progress(i, n, ctx, trade)。

    参数：
        use_numba: 是否启用 Numba JIT 编译（需已安装 numba）。
                   对 price_action 等需要逐K线计算 S/R 的策略效果显著。
    """
    if df.empty:
        raise ValueError("回测数据为空")
    data = df.copy()
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

    # ---- 向量化预计算指标序列（O(n)） ----
    closes = data["close"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    opens = data["open"].to_numpy(float)
    vols = data["volume"].to_numpy(float)
    n = len(closes)
    series = precompute_indicator_series(closes, opens, highs, lows, vols, backend=backend)

    # 预计算时间戳（毫秒）
    _ts = [int(t) for t in data.index.astype("int64") // 10**6]

    cash = cfg.start_cash
    position = 0.0
    # 持仓成本阵列（FIFO 逐笔记账）：分批建仓/部分平仓时 PnL 准确
    lots: list[tuple[float, float]] = []  # [(qty, price)]
    equity_curve: list[float] = []
    trades: list[dict] = []
    t0 = time.time()

    # 前视偏差防护：信号在第 i 根K线收盘后产生，必须在下一根（i+1）开盘成交
    pending_signal: Optional[Signal] = None

    # 判断是否需要逐K线计算 S/R（有状态策略；grid 不消费 sr/pa，跳过）
    need_sr = strategy.name in ("price_action",) or _is_ai_design(strategy)

    if use_numba:
        # Numba JIT 加速路径已禁用：其核心循环写死为双均线逻辑，与真实策略不符，
        # 对 price_action/grid 等策略会静默产出错误回测结果（曾发生），统一走标准路径。
        log.warning("[backtest] Numba 加速路径已禁用（核心循环与真实策略逻辑不符），使用标准向量化路径")

    # 标准路径（O(n) 指标预计算 + O(1) 快照）
    for i in range(n):
        ind = snapshot_at(series, i)
        # 关键位 + 价格行为（滑动窗口，仅用于有状态策略）
        if need_sr:
            a = max(0, i - 119)
            b = i + 1
            sr = support_resistance(highs[a:b], lows[a:b], closes[a:b],
                                    window=10, min_touches=2)
            seg = [list(x) for x in zip(_ts[a:b], opens[a:b], highs[a:b], lows[a:b],
                                        closes[a:b], vols[a:b])]
            pa = price_action_features(seg)
            ind["sr"] = sr
            ind["pa"] = pa

        price = float(closes[i])
        price_open = float(opens[i])

        # 1) 若上一根收盘产生了信号，用当前K线开盘成交
        trade: Optional[dict] = None
        if pending_signal is not None:
            sig = pending_signal
            pending_signal = None
            fill_price = price_open * (1 + cfg.slippage) if sig.side == "buy" else price_open * (1 - cfg.slippage)
            filled = False
            if sig.side == "buy" and cash > 0:
                qty = sig.qty if sig.qty else cash * sig.size_pct / fill_price
                qty = min(qty, cash / (fill_price * (1 + cfg.fee_rate)))
                if qty > 0:
                    cost = qty * fill_price
                    fee = cost * cfg.fee_rate
                    cash -= cost + fee
                    position += qty
                    lots.append((qty, fill_price))
                    filled = True
                    trade = {"ts": data.index[i].isoformat(), "symbol": cfg.symbol, "side": "buy",
                             "price": round(fill_price, 6), "qty": round(qty, 8),
                             "fee": round(fee, 6), "pnl": 0.0, "reason": sig.reason}
            elif sig.side == "sell" and position > 0:
                qty = sig.qty if sig.qty else position
                qty = min(qty, position)
                proceeds = qty * fill_price
                fee = proceeds * cfg.fee_rate
                # FIFO 摊销持仓成本（分批建仓的成本序列）
                remaining = qty
                cost_basis = 0.0
                while remaining > 1e-12 and lots:
                    lot_qty, lot_price = lots[0]
                    take = min(lot_qty, remaining)
                    cost_basis += take * lot_price
                    remaining -= take
                    if take >= lot_qty - 1e-12:
                        lots.pop(0)
                    else:
                        lots[0] = (lot_qty - take, lot_price)
                pnl = qty * fill_price - cost_basis - fee
                cash += proceeds - fee
                position -= qty
                trades.append({
                    "ts": data.index[i].isoformat(), "symbol": cfg.symbol, "side": "sell",
                    "price": round(fill_price, 6), "qty": round(qty, 8),
                    "fee": round(fee, 6), "pnl": round(pnl, 6), "reason": sig.reason,
                })
                filled = True
            # 只有实际成交才回调，避免"假成交"污染策略状态
            if filled:
                strategy.on_fill(cfg.symbol, sig.side, fill_price)

        # 2) 当前K线收盘后计算新信号
        ctx = {
            "symbol": cfg.symbol, "price": price, "position": position,
            "cash": cash, "indicators": ind, "timeframe": cfg.timeframe,
        }
        signal = strategy.on_candle(ctx)
        if signal:
            pending_signal = signal

        equity = cash + position * price
        equity_curve.append(round(equity, 4))

        if on_progress:
            try:
                on_progress(i, n, ctx, trade)
            except Exception as e:  # noqa: BLE001
                log.warning("[backtest] 进度回调异常: %s", e)

    # 末尾强制平仓（FIFO 成本摊销）
    if position > 0 and lots:
        fill_price = closes[-1] * (1 - cfg.slippage)
        qty = position
        proceeds = qty * fill_price
        fee = proceeds * cfg.fee_rate
        cost_basis = sum(lq * lp for lq, lp in lots)
        pnl = qty * fill_price - cost_basis - fee
        cash += proceeds - fee
        trades.append({
            "ts": data.index[-1].isoformat(), "symbol": cfg.symbol, "side": "sell",
            "price": round(fill_price, 6), "qty": round(qty, 8),
            "fee": round(fee, 6), "pnl": round(pnl, 6), "reason": "期末强制平仓",
        })
        if equity_curve:
            equity_curve[-1] = round(cash, 4)

    elapsed = time.time() - t0
    metrics = compute_metrics(equity_curve, trades, cfg.timeframe, cfg.start_cash)
    metrics["backtest_engine"] = "vectorized"
    metrics["elapsed_sec"] = round(elapsed, 4)
    return {"metrics": metrics, "equity_curve": equity_curve,
            "trades": trades, "symbol": cfg.symbol, "timeframe": cfg.timeframe,
            "strategy": cfg.strategy_name, "params": strategy.params}


def _is_ai_design(strategy) -> bool:
    try:
        from strategies import dynamic_names
        return strategy.name in dynamic_names()
    except Exception:  # noqa: BLE001
        return False
