"""共享撮合内核：双回测引擎（事件引擎 / 向量引擎）共用的逐K线撮合逻辑。

此前 backtest/engine.py 与 backtest/fast_engine.py 各维护一份几乎相同的
撮合循环（pending_signal → 限价单模拟 → FIFO 记账 → on_fill → 期末强制平仓），
任何撮合逻辑改动都要改两处、极易分叉。此处抽成单一事实来源，两个引擎只负责
"指标预计算方式 / 进度回调 / 是否记录买入成交"等外围差异。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from indicators.vectorized import snapshot_at
from strategies.base import Signal, Strategy

log = logging.getLogger(__name__)

# 限价单挂单最多尝试的K线数（超时取消）
_LIMIT_MAX_TRIES = 3


def _order_fee_rate(cfg, sig: Signal) -> float:
    """按订单类型选择费率：限价单（maker）用 maker_fee_rate，市价单用 taker_fee_rate。

    兼容旧配置：maker/taker 未显式设置时退回 cfg.fee_rate（单一费率口径）。
    """
    if getattr(sig, "order_type", None) == "limit":
        maker = getattr(cfg, "maker_fee_rate", None)
        if maker is not None:
            return float(maker)
        return float(cfg.fee_rate)
    taker = getattr(cfg, "taker_fee_rate", None)
    if taker is not None:
        return float(taker)
    return float(cfg.fee_rate)


def _limit_fill_ratio(cfg, sig: Signal, low: float, high: float) -> tuple[float, float]:
    """限价单模拟（partial/probabilistic 模式）：返回 (fill_price, fill_ratio)。

    未触及返回 fill_ratio=0（由调用方决定是否重挂）。
    """
    lp = float(sig.limit_price)
    if sig.side == "buy":
        if low <= lp <= high:
            fill_price = lp
            fill_ratio = float(high - lp) / (high - low + 1e-12) if high > low else 1.0
        else:
            fill_price = lp
            fill_ratio = 0.0
    else:
        if low <= lp <= high:
            fill_price = lp
            fill_ratio = float(lp - low) / (high - low + 1e-12) if high > low else 1.0
        else:
            fill_price = lp
            fill_ratio = 0.0
    return fill_price, fill_ratio


def _execute_fill(cfg, sig: Signal, fill_price: float, symbol: str,
                  ts: str, cash: float, position: float,
                  lots: list[tuple[float, float, int]],
                  trades: list[dict], strategy: Strategy,
                  record_buy_trade: bool, bar_volume: float = 0.0,
                  entry_i: int = -1, exit_i: int = -1,
                  stats: Optional[dict] = None) -> tuple[bool, float, float, Optional[dict]]:
    """执行一笔成交（买入/卖出）。返回 (filled, cash, position, trade_for_progress)。

    - filled: 是否实际成交（供 on_fill 回调与进度展示）
    - trade_for_progress: 进度回调用的成交 dict（fast_engine 记录买入成交；
      engine 不记录买入，仅返回卖出成交）
    - lots: (qty, price, entry_i) 三元组，entry_i 为建仓K线序号（持仓时长统计）
    - bar_volume: 当前K线成交量；cfg.participation_rate>0 时买入数量受
      volume×participation_rate 上限约束（流动性约束，避免全量成交失真）
    - exit_i: 当前K线序号（卖出时写入，供持仓时长统计）
    - stats: 可选统计累加器（P1-6）：stats["traded_notional"] 累计成交额、
      stats["total_fees"] 累计手续费。与 record_buy_trade 无关，保证 engine
      （不记录买单）也能统计换手与成本。
    """
    filled = False
    trade_for_progress: Optional[dict] = None
    fee_rate = _order_fee_rate(cfg, sig)
    if sig.side == "buy" and cash > 0:
        if sig.qty:
            qty = sig.qty
        else:
            qty = cash * sig.size_pct / fill_price
        # 流动性约束：成交量参与率（participation_rate=0 时不限制，保持旧行为）
        participation = getattr(cfg, "participation_rate", 0.0) or 0.0
        if participation > 0 and bar_volume > 0:
            qty = min(qty, bar_volume * participation)
        qty = min(qty, cash / (fill_price * (1 + fee_rate)))
        if qty > 0:
            cost = qty * fill_price
            fee = cost * fee_rate
            cash -= cost + fee
            position += qty
            lots.append((qty, fill_price, max(entry_i, 0)))
            if stats is not None:
                stats["traded_notional"] += qty * fill_price
                stats["total_fees"] += fee
            filled = True
            if record_buy_trade:
                trade_for_progress = {
                    "ts": ts, "symbol": symbol, "side": "buy",
                    "price": round(fill_price, 6), "qty": round(qty, 8),
                    "fee": round(fee, 6), "pnl": 0.0, "reason": sig.reason,
                    "entry_i": max(entry_i, 0),
                }
    elif sig.side == "sell" and position > 0:
        qty = sig.qty if sig.qty else position
        qty = min(qty, position)
        proceeds = qty * fill_price
        fee = proceeds * fee_rate
        # FIFO 摊销持仓成本（分批建仓的成本序列），同时摊销建仓K线序号
        remaining = qty
        cost_basis = 0.0
        first_entry_i = -1
        while remaining > 1e-12 and lots:
            lot_qty, lot_price, lot_entry_i = lots[0]
            take = min(lot_qty, remaining)
            cost_basis += take * lot_price
            if first_entry_i < 0:
                first_entry_i = lot_entry_i
            remaining -= take
            if take >= lot_qty - 1e-12:
                lots.pop(0)
            else:
                lots[0] = (lot_qty - take, lot_price, lot_entry_i)
        pnl = qty * fill_price - cost_basis - fee
        cash += proceeds - fee
        position -= qty
        if stats is not None:
            stats["traded_notional"] += qty * fill_price
            stats["total_fees"] += fee
        trade_for_progress = {
            "ts": ts, "symbol": symbol, "side": "sell",
            "price": round(fill_price, 6), "qty": round(qty, 8),
            "fee": round(fee, 6), "pnl": round(pnl, 6), "reason": sig.reason,
            "entry_i": first_entry_i if first_entry_i >= 0 else max(entry_i, 0),
            "exit_i": max(exit_i, 0),
        }
        trades.append(trade_for_progress)
        filled = True
    return filled, cash, position, trade_for_progress


def run_matching_loop(
    *,
    data: pd.DataFrame,
    cfg,
    strategy: Strategy,
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    series: dict[str, Any],
    need_sr: bool,
    sr_series: Optional[list] = None,
    pa_series: Optional[list] = None,
    volumes: Optional[np.ndarray] = None,
    record_buy_trades: bool = False,
    on_progress: Optional[Callable[[int, int, dict, Optional[dict]], None]] = None,
) -> dict:
    """逐K线撮合主循环（单一事实来源）。

    参数：
        volumes: 成交量序列（可选；参与率约束 cfg.participation_rate>0 时需要）
        record_buy_trades: fast_engine 为展示进度会记录买入成交 dict（仅用于
            on_progress，不参与 metrics 计算）；engine 不记录买入，保持原行为。
        on_progress: 进度回调 on_progress(i, n, ctx, trade)（fast_engine 使用）。
    返回 {cash, position, lots, equity_curve, trades, pending_state...} 供外层收尾。
    """
    n = len(closes)
    symbol = cfg.symbol
    cash = cfg.start_cash
    position = 0.0
    lots: list[tuple[float, float, int]] = []  # [(qty, price, entry_i)]
    equity_curve: list[float] = []
    trades: list[dict] = []
    # 前视偏差防护：信号在第 i 根K线收盘后产生，必须在下一根（i+1）开盘成交
    pending_signal: Optional[Signal] = None
    # 该挂单已尝试的K线数，与 pending_signal 同生命周期（实盘引擎的
    # _pending_signal=(sig, tries) 同构）。此前用 id(sig) 做旁路字典的键：
    # Signal 被回收后地址会被 Python 复用给新信号，新挂单于是继承旧单的
    # 超时次数、第一根未触及就被判超时取消，且字典只增不减。
    pending_tries = 0
    # P0-1：intrabar 止损/止盈触价建模（策略声明 protective_levels 时生效）
    intrabar_stops = bool(getattr(cfg, "intrabar_stops", False))
    # P0-3：每根K线按持仓价值收取的资金费率（永续合约，默认 0 关闭）
    funding_rate = float(getattr(cfg, "funding_rate", 0.0) or 0.0)
    # P1-6 换手/成本统计（_execute_fill 内部无条件累计，与 record_buy_trade 无关）
    stats: dict = {"traded_notional": 0.0, "total_fees": 0.0}

    def _emit_intrabar_exit(ctx: dict, trigger_price: float, reason: str,
                            bar_volume: float) -> Optional[dict]:
        """盘中触价卖出（long 持仓的止损/止盈），返回 trade dict 或 None。"""
        nonlocal cash, position
        fill_price = float(trigger_price) * (1 - cfg.slippage)
        sig = Signal(symbol, "sell", qty=position, strategy=strategy.name, reason=reason)
        filled, cash, position, trade = _execute_fill(
            cfg, sig, fill_price, symbol, data.index[i].isoformat(),
            cash, position, lots, trades, strategy,
            record_buy_trade=record_buy_trades, bar_volume=bar_volume,
            entry_i=i, exit_i=i, stats=stats)
        if filled:
            strategy.on_fill(symbol, "sell", fill_price)
            return trade
        return None

    for i in range(n):
        price_open, price_close = float(opens[i]), float(closes[i])
        bar_volume = float(volumes[i]) if volumes is not None else 0.0
        ind = snapshot_at(series, i)
        if need_sr:
            ind["sr"] = sr_series[i]  # type: ignore[index]
            ind["pa"] = pa_series[i]  # type: ignore[index]

        # 1) 若上一根收盘产生了信号，用当前K线开盘价成交（叠加滑点）
        trade_for_progress: Optional[dict] = None
        if pending_signal is not None:
            sig = pending_signal
            tries = pending_tries
            pending_signal = None
            pending_tries = 0
            fill_price = price_open * (1 + cfg.slippage) if sig.side == "buy" else price_open * (1 - cfg.slippage)
            skip_fill = False
            # ---- 限价单模拟（partial/probabilistic 模式） ----
            if (cfg.limit_order_model in ("partial", "probabilistic")
                    and getattr(sig, "order_type", None) == "limit"
                    and getattr(sig, "limit_price", None) is not None):
                fill_price, fill_ratio = _limit_fill_ratio(cfg, sig, float(lows[i]), float(highs[i]))
                if fill_ratio <= 0:
                    # 未触及：挂单最多 3 根K线（超时取消），本根不成交但仍正常推进
                    tries += 1
                    if tries < _LIMIT_MAX_TRIES:
                        pending_signal = sig  # 重新放回，下一根重试
                        pending_tries = tries
                    skip_fill = True
                else:
                    # 触及成交：按 fill_ratio 部分成交
                    qty_total = sig.qty if sig.qty else (cash * sig.size_pct / fill_price)
                    sig.qty = qty_total * fill_ratio
            # ---- 限价单结束 ----
            if not skip_fill:
                filled, cash, position, trade_for_progress = _execute_fill(
                    cfg, sig, fill_price, symbol, data.index[i].isoformat(),
                    cash, position, lots, trades, strategy,
                    record_buy_trade=record_buy_trades, bar_volume=bar_volume,
                    entry_i=i, exit_i=i, stats=stats)
                # 只有实际成交才回调，避免"假成交"污染策略状态
                if filled:
                    strategy.on_fill(symbol, sig.side, fill_price)

        # 1.5) P0-1 intrabar 止损/止盈触价：持仓中且策略声明保护位时，
        #      用当前K线 high/low 判断是否盘中触及（收盘价判断会漏掉盘中触发）。
        if intrabar_stops and position > 0 and not pending_signal:
            ctx_now = {
                "symbol": symbol, "price": price_close, "position": position,
                "cash": cash, "indicators": ind, "timeframe": cfg.timeframe,
                "high": float(highs[i]), "low": float(lows[i]),
            }
            try:
                levels = strategy.protective_levels(ctx_now)
            except Exception:  # noqa: BLE001
                levels = None
            if levels:
                stop = levels.get("stop")
                take_profit = levels.get("take_profit")
                # 保守次序：若同根K线既触及止损又触及止盈，先按止损处理（悲观）
                if stop is not None and float(lows[i]) <= float(stop):
                    trade_for_progress = _emit_intrabar_exit(ctx_now, float(stop), "盘中触价止损", bar_volume)
                elif take_profit is not None and float(highs[i]) >= float(take_profit):
                    trade_for_progress = _emit_intrabar_exit(ctx_now, float(take_profit), "盘中触价止盈", bar_volume)

        # 2) 当前K线收盘后计算新信号（指标已由 snapshot_at 取预计算快照，O(1)）
        ctx = {
            "symbol": symbol, "price": price_close, "position": position,
            "cash": cash, "indicators": ind, "timeframe": cfg.timeframe,
        }
        signal = strategy.on_candle(ctx)
        if signal:
            # 新信号替换仍在挂的旧限价单，超时计数一并归零
            pending_signal = signal
            pending_tries = 0

        # 3) P0-3 资金费率：每根K线按持仓价值收取（永续合约成本，默认 0 关闭）
        if funding_rate and position > 0:
            fund = position * price_close * funding_rate
            cash -= fund
            stats["total_fees"] += fund

        equity = cash + position * price_close
        equity_curve.append(round(equity, 4))

        if on_progress:
            try:
                on_progress(i, n, ctx, trade_for_progress)
            except Exception as e:  # noqa: BLE001
                log.warning("[backtest] 进度回调异常: %s", e)

    return {
        "cash": cash,
        "position": position,
        "lots": lots,
        "equity_curve": equity_curve,
        "trades": trades,
        "traded_notional": round(stats["traded_notional"], 6),
        "total_fees": round(stats["total_fees"], 6),
    }


def forced_liquidation(*, data: pd.DataFrame, cfg, closes: np.ndarray,
                       cash: float, position: float, lots: list[tuple[float, float, int]],
                       trades: list[dict], equity_curve: list[float],
                       stats: Optional[dict] = None) -> None:
    """期末强制平仓（FIFO 成本摊销）。修改 cash/position/trades/equity_curve 就地。"""
    if position > 0 and lots:
        fill_price = float(closes[-1]) * (1 - cfg.slippage)
        qty = position
        proceeds = qty * fill_price
        fee = proceeds * _order_fee_rate(cfg, Signal("", "sell"))
        cost_basis = sum(lq * lp for lq, lp, _li in lots)
        first_entry_i = min((li for _lq, _lp, li in lots), default=-1)
        pnl = qty * fill_price - cost_basis - fee
        cash += proceeds - fee
        if stats is not None:
            stats["traded_notional"] += qty * fill_price
            stats["total_fees"] += fee
        trades.append({
            "ts": data.index[-1].isoformat(), "symbol": cfg.symbol, "side": "sell",
            "price": round(fill_price, 6), "qty": round(qty, 8),
            "fee": round(fee, 6), "pnl": round(pnl, 6), "reason": "期末强制平仓",
            "entry_i": first_entry_i if first_entry_i >= 0 else len(closes) - 1,
            "exit_i": len(closes) - 1,
        })
        if equity_curve:
            equity_curve[-1] = round(cash, 4)
    return cash


def needs_sr(name: str, strategy) -> bool:
    """判断策略是否需要 S/R 与价格行为指标。

    检查策略的 executor 类型（动态策略）或名称（内置策略）是否为 price_action。
    AI 设计策略的 executor 可能是 price_action，但策略名是动态名。
    元控制器还要看它托管的子策略池：price_action 缺 sr/pa 时突破/回调判据
    恒不成立，作为子策略会静默产出零信号。
    """
    executor = getattr(strategy, "executor", name)
    if executor == "price_action":
        return True
    return "price_action" in (getattr(strategy, "managed_executors", None) or [])
