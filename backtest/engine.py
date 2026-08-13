"""事件驱动回测引擎：逐K线仿真撮合，计入手续费与滑点。"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from indicators.technical import compute_latest
from strategies.base import Signal, Strategy

log = logging.getLogger(__name__)


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


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """执行回测，返回 {metrics, equity_curve, trades}。"""
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

    cash = cfg.start_cash
    position = 0.0
    # 持仓成本阵列（FIFO 逐笔记账）：分批建仓/部分平仓时 PnL 准确——
    # 单点 entry_price 会被最后一次成交价覆盖、部分平仓后丢失成本，导致 pnl 失真
    lots: list[tuple[float, float]] = []  # [(qty, price)]
    equity_curve: list[float] = []
    trades: list[dict] = []
    # 前视偏差防护：信号在第 i 根K线收盘后产生，必须在下一根（i+1）开盘成交
    pending_signal: Optional[Signal] = None

    for i in range(len(data)):
        candle = data.iloc[i]
        price_open, price_close = float(candle["open"]), float(candle["close"])

        # 1) 若上一根收盘产生了信号，用当前K线开盘价成交（叠加滑点）
        if pending_signal is not None:
            sig = pending_signal
            pending_signal = None
            fill_price = price_open * (1 + cfg.slippage) if sig.side == "buy" else price_open * (1 - cfg.slippage)
            filled = False
            fee = 0.0
            if sig.side == "buy" and cash > 0:
                if sig.qty:
                    qty = sig.qty
                else:
                    qty = cash * sig.size_pct / fill_price
                qty = min(qty, cash / (fill_price * (1 + cfg.fee_rate)))
                if qty > 0:
                    cost = qty * fill_price
                    fee = cost * cfg.fee_rate
                    cash -= cost + fee
                    position += qty
                    lots.append((qty, fill_price))
                    filled = True
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

        # 2) 用截至当前的收盘价序列计算指标（不含未来数据）。
        # 只取最近 120 根窗口：与 fast_engine 的 S/R 滑动窗口口径一致，
        # 曾用全量历史使旧枢轴簇干扰 near_sup/near_res/broken_resistance，两引擎交易分叉
        hist = data.iloc[max(0, i - 119): i + 1]
        ohlcv = [[int(hist.index[j].timestamp() * 1000), float(hist.iloc[j]["open"]),
                  float(hist.iloc[j]["high"]), float(hist.iloc[j]["low"]),
                  float(hist.iloc[j]["close"]), float(hist.iloc[j]["volume"])] for j in range(len(hist))]
        ind = compute_latest(ohlcv)
        ctx = {
            "symbol": cfg.symbol, "price": price_close, "position": position,
            "cash": cash, "indicators": ind, "timeframe": cfg.timeframe,
        }
        signal = strategy.on_candle(ctx)
        if signal:
            pending_signal = signal

        equity = cash + position * price_close
        equity_curve.append(round(equity, 4))

    # 末尾强制平仓（与 fast_engine 一致，FIFO 成本摊销）：
    # 曾缺失此段导致期末持仓按市值计但无平仓交易，两引擎 trades/胜率/盈亏比不一致
    last_close = float(data.iloc[-1]["close"])
    if position > 0 and lots:
        fill_price = last_close * (1 - cfg.slippage)
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

    from .metrics import compute_metrics
    metrics = compute_metrics(equity_curve, trades, cfg.timeframe, cfg.start_cash)
    return {"metrics": metrics, "equity_curve": equity_curve,
            "trades": trades, "symbol": cfg.symbol, "timeframe": cfg.timeframe,
            "strategy": cfg.strategy_name, "params": strategy.params}