"""回测绩效指标。"""
import math
from typing import Any

import numpy as np

PERIODS_PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}


def compute_metrics(equity: list[float], trades: list[dict], timeframe: str = "1h",
                    start_cash: float = 10000.0) -> dict[str, Any]:
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

    # 夏普
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    std = rets.std(ddof=1)
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

    return {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "payoff_ratio": round(payoff_ratio, 4),
        "total_trades": len(pnls),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "final_equity": round(float(eq[-1]), 4),
    }