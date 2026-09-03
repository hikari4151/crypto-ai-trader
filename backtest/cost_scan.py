"""成本压力测试：同一策略在费用/滑点放大倍数下扫描，量化成本对收益的侵蚀。

P1-4 目标：回测收益对交易成本（手续费 + 滑点）的敏感性评估。把
fee_rate 与 slippage 放大到 1x/2x/3x/5x 组合跑同一回测，对比
总收益/夏普/回撤/交易数/成本拖累。若收益随成本快速恶化，说明策略
过度依赖高换手（成本敏感）；若几乎不变，说明策略对成本不敏感。

数据源/参数与普通回测完全一致（共享撮合内核），仅成本参数不同。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .engine import BacktestConfig

# 默认扫描组合：(fee 倍数, slippage 倍数)。含基准 1x1 与极端 5x5。
DEFAULT_FEE_MULT = (1.0, 2.0, 3.0, 5.0)
DEFAULT_SLIP_MULT = (1.0, 2.0, 5.0)


@dataclass
class CostScanConfig:
    """成本压力扫描配置。"""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy_name: str = "dual_ma"
    strategy_params: dict = field(default_factory=dict)
    start_cash: float = 10000.0
    base_fee_rate: float = 0.001
    base_slippage: float = 0.0005
    fee_mults: tuple[float, ...] = DEFAULT_FEE_MULT
    slip_mults: tuple[float, ...] = DEFAULT_SLIP_MULT
    run_backtest: Optional[Callable] = None  # 可注入回测函数（缺省用 run_backtest_fast）


def run_cost_scan(df: pd.DataFrame, cfg: CostScanConfig,
                  on_progress: Optional[Callable[[str], None]] = None) -> dict:
    """对每个 (fee×, slip×) 组合跑一次回测，返回逐组合指标表。

    返回结构：
    {
        "scan": [{ "fee_mult": x, "slip_mult": y, "fee_rate": .., "slippage": ..,
                   "metrics": {...} }, ...],
        "base": { "fee_rate": .., "slippage": .., "metrics": {...} },
        "sensitivity": { "fee_sensitive_ret_pct": .., "max_ret_drop_pct": .. },
    }
    sensitivity: fee 5x + slip 5x 相对基准的总收益跌幅（百分比）。
    """
    if df.empty:
        raise ValueError("扫描数据为空")

    run_fn = cfg.run_backtest
    if run_fn is None:
        from .fast_engine import run_backtest_fast

        def run_fn(data: pd.DataFrame, bcfg) -> dict:
            return run_backtest_fast(data, bcfg, bootstrap=False)  # P2-13 成本扫描只消费基础指标

    base_metrics = None
    rows: list[dict] = []
    fee_mults, slip_mults = cfg.fee_mults, cfg.slip_mults
    total = len(fee_mults) * len(slip_mults)
    idx = 0
    for fm in fee_mults:
        for sm in slip_mults:
            idx += 1
            if on_progress:
                on_progress(f"[cost-scan] {idx}/{total}: fee x{fm:.1f} slippage x{sm:.1f}")
            bcfg = BacktestConfig(
                symbol=cfg.symbol, timeframe=cfg.timeframe,
                strategy_name=cfg.strategy_name, strategy_params=dict(cfg.strategy_params),
                start_cash=cfg.start_cash,
                fee_rate=cfg.base_fee_rate * fm,
                taker_fee_rate=cfg.base_fee_rate * fm,
                slippage=cfg.base_slippage * sm,
            )
            try:
                res = run_fn(df, bcfg)
            except Exception as e:  # noqa: BLE001
                res = {"metrics": {"total_return": float("nan"),
                                   "sharpe": float("nan"),
                                   "max_drawdown": float("nan"),
                                   "total_trades": 0,
                                   "total_costs": 0.0,
                                   "error": str(e)}}
            m = res.get("metrics", {}) if isinstance(res, dict) else {}
            row = {
                "fee_mult": fm, "slip_mult": sm,
                "fee_rate": round(cfg.base_fee_rate * fm, 6),
                "slippage": round(cfg.base_slippage * sm, 6),
                "total_return": m.get("total_return"),
                "annual_return": m.get("annual_return"),
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
                "total_trades": m.get("total_trades"),
                "total_costs": m.get("total_costs"),
                "turnover": m.get("turnover"),
                "error": m.get("error"),
            }
            rows.append(row)
            if fm == 1.0 and sm == 1.0:
                base_metrics = m

    # 敏感性：基准 vs 最极端组合（fee 5x + slip 5x）
    worst = next((r for r in rows if r["fee_mult"] >= 5 and r["slip_mult"] >= 5), None)
    sensitivity = {}
    if base_metrics:
        base_ret = float(base_metrics.get("total_return", 0.0) or 0.0)
        sensitivity["base_ret"] = round(base_ret, 6)
        if worst and worst.get("total_return") is not None:
            worst_ret = float(worst["total_return"] or 0.0)
            sensitivity["worst_ret"] = round(worst_ret, 6)
            # 收益跌幅百分比（相对基准收益绝对值）
            if abs(base_ret) > 1e-9:
                drop = (base_ret - worst_ret) / abs(base_ret)
                sensitivity["worst_ret_drop_pct"] = round(drop * 100.0, 2)
            else:
                sensitivity["worst_ret_drop_pct"] = 0.0
    return {
        "scan": rows,
        "base": {"fee_rate": cfg.base_fee_rate, "slippage": cfg.base_slippage,
                 "metrics": base_metrics},
        "sensitivity": sensitivity,
    }