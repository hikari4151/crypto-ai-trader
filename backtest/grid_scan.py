"""通用参数网格扫描（2026-09-23，F-④）。

任意策略参数笛卡尔积批量回测（fast_engine 线程安全、快），返回每组合
{params, metrics}——参数敏感性/热力图数据源。cost_scan 是费用专用扫描，
本模块是通用参数网格。
"""
import copy
import logging
from itertools import product
from typing import Optional

log = logging.getLogger("backtest.grid_scan")


def run_param_grid(df, cfg, grid: Optional[dict[str, list]] = None) -> list[dict]:
    """对 grid（{参数名: [候选值]}）的笛卡尔积逐组合回测。

    返回 [{params: dict, metrics: dict}]（按组合顺序）；grid 为空 → 单组基线。
    fast_engine.run_backtest_fast 线程安全（每次新建策略实例），可并行调用。
    """
    from backtest.fast_engine import run_backtest_fast

    keys = list((grid or {}).keys())
    combos = list(product(*(grid[k] for k in keys))) if keys else [()]
    out: list[dict] = []
    for combo in combos:
        params = {k: v for k, v in zip(keys, combo)}
        cfg2 = copy.copy(cfg)
        cfg2.strategy_params = {**(cfg.strategy_params or {}), **params}
        try:
            r = run_backtest_fast(df, cfg2)
            metrics = (r or {}).get("metrics", {})
        except Exception as e:  # noqa: BLE001
            log.warning("[grid] 参数组合 %s 回测失败: %s", params, e)
            metrics = {}
        out.append({"params": params, "metrics": metrics})
    return out
