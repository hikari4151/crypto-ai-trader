"""回测结果 Schema 版本化（2026-09-24，E-②）。

定义回测指标/成交/结果的 pydantic 模型 + schema_version，防止字段漂移、
脏数据（如 start_cash≤0 的 NaN/inf）直接 500；API/DB 加载可统一校验。
版本历史：
  1- 初版指标集
  2- 2026-09-23 F-①：+calmar_ratio/sortino
  3- 2026-09-23 C-①：+avg_slippage / trades.slippage
"""
from typing import Any, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 3


class BacktestMetrics(BaseModel):
    """回测指标快照（必填核心 + 可选扩展；JSON 安全：无 NaN/inf）。"""
    model_config = {"extra": "allow", "allow_inf_nan": False}  # 防 NaN/inf 直接 500

    total_return: float
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    calmar_ratio: float = 0.0
    sortino: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_costs: float = 0.0
    avg_holding_bars: float = 0.0
    turnover: float = 0.0
    cost_drag: float = 0.0
    avg_slippage: float = 0.0
    schema_version: int = SCHEMA_VERSION


class BacktestTrade(BaseModel):
    """单笔成交（可扩展；slippage 为 C-① 不利滑点，未记录为 None→缺省）。"""
    model_config = {"extra": "allow"}

    ts: str
    symbol: str
    side: str
    price: float
    qty: float = 0.0
    fee: float = 0.0
    pnl: Optional[float] = None
    reason: str = ""
    slippage: Optional[float] = None


class BacktestResult(BaseModel):
    """回测结果整体（metrics + trades + equity + 元信息）。"""
    model_config = {"extra": "allow"}

    symbol: str
    timeframe: str
    strategy: str = ""
    metrics: BacktestMetrics
    trades: list[BacktestTrade] = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


def validate_backtest_result(data: dict) -> BacktestResult:
    """校验回测结果（指标/成交/equity）。字段缺失/类型错 → ValidationError。"""
    return BacktestResult.model_validate(data)