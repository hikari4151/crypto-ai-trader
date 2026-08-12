"""事件类型定义（事件驱动架构的核心数据契约）。"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    MARKET_TICKER = "market.ticker"
    MARKET_CANDLE = "market.candle"
    SIGNAL = "strategy.signal"
    ORDER_FILL = "order.fill"
    TRADE = "trade"
    RISK_BLOCKED = "risk.blocked"
    AI_ANALYSIS = "ai.analysis"
    AI_OPTIMIZED = "ai.optimized"
    AI_REVIEW = "ai.review"
    PORTFOLIO_UPDATE = "portfolio.update"
    ENGINE_STATE = "engine.state"
    SYSTEM = "system"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now())
    source: str = "system"