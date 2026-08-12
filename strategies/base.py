"""可插拔策略基类。用户自定义策略继承 Strategy 并实现 on_candle 即可。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Signal:
    symbol: str
    side: str                       # buy / sell
    size_pct: float = 0.5
    qty: Optional[float] = None
    order_type: str = "market"
    limit_price: Optional[float] = None
    reason: str = ""
    strategy: str = ""


class Strategy(ABC):
    name: str = "base"
    description: str = ""
    param_schema: dict[str, dict[str, Any]] = {}
    default_params: dict[str, Any] = {}

    def __init__(self) -> None:
        self.params: dict[str, Any] = dict(self.default_params)

    def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """热更新参数（含 AI 优化结果），按 schema 校验类型与范围，返回实际生效参数。"""
        schema = self.param_schema
        for k, v in params.items():
            if k not in schema:
                continue
            spec = schema[k]
            try:
                t = spec.get("type")
                if t == "int":
                    v = int(round(float(v)))
                elif t == "float":
                    v = float(v)
                elif t == "bool":
                    v = bool(v)
                elif t == "str":
                    choices = spec.get("choices") or []
                    if choices and str(v) not in choices:
                        continue
                    v = str(v)
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                self.params[k] = v
            except (TypeError, ValueError):
                continue
        return dict(self.params)

    def reset(self) -> None:
        """回测/重启时重置内部状态。"""
        pass

    @abstractmethod
    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        """处理一根新K线，返回信号（可空）。"""

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        """成交回调（用于记录入场价等状态）。"""