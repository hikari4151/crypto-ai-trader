"""简单网格示例策略：相对基准价下跌 x% 买一档、上涨 x% 卖一档。"""
from typing import Any, Optional

from .base import Signal, Strategy


class GridStrategy(Strategy):
    name = "grid"
    description = "网格策略：价格较基准下跌 grid_pct% 买入一档，上涨 grid_pct% 卖出一档"
    default_params = {
        "grid_pct": 0.02,
        "qty_per_grid": 0.001,
        "max_positions": 20,
    }
    param_schema = {
        "grid_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "网格间距"},
        "qty_per_grid": {"type": "float", "min": 0.0001, "max": 100, "label": "每档数量"},
        "max_positions": {"type": "int", "min": 1, "max": 200, "label": "最大档位数"},
    }

    def reset(self) -> None:
        self._base_price: Optional[float] = None

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        p = self.params
        price = ctx["price"]
        symbol = ctx["symbol"]
        position = ctx.get("position", 0.0)
        if self._base_price is None:
            self._base_price = price
            return None
        max_qty = p["max_positions"] * p["qty_per_grid"]
        if price <= self._base_price * (1 - p["grid_pct"]) and position < max_qty:
            self._base_price = price
            return Signal(symbol, "buy", qty=p["qty_per_grid"], strategy=self.name,
                          reason=f"网格买入：价格下跌 {p['grid_pct']:.1%}")
        if price >= self._base_price * (1 + p["grid_pct"]) and position >= p["qty_per_grid"]:
            self._base_price = price
            return Signal(symbol, "sell", qty=p["qty_per_grid"], strategy=self.name,
                          reason=f"网格卖出：价格上涨 {p['grid_pct']:.1%}")
        return None