"""双均线示例策略：快线上穿慢线买入、下穿卖出，附带止损/止盈。"""
from typing import Any, Optional

from .base import Signal, Strategy


class DualMAStrategy(Strategy):
    name = "dual_ma"
    description = "双均线交叉策略（快线上穿慢线买入 / 下穿卖出，含止损止盈）"
    default_params = {
        "fast_period": 10,
        "slow_period": 30,
        "size_pct": 0.5,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
    }
    param_schema = {
        "fast_period": {"type": "int", "min": 2, "max": 120, "label": "快线周期"},
        "slow_period": {"type": "int", "min": 5, "max": 300, "label": "慢线周期"},
        "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
        "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "止损比例"},
        "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.5, "label": "止盈比例"},
    }

    def reset(self) -> None:
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None
        self._entry_price: Optional[float] = None

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        p = self.params
        ind = ctx["indicators"]
        fast, slow = ind.get("ma_fast"), ind.get("ma_slow")
        price = ctx["price"]
        position = ctx.get("position", 0.0)
        if fast is None or slow is None or fast == 0 or slow == 0:
            return None
        symbol = ctx["symbol"]
        sig: Optional[Signal] = None

        if self._prev_fast is not None:
            cross_up = self._prev_fast <= self._prev_slow and fast > slow
            cross_down = self._prev_fast >= self._prev_slow and fast < slow
            if cross_up and position <= 0:
                sig = Signal(symbol, "buy", p["size_pct"], strategy=self.name,
                             reason=f"快线上穿慢线({p['fast_period']}/{p['slow_period']})")
            elif cross_down and position > 0:
                sig = Signal(symbol, "sell", 1.0, strategy=self.name, reason="快线下穿慢线")

        self._prev_fast, self._prev_slow = fast, slow

        # 止损 / 止盈（持仓中且没有交叉信号时）
        if sig is None and position > 0 and self._entry_price:
            chg = (price - self._entry_price) / self._entry_price
            if chg <= -p["stop_loss_pct"]:
                sig = Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止损 {chg:.2%}")
            elif chg >= p["take_profit_pct"]:
                sig = Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止盈 {chg:.2%}")
        return sig

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        if side == "buy":
            self._entry_price = price
        else:
            self._entry_price = None