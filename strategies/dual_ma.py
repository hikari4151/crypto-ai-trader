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
        # P1-5 风险预算仓位：>0 时按"单笔风险预算 / 止损距离"反推仓位，
        # 替代固定 size_pct（0=关闭，用固定 size_pct，保持旧行为）
        "risk_per_trade_pct": 0.0,
        # P1-5 ATR 止损距离：>0 时止损距离用 ATR×atr_stop_mult 而非 stop_loss_pct
        "atr_stop_mult": 0.0,
    }
    param_schema = {
        "fast_period": {"type": "int", "min": 2, "max": 120, "label": "快线周期"},
        "slow_period": {"type": "int", "min": 5, "max": 300, "label": "慢线周期"},
        "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
        "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "止损比例"},
        "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.5, "label": "止盈比例"},
        "risk_per_trade_pct": {"type": "float", "min": 0.0, "max": 0.1, "label": "单笔风险预算%"},
        "atr_stop_mult": {"type": "float", "min": 0.0, "max": 10.0, "label": "ATR止损倍数"},
    }

    def reset(self) -> None:
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None
        self._entry_price: Optional[float] = None

    def _stop_dist_pct(self, ctx: dict[str, Any]) -> float:
        """止损距离（占入场价的比例）：atr_stop_mult>0 时用 ATR，否则用 stop_loss_pct。"""
        p = self.params
        if float(p.get("atr_stop_mult", 0.0) or 0.0) > 0:
            atr_pct = (ctx.get("indicators") or {}).get("atr_pct", 0.0) or 0.0
            mult = float(p["atr_stop_mult"])
            dist = atr_pct * mult / 100.0  # atr_pct 是百分比，转为比例
            if dist > 0:
                return dist
        return float(p["stop_loss_pct"])

    def _entry_size(self, ctx: dict[str, Any]) -> float:
        """入场仓位：启用风险预算时按"单笔风险 / 止损距离"反推（封顶 size_pct）。"""
        p = self.params
        risk = float(p.get("risk_per_trade_pct", 0.0) or 0.0)
        if risk > 0:
            dist = self._stop_dist_pct(ctx)
            if dist > 0:
                return min(float(p["size_pct"]), max(0.05, risk / dist))
        return float(p["size_pct"])

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
                sig = Signal(symbol, "buy", self._entry_size(ctx), strategy=self.name,
                             reason=f"快线上穿慢线({p['fast_period']}/{p['slow_period']})")
            elif cross_down and position > 0:
                sig = Signal(symbol, "sell", 1.0, strategy=self.name, reason="快线下穿慢线")

        self._prev_fast, self._prev_slow = fast, slow

        # 止损 / 止盈（持仓中且没有交叉信号时；intrabar 触价由回测内核处理，
        # 此处的收盘判断保留作非 intrabar 路径的兜底）
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

    def protective_levels(self, ctx: dict[str, Any]) -> Optional[dict]:
        """持仓中且策略声明了止损/止盈位：返回触价位供回测内核盘中检查。"""
        if ctx.get("position", 0) > 0 and self._entry_price:
            p = self.params
            stop = self._entry_price * (1.0 - self._stop_dist_pct(ctx))
            tp = self._entry_price * (1.0 + float(p["take_profit_pct"]))
            return {"stop": stop, "take_profit": tp}
        return None