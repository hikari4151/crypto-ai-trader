"""纸面账户：无交易所 Key 也能完整跑通交易流程，支持自定义初始资金与手续费率。"""
from dataclasses import dataclass, field


@dataclass
class PaperAccount:
    start_cash: float
    fee_rate: float = 0.001        # 手续费率（默认 0.1%，可自定义）
    cash: float = field(init=False)
    positions: dict[str, dict] = field(default_factory=dict)  # symbol -> {"qty","avg_price"}

    def __post_init__(self) -> None:
        self.cash = self.start_cash

    def apply_fill(self, symbol: str, side: str, qty: float, price: float, fee: float = None) -> dict:
        # fee 缺省时按 fee_rate 计算
        if fee is None:
            fee = qty * price * self.fee_rate
        pos = self.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        if side == "buy":
            total_cost = qty * price + fee
            if total_cost > self.cash:
                raise ValueError("纸面账户余额不足")
            new_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + qty * price) / new_qty if new_qty else 0.0
            pos["qty"] = new_qty
            self.cash -= total_cost
        else:  # sell
            if qty > pos["qty"] + 1e-12:
                raise ValueError("纸面账户持仓不足")
            pos["qty"] -= qty
            self.cash += qty * price - fee
            if pos["qty"] < 1e-12:
                pos["qty"] = 0.0
        self.positions[symbol] = pos
        return {"cash": self.cash, "position": pos}

    def positions_value(self, last_prices: dict[str, float]) -> float:
        return sum(p["qty"] * last_prices.get(s, p["avg_price"]) for s, p in self.positions.items())