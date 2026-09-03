"""纸面账户：无交易所 Key 也能完整跑通交易流程，支持自定义初始资金与手续费率。

持仓成本基准与回测引擎对齐（M1）：FIFO 逐笔记账（lots 队列），
分批建仓/部分平仓时 PnL 与回测引擎的 lots.pop(0) 逻辑一致，
而非加权平均成本（avg_price）。

模块级函数 fifo_consume() 供 order_manager 实盘 FIFO 队列共用，单一实现。
"""
from dataclasses import dataclass, field


def fifo_consume(lots: list, qty: float) -> float:
    """FIFO 摊销并消费 lots 队列（与 backtest/engine.py 的 lots.pop(0) 逻辑一致）。

    从队列头部按 qty 反向摊销，消费队列中的 lot；返回已消费部分的加权平均成本价。
    调用方应自行保证 qty 不超 lots 总量（超出时仅消费队列中可覆盖的部分）。

    Returns:
        已消费部分的加权平均成本价；无消费时返回 0.0。
    """
    remaining = qty
    cost_basis = 0.0
    while remaining > 1e-12 and lots:
        lot_qty, lot_price = lots[0]
        take = min(lot_qty, remaining)
        cost_basis += take * lot_price
        remaining -= take
        if take >= lot_qty - 1e-12:
            lots.pop(0)
        else:
            lots[0] = (lot_qty - take, lot_price)
    consumed = qty - remaining
    if consumed <= 1e-12:
        return 0.0
    return cost_basis / consumed


@dataclass
class PaperAccount:
    start_cash: float
    fee_rate: float = 0.001        # 手续费率（默认 0.1%，可自定义）
    slippage: float = 0.0005       # 市价单滑点（默认与回测引擎一致；限价单不叠加）
    cash: float = field(init=False)
    positions: dict[str, dict] = field(default_factory=dict)  # symbol -> {"qty","lots":[(qty,price),...]}

    def __post_init__(self) -> None:
        self.cash = self.start_cash

    def apply_fill(self, symbol: str, side: str, qty: float, price: float, fee: float = None) -> dict:
        # fee 缺省时按 fee_rate 计算
        if fee is None:
            fee = qty * price * self.fee_rate
        pos = self.positions.get(symbol, {"qty": 0.0, "lots": []})
        cost_price = None
        if side == "buy":
            total_cost = qty * price + fee
            if total_cost > self.cash:
                raise ValueError("纸面账户余额不足")
            pos["qty"] += qty
            pos["lots"].append((qty, price))
            self.cash -= total_cost
        else:  # sell
            if qty > pos["qty"] + 1e-12:
                raise ValueError("纸面账户持仓不足")
            # FIFO 摊销成本（与 backtest/engine.py 的 lots 逐笔记账一致，
            # 与 order_manager._apply_fifo 共用 fifo_consume 实现）
            cost_price = fifo_consume(pos["lots"], qty)
            pos["qty"] -= qty
            if pos["qty"] < 1e-12:
                pos["qty"] = 0.0
                pos["lots"] = []
            self.cash += qty * price - fee
        self.positions[symbol] = pos
        return {"cash": self.cash, "position": pos, "cost_price": cost_price}

    def avg_price(self, symbol: str) -> float:
        """当前持仓 FIFO 剩余 lots 的加权平均成本（供估值/展示兜底）。"""
        pos = self.positions.get(symbol, {})
        lots = pos.get("lots", [])
        total_qty = sum(lq for lq, _lp in lots)
        if total_qty <= 1e-12:
            return 0.0
        return sum(lq * lp for lq, lp in lots) / total_qty

    def positions_value(self, last_prices: dict[str, float]) -> float:
        return sum(p["qty"] * last_prices.get(s, self.avg_price(s))
                   for s, p in self.positions.items())