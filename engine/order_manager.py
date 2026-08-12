"""订单执行：纸面模式本地撮合 / 实盘模式走交易所，均记录成交到数据库。"""
import logging
import time
from typing import Any, Optional

from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.manager import ExchangeManager
from exchange.paper import PaperAccount
from strategies.base import Signal

log = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, db: Database, bus: EventBus, paper: bool, paper_account: Optional[PaperAccount] = None) -> None:
        self._db = db
        self._bus = bus
        self.paper = paper
        self.paper_account = paper_account
        self.exchange: Optional[ExchangeManager] = None

    def attach_exchange(self, exchange: ExchangeManager) -> None:
        self.exchange = exchange

    async def place(self, signal: Signal, last_price: float, strategy_name: str) -> Optional[dict]:
        """下单并记录成交。返回成交信息或 None（未成交/跳过）。"""
        if self.paper:
            return await self._place_paper(signal, last_price, strategy_name)
        if self.exchange is None:
            log.warning("实盘模式但交易所未就绪，跳过订单 %s", signal)
            return None
        return await self._place_live(signal, last_price, strategy_name)

    def _resolve_qty(self, signal: Signal, price: float) -> float:
        """把信号换算成实际下单数量：
        - qty 明确则直接用
        - 比例下单（size_pct）按可用资金换算：买入用现金*比例/价（并留手续费），卖出用当前持仓
        与回测引擎（fast_engine）逻辑保持一致，避免实盘/纸面以 qty=0 成交。
        """
        if signal.qty:
            return signal.qty
        if not self.paper_account:
            return 0.0
        if signal.side == "buy":
            cash = self.paper_account.cash
            qty = cash * float(signal.size_pct or 0.5) / price
            # 留出手续费空间，避免 apply_fill 余额不足
            fee_rate = getattr(self.paper_account, "fee_rate", 0.001)
            qty = min(qty, cash / (price * (1 + fee_rate)))
            return qty
        # 卖出：全部持仓（或按比例）
        pos = self.paper_account.positions.get(signal.symbol, {}).get("qty", 0.0)
        pct = float(signal.size_pct or 1.0)
        return min(pos, pos * pct) if pct < 1.0 else pos

    async def _place_paper(self, signal: Signal, last_price: float, strategy_name: str) -> Optional[dict]:
        price = signal.limit_price or last_price
        qty = self._resolve_qty(signal, price)
        if qty <= 0:
            log.info("[order] 纸面订单跳过（数量为 0）：%s %s", signal.side, signal.symbol)
            return None
        # 使用账户配置的手续费率（默认 0.1%）
        fee_rate = getattr(self.paper_account, "fee_rate", 0.001) if self.paper_account else 0.001
        fee = qty * price * fee_rate
        fill = self.paper_account.apply_fill(signal.symbol, signal.side, qty, price, fee)
        trade_id = await self._record_trade(signal, price, qty, fee, strategy_name)
        await self._bus.publish(Event(EventType.ORDER_FILL, {
            "symbol": signal.symbol, "side": signal.side, "price": price, "qty": qty, "trade_id": trade_id,
        }, source="order_manager"))
        return {"price": price, "qty": qty, "fee": fee, "trade_id": trade_id, "cash": fill["cash"]}

    async def _resolve_live_qty(self, signal: Signal, price: float) -> float:
        """实盘下单数量换算：qty 明确则直接用；比例下单（size_pct）按真实余额换算。
        买入用报价币种可用现金，卖出用持仓数量，与纸面/回测口径对齐。
        """
        if signal.qty:
            return float(signal.qty)
        try:
            bal = await self.exchange.fetch_balance()
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 获取实盘余额失败，无法换算下单数量: %s", e)
            return 0.0
        if signal.side == "buy":
            quote = signal.symbol.split("/")[-1]
            cash = float(bal.get(quote, {}).get("free", 0.0) or 0.0)
            qty = cash * float(signal.size_pct or 0.5) / price
            # 留出手续费空间，避免下单被拒
            qty = min(qty, cash / (price * (1 + 0.001)))
            return max(qty, 0.0)
        # 卖出：按持仓比例
        base = signal.symbol.split("/")[0]
        pos = float(bal.get(base, {}).get("free", 0.0) or 0.0)
        pct = float(signal.size_pct or 1.0)
        return min(pos, pos * pct) if pct < 1.0 else pos

    async def _place_live(self, signal: Signal, last_price: float, strategy_name: str) -> Optional[dict]:
        try:
            price = signal.limit_price or last_price
            qty = await self._resolve_live_qty(signal, price)
            if qty <= 0:
                log.warning("[order] 实盘下单跳过（数量为 0，余额不足或比例换算失败）：%s %s", signal.side, signal.symbol)
                return None
            order = await self.exchange.create_order(
                signal.symbol, signal.order_type, signal.side,
                qty, signal.limit_price,
            )
            # 简化：市价单直接视为成交；限价单轮询一次状态
            status = order.get("status", "closed")
            if status != "closed":
                oid = order.get("id")
                if oid:
                    order = await self.exchange.fetch_order(oid, signal.symbol)
            filled = float(order.get("filled") or order.get("amount") or 0.0)
            price = float(order.get("average") or order.get("price") or 0.0)
            fee_info = order.get("fee") or {}
            fee = float(fee_info.get("cost") or 0.0)
            if filled <= 0:
                # 限价单未成交（挂单中）：不落库、不发事件、返回 None——
                # 曾把 qty=0 当成交处理，导致策略 on_fill 假触发、频率计数污染
                log.info("[order] 实盘订单未成交（挂单中），跳过记录: %s %s oid=%s",
                         signal.side, signal.symbol, order.get("id", ""))
                return None
            trade_id = await self._record_trade(signal, price, filled, fee, strategy_name, order.get("id", ""))
            await self._bus.publish(Event(EventType.ORDER_FILL, {
                "symbol": signal.symbol, "side": signal.side, "price": price, "qty": filled, "trade_id": trade_id,
            }, source="order_manager"))
            return {"price": price, "qty": filled, "fee": fee, "trade_id": trade_id}
        except Exception as e:  # noqa: BLE001
            log.exception("[order] 实盘下单失败: %s", signal)
            return None

    async def _record_trade(self, signal: Signal, price: float, qty: float, fee: float,
                            strategy_name: str, order_id: str = "") -> int:
        from core.database import Trade
        async with self._db.session() as s:
            trade = Trade(exchange="paper" if self.paper else "live", symbol=signal.symbol,
                          side=signal.side, price=price, qty=qty, value=price * qty, fee=fee,
                          strategy=strategy_name, order_id=order_id, reason=signal.reason)
            s.add(trade)
            await s.commit()
            await s.refresh(trade)
        return trade.id

    async def cancel(self, order_id: str, symbol: str) -> None:
        if not self.paper and self.exchange:
            await self.exchange.cancel_order(order_id, symbol)