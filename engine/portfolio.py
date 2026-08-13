"""钱包余额与持仓：定时快照，以最新价折算 USDT 展示，含浮动盈亏。"""
import logging
from typing import Any, Optional

from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.manager import ExchangeManager
from exchange.paper import PaperAccount

log = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, db: Database, bus: EventBus, paper: bool,
                 paper_account: Optional[PaperAccount], exchange: Optional[ExchangeManager]) -> None:
        self._db = db
        self._bus = bus
        self.paper = paper
        self.paper_account = paper_account
        self.exchange = exchange
        self._last_prices: dict[str, float] = {}

    def update_prices(self, prices: dict[str, float]) -> None:
        self._last_prices.update(prices)

    async def snapshot(self) -> dict:
        """返回 {equity, cash, positions_value, positions:[...]}，并落库资金曲线。"""
        if self.paper and self.paper_account:
            positions = []
            for sym, p in self.paper_account.positions.items():
                price = self._last_prices.get(sym, p["avg_price"])
                positions.append({
                    "symbol": sym, "qty": round(p["qty"], 8), "avg_price": round(p["avg_price"], 6),
                    "last_price": round(price, 6),
                    "value": round(p["qty"] * price, 4),
                    "unrealized_pnl": round((price - p["avg_price"]) * p["qty"], 4),
                })
            positions_value = sum(x["value"] for x in positions)
            equity = self.paper_account.cash + positions_value
            cash = self.paper_account.cash
        else:
            cash, positions, positions_value = 0.0, [], 0.0
            if self.exchange and self.exchange.has_private:
                bal = await self.exchange.fetch_balance()
                total = bal.get("total", {})
                used = bal.get("used", {})
                free = bal.get("free", {})
                for asset, qty in total.items():
                    if not qty:
                        continue
                    if asset == "USDT":
                        # 计价币：只计现金，不进持仓（曾生成 "USDT/USDT" 条目并可能双计）
                        cash += qty
                        continue
                    symbol = f"{asset}/USDT"
                    price = self._last_prices.get(symbol) or (await self._safe_ticker(symbol))
                    value = qty * (price or 0.0)
                    positions_value += value
                    positions.append({
                        "symbol": symbol, "qty": round(qty, 8), "avg_price": 0.0,
                        "last_price": round(price or 0.0, 6), "value": round(value, 4),
                        "unrealized_pnl": 0.0,
                        "free": round(free.get(asset, 0.0), 8), "used": round(used.get(asset, 0.0), 8),
                    })
                equity = cash + positions_value
            else:
                equity = self._last_prices.get("__equity__", 0.0)

        result = {"equity": round(equity, 4), "cash": round(cash, 4),
                  "positions_value": round(positions_value, 4), "positions": positions}
        from core.database import EquitySnapshot
        async with self._db.session() as s:
            s.add(EquitySnapshot(equity=equity, cash=cash, positions_value=positions_value))
            await s.commit()
        await self._bus.publish(Event(EventType.PORTFOLIO_UPDATE, result, source="portfolio"))
        return result

    async def _safe_ticker(self, symbol: str) -> Optional[float]:
        try:
            t = await self.exchange.fetch_ticker(symbol)
            return float(t.get("last") or 0.0)
        except Exception:  # noqa: BLE001
            return None