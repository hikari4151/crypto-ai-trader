"""钱包余额与持仓：定时快照，以最新价折算 USDT 展示，含浮动盈亏。

公共方法 fetch_live_balance() 供引擎侧每日对账与 API 侧复用。
"""
import logging
import time
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
        self._live_balance_cache: dict = {}
        self._live_balance_ts: float = 0.0

    async def fetch_live_balance(self, api_key: str, secret: str, password: str) -> dict:
        """拉取交易所真实钱包余额（供每日对账与 API 复用；close 由调用方负责）。

        引擎侧：self.exchange 已连接，直接 fetch_balance（30s TTL 缓存，
        异常保留旧缓存——交易所限流时口径不丢失）。
        self.exchange 为 None（API 轮询路径动态创建 manager 的场景）时返回空 dict。

        Returns:
            {"total": {asset: qty}, "free": {...}, "used": {...}} 原样余额结构。
            无 exchange 时返回 {"total": {}}。
        """
        if self.exchange is None:
            return {"total": {}}
        now = time.time()
        if now - self._live_balance_ts < 30 and self._live_balance_cache:
            return self._live_balance_cache
        bal = await self.exchange.fetch_balance()
        self._live_balance_cache = bal
        self._live_balance_ts = now
        return bal

    def update_prices(self, prices: dict[str, float]) -> None:
        self._last_prices.update(prices)

    async def snapshot(self, persist: bool = True) -> dict:
        """返回 {equity, cash, positions_value, positions:[...]}，并落库资金曲线。

        persist=False（API 轮询路径）：只读不落库——曾前端 5s 轮询 /summary
        每次落一条 EquitySnapshot（日增约 1.7 万行）；引擎 15s 快照循环仍落库。
        """
        if self.paper and self.paper_account:
            positions = []
            for sym, p in self.paper_account.positions.items():
                # M1：成本基准为 FIFO 剩余 lots 的加权平均成本（展示口径），
                # 与回测引擎 lots 逐笔记账一致，不再用单点 avg_price
                cost = self.paper_account.avg_price(sym)
                price = self._last_prices.get(sym, cost)
                positions.append({
                    "symbol": sym, "qty": round(p["qty"], 8), "avg_price": round(cost, 6),
                    "last_price": round(price, 6),
                    "value": round(p["qty"] * price, 4),
                    "unrealized_pnl": round((price - cost) * p["qty"], 4),
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
        if persist:
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