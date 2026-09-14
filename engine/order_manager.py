"""订单执行：纸面模式本地撮合 / 实盘模式走交易所，均记录成交到数据库。"""
import asyncio
import ccxt
import logging
import time
from typing import Optional

from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.manager import ExchangeManager
from exchange.paper import PaperAccount, fifo_consume
from exchange.symbols import base_currency, quote_currency
from strategies.base import Signal

from .protective_stop import ProtectiveStopManager

log = logging.getLogger(__name__)

# ccxt 订单终止态（各交易所/版本拼写不一，全部收进来）
_TERMINAL_ORDER_STATUS = {"closed", "canceled", "cancelled", "expired", "rejected"}


class OrderManager:
    def __init__(self, db: Database, bus: EventBus, paper: bool, paper_account: Optional[PaperAccount] = None) -> None:
        self._db = db
        self._bus = bus
        self.paper = paper
        self.paper_account = paper_account
        self.exchange: Optional[ExchangeManager] = None
        # open-order 注册表（key=order_id）：限价单撤单/未及时捕获的挂单，
        # 由对账循环每 ~30s 与交易所 open 订单比对，回灌迟到成交
        self._open_orders: dict[str, dict] = {}
        self._reconcile_task: Optional[asyncio.Task] = None
        # 实盘 FIFO 成本队列（仅 live 模式维护）：symbol -> [(qty, price), ...]
        # 与回测引擎 lots 逐笔记账同款——分批建仓/部分平仓时 PnL 口径一致
        self._live_lots: dict[str, list[tuple[float, float]]] = {}
        # 服务端兜底止损（仅实盘）：attach_exchange 时按配置构建，纸面恒为 None
        self.protective: Optional[ProtectiveStopManager] = None
        # 策略当前止损比例来源，由引擎注入（切换策略无需重建订单管理器）
        self._stop_pct_source = None

    def _apply_fifo(self, symbol: str, side: str, qty: float, price: float) -> Optional[float]:
        """实盘 FIFO 成本队列记账：buy 追加 lot，sell FIFO 摊销并返回成本价。

        摊销逻辑复用 fifo_consume（backtest/engine.py 的 lots.pop(0) 同款），
        与 PaperAccount.apply_fill 单一实现；仅 live 模式维护。
        卖出若超出队列覆盖（外部建仓的持仓），仅消费已覆盖部分并返回 None，
        交由 _record_fill 的 entry_price 兜底，避免用不完整成本算 PnL。
        """
        lots = self._live_lots.setdefault(symbol, [])
        if side == "buy":
            lots.append((qty, price))
            return None
        total_before = sum(lq for lq, _lp in lots)
        cost_price = fifo_consume(lots, qty)
        if total_before + 1e-12 < qty:
            # 卖出量超出队列覆盖（外部建仓/历史持仓）：成本不可信，兜底 entry_price
            log.warning("[order] 实盘 FIFO 队列未覆盖全部卖出量 %s: sell=%s queue=%s（本轮仅消费 %s），成本走兜底",
                        symbol, qty, total_before, min(total_before, qty))
            cost_price = None
        elif cost_price <= 0.0:
            # 队列为空/未消费到任何 lot：无成本参考
            cost_price = None
        if not lots:
            # 清仓后清理空 key，防 dict 无限增长（m3）
            self._live_lots.pop(symbol, None)
        return cost_price

    def attach_exchange(self, exchange: ExchangeManager) -> None:
        self.exchange = exchange
        if self.paper:
            self.protective = None
            return
        self.protective = ProtectiveStopManager(
            exchange, self._bus,
            enabled=bool(settings.protective_stop_enabled),
            default_stop_pct=float(settings.protective_stop_default_pct),
            buffer_pct=float(settings.protective_stop_buffer_pct),
            stop_pct_source=self._stop_pct_source,
        )

    def set_stop_pct_source(self, source) -> None:
        """引擎注入策略止损比例来源；attach_exchange 之前注入也能生效。"""
        self._stop_pct_source = source
        if self.protective is not None:
            self.protective._stop_pct_source = source

    def _position_book(self, symbol: str) -> tuple[float, float]:
        """FIFO 队列 → (未平仓量, 加权均价)。"""
        lots = self._live_lots.get(symbol) or []
        qty = sum(lq for lq, _lp in lots)
        if qty <= 0:
            return 0.0, 0.0
        return qty, sum(lq * lp for lq, lp in lots) / qty

    def _pending_sell_qty(self, symbol: str) -> float:
        """仍在交易所挂着的引擎卖单数量（这部分基础币已被锁定，不能重复计入兜底止损）。

        必须排除兜底止损自身——它同样是一条 side=="sell" 的挂单，
        若计进来，下一次对齐会算出"剩余可保护量 0"而把它自己撤掉。
        """
        total = 0.0
        for rec in self._open_orders.values():
            if rec.get("protective") or rec.get("symbol") != symbol or rec.get("side") != "sell":
                continue
            total += max(0.0, float(rec.get("qty") or 0.0) - float(rec.get("recorded_filled") or 0.0))
        return total

    async def _resync_protective(self, symbol: str, ref_price: float = 0.0) -> None:
        """把服务端兜底止损对齐到当前持仓；任何失败都不影响订单主链路。"""
        if self.paper or self.protective is None:
            return
        try:
            qty, entry = self._position_book(symbol)
            qty = max(0.0, qty - self._pending_sell_qty(symbol))
            rec = await self.protective.sync(symbol, qty, entry, ref_price or entry)
            if rec and rec.get("id"):
                self._register_protective({**rec, "symbol": symbol})
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 兜底止损同步失败 %s: %s", symbol, e)

    async def restore_live_state(self, symbol: str) -> dict:
        """重启回放：按 Trade 表重建实盘 FIFO 成本队列（仅 live 模式调用）。

        _live_lots 只活在进程内存里，重启即清空，此后每笔卖出都被判成
        "队列未覆盖全部卖出量"（成本走兜底、Trade.pnl 失真），策略 _entry
        也一起丢——price_action/dual_ma 的止损条件写作
        `position > 0 and self._entry`，于是实盘持仓在重启后再没有止损。
        回放口径与 _apply_fifo 完全一致（buy 追加、sell 用 fifo_consume 摊销）。

        Returns:
            {"qty": 未平仓量, "entry": 末笔买入价（无成交时 0.0）, "trades": 回放笔数}
        """
        from sqlalchemy import select
        from core.database import Trade
        async with self._db.session() as s:
            rows = (await s.execute(
                select(Trade).where(Trade.symbol == symbol, Trade.exchange == "live")
                .order_by(Trade.ts.asc(), Trade.id.asc()))).scalars().all()
        lots: list[tuple[float, float]] = []
        last_buy_price = 0.0
        for t in rows:
            qty = float(t.qty or 0.0)
            price = float(t.price or 0.0)
            if qty <= 0 or price <= 0:
                continue
            if t.side == "buy":
                lots.append((qty, price))
                last_buy_price = price
            elif t.side == "sell":
                fifo_consume(lots, qty)
        open_qty = sum(lq for lq, _lp in lots)
        if lots:
            self._live_lots[symbol] = lots
        else:
            self._live_lots.pop(symbol, None)
        log.info("[order] 实盘状态回放 %s: %d 笔成交 → 未平仓 %s，FIFO 剩余 %d lot，末笔买入 %s",
                 symbol, len(rows), open_qty, len(lots), last_buy_price)
        await self._restore_protective_stop(symbol)
        return {"qty": open_qty, "entry": last_buy_price, "trades": len(rows)}

    def _register_protective(self, rec: dict) -> None:
        """兜底止损单登记对账：它在交易所侧触发后，靠对账循环回灌成交，
        本地账本才不会残留一笔交易所早已卖掉的持仓。"""
        oid = str(rec.get("id") or "")
        symbol = str(rec.get("symbol") or "")
        if not oid or not symbol:
            return
        self._register_open_order(
            oid,
            Signal(symbol, "sell", qty=float(rec.get("qty") or 0.0),
                   strategy="protective_stop", reason="服务端兜底止损触发"),
            float(rec.get("qty") or 0.0), float(rec.get("trigger") or 0.0), protective=True)

    async def _restore_protective_stop(self, symbol: str) -> None:
        """重启后重建服务端兜底止损：能收养就收养（不撤不重挂），否则按回放出的持仓新挂。"""
        if self.paper or self.protective is None:
            return
        try:
            adopted = await self.protective.adopt(symbol)
            if adopted:
                self._register_protective({**adopted, "symbol": symbol})
            await self._resync_protective(symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 兜底止损重启重建失败 %s: %s", symbol, e)

    async def close(self) -> None:
        """关闭对账任务（引擎 stop 时调用，防任务泄漏到下次启动）。"""
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._reconcile_task = None

    def _ensure_reconcile(self) -> None:
        """惰性启动对账循环（首次实盘限价单挂出后）。"""
        if self._reconcile_task is None or self._reconcile_task.done():
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="order_reconcile")

    def _register_open_order(self, oid: str, signal: Signal, qty: float, limit_price: float,
                             protective: bool = False) -> None:
        """把撤单未成交的限价单登记进注册表（对账回灌迟到成交用）。"""
        self._open_orders[oid] = {
            "symbol": signal.symbol, "side": signal.side, "qty": qty,
            "strategy_name": signal.strategy or "", "placed_ts": time.time(),
            "limit_price": limit_price, "recorded_filled": 0.0,
            "reason": signal.reason, "protective": protective,
        }
        self._ensure_reconcile()

    async def _reconcile_loop(self) -> None:
        """对账循环：每 ~30s 跑一轮 _reconcile_once。

        循环体 try/except：单次失败不得杀死任务（参照 trading_engine._portfolio_loop 风格）。
        """
        while True:
            try:
                await self._reconcile_once()
            except Exception as e:  # noqa: BLE001
                log.exception("[order] 对账循环异常，30s 后继续: %s", e)
            await asyncio.sleep(30)

    async def _reconcile_once(self) -> None:
        """单轮对账：注册表里有、交易所 open 列表已无的订单复核——
        迟到成交（filled>0）→ 回灌 Trade 行 + ORDER_FILL(late) 事件。
        """
        if self.paper or not self.exchange or not self._open_orders:
            return
        open_ids: set[str] = set()
        try:
            open_orders = await self.exchange.fetch_open_orders()
            open_ids = {str(o.get("id")) for o in open_orders if o.get("id")}
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 对账拉取 open 订单失败: %s", e)
        for oid, rec in list(self._open_orders.items()):
            if oid in open_ids:
                continue  # 交易所仍挂单，保留注册表
            # 注册表有、交易所列表已无：撤单竞态/已成交/被取消——复核
            try:
                order = await self.exchange.fetch_order(oid, rec["symbol"])
            except Exception as e:  # noqa: BLE001
                log.warning("[order] 对账复核失败（保留注册表下轮再查）: oid=%s: %s", oid, e)
                continue
            filled = float(order.get("filled") or 0.0)
            if filled > 0:
                # 迟到成交（撤单竞态下已成交等）：回灌
                await self._backfill_fill(oid, rec, order)
            elif str(order.get("status") or "") in _TERMINAL_ORDER_STATUS:
                # 已终结且未成交（撤单/过期）：清理注册表
                log.info("[order] 对账：订单已终结未成交，清理注册表 oid=%s status=%s",
                         oid, order.get("status"))
                self._open_orders.pop(oid, None)
                if rec.get("protective") and self.protective is not None:
                    # 兜底止损在交易所侧已不存在（被外部撤销/过期）：
                    # 本地记录必须一起清掉，否则此后撤单永远失败、保护静默消失
                    self.protective.forget(rec["symbol"], oid)
                    await self._resync_protective(rec["symbol"])
            else:
                log.warning("[order] 对账：注册表有但交易所列表未见且仍 open，下轮复查 oid=%s", oid)

    async def _backfill_fill(self, oid: str, rec: dict, order: dict) -> None:
        """迟到成交回灌：记 Trade 行 + 发布 ORDER_FILL(late) 事件。

        引擎侧订阅 ORDER_FILL 后走公共记账方法 _record_fill（与即时成交同口径）。
        幂等：先移除注册表再落库——若 _record_trade 瞬时失败（DB 锁等），
        下轮对账不再重试（重复回灌会把 Trade/on_fill/日盈亏双计，
        比缺记更危险；缺记可经交易所余额对账发现）。
        """
        self._open_orders.pop(oid, None)
        side = rec["side"]
        # M2：delta 回灌——仅补录之前未记录的成交数量
        total_filled = float(order.get("filled") or 0.0)
        recorded = float(rec.get("recorded_filled") or 0.0)
        filled = max(0.0, total_filled - recorded)
        if filled <= 1e-12:
            log.info("[order] 对账：订单已关闭且无新增成交，清理注册表 oid=%s", oid)
            return
        price = float(order.get("average") or order.get("price") or 0.0)
        if price <= 0:
            # P2-7 顺手修复：交易所未返回 average 时用限价兜底，避免成交价 0 落库
            price = rec["limit_price"]
        fee_info = order.get("fee") or {}
        total_fee = float(fee_info.get("cost") or 0.0)
        # 部分成交已即时记账时，交易所返回的 fee 按新增量折算，避免手续费双计
        if total_filled > 0 and recorded > 0:
            fee = total_fee * (filled / total_filled)
        else:
            fee = total_fee
        sig = Signal(rec["symbol"], side, qty=filled, strategy=rec["strategy_name"],
                     reason=rec.get("reason") or "")
        if rec.get("protective") and self.protective is not None:
            # 兜底止损已在交易所侧了结（触发或被外部撤销）：清记录，避免去撤一个不存在的单
            self.protective.forget(rec["symbol"], oid)
        # 先落库，成功后 FIFO 队列才消费/追加——防 _record_trade 异常时队列污染
        trade_id = await self._record_trade(sig, price, filled, fee, rec["strategy_name"], oid)
        cost_price = self._apply_fifo(rec["symbol"], side, filled, price) if price > 0 else None
        await self._bus.publish(Event(EventType.ORDER_FILL, {
            "symbol": rec["symbol"], "side": side, "price": price, "qty": filled,
            "trade_id": trade_id, "fee": fee, "strategy_name": rec["strategy_name"],
            "cost_price": cost_price, "late": True,
        }, source="order_manager"))
        await self._resync_protective(rec["symbol"], price)
        log.info("[order] 迟到成交回灌: %s %s oid=%s filled=%s price=%s",
                 side, rec["symbol"], oid, filled, price)

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
        try:
            # 成交价建模与回测引擎对齐：
            # - 限价单：按限价成交，不叠加滑点（与 backtest/_matching._limit_fill_ratio 一致）
            # - 市价单：买入按 last_price*(1+slippage)、卖出按 last_price*(1-slippage)，
            #   与 backtest 的 open*(1±slippage) 同向。此前纸面恒按 last_price 成交
            #   且无滑点，导致纸面系统性比回测更乐观、实盘结果无法复现回测。
            if signal.limit_price is not None:
                price = signal.limit_price
            else:
                price = last_price
                slippage = getattr(self.paper_account, "slippage", 0.0005) if self.paper_account else 0.0005
                if slippage:
                    price = price * (1 + slippage) if signal.side == "buy" else price * (1 - slippage)
            # 数量在滑点后的成交价上换算（与回测 _execute_fill 的 cash*size_pct/fill_price 同口径）
            qty = self._resolve_qty(signal, price)
            if qty <= 0:
                log.info("[order] 纸面订单跳过（数量为 0）：%s %s", signal.side, signal.symbol)
                return None
            # 使用账户配置的手续费率（默认 0.1%）
            fee_rate = getattr(self.paper_account, "fee_rate", 0.001) if self.paper_account else 0.001
            fee = qty * price * fee_rate
            fill = self.paper_account.apply_fill(signal.symbol, signal.side, qty, price, fee)
        except ValueError as e:
            # 余额/持仓不足等边界：曾直接抛出中断整根 K 线处理
            # （事件被 bus 吞掉，后续止损止盈信号全部丢失）
            log.warning("[order] 纸面成交被拒（%s %s）: %s", signal.side, signal.symbol, e)
            return None
        except Exception as e:  # noqa: BLE001
            log.exception("[order] 纸面成交异常: %s", signal)
            return None
        trade_id = await self._record_trade(signal, price, qty, fee, strategy_name)
        await self._bus.publish(Event(EventType.ORDER_FILL, {
                "symbol": signal.symbol, "side": signal.side, "price": price, "qty": qty, "trade_id": trade_id,
                "fee": fee, "cost_price": fill.get("cost_price"), "strategy_name": strategy_name,
            }, source="order_manager"))
        return {"price": price, "qty": qty, "fee": fee, "trade_id": trade_id, "cash": fill["cash"],
                "cost_price": fill.get("cost_price")}

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
            quote = quote_currency(signal.symbol)
            cash = float(bal.get(quote, {}).get("free", 0.0) or 0.0)
            qty = cash * float(signal.size_pct or 0.5) / price
            # 留出手续费空间，避免下单被拒
            qty = min(qty, cash / (price * (1 + 0.001)))
            return max(qty, 0.0)
        # 卖出：按持仓比例
        base = base_currency(signal.symbol)
        pos = float(bal.get(base, {}).get("free", 0.0) or 0.0)
        pct = float(signal.size_pct or 1.0)
        return min(pos, pos * pct) if pct < 1.0 else pos

    def _market_spec(self, symbol: str) -> Optional[dict]:
        """交易所市场规格（精度/限额）。不可用时返回 None——规格是附加信息，
        绝不能因为它读不到就让下单链路崩掉。"""
        if self.exchange is None:
            return None
        try:
            return self.exchange.market(symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 读取市场规格失败（本轮跳过精度修正）%s: %s", symbol, e)
            return None

    def _fit_live_qty(self, symbol: str, qty: float, price: float, side: str) -> float:
        """按交易所规格修正下单数量：精度截断 + 最小名义额。

        曾把原始 float 直接发给交易所：数量带十几位小数 → 精度拒单，小额单 →
        minNotional 拒单，两类错误都只在 _place_live 的宽泛 except 里留一行日志。
        卖单截断为 0 时原样返回——把止损单截成 0 等于不平仓，宁可让交易所裁决。
        """
        if qty <= 0:
            return qty
        mkt = self._market_spec(symbol)
        if not mkt:
            return qty
        try:
            rounded = float(self.exchange.amount_to_precision(symbol, qty))
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 数量精度换算失败（原样下单）%s: %s", symbol, e)
            return qty
        if side == "sell":
            return rounded if rounded > 0 else qty
        cost_min = float(((mkt.get("limits") or {}).get("cost") or {}).get("min") or 0.0)
        if rounded <= 0 or (cost_min > 0 and price > 0 and rounded * price < cost_min):
            log.warning("[order] 买单低于交易所下限，跳过 %s: qty=%s 名义额=%s（最小 %s）",
                        symbol, rounded, rounded * price, cost_min)
            return 0.0
        return rounded

    def _fit_live_price(self, symbol: str, price: Optional[float]) -> Optional[float]:
        """限价按交易所 tick 截断（市价单传 None 即原样返回）。"""
        if not price or not self._market_spec(symbol):
            return price
        try:
            rounded = float(self.exchange.price_to_precision(symbol, price))
        except Exception as e:  # noqa: BLE001
            log.warning("[order] 价格精度换算失败（原样下单）%s: %s", symbol, e)
            return price
        return rounded or price

    async def _place_live(self, signal: Signal, last_price: float, strategy_name: str) -> Optional[dict]:
        try:
            return await self._place_live_inner(signal, last_price, strategy_name)
        finally:
            # 卖出前会主动撤掉服务端兜底止损（避免余额锁定/重复卖出），
            # 因此不论这笔成不成、跳不跳过，结束时都要重新对齐
            await self._resync_protective(signal.symbol, last_price)

    async def _place_live_inner(self, signal: Signal, last_price: float, strategy_name: str) -> Optional[dict]:
        try:
            if signal.side == "sell" and self.protective is not None:
                await self.protective.release(signal.symbol)
            # 限价先按 tick 截断，数量再按 stepSize/minNotional 修正：
            # 曾两者都原样发出，规格不符的单被子交易所拒（只在宽泛 except 里留日志）
            limit_price = self._fit_live_price(signal.symbol, signal.limit_price)
            price = limit_price or last_price
            qty = await self._resolve_live_qty(signal, price)
            qty = self._fit_live_qty(signal.symbol, qty, price, signal.side)
            if qty <= 0:
                log.warning("[order] 实盘下单跳过（数量为 0：余额不足/低于最小名义额/换算失败）：%s %s",
                            signal.side, signal.symbol)
                return None
            order = await self.exchange.create_order(
                signal.symbol, signal.order_type, signal.side,
                qty, limit_price,
            )
            # 简化：市价单直接视为成交；限价单轮询状态（重试 3 次，间隔 1s——
            # 曾单次 fetch_order，网络抖动时订单已成交却返回 None，
            # 系统无 Trade 记录/on_fill 不触发，下次同向信号重复下单）
            status = order.get("status", "closed")
            if status != "closed":
                oid = order.get("id")
                if oid:
                    for _ in range(3):
                        try:
                            order = await self.exchange.fetch_order(oid, signal.symbol)
                        except Exception as e:  # noqa: BLE001
                            # 轮询期间网络异常：订单在交易所侧仍挂单/可能已成交，
                            # 但本地无 Trade 记录、未进注册表——对账循环无从得知它的
                            # 存在，静默丢失 = 隐性敞口。立即登记注册表交给对账兜底
                            # （曾异常冒泡到外层宽泛 except 被吞，仅留一行日志）。
                            log.warning("[order] 轮询订单状态失败，登记对账兜底 oid=%s: %s",
                                        oid, e)
                            self._register_open_order(oid, signal, qty, price)
                            return None
                        if order.get("status") == "closed" or float(order.get("filled") or 0.0) > 0:
                            break
                        await asyncio.sleep(1.0)
                    status = order.get("status", "closed")
                    if status != "closed" and float(order.get("filled") or 0.0) <= 0:
                        # 轮询超时仍未成交：主动撤单（防限价单滞留交易所产生隐性敞口）
                        try:
                            await self.exchange.cancel_order(oid, signal.symbol)
                        except Exception as e:  # noqa: BLE001
                            log.warning("[order] 限价单撤单失败（保留注册表对账兜底）: %s %s oid=%s: %s",
                                        signal.side, signal.symbol, oid, e)
                        # 撤单竞态：撤销瞬间可能已成交，复核一次再判
                        try:
                            order = await self.exchange.fetch_order(oid, signal.symbol)
                        except (ccxt.NetworkError, asyncio.TimeoutError) as e:  # 撤单后复核：网络错误或超时属于已知异常类型
                            log.warning("[order] 撤单后复核失败: %s", e)
                        if order.get("status") != "closed" and float(order.get("filled") or 0.0) <= 0:
                            # 确认未成交（仍 open）：登记注册表，交给对账循环回灌迟到成交
                            self._register_open_order(oid, signal, qty, price)
                            log.warning("[order] 限价单超时未成交已撤单，登记对账: %s %s oid=%s",
                                        signal.side, signal.symbol, oid)
            # filled 口径：仅 status=="closed" 允许用 amount 兜底成交数量；
            # open 且 filled 为 None/0 → 保持 0（曾 `filled or amount` 在挂单未
            # 成交时误判全量成交——虚拟持仓/on_fill 假触发/频率计数污染）
            status = order.get("status", "closed")
            filled = float(order.get("filled") or 0.0)
            if status == "closed" and filled <= 0:
                filled = float(order.get("amount") or 0.0)
            price = float(order.get("average") or order.get("price") or 0.0)
            fee_info = order.get("fee") or {}
            fee = float(fee_info.get("cost") or 0.0)
            # 部分成交且仍挂单：剩余部分交易所继续撮合——落库成功后再登记
            # 对账注册表（M2），recorded_filled 才与已落库 Trade 一致
            partial_open = order.get("status") != "closed" and filled > 0
            if filled <= 0:
                # 限价单未成交（挂单中）：不落库、不发事件、返回 None——
                # 曾把 qty=0 当成交处理，导致策略 on_fill 假触发、频率计数污染
                log.info("[order] 实盘订单未成交（挂单中），跳过记录: %s %s oid=%s",
                         signal.side, signal.symbol, order.get("id", ""))
                return None
            # M1：成交入 FIFO 成本队列（buy 追加 lot / sell FIFO 摊销），
            # 返回成本价供 _record_fill 记账，与回测引擎 lots 口径一致。
            # 必须在 _record_trade 成功后执行，否则 _record_trade 抛出异常时
            # FIFO 队列已消费但 Trade 未落库，后续同 symbol 的 sell 成本基准偏移。
            trade_id = await self._record_trade(signal, price, filled, fee, strategy_name, order.get("id", ""))
            cost_price = self._apply_fifo(signal.symbol, signal.side, filled, price) if price > 0 else None
            if partial_open:
                oid = order.get("id", "")
                if oid:
                    self._register_open_order(oid, signal, qty, price)
                    self._open_orders[oid]["recorded_filled"] = filled
                    log.warning("[order] 订单部分成交且仍挂单（已登记对账，后续成交回灌）: %s %s oid=%s filled=%s/%s",
                                signal.side, signal.symbol, oid, filled, qty)
            await self._bus.publish(Event(EventType.ORDER_FILL, {
                "symbol": signal.symbol, "side": signal.side, "price": price, "qty": filled, "trade_id": trade_id,
                "fee": fee, "cost_price": cost_price, "strategy_name": strategy_name,
            }, source="order_manager"))
            return {"price": price, "qty": filled, "fee": fee, "trade_id": trade_id, "cost_price": cost_price}
        except Exception as e:  # noqa: BLE001  # 实盘下单整体 guard：含 DB 落库（_record_trade）、交易所交互、FIFO 记账等多环节，保留宽泛以防单点异常导致引擎无响应
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
        # commit 已完成 flush，自增主键已回填（expire_on_commit=False）；refresh 是
        # 多余的一次 SELECT 往返（信号→下单热路径）。返回 id 供 update_trade_pnl 使用。
        return trade.id

    async def cancel(self, order_id: str, symbol: str) -> None:
        if not self.paper and self.exchange:
            await self.exchange.cancel_order(order_id, symbol)