"""本地模拟交易所：让 simulated（模拟实盘）模式真正走 _place_live 代码路径。

动机：此前 simulated 被当作 paper（paper=True）处理，订单走 _place_paper
（纯内存 apply_fill），实盘订单状态机（create→轮询→撤单→复核→对账→FIFO）
从未被任何非实盘路径触达——纸面完备但实盘无证据。

本模块实现 OrderManager 所需的最小 ccxt 兼容接口，撮合语义与纸面对齐：
- 市价单：按 last_price ± slippage 立即成交（与 _place_paper 同向）
- 限价单：价格触及即成交，未触及挂单簿；支持撤单/对账
- 余额：初始 start_cash（quote）+ 成交持仓（base），fetch_balance 结构同 ccxt
- 故障注入：fail_fetch_order_once / fail_create_order_once 开关，供确定性测试
  （演练 _place_live 的异常分支：轮询失败→注册表兜底、下单失败→不落库）

用法（引擎 simulated 分支）：
    self.exchange = SimulatedExchange(start_cash, fee_rate, slippage)
    self.paper = False
    self.order_manager = OrderManager(db, bus, False, None)
    self.order_manager.attach_exchange(self.exchange)
    self.portfolio = PortfolioManager(db, bus, False, None, self.exchange)
"""
import time
from typing import Any, Optional


class SimulatedExchange:
    has_private = True  # 走实盘分支（_current_state 取 fetch_balance）

    def __init__(self, start_cash: float = 10000.0, fee_rate: float = 0.001,
                 slippage: float = 0.0005, symbol: str = "BTC/USDT") -> None:
        self._cash = float(start_cash)
        self._fee_rate = float(fee_rate)
        self._slippage = float(slippage)
        self._symbol = symbol
        # 持仓：{base: qty}（只维护本模拟交易所撮合产生的量）
        self._positions: dict[str, float] = {}
        # 挂单簿：{oid: {...}}（限价未成交）
        self._open: dict[str, dict] = {}
        # 成交历史：{oid: {...}}
        self._closed: dict[str, dict] = {}
        self._seq = 0
        # 故障注入（一次性）
        self.fail_fetch_order_once = False
        self.fail_create_order_once = False
        self._last_price = 0.0
        # 本地撮合标记：引擎据此跳过余额 30s 缓存（无网络成本、状态须实时）
        self.is_local_matching = True

    # ---------- 撮合核心 ----------
    def _next_oid(self) -> str:
        self._seq += 1
        return f"sim{self._seq}"

    def _apply_fee(self, qty: float, price: float) -> float:
        return qty * price * self._fee_rate

    def _fill(self, symbol: str, otype: str, side: str, amount: float,
              price: float) -> dict:
        """按成交价撮合：更新余额/持仓，返回 ccxt 风格 closed 订单。"""
        fee = self._apply_fee(amount, price)
        base, quote = self._split(symbol)
        if side == "buy":
            cost = amount * price + fee
            if cost > self._cash + 1e-9:
                raise ValueError(f"模拟账户余额不足: need={cost:.4f} cash={self._cash:.4f}")
            self._cash -= cost
            self._positions[base] = self._positions.get(base, 0.0) + amount
        else:
            have = self._positions.get(base, 0.0)
            if amount > have + 1e-9:
                raise ValueError(f"模拟账户持仓不足: sell={amount} have={have}")
            self._positions[base] = have - amount
            if self._positions[base] <= 1e-9:
                self._positions.pop(base, None)
            self._cash += amount * price - fee
        oid = self._next_oid()
        rec = {"id": oid, "symbol": symbol, "type": otype, "side": side,
               "amount": amount, "price": price, "average": price,
               "filled": amount, "status": "closed",
               "fee": {"cost": fee, "currency": quote},
               "timestamp": time.time() * 1000}
        self._closed[oid] = rec
        return dict(rec)

    def _split(self, symbol: str) -> tuple[str, str]:
        base, sep, _ = symbol.partition(":")
        parts = [p for p in base.split("/") if p]
        return (parts[0] if parts else symbol), (parts[1] if len(parts) > 1 else "USDT")

    # ---------- ccxt 兼容接口 ----------
    async def create_order(self, symbol: str, otype: str, side: str, amount: float,
                           price: Optional[float] = None, params: Optional[dict] = None) -> dict:
        if self.fail_create_order_once:
            self.fail_create_order_once = False
            import ccxt
            raise ccxt.NetworkError("simulated create_order failure")
        # 触发式委托（服务端兜底止损 triggerPrice 条件单）：挂单簿等待触及，
        # 不立即撮合——否则 protective stop 一挂就市价平仓（语义错误）
        trigger = (params or {}).get("triggerPrice")
        if trigger is not None:
            oid = self._next_oid()
            rec = {"id": oid, "symbol": symbol, "type": otype, "side": side,
                   "amount": amount, "price": float(trigger), "average": None,
                   "filled": 0.0, "status": "open",
                   "fee": {"cost": 0.0, "currency": "USDT"},
                   "triggerPrice": float(trigger),
                   "timestamp": time.time() * 1000}
            self._open[oid] = rec
            return dict(rec)
        if otype == "limit" and price is not None:
            # 限价单：挂单簿（是否成交由 fetch_order/外部触及模拟；引擎侧
            # 会轮询 fetch_order，测试可用 fetch_order 注入 closed 模拟触及）
            oid = self._next_oid()
            rec = {"id": oid, "symbol": symbol, "type": otype, "side": side,
                   "amount": amount, "price": price, "average": None,
                   "filled": 0.0, "status": "open",
                   "fee": {"cost": 0.0, "currency": "USDT"},
                   "timestamp": time.time() * 1000}
            self._open[oid] = rec
            return dict(rec)
        # 市价单：按 last_price ± slippage 立即成交
        lp = self._last_price or price or 50000.0
        exec_price = lp * (1 + self._slippage) if side == "buy" else lp * (1 - self._slippage)
        return self._fill(symbol, otype, side, amount, exec_price)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        if self.fail_fetch_order_once:
            self.fail_fetch_order_once = False
            import ccxt
            raise ccxt.NetworkError("simulated fetch_order failure")
        rec = self._open.get(order_id) or self._closed.get(order_id)
        if rec is None:
            return {"id": order_id, "symbol": symbol, "status": "canceled",
                    "filled": 0.0, "amount": 0.0}
        return dict(rec)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list:
        return [dict(r) for r in self._open.values()
                if symbol is None or r["symbol"] == symbol]

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        rec = self._open.pop(order_id, None)
        if rec is None:
            return {"id": order_id, "status": "canceled", "symbol": symbol}
        rec = dict(rec)
        rec["status"] = "canceled"
        self._closed[order_id] = rec
        return rec

    async def fetch_balance(self) -> dict:
        # 与 ccxt safe_balance 一致：同时保留两种访问结构——
        # 1) balance['free']['USDT']（按类型）  2) balance['USDT']['free']（按币种嵌套）
        total = {"USDT": self._cash}
        for base, qty in self._positions.items():
            total[base] = qty
        bal: dict[str, Any] = {"total": dict(total), "free": dict(total), "used": {}}
        for code, qty in total.items():
            bal[code] = {"free": qty, "used": 0.0, "total": qty}
        return bal

    async def fetch_ticker(self, symbol: str) -> dict:
        return {"symbol": symbol, "last": self._last_price or 50000.0}

    # ---------- 市场规格（下单前精度/最小名义额校验用） ----------
    def market(self, symbol: str) -> dict:
        return {
            "symbol": symbol, "contract": False,
            "precision": {"amount": 1e-8, "price": 1e-6},
            "limits": {"amount": {"min": 1e-6}, "cost": {"min": 0.0}},
        }

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{float(amount):.8f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{float(price):.6f}"

    async def close(self) -> None:
        pass

    def restore_state(self, cash: float, positions: dict,
                      closed: Optional[dict] = None,
                      seq: int = 0) -> None:
        """重启恢复：从 Trade 表回放重建账户状态（_start_locked simulated 分支调用）。

        SimulatedExchange 状态全在内存，stop→start 会重建实例丢现金/持仓。
        实盘有 _restore_live_position_state（Trade 回放重建 FIFO+入口价），
        本地模拟交易所同样需要恢复——否则重启后现金回 10000、持仓清零、
        成交历史丢失，策略误判持续开仓（正确性 0 容忍修复）。
        """
        self._cash = float(cash)
        self._positions = {k: float(v) for k, v in (positions or {}).items() if float(v) > 1e-9}
        self._closed = dict(closed or {})
        self._seq = int(max(seq, self._seq))

    # ---------- 测试辅助 ----------
    def set_last_price(self, price: float) -> None:
        self._last_price = float(price)

    def inject_open_fill(self, order_id: str, fill_price: float) -> None:
        """测试辅助：把挂单簿订单直接改为已成交（模拟限价单被触及）。"""
        rec = self._open.pop(order_id, None)
        if rec is None:
            raise KeyError(order_id)
        filled = self._fill(rec["symbol"], rec["type"], rec["side"],
                            rec["amount"], fill_price)
        # 保留原 oid，方便引擎按原 id 轮询到 closed
        self._closed.pop(filled["id"], None)
        rec.update({"id": order_id, "average": fill_price, "status": "closed"})
        self._closed[order_id] = rec
        return rec

    def inject_trigger_fill(self, order_id: str, exec_price: Optional[float] = None) -> dict:
        """测试辅助：触发式单（protective stop）被价格触及成交。

        模拟交易所侧触发单成交：按触发价（或指定成交价）撮合卖出，
        更新余额/持仓，订单置 closed。测试用它演练 protective→对账回灌链路。
        """
        rec = self._open.pop(order_id, None)
        if rec is None:
            raise KeyError(order_id)
        trigger = float(rec.get("triggerPrice") or rec.get("price") or 0.0)
        px = exec_price or trigger
        amount = float(rec.get("amount") or 0.0)
        base, quote = self._split(rec["symbol"])
        have = self._positions.get(base, 0.0)
        fill_qty = min(amount, have)
        if fill_qty > 0:
            fee = self._apply_fee(fill_qty, px)
            self._positions[base] = have - fill_qty
            if self._positions[base] <= 1e-9:
                self._positions.pop(base, None)
            self._cash += fill_qty * px - fee
            fee_out = {"cost": fee, "currency": quote}
        else:
            fee_out = {"cost": 0.0, "currency": quote}
        rec.update({"id": order_id, "status": "closed", "filled": fill_qty,
                    "average": px, "fee": fee_out})
        self._closed[order_id] = rec
        return dict(rec)
