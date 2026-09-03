"""实盘服务端兜底止损（hard stop）：成交后在交易所挂触发式市价卖单。

引擎的止损判定全部跑在本机进程里——进程崩溃、断网、机器休眠、被 OOM 干掉期间，
交易所在持仓侧不留任何保护，行情继续跌就是裸奔。这里挂的是最后一道防线，
触发价刻意比引擎软止损更深（buffer_pct），正常路径永远由引擎先平仓，
只有本地链路真的失效时它才会成交。

三条硬约束：
- 只在实盘生效（纸面/模拟撮合没有交易所侧订单）
- 引擎主动卖出前先撤掉它：币安现货条件单会锁定基础币余额，
  不撤会让引擎软止损的卖单因余额不足被拒，还可能同一笔持仓被卖两次
- 挂单/撤单失败只降级为告警 + SYSTEM 事件（软止损仍然在），绝不带崩下单主链路
"""
import logging
import time
from typing import Callable, Optional

from core.bus import EventBus
from core.events import Event, EventType

log = logging.getLogger(__name__)

# clientOrderId 前缀：重启后据此从交易所挂单里收养我们自己挂的兜底止损，
# 绝不误撤用户手工挂的单
STOP_TAG = "ps_"
# 触发价对齐容差（相对值）： within 之内不重挂，避免每根 K 线打一次交易所
_TRIGGER_TOL = 1e-4
_QTY_TOL = 1e-3


class ProtectiveStopManager:
    """维护 symbol -> 交易所兜底止损单 的一对一映射，负责挂/撤/收养/对齐。"""

    def __init__(self, exchange, bus: EventBus, *, enabled: bool = True,
                 default_stop_pct: float = 0.07, buffer_pct: float = 0.02,
                 stop_pct_source: Optional[Callable[[], Optional[float]]] = None) -> None:
        self.exchange = exchange
        self._bus = bus
        self.enabled = enabled
        self._default_pct = default_stop_pct
        self._buffer_pct = buffer_pct
        # 策略当前 stop_loss_pct 的来源（引擎在切换策略时不需要重建本对象）
        self._stop_pct_source = stop_pct_source
        # symbol -> {"id", "trigger", "qty"}
        self._stops: dict[str, dict] = {}

    # ---------------- 参数计算 ----------------
    def stop_distance(self) -> float:
        """止损距离（相对持仓均价）：策略软止损 + buffer，缺配置时用默认值。"""
        base = self._default_pct
        if self._stop_pct_source is not None:
            try:
                v = self._stop_pct_source()
                if v and float(v) > 0:
                    base = float(v)
            except Exception as e:  # noqa: BLE001
                log.warning("[stop] 读取策略止损比例失败，用默认值 %s: %s", self._default_pct, e)
        return min(max(base, 0.001), 0.5) + max(self._buffer_pct, 0.0)

    def _round_trigger(self, symbol: str, trigger: float) -> float:
        try:
            rounded = float(self.exchange.price_to_precision(symbol, trigger))
            return rounded or trigger
        except Exception:  # noqa: BLE001  # 规格读不到时原样发出，由交易所裁决
            return trigger

    def _is_contract(self, symbol: str) -> bool:
        """合约（USDT-M 等）需要 reduceOnly，防止兜底单触发后反向开空。"""
        try:
            mkt = self.exchange.market(symbol)
            if mkt is not None and "contract" in mkt:
                return bool(mkt.get("contract"))
        except Exception:  # noqa: BLE001  # 规格不可用时退回统一代码判断
            pass
        # parse_symbol 对现货也会把 settle 补成 quote，不能拿它判合约；
        # ccxt 统一代码里只有衍生品带冒号（BTC/USDT:USDT、BTC/USD:BTC）
        return ":" in symbol

    # ---------------- 主流程 ----------------
    async def sync(self, symbol: str, qty: float, entry: float,
                   ref_price: float = 0.0) -> Optional[dict]:
        """把兜底止损对齐到当前持仓；qty<=0 等价于撤单。返回挂单记录或 None。"""
        if not self.enabled or self.exchange is None:
            return None
        if qty <= 0 or entry <= 0:
            await self.release(symbol)
            return None
        trigger = self._round_trigger(symbol, entry * (1.0 - self.stop_distance()))
        if trigger <= 0:
            log.warning("[stop] 触发价计算异常，本轮不挂 %s: entry=%s", symbol, entry)
            return None
        # 止损卖单触发价必须低于现价，否则挂出去立刻成交（等价于市价平仓）
        if ref_price and trigger >= ref_price:
            log.warning("[stop] 触发价不低于现价，跳过挂单（避免瞬时市价平仓）%s: trigger=%s price=%s",
                        symbol, trigger, ref_price)
            return None
        existing = self._stops.get(symbol)
        if existing and self._aligned(existing, trigger, qty):
            return existing
        if existing and not await self._cancel(symbol, existing):
            # 旧单没能撤掉就先不挂新单——两笔止损会重复卖出同一笔持仓
            log.warning("[stop] 旧止损单撤销失败，保留旧单不重挂 %s: oid=%s", symbol, existing.get("id"))
            return existing
        params: dict = {"triggerPrice": trigger, "clientOrderId": f"{STOP_TAG}{int(time.time() * 1000)}"}
        if self._is_contract(symbol):
            params["reduceOnly"] = True
        try:
            order = await self.exchange.create_order(symbol, "market", "sell", qty, None, params)
        except Exception as e:  # noqa: BLE001
            log.warning("[stop] 兜底止损挂单失败（引擎软止损仍在）%s: %s", symbol, e)
            await self._publish("protective_stop_failed", symbol, trigger=trigger, qty=qty,
                                error=str(e)[:200])
            return None
        oid = str(order.get("id") or "")
        rec = {"id": oid, "trigger": trigger, "qty": qty}
        if oid:
            self._stops[symbol] = rec
        log.info("[stop] 服务端兜底止损已挂 %s: qty=%s trigger=%s（比引擎软止损更深）",
                 symbol, qty, trigger)
        await self._publish("protective_stop_placed", symbol, trigger=trigger, qty=qty, order_id=oid)
        return rec

    async def release(self, symbol: str) -> bool:
        """撤掉该标的的兜底止损（已清仓或引擎要自己卖出时）。

        返回是否真的撤掉了。撤单失败时**保留**本地记录：断网正是这道防线最
        该在的时刻，若此刻丢掉记录再重挂，交易所侧就同时躺着两笔止损，
        同一笔持仓会被卖两次。保留记录后 sync() 会因"对齐且未变"直接复用。
        """
        rec = self._stops.get(symbol)
        if not rec:
            return True
        if await self._cancel(symbol, rec):
            self._stops.pop(symbol, None)
            return True
        return False

    def forget(self, symbol: str, order_id: str = "") -> None:
        """交易所侧订单已了结（触发/被外部撤销）：只清本地记录，不再撤单。"""
        rec = self._stops.get(symbol)
        if rec and (not order_id or str(rec.get("id")) == str(order_id)):
            self._stops.pop(symbol, None)

    async def adopt(self, symbol: str) -> Optional[dict]:
        """重启收养：从交易所挂单中找回本进程此前挂的兜底止损（按 clientOrderId 前缀）。

        找不到返回 None——此时 sync() 会挂一笔新的。找到则避免"撤单+重挂"的裸奔窗口。
        """
        if not self.enabled or self.exchange is None:
            return None
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("[stop] 收养查询挂单失败 %s: %s", symbol, e)
            return None
        for o in open_orders or []:
            cid = str(o.get("clientOrderId") or (o.get("info") or {}).get("clientOrderId") or "")
            if not cid.startswith(STOP_TAG):
                continue
            trigger = float(o.get("triggerPrice") or (o.get("info") or {}).get("stopPrice") or 0.0)
            qty = float(o.get("remaining") or o.get("amount") or 0.0)
            oid = str(o.get("id") or "")
            if not oid or trigger <= 0 or qty <= 0:
                continue
            rec = {"id": oid, "trigger": trigger, "qty": qty}
            self._stops[symbol] = rec
            log.info("[stop] 收养交易所侧兜底止损 %s: oid=%s trigger=%s qty=%s",
                     symbol, oid, trigger, qty)
            return rec
        return None

    def status(self) -> dict:
        return {"enabled": self.enabled, "stop_distance": round(self.stop_distance(), 6),
                "stops": {k: dict(v) for k, v in self._stops.items()}}

    # ---------------- 内部 ----------------
    @staticmethod
    def _aligned(existing: dict, trigger: float, qty: float) -> bool:
        return (abs(float(existing.get("trigger") or 0.0) - trigger) <= trigger * _TRIGGER_TOL
                and abs(float(existing.get("qty") or 0.0) - qty) <= max(qty * _QTY_TOL, 1e-12))

    async def _cancel(self, symbol: str, rec: dict) -> bool:
        oid = str(rec.get("id") or "")
        if not oid:
            return True
        try:
            await self.exchange.cancel_order(oid, symbol)
            log.info("[stop] 兜底止损已撤 %s: oid=%s", symbol, oid)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("[stop] 兜底止损撤单失败 %s oid=%s: %s", symbol, oid, e)
            return False

    async def _publish(self, kind: str, symbol: str, **data) -> None:
        try:
            await self._bus.publish(Event(EventType.SYSTEM,
                                          {"kind": kind, "symbol": symbol, **data},
                                          source="protective_stop"))
        except Exception as e:  # noqa: BLE001
            log.debug("[stop] 事件发布失败: %s", e)
