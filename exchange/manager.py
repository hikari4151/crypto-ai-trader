"""ccxt REST 封装：读写权限分离（public 行情 / private 交易），支持代理。"""
import logging
from typing import Any, Optional

from config.settings import settings

log = logging.getLogger(__name__)


class ExchangeManager:
    """统一封装 Binance / OKX / Bybit / Bitget 等 ccxt 支持的交易所。

    - public client: 无密钥，只做行情
    - private client: 带密钥，做账户/下单/撤单
    - 自动注入本地代理（OKX / Bitget 等海外交易所网络受限时仍可连接）
    """

    def __init__(self, exchange_id: str, timeout_ms: int = 10000) -> None:
        self.exchange_id = exchange_id
        self._public: Any = None
        self._private: Any = None
        self._proxy = settings.resolved_proxy or None
        self._timeout_ms = timeout_ms

    def _client_cfg(self, **extra: Any) -> dict[str, Any]:
        cfg: dict[str, Any] = {"enableRateLimit": True, "timeout": self._timeout_ms}
        if self._proxy:
            # REST 用 aiohttp_proxy（ccxt python 特有，仅对 REST 生效）。
            # 注意：不要设置 httpProxy/httpsProxy——它们与 aiohttp_proxy 并存时
            # 会改变 REST 代理路径且对 WebSocket 无效（WS 代理用 wsProxy）
            cfg["aiohttp_proxy"] = self._proxy
        cfg.update(extra)
        return cfg

    async def start(self, api_key: str = "", secret: str = "", password: str = "",
                    load_markets: bool = True, testnet: bool = False) -> None:
        """初始化客户端。load_markets=False 用于只拉钱包/余额的场景（跳过市场加载，更快）。

        testnet=True 时对所有已建客户端调用 ccxt set_sandbox_mode(True)：
        - 实盘就绪验证（GAP-02）：在交易所 sandbox/testnet 上演练真实订单生命周期，
          不碰真实资金；
        - 不支持 sandbox 的交易所显式报错，避免"以为在沙箱实际打真单"。
        """
        import ccxt.async_support as ccxt
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"不支持的交易所: {self.exchange_id}")
        self._public = getattr(ccxt, self.exchange_id)(self._client_cfg())
        if testnet:
            try:
                self._public.set_sandbox_mode(True)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"{self.exchange_id} 不支持 testnet/sandbox: {e}") from e
        if load_markets:
            await self._public.load_markets()
        log.info("[%s] public client 就绪%s%s", self.exchange_id,
                 f"（代理 {self._proxy}）" if self._proxy else "",
                 "（testnet/sandbox）" if testnet else "")
        if api_key:
            cfg = self._client_cfg(apiKey=api_key, secret=secret)
            if password:
                cfg["password"] = password
            self._private = getattr(ccxt, self.exchange_id)(cfg)
            if testnet:
                try:
                    self._private.set_sandbox_mode(True)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"{self.exchange_id} 不支持 testnet/sandbox（private）: {e}") from e
            if load_markets:
                await self._private.load_markets()
            log.info("[%s] private client 就绪（已注入密钥，绝不写日志%s）", self.exchange_id,
                     "，testnet/sandbox" if testnet else "")

    @property
    def has_private(self) -> bool:
        return self._private is not None

    async def close(self) -> None:
        for c in (self._public, self._private):
            if c is not None:
                try:
                    await c.close()
                except Exception:  # noqa: BLE001
                    pass

    # ---------- 行情（public） ----------
    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._public.fetch_ticker(symbol)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", since: Optional[int] = None, limit: int = 500) -> list:
        return await self._public.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    # ---------- 交易（private） ----------
    def _require_private(self) -> Any:
        if self._private is None:
            raise RuntimeError("未配置交易密钥（只读模式）")
        return self._private

    async def fetch_balance(self) -> dict:
        return await self._require_private().fetch_balance()

    async def create_order(self, symbol: str, otype: str, side: str, amount: float,
                           price: Optional[float] = None, params: Optional[dict] = None) -> dict:
        return await self._require_private().create_order(symbol, otype, side, amount, price, params or {})

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._require_private().cancel_order(order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._require_private().fetch_order(order_id, symbol)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list:
        return await self._require_private().fetch_open_orders(symbol)

    async def fetch_positions(self, symbols: Optional[list] = None) -> list:
        return await self._require_private().fetch_positions(symbols)

    async def fetch_algo_open_orders(self, symbol: Optional[str] = None) -> list:
        """条件单/算法单（OKX 的 trigger/conditional/oco 等）——ccxt 统一 `fetch_open_orders`
        **不包含**它们（实测：OKX 兜底止损挂单成功但 `fetch_open_orders` 返回 0 条）。

        影响：`ProtectiveStopManager.adopt()` 重启收养、`_reconcile_once()` 对账都会
        误判“止损单不存在” → 重复挂单/裸奔窗口。此方法把 algo 挂单补进可见范围。

        其他交易所无对应实现时返回 []（不影响既有路径）。
        """
        if self.exchange_id != "okx" or self._private is None:
            return []
        out: list = []
        for ot in ("trigger", "conditional", "oco"):
            try:
                r = await self._private.private_get_trade_orders_algo_pending(
                    {"instType": "SPOT", "ordType": ot})
            except Exception as e:  # noqa: BLE001  # 单个 ordType 不支持不影响其它
                log.debug("[%s] algo 挂单查询失败 ordType=%s: %s", self.exchange_id, ot, e)
                continue
            for it in (r or {}).get("data") or []:
                inst = str(it.get("instId") or "")
                sym = inst.replace("-", "/") if inst else ""
                if symbol and sym and sym != symbol:
                    continue
                out.append({
                    "id": str(it.get("algoId") or ""),
                    "clientOrderId": it.get("algoClOrdId") or it.get("clOrdId") or "",
                    "symbol": sym, "side": it.get("side"),
                    "amount": float(it.get("sz") or 0.0), "remaining": float(it.get("sz") or 0.0),
                    "status": "open", "type": "trigger",
                    "triggerPrice": it.get("slTriggerPx") or it.get("triggerPx") or "",
                    "info": it,
                })
        return out

    # ---------- 市场规格（下单前精度截断 / 最小名义额校验用） ----------
    def _spec_client(self) -> Any:
        client = self._private or self._public
        if client is None:
            raise RuntimeError("交易所客户端未初始化")
        return client

    def market(self, symbol: str) -> dict:
        """ccxt 市场规格（含 precision/limits）；未 load_markets 或未知 symbol 会抛错。"""
        return self._spec_client().market(symbol)

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return self._spec_client().amount_to_precision(symbol, amount)

    def price_to_precision(self, symbol: str, price: float) -> str:
        return self._spec_client().price_to_precision(symbol, price)
