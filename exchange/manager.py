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
                    load_markets: bool = True) -> None:
        """初始化客户端。load_markets=False 用于只拉钱包/余额的场景（跳过市场加载，更快）。"""
        import ccxt.async_support as ccxt
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"不支持的交易所: {self.exchange_id}")
        self._public = getattr(ccxt, self.exchange_id)(self._client_cfg())
        if load_markets:
            await self._public.load_markets()
        log.info("[%s] public client 就绪%s", self.exchange_id, f"（代理 {self._proxy}）" if self._proxy else "")
        if api_key:
            cfg = self._client_cfg(apiKey=api_key, secret=secret)
            if password:
                cfg["password"] = password
            self._private = getattr(ccxt, self.exchange_id)(cfg)
            if load_markets:
                await self._private.load_markets()
            log.info("[%s] private client 就绪（已注入密钥，绝不写日志）", self.exchange_id)

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