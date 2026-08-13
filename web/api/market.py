"""实时行情 API（公开免费数据源）：K线、热门币种、涨跌幅榜。

设计要点：
- 免密钥：直连交易所公开 REST 接口（Binance / OKX / Bybit / Bitget）
- 走代理：自动使用 settings.resolved_proxy（解决海外接口访问受限）
- 后端聚合：前端不直连交易所，避免跨域与密钥暴露
- TTL 缓存：tickers 是全量拉取（请求重），缓存 3 秒让前端可秒级轮询而不触发限频
"""
import logging
import threading
import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException

from config.settings import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market", tags=["market"])

# ---------- 轻量 TTL 内存缓存（避免高频轮询打爆交易所限频） ----------
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_TTL_DEFAULT = 3.0   # tickers/overview 缓存 3 秒（支持前端秒级刷新）
_TTL_KLINES = 5.0    # K线响应较大，缓存 5 秒


def _cache_get(key: str, ttl: float = _TTL_DEFAULT) -> Optional[Any]:
    """读取缓存；ttl 按条目实际设置的过期时长判断（klines 用更长 TTL）。"""
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and time.time() - item[0] < ttl:
            return item[1]
    return None


def _cache_set(key: str, value: Any, ttl: float = _TTL_DEFAULT) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)
    # 防内存膨胀：超 500 条清掉过期项
    if len(_CACHE) > 500:
        now = time.time()
        with _CACHE_LOCK:
            expired = [k for k, v in _CACHE.items() if now - v[0] > _TTL_DEFAULT * 4]
            for k in expired:
                _CACHE.pop(k, None)


# 各交易所公开 REST 端点
# klines: 返回 [ts_ms, open, high, low, close, volume]
# tickers: 返回 {symbol, last, change_pct, quote_volume}
# 说明：Binance 默认用官方 .vision 数据域（中国大陆可直连）；其他交易所走本地代理
_BASES = {
    "binance": "https://data-api.binance.vision/api/v3",
    "okx": "https://www.okx.com/api/v5/market",
    "bybit": "https://api.bybit.com/v5/market",
    "bitget": "https://api.bitget.com/api/v2/spot/market",
}

_SYMBOL_MAP = {
    "binance": lambda s: s.upper().replace("-", "").replace("/", ""),
    "okx": lambda s: s.upper().replace("/", "-"),
    "bybit": lambda s: s.upper().replace("/", ""),
    "bitget": lambda s: s.upper().replace("/", ""),
}

_TIME_MAP = {
    "binance": lambda t: t,
    "okx": {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"},
    "bybit": {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D", "1w": "W"},
    "bitget": {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week"},
}

SUPPORTED = list(_BASES.keys())


_client: Optional[httpx.AsyncClient] = None
_client_lock = threading.Lock()


def _get_client() -> httpx.AsyncClient:
    """进程内复用 AsyncClient（TCP+TLS 握手复用）；代理配置变化时惰性重建。"""
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            proxy = settings.resolved_proxy or None
            _client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0), proxy=proxy)
        return _client


async def _fetch_json(url: str, params: Optional[dict] = None) -> Any:
    try:
        client = _get_client()
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"数据源 {e.response.status_code}: {e.response.text[:200]}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"数据源不可达: {e}")
    except Exception as e:  # noqa: BLE001
        log.exception("[market] fetch 失败 %s", url)
        raise HTTPException(status_code=502, detail=f"行情获取失败: {e}")


def _norm_symbol(exchange: str, symbol: str) -> str:
    if exchange not in _SYMBOL_MAP:
        raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
    return _SYMBOL_MAP[exchange](symbol)


# ---------- K线 ----------
async def _klines_binance(symbol: str, timeframe: str, limit: int) -> list:
    data = await _fetch_json(f"{_BASES['binance']}/klines", {
        "symbol": symbol, "interval": _TIME_MAP["binance"](timeframe), "limit": limit})
    return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in data]


async def _klines_okx(symbol: str, timeframe: str, limit: int) -> list:
    data = await _fetch_json(f"{_BASES['okx']}/candles", {
        "instId": symbol, "bar": _TIME_MAP["okx"].get(timeframe, "1H"), "limit": limit})
    rows = data.get("data", [])
    # OKX 返回从新到旧，需反转
    return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in reversed(rows)]


async def _klines_bybit(symbol: str, timeframe: str, limit: int) -> list:
    data = await _fetch_json(f"{_BASES['bybit']}/kline", {
        "category": "spot", "symbol": symbol, "interval": _TIME_MAP["bybit"].get(timeframe, "60"), "limit": limit})
    rows = data.get("result", {}).get("list", [])
    # Bybit 从新到旧
    return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in reversed(rows)]


async def _klines_bitget(symbol: str, timeframe: str, limit: int) -> list:
    data = await _fetch_json(f"{_BASES['bitget']}/candles", {
        "symbol": symbol, "granularity": _TIME_MAP["bitget"].get(timeframe, "1h"), "limit": limit})
    rows = data.get("data", [])
    # Bitget candles 字段顺序: [open, high, low, close, volume, ts, ...]
    return [[int(c[5]), float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])] for c in rows]


async def _klines(exchange: str, symbol: str, timeframe: str, limit: int) -> list:
    if exchange == "binance":
        return await _klines_binance(symbol, timeframe, limit)
    if exchange == "okx":
        return await _klines_okx(symbol, timeframe, limit)
    if exchange == "bybit":
        return await _klines_bybit(symbol, timeframe, limit)
    if exchange == "bitget":
        return await _klines_bitget(symbol, timeframe, limit)
    raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")


@router.get("/klines")
async def klines(exchange: str = "binance", symbol: str = "BTC/USDT",
                 timeframe: str = "1h", limit: int = 300):
    if exchange not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}，可选 {SUPPORTED}")
    # binance 的周期是直接透传；其余交易所用映射表校验
    allowed = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if timeframe not in allowed:
        raise HTTPException(status_code=400, detail=f"周期 {timeframe} 不支持")
    if isinstance(_TIME_MAP[exchange], dict) and timeframe not in _TIME_MAP[exchange]:
        raise HTTPException(status_code=400, detail=f"周期 {timeframe} 在 {exchange} 不支持")
    limit = max(10, min(int(limit), 1000))
    nsym = _norm_symbol(exchange, symbol)
    cache_key = f"kl|{exchange}|{nsym}|{timeframe}|{limit}"
    rows = _cache_get(cache_key, _TTL_KLINES)
    if rows is None:
        rows = await _klines(exchange, nsym, timeframe, limit)
        _cache_set(cache_key, rows, _TTL_KLINES)
    return {"exchange": exchange, "symbol": symbol, "timeframe": timeframe,
            "klines": rows, "updated_at": int(time.time() * 1000)}


# ---------- Ticker / 涨跌幅榜 ----------
def _norm_tickers(exchange: str, raw: Any) -> list[dict]:
    """把各交易所原始 ticker 响应统一成 [{symbol,last,change_pct,quote_volume}]。"""
    out: list[dict] = []
    if exchange == "binance":
        for t in raw or []:
            out.append({
                "symbol": t.get("symbol", ""), "last": float(t.get("lastPrice", 0) or 0),
                "change_pct": float(t.get("priceChangePercent", 0) or 0),
                "quote_volume": float(t.get("quoteVolume", 0) or 0),
            })
    elif exchange == "okx":
        for t in (raw or {}).get("data", []):
            out.append({
                "symbol": t.get("instId", ""), "last": float(t.get("last", 0) or 0),
                "change_pct": float(t.get("changePerc24h", 0) or 0),
                "quote_volume": float(t.get("volCcy24h", 0) or 0),
            })
    elif exchange == "bybit":
        for t in ((raw or {}).get("result", {}) or {}).get("list", []):
            out.append({
                "symbol": t.get("symbol", ""), "last": float(t.get("lastPrice", 0) or 0),
                "change_pct": float(t.get("price24hPcnt", 0) or 0) * 100,
                "quote_volume": float(t.get("turnover24h", 0) or 0),
            })
    elif exchange == "bitget":
        for t in (raw or {}).get("data", []):
            out.append({
                "symbol": t.get("symbol", ""), "last": float(t.get("lastPr", 0) or 0),
                "change_pct": float(t.get("changePct", 0) or 0),
                "quote_volume": float(t.get("quoteVolume", 0) or 0),
            })
    return out


@router.get("/tickers")
async def tickers(exchange: str = "binance", limit: int = 60):
    if exchange not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
    limit = max(5, min(int(limit), 200))
    cache_key = f"tk|{exchange}|{limit}"
    cached = _cache_get(cache_key)
    if cached:
        return {**cached, "updated_at": int(time.time() * 1000)}
    if exchange == "binance":
        raw = await _fetch_json(f"{_BASES['binance']}/ticker/24hr")
    elif exchange == "okx":
        raw = await _fetch_json(f"{_BASES['okx']}/tickers", {"instType": "SPOT"})
    elif exchange == "bybit":
        raw = await _fetch_json(f"{_BASES['bybit']}/tickers", {"category": "spot"})
    elif exchange == "bitget":
        raw = await _fetch_json(f"{_BASES['bitget']}/tickers")
    else:
        raise HTTPException(status_code=400, detail="不支持的交易所")
    items = _norm_tickers(exchange, raw)
    # 只保留主流币（以 USDT 计价且不是杠杆/衍生品），并过滤稳定币对
    if exchange == "okx":
        items = [i for i in items if i["symbol"].endswith("-USDT")]
        stables = ("USDC-USDT", "FDUSD-USDT", "TUSD-USDT", "DAI-USDT", "USDP-USDT", "PAX-USDT")
        items = [i for i in items if i["symbol"] not in stables]
    else:
        items = [i for i in items if i["symbol"].endswith("USDT") and i["symbol"].count("/") == 0]
        stables = ("USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT", "USDPUSDT", "PAXUSDT", "AEURUSDT")
        items = [i for i in items if i["symbol"] not in stables]
    # 按 24h 成交额排序取前 limit 名
    items.sort(key=lambda x: x["quote_volume"], reverse=True)
    result = {"exchange": exchange, "tickers": items[:limit]}
    _cache_set(cache_key, result)
    return {**result, "updated_at": int(time.time() * 1000)}


@router.get("/overview")
async def overview(exchange: str = "binance"):
    """热门币种 24h 行情总览（涨跌榜），供市场页首屏。"""
    if exchange not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
    data = await tickers(exchange, limit=40)
    tickers_list = data["tickers"]
    return {
        "exchange": exchange,
        "top_gainers": sorted([t for t in tickers_list if t["change_pct"] > 0],
                              key=lambda x: x["change_pct"], reverse=True)[:8],
        "top_losers": sorted([t for t in tickers_list if t["change_pct"] < 0],
                             key=lambda x: x["change_pct"])[:8],
        "top_volume": sorted(tickers_list, key=lambda x: x["quote_volume"], reverse=True)[:8],
    }
