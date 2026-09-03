"""钱包余额与持仓。

- 纸面模式：返回本地模拟账户
- 实盘模式：即使引擎未启动，也动态连接交易所 API 返回真实钱包数据
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# 实盘钱包缓存（5 秒）：避免前端轮询时反复连交易所
_live_cache: dict = {"ts": 0.0, "data": None}
_live_cache_lock = asyncio.Lock()


@router.get("/summary")
async def summary(db=Depends(get_db), engine=Depends(get_engine)):
    try:
        # 实盘模式优先：从 KV 读 mode + 配置
        mode = await db.kv_get("trading_mode", "paper")
        if mode == "live":
            cfg = await db.kv_json_get("live_trading_config") or {}
            exchange_id = cfg.get("exchange", "binance")
            api_key = await db.kv_get_secret(f"exchange_{exchange_id}_api_key") or ""
            secret = await db.kv_get_secret(f"exchange_{exchange_id}_secret") or ""
            password = await db.kv_get_secret(f"exchange_{exchange_id}_password") or ""
            if not api_key or not secret:
                # 未配置密钥但实盘模式：给出提示
                return {"equity": None, "cash": None, "positions_value": None,
                        "positions": [], "error": f"实盘模式需要配置 {exchange_id} 的 API Key/Secret"}
            # 5 秒缓存：避免前端轮询反复连交易所
            import time
            now = time.time()
            async with _live_cache_lock:
                if _live_cache["data"] and now - _live_cache["ts"] < 5:
                    return _live_cache["data"]
                result = await _fetch_live_balance(exchange_id, api_key, secret, password)
                _live_cache.update({"ts": now, "data": result})
                return result
        # 纸面模式：引擎内部快照（只读不落库，落库由引擎 15s 快照循环负责）
        return await engine.portfolio.snapshot(persist=False)
    except Exception as e:  # noqa: BLE001
        log.exception("[portfolio] 获取资产失败")
        return {"equity": None, "cash": None, "positions_value": None,
                "positions": [], "error": f"获取资产失败: {e}"}


async def _fetch_live_balance(exchange_id: str, api_key: str, secret: str, password: str) -> dict:
    """从交易所 API 拉取真实钱包数据，折算 USDT 展示。

    先做 socket 层快速可达性探测（~1s），连不通立即返回友好提示；
    之后才发起真正的钱包请求（总时长限制）。
    """
    from exchange.manager import ExchangeManager
    # socket 层探测：快速判断交易所网络是否可达（避免 aiohttp 长阻塞）
    if not await _socket_probe(exchange_id):
        log.warning("[portfolio] 实盘交易所不可达: %s", exchange_id)
        return {"equity": None, "cash": None, "positions_value": None,
                "positions": [],
                "error": f"{exchange_id} 网络不可达，请确认本地代理（FlClash/Clash）已开启"}
    mgr = ExchangeManager(exchange_id, timeout_ms=5000)
    try:
        return await asyncio.wait_for(
            _do_fetch(mgr, api_key, secret, password), timeout=8.0)
    except asyncio.TimeoutError:
        log.warning("[portfolio] 实盘钱包拉取超时")
        return {"equity": None, "cash": None, "positions_value": None,
                "positions": [], "error": "实盘钱包拉取超时（请确认本地代理已开启）"}
    except Exception as e:  # noqa: BLE001
        err = str(e)
        hint = "网络受限，请确认本地代理（FlClash/Clash）已开启" if (
            "timeout" in err.lower() or "RequestTimeout" in type(e).__name__
            or "proxy" in err.lower() or "Connection" in err
        ) else f"拉取失败: {err[:120]}"
        log.warning("[portfolio] 实盘钱包拉取失败: %s", err)
        return {"equity": None, "cash": None, "positions_value": None,
                "positions": [], "error": hint}
    finally:
        try:
            await mgr.close()
        except Exception:  # noqa: BLE001
            pass


_HOST_MAP = {
    "binance": ("data-api.binance.vision", 443),
    "okx": ("www.okx.com", 443),
    "bybit": ("api.bybit.com", 443),
    "bitget": ("api.bitget.com", 443),
}


async def _socket_probe(exchange_id: str) -> bool:
    """用 socket 快速探测交易所 host:443 连通性（1s 超时）。"""
    import socket
    host, port = _HOST_MAP.get(exchange_id, (f"api.{exchange_id}.com", 443))
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(loop.run_in_executor(None, _socket_connect, host, port), timeout=1.2)
        return True
    except Exception:  # noqa: BLE001
        return False


def _socket_connect(host: str, port: int) -> None:
    import socket
    s = socket.create_connection((host, port), timeout=1.0)
    s.close()


async def _do_fetch(mgr, api_key: str, secret: str, password: str) -> dict:
    """连接交易所并拉取钱包。close 由调用方负责。

    跳过 load_markets：钱包拉取不需要完整市场信息，显著减少耗时与失败点。
    通过 PortfolioManager.fetch_live_balance 复用引擎侧同一余额口径。
    """
    from engine.portfolio import PortfolioManager
    await mgr.start(api_key, secret, password, load_markets=False)
    # 用临时 PortfolioManager 包装 mgr，复用 fetch_live_balance 的缓存与异常处理
    pm = PortfolioManager(None, None, False, None, mgr)
    bal = await pm.fetch_live_balance(api_key, secret, password)
    positions = []
    cash = 0.0
    positions_value = 0.0
    total = bal.get("total", {})
    for asset, qty in total.items():
        if not qty:
            continue
        qty = float(qty)
        if asset == "USDT":
            # 计价币：只计现金，不进持仓（对齐 engine/portfolio.py:56-59 口径；
            # 曾同时追加 positions 条目并计入 positions_value → equity 双计虚高一份 USDT）
            cash += qty
            continue
        symbol = f"{asset}/USDT"
        price = await _safe_price(mgr, symbol)
        if not price:
            continue
        value = qty * price
        positions_value += value
        positions.append({
            "symbol": symbol, "qty": round(qty, 8), "avg_price": 0.0,
            "last_price": round(price, 6), "value": round(value, 4), "unrealized_pnl": 0.0,
        })
    equity = cash + positions_value
    return {"equity": round(equity, 4), "cash": round(cash, 4),
            "positions_value": round(positions_value, 4), "positions": positions,
            "source": "live_exchange"}


async def _safe_price(mgr, symbol: str) -> float | None:
    try:
        t = await mgr.fetch_ticker(symbol)
        return float(t.get("last") or 0.0) or None
    except Exception:  # noqa: BLE001
        return None
