"""交易所密钥管理：读写权限分离配置，Key 加密存储，支持界面修改与连接测试。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.settings import settings
from core.database import Database
from core.security import mask
from web.deps import get_db

router = APIRouter(prefix="/api/exchanges", tags=["exchanges"])


class ExchangeCredIn(BaseModel):
    exchange: str
    api_key: str = ""
    secret: str = ""
    password: str = ""
    read_only: bool = False   # 预留：仅填 Key 时自动只读


@router.get("/settings")
async def get_exchange_settings(db: Database = Depends(get_db)):
    out = {}
    for ex in ("binance", "okx", "bybit", "bitget"):
        key = await db.kv_get_secret(f"exchange_{ex}_api_key") or ""
        out[ex] = {
            "has_api_key": bool(key),
            "api_key_masked": mask(key),
            "has_secret": bool(await db.kv_get_secret(f"exchange_{ex}_secret")),
            "has_password": bool(await db.kv_get_secret(f"exchange_{ex}_password")),
        }
    return out


@router.put("/settings")
async def save_exchange_settings(body: ExchangeCredIn, db: Database = Depends(get_db)):
    prefix = f"exchange_{body.exchange}"
    if body.api_key:
        await db.kv_set(f"{prefix}_api_key", body.api_key, is_secret=True)
    if body.secret:
        await db.kv_set(f"{prefix}_secret", body.secret, is_secret=True)
    if body.password:
        await db.kv_set(f"{prefix}_password", body.password, is_secret=True)
    return {"ok": True, "message": f"{body.exchange} 密钥已加密保存"}


@router.post("/test")
async def test_exchange(body: ExchangeCredIn, db: Database = Depends(get_db)):
    from exchange.manager import ExchangeManager
    api_key = body.api_key or await db.kv_get_secret(f"exchange_{body.exchange}_api_key") or ""
    secret = body.secret or await db.kv_get_secret(f"exchange_{body.exchange}_secret") or ""
    password = body.password or await db.kv_get_secret(f"exchange_{body.exchange}_password") or ""
    if not api_key or not secret:
        raise HTTPException(status_code=400, detail="缺少 API Key / Secret")
    mgr = ExchangeManager(body.exchange)
    proxy = settings.resolved_proxy or ""
    try:
        await mgr.start(api_key, secret, password)
        bal = await mgr.fetch_balance()
        total = bal.get("total", {})
        non_zero = {k: v for k, v in total.items() if v}
        return {"ok": True, "message": "连接成功", "assets": non_zero,
                "proxy": proxy or None}
    except Exception as e:  # noqa: BLE001
        # 网络类错误：给用户明确提示（OKX/Bitget/Bybit 海外交易所通常需要本地代理）
        err = str(e)
        etype = type(e).__name__
        is_network = any(k in etype for k in (
            "RequestTimeout", "ConnectionError", "ConnectTimeout", "ReadTimeout",
            "TransportError", "NetworkError", "DDoSProtection", "ExchangeNotAvailable",
        ))
        if body.exchange in ("okx", "bitget", "bybit") and is_network:
            hint = ("；已自动使用本地代理 " + proxy) if proxy else "；未检测到可用本地代理（FlClash/Clash 默认 7890 端口）"
            raise HTTPException(
                status_code=502,
                detail=f"连接失败（网络受限）{hint}。请确认：① FlClash/V2Ray 代理已开启并处于连接状态 ② 代理端口与 .env 的 PROXY_URL 一致 ③ 交易所服务器可达。原始错误: {err}",
            )
        raise HTTPException(status_code=502, detail=f"连接失败: {err}")
    finally:
        await mgr.close()