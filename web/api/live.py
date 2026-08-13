"""实盘启动设置：交易模式切换（纸面/实盘）、实盘参数配置、启动状态。"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.database import Database
from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/live", tags=["live"])


class LiveConfigIn(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_cash: float = 10000.0


class PaperConfigIn(BaseModel):
    start_cash: float = 10000.0
    fee_rate: float = 0.001


class ModeIn(BaseModel):
    mode: str          # "paper" | "live"


@router.get("/config")
async def get_live_config(db: Database = Depends(get_db)):
    """读取实盘启动配置。"""
    cfg = await db.kv_json_get("live_trading_config") or {}
    mode = await db.kv_get("trading_mode", "paper")
    return {
        "mode": mode,
        "exchange": cfg.get("exchange", "binance"),
        "symbol": cfg.get("symbol", "BTC/USDT"),
        "timeframe": cfg.get("timeframe", "1h"),
        "start_cash": float(cfg.get("start_cash", 10000.0)),
    }


@router.put("/config")
async def save_live_config(body: LiveConfigIn, db: Database = Depends(get_db)):
    """保存实盘启动参数（下次启动生效）。"""
    await db.kv_json_set("live_trading_config", body.model_dump())
    return {"ok": True, "message": "实盘配置已保存"}


@router.get("/paper-config")
async def get_paper_config(db: Database = Depends(get_db)):
    """读取模拟模式配置（初始资金 + 手续费率）。"""
    cfg = await db.kv_json_get("paper_trading_config") or {}
    return {
        "start_cash": float(cfg.get("start_cash", 10000.0)),
        "fee_rate": float(cfg.get("fee_rate", 0.001)),
    }


@router.put("/paper-config")
async def set_paper_config(body: PaperConfigIn, db: Database = Depends(get_db), engine=Depends(get_engine)):
    """保存模拟模式配置（下次启动引擎生效）。"""
    if engine.running:
        raise HTTPException(status_code=400, detail="请先停止交易引擎再修改模拟配置")
    if body.start_cash <= 0:
        raise HTTPException(status_code=400, detail="初始资金必须大于 0")
    if body.fee_rate < 0 or body.fee_rate > 0.1:
        raise HTTPException(status_code=400, detail="手续费率需在 0 ~ 0.1 之间")
    await db.kv_json_set("paper_trading_config", body.model_dump())
    # 同步到引擎，下次启动生效
    engine.start_cash = body.start_cash
    engine.paper_fee_rate = body.fee_rate
    return {"ok": True, "message": "模拟配置已保存"}


@router.put("/mode")
async def set_mode(body: ModeIn, db: Database = Depends(get_db), engine=Depends(get_engine)):
    """切换交易模式（paper 纸面 / simulated 模拟实盘 / live 实盘）。切换前必须先停止引擎。"""
    if body.mode not in ("paper", "simulated", "live"):
        raise HTTPException(status_code=400, detail="mode 必须为 paper、simulated 或 live")
    if engine.running:
        raise HTTPException(status_code=400, detail="请先停止交易引擎再切换模式")
    if body.mode == "live":
        # 实盘模式必须有交易所密钥
        cfg = await db.kv_json_get("live_trading_config") or {}
        ex = cfg.get("exchange", "binance")
        key = await db.kv_get_secret(f"exchange_{ex}_api_key") or ""
        secret = await db.kv_get_secret(f"exchange_{ex}_secret") or ""
        if not key or not secret:
            raise HTTPException(status_code=400, detail=f"实盘需要先配置 {ex} 的 API Key/Secret")
    await db.kv_set("trading_mode", body.mode)
    return {"ok": True, "mode": body.mode}


@router.get("/status")
async def live_status(db: Database = Depends(get_db), engine=Depends(get_engine)):
    """当前模式与引擎状态。

    注意：mode 和实盘参数优先读 KV（用户已保存的配置），
    这样切换模式后立即生效，不依赖引擎是否启动。
    """
    mode = await db.kv_get("trading_mode", "paper")
    cfg = await db.kv_json_get("live_trading_config") or {}
    paper_cfg = await db.kv_json_get("paper_trading_config") or {}
    # 初始资金按模式取对应配置源（与引擎 _load_live_config 口径一致）：
    # live/simulated 用实盘配置（模拟实盘=验证实盘参数），paper 用模拟配置
    start_cash = float((cfg if mode != "paper" else paper_cfg).get("start_cash", engine.start_cash))
    return {
        "mode": mode,
        "running": engine.running,
        "exchange": cfg.get("exchange", engine.exchange_id),
        "symbol": cfg.get("symbol", engine.symbol),
        "timeframe": cfg.get("timeframe", engine.timeframe),
        "start_cash": start_cash,
        "paper": mode != "live",
    }
