"""风控规则动态调整 + 冷却状态查询。"""
import math

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from web.deps import get_engine


class RiskIn(BaseModel):
    # allow_inf_nan=False + 显式 isfinite 双保险：JSON NaN/Infinity 可绕过风控钳制
    max_loss_per_trade_usd: float | None = Field(default=None, allow_inf_nan=False)
    max_daily_loss_usd: float | None = Field(default=None, allow_inf_nan=False)
    min_order_value_usd: float | None = Field(default=None, allow_inf_nan=False)
    max_trades_per_hour: int | None = None
    max_position_pct: float | None = Field(default=None, allow_inf_nan=False)
    max_consecutive_losses: int | None = None
    cooldown_minutes: int | None = None


router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/rules")
async def get_rules(engine=Depends(get_engine)):
    return await engine.risk.get_rules()


@router.put("/rules")
async def update_rules(body: RiskIn, engine=Depends(get_engine)):
    payload = body.model_dump(exclude_none=True)
    for k, v in payload.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise HTTPException(status_code=400, detail=f"规则 {k} 必须为有限数值")
    rules = await engine.risk.update_rules(payload)
    return {"ok": True, "rules": rules}


@router.get("/cooldown")
async def cooldown_status(engine=Depends(get_engine)):
    """连续亏损冷却状态（供前端展示/轮询）。"""
    return engine.risk.cooldown_status()