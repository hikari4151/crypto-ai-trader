"""风控规则动态调整 + 冷却状态查询。"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from web.deps import get_engine


class RiskIn(BaseModel):
    max_loss_per_trade_usd: float | None = None
    max_daily_loss_usd: float | None = None
    min_order_value_usd: float | None = None
    max_trades_per_hour: int | None = None
    max_position_pct: float | None = None
    max_consecutive_losses: int | None = None
    cooldown_minutes: int | None = None


router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/rules")
async def get_rules(engine=Depends(get_engine)):
    return await engine.risk.get_rules()


@router.put("/rules")
async def update_rules(body: RiskIn, engine=Depends(get_engine)):
    rules = await engine.risk.update_rules(body.model_dump(exclude_none=True))
    return {"ok": True, "rules": rules}


@router.get("/cooldown")
async def cooldown_status(engine=Depends(get_engine)):
    """连续亏损冷却状态（供前端展示/轮询）。"""
    return engine.risk.cooldown_status()