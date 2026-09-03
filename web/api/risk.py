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
    # 冷却到期后是否需人工确认（0=自动恢复, 1=人工）—— 此前未声明，界面那个开关永远存不进去
    risk_manual_recovery: int | None = None
    # 闪崩/急拉/量能保护：RiskIn 未声明时 pydantic 会直接丢弃，engine.risk.update_rules 其实早已支持这些键
    flash_crash_5min_drop_pct: float | None = Field(default=None, allow_inf_nan=False)
    flash_crash_24h_drop_pct: float | None = Field(default=None, allow_inf_nan=False)
    flash_rally_5min_rise_pct: float | None = Field(default=None, allow_inf_nan=False)
    flash_cooldown_minutes: int | None = None
    max_vol_ratio: float | None = Field(default=None, allow_inf_nan=False)
    # 波动率目标仓位
    vol_target_enabled: float | None = Field(default=None, allow_inf_nan=False)
    vol_target_annual_pct: float | None = Field(default=None, allow_inf_nan=False)
    vol_target_lookback_hours: float | None = Field(default=None, allow_inf_nan=False)


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


@router.post("/clear-cooldown")
async def clear_cooldown(engine=Depends(get_engine)):
    """人工确认解除连续亏损冷却（冷却到期且开启人工确认模式时）。"""
    engine.risk.clear_cooldown()
    # 落库否则重启后"已人工确认"凭空失效，熔断再次挂起
    await engine.risk.persist_state()
    return {"ok": True, "cooldown": engine.risk.cooldown_status()}