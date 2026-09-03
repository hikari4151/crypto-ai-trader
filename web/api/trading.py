"""交易引擎启停 / 策略选择 / 参数热更新。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from web.deps import get_engine

router = APIRouter(prefix="/api/trading", tags=["trading"])


class StrategyIn(BaseModel):
    name: str


class ParamsIn(BaseModel):
    params: dict


@router.post("/start")
async def start(engine=Depends(get_engine)):
    await engine.start()
    return {"ok": True, "status": engine.status()}


@router.post("/stop")
async def stop(engine=Depends(get_engine)):
    await engine.stop()
    return {"ok": True, "status": engine.status()}


@router.get("/status")
async def status(engine=Depends(get_engine)):
    return engine.status()


@router.get("/strategies")
async def strategies(engine=Depends(get_engine)):
    # 内置 + 动态全量返回：回测/实盘下拉与「一键应用内置策略」流程都依赖本列表。
    # 曾过滤掉内置策略——回测页下拉里根本没有默认值 dual_ma，且应用内置策略后
    # loadStrategy 找不到 current，会静默跳到列表第一个其它策略。
    out = engine.list_strategies()
    # 当前策略用实例的实际参数（热更新后的），而不是默认参数
    for s in out:
        if s["name"] == engine.strategy.name:
            s["current_params"] = dict(engine.strategy.params)
            s["default_params"] = dict(engine.strategy.params)
    return {"current": engine.strategy.name, "strategies": out}


@router.post("/strategies/select")
async def select_strategy(body: StrategyIn, engine=Depends(get_engine)):
    try:
        result = await engine.select_strategy(body.name)
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/strategies/params")
async def update_params(body: ParamsIn, engine=Depends(get_engine)):
    return {"ok": True, **await engine.update_strategy_params(body.params)}