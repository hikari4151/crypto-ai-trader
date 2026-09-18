"""交易引擎启停 / 策略选择 / 参数热更新。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from web.deps import get_engine

router = APIRouter(prefix="/api/trading", tags=["trading"])


class StrategyIn(BaseModel):
    name: str


class ParamsIn(BaseModel):
    params: dict


class CloseAllIn(BaseModel):
    reason: str = "manual_close_all"


@router.post("/start")
async def start(engine=Depends(get_engine)):
    await engine.start()
    return {"ok": True, "status": engine.status()}


@router.post("/stop")
async def stop(engine=Depends(get_engine)):
    await engine.stop()
    return {"ok": True, "status": engine.status()}


@router.post("/close-all")
async def close_all(body: CloseAllIn, engine=Depends(get_engine)):
    """一键清仓（GAP-05）：引擎运行时按当前持仓市价卖出，绕开风控新开仓限制。

    注意：停引擎≠清仓；本接口必须在引擎运行时调用，用于人工紧急离场。
    """
    if not engine.running:
        raise HTTPException(status_code=400, detail="引擎未运行，无法平仓（停引擎≠清仓）")
    return await engine.close_all_positions(body.reason)


@router.get("/status")
async def status(engine=Depends(get_engine)):
    return engine.status()


@router.get("/alerts")
async def alerts(engine=Depends(get_engine)):
    """运行告警摘要（GAP-07 运维）：最近告警 + 分类计数 + CRITICAL 数。

    数据源：engine.monitor（断链/下单响应丢失/无保护持仓/风控冷却/总线降级）。
    完整历史见 data/logs/alerts.log（结构化 JSON，可被运维采集）。
    """
    mon = getattr(engine, "monitor", None)
    if mon is None:
        return {"counts": {}, "recent": [], "critical_recent": 0}
    return mon.summary()


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