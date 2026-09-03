"""持续进化引擎控制 API。"""
from datetime import timezone

from fastapi import APIRouter, HTTPException, Request

from web.deps import get_engine

router = APIRouter(prefix="/api/evolve", tags=["evolve"])


def _get_evolve(request: Request):
    """从请求 app.state 获取 EvolveEngine 实例。"""
    engine = request.app.state.engine
    if not hasattr(engine, "evolve"):
        raise HTTPException(status_code=404, detail="持续进化引擎未初始化")
    return engine.evolve


@router.post("/enable")
async def enable(request: Request):
    """启用持续进化。"""
    evolve = _get_evolve(request)
    await evolve.set_enabled(True)
    return {"ok": True, "enabled": True}


@router.post("/disable")
async def disable(request: Request):
    """禁用持续进化。"""
    evolve = _get_evolve(request)
    await evolve.set_enabled(False)
    return {"ok": True, "enabled": False}


@router.post("/resume")
async def resume(request: Request):
    """恢复持续训练（取消暂停状态）。"""
    evolve = _get_evolve(request)
    await evolve.resume()
    return {"ok": True, "paused": False}


@router.post("/pause")
async def pause(request: Request):
    """暂停持续训练。"""
    evolve = _get_evolve(request)
    await evolve.pause()
    return {"ok": True, "paused": True}


@router.post("/trigger/factor-miner")
async def trigger_factor_miner(request: Request):
    """手动触发一次因子挖掘训练（立即执行，等待完成）。"""
    evolve = _get_evolve(request)
    result = await evolve.trigger_factor_miner()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "训练失败"))
    return {"ok": True, **result}


@router.post("/trigger/meta-controller")
async def trigger_meta_controller(request: Request):
    """手动触发一次元策略控制器训练（立即执行，等待完成）。"""
    evolve = _get_evolve(request)
    result = await evolve.trigger_meta_controller()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "训练失败"))
    return {"ok": True, **result}


@router.post("/trigger/strategy-drl")
async def trigger_strategy_drl(request: Request):
    """手动触发一次策略 DRL 训练（立即执行，等待完成）。"""
    evolve = _get_evolve(request)
    result = await evolve.trigger_strategy_drl()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "训练失败"))
    return {"ok": True, **result}


@router.get("/status")
async def status(request: Request):
    """查询训练状态。"""
    evolve = _get_evolve(request)
    return evolve.status()


@router.get("/config")
async def get_config(request: Request):
    """查询当前持续进化可配置项（供面板编辑回填）。"""
    evolve = _get_evolve(request)
    return {"config": evolve.config_keys}


@router.post("/config")
async def post_config(changes: dict, request: Request):
    """热更新持续进化配置（训练标的池/时间周期/间隔/窗口/超参），
    写 settings + 引擎缓存属性并持久化回 config.yaml。"""
    evolve = _get_evolve(request)
    result = await evolve.apply_config(changes)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "配置更新失败"))
    return result


@router.get("/rounds")
async def rounds(model: str = "strategy_drl", limit: int = 200, request: Request = None):
    """训练轮次历史（P2-13 落库回看，供前端 fitness 曲线）。

    P4-E1：model 枚举校验（防任意字符串静默返回空数据）；运行时异常重新
    抛出为 500（此前全部吞掉返回 200 空列表，前端无法区分"无数据"与"出错"）；
    total 返回真实计数（此前是 limit 截断后的行数，分页语义错误）。
    """
    from sqlalchemy import select, func
    from core.database import EvolveRound
    if model not in ("factor_miner", "strategy_drl", "meta_controller"):
        raise HTTPException(status_code=400, detail=f"未知模型 {model}，可选：factor_miner / strategy_drl / meta_controller")
    evolve = _get_evolve(request)
    out: list = []
    total = 0
    try:
        async with evolve.db.session() as s:
            total = (await s.execute(
                select(func.count()).select_from(EvolveRound).where(EvolveRound.model == model)
            )).scalar_one()
            rows = (await s.execute(
                select(EvolveRound)
                .where(EvolveRound.model == model)
                .order_by(EvolveRound.id.desc())
                .limit(min(max(int(limit), 1), 1000)))).scalars().all()
            for r in reversed(rows):  # 升序返回
                out.append({
                    "id": r.id,
                    # P4-E1：ts 为 naive UTC（SQLite CURRENT_TIMESTAMP），
                    # 显式按 UTC 解释再转 epoch，避免本地时区偏移
                    "ts": r.ts.replace(tzinfo=timezone.utc).timestamp() if r.ts else None,
                    "model": r.model,
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "data_source": r.data_source,
                    "round_no": r.round_no,
                    "fitness": r.fitness,
                    "oos_ret": r.oos_ret,
                    "decay": r.decay,
                    "position_ratio": r.position_ratio,
                    "selected_factors": r.selected_factors,
                    "status": r.status,
                })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("[evolve] 读取训练轮次失败: %s", e)
        raise HTTPException(status_code=500, detail=f"读取训练轮次失败: {e}")
    return {"model": model, "rounds": out, "total": total}


@router.get("/versions")
async def versions(name: str = "factor_miner", request: Request = None):
    """列出指定模型的版本。"""
    evolve = _get_evolve(request)
    return {"versions": evolve.zoo.list_versions(name),
            "best_version": evolve.zoo.best_version(name),
            "best_fitness": evolve.zoo.best_fitness(name)}


@router.post("/rollback")
async def rollback(name: str, version: int, request: Request):
    """回退到指定版本。

    P4-E1：与训练写路径互斥（_train_lock）——后台 save_agent/save_agent_best
    对同一份 meta.json/best.json.gz 读改写，无锁时 rollback 与训练并发会互相
    覆盖（丢失版本记录/锚点分裂）。zoo.rollback 含 gzip 压缩与留档复制，
    放 to_thread 防阻塞事件循环。
    """
    import asyncio
    evolve = _get_evolve(request)
    async with evolve._train_lock:
        ok = await asyncio.to_thread(evolve.zoo.rollback, name, version)
    if not ok:
        raise HTTPException(status_code=404, detail=f"版本 v{version} 不存在")
    return {"ok": True, "version": version}


@router.post("/reset-anchor")
async def reset_anchor(payload: dict, request: Request):
    """重置回退基线（锚点）：解锁"每轮都不如历史最佳"的死循环。

    锚点可能是早期 demo 合成数据训出来的高分模型，真实行情永远够不到那个量级，
    于是每轮都判退化 → 进化只回退不前进。前端「删除模型」删不掉这个基线
    （它存在 meta.json 的 best_* 里），只能从这里清。归档式：权重不动、
    meta.json 留档，下一条通过 OOS 检验的模型重建基线。
    """
    evolve = _get_evolve(request)
    name = str((payload or {}).get("name") or "").strip()
    reason = str((payload or {}).get("reason") or "")
    result = await evolve.reset_anchor(name, reason)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "锚点重置失败"))
    return result