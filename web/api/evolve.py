"""持续进化引擎控制 API。"""
import json
from datetime import timezone
import re

from fastapi import APIRouter, HTTPException, Query, Request


router = APIRouter(prefix="/api/evolve", tags=["evolve"])
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+$")
_CONFIG_KEYS = {
    "symbols", "timeframe", "factor_miner_interval", "strategy_drl_interval",
    "rolling_window", "rollback_threshold", "factor_miner_episodes",
    "strategy_drl_episodes", "meta_episodes", "vol_penalty", "oos_min_bars",
    "cross_symbol_oos", "min_new_bars",
    # P0/P1：续训控成本 + 锚点量级闸 + 故障期暂停
    "train_time_budget", "min_train_gap_sec", "train_on_demo", "anchor_max_fitness",
    # P0-族群：策略 DRL 冠军/挑战者 K=2 族群开关
    "strategy_drl_population",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]*/[A-Z0-9][A-Z0-9._-]*(?::[A-Z0-9][A-Z0-9._-]*)?$")


def _validate_model_name(name: str) -> str:
    name = str(name or "")
    if not _MODEL_NAME_RE.fullmatch(name) or name in (".", ".."):
        raise HTTPException(status_code=400, detail="非法模型名")
    return name


def _parse_json_value(value, fallback):
    """把落库的 JSON 文本解析回对象；老行 str(list) 等非 JSON 格式原样透传。"""
    if value is None or value == "":
        return fallback
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:  # noqa: BLE001
        return value


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
    if result.get("busy"):
        raise HTTPException(status_code=409, detail=result.get("reason", "另一条训练正在进行中"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or result.get("reason") or "训练失败")
    return {"ok": True, **result}


@router.post("/trigger/meta-controller")
async def trigger_meta_controller(request: Request):
    """手动触发一次元策略控制器训练（立即执行，等待完成）。"""
    evolve = _get_evolve(request)
    result = await evolve.trigger_meta_controller()
    if result.get("busy"):
        raise HTTPException(status_code=409, detail=result.get("reason", "另一条训练正在进行中"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or result.get("reason") or "训练失败")
    return {"ok": True, **result}


@router.post("/trigger/strategy-drl")
async def trigger_strategy_drl(request: Request):
    """手动触发一次策略 DRL 训练（立即执行，等待完成）。"""
    evolve = _get_evolve(request)
    result = await evolve.trigger_strategy_drl()
    if result.get("busy"):
        raise HTTPException(status_code=409, detail=result.get("reason", "另一条训练正在进行中"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or result.get("reason") or "训练失败")
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
    unknown = sorted(set(changes) - _CONFIG_KEYS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"未知配置项: {', '.join(unknown)}")
    if "symbols" in changes:
        symbols = changes["symbols"]
        if not isinstance(symbols, list) or not symbols or len(symbols) > 50:
            raise HTTPException(status_code=422, detail="symbols 必须是 1-50 个交易对的列表")
        normalized = []
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise HTTPException(status_code=422, detail="symbols 必须全部是字符串")
            symbol = symbol.strip().upper()
            if not _SYMBOL_RE.fullmatch(symbol):
                raise HTTPException(status_code=422, detail=f"非法交易对: {symbol!r}")
            normalized.append(symbol)
        changes = {**changes, "symbols": normalized}
    if "cross_symbol_oos" in changes and changes["cross_symbol_oos"] is not None:
        if not isinstance(changes["cross_symbol_oos"], bool):
            raise HTTPException(status_code=422, detail="cross_symbol_oos 必须是布尔值")
    evolve = _get_evolve(request)
    result = await evolve.apply_config(changes)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "配置更新失败"))
    return result


@router.get("/rounds")
async def rounds(model: str = "strategy_drl",
                 limit: int = Query(200, ge=1, le=1000),
                 request: Request = None):
    """训练轮次历史（P2-13 落库回看，供前端 fitness 曲线）。

    P4-E1：model 枚举校验（防任意字符串静默返回空数据）；运行时异常重新
    抛出为 500（此前全部吞掉返回 200 空列表，前端无法区分"无数据"与"出错"）；
    total 返回真实计数（此前是 limit 截断后的行数，分页语义错误）。
    P2：limit 用 FastAPI Query 校验（此前裸 int() 对非法输入抛 ValueError → 500）。
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
                .limit(limit))).scalars().all()
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
                    "selected_factors": _parse_json_value(r.selected_factors, []),
                    "status": r.status,
                    "audit": _parse_json_value(getattr(r, "audit_json", None), {}),
                })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("evolve").exception("[evolve] 读取训练轮次失败")
        raise HTTPException(status_code=500, detail="读取训练轮次失败，请稍后重试")

    return {"model": model, "rounds": out, "total": total}


@router.get("/versions")
async def versions(name: str = "factor_miner", request: Request = None):
    """列出指定模型的版本（含身份/部署元数据摘要）。"""
    name = _validate_model_name(name)
    evolve = _get_evolve(request)
    return {"versions": evolve.zoo.list_versions(name),
            "best_version": evolve.zoo.best_version(name),
            "best_fitness": evolve.zoo.best_fitness(name),
            "best_info": evolve.zoo.best_info(name)}


@router.post("/rollback")
async def rollback(name: str, version: int = Query(..., ge=1), request: Request = None):
    """手动回退到指定版本（走统一部署入口：快照恢复 + 注册 + 运行时热加载）。

    P4-E1：与训练写路径互斥（_train_lock）——后台 save_agent/save_agent_best
    对同一份 meta.json/best.json.gz 读改写，无锁时 rollback 与训练并发会互相
    覆盖（丢失版本记录/锚点分裂）。手动回退落 manual_rollback 审计轮次。
    """
    name = _validate_model_name(name)
    evolve = _get_evolve(request)
    async with evolve._train_lock:
        result = await evolve.deploy_model(
            name, version, outcome="manual_rollback", reason="UI 手动回退")
    if not result.get("ok"):
        if "不存在" in str(result.get("error") or ""):
            raise HTTPException(status_code=404, detail=f"版本 v{version} 不存在")
        if result.get("runtime_reload", {}).get("status") == "failed":
            raise HTTPException(status_code=409,
                                detail=result.get("error") or "运行时模型热加载未完成")
        raise HTTPException(status_code=400,
                            detail=result.get("error") or "回退失败")
    return result


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


@router.get("/factor-strategies")
async def factor_strategies(request: Request):
    """列出全部已部署的进化组合因子策略（权重/OOS报告/版本/回测摘要）。

    实现：扫描 strategies 动态注册表，过滤 created_by=evolve_engine 且
    name 前缀 evolve_combo_；回测摘要优先取该策略最近一次部署（内存 status）。
    """
    from strategies import dynamic_names, get_dynamic
    evolve = _get_evolve(request)
    fm = evolve._factor_miner_status
    latest_name = fm.get("combo_strategy") or ""
    latest_bt = fm.get("combo_backtest")
    out = []
    for name in sorted(dynamic_names()):
        if not name.startswith("evolve_combo_"):
            continue
        spec = get_dynamic(name) or {}
        if spec.get("created_by") != "evolve_engine":
            continue
        params = spec.get("params") or {}
        combo_spec = params.get("combo_spec", "")
        weights = {}
        if combo_spec:
            try:
                weights = json.loads(combo_spec)
            except Exception:  # noqa: BLE001
                weights = {}
        em = spec.get("evolve_meta") or {}
        out.append({
            "name": name,
            "symbol": spec.get("base_symbol", ""),
            "timeframe": spec.get("base_timeframe", ""),
            "version": spec.get("version", ""),
            "weights": weights,
            "oos_report": em.get("oos_report"),
            "selected_factors": em.get("selected_factors", []),
            "fitness": em.get("fitness"),
            "deployed_at": em.get("deployed_at"),
            "backtest": latest_bt if name == latest_name else None,
        })
    return {"strategies": out, "count": len(out)}


@router.post("/factor/deploy")
async def deploy_factor(symbol: str = Query(..., description="交易对，如 BTC/USDT"),
                        request: Request = None):
    """手动重新部署进化组合因子策略（读最新 _cascade_weights_<symbol>.json）。

    symbol 走 query 参数（路径段含 / 无法路由）；与训练写路径互斥持 _train_lock。
    """
    if not _SYMBOL_RE.fullmatch(str(symbol or "")):
        raise HTTPException(status_code=400, detail=f"非法交易对: {symbol!r}")
    evolve = _get_evolve(request)
    wpath = evolve.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
    if not wpath.exists():
        raise HTTPException(status_code=404,
                            detail=f"{symbol} 暂无进化因子权重（先训练 factor_miner）")
    try:
        weights = json.loads(wpath.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"权重文件解析失败: {e}")
    if not isinstance(weights, dict) or not weights:
        raise HTTPException(status_code=400, detail="权重文件为空")
    # 伴随报告元数据（_cascade_report_<symbol>.json，训练落盘）——权重对应的真实
    # OOS 报告/fitness/选中因子；缺失或损坏时回退：携带上一版已部署 spec 的
    # evolve_meta（避免手动重部署把 OOS 报告覆盖成 null、fitness 用错成最新轮次值）。
    meta = {
        "fitness": float(evolve._factor_miner_status.get("fitness") or 0.0),
        "selected_factors": list(weights.keys()),
        "round_no": int(evolve._factor_miner_status.get("episode", 0)),
    }
    _report = None
    _companion_ok = False
    rpath = evolve.zoo.models_dir / f"_cascade_report_{symbol.replace('/', '_')}.json"
    if rpath.exists():
        try:
            _rep = json.loads(rpath.read_text(encoding="utf-8"))
            if isinstance(_rep, dict) and _rep.get("oos_report"):
                meta["fitness"] = float(_rep.get("fitness") or meta["fitness"])
                meta["selected_factors"] = _rep.get("selected_factors") or meta["selected_factors"]
                meta["round_no"] = int(_rep.get("round_no") or meta["round_no"])
                _report = _rep["oos_report"]
                _companion_ok = True
        except Exception:  # noqa: BLE001 伴随文件损坏时静默走回退
            pass
    if not _companion_ok:
        # 回退：沿用上一版 spec 的 OOS 报告与 fitness（同标同名策略）
        from strategies import get_dynamic
        _prev = get_dynamic(evolve._combo_strategy_name(symbol)) or {}
        _prev_em = (_prev.get("evolve_meta") or {})
        _report = _prev_em.get("oos_report")
        if _prev_em.get("fitness") is not None:
            meta["fitness"] = float(_prev_em["fitness"])
    async with evolve._train_lock:
        result = await evolve._deploy_factor_strategy(symbol, weights, _report, meta)
    return {"ok": True, **result}