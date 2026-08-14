"""AI 设置 / 测试 / 即时解读 / 策略工坊（设计+全自动优化）。Key 加密存储，绝不明文返回。"""
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai.client import AICallError, AINotConfigured
from core.database import Database
from core.security import mask
from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai"])

# 后台 AI 任务进度缓存: {task_id: {stage, stage_label, pct, message, done, error, result}}
_AI_TASKS: dict[int, dict] = {}
_AI_TASKS_LOCK = threading.Lock()


def _ai_next_task_id() -> int:
    with _AI_TASKS_LOCK:
        return max(list(_AI_TASKS.keys()) + [0]) + 1


def _ai_progress(task_id: int, stage: str, stage_label: str, pct: int, message: str = "") -> None:
    """更新后台 AI 任务进度。"""
    with _AI_TASKS_LOCK:
        _AI_TASKS[task_id] = {
            **_AI_TASKS.get(task_id, {}),
            "stage": stage, "stage_label": stage_label, "pct": pct,
            "message": message, "done": False, "running": True,
        }


def _ai_finish(task_id: int, result=None, error: str = "") -> None:
    from .tasks_cache import mark_done, prune_task_cache
    with _AI_TASKS_LOCK:
        _AI_TASKS[task_id] = mark_done({
            **_AI_TASKS.get(task_id, {}),
            "done": True, "running": False, "pct": 100,
            "result": result, "error": error,
        })
    # 防内存膨胀：完成后清理过期任务（保留最近 200 个、12 小时内的）
    prune_task_cache(_AI_TASKS, _AI_TASKS_LOCK)


class AISettingsIn(BaseModel):
    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # 温度/token 已由代码层按功能预设（client.AI_FEATURE_PARAMS），不再让用户手调。
    # 保留字段以兼容旧前端，但已不使用。


@router.get("/settings")
async def get_ai_settings(db: Database = Depends(get_db)):
    key = await db.kv_get_secret("ai_api_key") or ""
    from ai.client import AI_FEATURE_PARAMS
    return {
        "provider": await db.kv_get("ai_provider") or "openai",
        "base_url": await db.kv_get("ai_base_url") or "",
        "model": await db.kv_get("ai_model") or "",
        "api_key_masked": mask(key),
        "has_api_key": bool(key),
        # 温度/token 由代码层按功能预设，返回供前端展示说明
        "feature_params": AI_FEATURE_PARAMS,
    }


@router.put("/settings")
async def save_ai_settings(body: AISettingsIn, db: Database = Depends(get_db), engine=Depends(get_engine)):
    # 空串=清空该字段（允许用户重置 base_url/api_key 换回默认 provider）：
    # 曾用 if body.x: 跳过空值，导致配置一旦设置就无法清空
    await db.kv_set("ai_provider", body.provider or "openai")
    await db.kv_set("ai_base_url", body.base_url)
    await db.kv_set("ai_model", body.model)
    if body.api_key is not None:
        if body.api_key.strip():
            await db.kv_set("ai_api_key", body.api_key.strip(), is_secret=True)
        else:
            await db.kv_set("ai_api_key", "", is_secret=True)  # 清空密钥
    # invalidate_cache 是同步方法，不能 await
    engine.ai_client.invalidate_cache()
    return {"ok": True, "message": "AI 配置已保存（密钥已加密存储，温度/token 已按功能自动预设）"}


class AITestIn(BaseModel):
    """测试连接的临时配置（不落库）：留空则用已保存配置测试。"""
    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""


@router.post("/test")
async def test_ai(body: Optional[AITestIn] = None, db: Database = Depends(get_db),
                  engine=Depends(get_engine)):
    try:
        override = body.model_dump() if body else {}
        reply = await engine.ai_client.test_connection(override=override or None)
        return {"ok": True, "message": f"连接成功，模型回复: {reply[:100]}"}
    except AINotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AICallError as e:
        raise HTTPException(status_code=502, detail=str(e))


class AIModelsIn(BaseModel):
    provider: str = ""
    base_url: str = ""
    api_key: str = ""


@router.post("/models")
async def fetch_ai_models(body: AIModelsIn, engine=Depends(get_engine)):
    """拉取可用模型列表（Cherry Studio 式：填 Key+地址一键获取）。

    用表单临时值请求 {base_url}/models，不保存、不落日志、不回显 Key。
    """
    try:
        models = await engine.ai_client.list_models(body.base_url, body.api_key)
        return {"ok": True, "models": models}
    except AINotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AICallError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/analyze")
async def analyze_now(engine=Depends(get_engine)):
    snap = await engine._current_snapshot()
    if not snap:
        raise HTTPException(status_code=400, detail="暂无行情数据，请先启动引擎")
    try:
        result = await engine.analyst.analyze(snap)
        return {"ok": True, "result": result}
    except AINotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AICallError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/design-strategy")
async def design_strategy(payload: dict = {}, engine=Depends(get_engine)):
    """🧠 AI 设计策略：基于关键位+价格行为生成可执行策略。

    payload 可选:
    - symbol: 指定设计品种（如 BTC/USDT），默认引擎当前品种
    - timeframe: 指定周期（如 1h），默认引擎当前周期
    - strategy_type: 策略类型偏好（trend/breakout/mean_reversion/grid/custom）
    - custom_requirement: 用户自定义设计要求（自由打字）
    """
    symbol = (payload or {}).get("symbol") or ""
    timeframe = (payload or {}).get("timeframe") or ""
    strategy_type = (payload or {}).get("strategy_type") or ""
    custom_requirement = (payload or {}).get("custom_requirement") or ""
    try:
        spec = await engine.design_strategy(
            symbol=symbol, timeframe=timeframe,
            strategy_type=strategy_type, custom_requirement=custom_requirement)
        return {"ok": True, "strategy": spec}
    except AINotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AICallError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-optimize")
async def auto_optimize(payload: dict = {}, engine=Depends(get_engine)):
    """⚡ AI 全自动优化：后台执行，实时进度推送（阶段：取数→分析→优化→校验→应用）。"""
    import asyncio
    focus = payload.get("focus", "价格行为与关键位")
    task_id = _ai_next_task_id()

    def _worker():
        async def _run():
            try:
                _ai_progress(task_id, "snapshot", "获取行情快照", 5)
                snap = await engine._current_snapshot()
                if not snap:
                    raise RuntimeError("无法获取行情快照，请先启动引擎或稍后重试")
                _ai_progress(task_id, "perf", "分析近期表现", 15,
                             f"当前策略 {engine.strategy.name}")
                perf = await engine._recent_performance()
                backtests = await engine._recent_backtests(3)
                _ai_progress(task_id, "ai", "AI 正在分析价格行为与优化参数", 40,
                             "AI 思考中（依据 S/R、量能、近期盈亏）…")
                result = await engine.optimizer.optimize_price_action(
                    engine.strategy, perf, snap, focus, backtests, apply=False)
                if result is None:
                    raise RuntimeError("AI 优化失败（未配置 AI 或调用出错）")
                # 参数应用调度回主循环（持策略锁）：曾 worker 直接 update_params
                # 绕过锁，与 K 线处理并发写 params；asyncio.Lock 不能跨循环直接获取
                if result.get("params") and engine._loop is not None:
                    fut = asyncio.run_coroutine_threadsafe(
                        engine.apply_strategy_params(engine.strategy.name, result["params"]),
                        engine._loop)
                    fut.result(timeout=10.0)
                else:
                    log.warning("[ai] 无主循环引用，参数未应用（仅记录）: %s", result.get("params"))
                _ai_progress(task_id, "validate", "校验输出并热更新", 75,
                             "参数已过量化规则校验")
                result["market_features"] = {
                    "sr": snap.get("indicators", {}).get("sr"),
                    "pa": snap.get("indicators", {}).get("pa"),
                }
                _ai_finish(task_id, result=result)
            except Exception as e:  # noqa: BLE001
                log.exception("[ai] 自动优化失败")
                _ai_finish(task_id, error=str(e))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True, "task_id": task_id}


@router.get("/task/{task_id}")
async def ai_task_progress(task_id: int):
    """读取后台 AI 任务（优化/迭代）实时进度。"""
    with _AI_TASKS_LOCK:
        p = _AI_TASKS.get(int(task_id))
    if p is None:
        return {"found": False}
    return {"found": True, **p}


@router.post("/iterate-strategy")
async def iterate_strategy(payload: dict = {}, engine=Depends(get_engine)):
    """🧬 AI 重新思考迭代策略：后台执行，实时进度推送（阶段：取数→分析→反思→校验→注册）。

    payload: {"strategy_name": "dual_ma" | "price_action" | ...}
    """
    import asyncio
    strategy_name = (payload or {}).get("strategy_name", "")
    task_id = _ai_next_task_id()

    def _worker():
        async def _run():
            try:
                _ai_progress(task_id, "snapshot", "获取行情快照", 5)
                target = engine.strategy
                if strategy_name:
                    from strategies import get_strategy
                    target = get_strategy(strategy_name)
                _ai_progress(task_id, "perf", "分析近期表现与前代迭代", 15,
                             f"目标策略 {target.name}")
                snap = await engine._current_snapshot()
                if not snap:
                    raise RuntimeError("无法获取行情快照，请先启动引擎或稍后重试")
                guard_candles = await engine._fetch_vision_ohlcv(limit=800) or snap.get("candles", [])
                snap = {**snap, "candles": guard_candles}
                perf = await engine._recent_performance()
                backtests = await engine._recent_backtests(3)
                previous = await engine._previous_iterations(target.name)
                _ai_progress(task_id, "ai", "AI 批判性反思中", 45,
                             "审视失效点 + 对照前代改进效果…")
                result = await engine.iteration.iterate(target, perf, snap, backtests, previous)
                if result is None:
                    raise RuntimeError("AI 迭代失败（未配置 AI 或调用出错）")
                _ai_progress(task_id, "validate", "校验输出并注册", 80,
                             "参数已过量化规则校验与过拟合检测")
                result["market_features"] = {
                    "sr": snap.get("indicators", {}).get("sr"),
                    "pa": snap.get("indicators", {}).get("pa"),
                }
                # 迭代前后绩效对比（量化验证"每一代正向提升"，借鉴 freqtrade 的 backtest 验证习惯）
                try:
                    _ai_progress(task_id, "compare", "新旧参数回测对比", 90,
                                 "用同一数据验证迭代是否真正提升…")
                    candles = guard_candles or snap.get("candles", [])
                    if len(candles) >= 200:
                        import pandas as pd
                        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
                        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
                        from backtest.engine import BacktestConfig
                        from backtest.fast_engine import run_backtest_fast
                        old_params = dict(target.params or {})
                        new_params = dict(result.get("params") or {})
                        old_bt = run_backtest_fast(df, BacktestConfig(
                            symbol=snap.get("symbol", "BTC/USDT"), timeframe=snap.get("timeframe", "1h"),
                            strategy_name=target.name, strategy_params=old_params))["metrics"]
                        new_bt = run_backtest_fast(df, BacktestConfig(
                            symbol=snap.get("symbol", "BTC/USDT"), timeframe=snap.get("timeframe", "1h"),
                            strategy_name=result.get("name") or target.name,
                            strategy_params=new_params))["metrics"]
                        result["comparison"] = {
                            "old": {"total_return": old_bt["total_return"], "sharpe": old_bt["sharpe"],
                                    "max_drawdown": old_bt["max_drawdown"], "win_rate": old_bt["win_rate"]},
                            "new": {"total_return": new_bt["total_return"], "sharpe": new_bt["sharpe"],
                                    "max_drawdown": new_bt["max_drawdown"], "win_rate": new_bt["win_rate"]},
                            "improved": bool(new_bt["total_return"] > old_bt["total_return"]),
                            "note": "同数据快速回测对比（含手续费/滑点）",
                        }
                except Exception as e:  # noqa: BLE001
                    log.warning("[ai] 迭代对比回测失败: %s", e)
                _ai_finish(task_id, result=result)
            except Exception as e:  # noqa: BLE001
                log.exception("[ai] 策略迭代失败")
                _ai_finish(task_id, error=str(e))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True, "task_id": task_id}


@router.get("/strategies")
async def list_ai_strategies(db: Database = Depends(get_db)):
    """列出所有 AI 设计并保存过的策略。"""
    from core.database import AiStrategy
    from sqlalchemy import select
    async with db.session() as s:
        rows = (await s.execute(select(AiStrategy).order_by(AiStrategy.id.desc()).limit(50))).scalars().all()
        return [{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat(),
                 "spec": json.loads(r.spec_json or "{}")} for r in rows]


@router.get("/analyses")
async def list_analyses(limit: int = 20, db: Database = Depends(get_db)):
    from core.database import Analysis
    from sqlalchemy import select
    async with db.session() as s:
        rows = (await s.execute(select(Analysis).order_by(Analysis.ts.desc()).limit(limit))).scalars().all()
        return [{"id": a.id, "ts": a.ts.isoformat(), "symbol": a.symbol, "content": a.content} for a in rows]


@router.get("/auto-analysis-toggle")
async def get_auto_analysis_toggle(db: Database = Depends(get_db)):
    """自动行情分析开关状态。"""
    return {"enabled": (await db.kv_get("ai_auto_analysis_enabled", "1")) != "0"}


@router.put("/auto-analysis-toggle")
async def set_auto_analysis_toggle(payload: dict, db: Database = Depends(get_db)):
    """切换自动行情分析开关。"""
    enabled = bool(payload.get("enabled", True))
    await db.kv_set("ai_auto_analysis_enabled", "1" if enabled else "0")
    return {"ok": True, "enabled": enabled}


@router.get("/optimization-logs")
async def list_optimization_logs(limit: int = 50, db: Database = Depends(get_db)):
    from core.database import OptimizationLog
    from sqlalchemy import select
    async with db.session() as s:
        rows = (await s.execute(select(OptimizationLog).order_by(OptimizationLog.ts.desc()).limit(limit))).scalars().all()
        return [{"id": r.id, "ts": r.ts.isoformat(), "kind": r.kind, "summary": r.summary,
                 "suggestion": r.suggestion, "params": r.params_json} for r in rows]