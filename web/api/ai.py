"""AI 设置 / 测试 / 即时解读 / 策略工坊（设计+全自动优化）。Key 加密存储，绝不明文返回。"""
import asyncio
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

# P2-3：AI 类后台任务（自动优化/策略迭代共用）并发上限，超限 429
_AI_SEM = threading.Semaphore(2)


def _ai_next_task_id() -> int:
    with _AI_TASKS_LOCK:
        return max(list(_AI_TASKS.keys()) + [0]) + 1


def _run_on_loop(engine, coro_factory, timeout=30.0):
    """把引擎/DB 协程调度回主循环执行（后台线程调用；阻塞等待结果）。

    worker 线程新建事件循环直接 await 引擎 DB 方法（_recent_performance 等
    内部 db.session()）会跨 loop 复用主循环绑定的连接池 → 间歇性 RuntimeError；
    统一经 run_coroutine_threadsafe 调度回主循环（参照 apply_strategy_params
    的示范模式，见 auto_optimize worker）。
    """
    if engine is None or engine._loop is None:
        raise RuntimeError("主循环引用不可用")
    fut = asyncio.run_coroutine_threadsafe(coro_factory(), engine._loop)
    return fut.result(timeout=timeout)


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


async def _run_nearby_grid_scan(engine, center: dict) -> Optional[dict]:
    """以 AI 建议参数为中心做 ±20% 局部网格扫描。

    直接复用 backtest.fast_engine.run_backtest_fast + 本地 df（与 grid-scan 端点同构，
    避开 HTTP 往返）。尊重 _BT_SEM 并发上限（2 个），超限时跳过并 log.warning。
    """
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast
    from web.api.backtest import _BT_SEM

    # 检查并发上限：超限时跳过（不阻塞 AI 优化流程）
    if not _BT_SEM.acquire(blocking=False):
        log.warning("[grid-scan] 跳过网格扫描：回测并发已满")
        return None
    try:
        snap = await engine._current_snapshot()
        if not snap or len(snap.get("closes", [])) < 60:
            return None
        import pandas as pd
        df = pd.DataFrame(snap["candles"],
                          columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)

        # 参数候选：数值参数 ±20%；布尔/字符串不变
        keys, combos = list(center.keys()), [[]]
        for k in keys:
            v = center[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                step = max(abs(v) * 0.2, 1e-9)
                cands = [v - step, v, v + step]
            else:
                cands = [v]
            combos = [c + [x] for c in combos for x in cands]

        def _bt_one(params: dict) -> dict:
            cfg = BacktestConfig(symbol=engine.symbol, timeframe=engine.timeframe,
                                 strategy_name=engine.strategy.name,
                                 strategy_params=params)
            r = run_backtest_fast(df, cfg)
            m = r["metrics"]
            return {"params": params, "total_return": m["total_return"],
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"]}

        # 并行回测：run_backtest_fast 线程安全（get_strategy 每次新建实例 + df.copy，
        # 无共享可变状态），pandas/numpy 计算多释放 GIL；并发 4 与 _BT_SEM=2 总量协调
        _SCAN_CONCURRENCY = 4
        _scan_sem = asyncio.Semaphore(_SCAN_CONCURRENCY)

        async def _run_one(params: dict):
            async with _scan_sem:
                return await asyncio.to_thread(_bt_one, params)

        raw = await asyncio.gather(*[_run_one(dict(zip(keys, combo))) for combo in combos],
                                   return_exceptions=True)
        results = [r for r in raw if isinstance(r, dict)]

        results.sort(key=lambda x: -(x["total_return"] * 10 + x["sharpe"] * 2
                                      - x["max_drawdown"] * 5))
        best = results[0] if results else None
        log.info("[grid-scan] 以 AI 建议为中心扫描完成: %d 组", len(results))
        return {"ok": True, "scanned": len(results), "results": results[:5], "best": best}
    finally:
        _BT_SEM.release()


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


def _math_analysis(snap: dict) -> dict:
    """程序化数学行情分析：不依赖 AI 猜测，纯指标计算给出可复现的结论。

    输出：趋势判定(上涨/下跌/震荡)、支撑/阻力位置、入场/止损/止盈、
    盈亏比、胜率估计(基于价格在 S/R 区间位置 + 方向信号一致性)、
    期望收益%（量化到当前价 1 单位仓位）。数据不足时返回 {}，前端隐藏。
    """
    try:
        ind = snap.get("indicators") or {}
        closes = snap.get("closes") or []
        candles = snap.get("candles") or []
        if not closes or not candles:
            return {}
        last = float(closes[-1])
        if last <= 0:
            return {}
        # 趋势判定：五信号投票（复用方向引擎）
        from ai.direction_engine import compute_direction
        prog = compute_direction(candles)
        direction = prog.get("direction") or "neutral"
        score = int(prog.get("score") or 0)
        conf = float(prog.get("confidence") or 0.0)
        trend_label = {"long": "上涨趋势", "short": "下跌趋势", "neutral": "震荡行情"}.get(direction, "震荡行情")
        # 波动率（ATR% → 价格）
        atr_pct = float(ind.get("atr_pct") or 0.0)
        atr = last * atr_pct / 100.0 if atr_pct > 0 else last * 0.01
        # 支撑/阻力（数学聚类位，非猜测）
        sr = ind.get("sr") or {}
        support = sr.get("support")
        resistance = sr.get("resistance")
        support_levels = sr.get("support_levels") or []
        resistance_levels = sr.get("resistance_levels") or []
        # ---- 入场方案（数学规则，不做行情猜测）----
        entry = stop = tp = None
        rr = 0.0
        if direction == "long":
            # 做多：贴支撑入场（若距支撑 <1.5ATR），否则现价；止损 1.5ATR 下方；止盈取阻力或 2.5ATR
            entry = support if (support and last - support < 1.5 * atr) else last
            stop = entry - 1.5 * atr
            tp = resistance if (resistance and resistance > entry) else entry + 2.5 * atr
        elif direction == "short":
            entry = resistance if (resistance and resistance - last < 1.5 * atr) else last
            stop = entry + 1.5 * atr
            tp = support if (support and support < entry) else entry - 2.5 * atr
        else:
            # 震荡：区间策略——贴近下沿做多 / 上沿做空（同时给两个方案）
            lo = support or last - 2.0 * atr
            hi = resistance or last + 2.0 * atr
        if direction in ("long", "short") and entry and stop and tp:
            risk = abs(entry - stop)
            reward = abs(tp - entry)
            rr = round(reward / risk, 2) if risk > 0 else 0.0
        # ---- 胜率估计（数学口径，非玄学）----
        # 基础：价格在支撑/阻力区间内的相对位置 → 离支撑越近做多胜率越高；再按方向信号一致性微调
        span = None
        if support is not None and resistance is not None and resistance > support:
            span = (last - support) / (resistance - support)
        base = 0.5
        if span is not None:
            if direction == "long":
                base += (1.0 - span) * 0.25      # 贴支撑(+0.25)~贴阻力(0.0)
            elif direction == "short":
                base += span * 0.25               # 贴阻力(+0.25)~贴支撑(0.0)
            else:
                base += 0.02                       # 震荡中性微偏
        win_rate = min(0.85, max(0.35, base + conf * 0.10))
        # ---- 期望收益%（按 1 单位价格计算，含盈亏比）----
        ev_pct = None
        if direction in ("long", "short") and entry and stop and tp:
            ev_pct = round(
                win_rate * (tp - entry) / entry
                - (1 - win_rate) * (entry - stop) / entry, 5)
        return {
            "direction": direction,
            "trend_label": trend_label,
            "score": score,
            "confidence": round(conf, 3),
            "last_price": round(last, 6),
            "atr": round(atr, 6),
            "support": support, "resistance": resistance,
            "support_levels": support_levels, "resistance_levels": resistance_levels,
            "entry": round(entry, 6) if entry else None,
            "stop_loss": round(stop, 6) if stop else None,
            "take_profit": round(tp, 6) if tp else None,
            "risk_reward": rr,
            "win_rate": round(win_rate, 3),
            "expected_value_pct": ev_pct,
            "interval_low": round(last - 2.0 * atr, 6) if direction == "neutral" else None,
            "interval_high": round(last + 2.0 * atr, 6) if direction == "neutral" else None,
            "basis": "基于摆动点 S/R 聚类 + ATR 波动 + 方向五信号投票的纯数学计算（不依赖 AI 预测）",
        }
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 数学分析计算失败: %s", e)
        return {}


@router.post("/analyze")
async def analyze_now(payload: dict = {}, engine=Depends(get_engine)):
    """AI 行情解读：可选指定品种/周期，默认引擎当前品种/周期。

    payload 可选:
    - symbol: 解读品种（如 BTC/USDT），默认引擎当前品种
    - timeframe: 解读周期（如 1h），默认引擎当前周期
    """
    symbol = (payload or {}).get("symbol") or ""
    timeframe = (payload or {}).get("timeframe") or ""
    snap = await engine._current_snapshot()
    if not snap:
        raise HTTPException(status_code=400, detail="暂无行情数据，请先启动引擎")
    # 指定了品种或周期且与引擎当前不一致时，拉取对应K线构造快照（与 design_strategy 同口径）
    if (symbol or timeframe) and (symbol != engine.symbol or timeframe != engine.timeframe):
        use_symbol = symbol or engine.symbol
        use_tf = timeframe or engine.timeframe
        ohlcv = await engine._fetch_vision_ohlcv(limit=150, symbol=use_symbol, timeframe=use_tf)
        if ohlcv:
            from indicators.technical import compute_latest
            ohlcv = [list(c) for c in ohlcv]
            snap = {"symbol": use_symbol, "timeframe": use_tf, "candles": ohlcv,
                    "closes": [c[4] for c in ohlcv], "indicators": compute_latest(ohlcv)}
        else:
            log.warning("[ai] 指定品种 %s/%s 拉取失败，回退引擎当前快照", use_symbol, use_tf)
    try:
        # 注入真实持仓与近期交易（曾恒传 0/空）
        pos, recent = await engine._analysis_context(snap)
        result = await engine.analyst.analyze(snap, position=pos, recent_trades=recent)
        # 附加程序化数学结论：趋势判定 + 支撑/阻力 + 入场建议 + 期望/胜率（不依赖 AI 猜测）
        result["_math"] = _math_analysis(snap)
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


# ============ AI 后台任务公共基础设施 ============
# auto_optimize / iterate_strategy 的线程/EventLoop/并发控制/清理逻辑完全一致，
# 提取为公共函数：每个路由只需定义自己的 async _run() 协程，
# 由 _run_ai_task_background 统一管理 worker 线程、事件循环、信号量释放与 httpx 资源清理。


def _run_ai_task_background(task_id: int, engine, run_coro,
                            sem: threading.Semaphore) -> bool:
    """启动后台 AI 任务 worker 线程。返回 True 表示启动成功。

    封装了线程创建、事件循环、信号量释放、client 连接清理等全部基础设施。
    run_coro: 无参数 callable，返回 coroutine object（路由专属的业务逻辑）。
    失败时自动释放 sem。
    """
    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_coro())
        finally:
            sem.release()
            try:
                loop.run_until_complete(engine.ai_client.close_loop(loop))
            except Exception:  # noqa: BLE001
                pass
            loop.close()

    try:
        threading.Thread(target=_worker, daemon=True).start()
        return True
    except Exception:
        sem.release()
        raise


@router.post("/auto-optimize")
async def auto_optimize(payload: dict = {}, engine=Depends(get_engine)):
    """⚡ AI 全自动优化：后台执行，实时进度推送（阶段：取数→分析→优化→校验→应用）。"""
    if not _AI_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="AI 任务繁忙：最多 2 个并发任务，请稍后再试")
    focus = payload.get("focus", "价格行为与关键位")
    task_id = _ai_next_task_id()

    async def _run():
        try:
            _ai_progress(task_id, "snapshot", "获取行情快照", 5)
            snap = await engine._current_snapshot()
            if not snap:
                raise RuntimeError("无法获取行情快照，请先启动引擎或稍后重试")
            _ai_progress(task_id, "perf", "分析近期表现", 15,
                         f"当前策略 {engine.strategy.name}")
            try:
                perf = _run_on_loop(engine, lambda: engine._recent_performance())
            except TimeoutError:
                log.warning("[ai] 读取近期表现超时（主循环繁忙），降级继续")
                perf = None
            try:
                backtests = _run_on_loop(engine, lambda: engine._recent_backtests(3))
            except TimeoutError:
                log.warning("[ai] 读取近期回测超时（主循环繁忙），降级继续")
                backtests = []
            _ai_progress(task_id, "ai", "AI 正在分析价格行为与优化参数", 40,
                         "AI 思考中（依据 S/R、量能、近期盈亏）…")
            result = await engine.optimizer.optimize_price_action(
                engine.strategy, perf, snap, focus, backtests, apply=False)
            if result is None:
                raise RuntimeError("AI 优化失败（未配置 AI 或调用出错）")
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
            grid_center = result.get("grid_center") or {}
            if grid_center and engine.bus is not None:
                _ai_progress(task_id, "grid", "局部网格扫描（以 AI 建议为中心）", 70,
                             "参数范围 ±20%…")
                grid_result = await _run_nearby_grid_scan(engine, grid_center)
                if grid_result and grid_result.get("best"):
                    best = grid_result["best"]["params"]
                    if best and engine._loop is not None:
                        fut = asyncio.run_coroutine_threadsafe(
                            engine.apply_strategy_params(engine.strategy.name, best),
                            engine._loop)
                        fut.result(timeout=10.0)
                        log.info("[grid-scan] 应用最优参数: %s", best)
                    result["grid_scan"] = {"scanned": grid_result.get("scanned"),
                                           "best": best, "applied": bool(best)}
            _ai_finish(task_id, result=result)
        except Exception as e:  # noqa: BLE001
            log.exception("[ai] 自动优化失败")
            _ai_finish(task_id, error=str(e))

    _run_ai_task_background(task_id, engine, _run, _AI_SEM)
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
    if not _AI_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="AI 任务繁忙：最多 2 个并发任务，请稍后再试")
    strategy_name = (payload or {}).get("strategy_name", "")
    task_id = _ai_next_task_id()

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
            guard_candles = await engine._fetch_history_ohlcv(engine.GUARD_CANDLES) \
                or snap.get("candles", [])
            snap = {**snap, "candles": guard_candles}
            try:
                perf = _run_on_loop(engine, lambda: engine._recent_performance())
            except TimeoutError:
                log.warning("[ai] 读取近期表现超时（主循环繁忙），降级继续")
                perf = None
            try:
                backtests = _run_on_loop(engine, lambda: engine._recent_backtests(3))
            except TimeoutError:
                log.warning("[ai] 读取近期回测超时（主循环繁忙），降级继续")
                backtests = []
            try:
                previous = _run_on_loop(engine, lambda: engine._previous_iterations(target.name))
            except TimeoutError:
                log.warning("[ai] 读取历史迭代超时（主循环繁忙），降级继续")
                previous = []
            _ai_progress(task_id, "ai", "AI 批判性反思中", 45,
                         "审视失效点 + 对照前代改进效果…")
            validation_df = None
            try:
                candles = guard_candles or snap.get("candles", [])
                if len(candles) >= 200:
                    import pandas as pd
                    validation_df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
                    validation_df["timestamp"] = pd.to_datetime(validation_df["ts"], unit="ms", utc=True)
                    validation_df = validation_df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
            except Exception:  # noqa: BLE001
                validation_df = None
            result = await engine.iteration.iterate(
                target, perf, snap, backtests, previous,
                validation_df=validation_df,
                validation_symbol=snap.get("symbol", "BTC/USDT"),
                validation_timeframe=snap.get("timeframe", "1h"),
                require_improve=True,
                goal=(payload or {}).get("goal", ""))
            if result is None:
                raise RuntimeError("AI 迭代失败（未配置 AI 或调用出错）")
            _ai_progress(task_id, "validate", "校验输出并注册", 80,
                         "参数已过量化规则校验与回测验证")
            result["market_features"] = {
                "sr": snap.get("indicators", {}).get("sr"),
                "pa": snap.get("indicators", {}).get("pa"),
            }
            _ai_finish(task_id, result=result)
        except Exception as e:  # noqa: BLE001
            log.exception("[ai] 策略迭代失败")
            _ai_finish(task_id, error=str(e))

    _run_ai_task_background(task_id, engine, _run, _AI_SEM)
    return {"ok": True, "task_id": task_id}


@router.get("/strategies")
async def list_ai_strategies(db: Database = Depends(get_db)):
    """列出所有 AI 设计并保存过的策略。"""
    from core.database import AiStrategy
    from sqlalchemy import select
    from strategies.pine_utils import resolve_pine
    async with db.session() as s:
        rows = (await s.execute(select(AiStrategy).order_by(AiStrategy.id.desc()).limit(50))).scalars().all()
        out = []
        for r in rows:
            spec = json.loads(r.spec_json or "{}")
            # 老记录可能存的是图上零标记的代码，也可能压根没存：resolve_pine 一次
            # 完成「补标记 / 按执行器模板生成 / 给出不给代码的原因」，
            # 前端据此显示买卖点代码或原因，而不是一个空白面板
            code, note = resolve_pine(spec)
            spec["pine_code"] = code
            if note:
                spec["pine_note"] = note
            out.append({"id": r.id, "name": r.name,
                        "created_at": r.created_at.isoformat(), "spec": spec})
        return out


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