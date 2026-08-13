"""回测：后台线程执行（向量化引擎 + GPU 加速），实时进度推送，结果入库。

- 使用 fast_engine（向量化指标预计算，可选 GPU）
- 进度通过内存缓存 _PROGRESS 推送，前端轮询 /progress/{task_id}
"""
import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.database import Database
from web.deps import get_db, resolve_data_path
from web.api.tasks_cache import mark_done, prune_task_cache

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _load_csv_guarded(csv_path: str):
    """加载 CSV：先校验路径必须位于 data/ 目录内（防任意文件读取）。"""
    from backtest.data_loader import load_csv
    try:
        return load_csv(str(resolve_data_path(csv_path)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 实时进度缓存: {task_id: {i, n, price, equity, cash, position, last_trade, trades, done, elapsed}}
_PROGRESS: dict[int, dict] = {}
_PROGRESS_LOCK = threading.Lock()


class BacktestRequest(BaseModel):
    data_source: str = "demo"          # demo / csv / exchange
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy_name: str = "dual_ma"
    strategy_params: dict = {}
    start_cash: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    since: str = ""
    limit: int = Field(default=1000, ge=100, le=5000)   # 封顶防任意大值阻塞
    use_gpu: bool = False             # 是否启用 GPU 加速（需 cupy）
    stream: bool = True               # 是否推送实时进度


class GridScanIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy_name: str = "dual_ma"
    param_grid: dict = {}             # {参数名: [候选值]}
    base_params: dict = {}            # 非扫描参数的固定值
    start_cash: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    limit: int = Field(default=1000, ge=100, le=5000)
    top_n: int = 10                   # 返回绩效最好的前 N 组


@router.post("/run")
async def run_backtest_api(body: BacktestRequest, db: Database = Depends(get_db)):
    """启动异步回测，立即返回任务 id。"""
    task_id = await _schedule_backtest(body, db)
    return {"ok": True, "task_id": task_id}


@router.get("/progress/{task_id}")
async def get_progress(task_id: int):
    """读取回测实时进度（K线数据 + 成交 + 盈亏）。"""
    with _PROGRESS_LOCK:
        p = _PROGRESS.get(int(task_id))
    if p is None:
        return {"found": False}
    return {"found": True, **p}


def _make_progress_cb(task_id: int, n_total: int, symbol: str):
    """返回进度回调：写入 _PROGRESS 缓存（含最近K线/成交）。"""
    def cb(i: int, n: int, ctx: dict, trade: dict | None) -> None:
        ind = ctx.get("indicators", {})
        item = {
            "i": i, "n": n, "pct": round(i / max(1, n) * 100, 2),
            "price": ctx.get("price"), "cash": ctx.get("cash"),
            "position": ctx.get("position"),
            "close": ind.get("close"), "ma_fast": ind.get("ma_fast"),
            "ma_slow": ind.get("ma_slow"), "rsi": ind.get("rsi"),
            "macd": ind.get("macd"), "symbol": symbol,
            "ts": _fmt_ts(i),
        }
        if trade:
            item["last_trade"] = trade
        with _PROGRESS_LOCK:
            # 只保留最近 200 根K线快照，避免内存膨胀
            hist = _PROGRESS.get(task_id, {}).get("kline_history", [])
            hist.append(item)
            if len(hist) > 200:
                hist = hist[-200:]
            _PROGRESS[task_id] = {
                **_PROGRESS.get(task_id, {}),
                "i": i, "n": n, "pct": item["pct"], "last_price": item["price"],
                "kline_history": hist, "last_item": item,
                "running": True,
            }
        # 防内存膨胀：任务数超上限时清理过期已完成任务（每 200 根一次，
        # 避免长回测（2 万根）每根 K 线都加锁扫描缓存）
        if i % 200 == 0:
            prune_task_cache(_PROGRESS, _PROGRESS_LOCK)
    return cb


def _fmt_ts(i: int) -> str:
    return f"K{i}"


async def _schedule_backtest(body: BacktestRequest, db: Database) -> int:
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    # 写一条占位记录拿到 id
    from core.database import BacktestResult
    async with db.session() as s:
        row = BacktestResult(config_json=json.dumps(body.model_dump(), ensure_ascii=False),
                             metrics_json=json.dumps({"status": "running"}))
        s.add(row)
        await s.commit()
        await s.refresh(row)
        task_id = row.id

    # 初始化进度缓存
    with _PROGRESS_LOCK:
        _PROGRESS[task_id] = {"running": True, "pct": 0, "kline_history": [], "i": 0, "n": 0}

    def _sync_worker():
        """同步工作线程：数据加载 + 回测 + 入库。"""
        try:
            if body.data_source == "demo":
                df = generate_demo(timeframe=body.timeframe)
            elif body.data_source == "csv":
                df = _load_csv_guarded(body.csv_path)
            elif body.data_source == "exchange":
                df = asyncio.run(load_from_exchange(body.exchange, body.symbol, body.timeframe,
                                                    limit=body.limit))
            else:
                raise ValueError("未知数据源")
            cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                 strategy_name=body.strategy_name, strategy_params=body.strategy_params,
                                 start_cash=body.start_cash, fee_rate=body.fee_rate,
                                 slippage=body.slippage, start=body.since or None)
            # 选择后端：显式 GPU，但 GPU 真正不可用时自动回退 numpy
            from indicators.vectorized import gpu_available
            if body.use_gpu:
                if gpu_available():
                    backend = "cupy"
                else:
                    backend = "numpy"
                    log.warning("[backtest] GPU 不可用（缺 CUDA 头文件），回退 numpy 后端")
            else:
                backend = "numpy"
            cb = _make_progress_cb(task_id, len(df), body.symbol) if body.stream else None
            result = run_backtest_fast(df, cfg, on_progress=cb, backend=backend)

            # 更新进度为完成
            with _PROGRESS_LOCK:
                p = _PROGRESS.get(task_id, {})
                p["running"] = False
                p["pct"] = 100
                p["done"] = True
                p["metrics"] = result["metrics"]
                p["backend"] = result["metrics"].get("backtest_engine", backend)
                p["elapsed_sec"] = result["metrics"].get("elapsed_sec")
                _PROGRESS[task_id] = mark_done(p)
            prune_task_cache(_PROGRESS, _PROGRESS_LOCK)

            from core.database import BacktestResult
            from sqlalchemy import select
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _save():
                    # 更新占位记录（保持 task_id 不变，前端才能按 task_id 查到结果）
                    async with db.session() as s:
                        row = (await s.execute(select(BacktestResult).where(BacktestResult.id == task_id))).scalar_one_or_none()
                        if row is None:
                            row = BacktestResult(id=task_id)
                            s.add(row)
                        row.config_json = json.dumps(body.model_dump(), ensure_ascii=False)
                        row.metrics_json = json.dumps(result["metrics"], ensure_ascii=False)
                        row.equity_curve_json = json.dumps(result["equity_curve"])
                        row.trades_json = json.dumps(result["trades"], ensure_ascii=False)
                        await s.commit()
                loop.run_until_complete(_save())
            finally:
                loop.close()
        except Exception as e:  # noqa: BLE001
            log.exception("[backtest] 回测失败")
            with _PROGRESS_LOCK:
                p = _PROGRESS.get(task_id, {})
                p["running"] = False
                p["error"] = str(e)
                _PROGRESS[task_id] = mark_done(p)
            prune_task_cache(_PROGRESS, _PROGRESS_LOCK)
            from core.database import BacktestResult
            from sqlalchemy import select
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _save_err():
                    async with db.session() as s:
                        row = (await s.execute(select(BacktestResult).where(BacktestResult.id == task_id))).scalar_one_or_none()
                        if row is None:
                            row = BacktestResult(id=task_id)
                            s.add(row)
                        row.config_json = json.dumps(body.model_dump(), ensure_ascii=False)
                        row.metrics_json = json.dumps({"error": str(e)})
                        await s.commit()
                loop.run_until_complete(_save_err())
            finally:
                loop.close()

    # 线程池执行，避免阻塞事件循环
    threading.Thread(target=_sync_worker, daemon=True).start()
    return task_id


@router.get("/results")
async def list_results(limit: int = 20, db: Database = Depends(get_db)):
    from core.database import BacktestResult
    from sqlalchemy import select
    async with db.session() as s:
        rows = (await s.execute(select(BacktestResult).order_by(BacktestResult.id.desc()).limit(limit))).scalars().all()
        out = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            out.append({"id": r.id, "created_at": r.created_at.isoformat(),
                        "status": m.get("status", "done"),
                        "total_return": m.get("total_return"), "sharpe": m.get("sharpe"),
                        "max_drawdown": m.get("max_drawdown"), "win_rate": m.get("win_rate"),
                        "error": m.get("error"), "elapsed_sec": m.get("elapsed_sec")})
        return out


@router.get("/results/{result_id}")
async def get_result(result_id: int, db: Database = Depends(get_db)):
    from core.database import BacktestResult
    from sqlalchemy import select
    async with db.session() as s:
        row = (await s.execute(select(BacktestResult).where(BacktestResult.id == result_id))).scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="回测结果不存在")
        return {
            "id": row.id,
            "config": json.loads(row.config_json or "{}"),
            "metrics": json.loads(row.metrics_json or "{}"),
            "equity_curve": json.loads(row.equity_curve_json or "[]"),
            "trades": json.loads(row.trades_json or "[]"),
            "created_at": row.created_at.isoformat(),
        }


@router.post("/overfit")
async def detect_overfit_api(body: BacktestRequest):
    """过拟合检测：前推验证 + PBO 过拟合概率，识别 AI/参数搜索作弊。

    在运行后自动追加到回测结果，或独立调用。
    """
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.overfit import OverfitConfig, detect_overfit

    if body.data_source == "demo":
        df = generate_demo(timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = _load_csv_guarded(body.csv_path)
    elif body.data_source == "exchange":
        df = await load_from_exchange(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")

    if len(df) < 600:
        raise HTTPException(status_code=400, detail="过拟合检测需要至少 600 根K线（训练+验证）")

    cfg = OverfitConfig(symbol=body.symbol, timeframe=body.timeframe,
                        start_cash=body.start_cash, fee_rate=body.fee_rate,
                        slippage=body.slippage)
    try:
        # CSCV/PBO 计算量较大（多组合×多参数回测），放后台线程避免冻结事件循环
        report = await asyncio.to_thread(detect_overfit, df, body.strategy_name,
                                         body.strategy_params, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "verdict": report.verdict,
        "score": report.score,
        "is_ret": report.is_ret,
        "oos_ret": report.oos_ret,
        "decay": report.decay,
        "pbo": report.pbo,
        "oos_sharpe": report.oos_sharpe,
        "stability": report.stability,
        "n_folds": report.n_folds,
        "flags": report.flags,
        "folds": [{"fold": f.fold, "is_ret": f.is_ret, "oos_ret": f.oos_ret,
                   "oos_sharpe": f.oos_sharpe, "oos_drawdown": f.oos_drawdown,
                   "oos_trades": f.oos_trades,
                   "is_start": f.is_start, "oos_start": f.oos_start} for f in report.folds],
    }


@router.post("/grid-scan")
async def grid_scan(body: GridScanIn):
    """参数网格扫描：自动尝试参数组合，找出绩效最优者（配合过拟合检测防作弊）。

    对每个参数组合跑回测，按总收益+夏普综合排序，返回 Top-N。
    用线程池并发加速；结果仅返回指标，不入库。
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    # 加载数据（demo/csv 的 pandas 解析放后台线程）
    if body.data_source == "demo":
        df = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = await asyncio.to_thread(_load_csv_guarded, body.csv_path)
    elif body.data_source == "exchange":
        df = await load_from_exchange(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")

    # 生成参数组合
    keys = list(body.param_grid.keys())
    if not keys:
        raise HTTPException(status_code=400, detail="param_grid 不能为空")
    values = [body.param_grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        params = dict(body.base_params or {})
        for k, v in zip(keys, combo):
            params[k] = v
        combos.append(params)
    if len(combos) > 200:
        raise HTTPException(status_code=400, detail=f"参数组合过多({len(combos)})，请减少扫描范围（≤200）")

    def _run(params: dict) -> dict:
        try:
            cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                 strategy_name=body.strategy_name, strategy_params=params,
                                 start_cash=body.start_cash, fee_rate=body.fee_rate,
                                 slippage=body.slippage)
            r = run_backtest_fast(df, cfg)
            m = r["metrics"]
            return {"params": params,
                    "total_return": m["total_return"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
                    "total_trades": m["total_trades"], "profit_factor": m["profit_factor"]}
        except Exception:  # noqa: BLE001
            return None

    results = []
    # 注意：ThreadPoolExecutor.map 是同步迭代器，阻塞事件循环直到全部完成；
    # 整体包进 to_thread（map 内部 await 无法让出事件循环）
    with ThreadPoolExecutor(max_workers=min(8, len(combos))) as ex:
        for r in await asyncio.to_thread(lambda: list(ex.map(_run, combos))):
            if r is not None:
                results.append(r)

    # 综合评分：收益 + 夏普 - 回撤惩罚，防只看收益的过拟合陷阱
    for r in results:
        score = (r["total_return"] * 10 + r["sharpe"] * 2 - r["max_drawdown"] * 5)
        r["score"] = round(score, 4)
    results.sort(key=lambda x: -x["score"])
    best = results[0] if results else None

    return {"ok": True, "strategy": body.strategy_name,
            "scanned": len(results), "total_combos": len(combos),
            "results": results[:body.top_n], "best": best}


@router.post("/compare")
async def compare_strategies(body: BacktestRequest):
    """策略对比：同一数据上并列跑所有内置策略，输出绩效对比表（辅助选型）。"""
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast
    from strategies import list_strategies

    if body.data_source == "demo":
        df = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = await asyncio.to_thread(_load_csv_guarded, body.csv_path)
    elif body.data_source == "exchange":
        df = await load_from_exchange(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")

    # 只对比内置策略（AI/动态策略需要 model 或特殊参数，跳过）
    builtin = [s for s in list_strategies() if s.get("builtin") and s["name"] != "rl_adaptive"]
    if not builtin:
        raise HTTPException(status_code=400, detail="无可对比的内置策略")

    def _run(st: dict) -> dict:
        try:
            cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                 strategy_name=st["name"], strategy_params={},
                                 start_cash=body.start_cash, fee_rate=body.fee_rate,
                                 slippage=body.slippage)
            r = run_backtest_fast(df, cfg)
            m = r["metrics"]
            return {"name": st["name"], "description": st["description"],
                    "total_return": m["total_return"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
                    "total_trades": m["total_trades"], "profit_factor": m["profit_factor"]}
        except Exception:  # noqa: BLE001
            return None

    results = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(builtin)) as ex:
        for r in await asyncio.to_thread(lambda: list(ex.map(_run, builtin))):
            if r is not None:
                results.append(r)
    results.sort(key=lambda x: -(x["total_return"] * 10 + x["sharpe"] * 2 - x["max_drawdown"] * 5))
    return {"ok": True, "symbol": body.symbol, "timeframe": body.timeframe,
            "results": results}
