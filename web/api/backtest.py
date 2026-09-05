"""回测：后台线程执行（向量化引擎 + GPU 加速），实时进度推送，结果入库。

- 使用 fast_engine（向量化指标预计算，可选 GPU）
- 进度通过内存缓存 _PROGRESS 推送，前端轮询 /progress/{task_id}
"""
import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.database import Database
from web.deps import get_db, get_engine, resolve_data_path
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


def _require_backtest_data(df, source: str = "exchange"):
    """回测前确认数据源确实返回了足够的 OHLCV 数据。"""
    source_label = {"exchange": "交易所", "csv": "CSV", "demo": "演示数据"}.get(source, source)
    size = len(df) if df is not None else 0
    if size == 0:
        raise ValueError(f"{source_label}未返回有效 K 线")
    if size < 60:
        raise ValueError(f"{source_label}数据不足，至少需要 60 根 K 线（当前 {size} 根）")
    return df

# 实时进度缓存: {task_id: {i, n, price, equity, cash, position, last_trade, trades, done, elapsed}}
_PROGRESS: dict[int, dict] = {}
_PROGRESS_LOCK = threading.Lock()

# P2-3：回测类后台任务并发上限（CPU 密集线程，无上限可无限叠加拖垮服务器）。
# 非阻塞 acquire 失败 → 端点返回 429（新增状态码，不影响既有字段）
_BT_SEM = threading.Semaphore(3)

# P2：共享线程池，避免 grid-scan/compare 每请求新建池（线程创建开销 + 无限并发风险）。
# 与 _BT_SEM 配合：池内线程数 <= 并发上限，不额外竞争。
_BT_POOL = ThreadPoolExecutor(max_workers=4)

# 单次策略对比的上限：候选含内置+全部动态策略，过多会长时间占满共享回测线程池
_COMPARE_MAX = 12


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
    until: str = ""                 # 结束时间（含当日；空=不限制）
    limit: int = Field(default=1000, ge=100, le=5000)   # 封顶防任意大值阻塞
    use_gpu: bool = False             # 是否启用 GPU 加速（需 cupy）
    stream: bool = True               # 是否推送实时进度
    limit_order_model: str = "none"   # "none" | "partial" | "probabilistic"
    # ---- P0 成本/成交约束（缺省 = 沿用 fee_rate，向后兼容） ----
    maker_fee_rate: Optional[float] = None   # 限价单费率（None → fee_rate）
    taker_fee_rate: Optional[float] = None   # 市价单费率（None → fee_rate）
    funding_rate: float = 0.0                # 每根K线按持仓价值收取的资金费率
    participation_rate: float = 0.0          # 买单成交量参与率上限（0=不限）
    intrabar_stops: bool = True              # intrabar 止损/止盈触价建模
    cleaning_mode: str = "mark"              # 数据清洗：mark（默认标记）| replace


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
    limit_order_model: str = "none"   # "none" | "partial" | "probabilistic"
    # P2-10：并行扫描 worker 数（默认 1=串行；>1 用进程池，回测是纯 Python 撮合循环，
    # GIL 下线程不并行，多进程才能真正并行。CPU 密集场景建议 2~4）
    workers: int = Field(default=1, ge=1, le=8)
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    funding_rate: float = 0.0
    participation_rate: float = 0.0
    intrabar_stops: bool = True
    cleaning_mode: str = "mark"


class CostScanRequest(BacktestRequest):
    """成本压力测试：同一策略在费用/滑点放大倍数下的敏感性扫描（P1-4）。"""
    fee_mults: list[float] = [1.0, 2.0, 3.0, 5.0]
    slip_mults: list[float] = [1.0, 2.0, 5.0]


class PortfolioRequest(BaseModel):
    """多标的组合层回测请求（P2-11）。

    symbols: 标的列表（≥2）；data_source=demo 时对每个标的生成相同演示序列
    （相关矩阵=1 属预期，用于验证组合口径）；exchange/csv 时按标的加载真实数据。
    weights: equal（等权）| vol_inv（波动率倒数）。
    """
    data_source: str = "demo"          # demo / csv / exchange
    csv_paths: dict = {}               # {symbol: csv_path}（data_source=csv）
    exchange: str = "binance"
    symbols: list[str] = ["BTC/USDT", "ETH/USDT"]
    timeframe: str = "1h"
    strategy_name: str = "dual_ma"
    strategy_params: dict = {}
    weights: str = "equal"             # equal | vol_inv
    vol_lookback: int = 120
    start_cash: float = 20000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    funding_rate: float = 0.0
    participation_rate: float = 0.0
    intrabar_stops: bool = True
    cleaning_mode: str = "mark"
    limit: int = Field(default=1000, ge=100, le=5000)


def _grid_run_one(df, body: GridScanIn, params: dict) -> dict | None:
    """网格扫描单组合回测（模块级顶层函数：ProcessPoolExecutor 需要可 pickle）。

    df/body 由调用方传入：多进程 worker 里各自持有副本（Windows spawn 下
    DataFrame 经 pickle 传递；组合多且数据大时拷贝代价显著，见 P2-10 权衡）。
    """
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast
    try:
        cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                             strategy_name=body.strategy_name, strategy_params=params,
                             start_cash=body.start_cash, fee_rate=body.fee_rate,
                             slippage=body.slippage,
                             limit_order_model=body.limit_order_model,
                             maker_fee_rate=body.maker_fee_rate,
                             taker_fee_rate=body.taker_fee_rate,
                             funding_rate=body.funding_rate,
                             participation_rate=body.participation_rate,
                             intrabar_stops=body.intrabar_stops,
                             cleaning_mode=body.cleaning_mode)
        r = run_backtest_fast(df, cfg, bootstrap=False)  # P2-13 扫描只消费基础指标
        m = r["metrics"]
        return {"params": params,
                "total_return": m["total_return"], "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
                "total_trades": m["total_trades"], "profit_factor": m["profit_factor"]}
    except Exception:  # noqa: BLE001
        return None


@router.post("/run")
async def run_backtest_api(body: BacktestRequest, db: Database = Depends(get_db),
                           engine=Depends(get_engine)):
    """启动异步回测，立即返回任务 id。"""
    # P2-3：回测类后台任务并发上限 2，超限 429（前端 toast 可读）
    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 2 个并发回测，请稍后再试")
    try:
        task_id = await _schedule_backtest(body, db, engine)
    except Exception:
        _BT_SEM.release()  # 调度失败（如线程启动异常）不占名额
        raise
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


def _run_on_loop(engine, coro_factory, timeout=30.0):
    """把引擎/DB 协程调度回主循环执行（后台线程调用；阻塞等待结果）。

    SQLAlchemy 异步引擎连接池绑定 lifespan 主循环；worker 线程新开事件循环
    直接跑 db.session() 会跨 loop 复用连接池 → 间歇性 RuntimeError（回测落库
    静默失败）。统一经 run_coroutine_threadsafe 调度回主循环执行。
    注意：timeout 超时后底层协程无法取消（threadsafe future 语义），仍会迟到
    完成写入——属安全收敛，调用方日志需按"可能迟到落库"口径处理。
    """
    if engine is None or engine._loop is None:
        raise RuntimeError("主循环引用不可用")
    fut = asyncio.run_coroutine_threadsafe(coro_factory(), engine._loop)
    # 取走迟到协程的最终异常（超时后无人再 result()，否则打
    # "Future exception was never retrieved" 噪音日志）
    fut.add_done_callback(lambda f: f.exception())
    return fut.result(timeout=timeout)


async def _schedule_backtest(body: BacktestRequest, db: Database, engine) -> int:
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
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
        """同步工作线程：数据加载 + 回测 + 入库。

        worker 专属事件循环仅用于网络型数据加载（纯网络无 DB）；DB 落库统一
        经 _run_on_loop 调度回主循环（连接池绑定 lifespan 主循环，曾新开事件
        循环直接跑 session → 连接池跨 loop 复用间歇性 RuntimeError）。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            try:
                if body.data_source == "demo":
                    df = generate_demo(timeframe=body.timeframe)
                elif body.data_source == "csv":
                    df = _load_csv_guarded(body.csv_path)
                elif body.data_source == "exchange":
                    df = loop.run_until_complete(
                        load_klines_cached(body.exchange, body.symbol, body.timeframe,
                                           limit=body.limit))
                else:
                    raise ValueError("未知数据源")
                _require_backtest_data(df, body.data_source)
                cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                     strategy_name=body.strategy_name, strategy_params=body.strategy_params,
                                     start_cash=body.start_cash, fee_rate=body.fee_rate,
                                     slippage=body.slippage, start=body.since or None, end=body.until or None,
                                     limit_order_model=body.limit_order_model,
                                     maker_fee_rate=body.maker_fee_rate,
                                     taker_fee_rate=body.taker_fee_rate,
                                     funding_rate=body.funding_rate,
                                     participation_rate=body.participation_rate,
                                     intrabar_stops=body.intrabar_stops,
                                     cleaning_mode=body.cleaning_mode)
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

                def _save():
                    # 更新占位记录（保持 task_id 不变，前端才能按 task_id 查到结果）；
                    # 落库调度回主循环执行（曾 asyncio.new_event_loop 四件套跨 loop 用连接池）
                    async def _inner():
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
                    _run_on_loop(engine, _inner)

                try:
                    _save()
                except TimeoutError:
                    # 主循环被长任务阻塞（如 AI 网络调用）时调度超时：协程仍在主循环
                    # 排队，会迟到落库——进度已 done，按"可能迟到"告警而非判失败
                    log.warning("[backtest] 结果落库超时（可能迟到落库，进度已在内存完成）")
                except Exception as e:  # noqa: BLE001
                    # 落库失败不判任务失败：进度缓存已 done（宁可进度可见也不冒险跨 loop）
                    log.warning("[backtest] 结果落库失败（仅保留内存进度）: %s", e)
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

                def _save_err():
                    # 失败标记落库同样调度回主循环执行
                    async def _inner():
                        async with db.session() as s:
                            row = (await s.execute(select(BacktestResult).where(BacktestResult.id == task_id))).scalar_one_or_none()
                            if row is None:
                                row = BacktestResult(id=task_id)
                                s.add(row)
                            row.config_json = json.dumps(body.model_dump(), ensure_ascii=False)
                            row.metrics_json = json.dumps({"error": str(e)})
                            await s.commit()
                    _run_on_loop(engine, _inner)

                try:
                    _save_err()
                except TimeoutError:
                    log.warning("[backtest] 错误落库超时（可能迟到落库，进度已在内存完成）")
                except Exception as e2:  # noqa: BLE001
                    log.warning("[backtest] 错误落库失败（仅保留内存进度）: %s", e2)
        finally:
            # P2-3：任务结束释放并发名额（无论成败，先释放再清理）
            _BT_SEM.release()
            # P1-4 配套（模块 C 在 ai/client.py 提供 AIClient.close_loop(loop)）：
            # 关闭并移除本 worker 循环上创建的 httpx client（当前回测路径不调用
            # AIClient，防御性清理；C 未合入时 AttributeError 被吞，联调后生效）
            try:
                if engine is not None and getattr(engine, "ai_client", None) is not None:
                    loop.run_until_complete(engine.ai_client.close_loop(loop))
            except Exception:  # noqa: BLE001
                pass
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
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
    from backtest.overfit import OverfitConfig, detect_overfit

    if body.data_source == "demo":
        df = generate_demo(timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = _load_csv_guarded(body.csv_path)
    elif body.data_source == "exchange":
        df = await load_klines_cached(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")

    try:
        _require_backtest_data(df, body.data_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if len(df) < 600:
        raise HTTPException(status_code=400, detail="过拟合检测需要至少 600 根K线（训练+验证）")

    cfg = OverfitConfig(symbol=body.symbol, timeframe=body.timeframe,
                        start_cash=body.start_cash, fee_rate=body.fee_rate,
                        slippage=body.slippage)
    # 获取并发上限许可（CSCV/PBO 计算量大，需纳入上限）
    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 3 个并发回测，请稍后再试")
    try:
        # CSCV/PBO 计算量较大（多组合×多参数回测），放后台线程避免冻结事件循环
        report = await asyncio.to_thread(detect_overfit, df, body.strategy_name,
                                         body.strategy_params, cfg)
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
            "dsr": report.dsr,
            "dsr_trials": report.dsr_trials,
            "sharpe_ci": report.sharpe_ci,
            "ret_ci": report.ret_ci,
            "p_sharpe_pos": report.p_sharpe_pos,
            "flags": report.flags,
            "folds": [{"fold": f.fold, "is_ret": f.is_ret, "oos_ret": f.oos_ret,
                       "oos_sharpe": f.oos_sharpe, "oos_drawdown": f.oos_drawdown,
                       "oos_trades": f.oos_trades,
                       "is_start": f.is_start, "oos_start": f.oos_start} for f in report.folds],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _BT_SEM.release()


@router.post("/grid-scan")
async def grid_scan(body: GridScanIn):
    """参数网格扫描：自动尝试参数组合，找出绩效最优者（配合过拟合检测防作弊）。

    对每个参数组合跑回测，按总收益+夏普综合排序，返回 Top-N。
    用线程池并发加速；结果仅返回指标，不入库。
    """
    import itertools
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    # 加载数据（demo/csv 的 pandas 解析放后台线程）
    if body.data_source == "demo":
        df = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = await asyncio.to_thread(_load_csv_guarded, body.csv_path)
    elif body.data_source == "exchange":
        df = await load_klines_cached(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")
    try:
        _require_backtest_data(df, body.data_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
            return _grid_run_one(df, body, params)
        except Exception:  # noqa: BLE001
            return None

    # 获取并发上限许可（非阻塞：429 快速失败，避免排队积压）
    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 3 个并发回测，请稍后再试")
    try:
        results = []
        # P2-10：并行扫描。workers>1 走进程池（回测撮合是 Python 循环，GIL 下
        # 线程不并行，多进程才真正并行）；workers==1 用共享线程池向后兼容。
        # 共享 df 进多进程的拷贝代价 vs 串行收益：组合数 × 回测时长权衡，
        # 组合多（>8）且数据量大时进程池收益明显
        if body.workers > 1:
            import concurrent.futures as _cf
            import functools as _ft
            # 用模块级函数 + partial（df/body 经 pickle 传入），不能传局部闭包
            _worker = _ft.partial(_grid_run_one, df, body)
            with _cf.ProcessPoolExecutor(max_workers=body.workers) as pool:
                for r in await asyncio.to_thread(
                        lambda: list(pool.map(_worker, combos, chunksize=1))):
                    if r is not None:
                        results.append(r)
        else:
            for r in await asyncio.to_thread(lambda: list(_BT_POOL.map(_run, combos))):
                if r is not None:
                    results.append(r)
    finally:
        _BT_SEM.release()

    # 综合评分：收益 + 夏普 - 回撤惩罚，防只看收益的过拟合陷阱
    for r in results:
        score = (r["total_return"] * 10 + r["sharpe"] * 2 - r["max_drawdown"] * 5)
        r["score"] = round(score, 4)
    results.sort(key=lambda x: -x["score"])
    best = results[0] if results else None

    # 最优参数的缩水夏普 DSR：N 个组合里挑最大，需多重测试校正后
    # 才能回答"这个最优是实力还是运气"（trial_sharpes=全部候选的单周期SR）
    dsr_info = None
    if best is not None:
        try:
            import math as _math
            from backtest.metrics import PERIODS_PER_YEAR, deflated_sharpe_ratio, equity_returns
            ppy = PERIODS_PER_YEAR.get(body.timeframe, 8760)
            cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                 strategy_name=body.strategy_name, strategy_params=best["params"],
                                 start_cash=body.start_cash, fee_rate=body.fee_rate,
                                 slippage=body.slippage,
                                 limit_order_model=body.limit_order_model)
            rr = await asyncio.to_thread(run_backtest_fast, df, cfg)
            rets = equity_returns(rr.get("equity_curve") or [])
            trial_srs = [x["sharpe"] / _math.sqrt(ppy) for x in results
                         if x.get("sharpe") is not None and _math.isfinite(x["sharpe"])]
            dsr_info = deflated_sharpe_ratio(rets, trials=len(combos), trial_sharpes=trial_srs)
        except Exception:  # noqa: BLE001
            dsr_info = None

    return {"ok": True, "strategy": body.strategy_name,
            "scanned": len(results), "total_combos": len(combos),
            "results": results[:body.top_n], "best": best, "dsr": dsr_info}


@router.post("/compare")
async def compare_strategies(body: BacktestRequest):
    """策略对比：同一数据上并列跑所有内置策略，输出绩效对比表（辅助选型）。"""
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast
    from strategies import list_strategies

    if body.data_source == "demo":
        df = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = await asyncio.to_thread(_load_csv_guarded, body.csv_path)
    elif body.data_source == "exchange":
        df = await load_klines_cached(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")
    try:
        _require_backtest_data(df, body.data_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 对比全部可执行策略：内置 + AI/迭代/DRL 动态策略。
    # 曾只跑内置策略——用户训练出的 AI 策略从不参与选型对比。
    # 内置 rl_adaptive 无 model_path（跑不出意义）故排除；动态 rl_* 自带模型路径照常参与。
    candidates = [s for s in list_strategies()
                  if s["name"] != "rl_adaptive" or (s.get("default_params") or {}).get("model_path")]
    candidates = candidates[:_COMPARE_MAX]
    if not candidates:
        raise HTTPException(status_code=400, detail="无可对比的策略")

    def _run(st: dict) -> dict:
        try:
            cfg = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                                 strategy_name=st["name"], strategy_params={},
                                 start_cash=body.start_cash, fee_rate=body.fee_rate,
                                 slippage=body.slippage,
                                 limit_order_model=body.limit_order_model)
            r = run_backtest_fast(df, cfg, bootstrap=False)  # P2-13 对比扫描只消费基础指标
            m = r["metrics"]
            return {"name": st["name"], "description": st["description"],
                    "total_return": m["total_return"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
                    "total_trades": m["total_trades"], "profit_factor": m["profit_factor"]}
        except Exception:  # noqa: BLE001
            return None

    # 获取并发上限许可
    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 3 个并发回测，请稍后再试")
    try:
        results = []
        # 使用共享线程池
        for r in await asyncio.to_thread(lambda: list(_BT_POOL.map(_run, candidates))):
            if r is not None:
                results.append(r)
    finally:
        _BT_SEM.release()
    results.sort(key=lambda x: -(x["total_return"] * 10 + x["sharpe"] * 2 - x["max_drawdown"] * 5))
    return {"ok": True, "symbol": body.symbol, "timeframe": body.timeframe,
            "compared": len(candidates), "skipped": len(candidates) - len(results),
            "results": results}


@router.post("/cost-scan")
async def cost_scan_api(body: CostScanRequest):
    """成本压力测试（P1-4）：同一策略在费用/滑点放大倍数组合下的敏感性扫描。

    回答："这个策略是赚趋势钱，还是靠低摩擦才有利润？"
    若收益随成本快速恶化 → 成本敏感、高换手依赖，需降低频率/加大持仓时长。
    返回逐组合的收益/夏普/回撤与基准对照表。
    """
    from backtest.cost_scan import CostScanConfig, run_cost_scan
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached

    if body.data_source == "demo":
        df = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    elif body.data_source == "csv":
        df = await asyncio.to_thread(_load_csv_guarded, body.csv_path)
    elif body.data_source == "exchange":
        df = await load_klines_cached(body.exchange, body.symbol, body.timeframe,
                                      limit=body.limit)
    else:
        raise HTTPException(status_code=400, detail="未知数据源")
    try:
        _require_backtest_data(df, body.data_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 校验放大倍数合理（防 0/负值导致 fee_rate 无效）
    fee_mults = [float(m) for m in body.fee_mults if float(m) > 0]
    slip_mults = [float(m) for m in body.slip_mults if float(m) > 0]
    if not fee_mults or not slip_mults:
        raise HTTPException(status_code=400, detail="费用/滑点放大倍数必须为正")
    # 确保基准组合 1x1 在表内
    if 1.0 not in fee_mults:
        fee_mults = [1.0] + fee_mults
    if 1.0 not in slip_mults:
        slip_mults = [1.0] + slip_mults
    if len(fee_mults) * len(slip_mults) > 60:
        raise HTTPException(status_code=400, detail="扫描组合过多（≤60 组）")

    scan_cfg = CostScanConfig(
        symbol=body.symbol, timeframe=body.timeframe,
        strategy_name=body.strategy_name, strategy_params=body.strategy_params,
        start_cash=body.start_cash, base_fee_rate=body.fee_rate,
        base_slippage=body.slippage, fee_mults=tuple(fee_mults),
        slip_mults=tuple(slip_mults))

    # 获取并发上限许可
    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 3 个并发回测，请稍后再试")
    try:
        report = await asyncio.to_thread(run_cost_scan, df, scan_cfg)
    finally:
        _BT_SEM.release()

    return {"ok": True, "strategy": body.strategy_name,
            "scanned": len(report["scan"]), "base": report["base"],
            "sensitivity": report["sensitivity"], "scan": report["scan"]}


@router.post("/portfolio")
async def portfolio_api(body: PortfolioRequest):
    """多标的组合层回测（P2-11）：逐标的回测 → 加权组合权益 → 相关性矩阵。

    研究层组合视角：回答"多标的叠加后组合夏普/回撤/分散度"。
    支持 equal / vol_inv 权重。demo 源对每标的生成相同演示序列
    （相关矩阵≈1 属预期，用于验证口径）；exchange/csv 按标的加载真实数据。
    """
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
    from backtest.portfolio import PortfolioConfig, run_portfolio_backtest

    symbols = [s for s in body.symbols if s and s.strip()]
    if len(symbols) < 2:
        raise HTTPException(status_code=400, detail="组合回测需要至少 2 个标的")
    if len(symbols) > 20:
        raise HTTPException(status_code=400, detail="标的过多（≤20）")

    # 加载各标的数据
    data = {}
    for s in symbols:
        if body.data_source == "demo":
            data[s] = await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
        elif body.data_source == "csv":
            csv_path = (body.csv_paths or {}).get(s)
            if not csv_path:
                raise HTTPException(status_code=400, detail=f"缺少 {s} 的 csv 路径")
            data[s] = await asyncio.to_thread(_load_csv_guarded, csv_path)
        elif body.data_source == "exchange":
            data[s] = await load_klines_cached(body.exchange, s, body.timeframe, limit=body.limit)
        else:
            raise HTTPException(status_code=400, detail="未知数据源")
        try:
            _require_backtest_data(data[s], body.data_source)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{s}: {e}")

    pcfg = PortfolioConfig(
        symbols=symbols, strategy_name=body.strategy_name,
        strategy_params=body.strategy_params, weights=body.weights,
        vol_lookback=body.vol_lookback, start_cash=body.start_cash,
        timeframe=body.timeframe, fee_rate=body.fee_rate, slippage=body.slippage,
        maker_fee_rate=body.maker_fee_rate, taker_fee_rate=body.taker_fee_rate,
        funding_rate=body.funding_rate, participation_rate=body.participation_rate,
        intrabar_stops=body.intrabar_stops, cleaning_mode=body.cleaning_mode)

    if not _BT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="回测任务繁忙：最多 3 个并发回测，请稍后再试")
    try:
        report = await asyncio.to_thread(run_portfolio_backtest, data, pcfg)
    finally:
        _BT_SEM.release()

    return {"ok": True, "weights": report["weights"],
            "correlation": report["correlation"],
            "benchmark": report["benchmark"],
            "portfolio_metrics": report["portfolio_metrics"],
            "portfolio_equity_curve": report["portfolio_equity_curve"],
            "per_symbol": [{"symbol": ps["symbol"], "strategy": ps["strategy"],
                            "weight": ps["weight"], "metrics": ps["metrics"]}
                           for ps in report["per_symbol"]]}
