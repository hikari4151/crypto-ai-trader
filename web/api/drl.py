"""深度强化学习 API：训练动态自适应策略智能体 / 进度推送 / 模型管理。

训练在后台线程执行（与回测一致），进度写入内存缓存供前端轮询。
训练产物（智能体权重）保存为 JSON 到 data/models/，可一键应用为当前策略。
"""
import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.database import Database
from web.deps import get_db, get_engine, resolve_data_path
from web.api.tasks_cache import mark_done, prune_task_cache

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/drl", tags=["drl"])

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
PROGRESS_LOCK = threading.Lock()
_TRAINING: dict[int, dict] = {}

# P2-3：DRL 训练类后台任务（/train 线程 + /mine-factor to_thread 共用）并发上限，
# 超限 429（CPU 密集训练，无上限可无限叠加拖垮服务器）
_DRL_SEM = threading.Semaphore(2)


def _now() -> str:
    """UTC ISO 时间戳（曾引用 factors._now 造成 NameError，独立定义）。"""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _model_path(name: str) -> Path:
    """模型文件路径：名称白名单清洗（防路径遍历），并限定在 data/models 目录内。"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise HTTPException(status_code=400, detail="非法模型名")
    p = (MODEL_DIR / f"{name}.json").resolve()
    if not p.is_relative_to(MODEL_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法模型路径")
    return p


class TrainIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=1500, ge=100, le=5000)   # 封顶防任意大值阻塞
    episodes: int = 80
    hidden: list[int] = [64, 64]
    lr_actor: float = 0.004
    lr_critic: float = 0.01
    gamma: float = 0.99
    vol_penalty: float = 20.0
    cost_scale: float = 1.0
    min_trade_zone: float = 0.05
    entropy_coef: float = 0.05
    start_cash: float = 10000.0
    fee_rate: float = 0.001
    seed: int = 42
    use_gpu: bool = False     # 特征预计算走 GPU（需 cupy 可用，不可用自动回退 numpy）
    model_name: str = ""      # 自定义模型名，空则自动生成
    base_model: str = ""      # 续训：可选之前训练好的模型文件路径，作为起点继续增强
    force_train: bool = False # 强制自动训练：触发早停后不中断，继续训练到用户设置轮数
    early_stop_patience: int = 0  # 早停保护：验证收益连续 N 次无改善则提前停止（0=禁用）
    # ---- 真 PPO 训练参数 ----
    n_episodes: int = 4           # 每轮收集的轨迹数（批量平均降方差）
    ppo_epochs: int = 4           # PPO 多轮更新轮数（SB3 默认 10，小数据建议 3-5）
    mini_batch_size: int = 128    # PPO mini-batch 大小
    state_window: int = 1         # 状态窗口：堆叠最近 N 根K线特征（时序记忆，1=单点）
    factor_expression: str = ""   # 可选：注入因子信号列（因子表达式，如 close/ma(close,30)-1），
                                  # 部署时 rl_adaptive 必须配置同一表达式
    # ---- 可选奖励塑形（0=关闭；回撤/连亏/趋势一致性。QUANT_ADVICE 推荐默认开启，
    #     显式传 0 可关闭） ----
    reward_dd_penalty: float = 0.5       # 回撤加深惩罚系数（含 dd>12% 硬约束）
    reward_losing_penalty: float = 0.2   # 连亏惩罚（3 连亏起二次斜坡）
    reward_trend_align: float = 0.1      # 趋势一致性奖励（低波动趋势持仓同向加分）
    reward_val_gap_penalty: float = 0.5  # 过拟合衰减：训练收益>验证收益时衰减优势权重


def _validate_train(body: TrainIn) -> None:
    """训练参数边界校验（防 DoS/异常配置）。"""
    if not 1 <= body.episodes <= 5000:
        raise HTTPException(status_code=400, detail="episodes 须在 1-5000 之间")
    if not 1 <= len(body.hidden) <= 4 or any(not 4 <= x <= 512 for x in body.hidden):
        raise HTTPException(status_code=400, detail="hidden 须为 1-4 层，每层 4-512 维")
    if not 1 <= body.n_episodes <= 32:
        raise HTTPException(status_code=400, detail="n_episodes 须在 1-32 之间")
    if not 1 <= body.ppo_epochs <= 20:
        raise HTTPException(status_code=400, detail="ppo_epochs 须在 1-20 之间")
    if not 8 <= body.mini_batch_size <= 4096:
        raise HTTPException(status_code=400, detail="mini_batch_size 须在 8-4096 之间")
    if not 1 <= body.state_window <= 30:
        raise HTTPException(status_code=400, detail="state_window 须在 1-30 之间")
    if not 0 < body.lr_actor < 1 or not 0 < body.lr_critic < 1:
        raise HTTPException(status_code=400, detail="学习率须在 (0,1) 之间")
    if not 0 <= body.vol_penalty <= 1000:
        raise HTTPException(status_code=400, detail="vol_penalty 须在 [0,1000] 之间")


class FactorMineTrainIn(BaseModel):
    """RL 因子组合挖掘训练参数（AlphaForge 式：RL 学因子选择与权重）。"""
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 1500
    horizon: int = 1               # 预测未来几期收益
    episodes: int = 60
    n_episodes: int = 8            # 每轮收集轨迹数（因子环境轨迹短，可多收集）
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    hidden: list[int] = [64, 64]
    ic_window: int = 120           # 滚动 IC 窗口（因子表现特征）
    max_steps: int = 6             # 最多选几个因子组成组合
    corr_threshold: float = 0.85   # 冗余惩罚：与已选因子相关超阈值即罚
    lr_actor: float = 0.003
    lr_critic: float = 0.006
    seed: int = 42
    auto_register: bool = True     # 通过 OOS 安检后注册到自定义因子库


@router.post("/mine-factor")
async def mine_factor_by_rl(body: FactorMineTrainIn, db: Database = Depends(get_db)):
    """RL 因子组合挖掘：PPO 训练"因子选择智能体"，产出组合因子（接入因子体系）。

    与 /train（RL 直接出策略）不同：本端点输出"选中的因子组合 + 权重 + 合成因子"，
    由因子系统消费（回测/因子信号策略），RL 只负责学因子组合（执行层另由 rl_adaptive 承担）。
    CPU 训练在后台线程（to_thread），避免阻塞事件循环。
    """
    # 参数边界
    if not 1 <= body.episodes <= 1000:
        raise HTTPException(status_code=400, detail="episodes 须在 1-1000 之间")
    if not 1 <= body.horizon <= 30:
        raise HTTPException(status_code=400, detail="horizon 须在 1-30 之间")
    if not 30 <= body.ic_window <= 500:
        raise HTTPException(status_code=400, detail="ic_window 须在 30-500 之间")
    if not 1 <= body.max_steps <= 12:
        raise HTTPException(status_code=400, detail="max_steps 须在 1-12 之间")
    if not 0.5 <= body.corr_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="corr_threshold 须在 [0.5,1.0] 之间")
    if not 1 <= len(body.hidden) <= 4 or any(not 4 <= x <= 512 for x in body.hidden):
        raise HTTPException(status_code=400, detail="hidden 须为 1-4 层，每层 4-512 维")

    from drl.factor_miner import train_factor_miner
    from factors.engine import compute_factor_matrix

    df = await _load_df(body)  # async 版：exchange 源直接 await（曾用同步版嵌套 asyncio.run 必崩）
    if len(df) < 500:
        raise HTTPException(status_code=400, detail="RL 因子挖掘需要至少 500 根K线")
    # P2-3：DRL 训练类任务并发上限 2，超限 429（前端 toast 可读）
    if not _DRL_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="DRL 训练任务繁忙：最多 2 个并发训练，请稍后再试")
    try:
        mat = await asyncio.to_thread(compute_factor_matrix, df)
        cfg = {
            "episodes": body.episodes, "n_episodes": body.n_episodes,
            "ppo_epochs": body.ppo_epochs, "mini_batch_size": body.mini_batch_size,
            "hidden": tuple(body.hidden), "ic_window": body.ic_window,
            "max_steps": body.max_steps, "corr_threshold": body.corr_threshold,
            "h": body.horizon, "lr_actor": body.lr_actor, "lr_critic": body.lr_critic,
            "seed": body.seed,
        }
        result = await asyncio.to_thread(train_factor_miner, df, mat, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _DRL_SEM.release()

    report = result["report"]
    # 通过 OOS 安检后注册到自定义因子库（source=rl_mined）
    registered = False
    if body.auto_register and report.get("enabled") and report.get("valid"):
        name = f"rl_factor_combo_h{body.horizon}"
        custom = await db.kv_json_get("factor_custom", {})
        custom[name] = {
            "title": f"RL因子组合(h={body.horizon})",
            "category": "composite",
            "expression": f"<rl_factor_combo h={body.horizon} factors={result['selected_factors']}>",
            "logic": "PPO 训练的因子选择智能体选出的组合因子（IC 加权合成，AlphaForge 式）",
            "rank_ic": report.get("rank_ic"),
            "icir": report.get("icir"),
            "turnover": report.get("turnover"),
            "fitness": report.get("fitness"),
            "selected_factors": result["selected_factors"],
            "weights": result["weights"],
            "source": "rl_mined",
            "meta": result["meta"],
            "created_at": _now(),
        }
        await db.kv_json_set("factor_custom", custom)
        registered = True

    return {"ok": True,
            "selected_factors": result["selected_factors"],
            "weights": result["weights"],
            "report": report,
            "registered": registered,
            "meta": result["meta"]}


async def _load_df(body: TrainIn):
    """按数据源加载 OHLCV DataFrame（async：exchange 源直接 await，避免在事件循环中嵌套 asyncio.run）。"""
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached
    if body.data_source == "demo":
        return generate_demo(n=body.limit, timeframe=body.timeframe)
    if body.data_source == "csv":
        if not body.csv_path:
            raise HTTPException(status_code=400, detail="CSV 数据源需要 csv_path")
        return load_csv(str(resolve_data_path(body.csv_path)))
    if body.data_source == "exchange":
        return await load_klines_cached(body.exchange, body.symbol,
                                        body.timeframe, limit=body.limit)
    raise HTTPException(status_code=400, detail="未知数据源")


def _default_name(body: TrainIn) -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    base = body.model_name.strip() or f"rl_{body.data_source}_{ts}"
    # 避免覆盖已有文件
    name = base
    i = 1
    while (MODEL_DIR / f"{name}.json").exists():
        name = f"{base}_{i}"
        i += 1
    return name


@router.post("/train")
async def start_training(body: TrainIn, db=Depends(get_db), engine=Depends(get_engine)):
    """启动后台 DRL 训练，立即返回 task_id。"""
    _validate_train(body)
    # P2-3：DRL 训练类任务并发上限 2，超限 429（前端 toast 可读）
    if not _DRL_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="DRL 训练任务繁忙：最多 2 个并发训练，请稍后再试")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    task_id = _next_task_id()

    def _worker(db=db):
        async def _run():
            try:
                # worker 自建事件循环（ai.py/backtest.py 同模式）：
                # 数据加载直接 await；DB 写入经 engine._loop 调度回主循环
                df = await _load_df(body)
                cfg = body.model_dump()
                cfg["hidden"] = tuple(body.hidden)
                # 奖励塑形参数（QUANT_ADVICE 推荐默认已开启，显式传 0 可关闭）
                cfg["reward_dd_penalty"] = body.reward_dd_penalty
                cfg["reward_losing_penalty"] = body.reward_losing_penalty
                cfg["reward_trend_align"] = body.reward_trend_align
                cfg["reward_val_gap_penalty"] = body.reward_val_gap_penalty

                def on_progress(info):
                    with PROGRESS_LOCK:
                        _TRAINING[task_id] = {**_TRAINING.get(task_id, {}), **info,
                                              "running": True}
                    # 防内存膨胀：任务数超上限时清理过期已完成任务
                    prune_task_cache(_TRAINING, PROGRESS_LOCK)

                from drl import train_drl
                result = train_drl(df, cfg, on_progress=on_progress)

                # 续训模式：保存回原模型路径（覆盖原模型），策略名沿用 rl_{原模型名}
                # 而非新建模型文件——"在原先模型上改变"。模型名需通过白名单校验。
                if body.base_model:
                    try:
                        model_path = _model_path(body.base_model.strip())
                        if not model_path.exists():
                            raise FileNotFoundError(f"基础模型不存在: {body.base_model}")
                        strategy_name = f"rl_{model_path.stem}"
                    except Exception as e:  # noqa: BLE001
                        log.warning("[drl] 续训基础模型解析失败(%s)，回退新建模型: %s", body.base_model, e)
                        model_path = MODEL_DIR / f"{_default_name(body)}.json"
                        strategy_name = None
                else:
                    model_path = MODEL_DIR / f"{_default_name(body)}.json"
                    strategy_name = None

                # 记录实际使用的后端（GPU 不可用时自动回退 numpy）
                used_backend = result.get("backend", "numpy")
                with PROGRESS_LOCK:
                    _TRAINING[task_id] = {**_TRAINING.get(task_id, {}),
                                          "backend": used_backend}
                result["agent"].save(str(model_path))

                # 附加训练元数据（OOS报告/数据范围），供后续跨行情评估对比泛化
                try:
                    with open(model_path, "r", encoding="utf-8") as f:
                        _md = json.load(f)
                    _md["train_meta"] = {
                        "data_source": body.data_source,
                        "symbol": body.symbol,
                        "timeframe": body.timeframe,
                        "limit": body.limit,
                        "episodes": body.episodes,
                        "best_ret": result["best_ret"],
                        "best_val_ret": result.get("best_val_ret"),
                        "oos_report": result.get("oos_report"),
                    }
                    # 因子列标准化统计量（训练段拟合）——部署端 rl_adaptive 用同一
                    # mu/sd 标准化，否则喂原始因子值给模型，状态分布与训练完全不同
                    _md["factor_expression"] = result.get("factor_expression", "")
                    _md["factor_mu"] = float(result.get("factor_mu", 0.0))
                    _md["factor_sd"] = float(result.get("factor_sd", 1.0))
                    # 训练死区（min_trade_zone）写入模型：部署端调仓死区与训练严格一致
                    # （曾注册策略固定 buy_zone=0.02，与训练 0.05 不一致 → 部署更激进交易）
                    _md["min_trade_zone"] = float(getattr(body, "min_trade_zone", 0.05) or 0.05)
                    with open(model_path, "w", encoding="utf-8") as f:
                        json.dump(_md, f, ensure_ascii=False)
                except Exception as e:  # noqa: BLE001
                    log.warning("[drl] 模型元数据写入失败: %s", e)

                # 注册为可用策略（按 name 注册，engine 可切换）并持久化到 DB
                from strategies.rl_adaptive import RLAdaptiveStrategy
                from strategies import register_dynamic
                if strategy_name is None:
                    strategy_name = f"rl_{model_path.stem}"
                _risk_tips = ["训练数据分布外可能失效", "建议定期用新数据重训"]
                if result.get("deployment_blocked"):
                    _oos_reason = (result.get("oos_report") or {}).get("reason") or "OOS 硬门拦截"
                    _risk_tips = [f"疑似过拟合（{_oos_reason}），不建议实盘"] + _risk_tips
                spec = {
                    "name": strategy_name,
                    "title": f"RL自适应·{body.symbol}",
                    "description": f"深度强化学习训练的策略（{body.episodes}轮，{body.data_source}数据）"
                                   + ("（续训增强）" if body.base_model else ""),
                    "logic": "MDP建模+策略梯度训练，按价量/波动率/趋势状态动态调整目标仓位",
                    "executor": "rl_adaptive",
                    "param_schema": RLAdaptiveStrategy.param_schema,
                    "params": {"model_path": str(model_path), "buy_zone": 0.02},
                    "risk_tips": _risk_tips,
                    "created_by": "drl_train",
                    "version": "",
                    "base_symbol": body.symbol,
                    "base_timeframe": body.timeframe,
                    "episodes": body.episodes,
                    "model_path": str(model_path),
                    # 神经引擎的真实 Pine（不可导出时带原因）：曾缺失 →
                    # 前端给 rl_* 策略套通用模板，图上买卖点与模型无关
                    "pine_code": result.get("pine_code") or "",
                    "pine_note": result.get("pine_note") or "",
                }
                register_dynamic(strategy_name, spec)
                # 持久化到 AiStrategy 表（重启后 _load_dynamic_strategies 可恢复，全部策略可见）
                # P1-1：DB 写入调度回主循环执行（连接池绑定主循环；曾 asyncio.run 跨
                # loop 复用 → 间歇性 RuntimeError，训练完成但持久化静默丢失）
                try:
                    await _persist_strategy(db, strategy_name, spec, engine)
                    log.info("[drl] 策略 %s 已持久化", strategy_name)
                except Exception as e:  # noqa: BLE001
                    log.warning("[drl] 策略持久化失败: %s", e)

                with PROGRESS_LOCK:
                    # 长训练 history 降采样：几千/万轮压缩到最多 800 点（仅展示数据，不影响模型质量）
                    _hist = result["history"]
                    _max_points = 800
                    if len(_hist) > _max_points:
                        _step = len(_hist) / _max_points
                        _hist = [_hist[int(i * _step)] for i in range(_max_points - 1)] + [_hist[-1]]
                    _TRAINING[task_id] = mark_done({
                        **_TRAINING.get(task_id, {}),
                        "running": False, "done": True,
                        "model_name": model_path.stem,
                        "model_path": str(model_path),
                        "strategy_name": strategy_name,
                        "best_ret": result["best_ret"],
                        "best_val_ret": result.get("best_val_ret"),
                        "history": _hist,
                        "history_points": len(result["history"]),
                        "elapsed_sec": result["elapsed_sec"],
                        "oos_report": result.get("oos_report"),
                        # DRL 神经引擎 Pine 自动交易代码（训练产物）
                        "pine_code": result.get("pine_code") or "",
                        "pine_note": result.get("pine_note") or "",
                    })
                prune_task_cache(_TRAINING, PROGRESS_LOCK)
            except Exception as e:  # noqa: BLE001
                log.exception("[drl] 训练失败")
                with PROGRESS_LOCK:
                    _TRAINING[task_id] = mark_done({**_TRAINING.get(task_id, {}),
                                                    "running": False, "error": str(e)})
                prune_task_cache(_TRAINING, PROGRESS_LOCK)

        # worker 专属事件循环（ai.py/backtest.py 同模式）：
        # - 数据加载/DB 调度在其上运行
        # - finally 中关闭本循环上创建的 AIClient http client（防御性清理，P1-4 配套）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            # P2-3：任务结束释放并发名额（无论成败）
            _DRL_SEM.release()
            # P1-4 配套（模块 C 在 ai/client.py 提供 AIClient.close_loop(loop)）：
            # 关闭并移除本 worker 循环上创建的 httpx client（曾 loop.close() 后
            # client 残留 _http_clients 且连接未关 → 每任务泄漏；训练路径本身不
            # 直接调用 AIClient，此处为防御性清理）
            if engine is not None and getattr(engine, "ai_client", None) is not None:
                try:
                    loop.run_until_complete(engine.ai_client.close_loop(loop))
                except Exception:  # noqa: BLE001
                    pass
            loop.close()

    try:
        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        _DRL_SEM.release()  # 线程启动失败不占名额
        raise
    with PROGRESS_LOCK:
        _TRAINING[task_id] = {"running": True, "episode": 0, "episodes": body.episodes}
    return {"ok": True, "task_id": task_id}


@router.get("/train/{task_id}")
async def training_progress(task_id: int):
    """读取训练实时进度。"""
    with PROGRESS_LOCK:
        p = _TRAINING.get(int(task_id))
    if p is None:
        return {"found": False}
    return {"found": True, **p}


@router.get("/models")
async def list_models():
    """列出所有已训练模型。"""
    if not MODEL_DIR.exists():
        return {"models": []}
    models = []
    for f in sorted(MODEL_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        models.append({
            "name": f.stem, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1),
            "created_at": _ts(f),
        })
    return {"models": models, "dir": str(MODEL_DIR)}


@router.get("/models/{name}")
async def model_detail(name: str):
    """查看模型详情（配置/维度/训练轮数）。"""
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        actor_dims = data.get("actor", {}).get("dims", [])
        return {
            "name": name, "path": str(path),
            "state_dim": data.get("state_dim"), "n_actions": data.get("n_actions"),
            "gamma": data.get("gamma"), "epsilon": data.get("epsilon"),
            "actor_dims": actor_dims, "created_at": _ts(path),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"模型解析失败: {e}")


@router.post("/models/{name}/use")
async def use_model(name: str, engine=Depends(get_engine)):
    """应用模型为当前策略（引擎运行中则热切换）。"""
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")
    strategy_name = f"rl_{name}"
    try:
        from strategies.rl_adaptive import RLAdaptiveStrategy
        from strategies import register_dynamic, get_strategy
        from drl.pine_export import export_pine_from_model
        pine_code, pine_note = export_pine_from_model(path, name)
        register_dynamic(strategy_name, {
            "name": strategy_name, "title": f"RL自适应·{name}",
            "description": "深度强化学习训练的动态自适应策略",
            "logic": "MDP建模+策略梯度训练，动态调整目标仓位",
            "executor": "rl_adaptive",
            "param_schema": RLAdaptiveStrategy.param_schema,
            "params": {"model_path": str(path), "buy_zone": 0.02},
            "risk_tips": ["训练分布外可能失效"],
            "created_by": "drl_train", "version": "v1.0",
            "pine_code": pine_code, "pine_note": pine_note,
        })
        await engine.select_strategy(strategy_name)
        return {"ok": True, "strategy": strategy_name, "model": name}
    except Exception as e:  # noqa: BLE001
        log.exception("[drl] 应用模型失败")
        raise HTTPException(status_code=400, detail=f"应用模型失败: {e}")


@router.post("/models/{name}/evaluate")
async def evaluate_model(name: str):
    """用确定性策略评估模型在演示数据上的表现。"""
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")
    from drl import ACAgent, evaluate_agent
    from backtest.data_loader import generate_demo
    # 从模型文件读取元数据（architecture §2.3 根键；旧模型缺省兼容），
    # 按 train_drl 同口径重建 env（state_window/因子列/min_trade_zone）——
    # 曾缺省构造，state_window>1 或含因子列模型必然维度失配 500
    try:
        with open(path, "r", encoding="utf-8") as f:
            _data = json.load(f)
        cfg = _model_eval_cfg(_data)
        # 模型加载 + 完整 episode 评估放后台线程，避免冻结事件循环
        ev = await asyncio.to_thread(
            lambda: evaluate_agent(ACAgent.load(str(path)), generate_demo(timeframe="1h"), cfg))
        return {"ok": True, **ev}
    except Exception as e:  # noqa: BLE001
        log.exception("[drl] 模型评估失败")
        raise HTTPException(status_code=400, detail=f"评估失败: {e}")


def _model_extra_factors(model_data: dict, df) -> Optional[np.ndarray]:
    """按模型文件因子元数据在评估行情上标准化因子列（P1-9）。

    无因子表达式 → None（无因子列，旧模型兼容）；长度不符由 TradingEnv 抛
    ValueError（端点层 catch 并转 400 友好错误）。
    """
    expr = str(model_data.get("factor_expression", "") or "").strip()
    if not expr:
        return None
    from factors.mining import FactorExecutor
    vals = FactorExecutor(expr).eval(df).astype(float).to_numpy(float)
    mu, sd = float(model_data.get("factor_mu", 0.0)), float(model_data.get("factor_sd", 1.0))
    vals = (vals - mu) / (sd if sd and sd > 0 else 1.0)
    return np.nan_to_num(vals, nan=0.0).reshape(-1, 1)


def _model_eval_cfg(model_data: dict) -> dict:
    """模型文件根键（architecture.md §2.3）→ evaluate_agent cfg（缺省兼容）。"""
    return {
        "factor_expression": model_data.get("factor_expression", ""),
        "factor_mu": float(model_data.get("factor_mu", 0.0)),
        "factor_sd": float(model_data.get("factor_sd", 1.0)),
        "min_trade_zone": float(model_data.get("min_trade_zone", 0.05) or 0.05),
        "state_window": int(model_data.get("state_window", 1) or 1),
        "start_cash": 10000.0, "fee_rate": 0.001,
    }


async def _persist_strategy(db, strategy_name: str, spec: dict, engine) -> None:
    """持久化 AI 策略到 AiStrategy 表（P1-1：跨 loop DB 写入调度回主循环）。

    SQLAlchemy 异步引擎连接池绑定 lifespan 主循环；worker 线程直接跑 session
    会跨 loop 复用连接池 → 间歇性 RuntimeError（训练完成但持久化静默丢失）。
    经 run_coroutine_threadsafe 调度回 engine._loop 执行（同 web/api/ai.py
    apply_strategy_params / backtest.py _run_on_loop 模式），阻塞等待结果。
    engine._loop 不可用（CLI/单测环境）时降级为当前循环直接执行并 log warning。
    """
    import json as _json
    from core.database import AiStrategy
    from sqlalchemy import select as _sel

    async def _persist():
        async with db.session() as s:
            row = (await s.execute(_sel(AiStrategy).where(AiStrategy.name == strategy_name))).scalar_one_or_none()
            if row:
                row.spec_json = _json.dumps(spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=strategy_name, spec_json=_json.dumps(spec, ensure_ascii=False)))
            await s.commit()

    if engine is not None and getattr(engine, "_loop", None) is not None:
        fut = asyncio.run_coroutine_threadsafe(_persist(), engine._loop)
        fut.result(timeout=30.0)  # 阻塞等待主循环执行完成；超时/异常由调用方处理
    else:
        # 兜底：无主循环引用（CLI/单测环境），当前循环直接执行（与 ai.py 降级风格一致）
        log.warning("[drl] 无主循环引用，策略持久化降级为当前循环执行")
        await _persist()


class OosEvalIn(BaseModel):
    """跨行情 OOS 独立评估参数：换币种 / 周期 / 时间段。"""
    data_source: str = "demo"          # demo / exchange / csv
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=500, ge=100, le=5000)   # 评估用K线数量（时间段长度）
    csv_path: str = ""
    since: str = ""                    # 可选起始时间 ISO 字符串，空则取最近 limit 根


def _equity_stats(equities: list) -> dict:
    """权益序列 → 夏普 / 最大回撤。"""
    import numpy as np
    eq = np.asarray(equities, dtype=float)
    if len(eq) < 3:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    std = rets.std(ddof=1)
    sharpe = float(rets.mean() / std * np.sqrt(len(rets))) if std > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / (peak + 1e-9)).max())
    return {"sharpe": round(sharpe, 4), "max_drawdown": round(dd, 4)}


@router.post("/models/{name}/evaluate_oos")
async def evaluate_oos(name: str, body: OosEvalIn):
    """换一段未见过的行情（不同币种/周期/时间段）评估模型泛化能力。"""
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")

    from drl import ACAgent
    from drl.env import run_episode, TradingEnv
    from backtest.data_loader import generate_demo, load_csv, load_klines_cached

    # 读取模型文件根键元数据（architecture §2.3：state_window/factor_expression/
    # factor_mu/factor_sd/min_trade_zone；旧模型缺省兼容）
    model_data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"模型解析失败: {e}")
    try:
        agent = ACAgent.load(str(path))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"模型解析失败: {e}")

    # 读取训练元数据（训练时的数据范围与 OOS 报告），用于对比
    train_meta = model_data.get("train_meta") or {}

    # 1) 加载评估行情（demo/csv 的 pandas 解析放后台线程；exchange 源 await）
    try:
        if body.data_source == "demo":
            df = await asyncio.to_thread(generate_demo, n=body.limit, timeframe=body.timeframe)
        elif body.data_source == "csv":
            if not body.csv_path:
                raise HTTPException(status_code=400, detail="CSV 数据源需要 csv_path")
            csv_resolved = resolve_data_path(body.csv_path)
            df = await asyncio.to_thread(load_csv, str(csv_resolved))
        else:
            from datetime import datetime, timezone
            since = None
            if body.since:
                since = datetime.fromisoformat(body.since.replace("Z", "+00:00"))
            df = await load_klines_cached(body.exchange, body.symbol,
                                          body.timeframe, since=since, limit=body.limit)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"行情加载失败: {e}")

    if len(df) < 30:
        raise HTTPException(status_code=400,
                            detail=f"评估行情数据过少（仅{len(df)}根），请换更长时间段或更大 limit")

    # 2) 确定性评估（贪心策略，无探索噪声；完整 episode 放后台线程）
    # P1-9：按模型元数据同口径重建 env（state_window/因子列/min_trade_zone），
    # 曾缺省构造 → state_window>1 或含因子列模型必然维度失配 500
    try:
        env = TradingEnv(df, start_cash=10000.0, fee_rate=0.001,
                         vol_penalty=0.0, cost_scale=1.0,
                         min_trade_zone=float(model_data.get("min_trade_zone", 0.05) or 0.05),
                         extra_factors=_model_extra_factors(model_data, df),
                         state_window=int(model_data.get("state_window", 1) or 1))
        traj = await asyncio.to_thread(run_episode, env, lambda s: agent.greedy_action(s))
    except Exception as e:  # noqa: BLE001
        log.exception("[drl] OOS 评估失败")
        raise HTTPException(status_code=400, detail=f"评估失败: {e}")
    eqs = [info.get("equity") for info in traj.get("infos", []) if info.get("equity") is not None]
    stats = _equity_stats(eqs)

    # 3) 与训练 OOS 报告对比，量化泛化衰减
    oos_report = train_meta.get("oos_report") or {}
    train_oos_ret = oos_report.get("oos_ret")
    result = {
        "ok": True,
        "eval_symbol": body.symbol if body.data_source != "csv" else (csv_resolved.name if body.data_source == "csv" else "csv"),
        "eval_timeframe": body.timeframe,
        "eval_bars": len(df),
        "data_source": body.data_source,
        "total_ret": round(float(traj["total_ret"]), 6),
        "final_equity": round(traj["final_equity"], 2),
        "final_position_ratio": round(traj["final_position_ratio"], 4),
        "sharpe": stats["sharpe"],
        "max_drawdown": stats["max_drawdown"],
        "steps": traj["steps"],
        "train_meta": train_meta,
    }
    # Buy&Hold 基准：同一段行情的"躺平收益"，超额收益 = 策略收益 - 持有收益
    try:
        bh_ret = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 1 else 0.0
    except Exception:  # noqa: BLE001
        bh_ret = 0.0
    result["buy_hold_ret"] = round(bh_ret, 6)
    result["excess_ret"] = round(result["total_ret"] - bh_ret, 6)
    # 泛化对比：本段 OOS 收益 vs 训练时独立 OOS 段收益（同一模型，不同行情）
    if train_oos_ret is not None and abs(train_oos_ret) > 1e-9:
        decay = 1.0 - result["total_ret"] / train_oos_ret
        result["vs_train_oos"] = {
            "train_oos_ret": round(train_oos_ret, 6),
            "decay": round(decay, 4),
            "overfit_likely": bool(decay > 0.5),
        }
        result["verdict"] = (
            "疑似过拟合" if decay > 0.5 else
            "泛化良好" if decay <= 0.3 else "泛化一般")
    else:
        result["verdict"] = "独立行情评估（无训练 OOS 基准可对比）"
    return result


@router.post("/models/{name}/delete")
async def delete_model(name: str, db=Depends(get_db), engine=Depends(get_engine)):
    """删除已训练模型：删除模型文件 + 对应动态策略 + AiStrategy 记录。"""
    import os as _os
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"模型 {name} 不存在")
    strategy_name = f"rl_{name}"
    # 当前正在使用该模型策略时禁止删除
    if engine.strategy.name == strategy_name:
        raise HTTPException(status_code=400, detail="该模型策略正在使用中，请先切换其他策略再删除")
    try:
        # 1. 删除模型文件
        _os.remove(str(path))
        # 2. 移除动态策略注册
        from strategies import remove_dynamic, dynamic_names
        if strategy_name in dynamic_names():
            remove_dynamic(strategy_name)
        # 3. 删除 AiStrategy 记录（直接 async db 操作，不新开 event loop）
        from core.database import AiStrategy
        from sqlalchemy import select
        async with db.session() as s:
            row = (await s.execute(select(AiStrategy).where(AiStrategy.name == strategy_name))).scalar_one_or_none()
            if row:
                await s.delete(row)
                await s.commit()
        log.info("[drl] 模型 %s 已删除", name)
        return {"ok": True, "deleted": name}
    except Exception as e:  # noqa: BLE001
        log.exception("[drl] 删除模型失败")
        raise HTTPException(status_code=400, detail=f"删除模型失败: {e}")


def _next_task_id() -> int:
    with PROGRESS_LOCK:
        return max(list(_TRAINING.keys()) + [0]) + 1


def _ts(path: Path) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
