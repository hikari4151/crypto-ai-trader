"""量化因子系统 API：因子库 / 因子计算与IC分析 / AI因子挖掘 / 因子合成 / 模型因子。

数据源与回测一致：demo（演示数据）/ csv / exchange（交易所历史K线）。
因子计算纯向量化（pandas/numpy），支持 GPU 后端（与 fast_engine 对齐）。
"""
import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.database import Database
from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/factors", tags=["factors"])

# limit 封顶：防止任意大值触发超长 CPU 计算，冻结事件循环（已认证 DoS）
_LIMIT_CAP = 5000


class FactorComputeIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=1000, ge=100, le=_LIMIT_CAP)
    horizon: int = 1          # 未来几期收益做 IC
    use_gpu: bool = False


class FactorMineIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=1000, ge=100, le=_LIMIT_CAP)
    horizon: int = 1
    use_gpu: bool = False
    auto_register: bool = True   # 自动把有效因子注册进因子库


class FactorSynthIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=1000, ge=100, le=_LIMIT_CAP)
    horizon: int = 1
    weights: dict = {}           # {factor_key: weight}，空则 IC 加权自动合成
    top_n: int = 8
    method: str = "ic"           # ic=固定 IC 加权 / dynamic=滚动 IC 动态权重（AlphaForge 式，无前视）
    ic_window: int = 120         # dynamic 模式的滚动 IC 窗口
    use_gpu: bool = False


async def _load_df(body):
    """按数据源加载 OHLCV DataFrame（async：exchange 源直接 await；demo/csv 的 pandas 解析放后台线程）。"""
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    if body.data_source == "demo":
        return await asyncio.to_thread(generate_demo, timeframe=body.timeframe)
    if body.data_source == "csv":
        if not body.csv_path:
            raise HTTPException(status_code=400, detail="CSV 数据源需要 csv_path")
        from web.deps import resolve_data_path
        return await asyncio.to_thread(load_csv, str(resolve_data_path(body.csv_path)))
    if body.data_source == "exchange":
        return await load_from_exchange(body.exchange, body.symbol,
                                        body.timeframe, limit=body.limit)
    raise HTTPException(status_code=400, detail="未知数据源")


@router.get("/library")
async def factor_library():
    """内置因子库列表（含类别/说明/参数）。"""
    from factors.library import list_factors
    return {"factors": [f.meta() for f in list_factors()]}


@router.get("/custom")
async def list_custom_factors(db: Database = Depends(get_db)):
    """列出 AI 挖掘/用户合成的自定义因子（持久化于 KV）。"""
    custom = await db.kv_json_get("factor_custom", {})
    return {"factors": [{"name": k, **v} for k, v in custom.items()]}


@router.post("/compute")
async def compute_factors(body: FactorComputeIn):
    """计算因子矩阵 + IC/分组收益/相关性分析。"""
    from factors.analysis import (factor_correlation, factor_group_returns,
                                  factor_ic_table, find_redundant)
    from factors.engine import compute_factor_matrix
    df = await _load_df(body)
    try:
        # CPU 密集的因子矩阵/IC/分组/相关性全部放后台线程，避免冻结事件循环
        mat, ic_table, groups, corr, redundant = await asyncio.to_thread(
            _compute_pipeline, df, body.horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "n_candles": len(df),
        "n_factors": len(mat.columns),
        "factors": list(mat.columns),
        "ic_table": ic_table,
        "group_returns": groups,
        "correlation": {"columns": list(corr.columns),
                        "matrix": corr.round(4).values.tolist()}
                        if not corr.empty else {"columns": [], "matrix": []},
        "redundant_pairs": redundant[:20],
        "meta": [{"key": k, "name": _factor_name(k)} for k in mat.columns],
    }


@router.post("/usage")
async def factor_usage(body: FactorComputeIn):
    """因子使用场景：当前因子信号（多空/中性）+ 可落地的因子择时策略回测。

    让因子从"分析图表"变成"可直接交易的信号"：
    1. 每个因子的当前信号（z-score 阈值判断超买超卖/动量方向）
    2. 用最高 IC 因子生成 factor_signal 策略并回测，给出可执行方案
    """
    from factors.analysis import factor_ic_table
    from factors.engine import compute_factor_matrix
    from factors.library import get_factor
    df = await _load_df(body)
    try:
        mat = await asyncio.to_thread(compute_factor_matrix, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    close = df["close"]
    ic_table = factor_ic_table(mat, close, h=body.horizon, method="rank")
    latest = df.iloc[-1]

    # 每个因子的当前信号：z-score 方法
    def _signals():
        out = []
        for row in ic_table[:15]:
            key = row["key"]
            f = get_factor(key)
            if f is None:
                continue
            s = f.series(df)
            if s.notna().sum() < 20:
                continue
            last_val = float(s.iloc[-1])
            mu, sd = float(s.mean()), float(s.std())
            z = (last_val - mu) / (sd + 1e-12)
            # 信号判断：z 过高=超买(反向信号)/过低=超卖；结合 IC 方向
            ic_dir = 1 if row["rank_ic"] >= 0 else -1
            if z > 1.2:
                sig, label = "卖出", "超买"
            elif z < -1.2:
                sig, label = "买入", "超卖"
            else:
                sig, label = "中性", "区间"
            out.append({
                "key": key, "name": f.name, "category": f.category,
                "value": round(last_val, 6), "zscore": round(z, 3),
                "signal": sig, "label": label,
                "rank_ic": row["rank_ic"], "ic_dir": ic_dir,
            })
        return out
    signals = await asyncio.to_thread(_signals)

    # 用最佳方向因子生成因子择时策略并回测（使用场景：可直接用）
    best = next((s for s in signals if s["signal"] != "中性"), None)
    strategy_spec = None
    backtest = None
    if best:
        factor_map = {
            "mom_5": "mom_ma10", "mom_10": "mom_ma10", "mom_20": "mom_ma30",
            "roc_1": "mom_ma10", "macd_hist_pct": "macd_hist", "rsi_14": "rsi_osc",
            "vol_ratio": "vol_break", "bb_pos": "bb_pos",
        }
        fs = factor_map.get(best["key"], "macd_hist")
        mode = "reversal" if best["category"] in ("mean_reversion",) or best["label"] in ("超买", "超卖") else "trend"
        params = {"factor": fs, "mode": mode, "buy_threshold": 0.0, "sell_threshold": 0.0,
                  "stop_loss_pct": 0.03, "take_profit_pct": 0.06, "size_pct": 0.5}
        if fs == "rsi_osc":
            params["buy_threshold"] = 30.0
            params["sell_threshold"] = 70.0
        if fs == "bb_pos":
            params["mode"] = "reversal"
            params["buy_threshold"] = 0.2
            params["sell_threshold"] = 0.8
        from backtest.engine import BacktestConfig
        from backtest.fast_engine import run_backtest_fast
        bc = BacktestConfig(symbol=body.symbol, timeframe=body.timeframe,
                            strategy_name="factor_signal", strategy_params=params,
                            start_cash=10000.0, fee_rate=0.001)
        try:
            # 完整回测放后台线程，避免冻结事件循环
            r = await asyncio.to_thread(run_backtest_fast, df, bc)
            m = r["metrics"]
            backtest = {
                "strategy": "factor_signal", "params": params,
                "total_return": m["total_return"], "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
                "total_trades": m["total_trades"],
            }
            strategy_spec = {"name": "factor_signal", "params": params, "based_on": best["key"]}
        except Exception:  # noqa: BLE001
            pass

    return {"ok": True, "signals": signals,
            "recommended": strategy_spec, "backtest": backtest,
            "latest": {"close": float(latest["close"]), "ts": str(latest.name)}}


@router.post("/mine")
async def mine_factors(body: FactorMineIn, engine=Depends(get_engine),
                       db: Database = Depends(get_db)):
    """AI 挖掘新因子：AI 生成候选 + 程序进化变异（借鉴 QuantaAlpha/遗传编程），IC 反馈自进化。"""
    from factors.analysis import factor_ic_table
    from factors.engine import compute_factor_matrix
    from factors.mining import (evaluate_mined_factors, evolve_factors,
                                mine_prompt_system, mine_prompt_user, parse_ai_factors)
    from ai.client import AICallError, AINotConfigured

    df = await _load_df(body)
    try:
        mat = await asyncio.to_thread(compute_factor_matrix, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 现有因子 IC 表现（给 AI 参考，避免重复）
    ic_table = factor_ic_table(mat, df["close"], h=body.horizon, method="rank")
    existing = [{"name": r["key"], "rank_ic": r["rank_ic"]} for r in ic_table[:12]]

    # 上一轮 AI 挖掘结果（IC 反馈自进化）：从自定义因子库取 AI 挖掘的历史因子
    custom_factors = await db.kv_json_get("factor_custom", {})
    prev_feedback = [
        {"name": k, "expression": v.get("expression", ""),
         "rank_ic": v.get("rank_ic", 0), "icir": v.get("icir")}
        for k, v in custom_factors.items()
        if v.get("source") == "ai_mined" and v.get("rank_ic") is not None
    ][-8:]  # 最近 8 个作为反馈

    # 市场特征描述
    close = df["close"]
    ret = close.pct_change().dropna()
    market_desc = (
        f"数据量: {len(df)} 根K线, 周期: {body.timeframe}\n"
        f"区间收益: {(close.iloc[-1] / close.iloc[0] - 1) * 100:.1f}%, "
        f"日波动率均值: {ret.std() * 100:.2f}%\n"
        f"近期动量(20期): {(close.iloc[-1] / close.iloc[-21] - 1) * 100:.1f}%"
    )

    try:
        reply = await engine.ai_client.chat_json_validated([
            {"role": "system", "content": mine_prompt_system(prev_feedback)},
            {"role": "user", "content": mine_prompt_user(market_desc, existing)},
        ], feature="factor_mine", ctx={})
    except AINotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AICallError as e:
        raise HTTPException(status_code=502, detail=str(e))

    candidates = parse_ai_factors(reply)
    if not candidates:
        raise HTTPException(status_code=400, detail="AI 未生成合法因子表达式，请重试")

    # AI 验证 + 程序进化变异（纯计算，放后台线程避免冻结事件循环）
    def _mine_pipeline():
        results = evaluate_mined_factors(candidates, df, h=body.horizon)
        valid = [r for r in results if r.get("valid")]
        # 程序进化变异：对有效因子做窗口参数扰动，挖掘 IC 提升的变体（纯计算，无额外 AI 调用）
        evolved = evolve_factors(valid, df, h=body.horizon)
        # evolved 包含原有效因子 + 进化变体，按 |IC| 排序
        evolved.sort(key=lambda r: -abs(r.get("rank_ic", 0.0)))
        return evolved, valid
    evolved, valid = await asyncio.to_thread(_mine_pipeline)
    # 统计进化收益
    n_evolved = sum(1 for r in evolved if r.get("evolved_from"))

    # 自动注册有效因子到自定义库（含进化变体）
    if body.auto_register and evolved:
        custom = await db.kv_json_get("factor_custom", {})
        for r in evolved[:12]:
            custom[r["name"]] = {
                "title": r.get("title", r["name"]),
                "category": r.get("category", "momentum"),
                "expression": r["expression"],
                "logic": r.get("logic", ""),
                "expected_behavior": r.get("expected_behavior", ""),
                "rank_ic": r.get("rank_ic"),
                "icir": r.get("icir"),
                "source": "ai_mined",
                "evolved_from": r.get("evolved_from", ""),
                "created_at": _now(),
            }
        await db.kv_json_set("factor_custom", custom)

    return {"ok": True, "candidates": evolved, "valid": evolved,
            "n_evolved": n_evolved, "n_ai": len(valid),
            "auto_registered": body.auto_register}


@router.post("/synthesize")
async def synthesize(body: FactorSynthIn, db: Database = Depends(get_db)):
    """因子合成：IC 加权 / 动态滚动权重合成组合因子，返回权重与暴露度。"""
    from factors.analysis import factor_exposure, factor_ic_table
    from factors.engine import compute_factor_matrix
    from factors.mining import composite_factor, dynamic_composite, ic_weighted_composite
    df = await _load_df(body)
    if body.method == "dynamic" and not 30 <= body.ic_window <= 500:
        raise HTTPException(status_code=400, detail="ic_window 须在 30-500 之间")
    try:
        mat = await asyncio.to_thread(compute_factor_matrix, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    close = df["close"]

    # 合成 + 暴露度 + IC（CPU 密集，放后台线程）
    def _synth_pipeline():
        if body.weights:
            combo = composite_factor(mat, body.weights, method="manual")
            weights = body.weights
            selected = [{"key": k, "weight": v} for k, v in body.weights.items()]
            method_used = "manual"
        elif body.method == "dynamic":
            # 动态权重：滚动 IC 实时调权（因子衰减自动降权），无前视
            result = dynamic_composite(mat, close, h=body.horizon,
                                       ic_window=body.ic_window, top_n=body.top_n)
            combo = result["composite"]
            weights = result["weight_history"].iloc[-1].to_dict()
            selected = [{"key": k, "weight": round(float(v), 4), "dynamic": True}
                        for k, v in weights.items() if abs(v) > 1e-6]
            method_used = "dynamic"
        else:
            result = ic_weighted_composite(mat, close, h=body.horizon, top_n=body.top_n)
            combo = result["composite"]
            weights = result["weights"]
            selected = result["selected"]
            method_used = "ic"
        exposure = factor_exposure(weights, mat)
        combo_ic = _ic_of(combo, close, body.horizon)
        return combo, weights, selected, method_used, exposure, combo_ic

    combo, weights, selected, method_used, exposure, combo_ic = await asyncio.to_thread(_synth_pipeline)

    return {"ok": True, "method": method_used, "weights": weights, "selected": selected,
            "exposure": exposure, "composite_ic": combo_ic,
            "n_samples": int(combo.notna().sum())}


class ModelFactorIn(BaseModel):
    data_source: str = "demo"
    csv_path: str = ""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = Field(default=1000, ge=100, le=_LIMIT_CAP)
    horizon: int = 1                 # 预测未来几期收益
    hidden: list[int] = [32, 32]     # MLP 隐藏层
    epochs: int = 200
    lr: float = 0.002
    l2: float = 0.0001               # L2 正则（weight decay）
    patience: int = 25               # val 早停耐心值
    seed: int = 42
    auto_register: bool = True       # 通过安检后注册到自定义因子库


@router.post("/model-factor")
async def train_model_factor(body: ModelFactorIn, db: Database = Depends(get_db)):
    """监督学习因子（Qlib model-as-factor 范式）：MLP 预测未来收益，模型输出作为因子。

    防过拟合：train/val/OOS 严格时间切分 + 早停 + L2；安检判定段 = val（模型从未训练过），
    OOS 只报告不参与选择。CPU 训练放到后台线程（to_thread），避免阻塞事件循环。
    """
    # 参数边界（防 DoS/异常配置）
    if not 1 <= body.horizon <= 30:
        raise HTTPException(status_code=400, detail="horizon 须在 1-30 之间")
    if not 1 <= body.epochs <= 2000:
        raise HTTPException(status_code=400, detail="epochs 须在 1-2000 之间")
    if not 0 < body.lr < 1:
        raise HTTPException(status_code=400, detail="lr 须在 (0,1) 之间")
    if not 0 <= body.l2 < 1:
        raise HTTPException(status_code=400, detail="l2 须在 [0,1) 之间")
    if not 1 <= len(body.hidden) <= 4 or any(not 4 <= x <= 512 for x in body.hidden):
        raise HTTPException(status_code=400, detail="hidden 须为 1-4 层，每层 4-512 维")

    from factors.model_factor import fit_model_factor

    df = await _load_df(body)
    try:
        # CPU 密集训练放后台线程，避免冻结事件循环（回测/交易事件不受影响）
        result = await asyncio.to_thread(
            fit_model_factor, df,
            h=body.horizon, hidden=tuple(body.hidden),
            epochs=body.epochs, lr=body.lr, l2=body.l2,
            patience=body.patience, seed=body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    report = result["report"]
    # 通过安检后注册到自定义因子库（source=model_factor，同名重训以最新为准）
    registered = False
    if body.auto_register and report["valid"]:
        name = f"model_factor_h{body.horizon}"
        custom = await db.kv_json_get("factor_custom", {})
        val_gate = report["segments"]["val"]
        custom[name] = {
            "title": f"模型因子(h={body.horizon})",
            "category": "momentum",
            "expression": f"<model_factor h={body.horizon} hidden={list(body.hidden)} seed={body.seed}>",
            "logic": "MLP 在 13 维价量特征上预测未来收益，模型输出即因子（model-as-factor）",
            "rank_ic": val_gate.get("rank_ic"),
            "icir": val_gate.get("icir"),
            "turnover": val_gate.get("turnover"),
            "fitness": val_gate.get("fitness"),
            "oos_rank_ic": (report["segments"].get("oos") or {}).get("rank_ic"),
            "source": "model_factor",
            "meta": result["meta"],
            "created_at": _now(),
        }
        await db.kv_json_set("factor_custom", custom)
        registered = True

    return {"ok": True, "valid": report["valid"], "reason": report["reason"],
            "segments": report["segments"], "decay": report["decay"],
            "meta": result["meta"], "registered": registered}


def _ic_of(series, close, h):
    from factors.analysis import factor_ic
    r = factor_ic(series, close, h=h, method="rank")
    return {"rank_ic": r["rank_ic"], "icir": r["icir"]}


def _factor_name(key: str) -> str:
    try:
        from factors.library import get_factor
        f = get_factor(key)
        return f.name if f else key
    except Exception:  # noqa: BLE001
        return key


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _compute_pipeline(df, horizon: int):
    """因子矩阵 + IC 表 + 分组收益 + 相关性 + 冗余对（同步，供 to_thread 调用）。"""
    from factors.analysis import (factor_correlation, factor_group_returns,
                                  factor_ic_table, find_redundant)
    from factors.engine import compute_factor_matrix
    mat = compute_factor_matrix(df)
    close = df["close"]
    ic_table = factor_ic_table(mat, close, h=horizon, method="rank")
    groups = factor_group_returns(mat, close, h=horizon)
    corr = factor_correlation(mat)
    redundant = find_redundant(mat, threshold=0.85)
    return mat, ic_table, groups, corr, redundant
