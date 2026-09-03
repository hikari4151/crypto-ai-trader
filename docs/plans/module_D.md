# Module D: DRL+AI — 实施计划

> 依赖架构契约：docs/plans/architecture.md §2.6（DRL 默认值）、§2.7（grid_center）、§3（跨模块依赖）
> 独占文件：`web/api/drl.py`、`drl/env.py`、`ai/optimizer.py`、`web/api/ai.py`
> 任务：🟡8 DRL 默认值 + 🟡9 AI+网格串联
> 禁止编辑：`backtest/*`（grid-scan 复用端点，不越界）、模块 A/B/C/E 文件

---

## T1. 🟡8 drl/env.py 奖励塑形配置日志

**Files**：`drl/env.py`（`TradingEnv.__init__` :25-79）

**Interfaces**：
- 产出：`TradingEnv.__init__` 在塑形系数任一 >0 时打印 `[drl] 奖励塑形已启用: dd=%s losing=%s trend=%s`

**实现**（:78 附近，字段赋值完成后）：
```python
if reward_dd_penalty > 0 or reward_losing_penalty > 0 or reward_trend_align > 0:
    import logging
    logging.getLogger(__name__).info(
        "[drl] 奖励塑形已启用: dd=%s losing=%s trend=%s",
        reward_dd_penalty, reward_losing_penalty, reward_trend_align)
```

**防回归红线**：
- 塑形计算逻辑（step :209-227）零改动；系数=0 时行为不变（现有 A/B 路径）
- 日志仅为观测点，不影响训练结果 dict

**验证**：`POST /api/drl/train`（不带塑形参数，default 全 0）→ 日志**不出现**塑形启用行；带默认推荐值后（T2 改默认）→ 出现 `[drl] 奖励塑形已启用: dd=0.5 losing=0.2 trend=0.1`

---

## T2. 🟡8 web/api/drl.py TrainIn 默认值调整

**Files**：`web/api/drl.py`（`TrainIn` :52-87）

**Interfaces**（契约见 architecture.md §2.6）：
- 产出：`TrainIn.entropy_coef: float = 0.05`（原 0.03）
- 产出：`TrainIn.reward_trend_align: float = 0.1`（原 0.0）
- 产出：`TrainIn.reward_dd_penalty: float = 0.5`（原 0.0）
- 产出：`TrainIn.reward_losing_penalty: float = 0.2`（原 0.0）

**实现**：:67 及 :84-86 三处默认值字段修改。

**防回归红线**：
- `TrainIn` 其它字段（episodes/hidden/lr/gamma/vol_penalty/cost_scale/min_trade_zone/...）零改动
- `_validate_train`（:89-105）边界校验零改动（新默认值均在合法区间）
- Pydantic 模型字段名不变 → 前端旧提交不传这些参数也使用新默认（**预期行为变化**：训练行为不同，属 forward change，与 QUANT_ADVICE 一致）
- 推理路径（evaluate/evaluate_oos，:484-505）不使用 reward 塑形 → 不受影响

**验证**：
- `POST /api/drl/train` 不带塑形/熵参数 → `_worker` 内 `cfg["reward_dd_penalty"]=0.5` 等生效，训练日志出现 T1 的启用行
- 带显式 `rew_dd_penalty=0` 等覆盖 → 仍关闭（覆盖优先级不变）

---

## T3. 🟡9 ai/optimizer.py 返回 grid_center

**Files**：`ai/optimizer.py`（`optimize_price_action` :19-52）

**Interfaces**（契约见 architecture.md §2.7）：
- 产出：`optimize_price_action(..., apply=False)` 返回 dict 追加 `grid_center: dict`
- `apply=True`（旧行为）：返回不变（`grid_center=None` 或省略）

**实现**（:51 return 段）：
```python
log.info("[ai] 策略 %s 已按关键位/价格行为优化: %s", strategy.name, applied)
out = {"params": applied, "reason": result.get("reason", ""),
       "focus": result.get("focus", "")}
if not apply:
    out["grid_center"] = dict(new_params)   # AI 建议参数即网格扫描中心
return out
```

**防回归红线**：
- `apply=True` 路径（`_ai_scheduler` :496-500 `apply_strategy_params(name, result["params"])`）取 `result["params"]` 语义不变
- `apply=False` 返回现在多 `grid_center` 键——web/api/ai.py auto-optimize 消费者（T4）与新字段兼容；其它调用方（engine `_ai_scheduler`）只取 `params` 键，零影响
- OptimizationLog 写入（:42-50）不变

**验证**：
- 单测（mock client）：`optimize_price_action(apply=False)` → `grid_center == new_params`；`apply=True` → 无 grid_center 或 None

---

## T4. 🟡9 web/api/ai.py auto-optimize 触发局部网格扫描

**Files**：`web/api/ai.py`（`auto_optimize` :194-273）

**Interfaces**：
- 消费：`optimize_price_action` 返回的 `grid_center`；`/api/backtest/grid-scan`（用 HTTP 或复用 Service，见下）
- 产出：auto-optimize 完成后日志 `[grid-scan] 以 AI 建议为中心扫描` 与 `[grid-scan] 应用最优参数`；`result["grid_scan"]` 结果

**实现**：
1. `_run` 内 `optimize_price_action` 返回后（:227-239 参数应用段之后）插入：
   ```python
   grid_center = result.get("grid_center") or {}
   grid_result = None
   if grid_center and not engine.bus is None:
       _ai_progress(task_id, "grid", "局部网格扫描（以 AI 建议为中心）", 70, "参数范围 ±20%…")
       grid_result = await _run_nearby_grid_scan(engine, grid_center)
       if grid_result and grid_result.get("best"):
           best = grid_result["best"]["params"]
           if best and engine._loop is not None:
               fut = asyncio.run_coroutine_threadsafe(
                   engine.apply_strategy_params(engine.strategy.name, best), engine._loop)
               fut.result(timeout=10.0)
               log.info("[grid-scan] 应用最优参数: %s", best)
           result["grid_scan"] = {"scanned": grid_result.get("scanned"),
                                  "best": best, "applied": bool(best)}
   ```
2. 新增模块内协程（**不碰 backtest.py**）——局部扫描逻辑：
   ```python
   async def _run_nearby_grid_scan(engine, center: dict) -> Optional[dict]:
       """以 AI 建议参数为中心做 ±20% 局部网格扫描（调用 /api/backtest/grid-scan 同款函数）。
       直接复用 backtest.fast_engine + 本地 df（与 grid-scan 端点同构，避开 HTTP 往返）。"""
       from backtest.engine import BacktestConfig
       from backtest.fast_engine import run_backtest_fast
       import itertools
       snap = await engine._current_snapshot()
       if not snap or len(snap.get("closes", [])) < 60:
           return None
       import pandas as pd
       df = pd.DataFrame(snap["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
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
       results = []
       for combo in combos:
           params = dict(zip(keys, combo))
           try:
               cfg = BacktestConfig(symbol=engine.symbol, timeframe=engine.timeframe,
                                    strategy_name=engine.strategy.name, strategy_params=params)
               r = run_backtest_fast(df, cfg)
               m = r["metrics"]
               results.append({"params": params, "total_return": m["total_return"],
                               "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
           except Exception:
               continue
       results.sort(key=lambda x: -(x["total_return"] * 10 + x["sharpe"] * 2 - x["max_drawdown"] * 5))
       best = results[0] if results else None
       log.info("[grid-scan] 以 AI 建议为中心扫描完成: %d 组", len(results))
       return {"ok": True, "scanned": len(results), "results": results[:5], "best": best}
   ```

> 设计决策：不 HTTP 调 `/api/backtest/grid-scan`（需额外数据源请求与并发名额），直接复用在 backtest.py 用的 `BacktestConfig`+`run_backtest_fast` 走 `engine._current_snapshot()` 的最新快照 K 线。这样**不碰 backtest.py** 且数据实时。组合数限制：≤3 数值参数时候选 3^3=27 组，轻量。

3. `_ai_scheduler`（engine/trading_engine.py，**模块 C 独占**）：QUANT_ADVICE 建议每周自动优化+网格——C 侧在 T6（其 `_ai_scheduler`）已具备自动优化；是否叠网格扫描由 C 协调接入（架构文档 §8 记录）。D 交付纯函数 + auto-optimize 端点联动。

**防回归红线**：
- `auto_optimize` 现有阶段进度（snapshot/perf/ai/validate）不变，新增 `grid` 阶段为**追加**
- `_AI_TASKS` 进度键结构不变；`result` 新增 `grid_scan` 键（可选），其余键保留
- `run_backtest_fast`/`BacktestConfig` 在 backtest.py 中签名不动（我们只是调用方）
- 引擎繁忙/无快照时 `_run_nearby_grid_scan` 返回 None，**不影响 auto-optimize 主流程**

**验证**：
- 手工：`POST /api/ai/auto-optimize`（配 AI key）→ 完成后日志出现 `[grid-scan] 以 AI 建议为中心扫描` 与（若有更优）`[grid-scan] 应用最优参数`；`result.grid_scan.best.params` 出现在策略参数
- 单测（mock AIClient）：grid_center 存在 → `_run_nearby_grid_scan` 返回非空；center 为空 → None

---

## 交付验收清单（模块 D）

- [ ] `python -m compileall -q web/api/drl.py web/api/ai.py ai/optimizer.py drl/env.py`
- [ ] `python -m pytest -q`（现有全绿 + 新增 optimizer grid_center / env 日志测试）
- [ ] `POST /api/drl/train`（默认参数）→ 日志显示塑形启用行 [drl] dd=0.5 losing=0.2 trend=0.1
- [ ] `POST /api/ai/auto-optimize` → grid-scan 阶段出现且 result 含 grid_scan
- [ ] 回归：`apply=True` 的 `_ai_scheduler` 自动优化路径不受影响（engine 现有行为）
- [ ] 记录跨模块协调项（grid-scan 组合数上限、_ai_scheduler 每周网格挂载）到架构文档 §8