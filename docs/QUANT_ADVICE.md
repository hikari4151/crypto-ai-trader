# 专业量化视角改进建议（实现方案）

> 基于 crypto_ai_trader 代码库（2026-08-15 重构后状态）的实际代码分析。
> 优先级：🔴 高 > 🟠 中 > 🟡 低
> 每条含：目标文件、改动要点、核心代码思路、验证方法。

---

## 🔴 1. 回测报告加入买入持有基准

**目标文件**：`backtest/metrics.py`、`backtest/engine.py`、`backtest/fast_engine.py`、`web/api/backtest.py`（响应结构）

**现状**：`compute_metrics` 只算策略自身的收益/夏普/回撤，没有基准对比。

**改动要点**：

1. `metrics.py` 新增函数 `compute_benchmark(equity_curve: list[float], trades: list[dict], start_cash: float) -> dict`：
   - 买入持有 = 第一根 K 线全仓买入，最后一根 K 线全仓卖出（理论上用 `data["close"].iloc[0]` 和 `data["close"].iloc[-1]`——但 `equity_curve` 已经包含策略的每日权益曲线，买入持有的权益曲线 = `[start_cash] + [start_cash * close[i] / close[0] for i in 1..n]`）
   - 计算超额收益 = `total_return - bench_return`
   - 信息比率 = `(策略日收益 - 基准日收益).mean() / (策略日收益 - 基准日收益).std() * sqrt(252)`
   - 超额收益最大回撤（Calmar 比率类似）

2. `engine.py`/`fast_engine.py` 在 `run_backtest` 中传入 `close prices` 给 `compute_metrics`（或由 `compute_metrics` 自行从 `equity_curve` 反推基准——但 `equity_curve` 已含手续费和滑点，基准不应含这些，所以需要传原始 `close`）。

3. 返回结构追加 `{"benchmark": {"buy_hold_ret": ..., "excess_return": ..., "information_ratio": ..., "excess_max_drawdown": ...}}`。「基准」在结果 JSON 中始终存在，前端 `index.html` 中的权益曲线图同时绘制策略曲线与基准曲线（`renderEquity` 或者 `renderPnlChart` 中加一条虚线）。

**核心代码**：
```python
def compute_benchmark(closes: np.ndarray, start_cash: float) -> dict:
    if len(closes) < 2:
        return {"buy_hold_ret": 0.0, "excess_return": 0.0, "information_ratio": 0.0, "excess_max_drawdown": 0.0}
    bench_equity = start_cash * closes / closes[0]
    buy_hold_ret = (bench_equity[-1] / start_cash) - 1
    return {"buy_hold_ret": round(buy_hold_ret, 6), ...}
```

**验证**：`run.py backtest --source demo --strategy dual_ma` 输出含 `benchmark` 键；前端权益曲线出现两条线。

---

## 🔴 2. 因子挖掘跨品种验证

**目标文件**：`factors/mining.py`、`factors/model_factor.py`、`web/api/factors.py`、`tests/test_factor_oos.py`

**现状**：因子只在同一品种的 IS/OOS 时间段上验证，如果在 BTC 上挖的因子可能只在 BTC 趋势行情中有效，换到 ETH 就失效。

**改动要点**：

1. `mining.py` 的 `factor_quality_gate` 新增参数 `cross_validate_data: Optional[dict[str, pd.DataFrame]] = None` —— 传 `{"ETH/USDT": df_eth, "BNB/USDT": df_bnb}`。
2. 对每个跨品种数据：用该因子表达式计算 IC/ICIR/换手率。如果跨品种 IC 均值 < 0.01 或方向与主品种不一致，则标记 `cross_validation="failed"`。
3. `model_factor.py` 的 `fit_model_factor` 同样：保存模型时附带 `cross_val_ic` 到模型元数据。
4. `web/api/factors.py` 的 `POST /factors/mine` 端点新增可选参数 `cross_symbols`（逗号分隔的交易对），用户可从前端选择跨品种验证的数据源。
5. 前端因子挖掘结果页面显示 `cross_validation` 状态（✅ 通过 / ⚠ 未配置 / ❌ 未通过）。

**注意**：跨品种数据需要同周期（如都 1h），且长度足够（至少 200 根 K 线）。数据从 `_load_df` 或交易所实时拉取，不在回测时重新下载（优化：`DATA_DIR` 里缓存）。

**验证**：用 BTC 数据挖因子，ETH 数据做跨品种验证 → 因子 IC 两品种方向一致才算通过。

---

## 🔴 3. 每日全量对账

**目标文件**：`engine/trading_engine.py`、`engine/portfolio.py`、`web/api/portfolio.py`、`core/events.py`（`REPORT` 事件类型）

**现状**：`order_manager._reconcile_loop` 每 30s 对账未成交订单，但缺少每日定时拉取交易所全部持仓做全量对账。

**改动要点**：

1. `trading_engine.py` 在 `_portfolio_loop`（每 15s 运行）中加一个每日定时分支：`if now.hour == 0 and now.minute == 0 and now.second < 15 and not self._daily_reconciled_today`。
2. 定时分支调用 `portfolio.py` 的 `_do_fetch` 拉取交易所真实钱包数据，与本地 `self.portfolio` 中的持仓逐笔对比。
3. 差异处理：
   - 本地有、交易所无 → 记录告警日志 `[reconcile] MISSING POSITION`，发布 `SYSTEM` 事件（`bus` 已有）
   - 交易所多、本地无 → 记录告警 `[reconcile] UNKNOWN POSITION`（可能是手动交易所操作）
   - 数量偏差 > 5% → 告警 `[reconcile] QTY_MISMATCH`
4. 每日只执行一次（`_daily_reconciled_today` 标志），跨 0 点重置（`_daily_reconciled_today = False` 在 `_portfolio_loop` 的 `if now.hour != 0` 重置）。

**注意**：`_do_fetch` 在 `web/api/portfolio.py` 里是 async 函数，但 `_portfolio_loop` 在主循环内 async，可以直接调用。需要把 `_do_fetch` 抽取为 `portfolio.py` 的一个公共方法（或 `PortfolioManager` 的方法），引擎侧和 API 侧复用。

**验证**：模拟盘（simulated）启动引擎，过 0 点后检查日志出现 `[reconcile]` 记录；手动修改本地持仓（如插入测试 Trade），观察告警。

---

## 🟠 4. 连续亏损熔断 + 人工确认恢复

**目标文件**：`engine/risk.py`、`engine/trading_engine.py`、`web/api/risk.py`、`web/static/index.html`（风控设置页）

**现状**：`risk.cooldown_status` 已有冷却机制（连续亏损 N 笔后暂停新开仓，冷却时间后自动恢复）。但缺少"人工确认才能恢复"的选项。

**改动要点**：

1. `risk.py` 的 `cooldown_status` 新增 `manual_recovery: bool` 字段（`False` 自动恢复，`True` 需要人工确认）。`cooldown_status` 当前返回字典，追加 `manual_recovery` 键。
2. 当 `manual_recovery=True` 时，冷却时间到后 `check()` 的 `cooldown_active` 仍返回 True，直到用户调用 `POST /api/risk/clear-cooldown` 手动确认。
3. `trading_engine.py` 的 `_on_candle_locked_inner` 在 `risk.check()` 返回 False 时，若原因包含冷却 → 日志中注明"冷却中（人工确认模式）"。
4. 前端风控设置页新增开关"冷却后需人工确认"（状态由 `GET /api/risk/rules` 返回，`PUT /api/risk/rules` 写入 `kv_set` 的 `risk_manual_recovery` 键）。
5. 前端在仪表盘冷却指示器（`riskCooldown.active` 为 True 时显示"🛡 冷却中"）旁加一个"点击解除"按钮，POST 到 `/api/risk/clear-cooldown`。

**核心代码**（risk.py 新增）：
```python
@property
def cooldown_status(self) -> dict:
    manual = self._manual_recovery
    if not self._cooldown_active:
        return {"active": False, "remaining": 0, "manual_recovery": manual}
    remaining = max(0, self._cooldown_until - time.time())
    if remaining <= 0 and manual:
        return {"active": True, "remaining": 0, "manual_recovery": manual, "awaiting_clear": True}
    return {"active": remaining > 0, "remaining": int(remaining), "manual_recovery": manual}

async def clear_cooldown(self) -> None:
    self._cooldown_active = False
    self._cooldown_until = 0
```

**验证**：风控规则中设置"连续亏损 2 笔"，make 2 loosing trades → 冷却触发，冷却时间到后新开仓仍被阻塞（显示"人工确认"）；调用 `/api/risk/clear-cooldown` 后恢复。

---

## 🟠 5. 交易所断连自动平仓

**目标文件**：`exchange/ws_market.py`、`engine/trading_engine.py`、`config/settings.py`（新增参数）

**现状**：`ws_market` 的 `watch_ohlcv_for` 有指数退避重连（最多 30s 一次），但断连期间引擎继续运行，`_on_candle` 不触发（因为没有新 K 线），风控仍有效但无价格更新。

**改动要点**：

1. `ws_market.py` 新增 `last_klines_ts: dict[str, float]` 记录每个 symbol/tf 最后一次收到 K 线的时间戳，`_upsert` 中更新。
2. `ws_market.py` 新增 `max_stale_seconds: int = 300`（默认 5 分钟，`settings` 可配置）。
3. `trading_engine.py` 的 `_portfolio_loop`（每 15s 运行）中检查 `hub` 的 `last_klines_ts`——如果当前 symbol/tf 的 `time.time() - last_ts > max_stale_seconds`：
   - 如果 `mode == "live"` → 发布 `SYSTEM` 事件（`"kind": "market_stale"`）+ 如果有持仓（`self.order_manager` 或 `self.portfolio` 有仓位）→ 触发平仓（遍历所有持仓，逐个发 `SIGNAL` 事件，side=sell，理由="交易所断连自动平仓"）
   - 如果 `mode == "simulated"` → 只告警，不平仓（模拟盘无真实资金）
4. 断连恢复后（`last_klines_ts` 更新）→ 发布 `SYSTEM` 事件（`"kind": "market_recovered"`），自动平仓后不再恢复开仓（需人工确认）。

**注意**：自动平仓是**极端风控措施**，可能产生不必要的损失（如行情波动中平仓在低点）。建议默认关闭（`settings.max_stale_seconds: 0`），用户需显式配置才启用。

**验证**：模拟盘启动引擎，手动停止 `hub`（临时改代码让 `refresh` 不更新 `last_klines_ts`）→ 5 分钟后日志出现平仓信号。

---

## 🟠 6. 因子滚动 IC 衰变自动下线

**目标文件**：`factors/library.py`、`factors/analysis.py`、`web/api/factors.py`、`engine/trading_engine.py`（策略热切换时检查）

**现状**：因子上线后永久有效，即使 IC 已经衰变成负值，策略仍在使用。

**改动要点**：

1. `library.py` 的 `_LIB` 因子注册表增加元数据：`{"name": ..., "alive": True, "ic_decay": {"rolling_ic": [], "rolling_icir": 0.0, "last_updated": null}}`。
2. 新增 `periodic_ic_refresh` 功能（可放在 `analysis.py` 或 `factors/engine.py` 中）：
   - 每天定时（或每收到 100 根 K 线）用最新数据计算注册因子的 IC
   - 维护滚动 IC 队列（最近 30 个 IC 值）
   - 如果 `rolling_ic.mean() < -0.01` 且 `icir < -0.1` 持续 3 个周期 → `alive = False`（自动下线）
3. 下线的因子不再被 `factor_signal` 和 `rl_adaptive` 策略使用（`factor_signal.py` 的 `on_candle` 中检查因子状态，跳过 `alive=False` 的因子）。
4. 前端因子库页面示下线状态（❌ 已下线 / ✅ 活跃），保留历史记录（不删除，只标记）。
5. 如果 IC 随后恢复（`rolling_ic.mean() > 0.01` 持续 3 个周期）→ `alive = True`（自动恢复）。

**注意**：滚动 IC 需要最新行情数据，建议在引擎运行时（`_portfolio_loop` 或 `_ai_scheduler` 的空闲时段）异步计算。计算量轻（单因子 O(n)），不会阻塞主循环。

**验证**：手动注册一个故意反向的因子（如 `close.shift(1) - close`），观察其 IC 为负 → 3 个周期后自动下线 → factor_signal 策略不再使用它。

---

## 🟡 7. 回测加入限价单部分成交模拟

**目标文件**：`backtest/fast_engine.py`、`backtest/engine.py`、`indicators/vectorized.py`（可选）

**现状**：回测中所有订单都是市价成交（`fill_price = price_open * (1 +/- slippage)`），限价单的挂单/部分成交/不成交没有被模拟。

**改动要点**：

1. `BacktestConfig` 新增 `limit_order_model: str = "none"`（可选 `"none"` / `"partial"` / `"probabilistic"`）。
2. `"partial"` 模式：当 `signal.order_type == "limit"` 时，用以下规则模拟：
   - 买入限价：如果 `limit_price >= low[i]` 则成交（触及过限价），否则挂单等待下一根 K 线
   - 如果触底但未完全填满（`limit_price < low[i]` 但 `high[i] > limit_price`），成交比例 = `(high[i] - limit_price) / (high[i] - low[i])`（假设价格均匀分布）
   - 跨 K 线挂单：最多 N 根 K 线（默认 3），超时未成交则取消
3. `"probabilistic"` 模式：用历史 Tick 数据模拟（回测不用，因为无 tick 数据——用 `"partial"` 足够）。
4. 这个改动会改变回测结果（限价单不再是市价成交），所以**默认关闭**（`"none"` 保持原有行为）。

**验证**：`run.py backtest --source demo --strategy grid --limit-order-model partial` 对比两种模式的下单数量与成交价差异。

---

## 🟡 8. DRL reward 默认推荐值

**目标文件**：`web/api/drl.py`（`TrainIn` 默认值）、`drl/env.py`（reward 计算）、`docs/REFACTOR_PLAN.md` 或独立文档

**现状**：`entropy_coef=0.03`、`reward_dd_penalty=0.0`、`reward_losing_penalty=0.0`、`reward_trend_align=0.0`——塑形奖励全部关闭，用户从零开始调参。

**改动要点**：

1. `drl/env.py` 的 `TradingEnv` 在 `__init__` 中：如果 `reward_dd_penalty > 0` 或 `reward_losing_penalty > 0` 或 `reward_trend_align > 0`，打印配置日志 `[drl] 奖励塑形已启用: dd=%s losing=%s trend=%s`（让用户知道哪些打开了）。
2. `web/api/drl.py` 的 `TrainIn` 默认值调整：
   - `entropy_coef: float = 0.05`（从 0.03 提高，我们修了熵符号后，熵系数稍大促进探索）
   - `reward_trend_align: float = 0.1`（趋势对齐奖励，鼓励在趋势行情中持仓）
   - `reward_dd_penalty: float = 0.5`（回撤惩罚，默认轻量开启）
   - `reward_losing_penalty: float = 0.2`（连亏惩罚，默认轻量开启）
3. 新增训练页面提示文字（`index.html` 的 DRL 训练面板）："推荐默认值：entropy_coef=0.05, trend_align=0.1, dd_penalty=0.5, losing_penalty=0.2"。
4. 注意：这些默认值改变后，现有模型的训练行为会变（因为 reward 不同了）。不影响推理（`evaluate_agent` 不用 reward 塑形）。所以是forward change。

**验证**：`POST /api/drl/train` 不带这些参数（默认值）→ 训练日志出现 `[drl] 奖励塑形已启用: dd=0.5 losing=0.2 trend=0.1`。

---

## 🟡 9. AI 优化器 + 网格扫描的 warm-start pipeline

**目标文件**：`ai/optimizer.py`、`web/api/ai.py`、`web/api/backtest.py`（`grid-scan`）、`engine/trading_engine.py`（`_ai_scheduler`）

**现状**：AI 优化器输出参数（我们修了 apply=False 返回新参数的问题），网格扫描独立运行，两者没有串联。

**改动要点**：

1. `optimizer.py` 的 `optimize_price_action` 在 `apply=False` 时返回 `{"params": ..., "reason": ..., "focus": ..., "grid_center": params}` —— 把 AI 建议的参数也作为网格扫描的中心点。
2. `web/api/ai.py` 的 `auto-optimize` 完成后，如果引擎未繁忙且存在 `grid_center`，自动触发一次局部网格扫描（`grid-scan` 端点复用，但参数范围缩小到 AI 建议值的 ±20%）。
3. `grid-scan` 完成后，扫描结果中的 `best` 参数自动应用（`apply_strategy_params`）。
4. `_ai_scheduler` 每周自动优化 + 网格扫描（现有的 `ai_optimize_interval` 控制）。

**验证**：auto-optimize 完成后，观察日志出现 `[grid-scan] 以 AI 建议为中心扫描` 和 `[grid-scan] 应用最优参数`。

---

## 🟡 10. 因子库扩展（横截面/微观结构）

**目标文件**：`factors/library.py`、`factors/analysis.py`、`indicators/technical.py`（新增）

**现状**：因子库只有 MA/MACD/RSI/布林/ATR 等标准技术指标。

**改动要点**：

1. `library.py` 注册新因子（按需添加，不强制全部实现）：
   - 动量因子：`momentum_1w` = `close / close.shift(7) - 1`（1 周动量）
   - 波动率因子：`volatility_20` = `close.pct_change().rolling(20).std()`（20 日波动率）
   - 成交量异常：`vol_spike` = `volume / volume.rolling(20).mean() - 1`（成交量突增）
   - 价格位置：`price_position` = `(close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())`（20 日价格百分位）
   - 乖离率：`bias_20` = `(close - sma(close, 20)) / sma(close, 20)`（20 日乖离率）

2. 每个因子需在 `analysis.py` 的 `factor_quality_gate` 中注册默认参数（`min_ic=0.01`、`min_icir=0.1` 等），确保新因子不会因为质量门低而绕过安检。

3. 扩展后的因子库在 `factor_signal` 策略中自动可用（`factor_signal.py` 的 `on_candle` 遍历 `_LIB` 中 `alive=True` 的所有因子）。

**验证**：`/api/factors/library` 返回新增因子；`factor_signal` 策略回测交易数不为 0（demo 数据上至少部分因子有信号）。

---

## 实施顺序建议

这 10 项之间**没有强依赖**，可以并行开发。但按风险评估：

1. **先做 🔴3（每日全量对账）**— 实盘资金安全，改动量小，独立模块
2. **再做 🔴1（基准对比）**— 回测报告增强，纯增量，不破坏现有行为
3. **再做 🟠4 + 🟠5（熔断 + 断链平仓）**— 风控增强，与现有 risk.py 正交
4. **再做 🔴2（跨品种验证）**— 因子挖掘增强，需测试数据
5. **最后做 🟠6 + 🟡7-10（剩余项）**— 功能增强，无时间压力

总工作量估算：约 6-10 天（单人全栈），其中 🔴3 约 1 天、🔴1 约 0.5 天、🟠4+🟠5 约 1 天、🔴2 约 1-2 天、🟠6 约 1 天、🟡7-10 约 2-3 天。