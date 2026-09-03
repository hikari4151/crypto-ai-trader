# 优化需求模块拆分与接口契约（v2.0）

> 基于 `crypto_ai_trader` 项目代码分析，2026-08-16
> 批次：M1+M2+M3（第一批），M4+M5+M6（第二批）

---

## 1. 模块拆分与文件归属

### 第一批（M1+M2+M3）

| 模块 | 独占文件 | 共享文件 | 依赖模块 |
|------|---------|---------|---------|
| **M1 回测口径对齐** | `exchange/paper.py` | `engine/order_manager.py`、`engine/trading_engine.py`（`_record_fill`） | 无 |
| **M2 性能引擎优化** | `indicators/vectorized.py`、`indicators/technical.py`、`drl/env.py` | `backtest/engine.py`、`backtest/fast_engine.py`、`engine/trading_engine.py`、`engine/risk.py` | 建议在 M1 之后执行（共享 backtest/engine.py 和 fast_engine.py） |
| **M3 因子统计严谨性** | `factors/analysis.py`、`factors/mining.py`、`factors/model_factor.py` | 无 | 无 |

### 第二批（M4+M5+M6）

| 模块 | 独占文件 | 共享文件 | 依赖模块 |
|------|---------|---------|---------|
| **M4 DRL 口径统一** | `drl/agent.py`、`drl/env.py` | 无 | 无 |
| **M5 数据与风控增强** | `backtest/data_loader.py`、`backtest/metrics.py` | `engine/risk.py` | 无 |
| **M6 代码清理** | 无（临时文件删除） | `engine/order_manager.py`、`engine/trading_engine.py`、`engine/risk.py`、`drl/agent.py`、`factors/analysis.py` | 建议在所有模块之后执行（except 收紧可能影响其他模块） |

---

## 2. 文件冲突矩阵

| 文件 | 冲突模块 | 冲突类型 | 协调方案 |
|------|---------|---------|---------|
| `backtest/engine.py` | M1（成交口径）、M2（S/R 优化） | 同文件不同段 | M1 先改成交逻辑，M2 再改指标计算段。两段代码完全不重叠 |
| `backtest/fast_engine.py` | M1（成交口径）、M2（S/R 优化） | 同文件不同段 | 同上 |
| `engine/order_manager.py` | M1（FIFO 成本）、M6（except 收紧） | 同文件不同段 | M1 先改成本逻辑，M6 最后收紧 except |
| `engine/trading_engine.py` | M1（`_record_fill`）、M2（pop(0)→deque）、M6（except 收紧） | 同文件不同段 | M1 先改，M2 再改，M6 最后 |
| `engine/risk.py` | M2（pop(0)→deque，实际无 pop(0)）、M5（DB fallback）、M6（except 收紧） | 同文件不同段 | M5 先改，M6 最后收紧 |
| `drl/agent.py` | M4（vol_penalty）、M6（except 收紧） | 同文件不同段 | M4 先改，M6 最后收紧 |
| `factors/analysis.py` | M3（Newey-West + 门槛）、M6（except 收紧） | 同文件不同段 | M3 先改，M6 最后收紧 |

### 执行顺序建议

```
M1 → M3 → M2 → M4 → M5 → M6
     (M3 无冲突，可并行)
```

---

## 3. 接口契约表

### 3.1 函数签名变更

| 函数/类 | 所在文件 | 当前签名 | 变更后签名 | 变更说明 |
|---------|---------|---------|-----------|---------|
| `PaperAccount.positions` | `exchange/paper.py` | `dict[str, {"qty": float, "avg_price": float}]` | `dict[str, {"qty": float, "lots": list[tuple[float, float]]}]` | FIFO 逐笔记账替代加权平均成本 |
| `OrderManager._place_paper` | `engine/order_manager.py` | `async def _place_paper(self, signal, last_price, strategy_name)` | 同左（内部改为 FIFO 成本计算） | 内部实现变更，外部签名不变 |
| `TradingEngine._record_fill` | `engine/trading_engine.py` | 读 `paper_account.positions[].avg_price` | 读 `paper_account.positions[].lots` | 成本计算方式变更 |
| `TradingEngine._trade_times` | `engine/trading_engine.py` | `list[float]` | `deque[float]` | pop(0) → popleft() |
| `IncrIndicators._buf_*` | `indicators/vectorized.py` | `list[float]` | `deque[float]` | pop(0) → popleft() |
| `factor_ic` | `factors/analysis.py` | 返回 `{ic, rank_ic, icir, ic_std, ...}` | 新增 `{ic_std_nw: float}`；`icir` 改用 NW 修正 | 向后兼容 |
| `_rolling_ic` | `factors/analysis.py` | `def _rolling_ic(..., step=None)` | 同左 | 内部调用不变 |
| `DEFAULT_GATES["min_abs_ic"]` | `factors/analysis.py` | `0.01` | `0.03` | 收紧门槛 |
| `factor_quality_gate` | `factors/analysis.py` | 使用 `DEFAULT_GATES` | 同左（自动取新值） | 行为变化 |
| `train_drl` | `drl/agent.py` | val_env 使用 `vol_penalty=0.0` | val_env 使用 `vol_penalty=vol_penalty` | 口径统一 |
| `evaluate_agent` | `drl/agent.py` | 使用 `vol_penalty=0.0` | 使用 `vol_penalty=cfg.get("vol_penalty", 0.0)` | 口径统一 |
| `load_from_exchange` | `backtest/data_loader.py` | `async def load_from_exchange(..., limit=1000)` | 新增 `max_candles=10000` 参数；实现分页循环 | 向后兼容 |
| `RiskManager.get_rules` | `engine/risk.py` | 无 try/except，DB 异常传播 | 添加 try/except，DB 异常时返回 DEFAULT_RULES | 行为变化（fail-safe） |
| `compute_benchmark` | `backtest/metrics.py` | `def compute_benchmark(closes, start_cash, timeframe)` | 新增 `fee_rate=0.0, slippage=0.0` 参数 | 向后兼容 |
| 临时文件 | 根目录 + tests/ | 存在 | 删除 | 无 |

### 3.2 数据结构变更

| 数据结构 | 所在文件 | 变更内容 | 影响范围 |
|---------|---------|---------|---------|
| `PaperAccount.positions` 值 | `exchange/paper.py` | `{"qty": float, "avg_price": float}` → `{"qty": float, "lots": list[tuple[float, float]]}` | `order_manager.py`、`trading_engine.py`（`_record_fill`）、`portfolio.py` 读取 `avg_price` 的代码 |
| `factor_ic()` 返回字典 | `factors/analysis.py` | 新增 `ic_std_nw` 键 | `factor_quality_gate`、`factor_ic_table`、`periodic_ic_refresh` 以及 WebAPI 消费者 |
| `DEFAULT_GATES` | `factors/analysis.py` | `min_abs_ic: 0.01 → 0.03` | `factor_quality_gate`、`mining.py`、`model_factor.py` 中所有使用该常量的地方 |
| `compute_benchmark()` 返回 | `backtest/metrics.py` | `buy_hold_ret` 在 `fee_rate>0` 时略低 | 所有使用 `benchmark` 键的消费者（前端 equity 曲线等） |

### 3.3 行为变更（无接口变更）

| 行为 | 所在文件 | 变更内容 | 影响 |
|------|---------|---------|------|
| 纸面模式 PnL | `exchange/paper.py` + `order_manager.py` | FIFO 摊销替代加权平均 | 纸面模式 PnL 更接近回测，现有纸面交易历史可能略有差异 |
| backtest 引擎 S/R 计算 | `backtest/engine.py` + `fast_engine.py` | 预计算全量 + 按索引取值 | 回测结果逐位一致（仅速度提升） |
| DRL 验证评估 | `drl/agent.py` | vol_penalty 从 0.0 改为训练值 | 验证收益会降低（增加了惩罚），模型选择可能变化 |
| 风控规则 DB fallback | `engine/risk.py` | DB 不可用时使用默认规则而非拦截所有交易 | 更健壮，但 DB 异常期间风控规则无法更新 |

---

## 4. 依赖关系图

```
M1 (回测口径) ──→ 共享文件 ──→ M2 (性能优化) ──→ 共享文件 ──→ M6 (代码清理)
                     │
                     └──→ order_manager.py ──→ M6
                     
M3 (因子统计) ──→ factors/analysis.py ──→ M6

M4 (DRL 统一) ──→ drl/agent.py ──→ M6

M5 (风控增强) ──→ engine/risk.py ──→ M6
```

---

## 5. 已创建的计划文件

| 文件 | 对应模块 | 批次 |
|------|---------|------|
| `docs/plans/plan_m1_backtest_alignment.md` | M1 回测口径对齐 | 第一批 |
| `docs/plans/plan_m2_performance_optimization.md` | M2 性能引擎优化 | 第一批 |
| `docs/plans/plan_m3_factor_statistics.md` | M3 因子统计严谨性 | 第一批 |
| `docs/plans/plan_m4_drl_unification.md` | M4 DRL 口径统一 | 第二批 |
| `docs/plans/plan_m5_data_risk_enhancement.md` | M5 数据与风控增强 | 第二批 |
| `docs/plans/plan_m6_code_cleanup.md` | M6 代码清理 | 第二批 |