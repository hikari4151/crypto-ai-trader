# M1 回测口径对齐 — 实施计划

> 依赖架构契约：docs/plans/architecture.md（若有）
> 独占文件：`backtest/engine.py`、`backtest/fast_engine.py`、`engine/order_manager.py`、`exchange/paper.py`、`engine/trading_engine.py`（仅 `_record_fill` 段）
> 任务：统一回测引擎/纸面/实盘的成交时机和成本基准，使三者 PnL 口径一致

---

## 问题分析

| 环节 | 回测引擎 | 纸面模式 | 实盘模式 |
|------|---------|---------|---------|
| 成交时机 | 信号在 K 线收盘后产生，**次根 K 线开盘价成交** | 当前收盘价(last_price)成交 | 当前收盘价成交 |
| 成本基准 | FIFO 逐笔记账（lots 队列） | 加权平均成本（avg_price） | 交易所实际成交 |
| 滑点 | 成交价 `*(1±slippage)` | 无滑点 | 无滑点，交易所实际价 |

**目标**：三者 PnL 口径一致。回测引擎是真理基准，纸面/实盘向回测对齐。

---

## T1. 纸面成交时机对齐（order_manager.py + paper.py）

**Files**：`engine/order_manager.py`、`exchange/paper.py`

### 当前问题
- `order_manager.py:157`：`price = signal.limit_price or last_price`，使用当前收盘价成交
- `paper.py:apply_fill`：使用加权平均成本（avg_price），不记录分批建仓明细

### 改造方案

**order_manager.py 修改**：
- `_place_paper`: 不再用 `last_price` 作为成交价，而改为传递信号后等待下一根 K 线开盘价。
  但纸面模式没有"下一根 K 线"概念——在 `trading_engine.py` 的 `_on_candle_locked_inner` 中，当前 K 线收盘产生信号，下一个 K 线到来时才有开盘价。
  因此纸面成交价天然就是"当前 K 线收盘价"（信号产生时知道的最晚价格）。
  
  **决策**：纸面模式保持当前 K 线收盘价成交（与回测的"次根开盘价"在模拟行情下等价——因为收盘价和次根开盘价通常接近，但严格上说回测更保守）。关键是对齐成本基准（FIFO）。

- 将 `PaperAccount` 的加权平均成本改为 FIFO 逐笔记账（与回测引擎一致）

**PaperAccount 修改**：
- `apply_fill`: 将 `positions[symbol]` 从 `{"qty", "avg_price"}` 改为 `{"qty", "lots": [(qty, price), ...]}` 结构
- 卖出时用 FIFO 摊销成本，与回测引擎的 `lots` 逻辑一致

### 接口变更

```python
# exchange/paper.py
# 修改前
pos = {"qty": float, "avg_price": float}
# 修改后
pos = {"qty": float, "lots": list[tuple[float, float]]}  # [(qty, price), ...]
```

### 验证
- 纸面模式下的 PnL 与回测引擎在相同策略/数据上一致（差异仅来自成交时机：收盘 vs 次根开盘）

---

## T2. 实盘成交记账对齐（order_manager.py _place_live）

**Files**：`engine/order_manager.py`

### 当前问题
- `_place_live`: 实盘使用交易所实际成交价，成交后通过 `_record_fill` 记录
- `trading_engine.py _record_fill`: 卖出时计算 PnL 用 `paper_account.positions.get(symbol, {}).get("avg_price")` 或 `entry_price`

### 改造方案
- 实盘成交价由交易所决定，无法改变。但成本计算应向 FIFO 对齐
- 在 `engine/order_manager.py` 中维护实盘 FIFO 成本队列（类似回测的 `lots`）
- 卖出时用 FIFO 摊销计算 PnL

### 接口变更
- `OrderManager` 新增 `_live_lots: dict[str, list[tuple[float, float]]]` 字段
- `_record_fill` 中的成本计算逻辑从读 `avg_price` 改为读 FIFO 队列

---

## T3. 回测引擎确认（engine.py + fast_engine.py）

**Files**：`backtest/engine.py`、`backtest/fast_engine.py`

### 当前状态
两个引擎已经使用：
- 成交时机：信号在 K 线收盘后产生，次根 K 线开盘价成交（`pending_signal` + 开盘价逻辑）
- 成本基准：FIFO 逐笔记账（`lots: list[tuple[float, float]]`）
- 滑点：成交价叠加 `slippage`

**确认不动**：回测引擎的成交逻辑已是对齐的正确基准，无需修改。

---

## T4. TradingEngine._record_fill 成本计算对齐

**Files**：`engine/trading_engine.py`（`_record_fill` 方法，行 398-445）

### 当前问题
- 行 420-424：纸面模式用 `paper_account.positions.get(symbol, {}).get("avg_price")` 作为成本价
- 行 425-426：回退到策略的 `entry_price`（单点，会被最后一次买入价覆盖）

### 改造方案
- 纸面模式改用 FIFO 成本（从 `PaperAccount` 的 lots 队列计算）
- `entry_price` 回退路径保留为兜底

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| `exchange/paper.py` | `positions` 结构从 `avg_price` 改为 `lots` 队列；卖出时 FIFO 摊销 | 无冲突 |
| `engine/order_manager.py` | `_place_paper` 成交价确认；`_resolve_qty` 适配；新增 `_live_lots` 实盘 FIFO 队列 | 与 M6（except 收紧）有重叠，但改动区域不同 |
| `engine/trading_engine.py` | `_record_fill` 中成本计算从 `avg_price` 改为 FIFO | 与 M2（pop(0)→deque）有重叠，但改动区域不同 |

---

## 验收标准

- [ ] 纸面模式与回测引擎在相同策略/数据上的总 PnL 差异 < 5%（差异来自成交时机口径）
- [ ] `PaperAccount` 卖出时 FIFO 摊销与回测引擎的 `lots.pop(0)` 逻辑一致
- [ ] 实盘模式 `_record_fill` 的 PnL 计算使用 FIFO 成本，而非单点 `entry_price`
- [ ] 现有 `dual_ma`/`grid`/`price_action` 策略的纸面回测结果不倒退
- [ ] `python -m compileall -q backtest engine exchange`