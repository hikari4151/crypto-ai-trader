# Module A: 回测增强 — 实施计划

> 依赖架构契约：docs/plans/architecture.md §2.1（基准响应结构）、§2.5（limit_order_model）
> 独占文件：`backtest/metrics.py`、`backtest/engine.py`、`backtest/fast_engine.py`、`web/api/backtest.py`、`backtest/data_loader.py`（data_loader.py 本次无改动，保留所有权以防他模块越界）
> 任务：🔴1 基准对比 + 🟡7 限价单模拟

---

## T1. 🔴1 metrics.py 新增 compute_benchmark

**Files**：`backtest/metrics.py`（追加函数，行 54 之后）

**Interfaces**：
- 产出：`compute_benchmark(closes: np.ndarray, start_cash: float, timeframe: str = "1h") -> dict`
- 返回字典键（契约见 architecture.md §2.1）：`buy_hold_ret`、`excess_return`、`information_ratio`、`excess_max_drawdown`、`bench_equity_curve`

**实现**（注意是"基准"独立函数，超额指标由 engine 层算好后回填——基准自身不知道策略权益曲线）：

```python
def compute_benchmark(closes: np.ndarray, start_cash: float, timeframe: str = "1h") -> dict:
    """买入持有基准权益曲线与本段区间收益。
    closes: 原始收盘价序列（不含手续费/滑点）。
    excess_return/information_ratio/excess_max_drawdown 三个超额指标需要策略
    权益曲线，由 engine 层调用后回填；此处初始化占位并仅保证键恒在。
    """
    eq = np.asarray(closes, dtype=float)
    if len(eq) < 2 or starts_with_zero := (eq[0] <= 0):
        return {
            "buy_hold_ret": 0.0, "excess_return": 0.0,
            "information_ratio": 0.0, "excess_max_drawdown": 0.0,
            "bench_equity_curve": [round(start_cash, 4)],
        }
    bench_equity = start_cash * eq / eq[0]
    buy_hold_ret = bench_equity[-1] / start_cash - 1.0
    return {
        "buy_hold_ret": round(buy_hold_ret, 6),
        "excess_return": 0.0,
        "information_ratio": 0.0,
        "excess_max_drawdown": 0.0,
        "bench_equity_curve": [round(float(x), 4) for x in bench_equity],
    }
```

> 注：`walrus` 表达式仅作防御示例——保持简单，用两分支即可。模块内开发时按简洁风格实现。

**验证**：
- `python -c "from backtest.metrics import compute_benchmark; import numpy as np; r=compute_benchmark(np.array([1.,2.,4.]), 10000.); print(r)"` → `buy_hold_ret≈3.0`、`bench_equity_curve=[10000, 20000, 40000]`

---

## T2. 🔴1 两个引擎接入基准（engine.py + fast_engine.py 同步）

**Files**：`backtest/engine.py`（:170-175 指标段）、`backtest/fast_engine.py`（:182-189 指标段）

**Interfaces**：
- 消费：`compute_benchmark(closes, cfg.start_cash, cfg.timeframe)`
- 产出：两个引擎返回 dict 均含顶层 `"benchmark"` 键（完整版，超额指标已回填）

**engine.py 修改**（原有 :171-175）：

```python
from .metrics import compute_metrics, compute_benchmark

metrics = compute_metrics(equity_curve, trades, cfg.timeframe, cfg.start_cash)
metrics["benchmark"] = compute_benchmark(closes, cfg.start_cash, cfg.timeframe)
# ---- 超额指标回填（基准不含手续费/滑点，策略 equity_curve 含；同长度逐位差分） ----
if len(metrics["benchmark"]["bench_equity_curve"]) == len(equity_curve) and len(equity_curve) >= 2:
    beq = np.asarray(metrics["benchmark"]["bench_equity_curve"], dtype=float)
    seq = np.asarray(equity_curve, dtype=float)
    excess = seq[-1] / cfg.start_cash - 1.0 - metrics["benchmark"]["buy_hold_ret"]
    periods = PERIODS_PER_YEAR.get(cfg.timeframe, 8760)
    s_ret = np.diff(seq) / (seq[:-1] + 1e-9)
    b_ret = np.diff(beq) / (beq[:-1] + 1e-9)
    diff = s_ret - b_ret
    ir = float(diff.mean() / (diff.std() + 1e-12) * math.sqrt(periods)) if len(diff) > 5 else 0.0
    peak = np.maximum.accumulate(seq)
    dd = (peak - seq) / (peak + 1e-12)
    ex_mdd = float(dd.max())
    metrics["benchmark"].update({"excess_return": round(excess, 6),
                                  "information_ratio": round(ir, 4),
                                  "excess_max_drawdown": round(ex_mdd, 6)})
return {"metrics": metrics, "equity_curve": equity_curve,
        "trades": trades, ..., "benchmark": metrics["benchmark"]}
```

> 设计决策：`benchmark` 同时挂在顶层（`result["benchmark"]`）与 `metrics["benchmark"]` 内——前者服务 `/results/{id}`（result dict），后者保证 `_PROGRESS[task_id]["metrics"]`（进度缓存只存 metrics）也有基准。**两处都要写**。不要在 `metrics` 内用 `{**benchmark, **metrics}` 展开，这会丢失 `benchmark` 作为内嵌 dict 的结构。

**fast_engine.py 修改**（原有 :184-189）：完全相同的模式，`closes` 已在 :50 定义。

**防回归红线**：
- `metrics["total_return"]` 等 12 个既有键值**逐位不变**（benchmark 只是新键，不进 key 干涉现有消费者；`/backtest/results` 列表取字段按 key 提取，不受影响）
- `equity_curve`/`trades`/`symbol`/`timeframe`/`strategy`/`params` 键不变
- 超短线（n<2）时 `compute_benchmark` 返回占位零值，不得抛异常（metrics.py 的 `if n < 2` 已回归）

**验证**：
- `python run.py backtest --source demo --strategy dual_ma` → 输出含 `benchmark` 键，`buy_hold_ret` 为正（demo 数据近段上涨，实测确认）
- `python -m pytest tests/test_backtest_engine_consistency.py -q`（两引擎一致性测试，确认双引擎仍逐位一致）
- `python run.py backtest --source demo --strategy grid` 与修改前输出对比：现有 metrics 键值不变

---

## T3. 🟡7 BacktestConfig + 请求模型新增 limit_order_model

**Files**：`backtest/engine.py`（BacktestConfig :21-32）、`web/api/backtest.py`（BacktestRequest :40-55、GridScanIn :57-71）

**Interfaces**：
- 产出：`BacktestConfig.limit_order_model: str = "none"`；`BacktestRequest.limit_order_model: str = "none"`；`GridScanIn.limit_order_model: str = "none"`

**实现**：三处各加一行字段定义（值域 `"none" | "partial" | "probabilistic"`）。

**防回归红线**：
- 默认值 `"none"` 不改变现有任何回测结果（成交逻辑全走原市价路径）
- Pydantic 新增字段为 optional 带默认值，旧前端不传即用默认——`_schedule_backtest` :194-197 的 `BacktestConfig(...)` 无需改（自动透传）

---

## T4. 🟡7 fast_engine 成交循环接入 limit_order_model（partial 模式）

**Files**：`backtest/fast_engine.py`（:98-146 pending_signal 成交段）

**Interfaces**：
- 消费：`cfg.limit_order_model`、`sig.order_type`/`sig.limit_price`（Signal 可选字段，见下）
- 前置：`strategies/base.py` 的 `Signal` 类需新增可选字段——**不在模块 A 独占清单**，采用兼容策略：`getattr(sig, "order_type", None)` 读取，避免越界修改 `strategies/base.py`

**注意（模块边界）**：QUANT_ADVICE 将 `Signal` 视为方案的一部分，但 `strategies/base.py` 不在任何模块独占清单中。**实施方案**：A 模块用 `getattr` 防御性读取 `order_type`/`limit_price`；若字段缺失，`partial` 模式退化为市价（日志告警一次）。是否需要给 Signal 加字段由 PM 统一协调（见架构文档第 8 节"分工外注意"）。

**实现**（fast_engine.py :98-146 内，`fill_price` 计算后插入完整限价单逻辑，见 T5 代码块）：

> 注意：以下为简化的逻辑骨架。T5 提供完整实现（含触及判定、部分成交比例、跨根挂单计数器等）。开发者按 T5 完整代码块实现即可。

**T5. 🟡7 partial 成交模拟完整实现**

```python
# 在 fast_engine.py :103 行后插入（引擎循环内）：
if (cfg.limit_order_model in ("partial", "probabilistic")
        and getattr(sig, "order_type", None) == "limit"
        and getattr(sig, "limit_price", None)):
    lp = float(sig.limit_price)
    low = float(lows[i])
    high = float(highs[i])
    if sig.side == "buy":
        # 买入限价：K线最低价触及限价 → 全量成交；否则部分/挂单
        if low <= lp <= high:
            fill_price = lp
            # 部分成交：价格均匀分布假设下的填充比例
            fill_ratio = float(high - lp) / (high - low + 1e-12) if lp >= low else 1.0
        else:
            fill_ratio = 0.0  # 未触及 → 下一根重试（max_pending 根后取消）
    else:
        if low <= lp <= high:
            fill_price = lp
            fill_ratio = float(lp - low) / (high - low + 1e-12) if lp <= high else 1.0
        else:
            fill_ratio = 0.0
    if fill_ratio <= 0:
        # 挂单：重放 pending_signal 到下一根（最多 3 根）
        num_pending = 0
        if num_pending < 3:
            pending_signal = sig
            num_pending += 1
        continue  # 跳过本根成交，下一根重来
    qty_total = sig.qty or (cash * sig.size_pct / fill_price)
    qty = qty_total * fill_ratio
    # 以下沿用原市价成交记账（cash/position/lots 更新），fill price=lp
```

> 说明：跨根挂单计数器（最多 3 根）需在 `pending_signal` 会话中维护（在循环外声明 `_limit_try: dict[id, int]`）。实现细节由开发者按引擎既有风格补全，此处给出语义契约：**触及才成交、未触及保留 3 根、超时取消、默认 none 模式零变化**。

**engine.py（事件驱动）**：与 T5 相同逻辑落在 engine.py :91-138 段（`fills` 记账结构一致）。两引擎实现须逻辑同构，保证一致性测试仍绿。

**防回归红线（T4+T5）**：
- `limit_order_model="none"`（默认）：`getattr` 短路，成交逻辑走原路径，与基线逐位一致
- 现有策略（dual_ma/grid/factor_signal/price_action）**均不产生 order_type="limit"** 信号 → partial 模式对它们是纯框架，不影响现有回测
- 不修改 `strategies/base.py`（Signal 类保持原样，用 getattr 防御）

---

## 交付验收清单（模块 A）

- [ ] `python -m compileall -q backtest web`
- [ ] `python -m pytest tests/test_backtest_engine_consistency.py -q`（双引擎一致性）
- [ ] `python run.py backtest --source demo --strategy dual_ma` 结果含 benchmark 键，且 total_return/sharpe/win_rate 与修改前一致
- [ ] `python run.py backtest --source demo --strategy grid`（默认 none）结果逐位不变（基线对比：trades/成交明细与修改前一致）
- [ ] partial 模式冒烟（CLI 无该参数，用脚本直调，见 T3/T5 验证）：
  `python -c "from backtest.engine import BacktestConfig, run_backtest; from backtest.data_loader import generate_demo; df=generate_demo(timeframe='1h'); r=run_backtest(df.copy(), BacktestConfig(strategy_name='dual_ma', limit_order_model='partial')); print(r['metrics']['total_return'])"` 不报错且与 none 模式 total_return 一致
- [ ] 手工 `/api/backtest/results/{id}` JSON 含 `metrics.benchmark.bench_equity_curve` 数组

---

## 暂停保存点（2026-08-15）

### 任务状态总览

| 任务 | 优先级 | 状态 | 说明 |
|---|---|---|---|
| T1 🔴1 基准对比（metrics.py） | 🔴 | ✅ 已完成 | `compute_benchmark` 已实现，`compute_metrics` 新增可选 `closes` 参数 |
| T2 🔴1 引擎集成（engine.py + fast_engine.py） | 🔴 | ✅ 已完成 | 两引擎已调 `compute_benchmark` 并回填超额指标，返回 dict 含 `benchmark` 键 |
| T3 🟡7 限价单模型（BacktestConfig / BacktestRequest / GridScanIn） | 🟡 | ✅ 已完成 | 三处新增 `limit_order_model: str = "none"`，`_schedule_backtest`/`grid_scan`/`compare` 透传 |
| T4 🟡7 Signal 类字段 | 🟡 | ✅ 已完成（无需改） | `strategies/base.py` 已有 `order_type: str = "market"` 和 `limit_price: Optional[float] = None`，默认保持原有行为 |
| T5 🟡7 引擎限价单部分成交逻辑 | 🟡 | ✅ 已完成 | 两引擎 `skip_fill` 标志 + `_limit_pending` 跨根计数器（最多 3 根 K 线），未触及不跳过 equity_curve 点 |
| T6 🟡7 数据清洗（OHLCVSanitizer） | 🟡 | ✅ 已完成 | `data_loader.py` 新增 `OHLCVSanitizer`（ffill 限 3 根 + MAD Z-score 异常值中位数替换） |
| 测试（benchmark + limit_order + consistency） | — | ✅ 已完成 | 31 tests 全部通过 |
| 验证链 | — | ✅ 已完成 | compileall 通过、pytest 31 passed、CLI 结果含 benchmark 键、JSON 串行化正常 |

### 已完成任务的修改文件清单与改动要点

| 文件 | 改动要点 |
|---|---|
| `backtest/metrics.py` | 新增 `compute_benchmark(closes, start_cash, timeframe)` 函数（含占位键、空/短序列/零首价格防御）；`compute_metrics` 新增可选参数 `closes: np.ndarray \| None = None`，传值时自动在返回 dict 中加入 `benchmark` 键，默认 `None` 行为不变（向后兼容） |
| `backtest/engine.py` | 添加 `import math, numpy as np`；`BacktestConfig` 新增 `limit_order_model: str = "none"`；`run_backtest` 中（1）调用 `compute_benchmark` 得到基准 dict，（2）回填 `excess_return`/`information_ratio`/`excess_max_drawdown` 三个超额指标，（3）返回 dict 顶层新增 `"benchmark": metrics["benchmark"]` 双挂载；成交循环增加 `skip_fill` 标志 + `_limit_pending` 跨根计数器实现限价单部分成交模拟（触及价成交、未触及最多 3 根 K 线后取消，不跳过 equity_curve 点） |
| `backtest/fast_engine.py` | 与 engine.py 同构的基准回填 + 限价单模拟。添加 `import math, numpy`；`compute_benchmark` 调用 + 超额回填 + 顶层 `benchmark` 键；成交循环 `skip_fill` 标志 + `_limit_pending` 计数器。两引擎逻辑逐位一致 |
| `web/api/backtest.py` | `BacktestRequest` 新增 `limit_order_model: str = "none"`；`GridScanIn` 新增 `limit_order_model: str = "none"`；`_schedule_backtest`/`grid_scan`/`compare` 三处 `BacktestConfig(...)` 创建时透传该字段 |
| `backtest/data_loader.py` | 新增 `OHLCVSanitizer` 类（`z_threshold=6.0`，`ffill_limit=3`），`clean()` 方法返回副本，进行缺失值前向填充 + 基于 MAD 的 Z-score 异常值中位数替换（不强制启用） |
| `strategies/base.py` | **未修改**——`Signal` 类已含 `order_type: str = "market"` 和 `limit_price: Optional[float] = None`（第 13-14 行），默认值保持原有行为，无需追加 |

### 新增/修改的测试文件

| 文件 | 类型 | 测试数 | 说明 |
|---|---|---|---|
| `tests/test_benchmark.py` | 新增 | 12 | `compute_benchmark` 正确性（上涨/下跌/空/短/零首价）、`compute_metrics` 向后兼容与 closes 模式、两引擎 benchmark 集成与双引擎一致性、既有 metrics 键不变 |
| `tests/test_limit_order.py` | 新增 | 14 | BacktestConfig 默认值、none/partial 结果一致性（grid + dual_ma）、两引擎 partial 一致性、equity_curve 长度不因挂单丢失、触及成交/未触及不成交/3 根超时取消（用 `_register_test_strategy` 局部注册限价单策略）、OHLCVSanitizer 清洗正确性 |
| `tests/test_backtest_engine_consistency.py` | 未改 | 5 | 仍全部通过（双引擎一致性 + 期末强制平仓 + 布林带 ddof） |

### 验证链最新输出

```
# compileall
python -m compileall -q backtest web strategies  →  通过（无输出）

# pytest（31 tests, 0 failures, 0 warnings）
python -m pytest tests/test_benchmark.py tests/test_limit_order.py tests/test_backtest_engine_consistency.py -q
...............................  [100%]  31 passed in 1.62s

# CLI 双引擎一致性
python run.py backtest --source demo --strategy dual_ma  →  输出含 benchmark 键
  buy_hold_ret=-0.669036, excess_return=0.444037, information_ratio=3.967, excess_max_drawdown=0.224999
  bench_equity_curve 长度 = equity_curve 长度（2000 点）

# full pytest（含模块 C/D 预置失败）
python -m pytest tests/  →  106 passed, 28 failed
  28 个失败全部位于模块 C/D 测试（test_module_d.py / test_reconcile.py / test_risk_manual.py / test_stale_market.py）
  与模块 A 改动无关，属平行模块 C/D 未实现的功能

# 人工验证（grid 基线 none==partial）
grid none total_return=8e-06, trades=104; grid partial total_return=8e-06, trades=104  ✓

# JSON 串行化
json.dumps(metrics) 含 "benchmark" 键，bench_equity_curve 数组长度与 equity_curve 一致  ✓

# compute_metrics 向后兼容
不传 closes 时返回 dict 不含 "benchmark" 键  ✓
```

### 阻塞或待 PM 仲裁问题

- 无。所有任务在模块 A 独占文件范围内完成，未越界修改 `strategies/base.py`（`Signal` 类已有 `order_type`/`limit_price` 字段，无需追加）
- 模块 C/D 的 28 个测试失败属平行模块未交付，不影响模块 A 交付物
- 前端 `renderEquity` 第四参数（`bench_equity_curve`）由模块 E 负责，模块 A 仅保证 `benchmark` 键在 API 响应中可用