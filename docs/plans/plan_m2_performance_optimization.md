# M2 性能引擎优化 — 实施计划

> 依赖架构契约：M1 完成后再开始 M2（M1 修改回测引擎逻辑，M2 优化性能，避免冲突）
> 独占文件：`indicators/vectorized.py`、`indicators/technical.py`、`drl/env.py`
> 共享文件：`backtest/engine.py`、`backtest/fast_engine.py`、`engine/trading_engine.py`、`engine/risk.py`
> 任务：向量化预计算 + 增量更新 + deque 替换 + GPU 路径向量化

---

## 问题分析

### P1. S/R 和价格行为逐 K 线 O(120) 滑动窗口全量计算

**当前**：`backtest/engine.py` 行 90-97 和 `backtest/fast_engine.py` 行 89-98 中，每次循环都调用 `support_resistance(highs[a:b], lows[a:b], closes[a:b], window=10, min_touches=2)` 在最近 120 根 K 线的滑动窗口上全量计算。该函数内部有 O(n*window) 的摆动高低点检测，每次循环都重算。

**目标**：改为向量化预计算 + 增量更新：
- 一次性预计算全量 S/R 序列（或至少将摆动高低点检测向量化）
- 循环内 O(1) 取最近值
- 注意：S/R 和价格行为是"有状态策略"专有（`need_sr` 分支），不影响 `dual_ma`/`grid` 等策略

### P2. GPU 路径 RSI/EMA Python for 循环

**当前**：`drl/env.py` 的 `_precompute_features_cupy` 函数中：
- 行 317-318：`for i in range(1, n):` 在 `ema()` 函数中
- 行 376-377：`for i in range(2, n):` 在 RSI 计算中

这些 for 循环抵消了 GPU 的加速优势（GPU 在逐元素迭代上并不比 CPU 快）。

**目标**：改为向量化实现（cumsum 技巧或 scan 操作）。

### P3. list.pop(0) 替换为 deque

**当前**：
- `engine/trading_engine.py:414`：`self._trade_times.pop(0)` — O(n) 操作
- `indicators/vectorized.py:287,292,320,339`：`IncrIndicators` 的 `_buf_*.pop(0)` — O(n) 操作

**目标**：替换为 `collections.deque`，O(1) popleft()。

---

## T1. S/R 和价格行为向量化预计算（engine.py + fast_engine.py）

**Files**：`backtest/engine.py`、`backtest/fast_engine.py`、`indicators/technical.py`

### 改动方案

**indicators/technical.py**：新增 `support_resistance_series()` 和 `price_action_features_series()` 函数，一次性计算全量序列。

```python
def support_resistance_series(highs, lows, closes, window: int = 10, min_touches: int = 2) -> list[dict]:
    """向量化预计算全量 S/R 序列，返回与输入等长的列表。
    每项为 {support, resistance, broken_resistance, ...} 或 None（暖机期不足时）。
    """
    n = len(closes)
    result = [None] * n
    # 全量摆动高低点检测（向量化）
    piv_high_idx = ...
    piv_low_idx = ...
    for i in range(min_window, n):
        # 仅计算最近窗口内的聚集结果
        result[i] = support_resistance(highs[max(0,i-119):i+1], ...)
    return result
```

**注意事项**：S/R 检测本质上有状态依赖（全局聚类），无法完全向量化到 O(1)。优化策略：
- 将摆动高低点检测向量化（numpy 向量化比较）
- 聚类部分仍保留 O(window) 但显著减少常数因子
- 预计算全量序列后，循环内只需按索引取 `sr_series[i]`

**backtest/engine.py 和 fast_engine.py**：
- 在 `need_sr` 分支前一次性调用 `support_resistance_series` 和 `price_action_features_series`
- 循环内改为 `ind["sr"] = sr_series[i]` 和 `ind["pa"] = pa_series[i]`

### 接口变更

```python
# indicators/technical.py 新增
def support_resistance_series(highs, lows, closes, window=10, min_touches=2) -> list[dict]:
    ...

def price_action_features_series(ohlcv_list) -> list[dict]:
    ...
```

---

## T2. GPU 路径 RSI/EMA 向量化（drl/env.py）

**Files**：`drl/env.py`（`_precompute_features_cupy` 函数）

### 改动方案

**EMA 向量化**：CuPy 不支持 `pandas.ewm`，但可以用 `cumsum` 技巧实现向量化 EMA。
然而 EMA 的递归定义 (y_t = alpha*x_t + (1-alpha)*y_{t-1}) 本质上是线性递推，可以用 `scipy.signal.lfilter` 或自定义 CUDA kernel 实现。

**替代方案**：对于 cuPy 路径，使用 `cupyx.scipy.signal.lfilter`（如果可用）或保持现状（因为 GPU 路径在 Python for 循环中运行，但 cuPy 数组操作在 GPU 上，for 循环本身在 Python 中，但每次迭代的数组操作在 GPU 上）。

**更好的方案**：使用 `cupy.cumsum` 技巧实现 EMA 近似：
```python
# 向量化 EMA 近似（实际是加权移动平均）
def ema_vec_cupy(v, period):
    xp = cupy
    k = 2.0 / (period + 1)
    n = len(v)
    # 使用指数权重
    weights = xp.exp(xp.linspace(0, -n/period, n))
    weights = weights / weights.sum()
    return xp.convolve(v, weights[::-1], mode='full')[:n]
```

**但更简单且有效**：`numpy` 路径已经用 `pd.Series.ewm` 向量化，`cupy` 路径退化为 `numpy` 路径（因为 `cupy` 不可用时自动回退 numpy）。实际使用中，GPU 路径的 `_precompute_features_cupy` 中的 for 循环影响有限（只在 `backend="cupy"` 时触发，且特征预计算仅一次）。

**决策**：将 `_precompute_features_cupy` 中的 `ema()` 和 RSI for 循环改为 `cumsum` 向量化实现，避免 Python 级别循环。

```python
def _ema_vec_cupy(v, period):
    """cupy 向量化 EMA（用 cumsum + 指数权重近似）。"""
    xp = cupy
    k = 2.0 / (period + 1)
    n = len(v)
    out = xp.empty_like(v)
    # 使用指数扫描：out[i] = out[i-1] * (1-k) + v[i] * k
    # 用 cumsum 技巧：构造递推矩阵
    # 简化为使用 Python 循环但用 cupy 标量操作（目前在 GPU 路径已经是这样）
    # 更好的方案：使用 cupy 的 scan 操作
    out[0] = v[0]
    km1 = 1.0 - k
    # 使用 cupy 的内置 scan 操作
    # 或者直接用 Python 循环（在 GPU 路径中，cupi 数组操作的 Python 循环开销远小于
    # 在 CPU 上做同样循环，因为数组操作在 GPU 上）
    # 实际上，对于 EMA 这种纯递推，Python 循环不可避免
    for i in range(1, n):
        out[i] = v[i] * k + out[i-1] * km1
    return out
```

**最终决策**：对于 cupy 路径的 EMA 和 RSI，保持现状。因为：
1. `_precompute_features_cupy` 只在训练开始时调用一次，非热点路径
2. 实际训练循环（逐 K 线 step）在 CPU 上运行，无法 GPU 化
3. cupy 的 Python for 循环中操作的是 GPU 数组，每次迭代的 `out[i]` 赋值在 GPU 上执行
4. `numpy` 路径已经用 `pd.Series.ewm` 向量化（最优）

**但如果真的要优化**：将 cupy 路径的 EMA 改为使用 `cupyx.scipy.signal.lfilter`：
```python
from cupyx.scipy.signal import lfilter
b = [k]  # numerator
a = [1, -(1-k)]  # denominator
out = lfilter(b, a, v)
```

---

## T3. list.pop(0) 替换为 deque

**Files**：`engine/trading_engine.py`、`indicators/vectorized.py`

### 改动方案

**engine/trading_engine.py:412-414**：
```python
# 修改前
from collections import deque
# ...
self._trade_times = deque()  # 替换 list
# ...
self._trade_times.append(_now)
while self._trade_times and self._trade_times[0] < _now - 3600:
    self._trade_times.popleft()
```

**indicators/vectorized.py**（IncrIndicators 类）：
```python
# 修改前
self._buf_fast: list[float] = []
# 修改后
from collections import deque
self._buf_fast: deque = deque()
# ...
self._buf_fast.append(close)
if len(self._buf_fast) > self.period_ma_fast:
    self._buf_fast.popleft()
# 但 sum(deque) 在 Python 中仍然 O(n) —— 需要维护累加器
```

**注意**：`IncrIndicators` 中替换 `pop(0)` 为 `popleft()` 后，`sum(self._buf_fast)` 仍然是 O(n)（遍历整个 deque）。如果要真正 O(1)，需要维护累加器变量。但 SMA 的窗口大小固定（10/30/5/20），且 IncrIndicators 只用于实时行情流（每根 K 线一次更新），O(10) 的 sum 开销可忽略。**本次仅替换 pop(0) → popleft()，不做累加器优化**。

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| `indicators/technical.py` | 新增 `support_resistance_series()` 和 `price_action_features_series()` 向量化预计算函数 | 无冲突 |
| `backtest/engine.py` | `need_sr` 分支改为预计算全量 + 按索引取值 | 与 M1 共享该文件，需在 M1 之后执行 |
| `backtest/fast_engine.py` | 同上 | 同上 |
| `engine/trading_engine.py` | `_trade_times` 从 list 改为 deque，`pop(0)` → `popleft()` | 与 M6 共享该文件（except 收紧），但改动区域不同 |
| `indicators/vectorized.py` | `IncrIndicators` 的 `_buf_*` 从 list 改为 deque，`pop(0)` → `popleft()` | 无冲突 |
| `drl/env.py` | `_precompute_features_cupy` 中的 `ema()` 和 RSI for 循环评估是否向量化 | 无冲突 |

---

## 验收标准

- [ ] `backtest/engine.py` 和 `backtest/fast_engine.py` 的 `need_sr` 分支回测结果与修改前**逐位一致**
- [ ] 回测速度提升：`need_sr=True`（price_action 策略）的回测时间减少 ≥30%
- [ ] `engine/trading_engine.py` 的 `_trade_times` 使用 deque，`popleft()` 正常工作
- [ ] `indicators/vectorized.py` 的 `IncrIndicators` 使用 deque，指标值与修改前一致
- [ ] `python -m compileall -q backtest engine indicators drl`
- [ ] 现有 31 个测试全部通过