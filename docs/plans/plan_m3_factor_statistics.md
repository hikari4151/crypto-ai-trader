# M3 因子统计严谨性 — 实施计划

> 独占文件：`factors/analysis.py`、`factors/mining.py`、`factors/model_factor.py`
> 任务：IC 序列自相关修正 + 收紧因子筛选门槛

---

## 问题分析

### P1. IC 序列自相关导致 std 低估

**当前**：`factor_ic()` 函数（`factors/analysis.py` 行 48-70）中，ICIR 计算为：
```python
ic_series = _rolling_ic(factor_series, close, h=h, method=method, window=30)
ics = ic_series[ic_series.notna()]
icir = float(ics.mean() / (ics.std() + 1e-12)) if len(ics) > 5 else 0.0
```

`_rolling_ic` 默认 `step=window`（非重叠窗口），但即使非重叠，IC 序列仍可能存在自相关（相邻窗口的 IC 受重叠数据影响），尤其在 `step < window` 时更严重。`ics.std()` 低估了真实标准差，导致 ICIR 被高估。

**目标**：添加 Newey-West 修正估计量，调整 IC 序列的标准差估计。

### P2. min_abs_ic=0.01 过于宽松

**当前**：`DEFAULT_GATES["min_abs_ic"] = 0.01`（`factors/analysis.py` 行 194）

**目标**：收紧到 0.03（统计意义上更显著的门槛）。

---

## T1. Newey-West 修正 ICIR

**Files**：`factors/analysis.py`

### 改动方案

在 `factor_ic()` 中，对 IC 序列的自相关进行 Newey-West 修正：

```python
def _newey_west_std(series: np.ndarray, max_lags: int = None) -> float:
    """Newey-West 异方差自相关一致性标准差估计。
    
    HAC 估计量：Var(mean) = (1/N) * (gamma_0 + 2 * sum_{j=1}^{L} w_j * gamma_j)
    其中 gamma_j = 序列 j 阶自协方差，w_j = 1 - j/(L+1)（Bartlett 核）
    L = max_lags（默认取 N^(1/4) 经验法则）
    """
    n = len(series)
    if n < 10:
        return float(np.std(series, ddof=1))
    if max_lags is None:
        max_lags = int(n ** 0.25)  # Newey-West 1994 经验法则
    # 去均值
    demeaned = series - series.mean()
    # gamma_0 = 方差
    gamma_0 = np.sum(demeaned ** 2) / n
    # 自协方差
    var_hac = gamma_0
    for j in range(1, min(max_lags + 1, n - 1)):
        gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / n
        w = 1.0 - j / (max_lags + 1)  # Bartlett 核
        var_hac += 2 * w * gamma_j
    return float(np.sqrt(max(var_hac, 0)) / np.sqrt(n) * np.sqrt(n / (n - 1)))  # 无偏调整
```

在 `factor_ic()` 中：
```python
# 原代码
icir = float(ics.mean() / (ics.std() + 1e-12)) if len(ics) > 5 else 0.0
# 改为
nw_std = _newey_west_std(ics.to_numpy())
icir = float(ics.mean() / (nw_std + 1e-12)) if len(ics) > 5 else 0.0
```

同时保留 `ic_std` 作为原始标准差（用于对比），新增 `ic_std_nw` 键。

### 接口变更

```python
# factor_ic() 返回字典新增键
{
    "ic": ..., "rank_ic": ..., 
    "icir": ...,          # 基于 Newey-West 修正的 ICIR
    "ic_std": ...,        # 原始标准差（保留）
    "ic_std_nw": ...,     # 新增：Newey-West 修正标准差
    ...
}
```

---

## T2. 收紧因子筛选门槛

**Files**：`factors/analysis.py`、`factors/mining.py`、`factors/model_factor.py`

### 改动方案

**factors/analysis.py**：
```python
DEFAULT_GATES = {
    "min_abs_ic": 0.03,      # 原 0.01 → 0.03
    "min_abs_icir": 0.1,     # 保持不变
    "max_turnover": 0.5,     # 保持不变
    "min_samples": 20,       # 保持不变
}
```

**factors/mining.py**：同步更新默认参数
- 行 274：`top_n: int = 8, min_abs_ic: float = 0.01` → `min_abs_ic: float = 0.03`
- 行 297：`ic_window: int = 120, min_abs_ic: float = 0.01` → `min_abs_ic: float = 0.03`

**factors/model_factor.py**：
- 行 117：`val_ic < DEFAULT_GATES["min_abs_ic"]` 自动使用更新后的值（0.03）

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| `factors/analysis.py` | 新增 `_newey_west_std()` 函数；`factor_ic()` 中 ICIR 改用 NW 修正；`DEFAULT_GATES["min_abs_ic"]` 0.01→0.03 | 无冲突 |
| `factors/mining.py` | 默认参数 `min_abs_ic: float` 从 0.01 改为 0.03（两处函数签名） | 无冲突 |
| `factors/model_factor.py` | 无改动（自动引用 `DEFAULT_GATES`） | 无冲突 |

---

## 验收标准

- [ ] `_newey_west_std()` 在无自相关序列上返回与 `np.std()` 相近的值（差异 < 5%）
- [ ] `_newey_west_std()` 在正自相关序列上返回大于 `np.std()` 的值
- [ ] `factor_ic()` 返回的 `icir` 在修改后 <= 修改前（NW 修正不会降低 ICIR）
- [ ] `min_abs_ic=0.03` 后，`factor_quality_gate` 在现有因子库上过滤掉更多因子
- [ ] `python -m compileall -q factors`
- [ ] 现有因子相关测试通过