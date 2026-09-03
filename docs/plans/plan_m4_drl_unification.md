# M4 DRL 口径统一 — 实施计划

> 独占文件：`drl/agent.py`、`drl/env.py`
> 任务：统一训练/验证/评估的 vol_penalty 口径，确保选择依据与训练目标一致

---

## 问题分析

### 当前状态

`train_drl()` 函数（`drl/agent.py`）中：

| 环境 | vol_penalty | 用途 |
|------|------------|------|
| 训练环境 `TradingEnv`（行 377-386） | `vol_penalty` 默认 20.0 | 训练时奖励 = 收益 - 波动率惩罚 |
| 验证环境 `val_env`（行 431-436） | `vol_penalty=0.0` | 验证时评估纯收益（无惩罚） |
| OOS 环境 `oos_env`（行 540-545） | `vol_penalty=0.0` | 最终独立评估 |
| `evaluate_agent()`（行 648-652） | `vol_penalty=0.0` | 部署评估 |

**问题**：训练时智能体学会在高波动环境下降低仓位（因为惩罚），但验证时惩罚关闭，导致：
1. 验证收益 `val_ret` 不等于训练时的实际体验
2. 模型选择依据 `best_val_ret` 来自 `vol_penalty=0.0` 环境，但训练目标包含惩罚
3. 口径不一致，可能选出"在训练中表现差但验证时恰好好"的模型

---

## T1. 统一惩罚系数

**Files**：`drl/agent.py`

### 方案 A（推荐）：验证环境也使用 vol_penalty

验证环境与训练环境使用相同的 `vol_penalty`，确保选择依据与训练目标一致。

```python
# 修改前（行 431-436）
val_env = TradingEnv(val_df, ..., vol_penalty=0.0, ...)

# 修改后
val_env = TradingEnv(val_df, ..., vol_penalty=vol_penalty, ...)
```

同样修改 OOS 环境（行 540-545）和 `evaluate_agent()`（行 648-652）。

### 方案 B：改用 equity 指标（如夏普比）作为选择依据

保留 `vol_penalty=0.0` 的验证环境，但选择依据从 `val_ret`（总收益）改为 `val_sharpe`（夏普比）或 `val_equity`（最终权益）。

### 选择：方案 A

方案 A 更直接——验证环境与训练环境一致，选择依据与训练目标对齐。

---

## T2. 验证评估指标增强

**Files**：`drl/agent.py`

除 `val_ret` 外，额外记录 `val_sharpe`、`val_max_dd` 等指标，供更全面的模型选择。

```python
# 在 val 评估段（行 484-490）增加
v_equities = [info.get("equity") for info in vtraj.get("infos", []) if info.get("equity") is not None]
val_sharpe = _equity_sharpe(v_equities) if len(v_equities) > 2 else 0.0
val_max_dd = _equity_drawdown(v_equities) if len(v_equities) > 2 else 0.0
row["val_sharpe"] = round(val_sharpe, 4)
row["val_max_dd"] = round(val_max_dd, 4)
```

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| `drl/agent.py` | `train_drl()` 中 val_env/OOS_env/evaluate_agent 的 `vol_penalty` 从 0.0 改为使用训练时的 `vol_penalty`；新增 `val_sharpe`/`val_max_dd` 记录 | 无冲突 |
| `drl/env.py` | 无需改动 | 无冲突 |

---

## 验收标准

- [ ] 验证环境 `val_env` 的 `vol_penalty` 与训练环境一致
- [ ] OOS 评估环境 `oos_env` 的 `vol_penalty` 与训练环境一致
- [ ] `evaluate_agent()` 使用 `vol_penalty=vol_penalty`（从 cfg 读取）
- [ ] 训练历史记录新增 `val_sharpe`/`val_max_dd` 键
- [ ] `python -m compileall -q drl`
- [ ] 现有 DRL 训练测试通过（不影响已训练模型的推理）