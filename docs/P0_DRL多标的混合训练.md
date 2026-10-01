# P0 多标的混合训练（DRL 经验广度）— 设计与实施

> 日期：2026-10
> 相关文件：`drl/multi_asset.py`（新）、`drl/agent.py`、`drl/evolve_engine.py`、
> `config/settings.py`、`web/api/drl.py`、`web/static/index.html`

---

## 背景：DRL 训练偏离了目的

单标的 DRL 训练的本质局限（2026-10 诊断）：

1. **经验广度为零**：`train_drl` 只吃一个标的的一段行情，train/val/OOS 从同
   一段行情里切——"样本外验证"实际是"验证记住的这段行情的模式是否稳定"，
   不是"会不会交易"。
2. **经验规模太小**：持续进化每轮只训 8ep，纯 CPU 4-worker 并行，一个月也
   撑不起 DRL 应有的"人类几年经验"。
3. **能力天花板**：为装进 Pine 网络被压到 16×16，只能学"波动率依赖的仓位
   规则"，不是"跨标的可迁移的交易能力"。

**目的**：DRL 的核心价值是"用机器时间压缩人类经验"——几周/数月训练达到人类
几年的学习能力。这靠两样东西：经验规模（算力吞吐）+ 经验广度（见过的局面
覆盖人类一生的多样性）。本改动补上后者：**多标的混合训练**。

---

## 设计

### 核心机制：标的池 × 随机时间窗切片

- 训练时构建 `MultiAssetPool`（`drl/multi_asset.py`）：
  - **主标的** = 训练段 DataFrame（train_ratio 切分后）+ 预计算特征
    （与旧路径同源，单标的模式行为完全不变）。
  - **额外标的** = `extra_dfs: {name: df}`，每个标的整段预计算特征一次，
    episode 采样时按行切片（rolling 特征自带历史，无前视）。
- 每个 episode 以 `extra_mix_ratio` 概率随机选一个额外标的，在其数据范围内
  随机起点取 `window_len` 长切片；否则跑主标的全训练段。
- 网络共享：所有标的特征列同维（13 列 × state_window + 账户状态 + 可选因子），
  维度守卫排除列结构/特征维度不符的标的。

### 部署语义不变

- 额外标的**只作为经验来源**，不参与 val/OOS 判据；val 选模、OOS 硬门、
  回退保护、Pine 导出全以主标的为准（保证既有安全网不因多标而松动）。
- 单标的模式（不传 `extra_dfs`）路径字节级不变：`MultiAssetPool` 只有主标的，
  `sample()` 恒返回 start=0/全训练段，与旧 `_collect_one` 完全一致。

### 因子列的跨标一致性

主标的有因子列时，额外标的必须能重建同维度因子列才能lump成一个网络：

| 因子来源 | 额外标的能否复算 | 处理 |
|---------|-----------------|------|
| `factor_composite` 配方 | ✅ expanding-z 在任意标的 df 上复算（无前视，与部署端同口径） | 注入 `factor_builder`，全部标的带因子列 |
| `factor_expression` | ✅ 同一表达式对任意标的计算 | 同上 |
| `factor_values`（预计算数组，无配方无表达式） | ❌ 无法跨标重建 | 降级单标的训练 + 警告 |

### evolve 引擎集成

- `_train_strategy_drl_once` 默认把 `evolve_symbols` 池中除当前主标的外全部
  标的拉取进 `extra_dfs_pool`（尽力而为：单标的拉取失败不阻断）。
- `_training_identity` 的 config 指纹加入 `multi_symbol`/`multi_extra_names`：
  多标产物与单标产物**不可互比**（防止跨模式误回退）。
- 多标模式下**跳过跨标传导 OOS**：训练已见全部池内标的，跨标 OOS 不再构成
  样本外验证（主标的本地 OOS 门照常把关）。
- 经验回放缓冲区：多标模式下**不按主标轮换重建**（训练本体已是跨标混合采样，
  按主标重建会每轮清空缓冲，丢掉跨轮经验）。
- 单标模式（`evolve_strategy_drl_multi_symbol=False`）全部行为与旧版一致。

---

## 新配置

| 键 | 默认 | 说明 |
|----|------|------|
| `evolve_strategy_drl_multi_symbol` | True | 持续进化多标混合训练开关 |
| `evolve_strategy_drl_multi_mix_ratio` | 0.5 | 每条轨迹来自额外标的的概率 |
| `train_cfg.extra_mix_ratio` | 0.5 | 手动训练的多标混合比例 |
| `train_cfg.episode_window` | 0 | 采样切片长度（0=主标的训练段长度） |
| `TrainIn.extra_symbols` | [] | 手动训练 API：逗号分隔的额外标的名 |

---

# P0 并行采样上量（经验规模，2026-10）

多标混合补上"经验广度"后，下一块短板是"经验规模"：并行收集 worker 上限此前
硬编码 `min(n_episodes, 4)`——n_episodes 提到 16/32 时仍只有 4 条同时在跑，
每轮见过的局面翻倍但墙钟成本也随之翻倍。本次放开并行度：

- `_resolve_collect_workers(n_episodes, cfg_workers)`：解析并行线程数。
  0（默认）= 自动 `min(n_episodes, max(4, cpu_count))`——至少 4、不超
  episode 数（不造空线程）、不超物理核数（超线程并行收益递减）；
  显式值夹到 `[1, n_episodes]`。
- `train_drl` 的 `ThreadPoolExecutor` 用它（不再写死 4），result 透出
  `collect_workers` 供审计。
- 默认批量提升：持续进化 `evolve_strategy_drl_n_episodes` 8→16、手动训练
  API `TrainIn.n_episodes` 4→8；`collect_workers` 均可配置（0=自动）。
- 多标的模式下每 episode 是独立"标的×时间窗"切片，批量越大、每轮见过的
  局面越多，PPO 梯度越稳——与经验广度正交叠加。

安全设计：

- 确定性不变：per-worker rng（seed 派生）独立，同 seed 两次训练 history
  逐值一致（含 >4 线程，测试覆盖）。
- 内存口径随批量更新（settings 注释同步）：5m/5000根/state_window=4 单条
  轨迹 ≈1.24MB，n_episodes=16 → 每轮 16×8=128 条 ≈155MB；回放 capacity
  默认仍 64（覆盖最近半轮，旧轨迹经重要性比率自然衰减），可热调。
- 全部沿用既有防线：val 选模/OOS 硬门/回退保护/部署门不变，只改采样吞吐。

---

# P1 课程学习（能力层次，2026-10）

经验广度（多标的）+ 经验规模（并行上量）之后，第三块短板是"能力层次"：
agent 从随机初始化直接面对全部复杂行情——高波动段噪声大、奖励稀疏，第一性
原理上就难学。人类是先易后难，课程学习把同一逻辑搬给 agent：

- **难度定义**：窗口内已实现波动率（vol_5 特征）分位——低波动段（趋势清晰、
  奖励信号强）= 简单，高波动段 = 难。跨标的可比（"该窗口在本标历史中的难度
  百分位"）。
- **课程曲线**：`_allowed_difficulty(frac) = curriculum_start + (1-start)*frac`。
  frac=0（刚开始）只允许最低 `curriculum_start` 分位的窗口；frac=1（尾声）
  放开全部。进度由训练轮次驱动（ep/episodes）。
- **实现**：`MultiAssetPool.sample(rng, frac)` 拒绝采样——只在难度分位
  ≤ 当前允许值的起点里随机挑（最多重试，超限退回最低难度窗口）。
- **正交叠加**：多标的选标的 → 课程按难度过滤窗口，两者独立。

关键设计（向后兼容）：

- **默认关闭**（`curriculum=False`）：采样行为与旧路径完全一致，不改变任何
  既有模型/测试。
- **开启时单标的也走 pool**：主标的若不经过 pool，无法按难度过滤（主标在这
  段行情里也有高/低波动子段）——开启课程即统一走标的池采样。
- 课程难度过滤不参与 val/OOS 判据：val/OOS 仍按主标的整段/时间窗评估，
  只改训练采样分布。
- 确定性不破坏（per-worker rng 独立，拒绝采样消耗序列确定）。

配置：

| 键 | 默认 | 说明 |
|----|------|------|
| `evolve_strategy_drl_curriculum` | False | 持续进化课程开关 |
| `evolve_strategy_drl_curriculum_start` | 0.3 | 课程起始难度分位（0=只采最易） |
| `train_cfg.curriculum` / `curriculum_start` | False / 0.3 | 手动训练同参 |

---

## 调用方式

```python
# 手动训练（web API 或直接调用）
from drl.agent import train_drl
res = train_drl(main_df, cfg, extra_dfs={"ETH/USDT": eth_df, "SOL/USDT": sol_df})

# 持续进化引擎（默认开启，settings 可关）
# evolve_engine._train_strategy_drl_once 自动拉取 evolve_symbols 池全部标的
```

---

## 测试

`tests/test_drl_multi_symbol.py`（16 项）：

1. 单标的向后兼容：`multi_symbol=False`、结果可复现（同 seed 两次历史逐值一致）。
2. 采样正确性：切片特征维度一致、单标的恒主标全段、mix_ratio=0 只用主标的、
   过短标的排除、因子列复算、因子维度守卫降级。
3. 多标训练跑通：返回 `multi_symbol=True` + 额外标的名单，agent 维度正确。
4. API 层：`_load_extra_dfs` 跳过非法/失败/数据不足的标的；CSV 源不支持多标。
5. evolve 集成：多标模式把池内标的喂给 train_drl、跨标 OOS 跳过、单标模式
   跨标 OOS 照常执行；回放缓冲区跨标复用 vs 单标重建。

---

## 验证结果

- 全量测试套件通过（含既有 DRL/evolve 全部测试，无回归）。
- ruff + mypy（新增模块零错误；既有文件无新增类型错误）。
- 实际多标训练 smoke：ETH/SOL 双额外标的训练可跑通，4ep 0.3s，状态维度 28。
- 单标确定性：同 seed 两次训练 history 逐值一致。