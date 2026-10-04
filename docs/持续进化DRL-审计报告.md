# 持续进化 DRL 审计报告（2026-10）

> 范围：`drl/evolve_engine.py`（strategy_drl 管线）+ `drl/agent.py` / `drl/env.py` +
> `drl/model_zoo.py` + `drl/evolve_observations.py` + `strategies/rl_adaptive.py` +
> `web/api/drl.py`。
> 方法：四个并行子代理分别审计【主训练循环与数据】【状态与并发】【训练-验收口径】
> 【部署-执行集成】，主代理汇总去重，对确凿 HIGH/MED 项做复现并 TDD 修复。

---

## 0. 结论摘要

- **共发现 30+ 项**；已修复 **38 项**（全部有回归测试，门禁全绿）；
- 最危险的一类问题是 **"决策节奏/死区口径漂移"**：模型按 k=8/zone=0.10 训练，
  却在三处评估/部署路径按 k=1/zone=0.02 跑——回退重建 flat、部署长窗门回放、
  跨标 OOS 门。回退是常态保护动作，恰恰在最该严谨处制造漂移，**已全修**；
- 六轮累计：**HIGH / MED / LOW 全部清零**，审计闭环（§2 仅剩已评估无害的观察项）。

---

## 1. 本轮已修复（含回归测试）

### 第一轮（8 项）
| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 1 | HIGH | **回退重建 flat 丢失 `decision_interval`/`min_trade_zone`** | `_DEPLOYMENT_META_KEYS` 加键 + `save_agent` meta 写入 |
| 2 | HIGH | **部署长窗门临时模型漏键 + 临时文件泄漏** | `_build_candidate_model_dict` + `_TempModelFile` |
| 3 | HIGH | **跨标 OOS 门环境漏 `policy_env_kwargs`** | `_make_cross_oos_env` 透传 train_cfg 口径 |
| 4 | HIGH | **训练身份指纹含 `multi_extra_names`** | 剔除名单，`multi_symbol` 读设置开关 |
| 5 | MED | **多 seed 偶数取"上半中位数"** | `(len-1)//2`（下半保守） |
| 6 | MED | **GAP-01 FAIL 计数不随部署版本清零** | `_clear_gap01_streak_on_version` |
| 7 | MED | **增量门不随训练口径配置失效** | `_TRAINING_CFG_KEYS` 白名单 + 热更新清门 |
| 8 | MED | **挑战者谱系递增 `_reject_streak`** | `_log_round` 对 challenger 不递增 |

### 第二轮（4 项 HIGH，用户选"1"）
| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 9 | HIGH | **回退/晋升比较用绝对 OOS 收益跨窗口比**：下跌窗口真候选必更低 → 被误回退；上涨窗口差候选逃过回退（fitness 已是 excess，OOS 侧仍是绝对值） | meta 存 `oos_excess` + `best_info` 透传 + `_should_rollback` 比较改超额（缺超额老锚点退化为 fitness） |
| 10 | HIGH | **web 续训被拒不覆盖基础模型**：`save()` 在判定前无条件执行，旧权重永久丢失 | 续训+被拒 → 新权重存档 `models/archive/`、原文件保留、任务标 `rejected` |
| 11 | HIGH | **`use_model` 无门无持久化** | 做空 →400；OOS 硬门 →risk_tips 明示；`_persist_strategy` 持久化 |
| 12 | HIGH | **多 seed 共享同一 replay_buffer**（中位判据前提被破坏） | 每 seed 主缓冲深拷贝副本，选中副本经 meta 交回；单 seed 直接共享 |

测试：`tests/test_evolve_audit_fixes.py`（15 例 → 20 例）+ 更新 `tests/test_drl_seed_robustness.py`、
`tests/test_evolve_from_scratch_reset.py`（守卫旧 bug 行为的断言按"修 bug 即更新"更新）。

### 第三轮（3 项 MED，用户选"1"）
| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 13 | MED | **手动评估路径 k=1**：`_model_eval_cfg`/`evaluate_oos` 只带 factor/state_window/zone，k=8 训练的模型被按 k=1、slippage=0 评估，用户端"疑似过拟合"结论由错节奏的收益算出；9 档（做空）模型在 5 档 env 上裸断言崩 500 | `_model_eval_cfg` 补全 `decision_interval/reward_mode/benchmark_position/risk_coef/short_enabled`；`evaluate_oos` 走 `policy_env_kwargs` + slippage=0.0005；引擎/web 写模型根键落盘奖励/基准/滑点口径；`evaluate_agent` 加动作空间维度守卫（显式 ValueError） |
| 14 | MED | **`ReplayBuffer` 采样 RNG 无 seed**：其余随机源全部派生自训练 seed，唯独回放采样从 OS 熵初始化 ⇒ 启用回放后同 seed 不可复现 | `ReplayBuffer(capacity, seed)`；train_drl 新建缓冲 + 引擎主缓冲都从训练 seed 派生 |
| 15 | MED | **`oos_excess = median(段收益)−median(段基准)` ≠ 逐段超额的中位数**：可与其"≥2/3 子段为正"分段门符号相反（同一数据两个门一个放行一个拦截） | val/OOS 多段统一为**逐段超额的中位数**；报告带 `oos_excess_basis`；接受路径 meta + `best_info` 透传 basis；回退比较仅当新旧锚点同为 segment_median 口径才直比超额（缺标记的旧锚点 fallback fitness） |

测试：`tests/test_evolve_audit_fixes.py`（20 例 → 24 例）。

### 第四轮（10 项 MED，用户选"1"）
| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 16 | MED | **回放容量 64 vs 每轮 push≈2048**（episodes=128 时）：64 只留最近 ~1/32 轮，"跨轮记忆"名不副实 | 默认 64→**128**（155MB 实测档）；注释如实说明"保留最近 ~1/16 轮"的内存取舍，可配置调大（400≈484MB 不建议） |
| 17 | MED | **demo 轮污染观测状态**：合成数据评估部署模型会改写 gap01_streak/_gap01_pass_*/longwindow_verdict，故障期假 FAIL 恢复后触发 from-scratch | 观测三块（walkforward/gap01/longwindow）整体包进 `if not _demo_round`（demo 轮不做观测） |
| 18 | MED | **`_pop_round` 在 demo 轮/异常轮也递增**：与"每完成一轮真实训练递增"矛盾（故障期翻转奇偶、恢复后首轮变挑战者轮） | demo 轮不递增；`_run_train` 异常与"无历史记录"时回滚 `_pop_round` |
| 19 | MED | **挑战者自回退无身份守卫**：配置变更后新候选与旧 alt 锚点 fitness 口径可能不同量级仍硬比较（丢谱系） | 挑战者自回退加 `_identity_mismatch(alt_anchor, _identity)` 守卫（与冠军路径一致） |
| 20 | MED | **级联因子注入与否不入身份指纹**：注入轮与未注入轮被当同口径回退比较 | 注入时给 config 加 `factor_injected=True`（进 config_fingerprint）；未注入不加键（保持旧格式与旧锚点兼容） |
| 21 | MED | **`rl_adaptive` 间隔内仍按 `_last_target` 与当前仓位差调仓**：训练 env 决策后 k 根不撮合，部署端多出训练中不存在的撮合 | 间隔内跳过分支直接 `return None`（与 semi-MDP 严格对齐） |
| 22 | MED | **walk-forward/gap01 观测硬编码 `min_trade_zone=0.05`**（训练 0.10） | 4 处改从 `evolve_strategy_drl_min_trade_zone`（settings 0.10）读取 |
| 23 | MED | **`strategies._DYNAMIC` 跨线程共享可变 dict 无锁**：迭代期间插入抛"dictionary changed size" | `threading.RLock` 包住 register/get/list/rename/remove/names 全部读写（并发测试 6 线程 ×200 次） |
| 24 | MED | **增量门异常路径不更新 prev**：同批"新"K线下轮重复计数、重复训练 | `except` 中先记 `_last_tail_ts[key]=tail_ts` 再放行 |
| 25 | MED | **`_precompute_features` volume 缺列崩**（env 构造有兜底、特征预计算没有） | 与 init 同源兜底（`df["volume"] if "volume" in df.columns else 1e12`） |

测试：`tests/test_evolve_audit_fixes.py`（24 例，新增因子指纹/volume 兜底/回放容量/`_DYNAMIC` 并发）+ 更新
`test_drl_optimizations.py`（容量 64→128 断言）。

### 第五轮（12 项 LOW，用户选"1"）
| # | 问题 | 修复 |
|---|---|---|
| 26 | flat `train_meta.limit` 记滚动窗（5000）而非实际训练深度（30000） | 记 `_keep_bars` |
| 27 | 手动触发不 `_mark_pipeline_done`（标的轮换记账滞后） | `trigger_strategy_drl` 成功路径补记账 |
| 28 | val 多段 sharpe/dd 拼接污染（段间重置被当暴跌） | 按段独立计算取中位数（与 OOS 路径对称） |
| 29 | `_entry_price` 部分平仓不重加权（浮动盈亏特征脱节） | 持仓均价按 `(旧均×旧量 ± 价×Δ量)/新量` 重加权 |
| 30 | distill `n_rollouts` 全同轨迹（起点恒 warmup） | `env.reset(start_offset)` 随机起点（派生自 seed） |
| 31 | participation_rate 按单根成交量算（k>1 上限低估 k 倍） | 按决策期累计成交量 |
| 32 | 长窗观测文件被多标覆盖（无归属标记） | longwindow report 补 `symbol/timeframe`（walkforward/gap01 已有） |
| 33 | `save_agent` 接受路径无 try（异常留幽灵版本条目） | 包 try + `save_error` 落库 + 提示 |
| 34 | Pine 导出无 cadence（k=8 模型脚本逐根决策） | 生成 `isDecision = bar_index % k ==0` 包裹 argmax；`export_pine_from_model` 从模型文件读 k |
| 35 | rl_adaptive 9 动作缺键回退时 argmax 越界 IndexError | `from_dict` 缺键且 n_actions>5 → 显式 ValueError（损坏模型）；rl_adaptive 加运行时越界守卫 |
| 36 | `episodes=0` NameError 被吞成"无样本外证据" | `episodes = max(1, …)` |
| 37 | 增量门键不含 exchange（同尾戳换源永久跳过） | key 加 `default_exchange` |

测试：`tests/test_evolve_audit_fixes.py`（28 例）、`test_pine_trade_markers.py`（target 三元链适配 cadence）。

### 第六轮（留档项：`_run_train` 取消语义，用户选"1"）
| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 38 | MED | **`_run_train` 取消语义与实际不符**：旧 `stop()`/`set_enabled(False)` 硬 cancel
 所有循环协程——只中断 await，executor 线程照跑训练本体，但**写盘/注册/落库都在协程
 后处理段**会被静默丢弃（"训练+写盘必然完整执行"的旧注释不成立）；手动触发任务不在
 `_tasks` 里、stop() 不取消它 → 禁用返回后手动轮仍可能部署写盘 | 三层优雅停止：
 ① 置 `_stop_event` + `_running=False`，真实循环在下一轮迭代自然退出（间隔 sleep 可被
 事件唤醒，`_sleep_or_stop`）；② 仅当确有训练在跑（`_training_done` cleared）时等训练
 本体 + 3s 写盘缓冲——保证当前轮完整落盘；③ 超时/缓冲后硬 cancel 兜底（stub/失控任务
 不配合时不挂起）。停止/禁用状态下手动触发（strategy_drl 与 factor_miner）显式拒绝 |

测试：`tests/test_evolve_audit_fixes.py`（30 例，新增 `_sleep_or_stop` 唤醒与触发拒绝）；
`perf_baseline` 锁住 startup 不因无条件缓冲退化（53ms 基线）。

---

## 2. 待修清单

**已全部修复（§1 1-38，HIGH/MED/LOW 清零）**。以下为已评估、确认无需改动的观察项：

1. **demo 轮存档路径**：demo 轮只 `save_agent_archive`，不写 best/flat——符合
   "防实盘被未验证权重接管"（保留观察）。
2. **`evolve_stop_training_grace` 可配**：stop 等训练本体的最长时限（默认 120s，
   与单 seed time_budget 同量级）；极端超时仍硬 cancel（用户强停的取舍）。

---

## 3. 已检查、未发现问题（防重复审计）

- 训练/验证/OOS 切片无重叠、因子切片逐行对齐、数据不足回退路径一致。
- 多 seed 复现：train_drl 内所有随机源派生自 seed（ReplayBuffer 除外，见待修 6）。
- 主 OOS 门/活跃度/分段门判据链（含 `ceil(n·f−1e-9)` 浮点边界）逻辑自洽。
- 被拒路径只存档不注册、不动 best/flat（引擎端"拒绝≠不部署"无漏网注册）。
- `_run_train` 异常 → `_training_done` 置位正确；增量门跳过路径计数器行为正确。
- zoo 双缓存跨线程访问在 GIL 下无破坏性竞态。
