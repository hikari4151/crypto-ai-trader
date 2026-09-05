# 持续进化回退基线全链路优化设计

- 日期：2026-09-05
- 范围：`factor_miner`、`strategy_drl`、`meta_controller` 的回退基线、部署恢复与运行时同步
- 状态：设计已确认，待实现计划

## 1. 背景与根因

当前持续进化模块的回退动作主要更新 `ModelZoo` 的 `best.json.gz` 与兼容 flat 文件，但“文件已恢复”不等于“系统已经运行该版本”：

- `RLAdaptiveStrategy` 只在首次使用时加载 `_agent`，已有实例会继续使用旧内存权重；
- `TradingEngine` 没有统一的当前策略模型刷新入口；
- `save_agent_best` 通过 `agent.to_dict()` 重写 flat 文件，可能丢失因子表达式、标准化统计量、交易死区、级联信息等部署元数据；
- 手动 API 回退与自动回退路径不一致，手动操作不会统一刷新动态注册、运行时实例和审计状态；
- 锚点只按模型名共享，缺少 symbol、timeframe、窗口和配置身份，可能比较不可比的候选与基线；
- 回退后状态中的 fitness 仍可能是候选值，无法准确表达当前部署版本。

本设计的核心定义是：一次回退/接受只有在磁盘快照、锚点画像、动态注册表、当前运行实例四者一致时才算成功。

## 2. 目标与非目标

### 目标

1. 自动接受、自动回退、手动回退统一经过一个部署编排入口。
2. 版本恢复保留完整部署元数据，并以原子方式更新 best、flat 与锚点指针。
3. 当前运行中的 `rl_adaptive` 与 `meta_controller` 能在回退或接受后热加载对应模型。
4. 阻止不同 symbol、timeframe、窗口、状态维度或配置口径的结果互相比较。
5. 把候选表现、实际部署表现、回退结果和人工操作分别记录并暴露给 API/UI。
6. 因子挖掘的 OOS 不合格模型只能存档，不能成为 factor best 或级联部署基线。
7. 保留现有版本目录、flat 兼容格式和已有安全/OOS/跨标保护。

### 非目标

- 本轮不重写 PPO、因子搜索或 OOS 统计算法。
- 本轮不把模型目录完全拆成按 symbol/timeframe 的新目录体系。
- 本轮不删除旧版本文件；缺少新身份字段的旧版本按未知口径处理。
- 本轮不改变交易下单、撮合或风险规则。

## 3. 设计

### 3.1 规范化部署快照

在现有 `vN.json.gz` 的 `_meta` 中补充规范化部署字段。新保存的版本至少包含：

- 身份：`model`、`symbol`、`timeframe`、`rolling_window`、`state_window`、`factor_signature`、`config_fingerprint`、`window_tail_ts`；
- 评估：`fitness`、`oos_ret`、`data_source`、OOS gate 结果和跨标结果摘要；
- 部署：`factor_expression`、`factor_mu`、`factor_sd`、`min_trade_zone`、`pine_code`、级联因子引用信息；
- 版本：`version`、`timestamp`、`created_at`。

部署字段以版本快照为事实来源。回退时不能从当前 best 继承目标版本缺失的评估证据；缺失字段明确呈现为 unknown，避免旧 best 的 `oos_ret`、symbol 或来源被错误带入。

新增 `ModelZoo.restore_deployment(name, version, *, reason, outcome)`，职责是：

1. 校验目标版本文件存在且可解析；
2. 读取完整 payload，保留 `_meta` 和所有部署字段；
3. 分别写入临时 best gzip 与 flat 文件；
4. 使用 `os.replace` 原子替换 best/flat；
5. 最后更新 `meta.json` 的 `best_version`、`best_fitness` 和部署记录；
6. 任一步失败时保留原 best 与锚点指针，并抛出可识别异常。

`save_agent_best` 改为复用该完整快照恢复逻辑，或仅作为内部兼容包装，不再自行用 `agent.to_dict()` 重建部署文件。版本裁剪继续保证 best 不悬空，并补充 `max_versions=1` 的不变量测试。

### 3.2 统一部署编排

在 `EvolveEngine` 中新增 `deploy_model`，所有接受、自动回退和手动回退均调用它。编排顺序：

1. 在 `_train_lock` 内确认目标快照和当前部署状态；
2. 保存部署前快照信息，用于失败恢复；
3. 调用 `ModelZoo.restore_deployment`；
4. 依据完整快照重新生成/注册 `rl_evolve` 或元控制器动态策略，确保 Pine、模型路径和元数据来自同一版本；
5. 调用可选的 `deployment_reload` 异步回调刷新当前交易引擎实例；
6. 校验运行时加载的模型路径/版本与部署快照一致；
7. 写入 outcome、通知和系统事件。

`EvolveEngine` 不直接依赖 `TradingEngine`，由启动流程注入回调。没有交易引擎的测试环境可以只验证文件和动态注册层，并将运行时状态标记为 `skipped`。

交易引擎新增带 `_strategy_lock` 的 `reload_active_strategy_model`：

- 当前实例为 `rl_adaptive` 时，清理 `_agent`、因子元数据缓存并按当前 `model_path` 立即加载新快照；
- 当前实例为 `meta_controller` 时，通过 `update_params` 触发 `_meta_agent` 失效并重新加载；
- 当前策略不是目标策略时只更新注册表，不切换用户当前策略；
- 加载失败返回失败，不伪报部署成功。

自动回退和手动 API 回退都必须经过同一回调。若运行时刷新失败，编排器尝试恢复部署前快照；若恢复仍失败，返回明确的 `runtime_reload_failed`，并保留磁盘层失败证据供人工处理。

### 3.3 锚点可比性

训练轮开始时生成 `TrainingIdentity`：

```text
(model, symbol, timeframe, rolling_window, state_window,
factor_signature, config_fingerprint)
```

`config_fingerprint` 只覆盖影响状态维度、奖励、数据切分和部署门的配置，采用稳定排序后的结构化 JSON 计算 SHA-256。`factor_signature` 覆盖因子表达式、级联因子标识和标准化口径。

回退比较规则：

- 锚点与候选身份完全一致：按现有 OOS 优先、fitness 兜底规则执行 `_should_rollback`；
- 任一身份字段不一致：不执行退化比较，记录 `anchor_mismatch`，候选通过自身 OOS/部署门后可建立新锚点；
- 旧版本缺少身份字段：视为 `unknown`，不得进行跨口径回退比较，但保留旧权重作为可运行部署；
- demo 与 exchange 继续不可比；不可比不等于自动删除旧权重。

因子挖掘在 OOS/因子安检不合格时只调用 `save_agent_archive`，不得写 factor best、不得更新级联因子基线；训练轮仍落库，区分“研究结果”与“可部署基线”。

### 3.4 状态与审计

三条管线状态统一增加：

- `candidate_fitness`：当前训练候选值；
- `deployed_fitness`：当前 best 快照实际值；
- `deployed_version`、`deployed_source`；
- `last_outcome`：`accepted`、`rollback`、`oos_rejected`、`cross_rejected`、`demo_blocked`、`manual_rollback`、`failed`；
- `anchor_mismatch`：最近一次身份不一致的原因；
- `runtime_reload`：`reloaded`、`skipped`、`failed`。

回退轮的 `fitness` 仍记录候选值，另以 `deployed_fitness` 表示回退后实际基线。手动回退新增 `EvolveRound.status="manual_rollback"`，记录目标版本、源版本、当前 symbol/timeframe 和原因，并发布系统事件与通知。

API `/api/evolve/versions` 返回每个版本的身份和部署元数据摘要；`/api/evolve/rollback` 返回磁盘恢复、动态注册、运行时刷新和最终部署版本状态。错误响应区分版本不存在、身份不匹配、快照损坏和运行时刷新失败。

## 4. 数据流与失败处理

### 接受新模型

候选训练 → 生成身份与 OOS 报告 → 身份一致性检查 → OOS/跨标门 → 写入完整版本快照 → 原子部署 → 动态注册 → 运行时 reload → 更新状态/事件。

### 自动回退

候选训练 → 用同身份锚点比较 → 命中回退阈值 → 目标 best 版本完整恢复 → 动态注册 → 运行时 reload → `last_outcome=rollback`、`deployed_fitness=锚点值`。

### 手动回退

API 校验模型名与版本 → 获取训练锁 → 调用统一 `deploy_model` → 失败时恢复部署前状态 → 写 `manual_rollback` 审计记录并返回结构化结果。

禁止出现以下半成功状态：

- best 文件已换版本但 flat 文件仍是旧版本；
- 文件已回退但动态注册仍引用旧 Pine/路径；
- 当前交易策略仍缓存旧 agent 却 API 返回成功；
- 回退后 `deployed_fitness` 与 best 快照不一致；
- 目标版本缺失字段时继承当前 best 的旧 OOS 证据。

## 5. 测试策略

新增或扩展以下测试：

1. `ModelZoo` 完整部署元数据 round-trip：factor expression、mu/sd、trade zone、Pine、级联字段均保留。
2. 原子恢复故障注入：best、flat、meta 任一步写失败时旧部署和锚点指针不变。
3. `max_versions=1`、连续保存、回退和裁剪不变量：best 版本始终存在于版本文件中。
4. 自动回退后动态注册和当前 `RLAdaptiveStrategy` 实例加载目标版本；模型维度/元数据不匹配时明确失败。
5. 手动 `/api/evolve/rollback` 与自动回退共用部署入口，并生成 `manual_rollback`/`rollback` 审计记录。
6. `meta_controller` 回退触发 `_meta_agent` 清理和重新加载。
7. symbol、timeframe、state window、factor signature 或 config fingerprint 不一致时跳过比较并暴露 `anchor_mismatch`。
8. 因子 OOS rejected 只存 archive，不改变 factor best 或级联文件。
9. 回退后状态中的 `candidate_fitness`、`deployed_fitness`、`deployed_version` 和 `last_outcome` 一致。
10. 现有全量回归：锚点、OOS gate、API guard、动态策略恢复、交易引擎策略锁测试保持通过。

## 6. 迁移与兼容

- 现有 `vN.json.gz`、`best.json.gz` 和 flat 文件格式继续可读。
- 旧版本缺新身份字段时不删除、不自动改写；首次重新接受合格模型时补齐身份元数据。
- 旧 best 若缺部署字段，运行时 reload 失败必须显式报告缺失字段，不静默生成不完整 flat 文件。
- 数据库不要求破坏性迁移；若新增审计字段，使用现有 SQLite 幂等 DDL/兼容读取策略，并为旧表提供默认值。
- 保留 `docs/REDEV_GUIDE.md` 中 D5、D6、D7、D9、D11 红线：OOS 门、回退落库、重启注册、flat 兼容和训练周期解耦不能回退。

## 7. 验收标准

完成后必须满足：

- 自动或手动回退成功后，磁盘 best、flat、动态注册和当前运行实例指向同一版本；
- 任意一项部署恢复失败都不会被 API 报告为成功；
- 不同品种/周期/状态维度/配置口径不会互相触发回退比较；
- OOS 不合格因子不会成为可部署基线；
- UI/API 能同时展示候选表现与实际部署表现；
- `.venv\Scripts\python -m compileall -q ...`、`.venv\Scripts\python -m pytest -q`、前端语法检查和必要冒烟全部通过。
