# 进化因子使用场景：自动部署为组合因子策略（设计文档）

日期：2026-09-14
状态：已确认（用户审批通过）

## 1. 背景与问题

持续进化引擎（`drl/evolve_engine.py`）的 factor_miner 管线训练通过 OOS 安检后，
当前产物只有三样有明确去向：

1. RL 智能体 → `models/zoo`，供下一轮续训与版本管理；
2. 组合因子序列 `_cascade_factor_values_<symbol>.npy` → 仅供 strategy_drl 级联注入
   （内部训练依赖，不是使用场景）；
3. 组合权重 `_cascade_weights_<symbol>.json` → 代码注释写着"供部署端使用"，
   但**全工程没有任何读取方**——进化产出的因子没有落地为可用策略。

与此同时，运行时重建能力已经就绪：`strategies/factor_signal.py` 支持
`factor="combo" + combo_spec`（JSON：{因子key: 权重}），在策略自维护的滚动
OHLCV 缓冲上用 `factors.library.get_factor` 重建各因子、缓冲内 z-score 后按权重
加权求和，含因子下线隔离（`factor_is_live` 检查，下线因子自动跳过）。训练用的
因子矩阵列（`factors/engine.compute_factor_matrix` 用 `get_factor` 计算）与策略端
重建用的因子池同源，链路数据基础一致。

手动 RL 挖掘路径（`/api/drl/mine-factor`）在 UI 里已有 `applyRlCombo()` 一键
"应用为 factor_signal 组合策略"。**持续进化管线缺的正是同款闭环的自动化版。**

## 2. 目标

持续进化引擎 factor_miner 训练产物（组合因子权重 + OOS 报告）在 OOS 安检通过后
自动按标的部署为 factor_signal 组合因子策略（可回测、可实盘手动启用），并在进化
面板展示部署状态与回测摘要。

## 3. 设计决策（用户已确认）

1. **部署粒度**：按标的各建一个策略 `evolve_combo_<SYMBOL>`（如
   `evolve_combo_BTC_USDT` / `evolve_combo_ETH_USDT` / `evolve_combo_SOL_USDT`），
   与既有"品种周期独立训练"设计一致。
2. **触发方式**：训练通过 OOS 安检后**自动部署**（注册/更新对应标的策略）；
   进化面板保留**手动重新部署**按钮。
3. **部署后动作**：注册为可用策略 + 后台计算回测摘要进面板展示；**实盘由用户在
   实盘页手动启用**，不自动切换当前实盘策略。

## 4. 组件设计

### 4.1 部署方法 `EvolveEngine._deploy_factor_strategy(symbol, weights, report, meta)`

- 策略名：`evolve_combo_<symbol>`，其中 symbol 的 `/` 替换为 `_`（与既有
  `_cascade_weights_<symbol>.json` 的平铺命名一致）。
- spec 结构（与 `_register_rl_evolve` 同款规格，落 `AiStrategy` 表，
  重启后由 `strategies/dynamic_store.restore_from_db` 恢复）：

```python
spec = {
    "name": name,
    "title": f"进化组合因子·{symbol}",
    "description": f"持续进化引擎因子挖掘（factor_miner）产出的组合因子策略（{symbol}）",
    "logic": "RL因子挖掘选出的因子组合按IC方向加权合成，因子值z-score后与阈值比较产生买卖信号",
    "executor": "factor_signal",
    "param_schema": FactorSignalStrategy.param_schema,
    "params": {
        "factor": "combo",
        "combo_spec": json.dumps(weights, ensure_ascii=False),
        "mode": "trend",
        "buy_threshold": 0.0,
        "sell_threshold": 0.0,
        # 其余沿用 FactorSignalStrategy.default_params（stop_loss_pct 等）
    },
    "risk_tips": ["因子IC可能衰变，因子库下线后该因子自动跳过"],
    "created_by": "evolve_engine",
    "version": f"v{deploy_count}",
    "base_symbol": symbol,
    "base_timeframe": self._effective_timeframe(),
    "evolve_meta": {
        "fitness": meta.get("fitness"),
        "oos_report": report,          # factor_quality_gate 输出（valid/rank_ic/icir/turnover）
        "selected_factors": meta.get("selected_factors", []),
        "deployed_at": time.time(),
        "round_no": meta.get("round_no"),
        "backtest": None,              # 后台回测摘要回填
    },
}
```

- 执行：`register_dynamic(name, spec)` + `_upsert_ai_strategy(name, spec)`。
- **幂等覆盖**：同标的再次部署 = 覆盖更新同名 spec（version 递增）。
- 返回部署后的策略名与版本。

### 4.2 自动部署钩子（`_train_factor_miner_once`）

在 OOS 安检通过、保存 best 模型与 cascade 文件之后调用部署方法：

```python
if report valid:
    deploy_result = await self._deploy_factor_strategy(symbol, weights, report, meta)
    self._factor_miner_status["deployed_strategy"] = deploy_result["name"]
    self._factor_miner_status["deployed_version"] = deploy_result["version"]
    self._factor_miner_status["last_deploy_ts"] = time.time()
    self._factor_miner_status["deploy_error"] = ""
    # 触发后台回测摘要（fire-and-forget，不阻塞训练循环）
    asyncio.create_task(self._refresh_factor_strategy_backtest(deploy_result["name"], symbol, df))
```

部署失败（权重缺失等异常）只记 `deploy_error`，**不阻断训练流程**（部署是训练后
的附加动作，训练主体照常完成）。

### 4.3 后台回测摘要 `_refresh_factor_strategy_backtest(name, symbol, df)`

- 用本轮训练同一份 df（rolling_window 根 K 线）+ `BacktestConfig`：

```python
cfg = BacktestConfig(
    symbol=symbol,
    timeframe=self._effective_timeframe(),
    strategy_name=name,
    strategy_params=spec["params"],   # factor=combo + combo_spec
    start_cash=10000.0, fee_rate=0.001, slippage=0.0005,
)
```

- `await asyncio.to_thread(run_backtest, df, cfg)`（CPU 密集，放线程池）。
- 摘要字段：总收益、最大回撤、夏普、交易次数、基准收益。
- 回填 `_factor_miner_status["backtest_summary"]`；spec 的
  `evolve_meta["backtest"]` 仅内存更新、不重新落库（重启后回测摘要为空，
  由用户手动「去回测」或下一次自动部署重新生成——避免每次部署都多一次
  AiStrategy 表写放大）。
- 失败只记 `deploy_error`，不影响部署状态。

### 4.4 API（`web/api/evolve.py` 新增）

- `GET /api/evolve/factor-strategies`：列出全部已部署进化组合因子策略
  （策略名/标的/权重/OOS 报告/版本/回测摘要/部署时间/上线状态）。实现：从
  `strategies.get_dynamic` 过滤 `created_by == "evolve_engine"` 且
  executor 为 factor_signal 且 name 前缀 `evolve_combo_`，必要时并入
  `_factor_miner_status` 的最新部署信息。
- `POST /api/evolve/factor/{symbol}/deploy`：手动重新部署。读取
  `models_dir/_cascade_weights_<symbol_flat>.json`，构造 meta（fitness 取
  best 信息或状态值、selected_factors 取最近轮次），调用部署方法；
  权重文件不存在返回 404。

### 4.5 UI（`web/static/index.html` 进化面板）

factor_miner 模型卡片增加部署状态区：

- 已部署策略名（`evolveModel('factor_miner').deployed_strategy`，带版本与部署时间）；
- 回测摘要（总收益/最大回撤/夏普，`backtest_summary`）；
- 「部署为因子策略」按钮（手动重新部署，调
  `POST /api/evolve/factor/{symbol}/deploy`，symbol 用 status 里的当前标的）；
- 「去回测」按钮（跳转回测页并预填该策略名）。
- `deploy_error` 非空时以错误样式展示。

## 5. 错误处理与兼容性

- 权重文件缺失 / 组合因子全部下线 → `deploy_error` 记录，不阻断训练；
- 已下线因子由 `factor_signal` 端跳过（既有机制），部署时保留权重原文；
- 与既有 `applyRlCombo`（手动 RL 挖掘路径）并存，命名空间不同（`evolve_combo_*`
  与 `factor_signal` 内置策略名），互不覆盖；
- 启动恢复：`_restore_evolve_strategies` 只恢复 rl_evolve，本设计的部署策略
  由 `dynamic_store.restore_from_db` 通用恢复路径兜底（AiStrategy 表），无需新增
  启动钩子。

## 6. 测试

- 单测（`tests/`）：
  - spec 构造：命名规则、combo_spec 序列化、幂等覆盖、版本递增；
  - `_deploy_factor_strategy` 注册后 `get_strategy(name)` 可实例化且
    `combo_spec` 参数正确。
- 集成：
  - 模拟 factor_miner 安检通过 → 自动部署注册 + 落库 + status 更新
    （参考 `tests/test_evolve_oos_gate.py` 的构造方式）；
  - combo_spec 可被 factor_signal 端重建（复用
    `tests/test_factor_composite_runtime.py` 思路）。
- API 测试：`factor-strategies` 列表、手动 deploy、权重文件缺失 404 分支。

## 7. 范围外（YAGNI）

- 组合因子参数自动优化（buy/sell 阈值网格搜索）——部署用默认阈值，用户手动调；
- 自动切换实盘策略；
- 自动回退已部署策略（沿用现有部署/回退机制的手动入口）；
- 回测结果落 `backtest_results` 表（摘要只进 status / spec.evolve_meta）。
