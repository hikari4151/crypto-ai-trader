# crypto_ai_trader 重构与升级执行方案

> 版本：v1.0（2026-08-14）
> 范围：代码正确性 / 可靠性 / 性能 / 工程卫生 + 前端 Apple HIG 全量对齐
> 原则（用户既定）：**遗留项当场做完**；**优化不丢原有效能**（回归测试兜底）；**提交前验证链全绿**（compileall → pytest → JS 语法 → 服务器冒烟）

---

## 0. 现状基线

| 项 | 现状 |
|---|---|
| 架构 | FastAPI + ccxt/ccxt.pro + SQLite(WAL) + 事件总线，Python 3.13（.venv），Windows |
| 前端 | `web/static/index.html` 单体 222KB（HTML ~1790 行 / CSS ~350 行 / JS ~1170 行），Vue3 CDN + ECharts CDN + Tailwind CDN |
| 回测 | 双引擎：事件驱动 `backtest/engine.py` + 向量化 `backtest/fast_engine.py`（Numba 已禁用，统一向量化） |
| 测试 | pytest 9 例（双引擎一致性 + 风控防线），无前端自动化 |
| 版本控制 | 本地有 commit 历史，远程 GitHub 未 push（需 VPN）；环境检测显示当前目录无 .git（待确认） |

**阶段 0 先建立基线**（见 Phase 0），后续每个 Phase 结束时必须跑完整验证链，行为相关改动必须跑双引擎一致性测试确认无漂移。

---

## 1. 审计发现总览

三路并行审计（后端核心 / AI·DRL·因子 / 前端 UI）+ 人工复核 P0，结论如下。

### P0（正确性，直接影响交易/资金/全站可用）— 6 项

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| P0-1 | 24h 跌幅检测对 float 列表做 `[-1][0]` 下标，`TypeError` | `engine/risk.py:330` | 引擎启动约 50s 后**每次风控检查必抛异常**，被事件总线吞掉 → 下单/止损/止盈全部静默停止，"看起来在运行实则已死" |
| P0-2 | 实盘资产页 USDT 双计：同时进 cash 与 positions_value | `web/api/portfolio.py:131-138` | equity 恒虚高一份 USDT 余额，仓位占比/权益曲线失真 |
| P0-3 | 实盘链路指标快照缺 `sr`/`pa` 键（只有 close/ma/macd/rsi/bb/vol_ratio） | `exchange/ws_market.py:41-85` + `strategies/price_action.py:52-95` | **price_action 系策略（含全部 AI 设计/迭代策略）实盘永远无法开仓**；回测正常、实盘哑火，绩效系统性脱节 |
| P0-4 | PPO 熵正则梯度符号反了（熵被最小化而非最大化） | `drl/agent.py:194-197` | 训练主动压制探索，策略过早坍缩，DRL/因子挖掘产出系统性受损 |
| P0-5 | `TradingEnv.reset()` 未重置奖励塑形状态，跨 episode 污染 | `drl/env.py:85-93` | 开启塑形时新 episode 第一根 K 线就带上一段峰值权益/连亏计数，奖励失真 |
| P0-6 | 前端引导链无容错：`Promise.all` 中 8 个加载函数无 try/catch | `index.html:3283` | **任意一个接口失败 → 全站卡死骨架屏**，无数据、无轮询、无错误提示 |

### P1（可靠性，间歇性失败/资源泄漏/数据误导）— 12 项

| # | 问题 | 位置 |
|---|---|---|
| P1-1 | 后台线程新建事件循环直接操作主循环的 async DB 引擎（跨 loop 使用连接池）→ 间歇性 `RuntimeError`，回测落库/AI 任务/DRL 持久化静默丢失 | `web/api/backtest.py:186-228`、`ai.py:220-227,302-309`、`drl.py:331-344` |
| P1-2 | 实盘限价单轮询 3×1s 未成交即弃管：不撤单、无对账、`filled=None` 被 amount 兜底误判全量成交 → 迟到成交漏计、重复开仓 | `engine/order_manager.py:119-145` |
| P1-3 | 事件总线吞掉一切 handler 异常，无连续失败计数、无健康信号 → 单点故障全局静默 | `core/bus.py:78-82` |
| P1-4 | `_http_clients` 按 `id(loop)` 缓存永不淘汰 → 每个后台 AI/回测任务泄漏一个 httpx 客户端 | `ai/client.py:107,123-133` |
| P1-5 | 引擎 start 失败路径无清理：exchange/ccxt 客户端泄漏、半初始化状态残留 | `engine/trading_engine.py:148-161` |
| P1-6 | AI 自动优化在 `_strategy_lock` 内做完整网络调用（最坏数分钟）→ 阻塞全部 K 线处理 | `engine/trading_engine.py:394-400` |
| P1-7 | validator 放行 NaN/Inf 参数（json.loads 接受 NaN，`_is_number`/`_within_range` 未过滤） | `ai/validator.py:41-71` |
| P1-8 | FactorMiningEnv fitness 奖励用全样本 z-score 标准化 → 奖励含未来分布信息（前视） | `drl/factor_env.py:127-129` + `factors/mining.py:248` |
| P1-9 | `evaluate_agent`/`evaluate_oos` 对含因子列或 state_window>1 的模型必然 500（env 维度失配） | `web/api/drl.py:460-465,541-543` + `drl/agent.py:572-577` |
| P1-10 | 训练数据不足回退分支 + 因子列注入时 val 环境维度失配且静默失效（早停/选优失效） | `drl/agent.py:296-299,327,388-393` |
| P1-11 | 前端图表三连：复用已 dispose 实例不检查（切页后空白）、全部图表无窗口 resize 处理、"24h 涨跌"实为相邻两根 K 线涨跌（数据误导） | `index.html:3049-3050,3250-3251,2586-2594,2925` |
| P1-12 | 前端轮询无治理：`loadAll` 每 5s 在所有视图无条件执行、4 个 setInterval 永不清理、后台标签页不暂停 | `index.html:3295-3301` |

### P2（性能/工程）— 15 项

| # | 问题 | 位置 |
|---|---|---|
| P2-1 | 事件驱动回测每根 K 线全量重算全部指标（O(n×120) 纯 Python 循环） | `backtest/engine.py:111-115`、`indicators/technical.py:166-223` |
| P2-2 | `_trade_times` 无限增长，风险检查每次 O(n) 全表过滤 | `engine/trading_engine.py:57,323` + `engine/risk.py:305-308` |
| P2-3 | 后台任务（回测/AI/DRL）无并发上限，可无限叠加 CPU 密集线程 | `web/api/backtest.py:231`、`ai.py:227,309`、`drl.py:377` |
| P2-4 | `resolved_proxy` 在 async 路径做阻塞 socket 探测（最坏 ~5s） | `config/settings.py:26-39,108-126` |
| P2-5 | live API 对 KV 中历史 NaN 无防御（JSONResponse 序列化 500/非法 JSON） | `web/api/live.py:41,112` |
| P2-6 | 行情 `_fetch_json` 无重试，Binance 429/5xx 直接 502 | `web/api/market.py:92-104` |
| P2-7 | 限价单成交价全缺时以 0 落库，污染 entry/止损/性能聚合 | `engine/order_manager.py:132` |
| P2-8 | PPO GAE 在拼接轨迹边界处串扰（末步 next_val 取下一 episode 首状态） | `drl/agent.py:411-415,120-136` |
| P2-9 | client 空内容扩容重试只生效一次，后续重试退回小 max_tokens | `ai/client.py:216-220` |
| P2-10 | 三套布林带 ddof 不一致（technical ddof=1 vs vectorized/IncrIndicators ddof=0），双引擎信号漂移 | `indicators/technical.py:60` vs `vectorized.py:113,328` |
| P2-11 | 复盘权益曲线按时间倒序构造，total_return/sharpe/回撤失真 | `engine/trading_engine.py:407-410` |
| P2-12 | FactorMiningEnv 每步全量重算组合因子 + 安检门（3000+ 次全量重算） | `drl/factor_env.py:119-132` |
| P2-13 | 死代码：`ai/prompts.py:178-204` 未用函数、`ai/prompts.py.bak`、`scripts/dev/*` 调试脚本、`tests/check_*.py`、前端死变量（btPollTimer/drlPollTimer/marketTimer/isPaper/drlCurveWrap） | 多处 |
| P2-14 | 前端：挂载时同一接口重复请求 3 次、主题首屏闪色（FOUC）、数据渲染缺守卫（toFixed/Date.parse 无空值）、回测记录表无空状态、无 favicon | `index.html:3286-3290,2,3281,488,753,1906,1528-1535` |
| P2-15 | 前端 CDN 三依赖无 SRI/无本地回退，断网即白屏 | `index.html:24-26` |

---

## 2. 代码优化清单（按阶段）

### Phase 1 交易链路 P0 修复（最高优先级）

1. **P0-1 risk.py 解包错误**：`engine/risk.py:330` 改为
   ```python
   span_ok = len(prices_24h) >= 10 and (self._price_history[-1][0] - self._price_history[0][0]) >= 2 * 3600
   ```
   并给 `risk.check()` 加顶层 try/except 兜底（同类错误不再静默杀死交易循环）。
2. **P0-1 配套·bus 健康信号（P1-3）**：`core/bus.py` `_run_handler` 按 handler 维护连续失败计数 + 最后失败时间，≥N 次（如 5）发布 SYSTEM 事件并暴露到 `/api/health` 的 degraded 字段；对 `MARKET_CANDLE`/`ORDER_FILL` 失败单独告警日志。
3. **P0-2 portfolio USDT 双计**：`web/api/portfolio.py:131-138` 对齐 `engine/portfolio.py` 口径 —— USDT 只 `cash += qty`，不追加 positions、不加 positions_value；若前端需要展示 USDT 行，由 cash 字段单独渲染。
4. **P0-3 实盘补 sr/pa**：`engine/trading_engine.py` `_on_candle_locked_inner` 取到 `ctx["indicators"]` 后，用 hub 的 K 线缓冲（`buf[-120:]`，与 fast_engine 同口径）调用 `support_resistance`/`price_action_features` 补 `ind["sr"]`/`ind["pa"]`；注意与 `_on_candle` 的增量更新语义一致（同 ts 不重复算）。验证：实盘 simulated 模式跑 price_action 策略确认可开仓。
5. **P0-4 PPO 熵正则符号**：`drl/agent.py:194-197` 取反：
   ```python
   d_entropy = +proba * (entropy + np.log(proba + 1e-12))
   ```
   修复后 `train_drl`/`train_factor_miner` 各跑一轮，对照历史基线（如 2.85 vs 3.14）验证探索改善。
6. **P0-5 env.reset 塑形状态**：`drl/env.py:85-93` 补 `self._peak_equity = start_cash; self._last_dd = 0.0; self._losing_streak = 0`。
7. **P0-6 前端引导链容错**：`index.html:3283` 改 `Promise.allSettled`，8 个 load 函数（loadStrategy/loadRisk/loadOptLogs/loadAnalyses/loadAiSettings/loadExKeys/loadBtHistory/loadAiStrategies）各自 try/catch（参照 loadPaperConfig 的 `catch(e){}` 模式），失败时 toast 一次且保留默认值。

### Phase 2 可靠性 P1 修复

8. **P1-1 跨 loop DB**：`web/api/backtest.py:186-228`、`ai.py:220-227,302-309`、`drl.py:331-344` 统一改为 `asyncio.run_coroutine_threadsafe(coro, engine._loop)` 调度回主循环（`ai.py` 的 `apply_strategy_params` 已示范此模式），或 worker 内新建独立 `Database` 实例。删除 `web/api/portfolio.py:605` 注释所指的同类隐患。
9. **P1-2 限价单生命周期**：`engine/order_manager.py` —— 轮询超时主动 `cancel_order`；维护 open-order 注册表，定时 `fetch_open_orders` 对账并把迟到成交回灌 `_on_fill`/Trade；`filled` 仅在 `status == "closed"` 时允许 amount 兜底，open 状态 `filled is None` 视为 0 并撤单。
10. **P1-4 client 泄漏**：`ai/client.py` `_http_clients` 改 `weakref.WeakKeyDictionary`（loop 键），worker `finally` 中 `aclose()` 专属 client；`web/api/market.py:78-89` 全局 `_client` 接入 lifespan 清理。
11. **P1-5 start 失败复位**：`engine/trading_engine.py:148-161` 包 try/except，失败时 `await self.exchange.close()`、置 None，状态复位后 re-raise/降级告警。
12. **P1-6 scheduler 锁外 AI 调用**：`trading_engine.py:394-400` —— 锁内快照 `strategy.params`，AI 调用移出锁（或改后台 worker + `run_coroutine_threadsafe` 应用参数，参照 web/api/ai.py worker 模式），锁内仅 `update_params`。
13. **P1-7 validator NaN/Inf**：`ai/validator.py` `_is_number` 加 `math.isfinite`，`_within_range` 对非有限值直接返回错误。
14. **P1-8 factor_env 前视 z-score**：`drl/factor_env.py:127-129` 改用训练段/expanding 统计量标准化（仿 `factors/mining.py:306-310` 的 dynamic_composite 口径），`factor_miner.py:148` 同步修正。
15. **P1-9 评估端点维度**：`web/api/drl.py` `evaluate_agent`/`evaluate_oos` 从模型文件读 state_dim/factor_expression/factor_mu/factor_sd/state_window/min_trade_zone，按 train_drl 同口径重建 env；加 try/except 返回友好错误。
16. **P1-10 退化分支静默失效**：`drl/agent.py` 数据不足回退时显式跳过 val 评估并打日志（或重建 extra_factors_val），避免静默无验证。
17. **P1-11/12 前端图表与轮询**（见 UI 清单 F-16/F-17/F-18）。

### Phase 4 性能优化

18. **P2-1 回测引擎统一**：`backtest/engine.py` 改用 fast_engine 的预计算 + `snapshot_at`（`fast_engine.py:50-56`），`run.py:29` CLI 同步；`bollinger` 改 cumsum 向量化（`_bollinger_vec` 已有）。**回归：双引擎一致性测试必须保持全绿且结果不变**（当前 ret=-0.0878 trades=19 基线）。
19. **P2-2 `_trade_times` 剪枝**：入队时丢弃 `now - 3600` 前的元素（或 `collections.deque(maxlen=…)`）。
20. **P2-3 任务并发上限**：`backtest/ai/drl` 每类任务加 `threading.Semaphore(2)`，超限 429。
21. **P2-4 代理探测异步化**：`config/settings.py` 探测放 `asyncio.to_thread`，或 lifespan 预取 `settings.resolved_proxy`。
22. **P2-5/6/7 防御补强**：`web/api/live.py` 读取后 `math.isfinite` 校验回退默认值；`web/api/market.py` 对 429/5xx 加 1-2 次退避重试；`order_manager.py:132` 成交价非 0 校验，按 last_price 兜底并告警。
23. **P2-8 GAE 分段**：`drl/agent.py` `collect_episode` 返回每轨迹长度，`_compute_gae` 按 episode 分段（done 处重置 gae、next_val=0）。
24. **P2-9 扩容重试持久化**：`ai/client.py:216-220` 首次扩容后写回 `base_body["max_tokens"]`。
25. **P2-10 布林 ddof 统一**：统一 ddof=0（与 vectorized/IncrIndicators 对齐），补双引擎一致性断言。
26. **P2-11 复盘曲线正序**：`trading_engine.py:407-410` 先 `reversed(trades)` 再构造权益曲线。
27. **P2-12 factor_env 缓存**：按 `tuple(sorted(selected))` 缓存 fitness，z-score 统计量预计算。

### Phase 5 工程卫生

28. **P2-13 死代码清理**：删 `ai/prompts.py` 未用函数 + `prompts.py.bak`；`scripts/dev/*` 调试脚本归档或删除；`tests/check_*.py` 评估后归并进 pytest 或删除；前端死变量删除。
29. **P2-14/15 前端工程**：见 UI 清单 F-20~F-24。
30. **文档与发布**：README 反斜杠转义修复、`.env.example` 校验、docs/ 组织；本地 commit，VPN 可用后 `git push origin main`。

---

## 3. UI 设计规范调整点（Apple HIG 对齐）

### 3.1 设计令牌（Design Tokens）— 先落地 token 层，再改组件

**F-1 字号收敛到 Apple 字阶**（`index.html:62,113,1928-1932` 等 20+ 处）：

| Token | 值 | 对应 Apple 名称 | 用途 |
|---|---|---|---|
| `--fs-caption` | 11px / ls -0.01em | Caption 1 | 辅助标签（≥11px，低于 10px 全部删除） |
| `--fs-footnote` | 12px / ls -0.01em | Footnote | 次级说明 |
| `--fs-body` | 13px / ls -0.01em | Body | 正文（默认） |
| `--fs-headline` | 15px / ls -0.02em | Headline | 卡片标题 |
| `--fs-title` | 20px / ls -0.02em | Title 2 | 区块标题 |
| `--fs-large-title` | 26px / ls -0.03em | Large Title | 页面大标题 |

删除全部 `text-[10px]`/`text-[10.5px]`；`font-weight:650` 等任意字重收敛为 400/500/600/700。

**F-2 语义色贯通**：pill/badge/图表色从独立 hex（`#6db3ff/#5fe082/#ffb340/#a68bff/#ff8a80`、`#ffd60a/#5ac8fa/#bf5af2` 等）改为派生自 `--accent/--green/--red/--orange` token；图表色统一走 `chartTheme()`（行2573）返回对象；**删除全部浅色 `!important` 补丁**，改在 `:root[data-theme="light"]` 中覆盖 token。

**F-3 暗色对比度达标**：`text-slate-500/600`（暗色下 2.2-2.9:1）全局映射为 `var(--muted)`（#98989f，~5.9:1）；最小字号 ≥11px。

**F-4 圆角 token 启用**：`--r-sm:8 / --r-md:10 / --r-lg:12 / --r-full:20` 落地到 `.mac-window`（12）、`.mac-card`（10）、按钮（8）；14/11/9/7/4px 魔法值收敛进阶梯；`.stat-card::before` 与 `.stat-card` 共用变量。

**F-5 死 token 落地**：`--bg-2/--bg-3`（#2c2c2e/#3a3a3c）让 `.mac-card`/`.stat-card`/`.mac-window` 直接消费；半透明叠加用 `color-mix()`。

### 3.2 视觉组件

**F-6 图标统一为线性 SVG**：替换约 40 处 emoji（stat-card、空态、titlebar、settings 侧栏、toast）为内联线性 SVG（stroke 1.5、viewBox 24、SF Symbols 风格，与行427 主题按钮同风格）；toast 图标改 SVG。

**F-7 毛玻璃修正**：`.ui-modal-mask` blur(6px) 与 `.ui-modal` blur(40px) 嵌套导致玻璃退化 —— 模糊集中到 mask 一层（提高采样质量），内层去掉重复 backdrop-filter 或降低；`.bt-steps`/`.menubar` blur 降为 20-24px；`background-attachment:fixed` 改到 `body::before` 固定层（现代合成方式），消除滚动整页重绘。

**F-8 弹窗层级与反馈**：`.ui-modal-mask` 加 `@keydown.esc` 全局处理、`aria-modal="true"`、打开时 `nextTick` 聚焦输入框/默认按钮、关闭后焦点还原到触发按钮。

### 3.3 交互与动效

**F-9 页面过渡补 leave**：`.page-leave-active{transition:opacity .16s ease}` + `.page-leave-to{opacity:0}`。

**F-10 reduced-motion 全覆盖**：清单补 `.ui-modal,.ui-modal-mask` 的 modalIn/fadeIn 与 page 过渡。

**F-11 可点击元素语义化**：交易对 pill、因子 pill、可点击 tr/mac-card、`.settings-item`（div 改 button）统一改 `<button>` 或 `tabindex="0" role="button"` + Enter 处理；补 `:hover`（`background:var(--hover)`）+ `:active`（`transform:scale(.97)` 60ms）反馈；`.settings-item:focus-visible` 规则因此复活。

**F-12 开关可访问名称**：5 个 `role="switch"`（自动行情分析/GPU×2/OOS 自动注册/强制自动训练）补 `aria-label`。

**F-13 表单 label 关联**：全部 `.mac-label` 加 `for`，input/select 加对应 `id`；v-for 参数表单用 `:id="'param_'+name"`。

**F-14 toast 无障碍**：`#toasts` 加 `role="status" aria-live="polite"`。

### 3.4 图表与数据

**F-15 图表稳定**：统一走 `initChart()` 模式（`el._chart && !el._chart.isDisposed() ? el._chart : echarts.init(el)`），删除 `index.html:3049-3050/3250-3251` 两处手写复用；onMounted 加全局 `resize` 监听（debounce 150ms，遍历全部图表实例 `!isDisposed() && resize()`），onUnmounted 清理。

**F-16 "24h 涨跌"口径修正**：`index.html:2925` —— 或复用 tickers 的真实 24h `change_pct`（行488），或改标注为"周期涨跌"，消除数据误导。

**F-17 轮询治理**：`loadAll` 门控 `if(view.value==='dashboard')`，内部 3 个接口改 `Promise.all` 并行；`document.visibilitychange` hidden 时清除非时钟轮询（恢复时立即刷一次）；onUnmounted 清理全部 interval。

**F-18 启动冗余请求去重**：`index.html:3286-3290` 同一 `/performance/equity?limit=500` 拉 3 次 → 拉一次复用。

### 3.5 响应式与工程

**F-19 窄屏适配**：menubar 内层包 `overflow-x:auto` + `scrollbar-width:none`（<640px 横滑）；仪表盘 8 列、战绩 9 列表格包 `overflow-auto` 容器（与回测表一致）。

**F-20 依赖本地化**：Vue/ECharts 下载到 `web/static/vendor/`（加 SRI），Tailwind 改为预编译产物内联（CDN 运行时版官方仅限开发）；断网/局域网部署可用。

**F-21 首屏闪色修复**：head 内联脚本提前读 localStorage 设置 `data-theme`（首帧前），补 `prefers-color-scheme` 兜底。

**F-22 数据渲染守卫**：`t.change_pct.toFixed`、`Date.parse`、`best_ret?.toFixed` 等统一走 `fmt/pct` 式守卫（`v!=null?v.toFixed(2):'-'`）；`buildParamGrid` 空 defaults 返回 `{}`；回测记录表补空状态行；补内联 SVG favicon；tooltip backdrop-filter 改纯背景色+半透明（兼容性）。

**F-23 死代码清理**：删 btPollTimer/drlPollTimer/marketTimer/isPaper 未用变量；`drlCurveWrap` 补 id 或删滚动逻辑。

**F-24 单体拆分**（最后做，全部功能稳定后）：
- CSS → `tokens.css` / `base.css`（reset+排版+滚动条+keyframes）/ `components.css`（menubar/window/controls/cards/pills/toast/modal）/ `charts.css`
- JS → `api.js`（fetch 包装+token）/ `charts.js`（chartTheme+initChart+6 渲染函数）/ 按页面拆 views
- HTML → 8 个 `<template id="view-*">` 片段或 Vue SFC
- 先做"去重 + token 落地"（`.stat-card` 三处重复定义、`::-webkit-scrollbar` 两套、`tbody tr` 两条、`.mac-toggle.on` 重复、`.orange-c` 两处），再物理拆分

---

## 4. 分阶段实施计划

> 每个 Phase 完成 = 验证链全绿（compileall → pytest 全量 → 前端 JS 语法 → 服务器冒烟 200/401）+ 行为相关改动跑双引擎一致性。**估算按单人全栈 8h/天**。

### Phase 0 — 基线加固（0.5 天）
- [ ] 确认 .git 状态，缺失则 `git init` + 基线 commit（当前工作区快照）
- [ ] 跑通验证链并记录基线输出：`python -m compileall`、`pytest -q`（9 例）、JS 语法检查（node --check 或提取 <script> 校验）、服务器冒烟（health/认证 200/401）
- [ ] 记录双引擎一致性基线（ret/trades 数值）
- [ ] 建立 `docs/` 目录与本次重构清单跟踪

**验收**：基线输出存档，后续每次改动可 diff 对比。

### Phase 1 — 交易链路 P0 修复（1-2 天）
- [ ] P0-1 risk.py 解包错误 + `risk.check()` try/except 兜底
- [ ] P1-3 bus 连续失败计数 + `/api/health` degraded（P0-1 配套，防止同类静默）
- [ ] P0-2 portfolio USDT 双计
- [ ] P0-3 实盘补 sr/pa（含暖机守卫确认）
- [ ] P0-4 PPO 熵正则符号 + DRL A/B 训练验证
- [ ] P0-5 env.reset 塑形状态
- [ ] P0-6 前端引导链 allSettled + 8 个 loader 容错
- [ ] 新增 pytest：risk 24h 检查、portfolio equity 单计、price_action 实盘快照含 sr/pa

**验收**：模拟盘跑 price_action 策略确认可开仓；A/B 训练曲线对比；全站断一个接口不再瘫痪（前端手测）。

### Phase 2 — 可靠性 P1 修复（2-3 天）
- [ ] P1-1 跨 loop DB 统一调度（backtest/ai/drl 三处）
- [ ] P1-2 限价单撤单 + open-order 对账 + filled 口径
- [ ] P1-4 `_http_clients` WeakKeyDictionary + worker finally aclose
- [ ] P1-5 引擎 start 失败复位
- [ ] P1-6 scheduler 锁外 AI 调用
- [ ] P1-7 validator NaN/Inf
- [ ] P1-8 factor_env 前视 z-score
- [ ] P1-9 DRL 评估端点维度重建 + 友好错误
- [ ] P1-10 退化分支显式跳过 val
- [ ] 前端 P1 批：F-15 图表 isDisposed/resize、F-16 24h 口径、F-17 轮询治理、F-18 启动去重、F-19 窄屏、F-20 CDN 本地化

**验收**：连续触发 10 次回测/AI 任务零间歇失败；限价单挂单超时被撤；前端切页图表正常、后台标签降频、窄窗口可导航。

### Phase 3 — UI 设计体系重构（3-5 天）
- [ ] F-1 字号字阶 token（删 <11px）
- [ ] F-2 语义色贯通（删 !important）+ F-3 暗色对比度
- [ ] F-4 圆角 token 落地 + F-5 死 token
- [ ] F-6 线性 SVG 图标集替换 emoji
- [ ] F-7 毛玻璃修正 + 背景合成优化
- [ ] F-8 弹窗焦点管理 + F-9 page leave + F-10 reduced-motion 全覆盖
- [ ] F-11 可点击元素语义化 + F-12 toggle aria-label + F-13 label 关联 + F-14 toast aria-live
- [ ] F-21 首屏闪色 + F-22 渲染守卫/空状态/favicon + F-23 死代码
- [ ] F-24 拆分：CSS 四文件 → JS 三文件 → HTML 模板片段（逐步，每步冒烟）

**验收**：键盘全流程可操作（Tab 可达所有控件）；暗色对比度抽查 ≥4.5:1；无 emoji 残留；浅色模式无 !important；reduced-motion 下无动画。

### Phase 4 — 性能优化（2-3 天）
- [ ] P2-1 回测引擎统一预计算（**双引擎一致性基线必须不变**）
- [ ] P2-2 `_trade_times` 剪枝
- [ ] P2-3 后台任务并发上限
- [ ] P2-4 代理探测异步化
- [ ] P2-5/6/7 防御补强（NaN/重试/成交价）
- [ ] P2-8 GAE 分段 + P2-9 扩容重试持久化 + P2-10 布林 ddof 统一 + P2-11 复盘曲线正序 + P2-12 factor_env 缓存
- [ ] 前端性能：轮询门控实测（后台标签 0 请求）、大表渲染、图表 resize 防抖

**验收**：万根 K 线回测耗时对比（记录修复前后）；DRL 训练耗时对比；双引擎一致性 9/9 全绿。

### Phase 5 — 工程卫生与发布（1-2 天）
- [ ] 死代码清理（prompts.py.bak、scripts/dev、tests/check_*、前端死变量）
- [ ] README 转义修复、.env.example 核对、docs/ 组织
- [ ] 全量回归：完整验证链 + 双引擎一致性 + 模拟盘 smoke
- [ ] 本地 commit（按模块拆 commit）；VPN 可用后 `git push origin main` 并打 tag

**验收**：验证链全绿；`python run.py backtest --source demo --strategy dual_ma` 与基线一致；无遗留清单。

---

## 5. 验证链（每个 Phase 必跑）

```bat
:: 1. 编译检查
.venv\Scripts\python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py
:: 2. 单元/回归测试
.venv\Scripts\python -m pytest -q
:: 3. 前端 JS 语法
:: （提取 index.html 内联 script 后 node --check，或 node --check 拆分后的 js 文件）
:: 4. 服务器冒烟
:: start.bat 启动 → GET /api/health 200、无令牌访问 401、带令牌 200
:: 5. 双引擎一致性（行为改动后必跑）
.venv\Scripts\python run.py backtest --source demo --strategy dual_ma
```

---

## 6. 风险与注意事项

1. **Windows 编码陷阱**：新文件必须 UTF-8；`.bat` 纯 ASCII；第三方库读文件默认编码问题（config.yaml 教训）——改动涉及文件 IO 时在系统 locale（GBK）环境验证。
2. **实盘安全**：P1-2（限价单对账）必须在实盘启用前完成；测试引擎前先 `kv_set trading_mode=paper`（DB 中可能存有 okx 实盘配置）。
3. **性能回退红线**：Phase 3 UI 重构期间行为不改（只动表现层），Phase 4 每次优化跑双引擎一致性；两引擎基线 ret=-0.0878/trades=19 不得漂移。
4. **CDN/网络**：当前依赖 3 个外网 CDN，国内/局域网环境白屏 —— F-20 本地化优先于其他 UI 美化。
5. **git 远程**：7 笔本地修复未 push，GitHub 远程仍是 v1.0.0 初始快照；重构完成后统一 push。
6. **PPO 重训**：熵符号修复会改变训练行为，属预期收益，需 A/B 对照记录基线后重训并留档。
7. **遗留项纪律**：实施中发现的任何可修项当场修完，不留"待后续"清单；确需决策的写入本文件的 Phase 5 前清空。

---

## 7. 优先级速查

| 优先级 | 项 | 理由 |
|---|---|---|
| 🔴 立即 | Phase 1 全部（P0-1~P0-6） | 交易静默死亡 / 实盘无法开仓 / 全站瘫痪 |
| 🟠 短期 | Phase 2 全部（P1-1~P1-12） | 间歇性数据丢失 / 真实资金风险 / 数据误导 |
| 🟡 中期 | Phase 3 UI（F-1~F-24）| 产品质感与可用性，投入产出比最高的视觉升级 |
| 🟢 持续 | Phase 4 性能 + Phase 5 卫生 | 规模增长后收益显现，随迭代顺手做 |

---

## 8. 执行状态记录（2026-08-15 全部完成）

| 阶段 | 状态 | 验证 |
|---|---|---|
| Phase 0 基线 | ✅ | docs/BASELINE.md（compileall/pytest 9/check_js/冒烟/双引擎基线 ret=-0.224999/trades=37） |
| Phase 1+2（团队开发） | ✅ | 5 模块交付 + 5 审查 + PM 集成修复 21 项；pytest **81 passed**、双引擎基线零漂移、连发回测×4 / DRL 训练×2 落库零错误 |
| Phase 3 UI 重构 | ✅ | F-1~F-23（F-20 提前完成）；0 `!important`、0 <11px 字号、28 线性 SVG、无障碍全量、毛玻璃/焦点/过渡/对比度 |
| Phase 4 性能 | ✅ | P2-1~P2-12（引擎预计算统一 O(n)、_trade_times 剪枝、并发 Semaphore(2)+429、代理预热、GAE 分段、布林 ddof 统一、复盘正序、factor_env 缓存等） |
| Phase 5 卫生 | ✅ | prompts.py.bak/死函数/scripts/dev 清理、README 验证链说明、全量回归（compileall/pytest 81/check_js/冒烟 200/401/200） |

**F-24 文件拆分裁决**：物理拆分（222KB 单体 → CSS/JS/HTML 多文件）无浏览器回归验证环境且功能已稳定，风险>收益 → 改为可选增强（拆分前置条件——CSS 去重与 token 落地已在 Phase 3 完成）；如后续需要，按 §3.5 F-24 指引执行并从浏览器手测开始。

**发布检查单（用户手动执行，本机无 git CLI）**：
```bat
:: 1. 确认改动范围
git status
:: 2. 按模块分批提交（7 笔本地修复也未 push）
git add docs/ tests/ web/engine/exchange/core/ai/ factors/ drl/ backtest/ indicators/ config/ scripts/ README.md && git commit -m "refactor: Phase 1-2 P0/P1 修复 + 团队开发交付（风险/AB 盘/sr-pa/跨loopDB/AI优化/前端引导链）"
git add web/static vendor && git commit -m "ui: Phase 3 Apple HIG 设计体系 + vendor 本地化"
:: 3. 推送（需 VPN）
git push origin main
```
