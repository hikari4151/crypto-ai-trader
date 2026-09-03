# crypto_ai_trader 再开发指南（防反向升级）

> 版本：v1.0（2026-09-01 实测锚定）
> 读者：未来在这个仓库上再开发的任何人 / AI 助手。**开工前必读，改完对照 §4 验证链。**
> 目的：这个项目经历过多轮审计-修复-重构（见 §8 文档地图），大量"现在的写法"是踩坑后有意为之。
> 最危险的失败模式不是写出新 bug，而是把旧的修复"优化"回去（=反向升级）。本文档就是防这个的。

---

## 0. 快速事实卡

| 项 | 值 |
|---|---|
| 技术栈 | Python 3.13（.venv）+ FastAPI + ccxt/ccxt.pro + SQLite(WAL) + 事件总线 |
| 平台 | Windows（中文 locale=GBK），前端为单体 `web/static/index.html`（Vue3 CDN 版本地化，5674 行） |
| 入口 | `python run.py web`（子命令：web / backtest / cost-scan / portfolio / backfill） |
| 认证 | 所有 `/api/*`（除 `/api/health`）要求 `X-API-Token` 头；令牌经 Cookie `zx_api_token` 下发；**无 CORS 中间件、默认绑 127.0.0.1** |
| 数据库 | 主库 `data/trader.db`（aiosqlite）；K 线库 `data/klines.db`（独立 SQLite，复合主键 WITHOUT ROWID） |
| 令牌持久化 | `data/api_token.txt`（未配置 API_TOKEN 时自动生成并复用，重启不掉 401） |
| 代理 | `PROXY_URL` 或自动探测 Clash/V2Ray（prewarm 异步预热）；REST 走 `aiohttp_proxy`，WebSocket 走 `wsProxy` |
| 双回测引擎 | 事件驱动 `backtest/engine.py` + 向量化 `backtest/fast_engine.py`，共用撮合内核 `backtest/_matching.py` |

## 1. 当前基线（2026-09-01 实测，全部当场跑过）

| 验证项 | 命令 | 当前锚定值 |
|---|---|---|
| 编译检查 | `.venv\Scripts\python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py` | ✅ |
| 测试 | `.venv\Scripts\python -m pytest -q` | ✅ **434 passed**（数量随测试增长，以"全绿"为准） |
| 前端 JS 语法 | `.venv\Scripts\python scripts\check_js.py` | ✅（esprima，本机无 node） |
| 双引擎一致性 | `.venv\Scripts\python run.py backtest --source demo --strategy dual_ma` | **ret=-0.206239 / trades=37 / final_equity=7937.6077** |

**基线变更纪律**：
- 历史基线是 ret=-0.224999/trades=37（2026-08-15），2026-08-15 之后基准对比、成本模型（participation-rate/funding-rate/限价单成交模型）等**有意变更**使 ret 漂移到 -0.206239，trades=37 未变。本值即为新锚点。
- 以后跑出与 §1 不同的数值：**先假设自己改坏了**，查清原因；确认是有意变更后，更新本表并在下方记录（日期 / 新值 / 变更原因）。

| 日期 | ret / trades | 原因 |
|---|---|---|
| 2026-08-15 | -0.224999 / 37 | 成本模型增强前锚点（见 docs/BASELINE.md） |
| 2026-09-01 | -0.206239 / 37 | 成本模型/基准对比等有意变更后的新锚点（本文档建立） |

⚠️ **git 状态警告（2026-09-01）**：工作区约 145 个文件未提交（持续进化引擎、模型动物园、横截面因子、保护止损、通知、K 线本地库等大量功能只在未提交代码里），git 历史最新提交为 `4ec6ad6`（2026-08-28）。**再开发第一步：先把当前工作区整体 commit 作为基线快照**，否则任何回退/对比都失去参照。

## 2. 模块地图（当前真实状态）

### 启动流程（改 lifespan 前必读）
`run.py web` → uvicorn `web/main.py` **lifespan**，顺序敏感：
1. `setup_logging()` → `Database` + `db.init()`
2. **必须先** `strategies.dynamic_store.restore_from_db(db)`（AI 动态策略），**再**建 `TradingEngine` —— 因为引擎 `__init__` 里 `get_strategy(default_strategy)` 可能解析到动态策略，顺序反了启动即崩
3. `engine.evolve.start()`（进化引擎独立于交易引擎启动，失败仅告警不阻断）
4. `settings.prewarm_proxy()`；若 `trading_auto_start` 则 `engine.start()`
关闭序：engine.stop → ai_client.close → market.close_client → db.close。

### 目录职责

| 目录 | 职责与关键文件 |
|---|---|
| `core/` | `bus.py` 事件总线（带 handler 连续失败计数 → `/api/health` degraded）、`database.py`、`events.py`（事件名：market.ticker/candle、strategy.signal、order.fill、trade、risk.blocked、ai.analysis/optimized/review、portfolio.update、engine.state、system）、`notify.py`（TG/钉钉/邮件，fire-and-forget 永不抛错，60s 防抖） |
| `exchange/` | `manager.py` ccxt REST、`ws_market.py` ccxt.pro 行情+增量指标（按 ts 去重）、`paper.py` 纸面账户（FIFO）、`symbols.py` **统一 symbol 解析**（`parse_symbol`，禁 `split("/")`） |
| `indicators/` | `technical.py`（增量/单点）与 `vectorized.py`（批量/回测）——同一指标两实现必须同口径（ddof、暖机语义） |
| `strategies/` | 6 内置：dual_ma / grid / price_action / rl_adaptive / factor_signal / meta_controller + `_DYNAMIC` 动态策略表（AI 设计/迭代/DRL/仓库）；动态优先于内置 |
| `factors/` | `library.py` 因子库、`mining.py`（AI 挖掘 + dynamic_composite 滚动 IC 合成）、`analysis.py` 安检门（IC/ICIR/换手）、`model_factor.py`（监督 MLP + walk-forward）、`cross_section.py`（横截面因子，品种<5 出 warning） |
| `ai/` | `client.py`（重试/限速/JSON 稳健解析/扩容封顶 8000）、`prompts.py`（动态 schema 注入）、`schemas.py`（JSON Schema，`additionalProperties:false`）、`validator.py`（NaN/Inf/bool 拦截）、`market_analyst.py`、`optimizer.py`、`strategy_designer.py`、`iteration.py`、`evals.py`（提示词 golden set 评测） |
| `drl/` | `agent.py`（PPO：熵正则符号/GAE 分段）、`env.py`、`factor_env.py`（无前视 z-score + fitness 缓存）、`factor_miner.py`、`model_zoo.py`（版本化模型仓库，≤10 版，flat 兼容）、`evolve_engine.py`（三条训练管线：factor_miner/strategy_drl/meta_controller + 回退/OOS 门/跨标门/增量门）、`pine_export.py`（Actor→Pine v5） |
| `backtest/` | `_matching.py` **撮合单一事实来源**、`engine.py` 事件驱动、`fast_engine.py` 向量化、`metrics.py`（含 benchmark 键）、`overfit.py`（walk-forward + CSCV PBO）、`cost_scan.py`、`portfolio.py`（组合回测，研究层）、`binance_history.py`（data.binance.vision 月度 zip）、`kline_store.py`（K 线本地库） |
| `engine/` | `trading_engine.py`（调度中枢，三把锁）、`risk.py`（检查带顶层兜底 + cooldown/人工解除）、`order_manager.py`（限价单生命周期/对账）、`portfolio.py`、`protective_stop.py`（服务端兜底止损） |
| `web/api/` | 15 个路由：ai / backtest / data（K 线库+下载）/ drl / **evolve** / exchanges / factors / live / market / notify / performance / portfolio / risk / strategy_repo / trading + `tasks_cache.py`（进度缓存 TTL 清理） |
| `web/static/` | `index.html` 单体 + `vendor/`（vue.global.prod.js / tailwind.js / echarts.min.js，带 SRI）。**无拆分 js/css** |

### 引擎内部结构（trading_engine）
`__init__` 创建 OrderManager / PortfolioManager / RiskManager / AIClient / MarketAnalyst / ParamOptimizer / TradeReviewer / StrategyDesigner / StrategyIteration / EvolveEngine；三把锁 `_lifecycle_lock` / `_candle_lock`（K 线串行防重复下单）/ `_strategy_lock`（热切换）。订阅 MARKET_CANDLE / MARKET_TICKER / ORDER_FILL。

## 3. 禁止回退清单（核心）

> 每条 = 当年真实踩过的坑。改动涉及这些区域时，先读"为什么"，改完跑"怎么验"。
> 拿不准 = 保持现状。想改设计 = 先在本文档追加决策记录再动手。

### A. 回测与指标（双引擎一致性红线）

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| A1 | 信号当根 K 线成交 | `pending_signal` **次根开盘成交**（`_matching.py`） | 当根成交=前视偏差，回测虚高 | `test_backtest_engine_consistency.py` + `test_lookahead_fix.py` |
| A2 | 两引擎各写一套撮合 | 撮合逻辑统一在 `backtest/_matching.py`，两引擎只做数据准备 | 各写一套必然漂移（历史上漂移过） | 一致性测试全绿 |
| A3 | 单点 `entry_price` 记账 | FIFO 逐笔成本阵列（分批建仓/部分平仓 pnl 准确） | 单点记账使 pnl 失真、污染日盈亏 | `test_paper_fifo.py`（32 例） |
| A4 | 布林带 ddof=1 | 全项目统一 **ddof=0** | ddof 不一致曾致两引擎信号漂移 | 一致性测试 |
| A5 | 重新引入 Numba/JIT | Numba 核心与回退函数**已删除**，统一标准向量化 | 静默回退路径曾致崩溃且结果不可复现 | 回测能跑 + 基线一致 |
| A6 | vol_ratio 暖机爆炸值 | 前 4 根置 1.0；策略侧 `candles_count<30` 暖机守卫 | 曾产生 1e14 级假信号 | `test_factor_m3.py` |
| A7 | 复盘权益曲线倒序构造 | 按时间正序构造 | 倒序使 return/sharpe/回撤全错 | `test_optimizations_p0p1p2.py` |

### B. 交易与实盘链路（资金安全红线）

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| B1 | 无认证 / CORS 通配 / 0.0.0.0 | `AuthMiddleware`（X-API-Token + Cookie，hmac.compare_digest），默认 127.0.0.1，**无 CORS** | 曾裸奔；恢复=任何人可操控资金 | 冒烟：无令牌 401 |
| B2 | 删路径守卫 | `web/deps.py resolve_data_path` + drl 模型名白名单 `_model_path` | 曾有任意文件读取/路径遍历 | `test_*` 安全类 |
| B3 | 后台线程直连主 loop 的 async DB | 跨线程 DB 一律 `run_coroutine_threadsafe(engine._loop)` 或 worker 内新建独立 `Database` | 跨 loop 用连接池=间歇性 RuntimeError、落库静默丢失 | 连发回测×4 全落库；`grep run_coroutine_threadsafe web/api/` |
| B4 | 删实盘 sr/pa 补算 | `trading_engine._on_candle_locked_inner` 用 hub 缓冲补 `ind["sr"]`/`ind["pa"]` | 缺了 price_action 系（含全部 AI 设计/迭代策略）**实盘永不开仓**，回测正常实盘哑火 | 模拟盘跑 price_action 能开仓 |
| B5 | 吞掉 risk.check 异常 | `risk.check()` 顶层 try/except 兜底 + bus 连续失败计数 → `/api/health` 的 `degraded` | 静默异常曾使下单/止损全停、"看起来在跑实则已死" | `test_risk_guards.py`；health 含 degraded/bus 键 |
| B6 | 限价单"轮询超时即弃管" | 超时撤单 + open-order 对账 + 迟到成交回灌 + `filled` 口径（open+None=0） | 弃管=漏计成交、重复开仓 | `test_reconcile.py`、`test_limit_order.py` |
| B7 | USDT 双计 | USDT 只进 cash，不进 positions_value（web/api/portfolio 与 engine/portfolio 同口径） | 双计使 equity 恒虚高一份 | `test_optimizations_p0p1p2.py` |
| B8 | `symbol.split("/")` | 统一走 `exchange/symbols.py`（`parse_symbol`） | split 会拆坏合约符号 `BTC/USDT:USDT` | grep 全项目 `split("/")` |
| B9 | 保护止损语义 | `protective_stop.py` 三条硬约束：仅 live 生效 / 主动卖出前必撤（`release`）/ 失败只降级告警+SYSTEM 事件 | 破坏任何一条=交易所挂孤单或失去兜底 | `test_protective_stop.py` |
| B10 | 引擎 start 失败留半初始化 | 失败路径 `exchange.close()` + 状态复位 + 降级告警 | 半初始化残留=后续行为不可预测 | `test_engine_*` |
| B11 | scheduler 在 `_strategy_lock` 内做网络调用 | 锁内只快照参数，AI 调用在锁外（worker + threadsafe 应用） | 锁内网络调用曾阻塞全部 K 线处理 | 代码审查 trading_engine.py |

### C. AI 链路

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| C1 | optimizer 返回旧参数快照 | `optimize_*(apply=False)` 返回 **AI 建议的新参数** | 返回旧快照=AI 自动优化从未生效（曾长期如此无人发现） | `test_ai_guards.py`；优化日志看参数确实变化 |
| C2 | validator 放行 NaN/Inf/bool | `_is_number` 带 `math.isfinite`，bool 显式拦截（json.loads 接受 NaN！） | AI 返回 NaN 参数会直接进交易链路 | `test_validator_nan.py` |
| C3 | http client 按 `id(loop)` 永久缓存 | `WeakKeyDictionary` + worker `finally` 里 `aclose()` | 每个后台任务泄漏一个客户端 | 长跑内存曲线 |
| C4 | 删 AI 参数验证门 | `ai_optimize_validate=true`：AI 参数先回测+过拟合校验（walk-forward/PBO），不过则不应用 | 没有门=AI 可以把策略改成过拟合参数直接实盘 | `/api/ai/auto-optimize` 日志有 validate 环节 |
| C5 | max_tokens 扩容不封顶 | 扩容封顶 `_MAX_TOKEN_CAP=8000` 且首次扩容后写回 base_body | DeepSeek 上限 8192，曾扩容重试自毁 | `test_ai_guards.py` |
| C6 | schema 随意放开 | `ai/schemas.py` 全部 `additionalProperties:false`；`ai_use_json_schema` 默认关 + 400 自动回退 | 放开=AI 可注入未定义字段；默认关是因部分模型不支持 | `test_prompt_evals.py` |
| C7 | market_analyst 持久化缺 ts/source | 补程序时间戳 `ts`（短时反手防线依赖）+ `source=auto` 标记 | 缺 ts 曾使反手防线失效 | `test_ai_guards.py` |

### D. DRL 与持续进化引擎

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| D1 | PPO 熵正则符号取反回去 | 熵项梯度为**最大化**方向（`+proba*(entropy+log(proba+eps))`） | 符号反=主动压制探索、策略早坍缩 | `test_drl_fixes.py`；训练 entropy 应缓升后稳 |
| D2 | GAE 跨 episode 串扰 | 按 episode 分段（done 处 reset，next_val=0） | 串扰使优势估计失真 | `test_drl_optimizations.py` |
| D3 | env.reset 不重置塑形状态 | reset 补 `_peak_equity/_last_dd/_losing_streak` 复位 | 跨 episode 污染奖励 | `test_drl_fixes.py` |
| D4 | factor_env 全样本 z-score | 用训练段/expanding 统计量（无前视）+ 按 selected 组合缓存 fitness | 全样本标准化=奖励含未来信息 | `test_drl_*` |
| D5 | 删 OOS/安检门 | 因子门（\|rank_ic\|≥0.01、\|icir\|≥0.1、换手≤0.5）+ IS/OOS 双段验证 + evolve `_oos_gate_passed` + 跨标门 + 增量门（`min_new_bars`） | 删了=进化引擎自动产出过拟合因子/策略并部署 | `test_evolve_oos_gate.py`、`test_oos_split.py` |
| D6 | 回退轮不落库 | 回退路径也调 `_log_round(status="rollback")` 落 `evolve_rounds` | **训练回退是常态**（历史最佳难超越），不落库=前端曲线恒空白（2026-09-01 刚修） | 查 `evolve_rounds` 表有 rollback 行；进化曲线有数据 |
| D7 | rl_evolve 只在"训练成功"时注册 | `EvolveEngine.start()` → `_restore_evolve_strategies()`：存在 flat 模型即注册+落库 AiStrategy | 只在成功时注册=几乎永远不注册（2026-09-01 刚修） | 重启后 `/api/trading/strategies` 含 rl_evolve |
| D8 | 策略分类丢"进化"桶 | `/api/strategy-repo/all` 按 `created_by` 分类，`evolve_engine` → kind="进化" | 丢了会落到"AI设计"兜底桶，用户找不到进化产物 | 策略库页进化策略徽标正确 |
| D9 | ModelZoo 破坏 flat 兼容 | 版本化 `data/models/<name>/`（vN.json.gz）+ **保留兼容平铺 `<name>.json`** 双路径互通 | web/api/drl.py ACAgent 只认 flat 格式，砍一半=旧模型全 404 | `/api/drl/models` 能列出且能加载 |
| D10 | pine_export 放宽约束 | state_window>1 拒绝导出、bb_pos clip、min_trade_zone 死区、bar_index>60 门控 | 放宽=导出的 Pine 代码与模型行为不一致 | `test_pine_export*` |
| D11 | 训练周期重新绑定实盘周期 | `evolve_timeframe`（空=跟随 default_timeframe）+ `evolve_symbols` 可单元素；固定品种时跨标 OOS 自动降级同标评估（**预期行为，不是 bug**） | 解耦是 2026-08 特性；"降级"曾差点被当 bug 修回 | `docs/EVOLVE_SPEC_SYMBOL_TIMEFRAME.md` |
| D12 | 删 reward_val_gap_penalty | 训练奖励含 val-gap 衰减扣分项 | 删了=DRL 过拟合回升（2026-08 调优成果） | 训练日志 val_gap 项存在 |

### E. 前端（index.html 单体）

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| E1 | `Promise.all` 引导链 | `Promise.allSettled` + 每个 loader 独立 try/catch | 一个接口挂=全站骨架屏卡死 | 手动断一个接口，其余视图仍可用 |
| E2 | vendor 改回 CDN | Vue/ECharts/Tailwind 本地 `web/static/vendor/` + SRI | 断网/局域网即白屏 | 断网启动页面正常 |
| E3 | Vue setup 返回 `_`/`$` 开头绑定 | setup 暴露给模板的名字**禁止下划线/`$`开头**（如 `_fmt` → 用 `fmtUtil` 等命名） | Vue3 模板不渲染 `_/$` 开头绑定 → 菜单打不开/白屏（真实事故） | 页面手测菜单与相关功能 |
| E4 | 图表复用已 dispose 实例 | 统一 `initChart()`（isDisposed 检查）+ 全局 resize 防抖 + onUnmounted 清理 | 切页后空白、内存泄漏 | 切页往返后图表正常 |
| E5 | 轮询无治理 | `loadAll` 按视图门控 + `visibilitychange` 后台暂停 + onUnmounted 清 interval | 无治理=后台标签持续打接口 | 后台标签 DevTools 零请求 |
| E6 | 裸 `toFixed/Date.parse` | 统一 fmt/pct 守卫（null 安全） | 一个 null 字段渲染异常可中断整个 Vue 渲染 | OKX 切换等边界数据下页面不冻结 |
| E7 | 删 `!important` 补丁 / <11px 字号 | 主题切换走 `:root[data-theme=light]` token 覆盖；最小字号 11px | Phase 3 UI 重构红线（Apple HIG） | `grep '!important' index.html` ≈ 0 |
| E8 | 事件监听不清理 | onUnmounted 清理全部 interval/listener | 单页应用反复挂载泄漏 | 长会话内存 |

### F. 配置与环境（Windows 专项）

| # | 禁止回退到 | 现状（必须保持） | 为什么 | 怎么验 |
|---|---|---|---|---|
| F1 | 第三方库默认编码读文件 | `settings.py` 子类化 YamlConfigSettingsSource，`_read_file` **显式 encoding='utf-8'** | 中文 Windows=GBK，config.yaml 有中文注释时**启动即崩**（开发 shell UTF-8 复现不了，双击 bat 必现） | 双击 start.bat 能启动 |
| F2 | .bat 里写中文/非 ASCII | `start.bat` **纯 ASCII**；端口检测逻辑在 `scripts/check_port.py` | cmd 按 ANSI 解析 UTF-8 → 每行首字符被吞；for 块内嵌 python -c 括号冲突 | 双击启动 200 |
| F3 | 未配置令牌每次重启重生成 | 令牌持久化 `data/api_token.txt` | 重生成=浏览器旧 Cookie 全 401（"运行报错"） | 重启两次令牌一致 |
| F4 | 单端口硬编码 | start.bat 端口循环检测 8000-8010 | 占用即崩 | 8000 占用时自动切 8001 |
| F5 | ccxt 注入 httpProxy/httpsProxy | REST 用 `aiohttp_proxy` only；ccxt.pro WS 用 **`wsProxy`** | httpProxy 对 WS 无效，曾致 OKX 连接全断（回归事故） | 交易所测试连通 |
| F6 | 后台任务无并发上限 | backtest/ai/drl 每类 `threading.Semaphore(2)`，超限 429 | 无限叠加 CPU 密集线程 | 连发第 3 个任务得 429 |
| F7 | 任务进度缓存无限增长 | `tasks_cache.py` TTL 清理（ai/backtest/drl 三处接入） | 内存缓慢泄漏 | 长跑内存 |
| F8 | `pytest-asyncio` 缺失 | `.venv` 必须装 pytest-asyncio（pytest.ini `asyncio_mode=auto`） | 缺了 C/D 类异步测试全挂；重新装环境易漏 | pytest 全绿 |

## 4. 验证链（提交前必跑，顺序执行）

```bat
:: 1. 编译
.venv\Scripts\python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py
:: 2. 测试（当前 434 例）
.venv\Scripts\python -m pytest -q
:: 3. 前端 JS 语法（esprima，无需 node）
.venv\Scripts\python scripts\check_js.py
:: 4. 双引擎一致性（凡触碰回测/指标/撮合/策略语义，必跑并与 §1 基线对比）
.venv\Scripts\python run.py backtest --source demo --strategy dual_ma
:: 5. 冒烟（start.bat 启动后）
::    GET /api/health → 200 且含 degraded/bus 键
::    无令牌 GET /api/portfolio/summary → 401；带令牌 → 200
```

**全绿才允许 commit。** 双引擎基线漂移 = 必须解释（有意变更 → 更新 §1 表；否则 → 修复）。

## 5. "反向升级"症状速查表

| 症状 | 大概率是哪条红线被碰了 | 先查 |
|---|---|---|
| price_action 系策略回测有交易、实盘 0 交易 | B4 sr/pa 补算被删 | `grep sr trading_engine.py` `_on_candle_locked_inner` |
| 引擎"在跑"但不下单/不止损 | B5 风控兜底/失败计数被删 | `/api/health` 的 degraded；`data/logs/trader.log` |
| 进化引擎曲线空白、`evolve_rounds` 表空 | D6 回退轮落库被删 | DB `evolve_rounds`；log 的 `[evolve]` 行 |
| 重启后 rl_evolve 策略消失 | D7 启动自动注册被删 | `/api/trading/strategies` |
| 双击 start.bat 启动即崩 UnicodeDecodeError | F1 yaml 编码显式声明被删 | `settings.py` `_read_file` |
| .bat 运行错乱、`cho`/`EM` 类报错 | F2 bat 混入非 ASCII | `start.bat` 文件编码 |
| 重启后全站 401 | F3 令牌持久化被删 | `data/api_token.txt` 是否复用 |
| OKX/交易所连不上（走代理时） | F5 代理注入字段被改 | REST=aiohttp_proxy / WS=wsProxy |
| 间歇性 `RuntimeError: no running event loop`、回测/AI 结果偶尔不落库 | B3 跨 loop DB 直连回来了 | `grep run_coroutine_threadsafe web/api/` |
| 回测和实盘信号对不上、一致性测试红 | A1-A4（成交时点/ddof/撮合分叉） | 跑一致性测试，diff 两引擎信号 |
| AI 自动优化"跑了很多轮但参数从没变过" | C1 optimizer 返回值被改 | 优化日志 params 前后对比 |
| 某页面菜单点不开/整块白屏 | E3 setup 下划线绑定 | 检查 setup 返回的变量名 |
| 断网/局域网打开全白屏 | E2 vendor 被改回 CDN | `web/static/vendor/` 三文件 + SRI |
| DRL 训练 entropy 一路下降、策略很快收敛到单边 | D1 熵正则符号被反 | `drl/agent.py` 熵项符号 |
| 进化引擎产出策略明显过拟合 | D5 OOS/安检门被绕过 | 训练日志 OOS 段 |

## 6. 再开发工作流

1. **开工前**：commit 当前工作区（2026-09-01 有约 145 个未提交文件，务必先固化基线快照）→ 跑一遍 §4 验证链确认起点全绿 → 读本文档相关章节。
2. **设计中**：涉及 §3 红线条目，先想清楚"当年为什么这么写"（本文档"为什么"列 + §8 历史文档）；改设计先在 §3 追加决策记录。
3. **实现中**：行为变更必须补 pytest（本项目惯例：每轮修复伴随测试文件，如 `test_drl_fixes.py`、`test_optimizations_p0p1p2.py`）；Windows 下新文件一律 UTF-8（无 BOM），`.bat` 一律纯 ASCII。
4. **提交前**：§4 验证链全绿；触碰交易/回测语义的，跑双引擎一致性并对比 §1。
5. **提交后**：基线变了就更新 §1 表；新踩的坑/新红线随时补进 §3（本文档是活的）。
6. **测试引擎/回测前**：确认 DB 里 `trading_mode=paper`（DB 中可能残留 okx 实盘配置）。

## 7. 环境事实（2026-09-01）

- venv：`.venv/`（Python 3.13），全部命令用 `.venv\Scripts\python.exe`；git CLI **现在可用**（早期"无 git CLI"的记录已过时）。
- 无 node/npm：JS 检查用 `scripts/check_js.py`（esprima + ES2020 兼容层，已装 .venv，dev 依赖不入 requirements）。
- 网络：Binance 行情走官方 `.vision` 数据域可直连；OKX/Bitget/Bybit 需本地代理（Clash 7890 等，自动探测）；GitHub push 需 VPN。
- 端口：服务默认 8000，占用自动 8001-8010；冒烟测试曾用 8017 避让。
- 冒烟/手测遗留项历来留给用户浏览器端验证（本项目无浏览器自动化常驻环境）。

## 8. 文档地图（冲突时以本文档为准）

| 文档 | 定位 |
|---|---|
| **docs/REDEV_GUIDE.md（本文档）** | 当前状态 + 基线 + 禁止回退清单。**再开发唯一入口** |
| docs/BASELINE.md | 2026-08-14/15 Phase 0 基线存档（历史锚点，含 QA 验收记录） |
| docs/REFACTOR_PLAN.md | 2026-08-14 三路审计 + 5 阶段重构方案（已全部完成，§8 有执行状态） |
| docs/QUANT_ADVICE.md | 2026-08-15 量化建议 10 项落地记录（基准对比/熔断人工确认/因子 IC 衰减下线等） |
| docs/EVOLVE_SPEC_SYMBOL_TIMEFRAME.md | 进化引擎"固定 vs 轮换品种/周期"论证（D11 的设计依据） |
| docs/plans/ | 2026-08-15 团队开发模块契约（architecture + module_A~E + m1~m6） |
| docs/superpowers/specs/ | 三份功能设计 spec：AI 模型拉取（08-14）/ 持续进化引擎（08-19）/ UI macOS 重构设计（08-28，未实施） |
| README.md | 用户视角功能说明 + 快速开始 + 验证链命令 |
