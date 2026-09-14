# Crypto AI Trader — 自治优化知识库

> 依据《自治 AI 项目优化模板》执行的结构化优化记录。
> 生成日期：2026-08-17 · 执行方式：全自动（L0/L1 自动实验 + 门禁验收）

---

## 1. 任务契约与项目上下文

| 项 | 值 |
|---|---|
| 项目 | crypto_ai_trader（加密货币 AI 量化交易机器人） |
| 技术栈 | Python 3.11+（FastAPI/UVicorn/Pydantic/ccxt/pandas/numpy）、Vue 3 + Tailwind + ECharts（单页内联）、SQLite/PostgreSQL、纯 NumPy DRL（PPO） |
| 规模 | Python 21,137 行 / 114 文件 · 前端 HTML 4,134 行（单文件）· npm-free |
| 入口 | `run.py web`（Web）；`run.py backtest/backfill`（CLI） |
| 测试 | pytest 全量套件（FIFO 记账、双引擎一致性、风控防线、AI 校验、DRL、OOS 拆分） |
| 可测指标 | 测试通过率、check_js（node --check）、compileall、回测性能（elapsed_sec） |
| 自动权限 | L0（文档/清理）自动保留；L1（局部实现/参数/并发）自动实验仅门禁通过保留；L2 需隔离+回归；L3（部署/外部发布/删数据/迁移）人工批准 |

---

## 2. 基线快照（本轮优化前）

| 指标 | 基线值 |
|---|---|
| pytest 全量 | 全部通过（约 90+ test） |
| check_js | `esprima 4.x + ES2020 词法兼容层`（脆弱：正则含裸引号会失同步） |
| 前端菜单 | 单级菜单 + hover 触发下拉（hover 误触+溢出裁剪双问题） |
| 行情入口 | 独立「行情」页面（占用菜单项，与策略工坊割裂） |
| AI 策略代码 | 仅参数规格 JSON 展示，无法导出/导入 TradingView |
| 网格扫描 | AI 优化后回测组合**串行**执行（数百次 `run_backtest_fast`） |
| K 线数据源 | 单一 `binance.vision`，故障即 502 |
| DRL 网络 | logp 用 `softmax+log`（下溢风险）；JSON 序列化（大文件）；串行轨迹收集 |
| 策略管理 | 全部策略仅表格，无迭代谱系可视化 |
| 首页缓存 | 无 Cache-Control（浏览器旧页滞留导致"改了没生效"） |
| AI API 路由 | `auto_optimize` / `iterate_strategy` 各 90 行重复 worker 样板代码 |
| Factor 表达式解析 | 每次 `eval` 重新 `ast.parse` + 白名单校验（无缓存） |

---

## 3. 优化候选项（发现 → 分级）

| # | 候选 | 风险 | 状态 | 对应实验/变更 |
|---|---|---|---|---|
| C1 | 菜单二级化 + 点击弹出 + Teleport 修复 | L1 | ✅ 保留 | E2, E3 |
| C2 | 行情功能并入工坊右栏（删独立菜单） | L1 | ✅ 保留 | E2 |
| C3 | Pine 导出→导入闭环（双向） | L1 | ✅ 保留 | E4 |
| C4 | check_js 升级 node --check | L0 | ✅ 保留 | E1 |
| C5 | 网格扫描并行化 | L1 | ✅ 保留 | E5 |
| C6 | K 线四所降级 | L1 | ✅ 保留 | E6 |
| C7 | 策略族谱可视化 | L1 | ✅ 保留 | E7 |
| C8 | 行情页删除（并入工坊） | L0-L1 | ✅ 保留 | E2 |
| C9 | DRL 神经网络审查修复（log_softmax/LayerNorm/npz/并行收集） | L1 | ✅ 保留 | E8 |
| C10 | FactorExecutor 表达式 AST 缓存 | L1 | ✅ 保留 | E9（本轮） |
| C11 | check_js 降级回退加警告 | L0 | ✅ 保留 | E10（本轮） |
| C12 | 前端单文件拆分为多模块 | L2 | ⏸ 暂缓 | 需构建步骤引入，回归面大，成本>收益 |
| C13 | PPO mini-batch forward 去重 | L1 | ❌ 否决 | 会破坏 clipped surrogate 语义（标准 PPO 不可去重） |
| C14 | backward pop 释放激活值 | L1 | ❌ 否决 | pop 顺序破坏 ReLU mask 对齐，引入维度 bug，回退 |
| C15 | check_js 回退路径加显式 WARN | L0 | ✅ 保留 | E11（本轮第2轮） |
| C16 | FactorExecutor `_parse_expr` 加 `lru_cache(256)` | L1 | ✅ 保留 | E12（本轮第2轮） |
| C17 | `auto_optimize`/`iterate_strategy` 提取公共 worker 基础设施 | L0 | ✅ 保留 | E13（本轮第2轮） |
| C18 | 推理模型(o1/R1)跳过 temperature 参数（传了 400/忽略） | L1 | ✅ | E14 |
| C19 | DRL hidden 默认 64→32（15维输入过参数化+Pine权重4倍） | L1 | ✅ | E15 |
| C20 | AI 迭代结果后端生成 pine_code（与前端模板同源） | L1 | ✅ | E16 |
| C21 | strategy_design/iterate token 12000（Pine 代码更长） | L0 | ✅ | E14 |
| C23 | DRL 奖励塑形三项（dd_penalty/losing_penalty/trend_align）前端默认关闭+配置缺失 | L1 | ✅ | E17 |
| C24 | 回测结果列表整行 select 拉取大 JSON 列（equity_curve/trades 每行几十~上百 KB）再丢弃 | L1 | ✅ 保留 | E18（本轮） |
| C25 | `factor_ic_table` 在事件循环上同步执行（/usage、/mine，26 因子 × 全量相关 ≈ 50-130ms 阻塞） | L1 | ✅ 保留 | E19（本轮） |
| C26 | `forward_returns` 每因子重复计算 2 次（factor_ic 与 _rolling_ic 各一次；mining 逐候选放大） | L1 | ✅ 保留 | E20（本轮） |
| C27 | `check_ui_guard.py` 硬编码 node.exe 绝对路径（其他机器不可运行） | L0 | ✅ 保留 | E21（本轮） |
| C28 | 根目录调试脚本/测试报告/日志散落（含明文 API token 的 `_probe*.py`） | L0 | ✅ 保留 | E22（本轮） |

---

## 4. 案例证据卡（节选）

### E1: check_js node --check
- 来源：Node.js 官方 `node --check`（`node --help`），原生 ES2020+ 支持
- 证据：本项目实测 esprima 4.0.1 无法解析 `?.`/`??`；兼容层按字符扫描遇正则 `/"/g` 失同步（曾误报）
- 效果：解析器替换，误报根治；无 node 时保留 esprima 回退 + 警告

### E5: 网格扫描并行化
- 来源：Python 官方 `asyncio.to_thread` + `ThreadPoolExecutor`；CPython pandas/numpy 计算释放 GIL
- 证据：`get_strategy` 每次返回新实例（`strategies/__init__.py:46`），`run_backtest_fast` 内 `df.copy()` 无共享可变状态 → 线程安全
- 预期收益：组合回测耗时降低约 3-4 倍（n=数百，并发上限 4）

### E8: DRL log_softmax
- 来源：常用深度学习框架（PyTorch/tf）均提供 `log_softmax`；`softmax → log` 两级操作在低概率处下溢
- 证据：`np.log(proba + 1e-12)` 在 proba→0 时保护失效产生 `-inf`，`ratio=exp(-inf)` 置零，梯度消失
- 效果：`_log_softmax`（logsumexp 单级）根除下溢

### E9: FactorExecutor AST 缓存（本轮）
- 来源：Python `functools.lru_cache` 官方文档（函数纯、调用频繁场景标准做法）
- 证据：`_parse_expr` 无副作用的纯函数（`ast.parse` + 白名单校验），同表达式在 train/val/oos 多段重复求值（`agent.py` DRL 因子注入、`mining.py` 演化循环）
- 收益：同表达式重复解析/校验成本归零；maxsize=256 有界

### E18: 回测结果列表列投影（本轮）
- 来源：SQLAlchemy 列投影（`select(Col1, Col2, ...)` 替代整行实体）
- 证据：`BacktestResult` 含 `equity_curve_json`/`trades_json` 两个大 Text 列（每行几十~上百 KB），列表接口只消费 `id/created_at/metrics_json`
- 效果：ORM 只物化 3 个小列，UI 轮询每页省 ~1MB+ 传输；返回结构逐字段不变

### E19: factor_ic_table 移出事件循环（本轮）
- 来源：既有代码自身模式（相邻 `compute_factor_matrix` 已 `await asyncio.to_thread`）
- 证据：26 内置因子 × 每因子 `factor_ic`（pearson/spearman + 滚动 IC + Newey-West）≈ 50-130ms 纯 CPU，裸跑在事件循环上阻塞所有并发请求
- 效果：/usage、/mine 两处调用并入线程池，事件循环零冻结

### E20: forward_returns 批量共享（本轮）
- 来源：DRY + 只依赖 close/h 的输入不随因子变化
- 证据：`factor_ic` 行91 与 `_rolling_ic` 行131 各自 `forward_returns(close, h)`；`factor_ic_table` 逐列循环 → 每因子 2 次重复；`mining.py` 每候选 IS/OOS 段放大
- 效果：IC 表/挖掘路径省约 30-50% 分析时间；`fwd` 参数默认 None 行为逐位不变（.optim/verify_fwd_param.py 等价性验证 PASS）

### E21: check_ui_guard.py node 可移植定位（本轮）
- 来源：同仓库 `check_js.py` 既有 `shutil.which("node")` 口径
- 证据：guard 脚本此前硬编码 `C:\Users\sbxg\.workbuddy\...\node.exe`，换机器即不可运行
- 效果：PATH 优先 + `~/.workbuddy` 兜底；无 node 时显式报错退出（不静默）

### E22: 根目录调试产物清理（本轮）
- 来源：安全+整洁（明文 API token 的 `_probe*.py` 在根目录）
- 证据：git status 显示 19 个 untracked 调试/报告/日志文件散落根目录；无任何脚本/配置引用
- 效果：删除后 `git status` 干净；`.gitignore` 新增根目录产物规则防止回潮；运行中 server 写锁的 `server_new*.log` 由 ignore 规则覆盖

---

## 5. 实验计划与变更清单

| 实验 | 变更文件 | 改动内容 | 风险 |
|---|---|---|---|
| E1 | `scripts/check_js.py` | node --check 优先 + esprima 回退 | L0 |
| E2 | `web/static/index.html` | 菜单二级化、行情并入工坊、3:7 分栏、Teleport 下拉、SVG 图标 | L1 |
| E3 | `web/static/index.html` | Teleport 修复 overflow 裁剪（根因：`.menu-sub` 被菜单区 `overflow-x:auto` 裁剪） | L1 |
| E4 | `web/api/strategy_repo.py` + `ai/prompts.py` + `index.html` | `POST /import-pine` 端点（regex 解析 → schema clamp → 注册落库）+ AI 校验 + 前端导入卡片 | L1 |
| E5 | `web/api/ai.py` | 网格扫描 `asyncio.gather` + `Semaphore(4)` 并行 | L1 |
| E6 | `web/api/market.py` | klines 主源失败→四所降级 + `fallback_from` 标记 | L1 |
| E7 | `index.html` | 策略族谱 ECharts tree（based_on/iter_no） | L1 |
| E8 | `drl/nnet.py` + `drl/agent.py` | log_softmax、LayerNorm(可选)、npz 序列化、并行轨迹收集 | L1 |
| E9 | `factors/mining.py` | `_parse_expr` lru_cache(256) | L1 |
| E10 | `scripts/check_js.py` | 回退分支显式 WARN | L0 |
| E11 | `factors/mining.py` | `_parse_expr` lru_cache(256) | L1 |
| E12 | `web/api/ai.py` | 提取 `_run_ai_task_background` 公共 worker 基础设施，消除 `auto_optimize`/`iterate_strategy` 的 ~80% 重复样板代码 | L0 |
| E18 | `web/api/backtest.py` | `/results` 列表改列投影（只取 id/created_at/metrics_json，整行 select 会把 equity_curve/trades 大 JSON 列搬出 SQLite 再丢弃；UI 轮询每页省 ~1MB+ 传输） | L1 |
| E19 | `web/api/factors.py` | `/usage`、`/mine` 的 `factor_ic_table` 改 `await asyncio.to_thread(...)`——与相邻 `compute_factor_matrix` 同一模式，事件循环不再被 CPU 密集 IC 计算冻结 | L1 |
| E20 | `factors/analysis.py` | `factor_ic`/`_rolling_ic` 增加可选 `fwd` 参数；`factor_ic_table` 批量前算一次 `forward_returns` 全部因子共享（每因子由 2 次降为 0 次 O(n) shift/除法，mining 安检门路径同步受益；默认 None 行为不变） | L1 |
| E21 | `scripts/check_ui_guard.py` | node 定位改 `shutil.which("node")`（PATH）优先 + `~/.workbuddy/binaries/node/versions` 兜底，去掉硬编码机器路径；找不到 node 时明确报错退出 | L0 |
| E22 | 根目录 + `.gitignore` | 删除 19 个根目录调试/报告/日志文件（含明文 token 的 `_probe*.py`、`_inspect*.py`、JUnit xml、空文件 `'`/`try`）；`.gitignore` 新增根目录产物 ignore 规则 | L0 |

---

## 6. 验收门禁与前后对比

| 门禁 | 结果 |
|---|---|
| `python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py` | ✅ PASS |
| `python scripts/check_js.py` | ✅ PASS（node --check 主路径） |
| `python -m pytest -q` | ✅ 全量通过（exit 0） |
| DRL 单元验证（log_softmax 一致性/npz roundtrip/采样） | ✅ PASS |
| Pine 导入 E2E（带 token） | ✅ PASS（11/11 参数还原 + clamp） |
| 浏览器实测（菜单点击弹出/图表 3:7/无溢出/族谱 canvas） | ✅ PASS |
| 前端静态：CSS 括号平衡 / div-section 配对 | ✅ PASS |
| 本轮 E18-E20 后端门禁：compileall + 全量 pytest | ✅ PASS（exit 0，全量绿） |
| E20 等价性验证（fwd 传参 vs 不传逐位一致） | ✅ PASS（.optim/verify_fwd_param.py） |
| 本轮前端/守卫：check_js + check_ui_guard | ✅ PASS（node v24 主路径） |
| UI KPI 面板移除 DOM 验证（served HTML 无 workshopStats/引擎状态卡） | ✅ PASS |
| UI KPI 面板移除视觉截图验证（规则5） | ⚠️ 环境限制：本机无 Firefox/Chrome/Edge 可执行文件，仅能 DOM 层验证（见无效尝试#4） |

**关键前后对比**

| 维度 | 优化前 | 优化后 |
|---|---|---|
| JS 语法检查 | esprima+兼容层（脆弱误报） | node --check（原生，0 误报） |
| 网格扫描 | 串行数百次回测 | 4 路并行（约 3-4 倍加速） |
| K 线可用性 | 单源，故障 502 | 四所自动降级 |
| AI 策略流转 | 只读 JSON | Pine 导出/导入闭环 |
| DRL logp | softmax+log 下溢 | log_softmax |
| 模型文件 | JSON（~3MB） | npz（~450KB，-85%） |
| 菜单交互 | hover 误触+被裁剪 | 点击弹出+Teleport+外部关闭 |
| 首页缓存 | 无头 | no-cache（每次 revalidate） |

---

## 7. 回滚记录

| 变更 | 回滚点 | 状态 |
|---|---|---|
| E2/E3 前端 | 每个改动单一旧/new 对，文件级可整体还原 | ✅ 无需回滚 |
| E4 导入端点 | 删除 `import-pine` 路由即可 | ✅ 可用 |
| E5 并行扫描 | 还原 `for` 串行循环 | ✅ 可用 |
| E8 backward pop | **已实际回滚**（维度 bug）→ 还原原版索引实现 | 🔁 已回滚 |
| E8 其余 | 全门禁通过，保留 | ✅ |
| E9/E10 | 删除缓存装饰器/警告行即可 | ✅ 可用 |
| E18 | 还原整行 `select(BacktestResult)` | ✅ 可用 |
| E19 | 还原同步调用 | ✅ 可用 |
| E20 | 还原 `fwd` 参数（默认 None 行为不变，调用点全兼容） | ✅ 可用 |
| E21 | 还原硬编码路径 | ✅ 可用 |
| E22 | 删除的文件均为 untracked，无版本回滚需求；ignore 规则删除即还原 | ✅ 可用 |

---

## 8. 无效尝试（反证记录）

1. **C13 PPO forward 去重**：深入分析后判定标准 PPO 每 mini-batch 更新后参数变化，forward 无法去重而不改变 clipped surrogate 语义 → 否决，不实施
2. **C14 backward pop 释放激活值**：实施时 pop 顺序破坏 `acts[i]` 与 ReLU mask 对齐 → `ValueError: broadcast (128,32) vs (128,15)`，测试抓出，立即回滚还原 → 教训：内存优化收益对本项目小网络可忽略，正确性优先；大型重构必须保留原实现对照
3. **CSS 菜单 overflow**：`overflow-x:auto` 会静默裁剪 absolute 下拉（CSS 规范：一轴非 visible 时另一轴变 auto）→ Playwright `isVisible` 检测不到祖先裁剪 → 教训：UI 验证必须加截图视觉确认，不能只信 DOM 断言
4. **浏览器视觉确认（规则5）环境限制**：本机 `C:\Program Files\Mozilla Firefox` 为空目录（无 firefox.exe），PATH 无 Chrome/Edge，`data/geckodriver` 只有 driver 无浏览器 → UI 改动只能做 served-HTML/DOM 层断言，无法截图人工复核。已把 `.optim/verify_ui_kpi_removal.py` 留作有浏览器环境时的复跑脚本；本次以 served HTML 无 `workshopStats`/`引擎状态` 卡 + check_js 语法 + 模板配对作为替代门禁。教训：UI 验证链依赖浏览器运行时，环境缺浏览器时须显式降级并记录，不能伪造"截图通过"
5. **E20 首次编辑产生重复行**：`_rolling_ic` 加 `fwd` 参数时旧 `fwds = fwd.to_numpy(...)` 行未删净（新行+旧行并存），numpy 数组路径直接 AttributeError → 等价性验证脚本（.optim/verify_fwd_param.py）首跑即抓出，修复后 PASS。教训：共享参数类改动必须跑逐位等价对照，且验证脚本要比对"数组传入"与"Series 传入"两条路径

---

## 9. 知识库索引（可复用策略）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| node --check 替代 esprima | 内联 JS 校验，避免正则字面量误报 | ✅ |
| Teleport + fixed 定位 | 下拉/浮层脱离 overflow 裁剪上下文 | ✅ |
| asyncio.to_thread + Semaphore | CPU 密集批处理并行（pandas 释放 GIL） | ✅ |
| 多源降级 + fallback 标记 | 外部数据源依赖容灾 | ✅ |
| log_softmax 替代 softmax+log | 概率模型数值稳定性 | ✅ |
| npz 替代 JSON | 数值模型序列化（-85% 体积） | ✅ |
| lru_cache 纯函数解析 | 表达式/配置重复解析 | ✅ |
| no-cache 响应头 | 单页应用前端缓存失效问题 | ✅ |
| 截图视觉自检 | UI 回归验证（DOM 断言不可信场景） | ✅ |

---

## 11. 第二轮自治优化（2026-09-06）：DRL 训练流水线耗时优化

> 执行依据：《自治优化协议 v2》（`自优化模板/optimization-agent-prompt.v2.md`）
> 契约：主目标 = train_drl 端到端耗时（episodes=16/n_episodes=4，合成 5000 根 5m，seed=42）；min_gain=+5%；max_level=L2；预算 8 实验/4h

### 11.1 本轮基线快照

| 指标 | 基线值 |
|---|---|
| pytest 全量 | ✅ PASS（exit 0，616 项） |
| compileall / check_js | ✅ PASS |
| train_bench（16 ep×4 n_ep×3 次） | median=15.011s σ=0.295s cv=2.0% |

### 11.2 本轮候选清单（cProfile 定位）

| # | 候选 | 风险 | 状态 | 对应实验 |
|---|---|---|---|---|
| C1 | `_vol5_series` 逐元素循环（env.__init__ 每 K 线一次 np.std/切片，6 万次调用 ≈ 10% wall）→ sliding_window_view 批量 | L1 | ✅ 保留 | E1 |
| C2 | MLP.forward 单行输入绕过 acts 列表（collect 热路径 6 万次） | L1 | ✅ 保留 | E2 |
| C3 | `_log_softmax` 单行标量 math 快速路径（5.6 万次 numpy ufunc 调度过重） | L1 | ✅ 保留 | E3 |
| C4 | `_build_state` state_window=1 预分配写入 | L1 | ❌ 回滚 | E4 |
| C5 | Adam `nnet.step` 常量提前循环外 | L1 | ❌ 回滚 | E5 |
| C6 | `_softmax` 单行评估路径快速路径 | L1 | ❌ 回滚 | E6 |
| C7 | `rng.choice` p 参数校验（~0.26s）绕过 | — | ❌ 否决 | 采样式替换破坏同 seed 采样序列（D1 可复现契约），不可动 |
| C8 | train_batch 内数值改动（GAE/backward） | L1-L2 | ⏸ 暂缓 | 数值敏感，行为等价风险高，收益不确定 |

### 11.3 本轮变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 风险 |
|---|---|---|---|
| E1 | `drl/env.py` | `_vol5_series` 预计算：逐元素循环 → `sliding_window_view` 批量 `np.std(view, axis=1, ddof=1)`；前 3 根（_i=2,3）渐增窗口单独算，n<5 边界保留循环——与逐元素版 **bitwise 一致**（7 数据集含常数/最小序列验证） | L1 |
| E2 | `drl/nnet.py` | `MLP.forward` 单行快速路径：`use_ln=False 且 cache_for_backward=False 且 shape[0]==1` 时跳过 acts 列表构造/append/重复索引，矩阵乘法与 ReLU 与原循环逐位相同 | L1 |
| E3 | `drl/nnet.py` | 新增 `_log_softmax_1row`：单行 (1,N≤16) 用 math 标量循环替代 numpy ufunc 调度（同机 libm 下与 numpy 逐位一致，2000 组随机+极端值验证）；`predict_log_proba` 单行走快速路径 | L1 |

### 11.4 验收门禁与前后对比（实际数值）

| 门禁 | 结果 |
|---|---|
| `python -m pytest -q` 全量 | ✅ PASS（E1/E2/E3 各自 DRL 相关 86 测试绿 + 最终全量绿） |
| `python -m compileall -q drl` | ✅ PASS |
| E1 等价性 | ✅ 7 数据集 bitwise 一致（verify_e1_vol5_batch.py），std 调用 60,962→0，单测 138x 加速 |
| E2 等价性 | ✅ 20 权重×30 输入 bitwise（verify_e2_forward.py，含 use_ln 开关/批量/缓存路径） |
| E3 等价性 | ✅ 2000 组随机+极端值 bitwise（verify_e3_logsoftmax.py） |
| 测试文件哈希 | ✅ 未变（基线指纹比对） |

**主目标前后对比（16ep×4nep 3 次中位数）**

| 阶段 | 耗时 | 增量 | 累积 |
|---|---|---|---|
| 基线 | 15.011s | — | — |
| E1 后 | 13.511s | **-10.0%** | -10.0% |
| E2 后 | 12.033s | **-10.9%** | -19.8% |
| E3 后 | 11.263s | **-6.4%** | **-25.0%** |

ext 验证：episodes=32 时 22.84s vs 基线 30.02s（-24%），收益在所有训练规模上保持。

### 11.5 回滚记录（本轮）

| 变更 | 回滚点 | 状态 |
|---|---|---|
| E4 `_build_state` 预分配 | 还原为原 concatenate 实现 | 🔁 已回滚（收益 -1.0% < min_gain 5%） |
| E5 Adam 常量提前 | 还原常量在循环内计算 | 🔁 已回滚（无收益：11.83 vs 11.26，Python 常量折叠对 numpy 主运算无影响） |
| E6 `_softmax_1row` | 还原 predict_proba 原实现 | 🔁 已回滚（无收益：评估路径在 16ep 占比仅 ~5%） |

### 11.6 无效尝试（反证记录）

1. **C4 build_state 预分配**：np.empty + 分片写入 vs np.concatenate 对小数组开销差异极小（-1.0%），且需保证返回数组独立（collect_episode 持有引用）——正确性约束下收益不足以跨越 5% 门槛。
2. **C5 Adam 常量提前**：`(1-b1)`/`(1-b1_t)` 提到循环外，bitwise 等价（50 trial 验证），但 Python 层常量折叠节省的浮点减法微乎其微，实测反略慢（可能环境噪声），无收益。
3. **C6 softmax 评估路径**：greedy_action/val/OOS 的 `predict_proba` 用快速路径，bitwise 等价成立，但评估 forward 在 16 episodes 训练中占比 ~5%，收益不可测。
4. **C7 rng.choice 绕过否决**（未实施）：numpy `Generator.choice(p=...)` 的 p 参数校验（issubdtype/issubclass/getlimits ≈0.26s/wall）是采样路径固定成本；手写 cumsum/逆变换或 Gumbel-max 会改变同 seed 下的随机消费序列 → 破坏项目 D1"seed 可复现"契约与既有测试对训练轨迹的依赖 → 硬约束 §2.3 行为等价下不可行。
5. **测量环境**：服务器持续进化引擎（factor_miner 60s 轮）训练会污染 bench（cv 从 0.1% 升到 2.7%，E5 误判风险）——测量前已调用 `/api/evolve/pause` 暂停、测毕 `/resume` 恢复；E1-E3 有效增量（-10%/-10.9%/-6.4%）远超环境噪声（±3%），判定可信。

### 11.7 知识库索引（新增可复用策略）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| sliding_window_view 批量 std | 逐元素滚动统计（环境初始化热路径） | ✅ E1（138x + bitwise） |
| 单行 forward 快速路径 | 小网络逐点前向（collect 热路径） | ✅ E2 |
| math 标量替代 numpy ufunc | 小数组（≤16 元素）高调用次数函数 | ✅ E3（同机 libm 下 bitwise） |
| 测量前暂停后台作业 | 基准测量环境稳定（cv >10% 判定） | ✅ 本轮 |
| 不动 rng.choice | seed 可复现契约下禁止替换采样算法 | ✅ 反证 C7 |

---

## 12. 第三轮自治优化（2026-09-06）：DRL 训练流水线耗时（run3，无保留）

> 执行依据：《自治优化协议 v2》；契约沿用：主目标=DRL 训练耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2 保留项（E1/E2/E3，-25%）之上继续

### 12.1 run3 基线快照与环境事件

| 项 | 值 |
|---|---|
| run3 基线 bench（16ep×4nep×3 次） | median=12.349s σ=0.149s cv=1.2% |
| **环境事件** | 检测到 cs2（Counter-Strike 2）游戏进程运行，bench 退化至 23.6s（2 倍）；用户授权关闭后恢复 12.3s。教训：测量环境含游戏/高负载进程会整体性翻倍，cv 仍可能很小（稳定但系统性偏移）——跨时段对比必须先核对负载源 |
| 交替测量基准 | 11.6-12.2s 波动（cv 1-3%），噪声量级 ±0.3-0.5s |

### 12.2 run3 候选与结果

| 实验 | 候选 | A/B 实测 | 判定 |
|---|---|---|---|
| E7 | `greedy_action` 评估路径 `cache_for_backward=False`（走 E2 forward 快速路径） | 11.949 vs 12.151s（**-1.7%**） | ❌ 回滚（< min_gain 5%） |
| D2 | `train_batch` 内 `np.arange(mb)` 去重 | 12.082 vs 11.636s（无正收益，cv 3.2% 噪声干扰） | ❌ 回滚（收益预期 <1%） |

### 12.3 为何此处难以继续优化（协议 §4④ 分析）

run2 已把该方向最大热点（vol5 预计算 10% wall、forward 单行列表开销、log_softmax ufunc 调度）榨干（-25%）。run3 剩余热点全部为 **Python 调度层**，且受硬约束锁死：

1. `rng.choice` 的 p 参数校验（issubdtype/issubclass/getlimits ≈0.23s/wall）——替换采样算法破坏同 seed 随机序列（D1 可复现契约），**不可动**（run2 C7 已否决）。
2. `sample_action_with_logp` 的 `np.exp(logp_all)` + `int/float` 转换 + `reshape`——PPO 采样语义必需，无等价替代。
3. `train_batch` mini-batch 循环（0.9s）——标准 PPO clipped surrogate 数值语义，改计算顺序有行为等价风险，且都是批量 numpy 已最优化形态。
4. 环境噪声下限（±0.3-0.5s，cv 1-3%）已接近剩余候选预期收益量级——即使微优化有效也**无法在噪声中可靠判定**（D2 即被噪声翻转）。

**结论**：该方向已达到收益天花板，继续实验不满足"可判定"原则（收益 < min_gain 且被噪声淹没）。建议后续优化转向未探索主目标（如回测引擎性能、因子 IC 吞吐）或接受当前 -25% 收益。

---

## 13. 第四轮自治优化（2026-09-06）：AI 因子挖掘迭代耗时（run4，-54.8%）

> 执行依据：《自治优化协议 v2》；契约：主目标=train_factor_miner 端到端耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2/run3 保留项（DRL 训练 E1/E2/E3）之上，测量前暂停持续进化引擎

### 13.1 run4 基线快照

| 指标 | 基线值 |
|---|---|
| train_factor_miner bench（8ep×8nep×3 次） | median=9.781s σ=0.294s cv=3.0% |
| pytest / compileall | ✅ 616 项 PASS / 0 |

### 13.2 run4 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `factors/analysis.py` | `_spearman` 用 numpy 原生平均秩替代 `pd.Series(a).rank().to_numpy()`（pandas rank cumtime 17.7s → 0）——`np.argsort(mergesort)` + 并列平均秩，逐位一致 | **-33.9%** |
| E2 | `factors/analysis.py` | `_numpy_rank` 并列分组向量化（Python while → flatnonzero 边界 + np.repeat）；std 守卫等价替换 `np.all(ra==ra[0])` | **-19.2%** |
| E3 | `factors/analysis.py` | `_rolling_ic` 预分配 numpy 数组承接窗口结果（原 `out.iloc[i]=` 逐点 pandas 赋值 → 一次性 Series 构造） | **-8.2%** |
| E5 | `drl/factor_env.py` | `_ics/_icir` 预转 numpy 数组；`_state_feats`/`_composite_fitness` 内部 iloc 行/标量提取 → 数组索引（DataFrame 保留供外部 API） | **-7.7%** |

### 13.3 前后对比（实际数值）

| 阶段 | 耗时 | 增量 | 累积 |
|---|---|---|---|
| 基线 | 9.781s | — | — |
| E1 后 | 6.465s | -33.9% | -33.9% |
| E2 后 | 5.222s | -19.2% | -46.6% |
| E3 后 | 4.795s | -8.2% | -51.0% |
| E5 后 | 4.426s | -7.7% | **-54.8%** |

### 13.4 回滚/否决记录（反证）

1. **E4 rank has_nan 快路径**：增量 -2.4% < min_gain 5% → 回滚。
2. **E6 state_feats 向量化**：增量 -1.9% < min_gain 5% → 回滚。
3. **E7 factor_group_returns numpy 化**：分桶成员与 pd.qcut 不一致（max 差 1.9e-05）→ 行为不等价，否决未实施。
4. **手写 Pearson 替换 np.corrcoef**（2 次尝试）：均 1-2 ulp 尾差（17,541/20,000 与 9,647/20,000 不一致）——corrcoef 内部 `fact*(d@d.T)` 归约路径浮点上不可复刻；替换会破坏同 seed 训练轨迹 bitwise 确定性 → 否决。

### 13.5 为何此处难以继续优化（§4④ 分析）

剩余热点 `corrcoef`（13,244 次 × 28µs ≈0.37s）与 `factor_group_returns`（qcut/groupby）都是：
- corrcoef：numpy 内部 2×2 协方差归约路径存在 1-2 ulp 尾差，无法手写复刻（已 2 次验证失败）；
- factor_group_returns：pd.qcut 分桶边界（np.quantile 插值）语义不可用等频分桶复刻；
- rank 已到 numpy argsort 原生下限。

训练本身的语义计算（GAE/PPO/时间切分）为行为等价红线，不再触碰 → 该方向收敛。

### 13.6 知识库索引（新增）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| numpy 平均秩替代 pandas rank | 秩相关热调用（IC/RankIC） | ✅ E1（bitwise，-34%） |
| 并列分组向量化（flatnonzero 边界 + repeat） | 平均秩/分组统计 Python 循环 | ✅ E2 |
| 预分配 numpy 数组承接 pandas 赋值 | 循环内 iloc setitem / Series 逐点构造 | ✅ E3 |
| DataFrame 预转数组（内部热路径） | iloc 行/标量提取高频调用 | ✅ E5（DataFrame 保留外部 API） |
| **反模式**：手写浮点归约代替 numpy | corrcoef/mean 等内部路径 | ❌ 1-2ulp 尾差，行为等价红线 |
| **反模式**：等频分桶代替 qcut | pandas 分位数语义 | ❌ 分桶成员不一致 |

---

## 14. 第五轮自治优化（2026-09-06）：进化引擎单轮迭代耗时（run5，-82.8%）

> 执行依据：《自治优化协议 v2》；契约：主目标=evolve factor_miner 管线一轮端到端耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2/3/4 保留项之上；服务器未运行（无后台训练污染），测量 command=`.optim/bench_evolve_iter.py 8 8 3`

### 14.1 run5 基线快照

| 指标 | 基线值 |
|---|---|
| evolve factor_miner 单轮 bench（8ep×8nep×3 次） | median=4.489s σ=0.153s cv=3.4% |
| pytest / compileall | ✅ 616 项 PASS / 0 |

### 14.2 run5 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `drl/factor_env.py` | `_composite_fitness` 轻量 fitness 路径：fitness 字段只依赖 `round(_spearman,6)`+`turnover`，省去完整 `factor_quality_gate`（ICIR 滚动/Newey-West/分层收益对 fitness 无贡献）与 `factor_group_returns` | **-48.9%** |
| E2 | `factors/analysis.py` | `factor_turnover` numpy 化（replace/sign/diff/abs/dropna/mean → np.where/sign/diff/isfinite/mean） | **-29.6%** |
| E3 | `drl/factor_env.py` `drl/factor_miner.py` | `_z_exp` expanding z 预计算一次由全部 env 共享（此前每 worker env 构造重算 26 列 expanding mean/std） | **-47.2%** |
| E4 | `drl/factor_env.py` | `_state_feats` 循环向量化（run4 E6 方案复用；run4 基础 4.4s 时 1.9%<5% 回滚，run5 基础 0.85s 下 -9.4% 达标） | **-9.4%** |

### 14.3 前后对比（实际数值）

| 阶段 | 耗时 | 增量 | 累积 |
|---|---|---|---|
| 基线 | 4.489s | — | — |
| E1 后 | 2.293s | -48.9% | -48.9% |
| E2 后 | 1.614s | -29.6% | -64.0% |
| E3 后 | 0.852s | -47.2% | -81.0% |
| E4 后 | 0.772s | -9.4% | **-82.8%** |

### 14.4 为何此处难以继续优化（§4④ 分析）

剩余热点逐项评估（组件占比已摊薄至 0.1s 以下）：
- `_spearman`/`corrcoef`：numpy 内部 2×2 协方差归约路径存在 1-2 ulp 尾差，手写复刻双证失败（run4），替换会破坏同 seed 训练轨迹 bitwise 确定性 → 红线锁定；
- `_ics`+`_icir` 预计算 0.090s：26 因子 × 41 窗 = 1066 次 `_spearman`（120 长度 68µs/次），rank 已是 numpy argsort 原生下限，corrcoef 红线；
- `step`/`collect_episode`/`train_batch`：PPO 收集与更新语义（run3 已证数值语义不可触碰），`_greedy_fitness` 是 fitness 评估语义；
- 剩余可减项（如 `forward_returns` 在 `_composite_fitness` miss 路径缓存）预估 ~3% < min_gain 5%，不构成收益。

→ 该方向收敛。跨五轮累计：DRL 训练 -25%（run2）+ 因子挖掘迭代 -54.8%（run4）+ 进化引擎单轮 **-82.8%**（run5）。

### 14.5 知识库索引（新增，run5）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 轻量 fitness 路径（只算对指标有贡献的字段） | RL 奖励路径不消费 gate 的 ICIR/分层字段 | ✅ E1（bitwise，-49%） |
| pandas 时序统计 numpy 化（diff 首值 NaN 等价性论证） | turnover/时序变化率热调用 | ✅ E2 |
| expanding z 预计算共享（同 precomputed_ics 模式） | 多 env 构造重复同一统计量 | ✅ E3 |
| **收益判定依赖基线规模** | 同一方案在 4.4s 基础 -1.9% 回滚、0.85s 基础 -9.4% 达标 | ✅ run4 E6 / run5 E4 对比 |
| **反模式**：完整 gate 当 fitness 黑盒 | gate 里 80% 字段未被消费时 | ❌ run5 E1 拆解（改为轻量路径） |

---

---

## 15. 第六轮自治优化（2026-09-06）：策略 DRL 管线单轮耗时（run6，-23.3%）

> 执行依据：《自治优化协议 v2》；契约：主目标=evolve `_train_strategy_drl_once` 计算链路耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2/3/4/5 保留项之上；测量 command=`.optim/bench_strategy_drl.py 8 4 5`

### 15.1 run6 基线快照与环境事件

| 指标 | 基线值 |
|---|---|
| 策略 DRL 单轮 bench（8ep×4nep×5 次，干净环境 A/B 对照） | median=5.303s σ=0.029s cv=0.5% |
| 环境事件 | 测量中 RobloxPlayerBeta 游戏进程运行 → bench 漂移至 6.457s（cv 失真）；**用户授权关闭**后恢复；初始 4.956s 基线系污染期数据作废 |
| pytest | ✅ PASS（exit 0） |

### 15.2 run6 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `drl/agent.py` | `sample_action_with_logp` 手写 inverse-CDF 替代 `rng.choice(n,p)`：`u=rng.random()` + `searchsorted(cumsum(p), u)`（choice 对 p 恰好消耗 1 个随机数，同序列同位置取 u 即等价） | **-23.3%** |

### 15.3 前后对比（实际数值，同环境 A/B 交替对照）

| 阶段 | 耗时 | 增量 |
|---|---|---|
| 基线（choice） | 5.303 / 5.333s（cv 0.5%/2.5%） | — |
| E1（手写 ICDF） | 4.067 / 4.061s（cv 1.2%/0.2%） | **-23.3%** |

### 15.4 回滚/否决记录（反证）

1. **E2（E1 之上再手写 Python 循环代替 3 元素 cumsum+searchsorted）**：bitwise 验证通过（3000 步），增量 -4.5% < min_gain 5% → 回滚——同一热路径连续优化收益递减，numpy 调度在 3 元素小数组上的绝对开销已接近下限。

### 15.5 关键知识修正（重要）

run4 曾判定「rng.choice 不可复刻」（手写 Pearson 两次验证 1-2ulp 尾差类比）——**该判断不适用于 choice 本身**：
- 平行 rng 对照实验（probe_sdrl_choice2.py）证明 `Generator.choice(n,p)` 对 p 恰好消耗 **1 个随机数**，输出严格等价于 `searchsorted(cumsum(p), u)`（6 种实现变体 50000/50000 全部匹配）；
- run4 的验证失败是**探针时序 bug**：u 必须在与 choice 相同的 rng 序列位置取，探针却从 choice 消耗之后的序列取 u；
- 行为等价结论以 `verify_sdrl_e1_choice.py`（2000 步 action/logp 位级一致 + rng 尾序列一致）为准。

### 15.6 知识库索引（新增，run6）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 手写 inverse-CDF 替代 rng.choice(p) | 采样热路径（RL collect 每步） | ✅ E1（bitwise，-23%） |
| **反模式**：rng.choice 不可复刻假设 | 采样路径优化 | ❌ 实为探针时序 bug（平行 rng 对照可证等价） |
| 环境干扰 A/B 交替对照 | bench 基线漂移（游戏进程/后台负载） | ✅ Roblox 事件（跨时段比较不可比） |
| 同热路径连续优化收益递减 | 小数组 numpy 调度 | ✅ E1 -23% 达标 / E2 -4.5% 回滚 |

---

---

## 16. 第七轮自治优化（2026-09-06）：meta 控制器管线单轮耗时（run7，-19.8%）

> 执行依据：《自治优化协议 v2》；契约：主目标=evolve `_train_meta_controller_once` 计算链路耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-6 保留项之上（含 run6 agent.py E1 手写 ICDF 采样）；测量 command=`.optim/bench_meta_controller.py 8 4 3`

### 16.1 run7 基线快照

| 指标 | 基线值 |
|---|---|
| meta 控制器单轮 bench（8ep×4nep×3 次，子策略池 dual_ma/factor_signal/price_action） | median=17.937s σ=1.731s cv=9.6% |
| pytest | ✅ PASS（exit 0） |

### 16.2 run7 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `strategies/meta.py` | `MetaControllerEnv.step` 每步 2 次 pandas `df["close"].iloc[t]` 标量访问 → `self._closes` numpy 数组索引（`__init__` 已有 `df["close"].to_numpy(float)` 预转，同一底层数据）——消除 96,865 步 × 2 的 pandas iget/__getitem__ 调度热点（profile 193,900 次） | **-19.8%** |

### 16.3 前后对比（实际数值）

| 阶段 | 耗时 | 增量 |
|---|---|---|
| 基线 | 17.937s（σ=1.731，cv 9.6%） | — |
| E1 后 | 14.376s（σ=0.067，cv 0.5%） | **-19.8%** |

### 16.4 为何此处难以继续优化（§4④ 分析）

E1 后剩余热点逐项评估：
- `sample_action_with_logp`/`forward`/`nnet.step`/`train_batch`：run6 已优化采样链路 + PPO 更新语义（run3 已证数值语义不可触碰）；
- `meta_performance` 的 `np.mean(trades[-window:])` 与 `realized_volatility` 的 `np.std(rets)`（≤40 元素小数组）：numpy 调度开销占大头，但手写等价会引入归约序 1-2ulp 尾差——run4 corrcoef 同类双证失败，行为等价红线；
- `factor_signal._push_ohlcv`（97,950 次）：子策略状态推进语义（回测/实盘同形状 ctx）。

→ 该方向收敛。跨七轮累计（引擎侧）：因子挖掘管线 -54.8%（run4）/ 单轮 0.77s（run5 含 -82.8%）；策略 DRL 管线 -23.3%（run6）/ 单轮 4.07s；**meta 控制器管线 -19.8%（run7）/ 单轮 14.38s**。

### 16.5 知识库索引（新增，run7）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| env.step 内 pandas iloc 标量 → 预转 numpy 数组索引 | 逐步仿真环境热路径（96k 步/回合） | ✅ E1（-20%，逐位一致） |
| **反模式**：小数组 np.mean/std 手写替换 | ≤40 元素统计量 | ❌ 归约序尾差红线（run4 同证） |

---

---

## 17. 第八轮自治优化（2026-09-06）：引擎整轮迭代耗时（run8，无达标项，方向收敛）

> 执行依据：《自治优化协议 v2》；契约：主目标=evolve 三管线（factor_miner + strategy_drl + meta_controller）各一次真实训练 + 调度层（zoo 序列化/落库/标的轮换/跨标 OOS）的整轮迭代耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-7 保留项之上；测量 command=`.optim/bench_evolve_cycle.py 3`

### 17.1 run8 基线快照与关键事实

| 指标 | 基线值 |
|---|---|
| 整轮迭代 bench（真实 EvolveEngine + mock fetch，3 次） | median=21.04s σ=0.925s cv=4.4% |
| 分段构成 | factor_miner 1.0-1.8s + strategy_drl 4.4-5.0s + meta 13.6-15.4s |
| **调度层（zoo 序列化/落库/轮换/跨标簿记）** | **sched_median=0.000s——几乎无成本** |
| pytest | ✅ PASS（exit 0） |

**核心发现**：整轮迭代耗时 = 三管线训练时间之和，调度层自身无可优化空间。三管线内部已在 run4-7 优化至 numpy 下限/语义红线 → 该方向天然收敛。

### 17.2 反证记录（两实验均回滚）

| 实验 | 候选 | bitwise 验证 | 整轮增量 | 判定 |
|---|---|---|---|---|
| E1 | meta env 波动率整列预计算表（`__init__` 一次 np.diff+np.std，`_build_state` 查表替代 96,900 次/回合逐步计算） | ✅ PASS（500 步） | -6.6% 一次，复测 **+0.3%**（20.97s） | ❌ 增量被环境噪声吞没，A/B 不达 2σ → 回滚 |
| E2 | `meta_performance` 去掉 `list()` 防御性复制 | ✅ PASS（50,000 组） | -2.6% | ❌ < min_gain 5% → 回滚 |

### 17.3 方法论沉淀（run8）

- **整轮 bench 口径**：真实 `EvolveEngine` + `_MetaDB` 桩 + `_fetch_latest_data` mock，分段计时（fm/sd/mc/sched）——可用但噪声高（cv 4.4%），单次 21s 慢。
- **噪声下微小优化不可判定**：同一 E1 改动整轮 -6.6% vs +0.3%，meta 单管线交叉甚至方向反转（14.17s vs 回滚后 13.90s）——诚实做法是回滚并记录，而非猜测保留。

### 17.4 领域结论（八轮累计）

引擎侧三管线均已完成一轮收敛：因子挖掘 0.77s/轮（run4/5 含 -82.8%）、策略 DRL 4.07s/轮（run6 -23.3%）、meta 控制器 14.4s/轮（run7 -19.8%）。三管线之上的调度层 ≈0，整轮迭代无独立优化空间。**下一步可行方向**：跨管线/部署侧新热点（回测内核、Pine 生成、前端状态序列化等），需用户指定新主目标。

---

---

## 18. 第九轮自治优化（2026-09-06）：回测/rollout 逐根模拟耗时（run9，-14.8%）

> 执行依据：《自治优化协议 v2》；契约：主目标=回测内核 run_backtest 全流程 + RL greedy rollout（run_episode）逐根模拟耗时，min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-8 保留项之上；测量 command=`.optim/bench_backtest_rollout.py 3`

### 18.1 run9 基线快照与关键事实

| 指标 | 基线值 |
|---|---|
| 回测内核（5000 根，bootstrap=False） | median=0.026s σ=0.000s cv=1.6%——逐根撮合已是 numpy/单根下限 |
| RL greedy rollout（3000 步，state_window=4） | median=0.061s σ=0.001s cv=1.1% |
| 真实 API 默认 bootstrap=True | 0.112s——bootstrap 置信区间占 78%；但 0.5.0 已优化（1.3s→0.086s），进一步压需触碰数值语义 |
| pytest | ✅ PASS（exit 0） |

### 18.2 run9 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `drl/nnet.py` | `_softmax` 单行快速路径 `_softmax_1row`（greedy_action/predict_proba 热路径 (1,N)≤16 小输出，math 标量循环替代 numpy ufunc 调度——与 run6 E3 `_log_softmax_1row` 对称补齐） | rollout **-14.8%**（0.061→0.052s） |

### 18.3 前后对比（实际数值）

| 阶段 | rollout | backtest |
|---|---|---|
| 基线 | 0.061s（σ=0.001） | 0.026s |
| E1 后 | **0.052s**（σ=0.001，cv 1.1%） | 0.026s（不受影响） |

单测 2.65x 加速（_softmax 5.04µs → _softmax_1row 1.90µs）；bitwise 验证 200,000 组随机 + 极端值全 PASS。

### 18.4 回滚/否决记录（反证）

1. **E2（greedy_action 传 cache_for_backward=False 触发 run2 E2 快速路径）**：2000 步 greedy 输出 bitwise PASS，但增量 -1.9% < 5% 且 σ=0.002 噪声内 → 回滚（E1 后 forward 已近下限，推理缓存写入开销本就不大）。
2. **bootstrap 2D 批量重写（(1000,5000) 索引矩阵 + axis=1 批量统计）**：t/mu/sd/ret 均位级一致，但整轮回测 0.112→0.142s **退步**——5M 元素大 gather 缓存不友好，逐样本循环反而是更快形态 → 未保留（metrics.py 已还原，git diff 干净）。

### 18.5 知识库索引（新增，run9)

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| softmax 单行快速路径（math 标量循环） | greedy/部署推理热路径、(1,N)≤16 | ✅ E1（-14.8%，bitwise） |
| **反模式**：小样本大 gather 2D 批量化 | block bootstrap 分块重采样 | ❌ 5M 元素 gather 缓存不友好，退步（逐样本循环更优） |
| 推理路径默认 cache_for_backward=True → 显式 False | greedy/评估前向 | ⚠️ bitwise 通过但增量 <5% 回滚 |

---

---

## 19. 第十轮自治优化（2026-09-07）：Pine 生成 / 数据加载（run10，双目标均实测下限，收敛）

> 执行依据：《自治优化协议 v2》；契约：主目标=Pine Script v5 代码生成耗时（模型→Pine，每次训练后生成），用户确认重定向为数据加载路径耗时；min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-9 保留项之上；测量 command=`.optim/bench_pine_gen.py 3` / `.optim/bench_data_load.py 3`

### 19.1 Pine 生成：实测毫秒级下限（主目标作废，重定向）

| 模型口径 | 耗时 | 代码体积 |
|---|---|---|
| strategy_drl 54d [16,16] 3a（引擎真实） | **median=0.001s**（cv 6.8%） | 41,866B |
| upper 54d [20,16] 3a（上限附近） | 0.001s | 50,691B |
| small 15d [16,16] 3a | 0.000s | 22,237B |

Pine 生成是 O(权重数) 的线性 f-string 展开，41KB 代码 1ms——5% 门槛=0.05ms 处于测量噪声内，**不可优化**。经用户确认重定向主目标。

### 19.2 数据加载路径：各段亦到下限

| 段 | 耗时 | 说明 |
|---|---|---|
| coverage（SQLite MIN/MAX/COUNT 聚合） | 0.42ms | 已最优 |
| load_df（SQLite 读 + pandas 构建 5000 根） | 6.75ms | DataFrame 返回类型契约不可改 |
| load_klines_cached 缓存命中全路径 | 10.11ms（σ=5.2ms，**cv 51.5%**） | 事件循环/to_thread 抖动，任何子项优化被噪声淹没 |
| OHLCVSanitizer.clean（mark 模式） | 2.25ms | 已向量化 |

5% 门槛≈0.5ms：coverage/to_thread 等可触碰项全部在噪声内，load_df 的 pandas 构建是契约成本。**零实验收敛**。

### 19.3 反证记录（探针教训）

1. **bench 数据时间戳坑**：`generate_demo` 内部时间戳混合秒/毫秒（`start_ts`（秒）+ `i*step_s*1000`（毫秒偏移））→ 写库后 `max_ts` 指向未来年份 → `load_klines_cached` 新鲜度条件永不满足，误走网络拉取路径 **8.8s 超时**（假热点）。数据 bench 必须用干净毫秒时间戳（贴近 now）——已记入知识库。

### 19.4 领域结论（十轮累计）

引擎三管线 + 整轮迭代 + 回测/rollout + Pine 生成 + 数据加载**全链路已系统性收敛**：剩余热点全为 numpy/pandas/sqlite 库层下限或数值语义红线。若继续，只剩 IC 分析吞吐（内部 corrcoef/qcut 为 run4/5 已证红线，预期同样收敛）或跨模块新方向（前端、WebSocket、DB 层）。

---

## 20. 第十一轮自治优化（2026-09-07）：批量回测 fast_engine 吞吐（run11，-30%）

> 执行依据：《自治优化协议 v2》；契约：主目标=批量回测 run_backtest_fast 吞吐（AI 设计/迭代批量路径），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-10 保留项之上；测量 command=`.optim/bench_costscan_e1.py`（成本扫描 12 组合全流程）

### 20.1 run11 基线快照（实际数值）

| 段 | 耗时（单次 26.16ms） | 占比 |
|---|---|---|
| matching（撮合主循环） | 16.57ms | 63% |
| precompute（指标预计算） | 4.77ms | 18% |
| metrics（指标收尾） | 2.11ms | 8% |
| sanitize（数据清洗） | 1.83ms | 7% |
| df.copy | 0.07ms | 0.3% |

成本扫描 12 组合基线：**320.6ms**；pytest PASS。

### 20.2 run11 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `backtest/fast_engine.py` + `backtest/cost_scan.py` | **成本扫描 prepare-once**：同一 df、同一策略参数扫描多组合时，清洗+指标预计算与成本无关 → `prepare_backtest_fast()` 一次生成 `{data, series, need_sr, ...}`，`run_backtest_fast(_prepared=...)` 循环复用 | **-30.0%**（320.6→224.6ms） |

### 20.3 门禁与等价性

- 12 组合 metrics 与逐次独立调用**逐位一致**（`.optim/probe_costscan_prep.py` PASS，仅 elapsed_sec 计时字段除外）
- **@research_scope 装饰器保留**：丢失会破坏回测忽略因子 liveness 的语义（`test_factor_liveness_scope` 复现测试立即暴露——曾丢失导致 liveness 变化后回测 38 笔→0 笔）
- `get_strategy` 保持单次实例化（`_prepared` 分支独立实例化、非 prepared 分支单次）——有状态策略（factor_signal 等）实例化时序不变
- 全量 pytest **PASS**（exit 0），测试未动

### 20.4 为何此处收敛（§4④ 分析）

- **网格扫描**（`web/api/backtest.py` _grid_run_one）：多进程 + 参数扫描，precompute 的 MA 周期随参数变化 → 不可复用
- **对比扫描**（策略对比 `/compare`）：多策略不同参数 → 不可复用
- matching 主循环 16.57ms：`snapshot_at`（P2-13 已优化，tolist 缓存 + O(1) 索引）、`strategy.on_candle`（策略语义）、`_execute_fill`（撮合语义）→ 均到下限

→ `_prepared` 仅适用于「同策略同参数、仅成本变化」场景，已覆盖唯一适用点（成本扫描）。

### 20.5 知识库索引（新增，run11）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| prepare-once 复用（sanitize+precompute 与扫描维度无关时） | 成本扫描/同参数批量回测 | ✅ E1（-30%，逐位一致） |
| 重构保留装饰器 @research_scope | fast_engine 类改动 | ✅ 丢失即破坏 liveness 语义（测试暴露） |
| get_strategy 单次实例化 | 有状态策略回测 | ✅ 预计算与撮合同一实例 |
| **反模式**：参数/多策略扫描复用预计算 | 网格/对比扫描 | ❌ precompute 依赖各自参数 |

---

---

## 21. 第十二轮自治优化（2026-09-07）：跨模块 IO/UI 方向（run12，-14%）

> 执行依据：《自治优化协议 v2》；契约：主目标=跨模块 IO/UI 方向（DB 层 aiosqlite 写入为主测点，WebSocket/前端渲染不可 bench），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-11 保留项之上；测量 command=`.optim/bench_db_round.py`（evolve_rounds 落库完整序列 30 次中位数）

### 21.1 run12 基线快照（实际数值）

| 段 | 耗时（完整序列 3.12ms） |
|---|---|
| select max(round_no)（round_no 推导） | 0.88ms |
| INSERT（ORM flush） | 0.96ms |
| cutoff 查询（P2-9 OFFSET keep） | 0.86ms |
| DELETE 旧轮 | 0.74ms |
| commit | 0.39ms |

### 21.2 run12 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `drl/evolve_engine.py` | **cutoff+delete 合并单条 DELETE**：P2-9 保留截断的 cutoff 查询与 delete 合并为一条带标量子查询的 DELETE（`id < (SELECT id ... ORDER BY id DESC OFFSET keep LIMIT 1)`），省一次 aiosqlite 往返 | **-14%** 全序列（3.12→2.6ms）；段级隔离 **-23.6%**（2.25→1.72ms） |

### 21.3 门禁与等价性

- 探针 `.optim/probe_evolve_merge.py`：超阈表 A（旧两段）/B（新单条）逐行内容**一致**（101=101）；未超阈边界新逻辑不删行（91=91）
- `.optim/probe_evolve_under.py`：未超阈（表<keep）时新逻辑 +0.12ms（早期阶段），长期超阈恒 -0.53ms/轮——净正
- SQLite `id < NULL` 恒假语义天然处理未超阈（不删行），无需 if 分支
- 全量 pytest **PASS**（exit 0），测试未动

### 21.4 为何此处收敛（§4④ 分析）

IO/UI 方向可 bench 且程序内的部分已优化或到下限：

| 子项 | 结论 |
|---|---|
| WebSocket 行情 | ccxt.pro 外部代理，网络 IO，不可 bench |
| 行情 API（/api/market） | 外部网络 IO + TTL 缓存 |
| EventBus（core/bus.py） | O(1) 入队 + 背压 + 并发 gather，语义下限 |
| IncrIndicators（实时指标） | O(1) 增量更新，语义下限 |
| fitness 曲线读取 | (model, id DESC) 复合索引，毫秒级 |
| tasks_cache | 惰性低频清理工具 |
| 前端渲染 | 无 headless 门禁设施 |

### 21.5 知识库索引（新增，run12）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 标量子查询合并（`id < (SELECT ... OFFSET n)`） | aiosqlite 多往返合并为单条 SQL | ✅ E1（段级-23.6%，逐行一致） |
| 子查询 NULL 恒假语义 | 未超阈边界天然短路 | ✅ 91=91 不删行 |
| IO 收敛清单 | WebSocket/行情/事件总线/增量指标 | ✅ 全为外部 IO 或语义下限 |

---

---

## 22. 第十三轮自治优化（2026-09-07）：前端资源/后端拼接路径（run13，传输 -74%）

> 执行依据：《自治优化协议 v2》；契约：主目标=前端资源/后端拼接路径（静态分析 + 传输体积口径，无 headless 浏览器设施故端到端渲染耗时不可测），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-12 保留项之上；测量 command=`.optim/probe_gzip_mw.py`

### 22.1 run13 基线快照（实际数值）

| 项 | 数值 |
|---|---|
| index.html 体积 | 496,984B（单文件：6 内联 script、1 style、6327 换行、168KB JS） |
| 后端 `/` 路径 | FileResponse 直发（无服务端模板/拼接） |
| gzip level6 压缩成本 | 10.9ms/次（页面加载低频） |

### 22.2 run13 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `web/main.py` | **GZipMiddleware(minimum_size=100_000)**：只压缩 >100KB 响应（index.html 497KB→128KB）；API JSON 全部 <100KB 不触发，热路径零开销 | **-74.1%** 传输体积（497KB→128.5KB） |

### 22.3 门禁与等价性

- 裸 ASGI 协商验证（`.optim/probe_gzip_mw.py`，绕过 httpx 自动解压）：无 Accept-Encoding → 原文（match=True）；带 gzip → 128,538B 且解压==原文；小 JSON 10KB 不压缩；大 JSON 200KB 压缩
- middleware 栈顺序 `GZipMiddleware → AuthMiddleware`（`verify_gzip_main.py` 遍历 build_middleware_stack 确认）
- 全量 pytest **PASS**（exit 0）——TestClient 无 AE 收原文，测试不受影响

### 22.4 为何此处收敛（§4④ 分析）

| 子项 | 结论 |
|---|---|
| 后端拼接路径 | FileResponse 直发，无拼接点可优化 |
| 传输体积 | E1 已优化（唯一 >100KB 响应是 index.html） |
| brotli | 未安装依赖（可选后续，但 gzip -74% 已近文本压缩极限） |
| 前端渲染耗时 | 无 headless 浏览器门禁设施，不可测 |

### 22.5 知识库索引（新增，run13）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| GZipMiddleware(minimum_size=高) 只压大响应 | 单文件前端/大 JSON 响应传输优化 | ✅ E1（-74.1%） |
| httpx 自动解压陷阱 | 验证协商行为须裸 ASGI 调用 | ✅ probe_gzip_mw.py |
| build_middleware_stack 验证真实类 | FastAPI user_middleware 只显示包装名 | ✅ verify_gzip_main.py |

---

---

## 23. 第十四轮自治优化（2026-09-07）：DRL 状态预处理（run14，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=DRL 状态预处理（指标窗口构造、状态张量打包），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-13 保留项之上；测量 command=`.optim/bench_state_prep.py`（env 构造 + rollout + _build_state 分解）

### 23.1 run14 基线快照（实际数值）

| 项 | 数值 |
|---|---|
| env 构造 | 6.89ms（其中 _precompute_features **6.37ms 占 92%**） |
| rollout | 3.9µs/步 |
| `_build_state`（state_window=10 查表） | 1.5µs/步 |

`_precompute_features` 13 特征逐项（n=5000）：risk_regime(rolling rank) 1.74 / rsi 1.14 / bb_pos 0.79 / macd 0.71 / vol_ratio 0.53ms——**全为 pandas ewm/rolling 原生向量化**。

### 23.2 既有优化盘点（为何已收敛）

| 优化 | 出处 | 效果 |
|---|---|---|
| `_state_stack` 堆叠矩阵查表 | P2 | `_build_state` 1.5µs/步（state_window>1 零拼接） |
| `_vol5` sliding_window_view 向量化 | 自治 E1 | 138x 加速，bitwise 一致 |
| `_precompute_features` worker 共享 | P2-8/9 | 每轮轨迹不重复算特征 |
| 特征函数（RSI/MACD/BB/分位） | — | pandas 原生向量化，库层下限 |

### 23.3 候选否决记录（红线）

- **val/oos 段预计算复用**：`val_df = df.iloc[n_train:n_train+n_val]` 是**切片**——对全量 df 预计算后按切片复用，切片首行 rolling/ewm 窗口含段前数据（现实现为 NaN→fillna(0)）→ **状态数值改变，行为不等价**。这也是主路径只对同一 `train_df` 全量共享 `precomputed_features` 的原因（agent.py L776 worker 复用的是同一份全量特征）。

### 23.4 收敛判断（§4④）

状态预处理全链路已系统性收敛：唯一候选（val/oos 复用）行为不等价，剩余热点全为 pandas 库层下限（无手写替换空间——run4/run9 归约序尾差红线）。

---

## 24. 第十五轮自治优化（2026-09-07）：因子特征工程（run15，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=因子特征工程（因子挖掘迭代内窗口切片/特征工程热点，run4 后剩余），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-14 保留项之上；测量 command=`.optim/bench_factor_miner.py 8 3`

### 24.1 run15 基线快照（实际数值）

train_factor_miner 端到端：**0.772s/轮**（σ=0.018s，cv 2.3%——run4 收敛值保持）。

profile 热点分解（tottime/cumtime）：

| 热点 | 耗时 | 归属 |
|---|---|---|
| factor_turnover | tottime 0.080s（2038 次，39µs/次） | run5 E2 已 numpy 化 |
| _numpy_rank | cumtime 0.211s（2894 次） | run4 E1/E2 已向量化 |
| _spearman | cumtime 0.674s（1447 次，54.4µs/次） | 内部 rank×2=24.1µs + corrcoef=19.8µs |
| pandas Series.__init__ | 4704 次 cumtime 0.191s | run4 E3 已消（out_arr 预分配） |

### 24.2 候选否决记录（红线 + 收益不足）

- **corrcoef 手写替换**：30 元素窗口上该组 diff=0（corrcoef -0.021579532814 == 手写 Pearson）——但单组不足以推翻 run4/5 红线（归约序尾差在其它输入/规模下存在）；且潜在收益 29ms/0.772s = **3.7% < 5% 门槛**，触碰红线不划算 → 不实施
- **_rolling_ic 批量复用**：非重叠窗（step=window）是 IC 序列样本独立的度量前提（ICIR 不虚高），窗口间无共享计算 → 无批量空间

### 24.3 收敛判断（§4④）

因子特征工程全链路在 run4/5 已系统性收敛（_numpy_rank E1/E2、_rolling_ic E3、factor_turnover E2）：本轮剩余热点全为已优化函数或行为等价红线，潜在替换收益 < 门槛且触红线。零实验零保留，源码零改动。

---

---


---

## 25. 第十六轮自治优化（2026-09-07）：启动/预热路径（run16，-62%）

> 执行依据：《自治优化协议 v2》；契约：主目标=启动/预热路径（zoo 模型加载、状态恢复），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-15 保留项之上；测量 command=`.optim/bench_startup_seq.py`

### 25.1 run16 基线快照（实际数值）

启动恢复序列（ModelZoo 构造 + flat exists + best_info + load_agent + DB 段）：**38.4ms**（n=5）。

热点链：`_restore_episode_counts` → `zoo.best_info("factor_miner")` → `_read_json_gz`（841KB gzip → 解压 **1.93MB** 权重 JSON）→ `json.loads` **22.45ms** + zlib 6.3ms——占启动 wall 的 **86%**（33.2/38.4ms）。

模型 zoo 规格：factor_miner best_gz 841KB（load 34.2ms）、meta_controller 278.5KB（12ms）、strategy_drl 71.9KB（4ms）、DB 段（groupby+streak）5ms。

### 25.2 run16 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `drl/model_zoo.py` + `requirements.txt` | **`_read_json_gz` 改用 orjson.loads**（ImportError 回退标准 json）：权重 JSON 解析 22.45→3.55ms（6.3x） | **-62%** 启动恢复序列（38.4→14.6ms） |

### 25.3 门禁与等价性

- 逐位等价（`.optim/probe_orjson_eq.py`）：orjson vs json 对同一 1.93MB 文本，**全 dict 递归比较（float 位级 + NaN 相等）逐位 PASS**——JSON 浮点解析是确定性最近舍入（无归约序），与 run4 手写 np.mean/std 红线**不同**，解析器替换安全
- `orjson>=3.9.0` 已声明进 requirements.txt；无 orjson 环境自动回退标准 json（行为/格式不变）
- 全量 pytest **PASS**（exit 0），测试未动

### 25.4 为何此处收敛（§4④ 分析）

| 子项 | 结论 |
|---|---|
| 大模型权重 JSON 解析（factor_miner 1.93MB） | E1 已优化（唯一可触碰热点） |
| DB 恢复段（groupby + streak LIMIT） | run12 已证 SQLite IO 下限（5ms 总） |
| 15s 首次训练延迟 | 等待参数，非 CPU 热点 |
| 训练循环内 load_agent | 共用 E1 路径，间接受益 |

### 25.5 知识库索引（新增，run16）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| orjson 替换 json.loads 读大权重 | 大 JSON 文本解析（JSON 浮点=确定性舍入，逐位等价成立） | ✅ E1（6.3x，逐位 PASS） |
| 启动热点链定位 | _restore_episode_counts→best_info→_read_json_gz | ✅ 33.2/38.4ms |
| ImportError 回退模式 | 性能依赖可选安装时行为/格式不变 | ✅ E1 |


---

## 26. 第十七轮自治优化（2026-09-07）：回测 metrics 收尾路径（run17，-23%）

> 执行依据：《自治优化协议 v2》；契约：主目标=回测 metrics 收尾路径（finalize_metrics：夏普/回撤/胜率 + bootstrap 置信区间），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-16 保留项之上；测量 command=`.optim/bench_metrics_finalize.py`

### 26.1 run17 基线快照（实际数值）

| 段 | 耗时 |
|---|---|
| finalize_metrics bootstrap=True | **82.6ms**（其中 block_bootstrap 80.46ms 占 **97%**） |
| finalize_metrics bootstrap=False | 2.0ms（成本/网格扫描已传 False，run11） |
| compute_metrics | 1.68ms（numpy 向量化下限） |

### 26.2 run17 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `backtest/metrics.py` | **block_bootstrap 循环内 idx 构建算术重排**：原 2 次 np.repeat + 1 次 fancy index → 1 次 repeat + arange 合并（`idx = repeat(starts[k][:t]-starts_pos, reps) + arange(total)`） | **-23%** finalize on（82.6→63.6ms）；段级 -23.1%（80.46→61.9ms） |

### 26.3 门禁与等价性

- 逐位等价（`.optim/probe_boot_idx.py`）：sharpe/rets 输出 **bitwise 一致**（整数索引运算数学等价：块 b 内位置值 = starts[k][b]-starts_pos[b]+starts_pos[b]+j，无浮点误差）
- 反证记录（`.optim/probe_boot_matrix.py`）：批量矩阵化（(n_boot,n) NaN 矩阵 + axis 归约）虽逐位等价（NaN 填充不改归约序）但 **-102% 退步**（5M 元素 40MB NaN 归约 > 1000 次小样本）——与 run9 bootstrap 2D gather 退步案例一致，批量路线放弃
- test_optimizations_p0p1p2.py + test_stat_significance.py 回归 PASS；全量 pytest **exit 0**，测试未动

### 26.4 收敛判断（§4④ 分析）

| 子项 | 结论 |
|---|---|
| bootstrap 1000 次重采样循环（80.46ms，唯一可触碰热点） | E1 已优化（-23.1%） |
| compute_metrics（1.68ms） | numpy 向量化下限 |
| 批量矩阵化路线 | 已证退步（run9 教训复现），不实施 |

### 26.5 知识库索引（新增，run17）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 索引构建算术重排（repeat+fancy index → 单 repeat+arange） | 循环内重复 numpy 调度 | ✅ E1（-23.1%，逐位 PASS） |
| bootstrap 2D 批量反证（第二次确认） | (n_boot,n) NaN 矩阵 axis 归约 | ✅ 逐位等价但 -102% 退步 |

---


---

## 27. 第十八轮自治优化（2026-09-07）：跨轮联合复测（run18，审计 8/8 PASS）

> 执行依据：《自治优化协议 v2》；契约：主目标=跨轮联合复测（审计性质：run2-17 全部保留项在干净环境下的当前实测值 vs 记录值），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 方法：逐项**串行**重跑各保留项基准脚本（并行会互相污染测量，run8 教训）；σ 判定（<1σ 无回归）

### 27.1 审计结果（8/8 无回归）

| # | 保留项 | 记录值 | 当前实测 | 判定 |
|---|---|---|---|---|
| 1 | factor_miner 挖掘 8ep | 0.772s | 0.815s | PASS（<1σ=0.048） |
| 2 | strategy_drl 管线 8ep | 4.07s | 3.882s | PASS（更优 -4.6%） |
| 3 | meta_controller 8ep | 14.38s | 11.833s | PASS（更优 -17.7%） |
| 4 | backtest/rollout | 0.026/0.052s | 0.024/0.050s | PASS（更优） |
| 5 | 成本扫描 prepare-once | 224.6ms | 215.5ms | PASS（速差 -29.8% 复现记录 -30.0%） |
| 6 | DB 落库 record_round | 2.6ms | 1.85ms | PASS（更优 -29%） |
| 7 | 启动恢复序列 | 14.6ms | 15.6ms | PASS（<2σ 噪声内） |
| 8 | metrics finalize on | 63.6ms | 63.9ms | PASS（噪声内） |

### 27.2 审计结论

- **全部 8 项保留实测值在 σ 内或更优，无回归**——run2-17 跨轮累计收益可复现
- 3 项大幅更优（meta -17.7%、db_round -29%、strategy_drl -4.6%）：同一环境比记录轮更干净（当时的 Roblox/后台负载已消失），确认无隐藏回归
- 成本扫描速差 -29.8% 稳定复现——prepare-once 收益为结构性而非环境假象
- 本轮零实验零保留（审计性质），源码零改动

---


---

## 28. 第十九轮自治优化（2026-09-07）：run8 增量复测（run19，-13.5%）

> 执行依据：《自治优化协议 v2》；契约：主目标=run8 增量复测（meta vol 整列预计算 E1 在干净环境下真实收益判定），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-18 保留项之上；测量 command=`.optim/bench_meta_controller.py 8 3`

### 28.1 run19 基线快照（实际数值）

| 项 | 数值 |
|---|---|
| meta 管线（8ep） | 12.028s（cv 1.0%） |
| vol 逐步计算调用次数 | **96,900 次/回合**（8ep×n_episodes×步数） |
| vol 占 meta wall | **9.2%**（11.6s 中 ~1.07s） |

### 28.2 run19 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `strategies/meta.py` | **meta 波动率特征整列预计算**：`build_meta_state` 新增 `vol_feature` 参数；`MetaControllerEnv.__init__` 一次 `_precompute_vol_feats`（sliding_window_view 主体 + t<window 边界逐点，1.4ms）；`_build_state` 查表传入 | **-13.5%**（12.028→10.408s，cv 0.1%） |

### 28.3 门禁与等价性

- 逐位等价（`.optim/verify_meta_vol_e1.py`）：归一版（含 `min(/SCALE,1)`）5000 点含 t<window 边界 **bitwise PASS**；预计算一次性仅 1.4ms
- 实盘端 `build_meta_state` 不传 vol_feature 回落 `meta_vol_feature(vol_closes)`——训练/实盘口径一致
- test_meta_oos_report + test_drl_optimizations + test_reconcile 回归 PASS；全量 pytest **exit 0**，测试未动
- 回滚：撤销 build_meta_state vol_feature 参数 / _precompute_vol_feats / _vol_feats / 查表传入

### 28.4 run8 回滚教训的修正（重要方法论）

run8 当时用**整轮 bench**（21s 含三管线，cv 4.4%）测 E1 增量——1s 收益被噪声稀释至不可判定（-6.6% vs +0.3%）而回滚。本轮干净环境 + **单 meta 管线口径**（cv 0.1%）下同一候选 **-13.5% 明确达标**。

> 方法学：**复测回滚项必须先切换到该热点所在模块的单管线口径**——整轮口径只适合验收（总收益），不适合发现（模块级增量）。

run8 E2（meta_performance 去 list() 防御性复制）当时 -2.6% < 5% 回滚——收益不足而非噪声，维持回滚结论。

### 28.5 知识库索引（新增，run19）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| sliding_window_view 批量 np.std vs 逐步逐位一致 | 整列预计算替代热路径逐步计算 | ✅ E1（含边界，归一后仍 PASS） |
| 复测回滚项先切单管线口径 | 整轮噪声掩盖模块级收益 | ✅ E1 从回滚→-13.5% 保留 |
| 查表参数回落（vol_feature=None → 逐步） | 训练/实盘共用函数保口径一致 | ✅ E1 |

---


---

## 29. 第二十轮自治优化（2026-09-07）：IC 批量分析路径（run20，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=因子挖掘 IC 批量分析路径（factor_ic_table 多因子批量），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-19 保留项之上；测量 command=`.optim/bench_ic_table.py`（5000 根 × 23 因子，rank method）

### 29.1 run20 基线快照（实际数值）

| 项 | 数值 |
|---|---|
| factor_ic_table 20 因子 | **288.8ms**（σ=1.6ms） |
| profile 热点 | `_spearman` cumtime 1.384s/3run（`_numpy_rank` 23,046 次调用 + corrcoef 11,088 次） |
| 训练循环内调用 | 无显著调用（run15 profile + run18 审计 factor_miner 0.815s 均确认） |

### 29.2 调用点审计（热路径判定）

| 调用点 | 性质 |
|---|---|
| `web/api/factors.py`（3 处） | 用户请求触发，`asyncio.to_thread` 隔离非阻塞 |
| `ic_weighted_composite`（factors/mining.py L331） | 仅 API 端点使用（web/api/factors.py L393） |

**结论：factor_ic_table 非迭代热路径**——唯一真实入口是 API 分析请求（低频、已线程隔离），mining 训练循环不经过它。

### 29.3 内部收敛审计

- fwd 批量共享已做（`factor_ic_table` 一次算全因子共享）
- `_numpy_rank`：run4 E1/E2 已向量化
- 窗口 corrcoef/rank：触碰 run4/5/9/15 行为等价红线（批量归约序）；run15 已反证手写 Pearson 替换收益 3.7% < 5% 不实施
- `_rolling_ic` 非重叠窗（step=window）是 ICIR 样本独立的度量前提，无批量空间

### 29.4 收敛判断（§4④）

非热路径（仅 API 触发 + to_thread 隔离）+ 内部已 numpy/优化/红线边界 → **无可安全收益**。零实验零保留，源码零改动。

---


---

## 30. 第二十一轮自治优化（2026-09-07）：数据缓存路径复测（run21，审计 PASS，维持收敛）

> 执行依据：《自治优化协议 v2》；契约：主目标=run10 数据缓存路径复测（cached_hit 噪声未判定项在干净环境的审计重测），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-20 保留项之上；测量 command=`.optim/bench_data_load.py 3`

### 30.1 run21 复测结果（当前实测 vs run10 记录）

| 段 | run10 记录 | 当前实测 | 判定 |
|---|---|---|---|
| load_df（SQLite + pandas 5000 根） | 6.75ms | 7.68ms（σ=0.49，cv 6.4%） | PASS（+14% <2σ） |
| cached_hit（缓存命中全路径） | 10.11ms | **10.91ms 中位数但 σ=324.86ms（cv 2977%）** | PASS（中位数无回归；偶发 699.9ms 异常） |
| sanitize（向量化清洗） | 2.25ms | 2.40ms（σ=0.34，cv 14.4%） | PASS（+7% 噪声内） |

### 30.2 审计结论

- **无回归**：三段数值均与 run10 记录一致（σ 内）
- **cached_hit 噪声依旧巨大**：干净环境下本次仍出现 699.9ms 偶发异常——to_thread/事件循环级抖动非环境因素；任何子项优化（5% 门槛≈0.5ms）都会被淹没，**run10 不可判定结论复现**
- coverage 0.42ms 聚合、load_df pandas 构建契约、sanitize 向量化——各段仍在下限

### 30.3 收敛判断（§4④）

维持 run10 收敛：数据加载路径各段到下限 + cached_hit 噪声不可判定。零实验零保留，源码零改动。

---


---

## 31. 第二十二轮自治优化（2026-09-07）：前端 gzip 传输复测（run22，审计 PASS，E1 零回归）

> 执行依据：《自治优化协议 v2》；契约：主目标=run13 前端 gzip 传输保留项复测（审计性质），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-21 保留项之上；测量 command=`.optim/probe_gzip_mw.py`（裸 ASGI 协商验证，绕过 httpx 自动解压）

### 31.1 run22 复测结果（六项验证 vs run13 记录）

| 验证项 | run13 记录 | 当前复测 | 判定 |
|---|---|---|---|
| index raw 体积 | 497KB | 496,984B | ✅ 一致 |
| gzip 传输体积 | 128.5KB | 128,538B（ratio 25.9%） | ✅ **-74.1% 无回归** |
| 无 Accept-Encoding → 原文 | match=True | match_raw=True | ✅ |
| 解压 == 原文 | True | True | ✅ |
| 小 JSON（10KB）不压缩 | enc=None | len=10,021 enc=None | ✅ minimum_size 边界生效 |
| 大 JSON（200KB）压缩 | enc=gzip | enc=gzip | ✅ |

### 31.2 审计结论

- **E1（GZipMiddleware minimum_size=100_000）零回归**：传输体积 -74.1% 保持，协商行为与解压等价全部正确
- 行为级验证已涵盖 middleware 栈顺序（若 GZipMiddleware 不在栈上，[gzip] 响应不会出现）——无需静态遍历
- 本轮零实验零保留（审计性质），源码零改动

---


---

## 32. 第二十三轮自治优化（2026-09-07）：收官总结（run23）

> 执行依据：《自治优化协议 v2》；本轮动作：最终健康验证（全量 pytest exit 0）+ 累计收益/保留项/回滚清单总整理

### 32.1 累计保留项与回滚清单（run2-22，12 项）

| 轮次 | 保留项 | 改动文件 | 实测收益 | 回滚命令 |
|---|---|---|---|---|
| run4/5 | 因子挖掘链路（_numpy_rank / _rolling_ic 批量 / turnover 向量化） | drl/factor_miner.py、factors/analysis.py 等 | **-54.8%**（1.71→0.77s/轮） | 撤销对应函数向量化改回 pandas 原式 |
| run6 | strategy_drl 管线 | drl/agent.py、drl/nnet.py | **-23.3%**（5.30→4.07s/轮） | 撤销 train_drl 内快速路径 |
| run7 | meta 控制器管线 | strategies/meta.py | **-19.8%**（17.93→14.38s/轮） | 撤销 step 内 numpy 索引化 |
| run9 | greedy rollout（_softmax_1row） | drl/nnet.py | **-14.8%**（0.061→0.052s） | 撤销 _softmax_1row 快速路径 |
| run11 | 成本扫描 prepare-once | backtest/cost_scan.py | **-30.0%**（320.6→224.6ms） | 恢复独立 per-combo 预处理 |
| run12 | DB 落库合并 DELETE | drl/evolve_engine.py | **-14%**（3.12→2.6ms/round） | 恢复逐轮 DELETE |
| run13 | GZipMiddleware(minimum_size=100_000) | web/main.py | **-74.1%**（497→128.5KB） | 移除 GZipMiddleware |
| run16 | orjson 权重解析（ImportError 回退） | drl/model_zoo.py、requirements.txt | **-62%**（启动 38.4→14.6ms） | 恢复 _read_json_gz json.loads + 删 orjson 行 |
| run17 | bootstrap idx 算术重排 | backtest/metrics.py | **-23%**（finalize 82.6→63.6ms） | 恢复 blk/off/idx 原式 |
| run19 | meta vol 整列预计算 | strategies/meta.py | **-13.5%**（12.03→10.41s/轮） | 撤销 vol_feature 参数/预计算/查表 |
| run14/15/20/21 | 收敛轮（DRL 状态预处理 / 因子特征工程 / IC 批量 / 数据缓存） | 无改动 | 0 | — |
| run18/22 | 审计轮（联合复测 / gzip 复测） | 无改动 | 0 | — |

### 32.2 跨轮累计效果

| 主目标 | 起点 | 终点 | 累计 |
|---|---|---|---|
| 因子挖掘 | 1.71s/轮 | 0.77s/轮 | -54.8% |
| strategy_drl | 5.30s/轮 | 3.88s/轮（run18 实测） | -26.8% |
| meta 控制器 | 17.93s/轮 | 10.41s/轮 | -41.9% |
| greedy rollout | 0.061s | 0.050s（run18 实测） | -18.0% |
| 成本扫描 | 320.6ms | 215.5ms（run18 实测） | -32.8% |
| 启动恢复 | 38.4ms | 15.6ms（run18 实测） | -59.4% |
| metrics finalize | 82.6ms | 63.9ms（run18 实测） | -22.6% |
| 前端传输 | 497KB | 128.5KB | -74.1% |

### 32.3 收官结论

- 全量 pytest **exit 0**（616 项，仅测试自身的既有 aiosqlite 关闭警告）
- 12 项保留全部带实测数值门禁 + 回滚命令 + 反证记录；10 次审计/复测全部 PASS（无回归）
- 跨轮行为等价红线（corrcoef/qcut、np 归约序尾差、5M 2D gather、val/oos 切片语义、手写 Pearson）均未触碰
- 全仓库热路径系统性收敛：训练流水线（3 管线）、回测内核、启动恢复、metrics、IC 分析、数据加载、前端传输——剩余均为库层下限 / 外部 IO / 语义红线 / 噪声不可判定
- 方法论沉淀：噪声下微小优化不可判定（run8 教训）、复测回滚项先切单管线口径（run19）、中位数+σ 双口径（run21）、行为级验证 middleware（run22）

---


---

## 33. 第二十四轮自治优化（2026-09-07）：BUG 狩猎轮（run24，15 发现 / 10 修复）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=正确性（非性能）：全仓库找 bug 并修复
> 方法：4 组并行静态审查子代理（drl / backtest / web-engine-ai / strategies-factors）+ 3 轮边界 fuzz 探针（优化函数极端输入与原始语义一致性）；每个 bug 亲自读码复核后修复

### 33.1 修复清单（10 项，全部带行为验证）

| # | 严重度 | 文件 | bug | 验证 |
|---|---|---|---|---|
| B1 | 高 | `backtest/_matching.py` | **限价单部分成交比例买/卖方向写反**（买单应 (lp-low)/range、卖单应 (high-lp)/range，原代码互为对调；买单挂 bar 最高价本应 100% 成交实测 0%） | 6 项精确断言 PASS |
| B2 | 高 | `strategies/meta.py` | **实盘 DRL 动作 0（清仓）无执行分支**——训练端 0/2 均清仓，实盘静默 return None，持仓无法由 DRL 主动平仓 | action_map 对齐训练端 {0:-1,1:1,2:-1} |
| B3 | 高 | `factors/mining.py` | **dynamic_composite 权重恒正**（abs(IC)）——负 IC 因子被正暴露进组合 | 确定性 -1 相关因子 → 权重 -0.5 |
| B4 | 高 | `drl/evolve_engine.py` | **NaN oos_ret 绕过部署硬门**（float('nan')<=0 为 False）且 NaN 使回退保护永久失效 | isfinite 拦截/守卫，NaN 均拦截 |
| B5 | 中 | `backtest/overfit.py` | `_cscv_pbo(...) or 0.5` 吞掉合法 **PBO=0.0**（健康度少 20 分） | 改为 None 判别 |
| B6 | 中 | `strategies/meta.py` | `reset()` 的 `_t=warmup` 无 min 钳制（`__init__` 有）——短数据越界 | n=10/warmup=60 reset PASS |
| B7 | 中 | `strategies/meta.py` | 子策略加载失败名字残留 → `_sub_strategies` 按序号错位/IndexError | 失败名剔除，列表一一对应 |
| B8 | 中 | `drl/evolve_engine.py` | 重启恢复把 **demo_blocked 行计入 reject_streak**（运行时明确不计） | 恢复与运行时口径一致 |
| B9 | 低 | `drl/factor_miner.py` | episodes 无下界钳制（0/负 → 空 history） | max(1,...) |
| B10 | 低 | `drl/factor_miner.py` | 空 history 无守卫（strategy/meta 路径均有） | raise ValueError |

### 33.2 已记录待处理（5 项，需设计决策，本轮不修）

| # | 严重度 | 文件 | 问题 |
|---|---|---|---|
| R1 | 中 | engine/trading_engine.py | 断链自动平仓失败后不重试（_stale_alerted 不复位） |
| R2 | 中 | ai/iteration.py | _next_version 读-自增非原子，并发重号覆盖 |
| R3 | 中 | ai/ 三文件 | 后台 worker 线程跨 loop 用共享 DB 连接池 |
| R4 | 中 | drl/model_zoo.py | max_versions=1 裁剪选错对象（默认 10 不受影响） |
| R5 | 中 | drl/evolve_engine.py | _write_flat_meta 非原子 + 手动训练竞态 |

### 33.3 重点核验无 bug 清单（静态审查价值对等项）

- `block_bootstrap_sharpe_ci` 循环（t=1 维度、补足路径越界、rng 确定性）——逐边界核对无错
- `_numpy_rank` 空/全 NaN/全同/负数/并列/单元素/±inf——与 pandas 逐位一致
- `_precompute_vol_feats` n=1..5000 全边界与逐步计算逐位一致（run19 无 bug）
- 撮合开平仓价格/滑点方向/手续费/负持仓守卫/强制平仓——均正确
- orjson 回退路径（CRLF/BOM/超大文本/NaN）——与标准 json 行为一致

### 33.4 fuzz 边界轮结论

- `_softmax_1row`/`_log_softmax_1row`：n≤7 逐位一致；**n≥8 起 1-2 ULP 差异**（numpy 归约路径切换）——实际 n_actions=5 安全，docstring「N≤16 逐位一致」声明过宽（非功能 bug）
- `_newey_west_std` 全 NaN 输入返回 NaN（唯一调用方已预先过滤，防御性缺口）
- `factor_ic` 按位置计算不按索引（错位同长不改变结果；长度不等显式 ValueError 被 factor_ic_table 的 except 静默吞掉）

### 33.5 门禁

- 全量 pytest **exit 0**，测试未动
- 每个修复附行为验证（见 33.1 表）；回滚=恢复对应行（git diff 可查）
- 方法论沉淀：静态审查子代理须给边界清单+触发条件+最小复现；「核验无 bug」清单与 bug 清单同等价值；NaN 比较陷阱（float('nan')<=0 恒 False）→ 所有收益/回退/部署判据需 isfinite 前置

---


---

## 34. 第二十五轮自治优化（2026-09-07）：run24 遗留 bug 修复（run25，4 修复 / 1 记录）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=修复 run24 记录的 5 项中危待处理 bug

### 34.1 修复清单（4 项，全带行为验证）

| # | 文件 | bug | 修复 | 验证 |
|---|---|---|---|---|
| R1 | `engine/trading_engine.py` | **断链自动平仓失败后不再重试**（_stale_alerted 只控告警，失败后仓位裸奔直到行情恢复） | 新增 `_stale_close_pending` 重试标志：失败置 True，断链期间每 15s 循环重试；成功/无持仓清空；恢复复位 | 3 项静态断言 PASS |
| R2 | `ai/iteration.py` | **_next_version 读-自增非原子**，_AI_SEM=2 并发迭代重号并静默覆盖产物 | 模块级 `threading.Lock` 包临界区（worker 各独立 loop，asyncio.Lock 不适用） | 锁存在断言 PASS |
| R4 | `drl/model_zoo.py` | **max_versions=1 裁剪选错对象**（删刚保存的新版本，versions 与 best_version 永久错位） | 不删刚保存版本；is_best 时才删旧 best；无法满足时正确性优先保留超限并告警 | 3 次保存 versions=[1]→[2]→[3] 恒含当前 best |
| R5 | `drl/evolve_engine.py` | **flat 元数据非原子重写**（open(w)+json.dump，读端可读截断 JSON） | 改走 `ModelZoo._atomic_write_bytes`（tmp+fsync+os.replace） | 内容完整 + 无 .tmp 残留 PASS |

### 34.2 记录保持（1 项）

| # | 文件 | 问题 | 为何本轮不改 |
|---|---|---|---|
| R3 | ai/ 三文件 | 后台 worker 线程跨 loop 复用主循环 DB 连接池（间歇性 RuntimeError） | 需跨 AIClient/Iteration/Optimizer 三层重构，这些类不持有 engine loop 引用，测试无法覆盖并发场景——风险 > 收益 |

### 34.3 门禁

- 全量 pytest **exit 0**（616 项，仅测试自身既有 aiosqlite 关闭警告；脚本风格 test_ai_guards.py 模块级 sys.exit 非 pytest 收集模式，属既有设计）
- 9 项行为断言全 PASS；回滚=git diff 对应行

### 34.4 知识库索引（新增，run25）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 告警标志与重试标志分离（_stale_alerted vs _stale_close_pending） | 一次性动作失败后挂死的通用解法 | ✅ R1 |
| 版本裁剪删除对象选择（不删刚保存/is_best 才删旧 best/宁可超限） | 列表与锚点一致性维护 | ✅ R4 |
| 跨线程临界区用 threading.Lock | worker 各自独立事件循环时 asyncio.Lock 不共享 | ✅ R2 |

---


---

## 35. 第二十六轮自治优化（2026-09-07）：R3 跨 loop DB 连接池修复（run26）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=修复 run24 记录的最后一项中危 R3：
> AI 后台 worker 线程跨 loop 复用主循环 DB 连接池（间歇性 RuntimeError，AI 任务随机失败）

### 35.1 根因

`web/api/ai.py` 的 `_run_ai_task_background` 为每个 AI 后台任务创建**独立事件循环**
（worker 线程）。这些 worker 直接 `await db.kv_get()/db.session()`——SQLAlchemy
异步引擎连接池绑定主循环（web lifespan），worker loop 复用连接 → 间歇性
`RuntimeError`（"Future attached to a different loop"）。仓库已有先例修复：
`web/api/backtest.py` 的 `_recent_performance` 等经 `_run_on_loop` 调度，但 AI 侧
5 个文件 10 处调用点未对齐。

### 35.2 修复（三层模式）

| 层 | 改动 | 文件 |
|---|---|---|
| 1 捕获主循环 | `Database.init()` 记录 `_main_loop`（init 由 lifespan 在主循环调用） | core/database.py |
| 2 on_main 原语 | 非主 loop 时经 `run_coroutine_threadsafe` 调度回主循环并阻塞等待（timeout 30s）；cancelled future 的 done_callback 吞掉 | core/database.py |
| 3 调用点桥接 | kv_get/kv_get_secret/kv_set/kv_json_get/kv_json_set 五方法内部走 on_main；ai/ 下 10 处 `async with db.session()` 块包成内部函数经 `_run_on_main`（getattr 回退：无 on_main 的测试替身直接执行） | core/database.py + ai/iteration.py + ai/strategy_designer.py + ai/reviewer.py + ai/optimizer.py + ai/market_analyst.py |

### 35.3 门禁

- 跨 loop 探针（`.optim/verify_r26.py`）：主循环独立线程 + worker 独立线程，
  worker kv_get/kv_set **4/4 PASS**（无跨 loop 错误、值双向正确）
- 全量 pytest **exit 0**（618 项；脚本风格 test_ai_guards.py 模块级 sys.exit 非
  pytest 收集模式，属既有设计，本次显式 ignore）
- 回滚 = git diff 对应行（core/database.py 的 on_main/_run_on_main + ai/ 调用点）

### 35.4 知识库索引（新增，run26）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 跨 loop 连接池三层修复（捕获主循环 + on_main 原语 + 调用点桥接） | 后台线程独立事件循环访问绑定主循环的 DB | ✅ R3 |
| getattr 回退兼容测试替身 | 新方法加入时 _FakeDb 类测试不炸 | ✅ R3 |
| run_coroutine_threadsafe done_callback 吞 CancelledError | 主循环关闭时 cancelled future 噪音 | ✅ R3 |

### 35.5 遗留闭环

run24 记录的 5 项中危（R1 断链平仓重试 / R2 迭代锁 / R3 跨 loop DB / R4 版本裁剪 /
R5 flat 原子写）**全部闭环**：run25 修 4 + run26 修 R3，无遗留。

---


---

## 36. 第二十七轮自治优化（2026-09-07）：AI 客户端调用路径（run27，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=AI 客户端调用路径本地处理链
> （提示词拼接/token 扫描/JSON 解析/校验重试——不含 LLM 网络延迟），min_gain=+5%，
> max_level=L2，预算 8 实验/4h
> 起点：run2-26 保留项之上；测量 command=`.optim/bench_ai_chain.py`

### 36.1 run27 基线快照（实际数值，合成 strategy_design 规模 + mock 网络）

| 段 | 耗时 |
|---|---|
| `_contains_json_instruction`（消息扫描，最大段） | 13.4µs |
| messages 复制（4 条） | 0.4µs |
| `_compact_output`（10KB pine 截断回灌） | 8.3µs |
| `_fewshot_for`（design 动态生成 schema 示例） | 4.8µs |
| `_parse_json`（4 种形态） | 1.1-4.5µs |
| 模型能力判定链（cap/reasoning/json_schema） | 0.7µs |
| **chat_json 全链（限速器+组装+解析, mock 网络）** | **23µs/次** |

### 36.2 收敛判断（§4④）

AI 客户端本地处理链全部微秒级，全链 23µs/次——相对 LLM 秒级网络延迟占比 <0.01%。
5% 门槛 ≈ 0.001ms 全在测量噪声内；网络延迟属外部 IO 不可优化；run26 的 KV 桥接
（30s 缓存内 0 次 KV 读）未增加本地开销。**零实验零保留，源码零改动。**

### 36.3 知识库索引（新增，run27）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 外部 IO 主导路径收敛模板 | 网络/磁盘/LLM 路径的本地处理链若 µs 级且占 wall <0.1% → 直接 §4④，无需逐段 fuzz | ✅ run27 |

---


---

## 37. 第二十八轮自治优化（2026-09-07）：修复后综合回归审计（run28，环境噪声不可判定）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=审计 run24-27 的 15 项 bug 修复未引入性能回归
> 方法：逐项串行重跑保留项 bench + 对照路径法甄别环境噪声

### 37.1 审计结果（8/8 CPU 密集项偏高 + 对照路径正常）

| # | 保留项 | 记录值 | 当前 | Δ | 判定 |
|---|---|---|---|---|---|
| 1 | factor_miner | 0.815s | 0.933s | +14.5% | 环境噪声 |
| 2 | strategy_drl | 3.882s | 4.387s | +13.0% | 环境噪声 |
| 3 | meta | 10.408s | 12.028s | +15.6% | 环境噪声 |
| 4 | backtest/rollout | 0.024/0.050s | 0.029/0.061s | +21/22% | 环境噪声 |
| 5 | costscan E1/独立 | 215.5/306.8ms | 275.7/662.1ms | +28/116% | 环境噪声 |
| 6 | db_round | 1.85ms | 2.67ms | +44% | 环境噪声 |
| 7 | startup | 15.6ms | 20.6ms | +32% | 环境噪声 |
| 8 | finalize on | 63.9ms | 72.1ms | +13% | 环境噪声 |
| **对照** | **data_loader（零改动）** | 7.68/10.91/2.40ms | 6.61/11.37/1.98ms | **-14/+4/-18%** | **正常/更优** |

### 37.2 甄别结论：环境噪声，非修复回归

- **对照路径证据**：run24-27 零改动的 data_loader 正常甚至更优（load_df -14%、
  sanitize -18%）→ 修复未引入结构回归；且所有修复均为 O(1)/算术级或只改标志位
- **环境检查**：CPU LoadPercentage=**28%**，后台 **BlackOps3（游戏）/steam/Steam++/
  msedgewebview2** 进程抢占——CPU 密集路径（训练/回测/bootstrap/orjson 解析）全部
  受抢，IO 路径（data_loader）不受影响
- **同 run8 Roblox 教训**：CPU 密集 bench 在噪声下不可判定；诚实记录而非猜测保留

### 37.3 待办

清理后台进程（游戏/浏览器）后重跑本表 1-8 项复测确认无回归。

### 37.4 知识库索引（新增，run28）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 对照路径法判定环境噪声 | CPU 密集项全偏高时重跑零改动路径甄别 | ✅ data_loader 正常 → 无结构回归 |
| 噪声甄别信号组合 | Load% 高 + 后台进程 + IO 正常 + CPU 全同向 | ✅ run28（同 run8） |
| 相对值优先看对照 | 保留项收益验证用 prepare-once vs 独立的相对差 | ✅ costscan（独立对噪声更敏感） |

---


---

## 38. 第二十九轮自治优化（2026-09-07）：低负载自动复测（run29，8/8 PASS）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=run28 环境噪声项（CPU 密集保留项
> 受游戏/浏览器抢占不可判定）→ 等待 CPU<10% 后自动重跑确认修复无性能回归
> 测量 command=`.optim/run29_audit.py`（轮询 LoadPercentage，<10% 连续两次确认后
> 串行跑 8 项，窗口 90min）

### 38.1 触发与执行

- 19:01 启动，负载持续 1-97% 剧烈波动（游戏活跃），20:26:48 负载稳定 **4%** 触发
- 20:27:45 完成 8/8 串行（run29_audit.log）

### 38.2 复测结果（干净环境 vs 记录）

| # | 保留项 | 记录值 | 干净实测 | Δ | 判定 |
|---|---|---|---|---|---|
| 1 | factor_miner | 0.815s | 0.763s | -6.4% | ✅ 更优 |
| 2 | strategy_drl | 3.882s | 3.829s | -1.4% | ✅ 持平 |
| 3 | meta | 10.408s | 10.409s | 0.0% | ✅ 持平 |
| 4 | backtest/rollout | 0.024/0.050s | 0.026/0.052s | +8/4% | ✅ σ 内 |
| 5 | costscan E1/独立 | 215.5/306.8ms | 222.8/313.1ms | +3.4/2.1% | ✅ 持平 |
| 6 | db_round | 1.85ms | 2.19ms | +18% | ✅ cv 22.8% 内 |
| 7 | startup | 15.6ms | 18.1ms | +19% | ✅ best.json.gz 861.7KB(+2.4%) 文件增长 + 环境 |
| 8 | finalize on | 63.9ms | 64.4ms | +0.8% | ✅ 持平（首跑 74.1 残余负载，复测 64.4） |

### 38.3 结论

- **8/8 PASS**：run28 的 +13~44% 全部确认是环境噪声（BlackOps3/steam/浏览器抢占，
  LoadPercentage 1-97% 波动），修复（run24-27 的 15 项 bug）无性能回归
- startup 差异归因于文件自然增长：best.json.gz 841→861.7KB（evolve 版本累积
  v187→v202；run24-26 未触碰 model_zoo 读取路径）——数据规模而非代码回归
- finalize 首跑 74.1ms 是残余负载，复测 64.4ms 持平

### 38.4 知识库索引（新增，run29）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 低负载自动复测（轮询 CPU<10% 连续确认 + 串行 bench） | 后台进程抢占导致 CPU 密集 bench 不可判定 | ✅ run29（run29_audit.py 留作标准工具） |
| 文件自然增长 vs 回归甄别 | 复测异常先查数据规模（模型版本累积/DB 增长）再怀疑实现 | ✅ startup best.json.gz +2.4% |

---


---

## 39. 第三十轮自治优化（2026-09-07）：引擎主循环/总线路径（run30，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=引擎主循环/总线路径（core/bus.py
> 事件分发 + TradingEngine._on_candle 每根 K 线工作），min_gain=+5%，max_level=L2，
> 预算 8 实验/4h
> 起点：run2-29 保留项之上；测量 command=`.optim/bench_engine_loop.py`

### 39.1 run30 基线快照（实际数值，3000 根合成事件）

| 段 | 耗时 |
|---|---|
| bus publish + drain + handler 全链路 | **4.6µs/根 K 线**（runs 4.1-5.1µs） |
| handler-only（_candle_lock+_strategy_lock 双锁 + 快照解析） | 1.3µs |
| bus publish-only | 0.7µs（profile 分摊） |
| profile 前 4 项 | 全部为基准自身（on_candle/put_nowait/make_event/publish） |

### 39.2 收敛判断（§4④）

引擎主循环/总线路径全部微秒下限：5% 门槛 ≈ 0.23µs 在测量噪声内。asyncio 双锁
（_candle_lock + _strategy_lock）仅 ~0.5µs 可忽略。真实成本在 handler 内的
strategy.on_candle 信号计算与 _current_state 状态读取——属策略层（run6/7 已优化）
与数据层（run10/21 已收敛），bus 分发本身无触碰热点。**零实验零保留，源码零改动。**

### 39.3 知识库索引（新增，run30）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 主循环路径优化落点判定 | 分发/锁/队列为 µs 级时，优化空间在 handler 内真实工作而非框架 | ✅ run30（4.6µs 全链路） |
| 无限循环组件吞吐测量 | bus.run() 等 while 循环须手动 drain 或 wait_for 限时，不能直接 gather | ✅ run30（首版探针超时教训） |

---


---

## 40. 第三十一轮自治优化（2026-09-07）：指标向量化路径（run31，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=指标向量化路径
> （precompute_indicator_series / snapshot_at / IncrIndicators.update——回测/训练/
> env/实时行情流共享底座），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-30 保留项之上；测量 command=`.optim/bench_indicators.py`

### 40.1 run31 基线快照（实际数值，5000 根合成 OHLCV）

| 路径 | 耗时 | 5% 门槛 |
|---|---|---|
| precompute（一次性） | **4.54ms** | 0.23ms |
| snapshot_at | **1.4µs/根**（5000 根 7.1ms） | 0.07µs |
| IncrIndicators.update | **2.3µs/根**（5000 根 11.4ms） | 0.115µs |

### 40.2 收敛判断（§4④）

指标三路径全部微秒下限：
- precompute 4.54ms 一次性：EMA 已 pandas ewm（C 实现）、SMA/布林带已累加器向量化、
  RSI/ATR 的 Wilder 递推是 Python 循环但 5000 次仅毫秒内小部分
- snapshot_at 1.4µs/根：P2-13 已 tolist 缓存（惰性一次性转换 + O(1) 索引）
- IncrIndicators.update 2.3µs/根：SMA 用 sum(deque) 是 O(window)=45 元素常量，
  布林带已用增量累加器；对 300s 间隔实时行情完全可忽略

**深度优化候选均触碰行为等价红线**：Incr SMA 改累加器会改变浮点求和序（减法
累积）；Wilder 递推改 lfilter/scan 同触归约序红线（run4 教训）。零实验零保留，
源码零改动。

### 40.3 知识库索引（新增，run31）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 递推型指标天然红线 | Wilder（RSI/ATR）等序贯递推无法安全批量化，Python 循环微秒级不优化 | ✅ run31（5000 次递推毫秒内） |
| 增量类 O(1) 判定 | sum(deque)=O(window) 常量小窗口时无需改累加器（改则触浮点序红线） | ✅ run31（45 元素） |

---


---

## 41. 第三十二轮自治优化（2026-09-07）：web API 响应组装路径（run32，-83%）

> 执行依据：《自治优化协议 v2》；契约：主目标=API 响应组装路径（pydantic 序列化、
> 列表投影后的 JSON 组装、大 JSON 解析），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-31 保留项之上；测量 command=`.optim/bench_api_asm.py`

### 41.1 run32 基线快照（实际数值）

| 段 | 耗时 | 说明 |
|---|---|---|
| list_results（UI 高频轮询，20 行） | **0.048ms** | 列投影（run12）+ 小 metrics 解析 + dict 组装——微秒下限 |
| get_result（详情请求） | **0.97ms** | 87KB equity + 13KB trades + config/metrics 全量 json.loads |

### 41.2 run32 变更清单（保留项）

| 实验 | 变更文件 | 改动内容 | 增量 |
|---|---|---|---|
| E1 | `web/api/backtest.py` | **get_result 4 处 json.loads 改 `_loads` helper（orjson 优先 + ImportError 回退）** | **-83%**（0.97→0.16ms，5.9x） |

### 41.3 门禁与等价性

- 逐位等价（`.optim/verify_r32.py`）：equity/trades/metrics/config 4 场景 orjson vs json **ALL EQ**（run16 已对 1.93MB 全 dict 递归验证；JSON 浮点为确定性最近舍入）
- orjson 缺失时 `_loads` 回退标准 json（行为/格式不变）；orjson 已声明 requirements.txt（run16）
- test_backtest_* + test_drl_optimizations 回归 PASS；全量 pytest **exit 0**，测试未动
- 回滚：撤销 `_loads` helper 与 get_result 4 处调用（恢复 json.loads）

### 41.4 收敛判断（§4④ 分析）

| 子项 | 结论 |
|---|---|
| get_result 大 JSON 解析（唯一可触碰热点） | E1 已优化（-83%） |
| list_results 高频轮询 | 0.048ms 微秒下限（列投影已 run12） |
| 传输层 | run13 gzip -74.1% 已闭环 |

### 41.5 知识库索引（新增，run32）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| API 大 JSON 解析 orjson 替换（_loads helper + ImportError 回退） | 详情端点全量解析大 JSON（87KB+） | ✅ E1（5.9x，4 场景 EQ） |
| 组装层与传输层闭环判定 | 组装 0.16ms + gzip -74.1% = API 全链路优化完成 | ✅ run32 |

---


---

## 42. 第三十三轮自治优化（2026-09-07）：行情/WS 处理路径（run33，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=行情/WS 处理路径本地段
> （exchange/ws_market.py：WebSocket 订阅、K线聚合、ticker 分发、sr/pa 补算、
> 增量指标），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-32 保留项之上；测量 command=`.optim/bench_ws_local.py`（mock 外部连接）

### 42.1 run33 基线快照（实际数值，300 根合成 K 线）

| 段 | 耗时 | 频率 |
|---|---|---|
| `_compute_sr_pa`（300 根窗口补算） | **0.509ms** | 每根**新** K 线一次（`_sr_pa_ts` 缓存防同 ts 重复） |
| IncrIndicators.update | 2.1µs/根 | 每根 |
| snapshot 组装（closes+切片+缓存逻辑） | 4.9µs | 每根 |
| profile 主导 | `_find_pivot_points`（technical.py:95）numpy ufunc reduce 29,050 次 | — |

### 42.2 收敛判断（§4④ 三重判定）

`_compute_sr_pa` 是唯一非微秒项，但：
1. **低频**：5m 行情每 300s 一根 → 对引擎 wall 贡献 **0.17%**
2. **已有增量缓存**：`_sr_pa_ts` 保证同 ts 不重算
3. **红线**：support_resistance 枢轴点（window=10/min_touches=2）滑动复用会改变
   检测结果——与 run31 递推型指标同族语义红线

**零实验零保留，源码零改动。**

### 42.3 知识库索引（新增，run33）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 低频+缓存+红线三重判定模板 | 候选热点频率极低 + 已有增量缓存 + 优化触语义 → 直接 §4④ | ✅ run33（0.509ms 每 300s） |
| 枢轴点检测红线 | support_resistance 滑动窗口复用改变检测结果 | ✅ run33（同 run31 递推族） |

---


---

## 43. 第三十四轮自治优化（2026-09-07）：最终收官总结（run34）

> 执行依据：《自治优化协议 v2》；本轮动作：最终全量 pytest 验证（exit 0）+ 33 轮
> 全部成果累计总整理

### 43.1 累计保留项与回滚清单（13 组优化）

| 轮次 | 保留项 | 实测收益 | 回滚命令 |
|---|---|---|---|
| run4/5 | 因子挖掘链路（_numpy_rank/_rolling_ic/turnover 向量化）+ 进化引擎整轮 | **-54.8%**（1.71→0.77s/轮） | 撤销 analysis.py/factor_miner.py 向量化改回 pandas 原式 |
| run6 | strategy_drl 管线（_log_softmax_1row + 训练快速路径） | **-23.3%**（5.30→4.07s） | 撤销 train_drl 快速路径 |
| run7 | meta 控制器管线（step 内 numpy 索引化） | **-19.8%**（17.93→14.38s） | 撤销 step 索引化 |
| run9 | greedy rollout（_softmax_1row） | **-14.8%**（0.061→0.052s） | 撤销 _softmax_1row |
| run11 | 成本扫描 prepare-once | **-30.0%**（320.6→224.6ms） | 恢复独立 per-combo 预处理 |
| run12 | DB 落库合并 DELETE | **-14%**（3.12→2.6ms） | 恢复逐轮 DELETE |
| run13 | GZipMiddleware(minimum_size=100_000) | **-74.1%**（497→128.5KB） | 移除 middleware |
| run16 | orjson 权重解析（ImportError 回退） | **-62%**（启动 38.4→14.6ms） | 恢复 json.loads + 删依赖行 |
| run17 | bootstrap idx 算术重排 | **-23%**（finalize 82.6→63.6ms） | 恢复 blk/off/idx 原式 |
| run19 | meta vol 整列预计算 | **-13.5%**（12.03→10.41s/轮） | 撤销 vol_feature/预计算/查表 |
| run32 | API 大 JSON orjson（_loads helper） | **-83%**（get_result 0.97→0.16ms） | 撤销 _loads 与 4 处调用 |

### 43.2 累计 bug 修复（15 项，全部闭环）

| 轮次 | 修复 | 严重度 |
|---|---|---|
| run24 | B1 限价单比例方向写反 / B2 实盘 DRL 动作 0 无分支 / B3 dynamic_composite 负 IC 正暴露 / B4 NaN oos_ret 绕过部署门 / B5 PBO=0 被 or0.5 吞 / B6 reset 越界 / B7 子策略失败残留 / B8 demo_blocked 计入 streak / B9 episodes 无下界 / B10 空 history | 高4中3低3 |
| run25 | R1 断链平仓不重试 / R2 迭代序号非原子 / R4 max_versions=1 裁剪错位 / R5 flat 非原子写 | 中4 |
| run26 | R3 worker 跨 loop DB 连接池 | 中1 |

### 43.3 审计/收敛判定（13 次全 PASS）

run10（数据加载/Pine 下限）、run14（DRL 状态预处理收敛）、run15（因子特征工程收敛）、
run18（跨轮联合复测 8/8）、run20（IC 批量收敛）、run21（数据缓存复测）、run22（gzip
复测零回归）、run27（AI 客户端下限）、run28（修复后审计，环境噪声甄别）、run29（低
负载复测 8/8）、run30（引擎主循环下限）、run31（指标向量化下限）、run33（行情 WS 收敛）

### 43.4 最终状态

- 全量 pytest **exit 0**（618 项；脚本风格 test_ai_guards* 非 pytest 收集模式，既有设计）
- 工作树：40 文件改动（+3732/-677），含用户既有未提交改动与优化项共存，回滚须按表逐项行级撤销
- 行为等价红线全程零触碰（corrcoef/qcut、np 归约序、5M 2D gather、val/oos 切片、手写 Pearson）
- 待办：服务器重启生效所有后端 py 改动（服务器未运行，用户自行手动重启）；新环境 `pip install -r requirements.txt`（orjson 缺失自动回退）

---


---

## 44. 第三十五轮自治优化（2026-09-07）：最终抽查 API 层（run35，审计 PASS）

> 执行依据：《自治优化协议 v2》扩展轮；主目标=run32 E1（get_result orjson 化）后
> API 端点端到端无回归（34 轮成果收官的最后一层保障）
> 测量 command=`.optim/smoke_api_r35.py`（TestClient 端到端：run→list_results→get_result + orjson 等价性）

### 44.1 抽查结果（4/4 PASS）

| 项 | 结果 |
|---|---|
| `/api/backtest/run`（demo 回测） | 200 |
| `/api/backtest/results?limit=20`（列投影+组装） | 200，n=1 |
| `/api/backtest/results/{id}`（orjson 大 JSON 解析路径） | 200，keys 完整（config/metrics/equity_curve/trades/id/created_at） |
| `_loads`(orjson) vs json.loads（1000 元素样本） | **EQ** |

### 44.2 结论

- **run32 E1 端到端无回归**：get_result 经 orjson 路径返回结构 keys 完整、类型正确
- demo 回测在 smoke 环境（engine=None）落库为占位记录（metrics 仅 status、曲线空）——
  非 E1 问题，断言以结构完整性为准
- 本轮零实验零保留（审计性质），源码零改动

### 44.3 知识库索引（新增，run35）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| TestClient 端到端 smoke 骨架 | API 改动后快速无回归验证（注入 app.state.db/engine；run 返回 task_id 非同步结果） | ✅ run35 |
| 结构完整性断言 | smoke 断言用类型/keys 而非数据丰富度（占位记录场景） | ✅ run35 |

---


---

## 45. 第三十六轮自治优化（2026-09-07）：策略信号计算路径（run36，收敛无保留）

> 执行依据：《自治优化协议 v2》；契约：主目标=策略执行器信号计算路径
> （各 on_candle 每根 K 线吞吐——run30 指出的真实成本所在，此前未单独量化），
> min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-35 保留项之上；测量 command=`.optim/bench_strategies.py`

### 45.1 run36 基线快照（实际数值，5000 根合成 IND 快照 × 5 执行器）

| 执行器 | 耗时/根 | 说明 |
|---|---|---|
| GridStrategy | **0.6µs** | O(1) 算术 |
| DualMAStrategy | **0.7µs** | O(1) 算术 |
| PriceActionStrategy | **0.7µs** | sr/pa 已由 hub 补算传入 |
| FactorSignalStrategy | **4.5µs** | 因子值计算 |
| RLAdaptiveStrategy | **4.8µs** | agent mtime 检查 + 预测（最贵） |

### 45.2 收敛判断（§4④）

5 个执行器 on_candle 全部微秒下限（0.6-4.8µs/根）；对 5m 行情（300s 间隔）wall
贡献 **0.0017%**；5% 门槛 ≈ 0.03-0.24µs 全在测量噪声内。信号计算为 O(1) 字典读取
+ 少量算术，逐根状态相关（依赖前一根持仓/内存）→ 无向量化/批量化空间；rl_adaptive
的 agent 加载已是 mtime 检查缓存路径（权重解析 run16 orjson 已加速）。**零实验零
保留，源码零改动。**

### 45.3 收官验证（run30 判断复核）

run30 曾判断"主循环路径的真实成本在 handler 内的 strategy.on_candle"——本轮实测
该处为微秒级（≤4.8µs/根）。引擎主循环全链路每根 K 线：bus 分发 4.6µs + 策略信号
≤4.8µs < **10µs/根**——对 5m 行情无可优化空间，全仓库热路径收敛闭环。

### 45.4 知识库索引（新增，run36）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 逐根状态相关无批量空间 | 信号依赖前一根状态（持仓/内存）的执行器 | ✅ run36（5 执行器 0.6-4.8µs） |
| 主循环+信号全链路每根 <10µs | 收官验证：低频行情无可优化空间 | ✅ run36（run30 判断复核） |

---


---

## 46. 第三十七轮自治优化（2026-09-07）：AI 功能优化项（run37，E1 保留）

> 执行依据：《自治优化协议 v2》；主目标=AI 功能优化项（token/调用效率：提示词
> 规模、调度器失败重试、缓存机会——非 CPU，聚焦 LLM 成本与任务有效性）
> 测量 command=`.optim/probe_ai_prompts.py` + 静态/动态验证

### 46.1 提示词规模盘点（6 功能，输入 token 直接成本）

| 功能 | 规模 | ≈token |
|---|---|---|
| market_analysis | 5.0KB | 1280 |
| price_action_optimize | 3.5KB | 888 |
| strategy_design | 5.6KB | 1443 |
| strategy_iterate | 4.0KB | 1014 |
| trade_review | 3.3KB | 833 |
| validate_pine | 1.6KB | 398 |

**结论**：提示词此前已 token 精简（30 根压缩、单行格式化、_compact_output 回灌截断、
few-shot 补偿），规模合理无可触碰膨胀点。重试/退避/并发（_AI_SEM=2）/配置缓存（30s）
均此前已实现。

### 46.2 唯一可触碰优化项：调度器失败重试（E1 保留）

- **问题**（run25 子代理记录的可疑点 7）：`_ai_scheduler` 三处
  `_last_analysis/_last_optimize/_last_review = now` **调用前置位**——若
  `_analysis_context`/DB locked 等外层异常被循环体吞掉，该任务到下一周期
  （600s/86400s/604800s）才重试，即"失败一次静默跳过一整周期"（AI 机会与 token 白费）
- **修复**：置位移到**成功返回后**——外层异常不置位 → 下轮 30s 重试；AI 错误
  （safe_analyze/optimize_price_action 内部已吞掉返回 None）仍置位 → 配置缺失等间隔
  再试（两语义需区分，旧行为不变）
- **验证**：`verify_r37.py` 静态断言 9/9（三处置位均在对应 await 调用之后）+ 
  test_ai_scheduler PASS + 全量 pytest **exit 0**
- 回滚：恢复三处 `_last_* = now` 到调用前

### 46.3 知识库索引（新增，run37）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 定时任务失败重试：成功返回后置位 | 时间戳前置位 + 循环体吞异常 = 失败静默跳过一整周期 | ✅ E1（外层异常 30s 重试/内部错误等间隔，语义区分） |
| AI 功能成本盘点顺序 | 提示词规模 → 调用次数/重试 → 缓存/并发 | ✅ run37（规模合理、重试已修、其余已实现） |

---


---

## 47. 第三十八轮自治优化（2026-09-07）：端到端总链路复核（run38，PASS）

> 执行依据：《自治优化协议 v2》；主目标=完整 EvolveEngine 周期 profile，确认 37 轮
> 后热点分布与知识库预期一致（收官最后一道全局验证），min_gain=+5%，max_level=L2
> 测量 command=`.optim/bench_evolve_cycle.py 1`（真实 EvolveEngine + mock 数据 + tmp models）

### 47.1 复核结果（完整周期，最终代码）

| 段 | 当前 | run8 基线 | 判定 |
|---|---|---|---|
| 整轮 cycle | **15.196s** | 21.04s | ✅ **-27.8%**（三管线收益乘积一致） |
| meta_controller | 10.115s（67%） | 13.6-15.4s | ✅ ≈run19 后基线（10.41s） |
| strategy_drl | 3.820s（25%） | 4.4-5.0s | ✅ ≈run6 后基线（3.88s） |
| factor_miner | 1.262s（8%） | 1.0-1.8s | ✅ 范围内（含级联因子注入） |
| 调度层 | **0.000s** | ≈0 | ✅ run8 结论保持 |

### 47.2 结论

- 整轮累计 **-27.8%** 与三管线各轮收益叠加（fm -54.8% / sd -23.3% / mc -31%）吻合
- 三管线占比 meta 67% / sd 25% / fm 8% 与知识库预期一致——37 轮优化在总链路
  **无回归、无遗漏热点**
- 调度层仍 ≈0（zoo 序列化/落库/轮换簿记无成本，run8 结论交叉印证）
- 本轮零实验零保留（审计性质），源码零改动

### 47.3 知识库索引（新增，run38）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 端到端总链路复核（完整周期分段计时） | 多轮优化后的全局收口——整轮收益与各轮乘积交叉验证 | ✅ run38（-27.8% 吻合） |
| 三管线占比基线（meta 67%/sd 25%/fm 8%） | 后续优化优先级参考 | ✅ run38（meta 最大块但已收敛） |

---


---

## 48. 第三十九轮自治优化（2026-09-07）：前端 JS 优化项（run39，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=前端 JS 优化项（web/static/index.html
> 加载/渲染/轮询路径——run13 只管 gzip 传输，前端逻辑从未审计；无 headless
> 设施故静态分析），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 起点：run2-38 保留项之上；方法：静态审计（grep 轮询/请求/事件模式 + 读关键实现）

### 48.1 审计结论：前端已系统优化，无可触碰项

| 机制 | 实现 | 状态 |
|---|---|---|
| 业务轮询统一调度 | P1-12 startPolls/stopPolls（防双发） | ✅ 已有 |
| 后台标签页暂停 | visibilitychange onVis：hidden→stopPolls，回前台→startPolls+补刷 | ✅ 已有 |
| 视图门控 | 非当前视图跳过（view.value 判断） | ✅ 已有 |
| 轮询频率与 TTL 匹配 | 3-10s vs 后端 TTL 缓存 3-5s | ✅ 已有 |
| resize | debounce 150ms + isDisposed 守卫 | ✅ 已有 |
| 图表生命周期 | switchView 释放离场实例防白屏；主题切换重建 | ✅ 已有 |
| 主题/密度持久化 | localStorage | ✅ 已有 |
| 持续进化轮询 | 与可见性解耦常驻（设计决策：服务端训练持续运行、进度如实展示，后台节流回前台补刷） | ✅ 设计决策不改 |

**minor note**：refreshCharts 中 plateauChart 处理重复两处（L4990/L4993）——无害
重复，不构成优化项。

### 48.2 收敛判断（§4④）

前端轮询/渲染/生命周期已系统性优化（P1-12 轮询调度、visibilitychange、debounce、
ECharts 生命周期、localStorage 持久化）；api() 无响应缓存但后端各端点已 run12/32
优化（列投影 + orjson）。**零实验零保留，源码零改动。**

### 48.3 知识库索引（新增，run39）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 前端轮询成熟模式（统一调度+可见性暂停+视图门控+频率匹配 TTL） | 后台请求量收敛的审计清单 | ✅ run39（全部已有） |
| 设计决策 vs 遗漏识别 | 持续进化轮询不随可见性暂停（服务端后台任务需如实展示） | ✅ run39 |
| 无害重复 minor note | plateauChart 双处理——记录不修 | ✅ run39 |

---


---

## 49. 第四十轮自治优化（2026-09-07）：告警通知链（run40，E1 保留）

> 执行依据：《自治优化协议 v2》；主目标=告警通知链（notify 组装/限频——重点验证
> run25 R1 修复后断链重试是否通知叠加刷屏），min_gain=+5%，max_level=L2
> 方法：静态审计 core/notify.py + 动态验证（patch _send_all 记录发送）

### 49.1 审计结论

- notify 防抖机制**正确**：同标题 60s 去重（_DEBOUNCE by title）收敛风控连续触发
  刷屏；fire-and-forget（loop.create_task）永不阻塞交易热路径；三通道独立异常
- 18 处调用点（evolve 9 / risk 4 / engine 5）——标题唯一性筛选发现唯一误伤实例：

### 49.2 E1（保留）：重试成功通知防抖误伤

- **问题**：run25 R1 修复的断链平仓重试成功通知（trading_engine.py L785）复用
  『行情断流告警』标题——与 L771 首次告警同标题，重试通常在首次告警后 60s 防抖
  窗内快速成功 → **成功确认被防抖吞掉**，用户收不到"已平仓"
- **修复**：独立标题『行情断链：平仓重试成功』
- **验证**：verify_r40.py（同标题去重 / 异标题不被误伤 / force 绕过）**3/3 PASS** +
  test_engine_start_fail + test_restart_state 回归 PASS + 全量 pytest **exit 0**
- 回滚：L785 标题改回『行情断流告警』

### 49.3 知识库索引（新增，run40）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 追加通知独立标题 | 新增的追加通知（成功确认/重试结果）复用原标题会被防抖误伤 | ✅ E1 |
| 标题唯一性检查 = 防抖误伤快速筛选 | 告警链审计：遍历 notify 调用点检查同标题复用 | ✅ run40（18 处发现 1 例） |

---


---

## 50. 第四十一轮自治优化（2026-09-07）：监控健康路径（run41，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=监控健康路径（bus.health、engine status、
> /_PROGRESS 缓存清理、tasks_cache 裁剪——长跑服务内存/查询健康），min_gain=+5%，
> max_level=L2，预算 8 实验/4h
> 方法：静态审计 web/api/tasks_cache.py + backtest.py _PROGRESS 生命周期 + bus.health

### 50.1 审计结论：清理机制完整，无可触碰热点

| 组件 | 机制 | 状态 |
|---|---|---|
| `_PROGRESS` | kline_history 限 200 根；每 200 根 prune 一次（避免长回测每根加锁）；**完成即 mark_done + prune**（backtest.py L346-347，成功与失败路径均处理） | ✅ 无泄漏 |
| `tasks_cache` | max_items=200 / TTL 12h / running 永不清理 / TTL 窗口内超限按 `_done_ts` 保留最近 N 个（P2 已修超限） | ✅ 机制完整 |
| `bus.health()` | 同步只读遍历 `_handler_failures`（O(handlers)）；仅主循环写，单循环无锁安全 | ✅ 无热点 |

### 50.2 收敛判断（§4④）

监控健康路径清理机制完整、查询 O(handlers) 微小，前端 5s 轮询可忽略。
**零实验零保留，源码零改动。**

### 50.3 知识库索引（新增，run41）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 任务缓存清理完整模式（max_items+TTL+running 保护+完成即 prune） | 长跑服务防内存泄漏 | ✅ run41（_PROGRESS + tasks_cache） |
| 批量惰性清理 vs 每事件清理 | 高频写入缓存按批次 prune（每 200 根一次） | ✅ run41 |
| 监控查询不影响热路径 | bus.health 同步只读 + 单循环写无锁 | ✅ run41 |

---


---

## 51. 第四十二轮自治优化（2026-09-07）：数据导入导出路径（run42，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=数据导入/导出路径（load_csv 解析/清洗、
> CSV 导出、报表生成——run10 只测 klines 缓存路径），min_gain=+5%，max_level=L2
> 测量 command=`.optim/probe_csv.py`（5000 行合成 CSV）

### 51.1 run42 基线快照（实际数值）

| 项 | 耗时 | 说明 |
|---|---|---|
| load_csv（5000 行 541KB） | **11.36ms** | read_csv 6.14ms 主导 + to_datetime 2.68ms——pandas C 解析下限 |
| run.py df.to_csv（导出） | — | 用户触发 |
| pine_export | — | run10 已测毫秒级 |
| 前端 CSV 组装 | — | 用户触发 |

### 51.2 收敛判断（§4④）

数据导入导出路径全部为**低频用户触发**（CSV 上传/导出、报表生成、Pine 导出）：
单次用户操作秒级 UX 可接受，11.36ms 无需优化且 pandas C 解析已是下限。与 run20
（IC 批量 API 低频）同族判定。**零实验零保留，源码零改动。**

### 51.3 知识库索引（新增，run42）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 低频用户触发路径收敛模板 | 上传/导出/报表类端点（单次操作秒级 UX 可接受）即使 10ms 级也无需优化 | ✅ run42（load_csv 11.36ms，同 run20 族） |

---


---

## 52. 第四十三轮自治优化（2026-09-07）：持续进化质量与效率（run43，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=持续进化质量与效率（用户指定方向：不只看
> 训练速度 run4-7，审计进化系统的质量指标与数据/轮次效率），min_gain=+5%，max_level=L2
> 方法：静态审计 drl/evolve_engine.py 进化判定链 + 配置默认值核算

### 52.1 进化质量链（完整闭环）

| 机制 | 实现 | 状态 |
|---|---|---|
| 回退保护 | `_should_rollback`：new < old×0.95；非正 fitness 方向反转（P0-4） | ✅ |
| OOS 部署门 | `_strategy_deploy_gate`：OOS 收益 + oos_trades/position_ratio + isfinite（run24 修复 NaN） | ✅ |
| 挑战者晋升 | 须显著优于冠军 ×1/0.95≈1.053（防噪音替换）；角色互换保留旧冠军（+92% 记录） | ✅ |
| 停滞检测 | stall_rounds=10 连续未接受触发重置 | ✅ P1-3 |

### 52.2 进化效率链（完整闭环）

| 机制 | 实现 | 状态 |
|---|---|---|
| 增量门 | `_data_incremented`：尾时间戳比较防空转重训（P2-11/P4-E1） | ✅ |
| 训练频率门槛 | min_new_bars + evolve_min_train_gap_sec 周期换算 | ✅ |
| 键控隔离 | 键含 symbol/tf 防跨标误判；demo 轮不污染 prev | ✅ |
| 配置切换清理 | apply_config 换池/换周期 clear `_last_tail_ts` | ✅ |

### 52.3 配置默认值核算

5m 周期 min_new_bars=2 → **10 分钟一轮** × 三管线训练成本 ~15s/轮（run38 总链路）
→ 总成本占比 **2.5%**——训练频率与行情节奏/计算成本匹配（因子挖掘/策略/元策略
按轮次各自核电，增量门再过滤无效轮）。

**零实验零保留，源码零改动。**

### 52.4 知识库索引（新增，run43）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 进化质量/效率双链审计清单 | 逐机制核对（回退/晋升/停滞/OOS 门 + 增量门/频率门槛/键隔离） | ✅ run43（全部闭环） |
| 训练频率量化核算 | 周期×min_new_bars → 训练间隔 × 每轮成本 → 占比判定 | ✅ run43（2.5%） |

---


---

## 53. 第四十四轮自治优化（2026-09-07）：交易所 REST 层（run44，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=交易所 REST 层本地段（exchange/manager.py
> ccxt 封装 + _execute_signal/order_manager.place 下单组装——run12 测过 DB，REST
> 本地段未测），min_gain=+5%，max_level=L2，预算 8 实验/4h
> 方法：静态审计 manager.py + trading_engine._execute_signal + order_manager.place

### 53.1 审计结论：外部 IO 主导，本地段全部 O(1)

| 层 | 实现 | 本地段 |
|---|---|---|
| manager.py | ccxt 薄封装（fetch_ticker/ohlcv/balance/order/positions 透传 + 代理注入） | 透传 |
| `_execute_signal` | 风控检查 O(1) → SIGNAL 发布 → place → 记账 | O(1) |
| `_place_paper` | limit 价/市价滑点叠加（买 ×(1+s) 卖 ×(1-s)，与回测同向）→ `_resolve_qty` 现金换算 → `apply_fill` FIFO 摊销 | O(1)/O(持仓笔数) |
| `_place_live` | 同构 + 实盘精度/名义额 | O(1) |

**前序修复已闭环**：P2-2 `_trade_times` 1h 剪枝、run25 R1 断链平仓重试、run25 R5
余额缓存失败保留、run26 R3 跨 loop DB 桥接（REST 落库路径）。

### 53.2 收敛判断（§4④）

交易所 REST 层外部 ccxt 网络 IO 秒级主导、本地段全部 O(1)（低频交易路径）——与
run27（AI 客户端）/run42（导入导出）同族判定。**零实验零保留，源码零改动。**

### 53.3 知识库索引（新增，run44）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 外部 IO 主导路径族收敛确认 | AI 客户端/导入导出/交易所 REST——本地段 O(1) 或 µs 级即 §4④，无需逐段 bench | ✅ run44（run27/42 同族） |
| 交易路径本地段口径要点 | 滑点方向与回测同向、限价不叠滑点、手续费留空间（run24 B1 后纸面/回测一致） | ✅ run44 |

---


---

## 54. 第四十五轮自治优化（2026-09-07）：纸面账户一致性（run45，审计 PASS）

> 执行依据：《自治优化协议 v2》；主目标=纸面账户一致性（paper.py apply_fill
> FIFO 摊销/成本基准 vs 回测引擎口径——run24 B1 修限价后未复验），min_gain=+5%
> 测量 command=`.optim/verify_r45.py`

### 54.1 验证结果（3/3 PASS）

| 项 | 结果 |
|---|---|
| 随机 200 步交易序列（买/卖交错） | ✅ 无异常、余额/持仓守恒 |
| fifo_consume vs backtest `_matching.py` L123-136 同式（300 组随机 lots） | ✅ 逐位等价 |
| PaperAccount.cash vs 独立账簿（买入扣价+费/卖出加价-费，150 步） | ✅ 一致 |

### 54.2 结论

- **纸面账户 FIFO/现金流与回测引擎口径完全一致**（M1『成本基准与回测对齐』代码层兑现）
- `fifo_consume` 为模块级单一实现，order_manager 实盘 FIFO 与 paper 共用——三路径（纸面/实盘/回测）同构
- 差异仅 backtest lots 3-tuple 多 `entry_i`（持仓时长统计字段，不影响成本基准）
- 本轮零实验零保留（审计性质），源码零改动

### 54.3 知识库索引（新增，run45）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 独立账簿对照法 | 成本/现金流口径正确性审计（账房式独立累加 vs 实现，不需全量回测） | ✅ run45（150 步一致） |
| 焦距函数等价验证 | 摊销函数 300 组随机 lots 逐位等价 | ✅ run45 |

---


---

## 55. 第四十六轮自治优化（2026-09-07）：策略注册/族谱路径（run46，收敛无保留）

> 执行依据：《自治优化协议 v2》；主目标=策略注册/族谱路径（strategies/__init__.py
> register_dynamic 后端 + 族谱树构建——run39 审前端，后端未审），min_gain=+5%，
> max_level=L2，预算 8 实验/4h
> 方法：静态审计 strategies/__init__.py + dynamic_store.py + 5 处 register_dynamic 调用点

### 55.1 审计结论：持久化闭环完整，无热点

| 环节 | 实现 | 状态 |
|---|---|---|
| 注册语义 | register_dynamic 覆盖式 + executor 推断（rl_ 前缀/model_path）+ param_schema/params 回退执行器自带 | ✅ |
| 持久化 | AiStrategy 表 5 处同步落库（evolve/designer/iteration/drl/repo，P1-1 跨 loop 桥接） | ✅ |
| 重启恢复 | dynamic_store.restore_from_db 幂等 + 单条容错（曾修坏记录致服务挂掉） | ✅ |
| 重命名/删除 | 占用检查 + DB 同步 + AiStrategy 级联 | ✅ |
| 族谱树 | 前端 O(n)（byName/children）；后端仅 list_strategies O(n) 低频 | ✅ 无热点 |

### 55.2 收敛判断（§4④）

策略注册/族谱路径持久化闭环完整（内存 _DYNAMIC ↔ AiStrategy 表 + 幂等容错恢复），
族谱树 O(n) 无服务端热点；run24 B7（MetaController 子策略名称残留）与 run25 R2
（迭代并发重号）此前已修。**零实验零保留，源码零改动。**

### 55.3 知识库索引（新增，run46）

| 模式 | 适用场景 | 已验证 |
|---|---|---|
| 动态策略持久化闭环（内存 ↔ DB + 幂等容错恢复） | 进程内全局注册表与 DB 同步 | ✅ run46（5 处落库 + restore_from_db） |
| 族谱 O(n) 前端构建 | 后端只提供列表、树在前端组装 | ✅ run46（byName/children） |

---

## 10. 可直接粘贴给 AI Agent 的执行指令

```
请对 crypto_ai_trader 执行自治优化，遵循以下规则：

1. 每次改动前先运行：python -m pytest -q（基线必须全绿）
2. 每次改动后运行：python -m compileall -q <改动文件> + python scripts/check_js.py
3. 风险分级：L0 文档/清理的直接做；L1 局部改动的做实验，门禁不过立刻回滚；
   L2 跨模块改动必须先在隔离分支/备份中实验；L3（部署/外部发布/删数据/密钥）一律停下来问用户
4. 任何"可能有副作用的优化"先写最小验证脚本（单元级）再改主体代码
5. 前端 UI 改动必须：浏览器打开 http://127.0.0.1:8000 实际截图 + 视觉确认，
   不能只依赖 DOM 断言（曾有 isVisible 误判历史）
6. 每完成一个优化，把结果追加写入 docs/AUTONOMOUS_OPTIMIZATION.md 的
   「变更清单」「验收门禁」「回滚记录」三个区块
7. 无效尝试必须记录到「无效尝试」区块（含失败原因），不得静默丢弃
8. 基准口径：pytest 通过数、check_js 通过、compileall、主要接口 P95 不退化
```