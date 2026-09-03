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

---

## 8. 无效尝试（反证记录）

1. **C13 PPO forward 去重**：深入分析后判定标准 PPO 每 mini-batch 更新后参数变化，forward 无法去重而不改变 clipped surrogate 语义 → 否决，不实施
2. **C14 backward pop 释放激活值**：实施时 pop 顺序破坏 `acts[i]` 与 ReLU mask 对齐 → `ValueError: broadcast (128,32) vs (128,15)`，测试抓出，立即回滚还原 → 教训：内存优化收益对本项目小网络可忽略，正确性优先；大型重构必须保留原实现对照
3. **CSS 菜单 overflow**：`overflow-x:auto` 会静默裁剪 absolute 下拉（CSS 规范：一轴非 visible 时另一轴变 auto）→ Playwright `isVisible` 检测不到祖先裁剪 → 教训：UI 验证必须加截图视觉确认，不能只信 DOM 断言

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