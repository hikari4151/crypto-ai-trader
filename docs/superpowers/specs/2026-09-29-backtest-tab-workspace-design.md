# 回测页 Tab 工作区 UI 重设计

- 日期：2026-09-29
- 状态：设计已经用户逐屏确认（可视化伴侣：布局方案 C → 视觉细化稿 → 磁贴光晕式 + 下划线 Tab）
- 设计稿留存：`.superpowers/brainstorm/1-1790500775/content/`（backtest-layout-options / backtest-style-refinement）
- 实施入口：`web/static/index.html` 中 `view==='backtest'` 的模板（3045-3425 行附近，行号会漂移以内容锚为准）与 bt 局部样式区（`.bt-layout` 等）

## 1. 背景与目标

现版回测页 7 个区块纵向堆叠（配置双栏 + 结果 + 实时监控 + 过拟合 + 扫描 + 对比 + 历史），用户判定「布局、层级、视觉都要整体重做」：

1. **信息层级乱**——7 区块无主次，跑完回测后结果淹没在长页面里；
2. **视觉不够精致**——卡片/间距/图表质感需升级；
3. **布局结构要重新组织**——双栏 + 下方堆叠改为新的分区方式。

核心原则（用户选定）：**执行流优先**——打开页面第一眼是「配置参数 → 开始回测」的高效执行流，结果与工具为执行服务。

一致性约束（用户选定）：**沿用全站深色体系，允许局部突破**——复用色板/字体/mac-* 组件，允许新组件形态（紧凑配置条、下划线主 Tab、运行状态条、指标磁贴、画布质感）。

视觉参考：用户未指定具体产品，由设计流程给出三方案（三步执行流 / 双栏工作台 / 单栏 Tab）后用户选定 **方案 C：单栏 Tab 工作区**。

## 2. 范围

### 2.1 在范围内

- `view==='backtest'` 整页模板重排：页头、配置、运行状态、结果、四个主 Tab、分析工具子 Tab；
- bt 局部 CSS 重写（新增配置条/状态条/下划线 Tab/磁贴光晕/画布质感；退役 `.bt-layout` 双栏规则）；
- `btGoto` 等锚点导航适配 Tab 切换（方法名不变，行为改为「先切 Tab 再滚动」）；
- 新增契约测试（沿用 `tests/test_backtest_ui_contract.py` 既有文件追加）。

### 2.2 不在范围内（验收红线）

1. **零后端改动**：`web/api/backtest.py`（及全部 `web/api/*`、`core/`、`engine/`）零 diff；
2. **零数据绑定改名**：既有 Vue 字段与方法全部保留原名——`bt/btResult/btMetrics/btAdvice/btLegend/btTradesAll/btTradesShown/btTradeSearch/btTradeShowAll/btLive/btHistoryGroups/btHistoryLoading/btHistoryError/btPlans/btRunning/btRange/overfitResult/gridScanResult/compareResult/plateauInfo/btLocalCoverage/btSymbolOptions/btStrategyOptions/btSymbolManual/btGoto/runBacktest/loadBtResult/loadBtHistory/applyBtPlan/saveBtPlan/delBtPlan/setBtRange/setBtSymbol/setBtStrategy/addBtSymbolManual/runGridScan/compareStrategies/runOverfit/useGridBest/runOverfitForBest/exportBtTradesCsv/formDrop/toggleFormDrop/switchView` 等；
3. **ECharts 容器与渲染不动**：`#btChart`、`#plateauChart` 的初始化/主题/重绘逻辑不改（只允许随容器移动位置）；
4. **回测轮询/进度推送策略不动**（`btLive` 的轮询与进度更新节奏保持现状）；
5. `tests/test_backtest_ui_contract.py` 既有 3 个测试（timeframe 数组字面量、ctxbar/menubar 规则）必须继续通过——尤其 `assert "['5m','15m','30m','1h','4h','1d']" in HTML`，周期分段的数组字面量不得改写。

## 3. 页面结构（方案 C 终稿）

自上而下单列：

```
┌ 页头：历史回测 + 副标题          [高级参数] [保存方案 ▾] [开始回测]
├ 紧凑配置条（常驻，直接开跑所需字段）
├ 运行状态条（仅 btRunning/btLive.running 时出现）
├ 主 Tab 行（下划线式）：结果 | 实时监控[运行徽标] | 分析工具 | 历史记录
└ Tab 主体
    ├ 结果：指标磁贴 ×N + 建议横幅 + 净值曲线 + 逐笔成交
    ├ 实时监控：进度 + 4 磁贴 + 实时K线表 + 最近成交表
    ├ 分析工具（内层子 Tab）：过拟合报告 | 参数扫描 | 策略对比
    └ 历史记录：分组表格
```

### 3.1 页头

- 左：**不放标题**——顶栏 `.tb-name` 已显示「历史回测」（`viewTitle`），页头重复放大标题会同屏两份（与持续进化页同裁定）；页头仅保留右侧按钮区（右对齐）。按钮全部站内 SVG 图标，**禁止 emoji**：
  - **高级参数**（新入口：打开高级参数浮层，见 §3.2）；
  - **保存方案 ▾**（现 `formDrop==='btPlan'` 二级菜单原样迁移）；
  - **开始回测**（现 `runBacktest()` 主 CTA，`btRunning` 禁用与「回测中…」文案不变）。

### 3.2 紧凑配置条（新组件，执行流优先的补偿设计）

一行 `flex-wrap` 卡片条（`.bt-configbar`），字段从现配置面板**上移**、开跑所需全部常驻：

| 字段 | 控件（原样迁移） | 绑定 |
|---|---|---|
| 数据来源 | `mac-seg` 三选（演示/交易所/CSV） | `bt.data_source` |
| 交易所 / CSV 路径 | 条件显示（`mac-select` / `input`） | `bt.exchange` / `bt.csv_path` |
| 交易对 | `menu-drop` 二级菜单（含手输） | `bt.symbol` + `formDrop==='btSymbol'` |
| 周期 | `mac-seg` 六选 | `bt.timeframe`（数组字面量不动） |
| 策略 | `menu-drop` 二级菜单（含 kind pill） | `bt.strategy_name` + `formDrop==='btStrategy'` |
| 起 / 止 | 两个 `date input` | `bt.since` / `bt.until` |
| 本地覆盖提示 | 现 `btLocalCoverage` / 「留空则使用最新 K 线 + 数据管理跳转」提示行（跟在日期字段下） | 原样 |
| 工具三键 | 参数扫描 / 策略对比 / 过拟合检测 | `runGridScan/compareStrategies/runOverfit`，loading 禁用不变 |

- **高级参数浮层**（新）：页头按钮点开，复用 `menu-drop`（Teleport + `popPos` + 点外部关闭 + Esc 关闭，与持续进化页「配置/触发」浮层同构）；内容 = 现「高级参数」整段原样迁移：初始资金/手续费率/滑点/K 线数量/GPU 加速（`bt.start_cash/fee_rate/slippage/limit/use_gpu`）。**不放日期相关提示**（那些留在配置条日期字段下方）。
- 窄屏（≤700px）配置条纵向堆叠；所有字段宽度 `min-width:0` 防溢出。

### 3.3 运行状态条（新组件）

- 显示条件：`btRunning || btLive.running`（与现监控窗条件一致）；
- 内容：「回测中 N%」+ 进度条（`btLive.pct`）+ 关键数（价格 `mktPrice(btLive.last_price)` / 持仓 / 现金）+ 右侧「查看实时监控 →」；
- 点击整条或「查看实时监控」→ 切到「实时监控」Tab；
- 样式：紫色淡渐变底 + `rgba(125,59,237,.4)` 描边（与持续进化运行态同语言）；
- 进度条颜色用 `--accent`，**不用**现模板里的 `bg-blue-500`（Tailwind 任意色，改为令牌）。

### 3.4 主 Tab（新组件：下划线式）

- 新状态 `btTab = ref('result')`，枚举 `result | monitor | tools | history`；
- 形态：线性 Tab 行 + `--accent` 下划线指示条（`.bt-tabs` / `.bt-tab.on::after`），底部 1px `--border` 分隔线；
- **徽标**：`btLive.running` 时「实时监控」Tab 挂琥珀徽标（运行中）；配置条任一字段变化后（`bt` 深监听或 `watch` 全字段）「结果」Tab 挂「配置已改」徽标，`loadBtResult/runBacktest` 成功后清除；
- **自动切换规则**（Tab 行为的核心）：
  1. 点「开始回测」→ 切到 `result`（跑完直接看结果）；运行中用户可自由切到 monitor；
  2. `runGridScan/compareStrategies/runOverfit` 任一发起 → 切到 `tools` 并把子 Tab 设为对应项（`btToolTab = ref('overfit'|'gridscan'|'compare')`）；结果到达时若用户仍在 `tools` 保持，否则不动（不打断用户）；
  3. 历史行点击 `loadBtResult(r.id)` → 成功后切回 `result`；
  4. 状态条点击 → `monitor`。
- `btGoto(section)` 改写为「section → btTab 映射 + `nextTick` 滚动到对应锚点」，方法名与调用点不变。

### 3.5 Tab 内容（现区块整体迁入，功能零丢失）

| Tab | 迁入内容 | 迁移注意 |
|---|---|---|
| result | 指标磁贴行（`btMetrics.cards` v-for）+ `btAdvice` 横幅 + 净值曲线窗（`#btChart`、`btRange` 分段、`btLegend` 图例）+ 逐笔成交窗（搜索/导出 CSV/表格/显示全部）+ 空态（「设置配置条 → 点开始回测」文案沿用） | 曲线容器 ID 不变；画布质感沿用持续进化（`.bt-chart-canvas`：`linear-gradient(180deg,rgba(125,59,237,.05),transparent)` + `1px dashed var(--border)` + `var(--r-md)`） |
| monitor | 现「回测实时监控」窗整体：4 磁贴（当前价格/持仓/现金/最近成交盈亏）+ 实时 K 线表 + 最近成交表 + 进度 | 原窗的 `v-if` 条件移除（改由 Tab 显隐承担）；进度信息与状态条同源 |
| tools | 内层子 Tab `.bt-subtabs`（下划线同族的小分段）：过拟合报告（`overfitResult`，判定徽标 `✓/✗` 保留为文字、`⚠` 换 `#i-alert` SVG——沿用持续进化页裁定）/ 参数网格扫描（`gridScanResult` + `plateauInfo` 热力图 `#plateauChart` + 应用最优参数）/ 策略对比（`compareResult` 表格） | 三块现窗体整体迁入；`v-if` 判定保留；热力图容器 ID 不变 |
| history | 现「历史回测记录」窗整体：分组表（`btHistoryGroups`）、加载态/错误态/空态、刷新与「新建回测」按钮（后者改为切回 `result` + 滚配置条） | 行点击 `loadBtResult` 不变 |

### 3.6 指标磁贴（新组件，状态光晕式——用户选定）

- 结构沿用持续进化 KPI 磁贴：标签行 / 大数值行（`tabular-nums`）/ 副行，`grid-auto-rows:1fr` + 三行 `nowrap+ellipsis`；
- 数据 = 现 `btMetrics.cards`（`label/val/cls/sub`）**零新计算**；
- 光晕规则：`label==='总收益'` 且 `cls=pos` → 绿光晕；`label==='最大回撤'` → 红光晕；其余磁贴素色（数字色仍按 `cls`）；
- 网格：`repeat(auto-fit,minmax(150px,1fr))` 全档自适应；`≤700px` 收窄为 `minmax(120px,1fr)`（磁贴数由 `btMetrics.cards` 决定，不写死列数）。

## 4. 视觉规范

- 令牌同全站（`--bg-0/1/2`、`--border/--border2`、`--accent #7D3BED`、`--green #0DCC82`、`--red #F5475E`、`--orange #F59E0A`）；无新色值；
- **禁止 emoji**：回测视图内新增行不用任何表情符号；现存 `⚠`（过拟合标题栏「无法判定」、提示处）换 `#i-alert` SVG 或「警告：」文字；`✓/✗` 判定文字保留（持续进化裁定）；删除按钮 `✕` 改为文字「删除」；
- 主 Tab 下划线 `2px var(--accent)`，指示条过渡 `transition:transform .15s`；
- 运行态紫光晕、磁贴光晕、曲线画布质感三种「突破形态」与持续进化页完全同源（同 rgba 值、同结构类名前缀 `.bt-`）；
- 断点：`≤1100px` 磁贴 4 列；`≤700px` 配置条纵排、Tab 行横向滚动（`overflow-x:auto` + 隐藏滚动条样式沿用站内做法）、磁贴 2 列；
- 退役样式：`.bt-layout` 双栏 grid 及其断点规则、原配置面板 `mac-form-section` 堆叠容器的布局职责（分段标题样式 `mac-form-section-title` 可复用于浮层与配置条分组）。

## 5. 错误与空态

| 场景 | 表现 |
|---|---|
| 未跑过回测 | `result` Tab 现空态文案沿用（「在左侧配置参数…」改为「设置上方配置，点击开始回测」） |
| 历史加载失败 | `btHistoryError` 现错误态 + 重试按钮原样（在 history Tab 内） |
| 回测运行中 | 状态条出现、CTA 禁用「回测中…」、监控 Tab 徽标；用户可自由切 Tab |
| 工具无结果 | 子 Tab 内显示各自现空态/引导（未运行时提示「先点配置条上的对应工具」） |
| 配置条字段非法/后端报错 | 沿用现 toast / 错误提示，不新增处理逻辑 |

## 6. 不变项清单（验收红线）

1. `web/api/*`、`core/`、`engine/`、`drl/` 零 diff；
2. §2.2(2) 列出的全部字段/方法原名可用且模板中有引用；
3. `#btChart`、`#plateauChart` 唯一且渲染逻辑不改；
4. 既有轮询与进度推送策略不改；
5. `tests/test_backtest_ui_contract.py` 既有 3 测试继续通过；
6. 方案保存/应用、CSV 导出、历史加载、参数扫描「应用最优参数」、过拟合检测等按钮链路逐一可用。

## 7. 测试与验收

- **契约测试**（追加到 `tests/test_backtest_ui_contract.py`，TDD 先红后绿）：Tab 结构存在（`btTab`/`.bt-tabs`/四 Tab 枚举）、配置条字段齐全、状态条条件、磁贴光晕类、画布质感、`btGoto` 的 Tab 映射、回测视图 emoji 扫描（位码表法，同持续进化）、退役类（`.bt-layout`）消失；
- **回归**：`pytest tests/test_backtest_ui_contract.py` + 持续进化契约（防跨页污染）+ `scripts/check_js.py` + `scripts/build_frontend.py --check-only` + 后端零 diff 红线；
- **人工验收**（三档宽度 1440/1000/700 + 交互）：配置条直跑流程、四个 Tab 切换与自动切换规则、状态条→监控、历史行点击→回结果、高级参数浮层（点外部/Esc 关闭、锚定）、工具子 Tab、空态/运行态、无横向滚动、控制台无报错；
- **scoped staging**：与并行工作流共存，提交仅含本任务 hunk。

## 8. 风险与对策

- **Tab 化后 `v-if` 复活的容器**：监控/工具/历史内容若用 `v-show` 常驻 DOM，`#plateauChart` 等 ECharts 容器在隐藏期 resize 会出错——对策：隐藏 Tab 用 `v-if`（按现有 `v-if` 迁移最自然），ECharts 首次可见时按现逻辑初始化；`btGoto` 切换用 `nextTick` 后滚动；
- **配置条 watch 误报「配置已改」**：`loadBtResult` 回填结果时也会写 `bt`——用「回填期间暂停 watch 标记」（布尔 guard）或对比指纹，实施计划中定稿；
- **`formDrop` 单槽冲突**：交易对/策略/保存方案/高级参数四个浮层共用 `formDrop` 单槽 + `popPos`——互斥行为与现状一致，高级参数新增 key `btAdvanced` 即可，无并发冲突；
- **历史「新建回测」按钮**原为滚动锚点，Tab 化后需先切 `result` 再滚配置条（§3.4 规则覆盖）。
