# 持续进化页 KPI 仪表盘 UI 重设计

- 日期：2026-09-27
- 状态：设计已经用户逐屏确认（可视化伴侣 v1→v4 稿，v4 为最终稿）
- 设计稿留存：`.superpowers/brainstorm/1-1790500775/content/`（layout-options / style-refinement v2-v4）
- 实施入口：`web/static/index.html` 中 `view==='evolve'` 的模板（3428-3976 行）与 evolve 页局部样式区（1336-1423 行）

## 1. 背景与目标

现版持续进化页（2026-09-07 PDF 工作台改造）功能完整，但用户判定为「布局、层级、视觉都要整体重做」：

1. **信息层级乱**——状态、曲线、模型卡、配置混在一起，一眼看不出「现在训练得怎么样、有没有问题」；
2. **视觉不够精致**——卡片、间距、配色需要升级成更现代的设计；
3. **布局要重新组织**——含页头操作区、双栏主区、下方 DRL 实验台的整页重排。

新设计的核心原则（用户选定）：**监控优先**——打开这页第一眼回答「训练得怎么样」，健康度置顶，操作入口其次。

一致性约束（用户选定）：**沿用全站深色体系，允许局部突破**——复用现有色板/字体/mac-* 组件，允许为该页引入新组件形态（KPI 磁贴、状态光晕、下拉浮层）。

## 2. 范围

### 2.1 在范围内

- `view==='evolve'` 整页模板重排：页头、主工作区、模型状态、DRL 实验台容器、配置/触发入口；
- evolve 页局部 CSS 重写（新增 KPI 磁贴、下拉浮层样式；退役 evolve-layout 双栏规则）；
- 新增 KPI 数据的**纯前端计算**（复用现有 API，不新增/修改任何后端接口）。

### 2.2 不在范围内

- **零后端改动**：`web/api/evolve.py`、`drl/evolve_engine.py` 不动；
- **零数据绑定改动**：所有 Vue 响应式字段、方法名、API 路径、轮询策略（status 5s / rounds 10s、与 visibilitychange 解耦）、ECharts 容器 ID `evolveCurveChart` 全部保留原名原逻辑；
- 其他页面、侧边栏、顶栏不动；
- 「导出进化日志」按钮维持 disabled（后端无接口，非本次范围）。

## 3. 页面结构（v4 最终稿）

自上而下单列全宽（**右栏取消，不再有 evolve-layout 双栏**）：

```
┌ 页头：持续进化 + 副标题          [暂停] [启用] [配置/触发] [导出进化日志] [刷新状态]
├ KPI 磁贴 ×7（4 列 × 2 行，状态光晕）
├ 训练轮次曲线面板（全宽）：标题+双分段控件 / 统计行 / 图表 / 图例
├ 「模型状态 · 部署与成果」标签 + 模型两卡（2 列）
└ 手动 DRL 实验台（折叠条，展开后四模式，容器与逻辑不变）
```

### 3.1 页头

- 左：标题「持续进化」+ 副标题「自优化模板 · 轮次迭代 · 版本回退」（与现有 `META` 一致）；
- 右按钮区，全部使用站内 SVG 图标（`#i-pause`/`#i-play`/`#i-power`/`#i-refresh`/`#i-settings`），**禁止任何 emoji/符号表情**：
  - 暂停/恢复（现有 `evolvePause`/`evolveResume`，条件与禁用逻辑不变）；
  - 启用（现有 `evolveEnable`，仅未启用时显示）；
  - **配置 / 触发**（新按钮，紫色描边，见 §3.6）；
  - 导出进化日志（维持 disabled + title 提示）；
  - 刷新状态（现有 `fetchEvolveStatus`）。
- 实现细化（2026-09-27）：页头不再重复放大标题与副标题——顶栏 `.tb-name` 已显示 `viewTitle/viewSub`（持续进化 + 自优化模板 · 轮次迭代 · 版本回退），页头仅保留操作按钮区（右对齐），避免同屏两份标题。

### 3.2 KPI 磁贴 ×7（新组件）

统一结构：**标签行（状态点+名称；状态点仅两循环与引擎保活三张）/ 大数值行 / 副文案行**；`grid-auto-rows:1fr` 等行高，三行均 `nowrap + ellipsis`，任何文案不得换行挤压（修复 v2 稿「运行中」扭曲问题）。

**质感 = 状态光晕式（用户选定 B）**：

| 状态 | 样式 |
|---|---|
| 运行中/健康 | 绿光晕：`border rgba(13,204,130,.45)` + `box-shadow 0 0 14px rgba(13,204,130,.12)` + 顶部淡绿渐变 |
| 暂停 | 琥珀光晕（同构，`#F59E0A`） |
| 告警（锚点锁死/保活降级） | 红光晕（同构，`#F5475E`） |
| 无状态 KPI | 素色 1px 边框 `--border`，零光效 |

**7 个磁贴与数据源映射（全部现有 API，前端计算）**：

| # | 磁贴 | 主数值 | 数据源（同源字段） | 光晕 |
|---|---|---|---|---|
| 1 | 因子挖掘 | 运行中/已暂停/已停用 | `GET /api/evolve/status` → `factor_miner.active`、`supervisor.loops_alive.factor_miner`；副行 = `next_run` 倒计时 + `factor_miner_interval`s/轮 | 运行绿/暂停琥珀 |
| 2 | 策略 DRL | 同上 | `strategy_drl.*`、`loops_alive.strategy_drl`、`strategy_drl_interval` | 同上 |
| 3 | 引擎保活 | 正常/降级 | `supervisor.degraded` 取反；副行 = `heartbeat_age` 秒数 + `restarts_by_loop` 自愈次数合计 | 正常素色/降级红 |
| 4 | 锚点健康 | `X / 2` | 现有 `evolveAnchorIssues` computed：`2 - evolveAnchorIssues.length`；副行 = 第一条 issue 的 `label`（有告警时） | 有告警红/全健康绿 |
| 5 | 累计轮次 | 两模型合计 | `factor_miner.rounds_total + strategy_drl.rounds_total`（`_pipeline_view` 已返回，零额外请求）；副行 = 当前曲线模型的今日轮次（`evolveRounds[model]` 按 `ts` 过滤当日，标注模型简称） | 素色 |
| 6 | 最佳 fitness | 两模型最大 fitness | 两模型 status 的 `fitness` 取最大，正值绿/负值红；副行 = 模型名 + `deployed_version`（如有）`v49 已部署` | 素色 |
| 7 | OOS 通过率 | 近 30 轮未拦截占比 | 与曲线红点**同一 `evolveVisual` 口径**：当前曲线模型最近 30 轮中非拦截轮占比；副行 = `20 过 / 10 拦` | 素色 |

> 同源约束（历史教训，`evolveCurveStat` 曾因两套口径打架）：#5/#7 的轮次分类必须复用 `evolveVisual` 的分类结果，**不另算一套**。

**磁贴交互**：#1-#2 点击 → `setEvolveCurveModel(对应模型)`（曲线切到该模型）；#4 有告警点击 → 打开配置/触发浮层并滚动到锚点段。

### 3.3 训练轮次曲线面板（全宽）

- 标题行：「训练轮次曲线」+ 模型分段（因子挖掘/策略 DRL）+ 范围分段（全部/最近 120/最近 50），沿用现有 `mac-seg` 与 `setEvolveCurveModel`/`setEvolveRange`；
- **统计行（独立成行，修复 v3 稿显示问题）**：现 `evolveCurveStat` 文案从标题行挪到标题下方独立一行，竖线分隔样式，空态显示「红点 = 拦截 / 回退轮次」；`evolveCurveNote`（聚合说明）保持在图例行；
- 图表：容器 ID `evolveCurveChart` 不变（ECharts 渲染、dataZoom、指纹门控逻辑全不动），高度由 220px 提升至 **260px**（全宽后比例更舒展）；
- 图例行：现有四色图例 + `evolveCurveNote`。

### 3.4 模型状态两卡

- 保留 `EVOLVE_MODELS` 配置驱动的 `v-for` 单模板（不回退成两份复制粘贴）；
- 卡片升级：所有字段/按钮/条件渲染照旧（候选/部署/上轮结论/热加载、锚点不一致、last_error、OOS 拦截、跨标 OOS、组合因子部署/回测/IC/部署与去回测按钮）；
- 视觉：实现细化：**部署旗舰卡紫色描边高亮：`deployed_fitness` 非空且为两模型中最高者（`evolveAccentModel` computed）；均无部署则不高亮**（`rgba(125,59,237,.5)` + 微光）；fitness 行保留上下边框分隔与大数字（正值绿/负值红）；
- 「查看版本」等按钮改用统一 `mac-btn btn-sm`，不使用内联 padding 拼凑（逐步清理本页内联样式）。
- 实现细化（2026-09-27 最终审查）：「查看版本」等按钮保留既有内联 padding——`.btn-sm` 全站未定义 padding，内联值承重；全站按钮 padding 收归 `.btn-sm` 另行立项。

### 3.5 手动 DRL 实验台

- 保持页底折叠条形态（`drlPanelOpen` 逻辑不变），标题 + 四模式摘要 pill + 展开按钮；
- 展开后内部四模式（RL 因子挖掘/自适应策略训练/已训练模型库/跨行情 OOS）的模板、表单、结果卡、Pine 代码块**全部原样保留**，只换容器标题样式。

### 3.6 配置 / 触发：页头按钮 + 下拉二级菜单（用户指定）

- 页头「配置 / 触发」按钮，点击弹出**下拉浮层**（用户否决了右缘把手抽屉方案）；
- 实现复用站内 `menu-drop` 模式：浮层 `Teleport to body`、按按钮位置锚定（按钮下方右对齐）、点外部/Esc 关闭、`max-height:70vh` 内部滚动；
- 浮层内容 = 原右栏三张卡的全部功能，零丢失：
  1. **训练标的池**（现 `formDrop==='evolveSymbols'` 多选二级菜单，含手输添加）+ **训练时间周期**（`mac-select`）；
  2. **手动触发训练**两键（`triggerFactorMiner`/`triggerStrategyDrl`，禁用与「训练中…」文案逻辑不变）；
  3. **回退基线（锚点）**：`evolveAnchorIssues` 列表 + 「重置锚点」（`resetEvolveAnchor`），空态时不渲染该段；
  4. **保存配置**（`saveEvolveConfig`，toast 反馈不变）；
- 窄屏（≤700px）浮层宽度 `min(330px, 92vw)`。
- 实现细化（2026-09-27）：浮层定位用既有 `popPos` 两段式（首帧 + nextTick 实测重定位）+ `_evolvePanelLeft` 按钮右缘右对齐覆盖（`w=min(330, innerWidth*0.92)`，8px 视口钳制），`evolvePanelStyle` 经 `:style` 绑定。另：evolve 视图内不再有 `⚠` 文本，警告图标统一 `#i-alert` SVG。
- Esc 关闭：浮层打开期间注册 document keydown，Escape 关闭并注销（与点外部关闭同路径）。

### 3.7 版本回退弹层

- `evolveVersionsOpen` 弹层（680px 表格 + 回退按钮 + 空态）**原样保留**，不改交互，仅保证与新页面视觉一致（沿用现样式即可）。

## 4. 视觉规范

- 沿用设计令牌：`--bg-0 #0A0A0A` / `--bg-1 #141417` / `--bg-2 #1C1C21` / `--border #262629` / `--text #FAFAFA` / `--muted #A1A1AB` / `--text3 #70707A` / `--accent #7D3BED` / `--green #0DCC82` / `--red #F5475E` / `--orange #F59E0A`；
- 字号体系沿用 `--fs-caption/--fs-body/--fs-footnote/--fs-micro`，KPI 主数值 `~16-17px SemiBold + tabular-nums`；
- **禁止 emoji**：所有图标用站内 SVG sprite（`#i-*`），文案层面不出现 ⚠️📈🧪 之类的表情符号（v2 稿用户明确要求）；
- 卡片：`border-radius 8px`、1px 细边框、无堆叠阴影；光晕只用于 KPI 磁贴与部署版模型卡；
- 响应式断点：`>1100px` KPI 4 列 / 模型 2 列；`700-1100px` KPI 2 列 / 模型 2 列；`<700px` KPI 1 列 / 模型 1 列，页头按钮保持文字不缩图标（沿用现断点习惯）；
- 退役样式：`.evolve-layout/.evolve-main-column/.evolve-side-column` 双栏规则及其 `grid-row:1` 相关注释块（该布局不再存在，注释里的推演结论随规则一起移除，避免误导后人）。

## 5. 错误与空态

| 场景 | 表现 |
|---|---|
| status 轮询连续 ≥3 次失败（`evolveFailStreak≥3`，与页顶黄色提示条同口径）→ KPI 全部素色，主数值 `—`，副行「状态未知」；单次失败保留上次值（避免抖动误清空） | 页头按钮维持现有禁用态 |
| 轮次加载失败 | 统计行显示现有「训练轮次加载失败，请稍后重试」；图表区沿用现有 `evolveRoundsErr` 分支 |
| 无训练轮次 | 图表区现有空态文案不变；#5/#7 KPI 显示 `—` |
| 无锚点告警 | #4 磁贴绿光晕 `2 / 2`；浮层锚点段不渲染 |
| 引擎禁用/暂停 | #1-#2 磁贴如实反映（停用态用素色，不误用绿色） |

## 6. 不变项清单（验收红线）

1. `web/api/evolve.py`、`drl/evolve_engine.py` 零 diff；
2. 模板中所有既有响应式字段与方法仍有引用（`fetchEvolveStatus/evolvePause/evolveResume/evolveEnable/saveEvolveConfig/triggerFactorMiner/triggerStrategyDrl/resetEvolveAnchor/showEvolveVersions/evolveRollback/deployEvolveFactor/goEvolveComboBacktest/setEvolveCurveModel/setEvolveRange/drlPanelOpen/drlMode/...`）；
3. `#evolveCurveChart` 唯一且仍归持续进化曲线渲染；
4. 轮询策略不变：status 5s、rounds 10s、与页面可见性解耦、回前台补刷；
5. DRL 实验台四模式与版本回退弹层功能零丢失。

## 7. 测试与验收

- **回归**：现有 `tests/test_evolve_*.py`（anchor/api_guards/audit/deployment/keepalive/runtime_reload）全部通过——因后端零改动，预期直接通过；
- **静态检查**：模板字段/方法引用完整性（对照 §6.2 清单逐项 grep）；evolve 视图模板内无 emoji 字符；
- **视觉验收**（1440 / 1000 / 700 三档）：KPI 磁贴行高等高、无文案扭曲换行；配置/触发浮层锚定正确、点外部关闭、窄屏不溢出；曲线统计行独立成行不与分段按钮挤压；模型卡部署版紫描边；页面无横向滚动条；
- **交互点检**：KPI #1-#2 点击切曲线模型；#4 点击开浮层；暂停/恢复/启用/刷新/保存配置/手动触发/查看版本/回退/重置锚点/实验台四模式逐一可用。

## 8. 风险与对策

- **KPI 计算污染现有 computed**：新增独立 computed（如 `evolveKpiTiles`），不修改 `evolveCurveStat/evolveVisual` 内部逻辑，同源读取、各自输出；
- **下拉浮层遮挡**：Teleport 到 body + z-index 对齐现有 `menu-sub` 层级；实测与版本弹层不同时打开（两个入口互斥交互，无冲突）；
- **浮层内表单状态**：沿用现有 `evolveConfig` 对象与「轮询不回填正在编辑的表单」策略，浮层开合不触发重新回填。
