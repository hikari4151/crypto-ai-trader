# 持续进化 PDF 设计稿 UI 改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将持续进化页主内容区重排为用户 PDF 设计稿中的深色双栏工作台，同时完整保留现有功能、数据绑定和侧边栏。

**Architecture:** 在现有 CDN Vue 单文件页面中，仅重写 `view==='evolve'` 的页面模板布局，并补充持续进化页专用 CSS。现有状态对象、计算属性、方法、API 请求和 ECharts 容器 ID 保持不变；通过 CSS Grid 实现桌面双栏和窄屏单栏，通过统一的状态带、卡片和右栏操作区承载已有功能。

**Tech Stack:** Vue 3 CDN、现有 `web/static/index.html`、Tailwind utility classes、现有 macOS 风格 CSS 变量、ECharts、pytest。

## Global Constraints

- 只改持续进化主内容区，不修改侧边栏、全局导航或其他页面。
- 保留现有持续进化 API、轮询、状态对象、配置保存、训练触发、版本查看、版本回退、锚点重置和手动 DRL 实验台逻辑。
- 保留 `evolveStatus`、`evolveConfig`、`evolveRounds`、`EVOLVE_MODELS`、DRL 状态及现有方法名和关键图表容器 ID。
- 以用户提供的 PDF 为唯一页面结构依据；不引入新的产品功能或后端接口。
- 桌面端为左主栏 + 右操作栏，窄屏堆叠，任何按钮、标签、模型卡和表单都不得横向溢出。
- 颜色沿用现有 CSS 变量：主动作 `var(--accent)`，成功绿色，警告橙色，危险红色；不增加渐变、插图或新的侧边栏样式。
- 每个任务完成后运行对应的静态检查或测试；不重置或覆盖工作区中与本任务无关的已有修改。

---

### Task 1: 建立持续进化页布局骨架

**Files:**
- Modify: `web/static/index.html:2558-2783`（持续进化主页面模板）
- Modify: `web/static/index.html`（持续进化页局部样式区，保持在同一文件现有样式体系内）

**Interfaces:**
- Consumes: 现有 `view`, `evolveStatus`, `evolveFailStreak`, `evolveAnchorIssues`, `evolveAnchorLocked`, `EVOLVE_MODELS`, `evolveModel`, `evolveConfig`, `drlPanelOpen`。
- Produces: 设计稿对应的持续进化主页面 DOM 骨架，继续暴露 `#evolveCurveChart` 给现有 ECharts 渲染逻辑，继续暴露所有现有事件处理入口。

- [ ] **Step 1: 记录现有页面关键绑定清单**

在编辑前用 `rg` 确认以下绑定仍存在并记录行号：

```bash
rg -n "evolveStatus|evolveAnchorIssues|evolveConfig|evolveCurveChart|EVOLVE_MODELS|drlPanelOpen|triggerFactorMiner|triggerStrategyDrl|triggerMetaController" web/static/index.html
```

预期：这些字段和方法在脚本区存在，且 `#evolveCurveChart` 仅用于持续进化曲线。

- [ ] **Step 2: 替换页面外层为标题区、状态带和双栏容器**

保留标题和副标题，按 PDF 增加标题区右侧操作按钮；把原先四个纵向 `mac-window` 主块改为以下结构语义：

```html
<section v-if="view==='evolve'" class="evolve-page">
  <div class="evolve-page-head">
    <div>标题和副标题</div>
    <div class="evolve-page-actions">导出进化日志 / 刷新状态</div>
  </div>
  <div class="evolve-status-bar">运行状态、标的池、模块状态、下一次训练、暂停/禁用/立即训练</div>
  <div class="evolve-layout">
    <main class="evolve-main-column">
      训练轮次曲线
      模型状态
    </main>
    <aside class="evolve-side-column">
      训练配置
      回退基线
      手动触发训练
    </aside>
  </div>
  手动 DRL 实验台
</section>
```

不要删除现有事件绑定；按钮只移动到新容器。

- [ ] **Step 3: 添加只作用于持续进化页的 CSS**

在现有样式区域加入命名空间为 `.evolve-page` 的样式，至少包含：

```css
.evolve-layout { display:grid; grid-template-columns:minmax(0,1fr) 370px; gap:16px; align-items:start; }
.evolve-main-column, .evolve-side-column { min-width:0; display:grid; gap:16px; align-content:start; }
.evolve-status-bar { display:grid; grid-template-columns:auto repeat(4,minmax(0,1fr)) auto; gap:14px; align-items:center; }
.evolve-model-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
@media (max-width: 1100px) { .evolve-layout { grid-template-columns:1fr; } .evolve-side-column { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 760px) { .evolve-status-bar, .evolve-side-column, .evolve-model-grid { grid-template-columns:1fr; } }
```

结合现有变量补充卡片内边距、按钮换行、表格/长标签的 `min-width:0` 和 `overflow-wrap`，不要修改全局 `.mac-window` 或侧边栏样式。

- [ ] **Step 4: 运行结构静态检查**

运行：

```bash
python - <<'PY'
from pathlib import Path
s = Path('web/static/index.html').read_text(encoding='utf-8')
assert "view==='evolve'" in s
for token in ('evolve-layout', 'evolve-status-bar', 'evolveCurveChart', 'evolve-model-grid'):
    assert token in s, token
print('evolve layout markers: PASS')
PY
```

预期：输出 `evolve layout markers: PASS`。

- [ ] **Step 5: Commit**

```bash
git add web/static/index.html
git commit -m "style: reshape continuous evolution page layout"
```

若工作区中存在用户未提交修改，不要提交无关文件；提交前用 `git diff -- web/static/index.html` 确认差异只涉及本任务页面和局部样式。

---

### Task 2: 按 PDF 重排运行状态、配置、锚点和触发区

**Files:**
- Modify: `web/static/index.html`（Task 1 中持续进化模板）

**Interfaces:**
- Consumes: `evolveStatus`, `evolveFailStreak`, `evolveAnchorIssues`, `evolveAnchorLocked`, `evolveConfig`, `evolveSymbolOptions`, `evolveSymbolsLabel`, `evolveTfOptions`, `formDrop`, `formDropStyle`。
- Produces: 设计稿顶部状态条及右栏三个操作面板，保留现有暂停、恢复、启用、禁用、刷新、保存配置、重置锚点和三种立即训练入口。

- [ ] **Step 1: 将运行状态压缩为单行状态带**

把旧的“运行状态”窗口内容收拢到 `.evolve-status-bar`，保留：

- `evolveStatus.running`、`evolveStatus.paused`、`evolveStatus.enabled` 的状态显示；
- `evolveStatus.symbols` / `current_symbol` 的标的池；
- `factor_miner.active`、`strategy_drl.active`、`meta_controller.active`；
- `next_run` 或 `next_train` 可用值；
- 暂停、恢复、启用、禁用、刷新和设计稿中的“立即训练”主按钮。

连续失败提示 `evolveFailStreak >= 3` 保留，但放在状态带上方作为窄告警行，避免它把主栏内容向下推得过多。

- [ ] **Step 2: 将训练配置收拢到右栏卡片**

把现有训练标的池下拉和时间周期选择移动到 `.evolve-side-column` 的训练配置面板，保留：

```html
@click.stop="toggleFormDrop('evolveSymbols', $event)"
@click="toggleEvolveSymbol(s)"
@keydown.enter.prevent="addEvolveSymbolManual()"
<mac-select v-model="evolveConfig.timeframe" :options="evolveTfOptions"></mac-select>
@click="saveEvolveConfig()"
```

保留 `Teleport` 菜单结构，确保下拉菜单仍通过 `formDropStyle` 定位。

- [ ] **Step 3: 将锚点问题改为右栏常驻告警卡**

保留 `v-if="evolveAnchorIssues.length"`，但使用设计稿中的告警卡结构：每个 issue 展示 `a.tag`、`a.label`、`a.fitness`、`a.source`、`a.streak`、`a.stallRounds`、`a.reason`，并在 `a.resettable` 时调用：

```html
@click="resetEvolveAnchor(a.model)"
```

使用 `evolveAnchorLocked` 决定危险边框，否则使用警告边框；不要再让锚点卡占据左侧主栏。

- [ ] **Step 4: 将手动训练触发器改为右栏纵向按钮组**

保留三组现有按钮与禁用条件：

```html
@click="triggerFactorMiner()"
@click="triggerStrategyDrl()"
@click="triggerMetaController()"
```

按钮标签继续反映 `active` 状态，例如“因子挖掘中…”、“策略训练中…”、“元策略训练中…”。按 PDF 让按钮在右栏垂直堆叠并占满可用宽度。

- [ ] **Step 5: 运行绑定检查**

```bash
python - <<'PY'
from pathlib import Path
s = Path('web/static/index.html').read_text(encoding='utf-8')
for token in ('toggleFormDrop(\'evolveSymbols\'', 'saveEvolveConfig()', 'resetEvolveAnchor', 'triggerFactorMiner()', 'triggerStrategyDrl()', 'triggerMetaController()'):
    assert token in s, token
print('evolve controls: PASS')
PY
```

预期：输出 `evolve controls: PASS`。

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html
git commit -m "style: align evolution controls with pdf layout"
```

---

### Task 3: 按 PDF 重排曲线、模型卡和版本弹层

**Files:**
- Modify: `web/static/index.html`（持续进化模板和局部样式）

**Interfaces:**
- Consumes: `evolveCurveModel`, `evolveCurveHasData`, `evolveRoundsErr`, `EVOLVE_MODELS`, `evolveModel`, `showEvolveVersions`, `evolveVersionsOpen`, `evolveVersionTitle`, `evolveVersions`, `evolveBestVersion`, `evolveRollback`, `closeEvolveVersions`。
- Produces: 左栏曲线和三卡横排模型区，以及 PDF 风格版本弹层。

- [ ] **Step 1: 将训练轮次曲线放入主栏首卡**

保留三个模型切换按钮和以下图表容器：

```html
<div id="evolveCurveChart" v-if="evolveCurveHasData"></div>
```

空数据时保留“加载失败”和“暂无训练轮次记录”两种文案分支。将图例信息压缩到曲线卡底部，保留红点代表拦截/回退、蓝线代表接受/训练轮次和最佳 fitness/接受率摘要。

- [ ] **Step 2: 将模型状态改为三卡横排**

继续使用单一 `v-for`：

```html
<div v-for="m in EVOLVE_MODELS" :key="m.key" class="evolve-model-card">
```

每张卡至少保留：

- `m.name`、`m.icon`、PPO pill；
- `evolveModel(m.key).fitness`；
- symbol、episode、last_run；
- 候选 fitness、部署 fitness、版本、last outcome、runtime reload；
- anchor mismatch、last error、oos rejected、cross OOS、meta 子策略和 next run；
- `@click="showEvolveVersions(m.key)"`。

不要改变 `EVOLVE_MODELS` 数据数组和 `evolveModel()` 方法。

- [ ] **Step 3: 用边框/状态色表达模型异常**

卡片边框继续由 `m.border` 驱动；当模型有 `last_error`、`oos_rejected` 或 `anchor_mismatch` 时，在卡片内部只增加紧凑提示行，不新增图标装饰或改变数据逻辑。

- [ ] **Step 4: 对齐版本弹层**

保留 `evolveVersionsOpen` 独立遮罩和 `closeEvolveVersions()`；弹层表格保留版本、fitness、训练时间和操作列：

```html
<tr v-for="v in evolveVersions" :key="v.version">
```

最佳版本高亮，最佳版本禁用回退；非最佳版本继续调用：

```html
@click="evolveRollback(evolveVersionName, v.version)"
```

无版本时保留 `该模型暂无可回退版本` 空态。

- [ ] **Step 5: 运行模板绑定和重复 ID 检查**

```bash
python - <<'PY'
from pathlib import Path
import re
s = Path('web/static/index.html').read_text(encoding='utf-8')
assert s.count('id="evolveCurveChart"') == 1
for token in ('EVOLVE_MODELS', 'showEvolveVersions', 'evolveVersionsOpen', 'evolveRollback', 'evolveBestVersion'):
    assert token in s, token
print('curve/models/modal bindings: PASS')
PY
```

预期：输出 `curve/models/modal bindings: PASS`。

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html
git commit -m "style: refine evolution curve model cards and versions"
```

---

### Task 4: 按 PDF 收拢手动 DRL 实验台并验证窄屏

**Files:**
- Modify: `web/static/index.html`（手动 DRL 实验台模板和局部样式）

**Interfaces:**
- Consumes: `drlPanelOpen`, `drlMode`, `drlMineCfg`, `drlMineResult`, `drlCfg`, `drlTrain`, `drlModels` 及现有 DRL 方法。
- Produces: PDF 风格底部实验台，保留四种模式、所有参数、GPU/早停/续训/强制训练控制、训练进度和结果反馈。

- [ ] **Step 1: 保留实验台独立窗口与折叠入口**

保留：

```html
<button @click="drlPanelOpen=!drlPanelOpen">{{drlPanelOpen?'收起':'展开'}}</button>
<div v-show="drlPanelOpen">
```

将实验台标题、模式 tab、说明文案、参数区和进度区统一收进 `.evolve-lab`，不改变其显示逻辑。

- [ ] **Step 2: 将参数字段统一为紧凑网格**

保留已有字段和 `v-model.number`，使用响应式网格承载：

```html
<div class="evolve-parameter-grid">
  训练轮数 / K线数量 / 波动率风险惩罚 / 熵正则 / 回撤惩罚 / 连亏惩罚 / 趋势一致奖励 / 过拟合衰减
</div>
```

第二组学习率、折扣因子、起始资金字段继续使用同一网格；不删除因子挖掘模式的 `drlMine*` 字段。

- [ ] **Step 3: 将 GPU、续训、早停和训练按钮对齐到一行或两行**

保留以下现有行为与字段：

```html
@click="drlCfg.use_gpu=!drlCfg.use_gpu"
@click="drlCfg.force_train=!drlCfg.force_train"
<select v-model="drlCfg.base_model">
@click="startDrlTrain()"
@click="loadDrlModels()"
```

在窄屏下自动换行，确保控件和辅助文案不会溢出。

- [ ] **Step 4: 保留训练中、完成和 OOS 结果状态**

保留 `drlTrain.running` 进度条、episode/episodes、total_ret、best_ret、epsilon、policy_loss，以及 `drlTrain.done`、错误、OOS 报告和应用模型结果区域。

- [ ] **Step 5: 运行字段保留检查**

```bash
python - <<'PY'
from pathlib import Path
s = Path('web/static/index.html').read_text(encoding='utf-8')
for token in ('drlPanelOpen', 'drlMode', 'drlMineCfg', 'drlCfg.use_gpu', 'drlCfg.base_model', 'startDrlTrain()', 'drlTrain.running', 'drlTrain.done'):
    assert token in s, token
print('manual DRL lab bindings: PASS')
PY
```

预期：输出 `manual DRL lab bindings: PASS`。

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html
git commit -m "style: compact manual drl evolution lab"
```

---

### Task 5: 运行测试和浏览器级验收

**Files:**
- Modify: `web/static/index.html` only if verification exposes a layout or binding regression.
- Test: `tests/test_evolve_oos_gate.py`, `tests/test_evolve_population.py`, `tests/test_drl_optimizations.py` and any existing web/static smoke checks.

**Interfaces:**
- Consumes: Task 1-4 的完整页面实现。
- Produces: 通过后端回归测试、静态模板检查和桌面/窄屏视觉验收的持续进化页。

- [ ] **Step 1: 运行针对持续进化的 pytest**

```bash
pytest -q tests/test_evolve_oos_gate.py tests/test_evolve_population.py tests/test_drl_optimizations.py
```

预期：全部通过；若失败，区分是否由本次仅前端模板改动导致，不修改后端逻辑来掩盖失败。

- [ ] **Step 2: 启动 Web 服务并确认端口**

按仓库现有启动方式启动服务；如果默认端口已占用，选择另一个空闲端口。确认首页可访问后记录本地 URL。

- [ ] **Step 3: 浏览器检查桌面布局**

打开持续进化页并检查：

- 侧边栏与其他页面未改变；
- 标题区和状态带为一行主信息；
- 左侧曲线和模型横排卡可见；
- 右侧配置、锚点告警、手动触发按钮纵向排列；
- 模型卡没有文字或按钮溢出；
- 版本弹层打开后页面主体不跳动；
- 手动 DRL 实验台展开时参数网格、进度条和结果区完整可见。

- [ ] **Step 4: 浏览器检查窄屏布局**

将窗口切换到窄屏宽度，确认：

- 双栏堆叠为单栏；
- 状态带、右栏面板和模型卡改为单列；
- 触发按钮、训练配置和版本表不产生横向滚动；
- 下拉菜单仍能打开且不被父容器裁切；
- 实验台参数和按钮自动换行。

- [ ] **Step 5: 检查无数据和异常状态**

在不改变后端数据的前提下，验证模板分支：

- 曲线无数据和加载失败提示；
- `evolveFailStreak >= 3` 的陈旧提示；
- `evolveAnchorIssues.length` 为 0 时右栏不显示空告警；
- 模型无版本时的空态；
- 模型错误/OOS 拒绝/锚点不一致提示。

- [ ] **Step 6: 汇总差异并做最终静态检查**

```bash
git diff --check
git status --short
rg -n "view==='evolve'|evolve-layout|evolve-status-bar|evolveCurveChart|drlPanelOpen" web/static/index.html
```

预期：无空白错误；工作区只包含本次页面改造及必要规格/计划文件，侧边栏文件无修改。

- [ ] **Step 7: Commit**

```bash
git add web/static/index.html docs/superpowers/specs/2026-09-07-continuous-evolution-pdf-ui-design.md docs/superpowers/plans/2026-09-07-continuous-evolution-pdf-ui.md
git commit -m "feat: redesign continuous evolution page from pdf"
```
