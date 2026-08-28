# UI macOS 深度优化设计文档

> 日期：2026-08-28
> 目标：在保留顶部菜单栏的前提下，通过"组件套件先行"提升全站表格可读性、配置表单易用性与动效手感，使整体更接近 macOS 原生体验。
> 方案：B（组件套件先行 + 系统化落地）。已与用户确认：全都要、分阶段；保留顶部菜单栏；痛点为「表格太密 / 配置表单不好用 / 动效手感弱」。

---

## 0. 现状与约束

- 单文件 `web/static/index.html`，约 5116 行；20 张 `<table>`、43 个 `.mac-window`、9 个视图（仪表盘 / AI与API / 策略与工坊[含全部策略] / 历史回测 / 量化研究 / 持续进化 / 历史战绩 / 实盘启动）。
- 设计 token 已成熟：毛玻璃、`color-mix` 语义色、Apple 字阶（`--fs-*`）、圆角阶梯（`--r-*`）、spring 缓动（`--ease-spring`）、暗/浅双主题。
- 页面转场现状：`page` transition 仅 opacity + translateY(6px) 淡入，`mode="out-in"`。
- **验证约束**：内置浏览器视口隐藏，无法截图；验证方式为脚本断言计算样式/结构 + `check_js` 语法 + 用户在真实浏览器反馈迭代。
- **范围红线**：不做 F-24 文件拆分；不改后端、不动业务逻辑；本地化依赖（vendor/）不变；`prefers-reduced-motion` 全量降级；暗/浅双主题一致。

## 1. 组件套件（新增，单一来源）

在现有组件块之后扩出以下 token 驱动组件：

### 1.1 `.mac-table` 表格族
- 容器 `.mac-table`：圆角 `--r-md`、1px `--border`、`overflow:auto`、内嵌 `table`。
- 吸顶表头：`thead th{position:sticky;top:0}`，背景 `--bg-2`，保证滚动可读。
- 行悬停：`tbody tr:hover` → `var(--hover)`；可选斑马 `--zebra`。
- 数字对齐：`.num{text-align:right;font-variant-numeric:tabular-nums}`。
- 可排序表头：`.th-sort` + 升/降箭头（chevron），`aria-sort`。
- 密度：`.mac-table--compact`（行高收紧）/ 默认 / `.mac-table--comfort`。
- 空态：`.mac-table-empty`（居中 muted 文案 + 图标）。
- 加载骨架：`.mac-table-skeleton` 行。
- 次要列降级：`.cell-muted{color:var(--muted)}`。

### 1.2 `.mac-form` 表单族（系统设置式）
- `.mac-form-section`：分组卡片（`--bg-2`、圆角 `--r-md`、内边距），带可选分组标题 `.mac-form-section-title`。
- `.mac-form-row`：一行 = 标签（左，`--fs-body`）+ 控件（右）+ 说明（下，`--fs-caption`、`--muted`）；窄屏自动堆叠。
- `.mac-form-hint`：内联帮助文案。
- 复用现有 `mac-input / mac-toggle / mac-seg / mac-btn`；校验错误 `.mac-form-err`（`--red-text` 行内，不打断）。

### 1.3 `.mac-list-row` 列表行
用于全部策略 / 因子库等列表：左图标 + 标题/副标题 + 右侧动作，悬停高亮。

### 1.4 `.mac-toolbar` 工具条
表格/列表上方：搜索输入 + 密度切换（`mac-seg`）+ 筛选，间距统一。

## 2. 三大痛点落地

### 2.1 表格可读性（痛点 1）
把 20 张表套上 `.mac-table`，重点：量化研究、历史战绩、全部策略、回测记录。
- 次要列 `cell-muted`、数字 `num` 对齐。
- 密集表加密度切换（工具条内）。
- 空态/加载态补齐。

### 2.2 配置表单原生化（痛点 2）
AI 设置（settings 视图）、风控、回测参数三处改为 `.mac-form-section` 分组 + `.mac-form-row` + 内联帮助 + 清晰标签。

### 2.3 动效手感（痛点 3）
- 页面转场：`--ease-spring`，enter 加轻微 `scale(.995)`→`1`，保持 `out-in`。
- 列表/卡片/表格行错落入场（`nth-child` 递增 30-40ms）。
- 悬停/按压微交互加深（已有 `--press-ms`，统一应用）。
- 统一 `:focus-visible` 焦点环（2px `--accent`，offset 2）。
- `prefers-reduced-motion` 下全部动画 `none`。

## 3. 顶部菜单栏精修（保留，轻量、最后）
分组与溢出、激活/按压态、键盘焦点、下拉打磨。可选项。

## 4. 空态 / 加载 / 错误
表格空态文案、加载骨架、表单行内校验错误。

## 5. 结构与范围
- 仅改 `web/static/index.html`：扩 CSS 组件块 + 少量模板类名/包裹替换。
- 不新增文件、不改后端、不动业务逻辑。
- 保持 `check_js` 通过、暗/浅主题一致、reduced-motion 降级。

## 6. 验证方式
- 脚本断言：关键组件计算样式（圆角/间距/对齐/焦点环）符合规范；暗/浅双主题切换。
- `check_js`：内联 JS 语法。
- 双引擎一致性不受影响（不改后端）。
- 用户在真实浏览器看效果反馈，迭代。

## 7. 分阶段实施
1. **Phase 1**：组件套件 CSS + 表格可读性（收益最大）。
2. **Phase 2**：配置表单原生化（settings / 风控 / 回测参数）。
3. **Phase 3**：动效纵深（转场 / 入场 / 微交互 / 焦点）。
4. **Phase 4**：菜单栏精修 + 空态/加载/错误收尾。

## 8. 非目标（YAGNI）
- 不做侧边栏导航重构（用户明确保留顶部菜单栏）。
- 不做 F-24 文件拆分。
- 不引入新的前端依赖/构建步骤。
- 不改后端与数据结构。
