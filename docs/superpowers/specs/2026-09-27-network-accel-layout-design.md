# 网络加速页 UI 布局重构 · 设计文档

日期：2026-09-27 ｜ 范围：`web/static/index.html` 单文件 ｜ 性质：纯布局与样式重构

## 1. 背景与问题

「系统设置 → 网络加速」（`settingsTab==='proxy'`，`web/static/index.html` L2267–L2304）存在三个具体问题：

1. **结构 bug**：`.settings-wrap` 是 `display:flex`（横向，L582），但该 tab 渲染了**两个并列的 `.settings-panel` 兄弟节点**（L2267 与 L2279）。其他 8 个设置页都只渲染一个 panel，因此网络加速页实际是**左右分栏挤压**，而非预期的上下堆叠。
2. **样式缺失**：标题使用的 `.settings-head` 类在整份 CSS 中**无任何定义**，标题呈无样式裸文字，与其他设置页统一的 `<h2 class="text-[length:var(--fs-body)] font-bold mb-1">` 不一致。
3. **视觉语言不统一**：大量裸内联 `style="..."`；节点状态用纯文本（`可连 120ms`／`不可达`）而非项目现成的 `pill` / `mac-card` 体系；无状态总览。

## 2. 目标与非目标

**目标**
- 修复双 panel 横向分栏 bug，恢复与其他设置页一致的单 panel 结构。
- 统一标题层级与视觉语言（`<h2>` + 副标题 + `mac-card` + `pill`）。
- 用「顶部状态条 + 三步向导 + 节点卡片网格」重组信息，提升可读性与扫读效率。

**非目标（红线）**
- 不改任何 JS 逻辑、函数、接口、返回字段。
- 不新增接口、不新增请求、不改变数据流。
- 不触碰其余 8 个设置页 tab 与全局 CSS 类。
- 不做节点排序/重测/一键清空等新交互（超出既定范围）。

## 3. 已确认的决策

| 决策点 | 结论 |
|---|---|
| 改造范围 | 仅布局与样式重构，不动 JS 与接口 |
| 布局方案 | **C · 三步向导式**（浏览器可视化三选一，用户点选 C） |
| 手动代理与状态 | **顶部加一条状态条**（补齐 C 的两个短板），手动地址保留在步骤③底部 |
| 流程确认 | 用户授权：此类小决策不再逐一询问，直接执行 |

## 4. 信息架构（最终形态）

单个 `.settings-panel`，纵向五段：

```
<h2>网络加速</h2>
<p>副标题：三步完成配置 · 仅本软件生效（解释出站请求范围）</p>
[状态条]  当前模式 │ 生效地址 │ hysteria 客户端 │ 已选节点
[步骤 1]  准备客户端（+ pill 状态 / 自动下载按钮）
[步骤 2]  粘贴订阅或节点文本（textarea + 扫描按钮 + proxyScanMsg）
[步骤 3]  选择要使用的节点（卡片网格）
          ── 或者手动填写地址 ──
          [input proxyUrl] [保存并应用]
```

## 5. 状态条字段映射（零新增接口）

| 字段 | 来源 | 展示 |
|---|---|---|
| 当前模式 | `proxyAuto` | `自动探测本机代理` / `手动指定` |
| 生效地址 | `proxyResolved`，回落 `proxyUrl` | 均为空 → `—` |
| hysteria 客户端 | `hyClient` | `pill`：`installed`=绿`已就绪`；`downloading`=橙`下载中…`；`error`=红`下载失败`；否则灰`未安装` |
| 已选节点 | `proxyUrl` 首段协议 + 地址 | 空 → `未设置` |

## 6. 视觉规范

- **状态条**：`mac-card` 内 `display:grid;grid-template-columns:repeat(4,1fr)`，每格 `mac-label` 小标 + 数值；窄屏降 2 列（`max-width` 断点或 `repeat(auto-fit,minmax(150px,1fr))`）。
- **步骤圆点**：20px 圆形，`background:var(--accent)`、白色 700 字重数字，与 `mac-form-section-title` 图标气质对齐；步骤标题 600 字重 + 状态 `pill`；步骤卡本体 `mac-card`。
- **节点卡片**：`display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr))`（3→2→1 列自然降级），卡内：名称（600）／`type · server:port`（muted）／延迟 pill + 选择按钮。
- **延迟 pill 三档**：`<150ms` 绿、`150–400ms` 橙、`不可达/需客户端` 灰（复用现有 `.pill` 的 `green/orange` 与默认色）。
- **选中态**：当前生效节点加 accent 描边（`border-color:var(--accent)`）+ 主按钮文案 `使用中`。
- **列表容器**：`max-height` + `overflow:auto`，避免撑长设置页。

## 7. 边界态（全部复用现有状态，不加逻辑）

| 条件 | 展示 |
|---|---|
| `hyClient !== 'installed'` | 橙 pill `未安装` / 红 pill `下载失败` + `自动下载` 按钮（`v-if` 沿用现状） |
| `proxyNodes.length === 0` | 虚线空态框：`还没有节点——先在上一步粘贴订阅并扫描` |
| `proxyScanning` | 扫描按钮文案 `扫描中…` + `proxyScanMsg` 橙色提示 |
| `proxyScanMsg` 非空 | 原样显示在扫描按钮右侧（现状保留） |
| 节点不可达 / 需客户端 | 选择按钮 `disabled`（`:disabled` 表达式原样保留） |

## 8. 错误处理

- 状态条字段缺失一律渲染 `—`，不抛错、不新增 try/catch。
- `loadProxy()` / `saveProxy()` 等现有 `catch` 行为保持不变。

## 9. 改动清单

**唯一改动文件：`web/static/index.html`**

1. **L2267–L2304**：合并两个 `settings-panel` 为一个；`.settings-head` → `<h2>`；重建为状态条 + 三步骤结构。
2. **`<style>` 区**（约 L842 `mac-form-section` 附近新增）：新增约 40 行本页专用类——`proxy-status`、`proxy-step`、`proxy-node-grid` 等，全部走现有 CSS 变量（`var(--accent)` / `var(--border)` / `var(--r-md)` / `var(--fs-*)`），不修改已有全局类。

**明确不动**：`saveProxy` / `scanProxy` / `chooseProxy` / `downloadHyClient` / `loadProxy` 五个函数；所有 `v-model` / `@click` / `:disabled` / `:key` 绑定表达式（仅挪位置）；`settingsTab==='proxy'` 判断；`/proxy` 系列接口；其余 8 个 tab。

**Vue 模板注意**（L1918 项目内既有教训）：in-DOM 模板中自定义组件必须成对闭合，`/>` 会吞掉后续兄弟节点。

## 10. 验证方案

1. 访问 `?tab=proxy` 查看该页；切换其余 8 个 tab 确认无串扰。
2. 控制台无 Vue 模板编译报错。
3. 四态手测：默认态 / 粘贴订阅扫描 / 点选节点 / 清空输入保存。
4. 窄窗口拖动：状态条 4→2 列，节点网格 3→2→1 列。
5. 静态检查：`.settings-head` 全文 0 处；`settingsTab==='proxy'` 的 `.settings-panel` 恰好 1 个。

## 11. 交付物

- `web/static/index.html` 改动
- 本设计文档
- 实施计划（由 writing-plans 产出）
