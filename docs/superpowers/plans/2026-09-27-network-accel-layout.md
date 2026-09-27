# 网络加速页 UI 布局重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复「系统设置 → 网络加速」双 `.settings-panel` 横向分栏 bug，并把该页重排为「状态条 + 三步向导 + 节点卡片网格」，视觉语言对齐现有 `mac-*` 组件套件。

**Architecture:** 单文件改动（`web/static/index.html`）。HTML 模板层把两个并列 panel 合并为一个、重建信息层级；CSS 层新增一段仅作用于 `.proxy-*` / `.pnode` 的局部样式，全部走既有 CSS 变量。**不改任何 JS 函数、接口、v-model/@click/:disabled 绑定表达式。**

**Tech Stack:** Vue 3 in-DOM template + 手写 CSS（token 驱动），Tailwind CDN 类名（`text-[length:var(--fs-caption)]` 等）既有用法沿用。

**Spec:** `docs/superpowers/specs/2026-09-27-network-accel-layout-design.md`

## Global Constraints

- 唯一改动文件：`web/static/index.html`（严禁触碰其余 8 个 `settingsTab` 面板）
- JS 红线：`saveProxy / scanProxy / chooseProxy / downloadHyClient / loadProxy` 五个函数一行不改；`v-model` `@click` `:disabled` `:key` 绑定表达式**原样保留**
- 不新增接口、不新增 ref/computed/method；模板内只允许内联三元/字符串方法等**纯模板表达式**
- in-DOM 模板铁律（项目 L1918 已有教训注释）：自定义组件必须成对闭合，`/>` 会吞掉后续兄弟节点
- 新增 CSS 只用既有变量：`var(--accent) var(--border) var(--border2) var(--r-md) var(--bg-2) var(--text) var(--muted) var(--fs-*) var(--lh-*)`；类名前缀限定 `proxy-` / `ps-` / `pnode`，不改已有全局类
- 删除 `.settings-head` 用法（该类全项目无定义），标题改 `<h2 class="text-[length:var(--fs-body)] font-bold mb-1">`，与其他 8 个设置页一致

---

### Task 1: 注入「网络加速」局部样式块

**Files:**
- Modify: `web/static/index.html:859`（`.mac-form-err` 行之后、`/* ---- 风控规则` 注释块之前）

**Interfaces:**
- Consumes: 既有 CSS 变量与 `.mac-card` / `.pill` / `.mac-input` / `.mac-btn` 类
- Produces: 类 `proxy-status` `ps-cell` `ps-val` `proxy-step` `ps-num` `ps-body` `ps-title` `ps-hint` `ps-actions` `proxy-node-grid` `pnode` `pnode-name` `pnode-meta` `pnode-foot` `proxy-or` `proxy-empty` `ps-manual` —— Task 2 的模板将引用这些类名，**必须逐字一致**

- [ ] **Step 1: 定位插入点**

Read `web/static/index.html` 第 855–862 行，确认 `.mac-form-err{...}` 行与其后的 `/* ---- 风控规则：组内两列（2026-09-19 第十轮）----` 注释相邻。

- [ ] **Step 2: 插入样式块**

在 `.mac-form-err{...}` 行之后插入（原样复制，勿改类名）：

```css
  /* ---- 网络加速（proxy tab）：顶部状态条 + 三步向导（2026-09-27 布局重构）----
     此前该 tab 写了两个并列 .settings-panel，而 .settings-wrap 是横向 flex ⇒ 左右分栏挤压。
     合并为单 panel 后，本段负责状态条/步骤/节点卡片三块的局部排版，类名一律带 proxy-/ps-/pnode 前缀，
     只用既有 token，不与全局组件类发生耦合。 */
  .proxy-status{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;
    background:color-mix(in srgb,var(--bg-2) 55%,transparent);border:1px solid var(--border);
    border-radius:var(--r-md);padding:11px 14px;margin-bottom:14px}
  .proxy-status .ps-cell{min-width:0}
  .proxy-status .mac-label{margin-bottom:3px}
  .proxy-status .ps-val{display:block;font-size:var(--fs-body);font-weight:600;color:var(--text);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .proxy-step{display:flex;gap:11px;align-items:flex-start;margin-bottom:12px}
  .proxy-step .ps-num{width:20px;height:20px;flex:none;border-radius:50%;background:var(--accent);color:#fff;
    font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;margin-top:13px}
  .proxy-step .ps-body{flex:1;min-width:0}
  .proxy-step .ps-title{display:flex;align-items:center;gap:8px;font-size:var(--fs-body);
    font-weight:600;margin-bottom:8px;flex-wrap:wrap}
  .proxy-step .ps-title .mac-btn{margin-left:auto}
  .proxy-step .ps-hint{font-size:var(--fs-caption);color:var(--muted);line-height:var(--lh-body)}
  .proxy-step .ps-actions{display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap}
  .proxy-node-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;
    max-height:260px;overflow:auto}
  .pnode{border:1px solid var(--border);border-radius:var(--r-md);padding:8px 10px;
    background:color-mix(in srgb,var(--bg-2) 45%,transparent)}
  .pnode.on{border-color:var(--accent);box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 45%,transparent)}
  .pnode .pnode-name{font-size:var(--fs-footnote);font-weight:600;color:var(--text);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .pnode .pnode-meta{font-size:var(--fs-caption);color:var(--muted);margin-top:1px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .pnode .pnode-foot{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:7px}
  .proxy-or{display:flex;align-items:center;gap:10px;font-size:var(--fs-caption);color:var(--muted);margin:12px 0 8px}
  .proxy-or::before,.proxy-or::after{content:"";flex:1;height:1px;background:var(--border)}
  .proxy-empty{border:1px dashed var(--border2);border-radius:var(--r-md);padding:16px;text-align:center;
    font-size:var(--fs-caption);color:var(--muted)}
  .ps-manual{display:flex;gap:8px}
  .ps-manual .mac-input{flex:1;min-width:0}
```

- [ ] **Step 3: 静态校验 CSS 未破坏结构**

Run: `.venv\Scripts\python.exe scripts/build_frontend.py --check-only --require-manifest`
Expected: `JS 通过 | CSS 括号平衡 通过`（括号平衡是本步的关键信号）

---

### Task 2: 重建 proxy 面板模板（合并双 panel）

**Files:**
- Modify: `web/static/index.html:2267-2304`（整段替换；行号在 Task 1 插入后会后移，以内容定位）

**Interfaces:**
- Consumes: Task 1 产出的全部类名；既有 ref `proxyUrl` `proxyResolved` `proxyAuto` `proxySource` `proxyNodes` `proxyScanMsg` `proxyScanning` `hyClient`；既有函数 `saveProxy` `scanProxy` `chooseProxy` `downloadHyClient`
- Produces: 单个 `.settings-panel[v-if="settingsTab==='proxy'"]`；所有既有绑定原样保留

- [ ] **Step 1: 定位旧段落**

Read 找到这两行作为边界：
- 起：`<div class="settings-panel" v-if="settingsTab==='proxy'">`（含 `.settings-head`）
- 止：紧跟其后的第二个兄弟 `</div>`，其特征是 `style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)"` 那个 `<div v-if="settingsTab==='proxy'" ...>` 的闭合处（即 `<div class="settings-panel" v-if="settingsTab==='about'">` 之前）

确认旧段共两个 `v-if="settingsTab==='proxy'"` 开头的顶层 div。

- [ ] **Step 2: 用下述内容整体替换（逐字，勿改绑定表达式）**

```html
          <!-- 网络加速：状态条 + 三步向导（2026-09-27 布局重构）
               原为两个并列 .settings-panel，而 .settings-wrap 是横向 flex ⇒ 左右分栏挤压，已合并为单 panel；
               原标题 .settings-head 在 CSS 中无定义（全项目仅此一处使用），改用与其他设置页同款 h2。 -->
          <div class="settings-panel" v-if="settingsTab==='proxy'">
            <h2 class="text-[length:var(--fs-body)] font-bold mb-1">网络加速</h2>
            <p class="text-[length:var(--fs-caption)] text-slate-500 mb-4">三步完成配置。所有出站请求（交易所 / 行情 / 通知 / AI）走代理加速，<b>仅本软件生效</b>，不影响系统其他程序。</p>

            <!-- ① 状态条：只读现有 ref，不新增接口 -->
            <div class="proxy-status">
              <div class="ps-cell">
                <span class="mac-label">当前模式</span>
                <span class="ps-val">{{proxyAuto ? '自动探测本机代理' : (proxyUrl ? '手动指定' : '未设置')}}</span>
              </div>
              <div class="ps-cell">
                <span class="mac-label">生效地址</span>
                <span class="ps-val" :title="proxyResolved || proxyUrl">{{proxyResolved || proxyUrl || '—'}}</span>
              </div>
              <div class="ps-cell">
                <span class="mac-label">已选节点 / 手动地址</span>
                <span class="ps-val" :title="proxyUrl">{{proxyUrl || '未设置'}}</span>
              </div>
              <div class="ps-cell">
                <span class="mac-label">hysteria 客户端</span>
                <span class="ps-val">
                  <span v-if="hyClient==='installed'" class="pill green">已就绪</span>
                  <span v-else-if="hyClient==='downloading'" class="pill orange">下载中…</span>
                  <span v-else-if="hyClient==='error'" class="pill orange">下载失败</span>
                  <span v-else class="pill">未安装</span>
                </span>
              </div>
            </div>

            <!-- 步骤 1：客户端 -->
            <div class="proxy-step">
              <span class="ps-num">1</span>
              <div class="ps-body mac-card">
                <div class="ps-title">准备客户端
                  <span v-if="hyClient==='installed'" class="pill green">已就绪</span>
                  <span v-else-if="hyClient==='downloading'" class="pill orange">下载中…</span>
                  <span v-else-if="hyClient==='error'" class="pill orange">下载失败</span>
                  <span v-else class="pill">未安装</span>
                  <button v-if="hyClient!=='installed' && hyClient!=='downloading'" class="mac-btn btn-sm" @click="downloadHyClient()">自动下载</button>
                </div>
                <div class="ps-hint">hysteria2 节点需要本机客户端；socks5 / http / ss 节点可直接使用，无需此步。</div>
              </div>
            </div>

            <!-- 步骤 2：扫描 -->
            <div class="proxy-step">
              <span class="ps-num">2</span>
              <div class="ps-body mac-card">
                <div class="ps-title">粘贴订阅或节点文本</div>
                <textarea class="mac-input" v-model="proxySource" rows="3" placeholder="订阅 URL 或粘贴节点文本（Clash YAML / 每行 socks5:// http:// ss://）" style="width:100%;box-sizing:border-box"></textarea>
                <div class="ps-actions">
                  <button class="mac-btn primary" @click="scanProxy()">{{proxyScanning?'扫描中…':'扫描节点'}}</button>
                  <span class="text-[length:var(--fs-caption)] text-slate-500">{{proxyScanMsg}}</span>
                </div>
              </div>
            </div>

            <!-- 步骤 3：选节点 + 手动地址 -->
            <div class="proxy-step">
              <span class="ps-num">3</span>
              <div class="ps-body mac-card">
                <div class="ps-title">选择要使用的节点
                  <span v-if="proxyNodes.length" class="pill">共 {{proxyNodes.length}} 个</span>
                </div>
                <div v-if="proxyNodes.length" class="proxy-node-grid">
                  <div v-for="nd in proxyNodes" :key="nd.name+nd.url+nd.server"
                       class="pnode" :class="(proxyUrl && nd.server && proxyUrl.indexOf(nd.server)>=0) ? 'on' : ''">
                    <div class="pnode-name" :title="nd.name">{{nd.name}}</div>
                    <div class="pnode-meta" :title="nd.type+' · '+(nd.server||'-')+':'+(nd.port||'-')">{{nd.type}} · {{nd.server||'-'}}:{{nd.port||'-'}}</div>
                    <div class="pnode-foot">
                      <span v-if="nd.reachable" class="pill" :class="nd.delay_ms<150?'green':(nd.delay_ms<=400?'orange':'')">{{nd.delay_ms}}ms</span>
                      <span v-else-if="!nd.supported" class="pill">需客户端</span>
                      <span v-else class="pill">不可达</span>
                      <button class="mac-btn btn-sm" :disabled="!((nd.supported && nd.reachable) || nd.type==='hysteria2' || nd.type==='hysteria')" @click="chooseProxy(nd)">选择</button>
                    </div>
                  </div>
                </div>
                <div v-else class="proxy-empty">还没有节点——先在上一步粘贴订阅并点击「扫描节点」；订阅链接不适用于下方手动地址。</div>

                <div class="proxy-or">或者手动填写地址</div>
                <div class="ps-manual">
                  <input class="mac-input" v-model="proxyUrl" placeholder="单个代理地址 如 socks5://ip:port 或 http://ip:port" @keyup.enter="saveProxy()"/>
                  <button class="mac-btn primary" @click="saveProxy()">保存并应用</button>
                </div>
              </div>
            </div>
          </div>
```

- [ ] **Step 3: 确认旧段落已完全消失、且未误伤相邻 tab**

Run: `.venv\Scripts\python.exe -c "import io,re;p='web/static/index.html';s=io.open(p,encoding='utf-8').read();print('proxy-panels=',s.count(chr(34)+'proxy'+chr(34)+''));print('settings-head=',s.count('settings-head'));print('about-panel=',s.count(\"settingsTab==='about'\"))"`
Expected: `proxy-panels= 1`（`settingsTab==='proxy'` 只剩 1 处 `settings-panel`）、`settings-head= 0`、`about-panel= 1`

---

### Task 3: 校验 + 提交

**Files:**
- Modify: 无（只跑校验与提交）

**Interfaces:**
- Consumes: Task 1、Task 2 的改动
- Produces: 一次 commit

- [ ] **Step 1: JS 语法与构建校验**

Run: `.venv\Scripts\python.exe scripts/check_js.py`
Expected: `JS 语法安全检查通过`

Run: `.venv\Scripts\python.exe scripts/build_frontend.py --check-only --require-manifest`
Expected: `[verify][OK] build_frontend`（若 manifest 不一致，说明 index.html 被构建流程消费，需按输出重建）

- [ ] **Step 2: 模板自定义组件闭合检查（in-DOM 铁律）**

Run: `.venv\Scripts\python.exe -c "import io;s=io.open('web/static/index.html',encoding='utf-8').read();b=s.split(\"settingsTab==='proxy'\")[1].split(\"settingsTab==='about'\")[0];import re;print('mac-field self-close:',len(re.findall(r'<mac-[a-z]+[^>]*/>',b)));print('svg use:',b.count('<use '))"`
Expected: `mac-field self-close: 0`（新段落不含任何自闭合自定义组件）

- [ ] **Step 3: 全量校验（pre-commit 同款套件）**

Run: `.venv\Scripts\python.exe scripts/verify_all.py`
Expected: `[verify][ALL PASS]`（含 check_lock / ruff / mypy / validate_config / check_doc_sync / check_js / build_frontend / compileall）

- [ ] **Step 4: 提交**

```bash
git add web/static/index.html
git commit -m "refactor(web): 网络加速页布局重构——单 panel 修复分栏 bug + 状态条三步向导 + 节点卡片网格"
```

Expected: pre-commit 全绿，commit 成功

- [ ] **Step 5: 人工可视验收（交给用户）**

打开 `?tab=proxy`：状态条 4 格 / 三步骤圆点 / 节点卡片 3 列；来回切其余 8 个 tab 确认无串扰；窄窗口拖动验证状态条 4→2 列、节点网格 3→2→1 列。
