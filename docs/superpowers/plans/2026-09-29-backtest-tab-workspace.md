# 回测页 Tab 工作区 UI 重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把回测页（`view==='backtest'`）重排为「页头 → 紧凑配置条 → 运行状态条 → 下划线主 Tab（结果/实时监控/分析工具/历史记录）」的单栏 Tab 工作区，功能与数据绑定零丢失。

**Architecture:** 单文件改造：`web/static/index.html` 内重写 backtest 视图模板与 bt 局部 CSS；新增 `btTab/btToolTab/btConfigDirty` 脚本状态与自动切换规则（纯前端状态，不新增任何后端调用）；现 7 个区块按 Tab 迁移容器；契约测试沿用 `tests/test_backtest_ui_contract.py` 静态断言模式。

**Tech Stack:** CDN Vue 3（单文件内联）、原生 CSS（既有令牌）、ECharts（容器与渲染逻辑不动）、pytest 静态契约测试。

**设计规范（唯一依据）:** `docs/superpowers/specs/2026-09-29-backtest-tab-workspace-design.md`
**设计稿留存:** `.superpowers/brainstorm/1-1790500775/content/`（backtest-layout-options / backtest-style-refinement）

## Global Constraints

1. **零后端改动**：`web/api/*`、`core/`、`engine/`、`drl/`、`backtest/` 任何 py 文件零 diff。
2. **零数据绑定改名**：`bt/btResult/btMetrics/btAdvice/btLegend/btTradesAll/btTradesShown/btTradeSearch/btTradeShowAll/btLive/btStep/btHistoryGroups/btHistoryLoading/btHistoryError/btPlans/btRunning/btRange/overfitResult/gridScanResult/compareResult/plateauInfo/btLocalCoverage/btSymbolOptions/btStrategyOptions/btSymbolManual/btGoto/openBtResult/runBacktest/loadBtResult/loadBtHistory/applyBtPlan/saveBtPlan/delBtPlan/setBtRange/setBtSymbol/setBtStrategy/addBtSymbolManual/runGridScan/compareStrategies/runOverfit/useGridBest/runOverfitForBest/exportBtTradesCsv/formDrop/toggleFormDrop/switchView/btLegend` 等全部原名保留。
3. **ECharts 容器与渲染不动**：`#btChart`、`#plateauChart` 的 init/主题/重绘逻辑不改；两 ID 唯一。
4. **回测轮询/进度推送策略不动**：`runBacktest` 内 500ms progress 轮询、有界重试、`btLive` 更新节奏逐字保留（本计划只改其中的 Tab/滚动钩子行）。
5. **既有契约测试必须继续通过**：`tests/test_backtest_ui_contract.py` 的 3 个既有测试，尤其 `assert "['5m','15m','30m','1h','4h','1d']" in HTML`（周期分段数组字面量不得改写）。
6. **禁止 emoji**：回测视图新增行不用任何表情符号；现存 `⚠` 换 `#i-alert` SVG 或「警告：」文字；`✓/✗` 判定文字保留；删除按钮 `✕` 改文字「删除」。
7. **只用既有令牌/rgba**，不引入新色值（光晕/状态条/画布质感只用 `--accent/--green/--red/--orange/--border` 及其 rgba）。
8. **Scoped staging（强制）**：本仓库有并行工作流会 `git add` 整个 index.html 并可能提交 WIP。每任务提交必须用过滤 patch + `git apply --cached` 只暂存本任务 hunk（流程见 `.superpowers/sdd/2026-09-27-continuous-evolution-kpi-dashboard-ui/task-1-report.md` §"Scoped-staging procedure"，若该目录已清理则用 `git diff --output=patch` 生成后手工剔除他人 hunk）。提交后立即 `git show HEAD` 校验无 proxy/net-accel/localStorage 等外来内容；若 pre-commit 门禁被**并行工作流的脏文件**（非本任务文件的 ruff 错误）拦住：先手动跑 `scripts/check_js.py` + `scripts/build_frontend.py --check-only` + 本任务契约测试全绿，再 `--no-verify` 提交并在报告中记录（先例：27f591d、e99b9a3）；不得修改他人文件来过门禁。
9. TDD：每任务先写契约测试跑红，再实现跑绿，然后 commit。

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `web/static/index.html` | 修改 | 唯一实现载体：backtest 模板（`<!-- ========== 历史回测 ========== -->` 至配对 `</section>`）、bt 局部 CSS、script 区（Tab 状态/btGoto/自动切换/徽标 watch/Esc） |
| `tests/test_backtest_ui_contract.py` | 修改 | 追加契约测试（文件已有 3 个既有测试不得改动） |

> 行号会随并行提交漂移，全部用**内容锚**定位。

---

### Task 1: Tab 状态与导航脚本层

**Files:**
- Modify: `web/static/index.html`（script 区：`const btStep = ref('config');` 附近插入新状态与 watch；`runBacktest` / `openBtResult` / `loadBtResult` / `runGridScan` / `compareStrategies` / `runOverfit` 的钩子行；`btGoto` 整体替换；onMounted 的监听注册区加 Esc；setup return 追加导出）
- Test: `tests/test_backtest_ui_contract.py`（追加）

**Interfaces:**
- Consumes: 既有 `bt`（reactive，7444 行附近）、`btStep`、`formDrop`、`nextTick`/`watch`（已从 Vue 解构，4330 行）、`btGoto` 的既有调用方（历史窗「新建回测」按钮）
- Produces（Task 2-5 依赖，setup return 必须导出）:
  - `btTab` — ref，`'result'|'monitor'|'tools'|'history'`，初值 `'result'`
  - `btToolTab` — ref，`'overfit'|'gridscan'|'compare'`，初值 `'overfit'`
  - `btConfigDirty` — ref<boolean>（「配置已改」徽标）
  - `btTileGlow(c)` — `(c:{label:string,cls:string})=>''|'glow-g'|'glow-r'`（Task 5 模板用；本任务先定义并导出）
  - `btGoto(key:string):void` — 签名不变，行为改为「切 Tab + 滚动」

- [ ] **Step 1: 写失败契约测试**

在 `tests/test_backtest_ui_contract.py` 末尾追加（保持文件既有 3 测试原样）：

```python
def _bt_view():
    start = HTML.find("<!-- ========== 历史回测 ========== -->")
    assert start != -1, "回测视图模板起点丢失"
    return HTML[start:HTML.find("</section>", start)]


def test_bt_tab_state_defined_and_exported():
    assert "const btTab = ref('result')" in HTML
    assert "const btToolTab = ref('overfit')" in HTML
    assert "const btConfigDirty = ref(false)" in HTML
    anchor = HTML.find("return {view, settingsTab")
    assert anchor != -1
    ret = HTML[anchor:]
    for name in ("btTab", "btToolTab", "btConfigDirty", "btTileGlow"):
        assert name in ret, name + " 未从 setup 导出"


def test_bt_goto_switches_tab():
    body = HTML[HTML.find("function btGoto"):]
    body = body[:body.find("}", body.find("function btGoto")) + 1]
    assert "BT_TAB_OF_SECTION" in body
    assert "btTab.value" in body


def test_bt_auto_switch_wiring():
    # 开跑→结果；工具发起→工具对应子 Tab；结果打开→结果 Tab
    assert "btTab.value = 'result'" in HTML
    assert "btToolTab.value = 'gridscan'" in HTML
    assert "btToolTab.value = 'compare'" in HTML
    assert "btToolTab.value = 'overfit'" in HTML


def test_bt_config_dirty_watch():
    start = HTML.find("let btBaseline")
    assert start != -1
    body = HTML[start:HTML.find("async function runOverfit", start)]
    assert "watch(bt," in body or "watch(bt ," in body
    assert "{deep:true}" in body or "{ deep: true }" in body


def test_bt_esc_closes_formdrop():
    start = HTML.find("onKeyEsc")
    assert start != -1
    body = HTML[start:start + 400]
    assert "'Escape'" in body and "formDrop.value = ''" in body


def test_bt_tile_glow_rule():
    start = HTML.find("function btTileGlow")
    assert start != -1
    body = HTML[start:start + 500]
    assert "'总收益'" in body and "'最大回撤'" in body
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 新增 7 个 FAIL（既有 3 个保持绿）

- [ ] **Step 3: 插入 Tab 状态与配置徽标 watch（定位锚：`const btStep = ref('config');`，插到其后）**

```js
    // ---- 回测页 Tab 工作区（2026-09-29 重设计，spec §3.4）----
    const btTab = ref('result');        // result | monitor | tools | history
    const btToolTab = ref('overfit');   // overfit | gridscan | compare（分析工具内层子 Tab）
    const btConfigDirty = ref(false);   // 「配置已改」徽标：配置相对上次开跑有变化
    // 基线快照：开跑时重置；deep watch 对比序列化值，避免逐字段监听漏项
    let btBaseline = JSON.stringify(bt);
    function btMarkClean() {
      nextTick(() => { btBaseline = JSON.stringify(bt); btConfigDirty.value = false; });
    }
    watch(bt, () => {
      try { btConfigDirty.value = JSON.stringify(bt) !== btBaseline; } catch (e) { /* 序列化失败不阻塞 */ }
    }, {deep:true});
    // 指标磁贴光晕（spec §3.6）：总收益(正)绿光晕 / 最大回撤红光晕 / 其余素色
    function btTileGlow(c) {
      if (!c) return '';
      if (c.label === '总收益' && c.cls === 'pos') return 'glow-g';
      if (c.label === '最大回撤') return 'glow-r';
      return '';
    }
    // section → Tab 映射（btGoto 与自动切换共用一份，防两处口径漂移）
    const BT_TAB_OF_SECTION = {
      config:'result', result:'result', monitor:'monitor',
      overfit:'tools', gridscan:'tools', compare:'tools', history:'history',
    };
```

- [ ] **Step 4: 替换 `btGoto`（整体替换现有函数体，签名不变）**

现函数（锚 `function btGoto(key){`）内容为 `btStep.value=key; ... scrollIntoView`。替换为：

```js
    function btGoto(key){
      // Tab 工作区：先切到 section 所属 Tab，渲染完成后再滚动（v-if 容器 nextTick 才存在）
      btStep.value = key;
      btTab.value = BT_TAB_OF_SECTION[key] || 'result';
      if (key === 'overfit' || key === 'gridscan' || key === 'compare') btToolTab.value = key;
      nextTick(() => {
        const el = document.getElementById('bt-' + key);
        if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
      });
    }
```

- [ ] **Step 5: 自动切换钩子（六处，逐处最小改动）**

1. `async function runBacktest()` 开头两行 `btRunning.value=true; btResult.value=null;` 之后，**删除**原有两行：
   ```js
         btStep.value = 'monitor';
         nextTick(()=>btGoto('monitor'));
   ```
   替换为：
   ```js
         btTab.value = 'result';   // spec §3.4①：开跑即停在结果 Tab，跑完直接可见
         btMarkClean();            // 基线 = 本次开跑时的配置
   ```
   （函数其余部分——progress 轮询、重试、`btStep.value='result'`、`openBtResult(r)`——逐字保留。）
2. `function openBtResult(r){` 内、`btResult.value = r;` 之后加一行：
   ```js
       btTab.value = 'result';   // spec §3.4③：历史加载/跑完统一回结果 Tab
   ```
3. `async function loadBtResult(id)` 内 `openBtResult(r);` 之后加一行：
   ```js
         btConfigDirty.value = false;   // spec §3.4 徽标：加载结果后清除（不改基线）
   ```
4. `async function runOverfit()` 第一行（`btRunning...` 类似位置，函数体最前）加：
   ```js
       btTab.value = 'tools'; btToolTab.value = 'overfit';   // spec §3.4②
   ```
5. `async function runGridScan()` 函数体最前加：
   ```js
       btTab.value = 'tools'; btToolTab.value = 'gridscan';
   ```
6. `async function compareStrategies()` 函数体最前加：
   ```js
       btTab.value = 'tools'; btToolTab.value = 'compare';
   ```
   （结果到达时不再强切——用户离开 tools 不打断，spec §3.4② 后半。）

- [ ] **Step 6: Esc 关闭 formDrop（定位 onMounted 内 `document.addEventListener('click', onDocClick);`）**

在该行**之前**插入：

```js
      // spec §3.2：浮层 Esc 关闭（复用 menu-drop 单槽——Esc 清空即全部收起）
      onKeyEsc = (e) => { if (e.key === 'Escape' && formDrop.value) formDrop.value = ''; };
      document.addEventListener('keydown', onKeyEsc);
```

并在 `let onDocClick` 声明处（同一监听声明区）补 `let onKeyEsc = null;`。清理：定位 `document.removeEventListener('click', onDocClick)`（onUnmounted 清理区）；若存在则在其后镜像加 `if (onKeyEsc) document.removeEventListener('keydown', onKeyEsc);`；若该处并无 click 移除，则不加移除并保持与 click 监听同等生命周期（与现状一致即可，报告中注明属于哪种情况）。

- [ ] **Step 7: setup return 追加导出**

定位 return 中现有行 `bt, btRunning, btResult, btHistory, ... btStep, btGoto, openBtResult, ...`，在 `btStep, btGoto,` 附近追加：

```js
      btTab, btToolTab, btConfigDirty, btTileGlow,
```

- [ ] **Step 8: 跑测试确认通过 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 10/10 PASS（3 旧 + 7 新）

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: 通过

- [ ] **Step 9: Commit（scoped staging，Global Constraint 8）**

```powershell
git commit -m "feat(web): 回测页 Tab 状态层——btTab/btToolTab/配置徽标 watch/btGoto 切 Tab/Esc 关闭浮层"
```
（提交内容：本任务 hunk + 测试文件；提交后 `git show HEAD` 校验。）

---

### Task 2: 页头 + 紧凑配置条 + 高级参数浮层

**Files:**
- Modify: `web/static/index.html`（backtest 模板：page-head 行、`.bt-configbar` 新容器与字段迁移、高级参数 Teleport 浮层；删除 `#bt-config` mac-window 与四个 `mac-form-section`；bt 局部样式区新增 `.bt-configbar`/`.bt-advpop` 规则）
- Test: `tests/test_backtest_ui_contract.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `formDrop/toggleFormDrop`（`btAdvanced` 为新 key）、既有绑定 `bt.*`、`btLocalCoverage`、`btSymbolOptions/btStrategyOptions`、`runGridScan/compareStrategies/runOverfit`
- Produces: `id="bt-config"` 落到配置条上（`btGoto('config')` 依赖此 ID）；`.bt-configbar`、`.bt-advpop` CSS；高级参数浮层节点（`formDrop==='btAdvanced'`）

- [ ] **Step 1: 写失败契约测试**

```python
def _bt_script_body():
    start = HTML.find("async function runOverfit")
    return HTML[start:]


def test_bt_configbar_fields():
    v = _bt_view()
    assert 'class="bt-configbar"' in v
    assert 'id="bt-config"' in v          # btGoto('config') 的锚点落在配置条
    for binding in ("bt.data_source", "bt.symbol", "bt.timeframe", "bt.strategy_name",
                    "bt.since", "bt.until"):
        assert binding in v, binding
    # 工具三键必须位于配置条内（严格包含断言）
    bar = v[v.find('bt-configbar'):]
    for call in ("runGridScan()", "compareStrategies()", "runOverfit()"):
        assert call in bar, call + " 不在配置条内"
    # 周期数组字面量不得改写（既有测试同源，双保险）
    assert "['5m','15m','30m','1h','4h','1d']" in v


def test_bt_advanced_popup():
    v = _bt_view()
    assert "formDrop==='btAdvanced'" in v
    for binding in ("bt.start_cash", "bt.fee_rate", "bt.slippage", "bt.limit", "bt.use_gpu"):
        assert binding in v, binding


def test_bt_config_sections_retired():
    v = _bt_view()
    assert "mac-form-section" not in v          # 旧四段堆叠容器全部迁走
    assert 'id="bt-config" class="mac-window"' not in v
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 新增 3 个 FAIL

- [ ] **Step 3: 重写 page-head + 建配置条（整段替换）**

1. **page-head**：定位 `<div class="page-head">` 至其配对 `</div>`（含保存方案 menu-drop 与开始回测按钮的整块）。整块替换为（保存方案块与开始回测按钮**原样剪切**回来，前面新增高级参数入口）：

```html
      <div class="page-head">
        <div></div>
        <div class="page-actions">
          <div class="menu-drop" :class="{open: formDrop==='btAdvanced'}">
            <button class="mac-btn" data-form-key="btAdvanced" :aria-expanded="formDrop==='btAdvanced'?'true':'false'" aria-haspopup="true" @click.stop="toggleFormDrop('btAdvanced', $event)">
              <svg width="13" height="13"><use href="#i-sliders"/></svg>高级参数<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="margin-left:2px"><path d="M6 9l6 6 6-6"/></svg>
            </button>
            <Teleport to="body">
              <div v-if="formDrop==='btAdvanced'" ref="formDropMenu" class="menu-sub bt-advpop" :style="formDropStyle" @click.stop style="min-width:320px;overflow-y:auto">
                <div class="bt-advpop-title">高级参数</div>
                <div class="mac-field-row">
                  <label class="fr-label" for="btStartCash">初始资金</label>
                  <div class="fr-ctrl"><input id="btStartCash" v-model.number="bt.start_cash" class="mac-input" type="number" min="0"/><span class="fr-unit">USDT</span></div>
                </div>
                <div class="mac-field-row">
                  <label class="fr-label" for="btFeeRate">手续费率</label>
                  <div class="fr-ctrl"><input id="btFeeRate" v-model.number="bt.fee_rate" class="mac-input" type="number" step="0.0001" min="0" max="0.1" title="默认 0.001 · 范围 0 ~ 0.1"/></div>
                </div>
                <div class="mac-field-row">
                  <label class="fr-label" for="btSlippage">滑点</label>
                  <div class="fr-ctrl"><input id="btSlippage" v-model.number="bt.slippage" class="mac-input" type="number" step="0.0001" min="0" title="默认 0 · 按成交价比例计的滑点成本"/></div>
                </div>
                <div class="mac-field-row">
                  <label class="fr-label" for="btLimit">K 线数量</label>
                  <div class="fr-ctrl"><input id="btLimit" v-model.number="bt.limit" class="mac-input" type="number" min="100" max="5000" placeholder="留空用最新"/><span class="fr-unit">根</span></div>
                </div>
                <div class="mac-field-row">
                  <label class="fr-label" for="btGpuToggle">GPU 加速</label>
                  <div class="fr-ctrl"><div id="btGpuToggle" class="mac-toggle" role="switch" aria-label="回测 GPU 加速" :aria-checked="bt.use_gpu?true:false" tabindex="0" :class="bt.use_gpu?'on':''" @click="bt.use_gpu=!bt.use_gpu" @keydown.enter.prevent="bt.use_gpu=!bt.use_gpu" @keydown.space.prevent="bt.use_gpu=!bt.use_gpu"></div><span class="text-[length:var(--fs-caption)]" :class="bt.use_gpu?'pos':'text-slate-500'">{{bt.use_gpu?'已启用':'未启用'}}</span></div>
                </div>
              </div>
            </Teleport>
          </div>
          <!-- 保存方案 menu-drop：原块原样剪切至此（含 Teleport 与方案列表/删除/保存按钮，零改动） -->
          <!-- 开始回测按钮：原块原样剪切至此（runBacktest 调用与 disabled/文案不变） -->
        </div>
      </div>
```

2. **配置条**：紧随 page-head 之后插入；内容 = 从旧 `#bt-config` 窗内**剪切**（不是复制，旧窗随后整体删除）四段字段，按 `.bt-configbar` 重排（`.fld` 小标签组，字段控件原样）：

```html
      <div class="bt-configbar" id="bt-config">
        <div class="bt-cb-group">
          <div class="bt-cb-lb">数据来源</div>
          <div class="mac-seg">
            <button :class="bt.data_source==='demo'?'active':''" @click="bt.data_source='demo'">演示数据</button>
            <button :class="bt.data_source==='exchange'?'active':''" @click="bt.data_source='exchange'">交易所</button>
            <button :class="bt.data_source==='csv'?'active':''" @click="bt.data_source='csv'">CSV 文件</button>
          </div>
          <template v-if="bt.data_source==='exchange'">
            <mac-select v-model="bt.exchange" :options="exchangeOptions" style="width:120px"></mac-select>
          </template>
          <template v-if="bt.data_source==='csv'">
            <input id="btCsvPath" v-model="bt.csv_path" class="mac-input" style="width:170px" placeholder="/path/to/btc_usdt_1h.csv"/>
          </template>
        </div>
        <div class="bt-cb-group">
          <div class="bt-cb-lb">交易对</div>
          <!-- 原 btSymbol menu-drop 整块原样剪切（含 Teleport 二级菜单与手输行），外层样式改 width:auto -->
        </div>
        <div class="bt-cb-group">
          <div class="bt-cb-lb">周期</div>
          <!-- 原周期 mac-seg 整块原样剪切（数组字面量逐字保留） -->
        </div>
        <div class="bt-cb-group">
          <div class="bt-cb-lb">策略</div>
          <!-- 原 btStrategy menu-drop 整块原样剪切（含 kind pill 与提示行「参数跟随策略默认值…」） -->
        </div>
        <div class="bt-cb-group">
          <div class="bt-cb-lb">时间范围</div>
          <div class="bt-cb-dates">
            <input id="btSince" type="date" v-model="bt.since" class="mac-input" aria-label="开始日期"/>
            <span class="bt-cb-sep">→</span>
            <input id="btUntil" type="date" v-model="bt.until" class="mac-input" aria-label="结束日期"/>
          </div>
          <div class="bt-cb-hint" v-if="btLocalCoverage">{{btLocalCoverage}}</div>
          <div class="bt-cb-hint" v-else>留空则使用最新 K 线；批量下载前往 <button class="mac-btn btn-sm" style="padding:1px 8px" @click="switchView('settings'); settingsTab='data'">系统设置 · 数据管理</button></div>
        </div>
        <div class="bt-cb-tools">
          <button class="mac-btn" @click="runGridScan()" :disabled="gridScanLoading">{{gridScanLoading?'扫描中…':'参数扫描'}}</button>
          <button class="mac-btn" @click="compareStrategies()" :disabled="compareLoading">{{compareLoading?'对比中…':'策略对比'}}</button>
          <button class="mac-btn" @click="runOverfit()" :disabled="overfitLoading">{{overfitLoading?'检测中…':'过拟合检测'}}</button>
        </div>
      </div>
```

3. **删除旧窗**：整块删除 `<div class="mac-window" id="bt-config">` … 配对 `</div>`（内含四段 `mac-form-section` 与 `.bt-tools`——字段已全部移出）。注意：交易对/策略 menu-drop 的**整块含 Teleport**一并移动，移动后检查 `.menu-drop` 外层不要残留旧样式属性（保持 `style="width:100%"` 的改为不设宽或 `width:auto`）。

- [ ] **Step 4: 局部样式新增（bt 样式区锚：`.bt-layout` 规则附近）**

```css
  /* ---- 紧凑配置条（spec §3.2）：开跑所需字段常驻一行，窄屏换行 ---- */
  .bt-configbar{display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;
    background:var(--bg-1);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
  .bt-cb-group{display:flex;flex-direction:column;gap:5px;min-width:0}
  .bt-cb-lb{font-size:var(--fs-micro);color:var(--text3);font-weight:600;letter-spacing:var(--fs-micro-ls)}
  .bt-cb-group .mac-seg{width:auto}
  .bt-cb-dates{display:flex;align-items:center;gap:6px}
  .bt-cb-dates .mac-input{width:130px}
  .bt-cb-sep{color:var(--text3)}
  .bt-cb-hint{font-size:var(--fs-caption);color:var(--muted);max-width:320px;line-height:1.5}
  .bt-cb-tools{display:flex;gap:8px;margin-left:auto;align-items:center;flex-wrap:wrap}
  /* ---- 高级参数浮层（menu-drop 同族） ---- */
  .bt-advpop{flex-direction:column;gap:8px;padding:11px 12px;max-height:70vh}
  .bt-advpop-title{font-size:var(--fs-body);font-weight:650;color:var(--text)}
  .bt-advpop .mac-field-row{align-items:center}
```

- [ ] **Step 5: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 13/13 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py` → 通过

- [ ] **Step 6: Commit（scoped staging）**

```powershell
git commit -m "feat(web): 回测页紧凑配置条 + 高级参数浮层——四段表单迁移，#bt-config 锚点落条"
```

---

### Task 3: 主 Tab 行 + 结果 Tab + 退役双栏

**Files:**
- Modify: `web/static/index.html`（模板：Tab 行插入、结果区包 `v-if`、`.bt-layout` 壳删除；样式：`.bt-tabs` 下划线规则 + 退役 `.bt-layout` 系列）
- Test: `tests/test_backtest_ui_contract.py`（追加）

**Interfaces:**
- Consumes: Task 1 `btTab/btConfigDirty/btTileGlow`、Task 2 配置条、既有结果区内容（`btMetrics.cards` 磁贴行、`btAdvice`、`#btChart` 窗、逐笔成交窗、空态）
- Produces: `.bt-tabs`/`.bt-tab` 样式与徽标结构（Task 4/5 复用）；`id="bt-result"` 移入 Tab 容器

- [ ] **Step 1: 写失败契约测试**

```python
def test_bt_main_tabs():
    v = _bt_view()
    assert 'class="bt-tabs"' in v
    for label in ("结果", "实时监控", "分析工具", "历史记录"):
        assert label in v
    assert "btTab==='monitor'" in v or 'btTab==="monitor"' in v


def test_bt_result_tab_wrapped():
    v = _bt_view()
    assert "btTab==='result'" in v
    assert 'id="bt-result"' in v
    # 结果内容（磁贴行/曲线/成交）必须存在于视图中（迁移不丢件）
    for marker in ("btMetrics.cards", 'id="btChart"', "btTradesShown"):
        assert marker in v, marker


def test_bt_layout_retired():
    assert ".bt-layout" not in HTML
    assert "bt-layout" not in _bt_view()


def test_bt_tab_css_underline():
    assert ".bt-tab.on::after" in HTML or ".bt-tab.on:before" in HTML
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 新增 4 个 FAIL

- [ ] **Step 3: 插入主 Tab 行**

位置：`.bt-configbar` 之后（Task 2 的运行状态条位置由 Task 5 填充，此处先留注释占位 `<!-- 运行状态条：Task 5 -->`）：

```html
      <!-- 主 Tab 行（spec §3.4：下划线式 + 自动切换徽标） -->
      <div class="bt-tabs" role="tablist">
        <button class="bt-tab" :class="btTab==='result'?'on':''" role="tab" :aria-selected="btTab==='result'?'true':'false'" @click="btTab='result'">
          结果<span v-if="btConfigDirty" class="bt-tab-badge warn">配置已改</span>
        </button>
        <button class="bt-tab" :class="btTab==='monitor'?'on':''" role="tab" :aria-selected="btTab==='monitor'?'true':'false'" @click="btTab='monitor'">
          实时监控<span v-if="btLive.running" class="bt-tab-badge run">运行中</span>
        </button>
        <button class="bt-tab" :class="btTab==='tools'?'on':''" role="tab" :aria-selected="btTab==='tools'?'true':'false'" @click="btTab='tools'">分析工具</button>
        <button class="bt-tab" :class="btTab==='history'?'on':''" role="tab" :aria-selected="btTab==='history'?'true':'false'" @click="btTab='history'">历史记录</button>
      </div>
```

- [ ] **Step 4: 结果区包 Tab 容器 + 拆双栏壳**

1. 定位 `<div class="space-y-4 bt-section" id="bt-result">`（Task 2 后该行仍在旧布局内）。将其**替换**为：

```html
      <div v-if="btTab==='result'" class="space-y-4 bt-section" id="bt-result">
```

2. 删除外层 `.bt-layout` 壳：`<div class="bt-layout">` 一行与其配对 `</div>`（Task 2 已清空左列 config 窗，壳内现仅剩结果 div；删除后 configbar/tabs/结果 div 成为 section 直接子元素）。若 `.bt-layout` 内还有包裹结果区的多余 div（如 col 容器），一并拆除——**以「删除后标签配平 + 结果区成为 section 直接子元素」为准**，用 grep 验证 `bt-layout` 全文 0 出现。
3. 空态文案更新：`v-else` 空态里 `在左侧配置参数，点击「开始回测」` 改为 `设置上方配置，点击「开始回测」`。

- [ ] **Step 5: Tab CSS（bt 样式区新增）**

```css
  /* ---- 主 Tab（spec §3.4 下划线式） ---- */
  .bt-tabs{display:flex;gap:18px;border-bottom:1px solid var(--border);padding:0 4px;overflow-x:auto;scrollbar-width:none}
  .bt-tabs::-webkit-scrollbar{display:none}
  .bt-tab{position:relative;background:none;border:0;padding:8px 1px 9px;font-family:inherit;
    font-size:var(--fs-caption);color:var(--muted);cursor:pointer;white-space:nowrap;
    display:inline-flex;align-items:center;gap:6px;transition:color .15s}
  .bt-tab:hover{color:var(--text)}
  .bt-tab.on{color:var(--text);font-weight:600}
  .bt-tab.on::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:2px;
    background:var(--accent);border-radius:2px}
  .bt-tab-badge{font-size:var(--fs-micro);padding:1px 7px;border-radius:99px;white-space:nowrap}
  .bt-tab-badge.warn{background:rgba(245,158,10,.15);color:var(--orange);border:1px solid rgba(245,158,10,.35)}
  .bt-tab-badge.run{background:rgba(13,204,130,.13);color:var(--green);border:1px solid rgba(13,204,130,.35)}
```

退役：删除 `.bt-layout{...}` 规则及其媒体查询内所有 `bt-layout` 引用（含注释块）。保留 `.bt-section/.bt-advice/.bt-legend` 等仍被引用的规则。

- [ ] **Step 6: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 17/17 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py` → 通过
自查：标签配平（`.bt-layout` 0 出现、section 子元素序 configbar → 注释 → tabs → result）。

- [ ] **Step 7: Commit（scoped staging）**

```powershell
git commit -m "feat(web): 回测页主 Tab 行（下划线+徽标）与结果 Tab 化，退役 bt-layout 双栏"
```

---

### Task 4: 监控 / 分析工具 / 历史三个 Tab 迁移

**Files:**
- Modify: `web/static/index.html`（模板：三个既有区块包 Tab 容器 + 工具子 Tab + 各自空态）
- Test: `tests/test_backtest_ui_contract.py`（追加）

**Interfaces:**
- Consumes: Task 1 `btTab/btToolTab`；既有区块（回测实时监控窗 `id="bt-monitor"`、过拟合窗 `bt-overfit`、扫描窗 `bt-gridscan`、对比窗 `bt-compare`、历史窗 `bt-history`）
- Produces: 三 Tab 的 `v-if` 容器、`.bt-subtabs` 子 Tab 结构与样式

- [ ] **Step 1: 写失败契约测试**

```python
def test_bt_monitor_tab():
    v = _bt_view()
    assert "btTab==='monitor'" in v
    assert 'id="bt-monitor"' in v
    assert "btLive.kline_history" in v


def test_bt_tools_tab_with_subtabs():
    v = _bt_view()
    assert "btTab==='tools'" in v
    assert 'class="bt-subtabs"' in v
    assert "btToolTab==='overfit'" in v
    assert "btToolTab==='gridscan'" in v
    assert "btToolTab==='compare'" in v
    for anchor in ("id=\"bt-overfit\"", "id=\"bt-gridscan\"", "id=\"bt-compare\"", "id=\"plateauChart\""):
        assert anchor in v, anchor


def test_bt_history_tab():
    v = _bt_view()
    assert "btTab==='history'" in v
    assert 'id="bt-history"' in v
    assert "btHistoryGroups" in v
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 新增 3 个 FAIL

- [ ] **Step 3: 监控 Tab 迁移**

定位 `<!-- 回测实时监控窗口 -->` 注释与其 `<div v-if="btLive.running || btLive.done || btRunning" class="mac-window bt-section" id="bt-monitor" ...>`。改为：

```html
      <div v-if="btTab==='monitor'" class="bt-section" id="bt-monitor">
        <div v-if="btLive.running || btLive.done || btRunning" class="mac-window" style="border-color:rgba(125,59,237,.35)">
          <!-- 窗内内容（titlebar/进度/4 磁贴/两表）原样保留，零改动 -->
        </div>
        <div v-else class="mac-window"><div class="mac-body"><div class="empty-state"><span class="e-ico"><svg><use href="#i-clock"/></svg></span><span class="e-txt">暂无运行中的回测——在配置条点击「开始回测」</span></div></div></div>
      </div>
```

（原窗 titlebar 内的进度文案/进度条**保留**——状态条是它的常驻摘要，两者同源不同位。原窗体的 `style="border-color..."` 挪到内层 mac-window 上。）

- [ ] **Step 4: 分析工具 Tab（子 Tab + 三块迁移）**

在监控 Tab 之后插入容器，把过拟合/扫描/对比三个既有窗**整体剪切**进来：

```html
      <div v-if="btTab==='tools'" class="bt-section" id="bt-tools">
        <div class="bt-subtabs">
          <button class="bt-subtab" :class="btToolTab==='overfit'?'on':''" @click="btToolTab='overfit'">过拟合报告</button>
          <button class="bt-subtab" :class="btToolTab==='gridscan'?'on':''" @click="btToolTab='gridscan'">参数扫描</button>
          <button class="bt-subtab" :class="btToolTab==='compare'?'on':''" @click="btToolTab='compare'">策略对比</button>
        </div>
        <!-- 过拟合块：原窗整体移入，外层再包一层子 Tab 判定 -->
        <div v-if="btToolTab==='overfit'" class="bt-tool-pane">
          <div v-if="overfitResult" class="mac-window bt-section" id="bt-overfit" ...>原窗体原样</div>
          <div v-else class="mac-window"><div class="mac-body"><div class="empty-state"><span class="e-ico"><svg><use href="#i-shield"/></svg></span><span class="e-txt">尚未检测——点配置条「过拟合检测」</span></div></div></div>
        </div>
        <!-- 参数扫描块：原窗（含 #plateauChart 热力图与应用最优参数）整体移入，同构 v-if/v-else -->
        <div v-if="btToolTab==='gridscan'" class="bt-tool-pane">
          <div v-if="gridScanResult" class="mac-window bt-section" id="bt-gridscan" ...>原窗体原样</div>
          <div v-else class="mac-window"><div class="mac-body"><div class="empty-state"><span class="e-ico"><svg><use href="#i-sliders"/></svg></span><span class="e-txt">尚未扫描——点配置条「参数扫描」</span></div></div></div>
        </div>
        <!-- 策略对比块：同构 -->
        <div v-if="btToolTab==='compare'" class="bt-tool-pane">
          <div v-if="compareResult" class="mac-window bt-section" id="bt-compare" ...>原窗体原样</div>
          <div v-else class="mac-window"><div class="mac-body"><div class="empty-state"><span class="e-ico"><svg><use href="#i-trophy"/></svg></span><span class="e-txt">尚未对比——点配置条「策略对比」</span></div></div></div>
        </div>
      </div>
```

要点：三个原窗的 `v-if="overfitResult/gridScanResult/compareResult"` 判定**移到内层**（外层是子 Tab 判定）；原窗 id、边框色 style、内部结构零改动；`#plateauChart` 随窗迁移（`v-if` 隐藏期不渲染 ECharts，首次可见按现逻辑初始化——若 `renderPlateau` 依赖 DOM 可见性报错，报告中记录并改为 `v-show` 该 pane，禁止改渲染函数本身）。

子 Tab CSS：

```css
  .bt-subtabs{display:inline-flex;background:var(--bg-2);border:1px solid var(--border);
    border-radius:6px;padding:2px;gap:2px;margin-bottom:10px}
  .bt-subtab{border:0;background:none;font-family:inherit;font-size:var(--fs-caption);
    color:var(--muted);padding:4px 12px;border-radius:4px;cursor:pointer;transition:background .15s,color .15s}
  .bt-subtab:hover{color:var(--text)}
  .bt-subtab.on{background:var(--bg-3);color:var(--text)}
  .bt-tool-pane{min-width:0}
```

- [ ] **Step 5: 历史 Tab 迁移**

定位 `<div class="mac-window bt-section" id="bt-history">`（整块到配对 `</div>`）。改为外层包 Tab 判定：

```html
      <div v-if="btTab==='history'" class="bt-section" id="bt-history">
        <div class="mac-window">…历史窗体原样（含 loading/error/空态、刷新按钮、新建回测按钮）…</div>
      </div>
```

（历史行点击 `loadBtResult` → Task 1 已接自动切回结果；「新建回测」按钮 `btGoto('config')` → Task 1 映射为切 result + 滚配置条，无需改。）

- [ ] **Step 6: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 20/20 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py` → 通过
自查：四个 Tab 容器互斥（同一时刻只渲染一个的条件链完整）、原窗 id 全部保留、标签配平。

- [ ] **Step 7: Commit（scoped staging）**

```powershell
git commit -m "feat(web): 回测页监控/分析工具(子Tab)/历史三 Tab 迁移——区块整体搬入，空态引导补齐"
```

---

### Task 5: 视觉收口（磁贴光晕 / 状态条 / 画布质感 / 断点 / emoji 清扫）

**Files:**
- Modify: `web/static/index.html`（模板：磁贴 glow 绑定、运行状态条插入、曲线画布包裹、emoji 清扫；样式：磁贴/状态条/画布/断点规则）
- Test: `tests/test_backtest_ui_contract.py`（追加）

**Interfaces:**
- Consumes: Task 1 `btTileGlow`；既有 `btMetrics.cards`、`btLive`、`btRunning`、`#btChart` 容器
- Produces: `.bt-tiles`/`.glow-g`/`.glow-r`、`.bt-runbar`、`.bt-chart-canvas`、断点规则

- [ ] **Step 1: 写失败契约测试**

```python
def test_bt_metric_tiles_glow():
    v = _bt_view()
    assert 'class="bt-tiles"' in v
    assert "btTileGlow(c)" in v


def test_bt_runbar():
    v = _bt_view()
    assert 'class="bt-runbar"' in v
    assert "btLive.pct" in v
    assert "btTab='monitor'" in v   # 状态条点击切监控


def test_bt_curve_canvas_texture():
    v = _bt_view()
    assert ".bt-chart-canvas" in v
    assert "linear-gradient(180deg,rgba(125,59,237,.05)" in HTML


def test_bt_view_no_emoji():
    v = _bt_view()
    blocked = "".join(map(chr, (0x1F4C8, 0x1F9EA, 0x1F9E0, 0x1F916, 0x1F9EC,
                                0x23F8, 0x26A1, 0x231A, 0x1F680, 0x1F3AF, 0x1F527,
                                0x26A0, 0x2715)))   # ⚠ 与 ✕ 都必须清掉；✓/✗ 保留
    assert not any(ch in v for ch in blocked)


def test_bt_breakpoints():
    assert "repeat(auto-fit,minmax(150px,1fr))" in HTML
    assert "minmax(120px,1fr)" in HTML
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 新增 5 个 FAIL

- [ ] **Step 3: 指标磁贴（结果 Tab 内，替换现 `grid grid-cols-2 xl:grid-cols-4` 行）**

原行：`<div class="grid grid-cols-2 xl:grid-cols-4 gap-3"><div v-for="c in btMetrics.cards" ... class="stat-card">...` 替换为：

```html
            <div class="bt-tiles">
              <div v-for="c in btMetrics.cards" :key="c.label" class="stat-card bt-tile" :class="btTileGlow(c)">
                <div class="mac-label">{{c.label}}</div>
                <div class="val fade-num" :class="c.cls">{{c.val}}</div>
                <div class="sub">{{c.sub}}</div>
              </div>
            </div>
```

- [ ] **Step 4: 运行状态条（Task 3 留的占位注释处替换）**

```html
      <!-- 运行状态条（spec §3.3）：跑测时常驻，点击进监控 Tab -->
      <div v-if="btRunning || btLive.running" class="bt-runbar" role="button" tabindex="0"
           @click="btTab='monitor'" @keydown.enter.prevent="btTab='monitor'">
        <span class="bt-runbar-title">回测中 {{btLive.pct||0}}%</span>
        <span class="bt-runbar-prg"><i :style="{width:(btLive.pct||0)+'%'}"></i></span>
        <span v-if="btLive.last_price!=null">价格 {{mktPrice(btLive.last_price)}}</span>
        <span>持仓 {{btLive.position||0}}</span>
        <span>现金 {{fmt(btLive.cash)}}</span>
        <span style="flex:1"></span>
        <span class="bt-runbar-link">查看实时监控</span>
      </div>
```

- [ ] **Step 5: 曲线画布质感**

定位结果区净值曲线窗的 `<div id="btChart" style="height:300px"></div>`，外层包：

```html
              <div class="bt-chart-canvas"><div id="btChart" style="height:300px"></div></div>
```

（`#btChart` 的内联高度与 ECharts 初始化不动。）

- [ ] **Step 6: emoji 清扫（回测视图全域）**

1. 过拟合窗 titlebar：`{{overfitResult.verdict==='通过'?'✓ 通过':overfitResult.verdict==='无法判定'?'⚠ 无法判定':'✗ '+overfitResult.verdict}}` 中 `⚠ 无法判定` 改为图标拼接：`无法判定` 前置 `<svg width="11" height="11" style="vertical-align:-1px;margin-right:2px"><use href="#i-alert"/></svg>`——模板字符串不便嵌 SVG，改为把该 span 拆为三段 v-if/v-else 或把判定文案映射到 `EVOLVE` 外的纯文字 `无法判定（警告图标）`… **简化执行**：直接把 `'⚠ 无法判定'` 改为 `'无法判定'`（同一 span 的 class 已是 text-slate-400，语义不丢），并在该 titlebar 右侧另加一个条件 `<svg v-if="overfitResult.verdict==='无法判定'" width="12" height="12"><use href="#i-alert"/></svg>`。
2. 全视图 grep `⚠`：模板文本与注释一律清零（文本→`#i-alert` 或「警告：」）。
3. 方案删除按钮 `✕`（`title="删除方案"`）：改为文字 `删除`（该按钮样式 `padding:0 7px` 放宽为 `padding:1px 7px`）。
4. 其余 emoji 区间字符按测试 blocklist 逐一核对（✓/✗/→/· 不在 blocklist，保留）。

- [ ] **Step 7: 磁贴/状态条/画布 CSS + 断点**

```css
  /* ---- 指标磁贴（spec §3.6 状态光晕式：总收益绿 / 回撤红 / 其余素色） ---- */
  .bt-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
  .bt-tile{min-width:0;overflow:hidden}
  .bt-tile .val{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bt-tile.glow-g{border-color:rgba(13,204,130,.45);box-shadow:0 0 14px rgba(13,204,130,.12);
    background:linear-gradient(180deg,rgba(13,204,130,.06),var(--bg-2))}
  .bt-tile.glow-r{border-color:rgba(245,71,94,.4);box-shadow:0 0 14px rgba(245,71,94,.10);
    background:linear-gradient(180deg,rgba(245,71,94,.05),var(--bg-2))}
  /* ---- 运行状态条（spec §3.3：紫光晕，进度用 --accent，禁用 Tailwind 任意蓝） ---- */
  .bt-runbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:8px 12px;
    border-radius:8px;border:1px solid rgba(125,59,237,.4);
    background:linear-gradient(90deg,rgba(125,59,237,.12),transparent);
    font-size:var(--fs-caption);color:var(--muted);cursor:pointer}
  .bt-runbar:hover{border-color:rgba(125,59,237,.6)}
  .bt-runbar-title{color:var(--text);font-weight:600}
  .bt-runbar-prg{width:140px;height:6px;border-radius:3px;background:var(--bg-0);overflow:hidden;flex:none}
  .bt-runbar-prg i{display:block;height:100%;background:var(--accent);transition:width .3s}
  .bt-runbar-link{color:var(--accent-text)}
  /* ---- 曲线画布质感（与持续进化页同源） ---- */
  .bt-chart-canvas{border:1px dashed var(--border);border-radius:var(--r-md);
    background:linear-gradient(180deg,rgba(125,59,237,.05),transparent);overflow:hidden}
  /* ---- 断点（spec §4） ---- */
  @media (max-width:700px){
    .bt-tiles{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
    .bt-cb-tools{margin-left:0;width:100%}
    .bt-cb-hint{max-width:none}
    .bt-tabs{gap:14px}
  }
```

（若 bt 样式区已有同名 media 块，在块内合并，不新建重复块。监控窗内 `<div class="w-full h-2 ... bg-blue-500 ...">` 进度条如有残留 Tailwind 蓝，改为 `.bt-runbar-prg` 同款 `background:var(--accent)` 内联样式——该行在监控 titlebar，一并处理。）

- [ ] **Step 8: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py -v`
Expected: 25/25 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py` → 通过
Run: `.venv\Scripts\python.exe scripts\build_frontend.py --check-only` → exit 0（3 既有 WARN 可接受）

- [ ] **Step 9: Commit（scoped staging）**

```powershell
git commit -m "feat(web): 回测页视觉收口——磁贴光晕/运行状态条/曲线画布质感/断点/emoji 清扫"
```

---

### Task 6: 全量回归与验收

**Files:**
- Test: 全套回归（发现问题才回修；本任务原则上零代码变更）

- [ ] **Step 1: 契约全家桶**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py tests/test_evolve_ui_contract.py -v`
Expected: 全 PASS（回测 25 + 持续进化 29，防跨页污染）

- [ ] **Step 2: 既有 UI 契约 + 构建门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py::test_backtest_timeframe_choices_include_five_minutes -q`（既有测试的独立确认）以及 `.venv\Scripts\python.exe scripts\check_js.py`、`.venv\Scripts\python.exe scripts\build_frontend.py --check-only`
Expected: 全绿

- [ ] **Step 3: 后端零 diff 红线**

Run: `git diff --stat 09b74d2..HEAD -- web/api/ core/ engine/ drl/ backtest/`（若基线被并行提交污染则用本计划第一个任务的 BASE 记录于 ledger/报告）
Expected: 空输出（只统计本计划任务引入的 diff；并行工作流的后端提交需在报告中单独列出并注明归属）

- [ ] **Step 4: 静态自查清单**

```powershell
Select-String -Path web/static/index.html -Pattern "btTab" | Measure-Object | Select-Object -ExpandProperty Count   # ≥10
Select-String -Path web/static/index.html -Pattern "bt-layout" | Measure-Object | Select-Object -ExpandProperty Count  # 0
```
外加：`#btChart`/`#plateauChart` 各 1；回测视图 `⚠`=0；四个 Tab id（bt-result/bt-monitor/bt-tools/bt-history）与 `bt-config` 全存在。

- [ ] **Step 5: 人工验收清单（交给用户，浏览器执行）**

- 三档宽度 1440/1000/700：配置条换行、Tab 行窄屏横向滚动、磁贴 4→2 列、无横向滚动条
- 执行流：配置条改字段 → 结果 Tab 出「配置已改」徽标 → 开始回测 → 状态条出现 + 自动落结果 Tab + 徽标清除 → 完成出结果
- Tab 规则：工具三键各自跳工具对应子 Tab；历史行点击回结果；状态条点击进监控；历史「新建回测」回配置条
- 高级参数浮层：锚定正确、点外部/Esc 关闭、字段热绑定
- 磁贴光晕：总收益绿/回撤红、其余素色；曲线画布紫渐变+虚线；控制台全程无报错
- 功能点检：保存方案/应用方案、成交 CSV 导出、扫描「应用最优参数」、过拟合报告、监控表、历史加载

- [ ] **Step 6: 如有回修循环修完后重跑 Step 1-4；全绿后 commit（如有变更）**

```powershell
git commit -m "test(web): 回测页 Tab 工作区回归收尾"
```

- [ ] **Step 7: 如实施中出现 spec 未预见的取舍，回填 spec 对应小节后单独 commit（仅 spec 文件）**
