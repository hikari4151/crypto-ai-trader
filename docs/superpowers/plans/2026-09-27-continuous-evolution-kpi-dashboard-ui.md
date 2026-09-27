# 持续进化页 KPI 仪表盘 UI 重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把持续进化页（`view==='evolve'`）整页重排为「页头操作区 → 8 个状态光晕 KPI 磁贴 → 全宽训练曲线 → 模型三卡 → DRL 实验台折叠条」的监控优先布局，配置/触发改为页头按钮弹出的下拉浮层，功能与数据绑定零丢失。

**Architecture:** 单文件改造：`web/static/index.html` 内重写 evolve 视图模板与局部 CSS，新增的 KPI 数据全部由现有 API（`/evolve/status`、`/evolve/rounds` 已拉取数据）在前端 computed 中计算，后端零改动。测试采用仓库既有的「静态 UI 契约」模式（`tests/test_backtest_ui_contract.py` 同款：读 index.html 断言结构），配合 `scripts/check_js.py` 与 `scripts/build_frontend.py --check-only` 门禁。

**Tech Stack:** CDN Vue 3（单文件内联）、原生 CSS（复用 `--bg/--border/--accent` 等既有设计令牌）、ECharts（容器与渲染逻辑不动）、pytest 静态契约测试。

**设计规范（唯一依据）:** `docs/superpowers/specs/2026-09-27-continuous-evolution-kpi-dashboard-design.md`
**设计稿留存:** `.superpowers/brainstorm/1-1790500775/content/`（v4 为终稿）

## Global Constraints

1. **零后端改动**：`web/api/evolve.py`、`drl/evolve_engine.py` 不允许出现任何 diff。
2. **零数据绑定改名**：既有 Vue 响应式字段/方法（`fetchEvolveStatus/evolvePause/evolveResume/evolveEnable/saveEvolveConfig/triggerFactorMiner/triggerStrategyDrl/triggerMetaController/resetEvolveAnchor/showEvolveVersions/evolveRollback/deployEvolveFactor/goEvolveComboBacktest/setEvolveCurveModel/setEvolveRange/drlPanelOpen/drlMode/fetchEvolveRounds/renderEvolveCurve/evolveCurveStat/evolveAnchorIssues/...`）全部保留原名。
3. **`#evolveCurveChart` 唯一**，仍归持续进化曲线渲染；ECharts 渲染逻辑（dataZoom、指纹门控、缩放记忆）不动。
4. **轮询策略不动**：status 5s、rounds 10s、与 visibilitychange 解耦（`startEvolvePolls` 块不改）。
5. **禁止 emoji**：evolve 视图模板内不出现 📈🧪🧠🤖🧬⏸⚡ 等表情符号；图标一律用站内 SVG sprite（`#i-*`）。
6. **设计令牌**：只用既有变量（`--bg-0 #0A0A0A`、`--bg-1 #141417`、`--bg-2 #1C1C21`、`--border #262629`、`--text #FAFAFA`、`--muted #A1A1AB`、`--text3 #70707A`、`--accent #7D3BED`、`--green #0DCC82`、`--red #F5475E`、`--orange #F59E0A`）；不引入新色值（光晕用既有色的 rgba）。
7. **KPI 轮次口径同源**：#6/#8 磁贴的轮次分类必须引用既有 `EVOLVE_STOP_STATUS`/`evolveVisual`，不得另写一套集合。
8. 图表高度 **220px → 260px**（两处内联样式都要改）。
9. 每个任务结束跑该任务的测试 + `check_js`，**全绿才 commit**；commit 会触发仓库 pre-commit 门禁（ruff/mypy/check_js/build_frontend），失败必须修。

## 对规格的三处细化（写入计划即生效，最终回填 spec）

- **页头不放大标题**：顶栏 `.tb-name` 已显示「持续进化」+「自优化模板 · 轮次迭代 · 版本回退」（`viewTitle/viewSub`），页头再放一份会同屏重复；页头仅保留按钮区（右对齐）。
- **部署版高亮规则具体化**：`deployed_fitness` 非空**且为三模型最高**的那张卡加紫描边（三模型往往都有部署值，逐卡高亮无意义）。
- **`⚠` 文本符号清除**：模型卡 warning 的 `<span>⚠</span>` 换成 `#i-alert` SVG（emoji 测试按此口径执行）。

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `web/static/index.html` | 修改 | 唯一实现载体：evolve 模板（原 3428-3976 行区域）、evolve 局部 CSS（原 1336-1423 行区域）、setup 脚本区新增 computed、`onDocClick` 扩展、setup return 追加 |
| `tests/test_evolve_ui_contract.py` | 新建 | 持续进化页静态 UI 契约（分任务增量添加断言） |

> 单文件是本仓库既定模式（全站 9 个视图都在 index.html），不拆分。

---

### Task 1: KPI/浮层数据层（script 区 computed + setup 导出）

**Files:**
- Modify: `web/static/index.html`（脚本区：`evolveCurveNote` computed 之后、`let evolveCurveChart = null;` 之前插入新 computed；setup return 的 evolve 段追加导出）
- Test: `tests/test_evolve_ui_contract.py`（新建）

**Interfaces:**
- Consumes: `evolveStatus`（reactive，含 `supervisor.loops_alive/degraded/heartbeat_age/restarts_by_loop`、`factor_miner/strategy_drl/meta_controller` 的 `fitness/next_run/deployed_fitness/deployed_version/rounds_total`、`*_interval`、`running/enabled/paused`）、`evolveRounds`（按模型缓存的轮次数组，行含 `ts/status`）、`evolveRoundsTotal`、`evolveCurveModel`、`evolveVisual(model)`（返回 `{rows, stopRounds, ...}`）、`EVOLVE_STOP_STATUS`、`evolveAnchorIssues`、`EVOLVE_MODELS`、`EVOLVE_MODEL_LABELS`、`clock`（秒级时钟 ref）、`setEvolveCurveModel`（模板用）
- Produces（Task 2-5 依赖，setup return 必须导出）:
  - `evolveKpiTiles` — computed<Array<{key, label, value, glow:'green'|'amber'|'red'|'plain', valueCls:''|'pos'|'warn'|'neg', sub:string, model?:string, anchor?:boolean, dot?:string}>>
  - `evolvePanelOpen` — ref<boolean>
  - `toggleEvolvePanel(force?: boolean): void`
  - `kpiTileClick(t: KpiTile, ev: MouseEvent): void` — KPI 统一点击处理（model 分支切曲线；anchor 分支 **必须 `ev.stopPropagation()`**，否则同一次点击冒泡到 document 的 `onDocClick` 会把刚打开的浮层立即关掉）
  - `evolveAccentModel` — computed<string>（部署旗舰模型 key，无部署返回 `''`）

- [ ] **Step 1: 写失败契约测试**

新建 `tests/test_evolve_ui_contract.py`：

```python
"""持续进化页 KPI 仪表盘重设计（2026-09-27）UI 契约。

规范：docs/superpowers/specs/2026-09-27-continuous-evolution-kpi-dashboard-design.md
模式与 tests/test_backtest_ui_contract.py 同款：读 index.html 断言结构。
"""
from pathlib import Path


HTML = Path("web/static/index.html").read_text(encoding="utf-8")


def _setup_return() -> str:
    """setup() 的 return 对象片段（新增响应式量必须出现在这里才能进模板）。"""
    anchor = HTML.find("return {view, settingsTab")
    assert anchor != -1, "setup return 锚点丢失（组件结构被改动？）"
    return HTML[anchor:]


def _evolve_view() -> str:
    start = HTML.find("<!-- ========== 持续进化引擎 ========== -->")
    assert start != -1, "持续进化视图模板起点丢失"
    return HTML[start:HTML.find("</section>", start)]


def test_kpi_tiles_computed_defined():
    assert "const evolveKpiTiles = computed" in HTML


def test_kpi_computed_exported_from_setup():
    assert "evolveKpiTiles" in _setup_return()


def test_loop_kpi_uses_supervisor_loops_alive():
    start = HTML.find("const evolveKpiLoops = computed")
    assert start != -1, "三循环 KPI computed 缺失"
    body = HTML[start:HTML.find("const evolveKpiTiles", start)]
    assert "loops_alive" in body
    assert "supervisor" in body


def test_rounds_kpi_uses_pipeline_rounds_total():
    start = HTML.find("const evolveKpiTiles = computed")
    body = HTML[start:HTML.find("const evolveAccentModel", start)]
    assert "rounds_total" in body, "累计轮次必须用 _pipeline_view.rounds_total（零额外请求）"


def test_oos_kpi_shares_curve_stop_status():
    # 通过率与曲线红点必须同源：同一 EVOLVE_STOP_STATUS + evolveVisual
    start = HTML.find("const evolveKpiTiles = computed")
    body = HTML[start:HTML.find("const evolveAccentModel", start)]
    assert "EVOLVE_STOP_STATUS" in body
    assert "evolveVisual(" in body


def test_panel_state_exported():
    assert "evolvePanelOpen" in _setup_return()
    assert "toggleEvolvePanel" in _setup_return()
    assert "kpiTileClick" in _setup_return()


def test_accent_model_defined_and_exported():
    assert "const evolveAccentModel = computed" in HTML
    assert "evolveAccentModel" in _setup_return()


def test_kpi_stale_state_fallback():
    # §5 空态：连续轮询失败时 KPI 必须置灰，不能拿陈旧数据当实时
    start = HTML.find("const evolveKpiTiles = computed")
    body = HTML[start:HTML.find("const evolveAccentModel", start)]
    assert "evolveFailStreak" in body
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: FAIL（`const evolveKpiTiles = computed` not found 等，全部红）

- [ ] **Step 3: 插入 computed（定位锚：`let evolveCurveChart = null;`，插到它前面）**

```js
    // ---- 配置/触发页头浮层（2026-09-27 重设计）：菜单按钮点开，点外部收起 ----
    const evolvePanelOpen = ref(false);
    function toggleEvolvePanel(force) {
      evolvePanelOpen.value = (force === true) ? true : !evolvePanelOpen.value;
    }
    // KPI 磁贴统一点击：前三张切曲线模型；锚点磁贴开浮层。
    // ⚠️ anchor 分支必须 stopPropagation：否则本次点击冒泡到 document 的
    //    onDocClick（此时 evolvePanelOpen 刚变 true）会把浮层立刻关掉，表现为"点了打不开"。
    function kpiTileClick(t, ev) {
      if (t.model) { setEvolveCurveModel(t.model); return; }
      if (t.anchor) { ev.stopPropagation(); toggleEvolvePanel(true); }
    }
    // ---- KPI 磁贴数据层：8 磁贴全部由现有数据前端计算，后端零改动 ----
    // 轮次口径与曲线红点同源（evolveVisual / EVOLVE_STOP_STATUS），杜绝"文本与图不符"。
    const evolveKpiLoops = computed(() => {
      const sup = evolveStatus.supervisor || {};
      void clock.value;                 // 挂秒级时钟：下次训练倒计时每秒刷新
      const nowSec = Date.now() / 1000;
      const defs = [
        { key:'factor_miner', label:'因子挖掘', intKey:'factor_miner_interval' },
        { key:'strategy_drl', label:'策略 DRL', intKey:'strategy_drl_interval' },
        { key:'meta_controller', label:'元策略', intKey:'meta_controller_interval' },
      ];
      return defs.map(d => {
        const p = evolveStatus[d.key] || {};
        let state, glow, valueCls;
        if (!evolveStatus.running || !evolveStatus.enabled) { state='已停用'; glow='plain'; valueCls=''; }
        else if (evolveStatus.paused) { state='已暂停'; glow='amber'; valueCls='warn'; }
        else if (sup.loops_alive && sup.loops_alive[d.key] === false) { state='已停训'; glow='red'; valueCls='neg'; }
        else { state='运行中'; glow='green'; valueCls='pos'; }
        const next = Number(p.next_run || 0);
        const interval = Number(evolveStatus[d.intKey] || 0);
        let sub = interval ? interval + 's/轮' : '';
        if (state === '运行中' && next > nowSec) {
          const left = Math.ceil(next - nowSec);
          const mm = String(Math.floor(left / 60)).padStart(2, '0');
          const ss = String(left % 60).padStart(2, '0');
          sub = `下次 ${mm}:${ss} · ` + sub;
        } else if (state === '已暂停') sub = '随引擎暂停 · ' + sub;
        else if (state === '已停用') sub = sub || '引擎未启用';
        return { key:d.key, label:d.label, value:state, glow, valueCls, sub, model:d.key, dot:glow };
      });
    });
    const evolveKpiTiles = computed(() => {
      const loops = evolveKpiLoops.value;
      const sup = evolveStatus.supervisor || {};
      // ⑤ 空态（spec §5）：连续 ≥3 次状态轮询失败 ⇒ 全部置灰，不拿陈旧数据当实时
      if (evolveFailStreak.value >= 3) {
        const blank = { value:'—', glow:'plain', valueCls:'', sub:'状态未知' };
        return [
          ...loops.map(l => ({ key:l.key, label:l.label, ...blank, model:l.key })),
          { key:'keepalive', label:'引擎保活', ...blank },
          { key:'anchor', label:'锚点健康', ...blank, anchor:true },
          { key:'rounds', label:'累计轮次', ...blank },
          { key:'best', label:'最佳 fitness', ...blank },
          { key:'oos', label:'OOS 通过率', ...blank },
        ];
      }
      // ④ 引擎保活：supervisor 降级判定 + 心跳/自愈
      let kaVal = '—', kaGlow = 'plain', kaCls = '', kaSub = '状态未知';
      if (evolveStatus.running && evolveStatus.enabled) {
        kaVal = sup.degraded ? '降级' : '正常';
        kaGlow = sup.degraded ? 'red' : 'plain';
        kaCls = sup.degraded ? 'neg' : '';
        const hb = sup.heartbeat_age != null ? Math.round(sup.heartbeat_age) : null;
        const rs = Object.values(sup.restarts_by_loop || {}).reduce((s, n) => s + (Number(n) || 0), 0);
        kaSub = (hb != null ? `心跳 ${hb}s 前` : '心跳未知') + ` · 自愈 ${rs} 次`;
      }
      // ⑤ 锚点健康：复用现有 evolveAnchorIssues（3 - 告警数）
      const issues = evolveAnchorIssues.value;
      const anchorTile = {
        key:'anchor', label:'锚点健康', value:(3 - issues.length) + ' / 3',
        glow: issues.length ? 'red' : 'green', valueCls:'',
        sub: issues.length ? (issues[0].label + ' · ' + issues[0].tag) : '三模型基线正常',
        anchor: true,
      };
      // ⑥ 累计轮次：_pipeline_view.rounds_total 三模型合计（status 同源，零额外请求）
      const roundsSum = EVOLVE_MODELS.reduce(
        (s, m) => s + (Number(evolveStatus[m.key] && evolveStatus[m.key].rounds_total) || 0), 0);
      const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);
      const curRows = evolveRounds[evolveCurveModel.value] || [];
      const todayN = curRows.filter(r => r.ts && r.ts * 1000 >= todayStart.getTime()).length;
      const curLabel = EVOLVE_MODEL_LABELS[evolveCurveModel.value] || evolveCurveModel.value;
      // ⑦ 最佳 fitness：三模型最大者，数值色按正负
      let best = null;
      for (const m of EVOLVE_MODELS) {
        const f = Number(evolveStatus[m.key] && evolveStatus[m.key].fitness);
        if (!Number.isFinite(f)) continue;
        if (!best || f > best.f) {
          best = { f, name: m.name,
                   ver: Number(evolveStatus[m.key] && evolveStatus[m.key].deployed_version) || 0 };
        }
      }
      // ⑧ OOS 通过率：近 30 轮，拦截口径 = EVOLVE_STOP_STATUS（与曲线红点同源）
      const v = evolveVisual(evolveCurveModel.value);
      const recent = (v.rows || []).slice(-30);
      const stopN = recent.filter(r => EVOLVE_STOP_STATUS.has(r.status)).length;
      const passN = recent.length - stopN;
      const passPct = recent.length ? Math.round(passN / recent.length * 100) : null;
      return [
        ...loops,
        { key:'keepalive', label:'引擎保活', value:kaVal, glow:kaGlow, valueCls:kaCls, sub:kaSub },
        anchorTile,
        { key:'rounds', label:'累计轮次', value: roundsSum.toLocaleString('en-US'),
          glow:'plain', valueCls:'', sub:`今日（${curLabel}）+${todayN}` },
        { key:'best', label:'最佳 fitness', glow:'plain',
          value: best ? (best.f >= 0 ? '+' : '') + best.f.toFixed(4) : '—',
          valueCls: best ? (best.f >= 0 ? 'pos' : 'neg') : '',
          sub: best ? `${best.name}${best.ver ? ' · v' + best.ver + ' 已部署' : ''}` : '尚无训练记录' },
        { key:'oos', label:'OOS 通过率', glow:'plain', valueCls:'',
          value: passPct == null ? '—' : passPct + '%',
          sub: recent.length ? `${passN} 过 / ${stopN} 拦` : '暂无轮次数据' },
      ];
    });
    // 部署旗舰高亮：deployed_fitness 非空且最高者；均无部署则不高亮
    const evolveAccentModel = computed(() => {
      let best = null;
      for (const m of EVOLVE_MODELS) {
        const f = evolveStatus[m.key] ? evolveStatus[m.key].deployed_fitness : null;
        if (f == null || !Number.isFinite(Number(f))) continue;
        if (!best || Number(f) > best.f) best = { key: m.key, f: Number(f) };
      }
      return best ? best.key : '';
    });
```

- [ ] **Step 4: setup return 追加导出**

定位 setup return 中现有行 `evolveRounds, evolveCurveModel, evolveCurveHasData, evolveCurveStat, fetchEvolveRounds, renderEvolveCurve, setEvolveCurveModel, setEvolveRange, evolveCurveNote, evolveExtra,`，改为在该行末尾追加：

```js
      evolvePanelOpen, toggleEvolvePanel, kpiTileClick, evolveKpiTiles, evolveAccentModel,
```

- [ ] **Step 5: 跑测试确认通过 + JS 语法门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: PASS（8 个测试全绿）

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: `JS 语法完全通过`

- [ ] **Step 6: Commit**

```powershell
git add web/static/index.html tests/test_evolve_ui_contract.py
git commit -m "feat(web): 持续进化页 KPI 数据层——8 磁贴 computed + 配置/触发浮层状态 + 部署旗舰高亮"
```

---

### Task 2: 页头按钮区 + KPI 磁贴模板与光晕 CSS

**Files:**
- Modify: `web/static/index.html`（evolve 模板页头行；页头后插入 KPI 网格；局部样式区新增 `.evolve-kpi*` 规则）
- Test: `tests/test_evolve_ui_contract.py`（追加断言）

**Interfaces:**
- Consumes: Task 1 的 `evolveKpiTiles`、`evolvePanelOpen`、`toggleEvolvePanel`；既有 `evolvePause/evolveResume/evolveEnable/fetchEvolveStatus/setEvolveCurveModel`
- Produces: `.evolve-kpis` 网格容器、`.evolve-panel` 按钮包装（Task 3 在其中填充 Teleport 浮层）

- [ ] **Step 1: 写失败契约测试**

在 `tests/test_evolve_ui_contract.py` 末尾追加：

```python
def test_kpi_grid_rendered_in_evolve_view():
    v = _evolve_view()
    assert 'class="evolve-kpis"' in v
    assert 'v-for="t in evolveKpiTiles"' in v
    assert "kpiTileClick(t, $event)" in v


def test_header_has_panel_button():
    v = _evolve_view()
    assert "toggleEvolvePanel()" in v
    assert "配置 / 触发" in v


def test_header_keeps_existing_actions():
    v = _evolve_view()
    for call in ("evolvePause()", "evolveResume()", "evolveEnable()", "fetchEvolveStatus()"):
        assert call in v
    assert 'disabled title="进化日志导出尚未提供接口"' in v


def test_evolve_view_has_no_emoji():
    v = _evolve_view()
    blocked = "📈🧪🧠🤖🧬⏸⚡⌚🚀🎯🔧"
    assert not any(ch in v for ch in blocked)


def test_kpi_glow_css_present():
    assert ".evolve-kpi.on-green{border-color:rgba(13,204,130,.45)" in HTML
    assert ".evolve-kpi.on-amber" in HTML
    assert ".evolve-kpi.on-red" in HTML
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 新增 5 个测试 FAIL（旧测试保持绿）

- [ ] **Step 3: 重写页头 + 插入 KPI 网格**

定位唯一锚点行（单行长文本）：以 `<div class="page-head evolve-page-head"><div class="page-actions">` 开头、`刷新状态</button></div></div>` 结尾的整行，**整行替换**为：

```html
      <div class="page-head evolve-page-head"><div class="page-actions">
        <button class="mac-btn" v-if="!evolveStatus.paused" @click="evolvePause()" :disabled="!evolveStatus.running || !evolveStatus.enabled" title="暂停持续训练（不停止进程）"><svg width="13" height="13"><use href="#i-pause"/></svg>暂停</button>
        <button class="mac-btn primary" v-else @click="evolveResume()" :disabled="!evolveStatus.running || !evolveStatus.enabled" title="恢复持续训练"><svg width="13" height="13"><use href="#i-play"/></svg>恢复</button>
        <button class="mac-btn" v-if="!evolveStatus.enabled" @click="evolveEnable()" :disabled="!evolveStatus.running" title="启用自动进化"><svg width="13" height="13"><use href="#i-power"/></svg>启用</button>
        <span class="evolve-panel" @click.stop>
          <button class="mac-btn evolve-panel-btn" :class="{open:evolvePanelOpen}" data-form-key="evolvePanel" aria-haspopup="true" :aria-expanded="evolvePanelOpen?'true':'false'" @click.stop="toggleEvolvePanel()"><svg width="13" height="13"><use href="#i-settings"/></svg>配置 / 触发</button>
          <!-- 浮层本体：Task 3 在此填充 Teleport -->
        </span>
        <button class="mac-btn" disabled title="进化日志导出尚未提供接口"><svg width="13" height="13"><use href="#i-download"/></svg>导出进化日志</button>
        <button class="mac-btn" @click="fetchEvolveStatus()" :disabled="!evolveStatus.running"><svg width="13" height="13"><use href="#i-refresh"/></svg>刷新状态</button>
      </div></div>
      <!-- KPI 磁贴 ×8（状态光晕）：前三张点按切换曲线模型，锚点磁贴点按开浮层 -->
      <div class="evolve-kpis">
        <button v-for="t in evolveKpiTiles" :key="t.key" type="button" class="evolve-kpi"
          :class="t.glow!=='plain' ? 'on-'+t.glow : ''"
          @click="kpiTileClick(t, $event)"
          :title="t.model ? ('切换曲线到'+t.label) : (t.anchor ? '打开配置/触发查看锚点详情' : '')">
          <span class="evolve-kpi-lab"><i v-if="t.dot" class="evolve-kpi-dot" :class="t.dot"></i>{{t.label}}</span>
          <span class="evolve-kpi-val" :class="t.valueCls">{{t.value}}</span>
          <span class="evolve-kpi-sub">{{t.sub}}</span>
        </button>
      </div>
```

同时把页头下方三条历史注释（`运行状态窗口已移除（2026-09-19）…`）更新末句为：`「立即训练」入口已并入页头「配置 / 触发」浮层（2026-09-27）。`

- [ ] **Step 4: 局部样式区新增 KPI/页头 CSS**

定位样式区锚 `/* ---------- 持续进化页 PDF 工作台：仅作用于 evolve 视图 ---------- */`，把该注释改为 `/* ---------- 持续进化页 KPI 仪表盘（2026-09-27 重设计）：仅作用于 evolve 视图 ---------- */`，并在 `.evolve-page-head{min-height:52px}` 规则后插入：

```css
  .evolve-page-head{justify-content:flex-end}
  /* ---- KPI 磁贴：等行高 + 三行文字各就各位（nowrap+ellipsis），任何文案不得换行挤压 ---- */
  .evolve-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;grid-auto-rows:1fr}
  .evolve-kpi{display:flex;flex-direction:column;justify-content:space-between;gap:4px;min-width:0;
    padding:11px 12px;border-radius:8px;text-align:left;cursor:pointer;font-family:inherit;
    background:var(--bg-1);border:1px solid var(--border);overflow:hidden;
    transition:border-color .15s, box-shadow .15s}
  .evolve-kpi:hover{border-color:var(--border2)}
  .evolve-kpi:focus-visible{outline:2px solid var(--accent-text);outline-offset:1px}
  .evolve-kpi.on-green{border-color:rgba(13,204,130,.45);box-shadow:0 0 14px rgba(13,204,130,.12);
    background:linear-gradient(180deg,rgba(13,204,130,.06),var(--bg-1))}
  .evolve-kpi.on-amber{border-color:rgba(245,158,10,.45);box-shadow:0 0 14px rgba(245,158,10,.10);
    background:linear-gradient(180deg,rgba(245,158,10,.06),var(--bg-1))}
  .evolve-kpi.on-red{border-color:rgba(245,71,94,.45);box-shadow:0 0 14px rgba(245,71,94,.10);
    background:linear-gradient(180deg,rgba(245,71,94,.06),var(--bg-1))}
  .evolve-kpi-lab{display:flex;align-items:center;gap:5px;font-size:var(--fs-micro);color:var(--text3);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:none}
  .evolve-kpi-dot{width:6px;height:6px;border-radius:50%;flex:none;background:var(--muted)}
  .evolve-kpi-dot.green{background:var(--green)}
  .evolve-kpi-dot.amber{background:var(--orange)}
  .evolve-kpi-dot.red{background:var(--red)}
  .evolve-kpi-val{font-size:16px;font-weight:650;color:var(--text);font-variant-numeric:tabular-nums;
    letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:none;line-height:1.25}
  .evolve-kpi-val.pos{color:var(--green)}
  .evolve-kpi-val.warn{color:var(--orange)}
  .evolve-kpi-val.neg{color:var(--red)}
  .evolve-kpi-sub{font-size:var(--fs-micro);color:var(--text3);white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;flex:none}
  /* ---- 配置/触发页头按钮与浮层 ---- */
  .evolve-panel{position:relative;display:inline-flex}
  .evolve-panel-btn.open{border-color:var(--accent);color:var(--accent-text);background:var(--accent-soft)}
  .evolve-panel-pop{width:min(330px,92vw);gap:0;padding:11px 12px}
  .evolve-pop-head{display:flex;align-items:center;gap:6px;font-size:var(--fs-body);font-weight:650;
    color:var(--text);padding-bottom:8px}
  .evolve-pop-group{padding:9px 0;border-top:1px solid var(--border);min-width:0}
  .evolve-pop-group:first-of-type{border-top:0;padding-top:2px}
  .evolve-pop-group .mac-label{margin-bottom:6px}
```

- [ ] **Step 5: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 全部 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: 通过

- [ ] **Step 6: Commit**

```powershell
git add web/static/index.html tests/test_evolve_ui_contract.py
git commit -m "feat(web): 持续进化页头（配置/触发按钮）+ 8 状态光晕 KPI 磁贴"
```

---

### Task 3: 配置 / 触发下拉浮层（侧栏三卡内容迁移）

**Files:**
- Modify: `web/static/index.html`（页头 `.evolve-panel` 内填充 Teleport 浮层；从原侧栏剪切三卡内容；`onDocClick` 扩展）
- Test: `tests/test_evolve_ui_contract.py`（追加断言）

**Interfaces:**
- Consumes: Task 1 `evolvePanelOpen/toggleEvolvePanel`；Task 2 `.evolve-panel` 包装；既有 `formDrop/toggleFormDrop`（标的池多选）、`saveEvolveConfig`、`triggerFactorMiner/triggerStrategyDrl/triggerMetaController`、`evolveAnchorIssues/evolveAnchorLocked/resetEvolveAnchor`、`evolveConfig/evolveTfOptions`
- Produces: `.evolve-panel-pop` 浮层（含四段：标的池+周期+保存 / 手动触发 / 锚点）；浮层内标的池二级菜单的 Teleport 节点追加类 `evolve-symbols-pop`（供点外部判定"算内部"）

- [ ] **Step 1: 写失败契约测试**

`tests/test_evolve_ui_contract.py` 末尾追加：

```python
def test_panel_pop_contains_all_functionality():
    v = _evolve_view()
    assert 'class="menu-sub evolve-panel-pop"' in v
    assert "保存配置（热更新）" in v
    assert "triggerFactorMiner()" in v
    assert "triggerStrategyDrl()" in v
    assert "triggerMetaController()" in v
    assert "resetEvolveAnchor(a.model)" in v
    assert "formDrop==='evolveSymbols'" in v


def test_old_side_cards_removed_from_view():
    v = _evolve_view()
    assert "evolve-side-config" not in v
    assert "evolve-side-trigger" not in v
    assert "evolve-side-anchor" not in v


def test_doc_click_closes_panel():
    start = HTML.find("onDocClick = (e)")
    end = HTML.find("document.addEventListener('click', onDocClick)")
    assert start != -1 and end > start
    body = HTML[start:end]
    assert "evolvePanelOpen" in body and ".evolve-panel" in body
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 新增 3 个 FAIL

- [ ] **Step 3: 填充浮层（页头 `.evolve-panel` 内，替换 `<!-- 浮层本体：Task 3 在此填充 Teleport -->` 注释行）**

```html
          <Teleport to="body">
            <div v-if="evolvePanelOpen" class="menu-sub evolve-panel-pop" @click.stop>
              <div class="evolve-pop-head"><span>配置 / 触发</span><span style="flex:1"></span><button class="mac-btn btn-sm" style="padding:2px 9px" @click="toggleEvolvePanel()">收起</button></div>
              <div class="evolve-pop-group">
                <div class="mac-label">训练标的池</div>
                <div class="menu-drop" :class="{open: formDrop==='evolveSymbols'}" style="width:100%">
                  <!-- 原训练配置卡的「标的池 menu-drop 整块」原样移入（含按钮、Teleport 二级菜单、手输添加行） -->
                </div>
                <div class="mac-label" style="margin-top:9px">训练时间周期</div>
                <mac-select v-model="evolveConfig.timeframe" :options="evolveTfOptions" style="width:100%"></mac-select>
                <button class="mac-btn primary" style="width:100%;margin-top:10px;justify-content:center" @click="saveEvolveConfig()"><svg width="13" height="13"><use href="#i-check"/></svg>保存配置（热更新）</button>
              </div>
              <div class="evolve-pop-group">
                <div class="mac-label">手动触发训练</div>
                <div class="evolve-trigger-stack">
                  <!-- 原「手动触发训练」卡的 3 个按钮原样移入（含 :disabled 条件与「训练中…」文案） -->
                </div>
              </div>
              <div class="evolve-pop-group" v-if="evolveAnchorIssues.length" :style="{borderColor: evolveAnchorLocked ? 'rgba(245,71,94,.45)' : 'rgba(245,158,10,.3)'}">
                <div class="mac-label">回退基线（锚点）</div>
                <div class="evolve-anchor-item" v-for="a in evolveAnchorIssues" :key="a.model">
                  <!-- 原「回退基线（锚点）」卡的 v-for 条目原样移入（pill/label/原因/重置按钮） -->
                </div>
              </div>
            </div>
          </Teleport>
```

**移入内容的精确来源（原样剪切，不重写）：**

1. 标的池块：原 `<div class="mac-window evolve-side-config">` 卡内，从 `<div class="menu-drop" :class="{open: formDrop==='evolveSymbols'}"...>` 到其配对 `</div>`（含 `<button class="mac-input" ... :data-form-key="'evolveSymbols'"`、`<Teleport to="body">` 二级菜单、手输添加行）→ 粘进浮层第一个 `menu-drop` 占位注释处；
2. 二级菜单 Teleport 节点 `<div v-if="formDrop==='evolveSymbols'" ref="formDropMenu" class="menu-sub" ...>` 追加类名：改为 `class="menu-sub evolve-symbols-pop"`（点外部判定需要）；
3. 触发按钮：原 `<div class="mac-window evolve-side-trigger">` 卡的 `.evolve-trigger-stack` 内 3 个 `<button>` → 粘进浮层 `.evolve-trigger-stack` 占位注释处；
4. 锚点条目：原 `<div class="mac-window evolve-side-anchor">` 卡 `.mac-body` 内的 `<div class="evolve-anchor-item" ...>` 整块 → 粘进浮层锚点占位注释处。

- [ ] **Step 4: 从原位删除三张侧栏卡（内容已移入浮层，整卡删除）**

依次删除（连同各自的开闭标签）：
- `<div class="mac-window evolve-side-config">` 整卡
- `<div v-if="evolveAnchorIssues.length" class="mac-window evolve-side-anchor" ...>` 整卡
- `<div class="mac-window evolve-side-trigger">` 整卡

删除后 `.evolve-side-column` 容器暂时为空——**该容器的移除属于 Task 4**，本任务不碰布局结构。

- [ ] **Step 5: 扩展 `onDocClick`（点外部关闭浮层）**

定位 `onDocClick = (e) => {` 块，在 `if (formDrop.value && ...)` 行之后追加一行：

```js
        // 配置/触发页头浮层：点外部收起（浮层本体与内部标的池二级菜单都算"内部"）
        if (evolvePanelOpen.value && e.target.closest &&
            !e.target.closest('.evolve-panel, .evolve-panel-pop, .evolve-symbols-pop')) evolvePanelOpen.value = false;
```

- [ ] **Step 6: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 全部 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: 通过

- [ ] **Step 7: Commit**

```powershell
git add web/static/index.html tests/test_evolve_ui_contract.py
git commit -m "feat(web): 持续进化配置/触发改为页头按钮下拉浮层（标的池/周期/触发三键/锚点/保存，功能零丢失）"
```

---

### Task 4: 布局扁平化 + 曲线面板改造（独立统计行 / 260px）

**Files:**
- Modify: `web/static/index.html`（evolve 模板：拆除 `.evolve-layout` 双栏壳；曲线面板改造；模型区分区标签；局部样式：退役双栏规则）
- Test: `tests/test_evolve_ui_contract.py`（追加断言）

**Interfaces:**
- Consumes: Task 3 之后 `.evolve-side-column` 已空；既有 `evolveCurveStat`、`evolveCurveNote`、`setEvolveCurveModel/setEvolveRange`、`EVOLVE_MODELS` 模型卡
- Produces: `.evolve-page` 直接子元素序列（head → kpis → curve 窗口 → 分区标签 → models 窗口 → lab）；`.evolve-curve-stats` 统计行

- [ ] **Step 1: 写失败契约测试**

`tests/test_evolve_ui_contract.py` 末尾追加：

```python
def test_dual_column_layout_retired():
    assert ".evolve-layout" not in HTML
    assert ".evolve-main-column" not in HTML
    assert ".evolve-side-column" not in HTML
    v = _evolve_view()
    assert "evolve-layout" not in v
    assert "evolve-main-column" not in v
    assert "evolve-side-column" not in v


def test_curve_stats_row_standalone():
    v = _evolve_view()
    assert 'class="evolve-curve-stats"' in v
    assert "{{evolveCurveStat}}" in v
    # 统计文案不得再挤在标题栏里
    titlebar = v[v.find("训练轮次曲线"):v.find("evolve-curve-tabs")]
    assert "evolveCurveStat" not in titlebar


def test_chart_height_260():
    v = _evolve_view()
    assert v.count("height:260px") >= 2
    assert "height:220px" not in v


def test_model_section_label_present():
    assert "模型状态 · 部署与成果" in _evolve_view()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 新增 4 个 FAIL

- [ ] **Step 3: 拆除双栏壳（模板）**

evolve 模板内按锚点依次操作：
1. 删除 `<div class="evolve-layout">` 一行；
2. 删除 `<div class="evolve-side-column">` 一行及紧随其后的空行/配对 `</div>`（Task 3 已清空卡，只剩壳）；
3. 删除 `<div class="evolve-main-column">` 一行及其配对 `</div>`（位于版本弹层 `<transition name="fade">` 之前的 `</div>`）；
4. 拆除后顺序应为：KPI 网格 → `<!-- 训练轮次曲线 -->` 注释 + `mac-window evolve-main-curve` → `mac-window evolve-main-models` → 版本弹层 → lab。若 `.evolve-page` 的 `space-y-3.5` 与 `.evolve-page{gap:16px}` 重复控制间距，保留 CSS gap，模板 `class="space-y-3.5 evolve-page"` 里的 `space-y-3.5` 改为仅 `evolve-page`（CSS `display:flex;flex-direction:column;gap:12px` 统一负责）。

- [ ] **Step 4: 曲线面板改造（模板）**

1. 标题栏删除统计 span：`<span class="text-[length:var(--fs-caption)] text-slate-500 ml-auto">{{evolveCurveStat}}</span>` 整个删除；
2. 在 `.evolve-curve-tabs` div 结束标签之后、`<div class="relative evolve-chart-wrap"...>` 之前插入：

```html
          <div class="evolve-curve-stats"><span>{{evolveCurveStat}}</span></div>
```

3. 两处 `style="height:220px"` 改为 `style="height:260px"`（`.evolve-chart-wrap` 与 `#evolveCurveChart`，用 `grep -n "height:220px" web/static/index.html` 确认 evolve 视图内无残留）；
4. 模型窗口 titlebar 之后、`<div class="evolve-model-grid">` 之前插入分区标签：

```html
          <div class="lab">模型状态 · 部署与成果</div>
```

- [ ] **Step 5: 局部样式：退役双栏规则 + 新增统计行样式**

1. 删除整块双栏 CSS（含那段 `grid-row:1` 长注释）：从 `.evolve-layout{display:grid;...}` 到 `.evolve-main-column > *,.evolve-side-column > *{min-width:0}` 为止（含中间注释），以及 `.evolve-main-column{grid-column:1}` / `.evolve-side-column{grid-column:2}`；
2. 保留 `.evolve-panel .mac-body{min-width:0}` 等仍被引用的规则；
3. 在 `.evolve-curve-tabs` 规则后插入：

```css
  .evolve-curve-stats{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;
    font-size:var(--fs-caption);color:var(--muted)}
```

4. 删除 `@media (max-width:900px){ ... }` 中仅服务于双栏的规则（`.evolve-layout{grid-template-columns:1fr}` 与列容器 `grid-column/grid-row` 重置），若该 media 块删空则整块移除（`.evolve-lab` 相关规则在 Task 5 的断点重写中统一处理）。

- [ ] **Step 6: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 全部 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: 通过

- [ ] **Step 7: Commit**

```powershell
git add web/static/index.html tests/test_evolve_ui_contract.py
git commit -m "feat(web): 持续进化页布局扁平化——退役双栏、曲线全宽独立统计行、图表 260px"
```

---

### Task 5: 模型卡部署旗舰高亮 + 警告图标化 + 实验台模式 pills + 断点重写

**Files:**
- Modify: `web/static/index.html`（模型卡 style 绑定；`⚠` 文本清除；lab titlebar pills；断点 media 块）
- Test: `tests/test_evolve_ui_contract.py`（追加断言）

**Interfaces:**
- Consumes: Task 1 `evolveAccentModel`；既有 `drlMode/drlPanelOpen/drlPanelOpen`、`EVOLVE_MODELS` v-for、`.pill` 样式
- Produces: `.evolve-model-card.evolve-accent`、`.evolve-lab-modes`、新断点（1100/700）

- [ ] **Step 1: 写失败契约测试**

`tests/test_evolve_ui_contract.py` 末尾追加：

```python
def test_accent_binding_on_model_card():
    v = _evolve_view()
    assert "evolveAccentModel" in v
    assert "evolve-accent" in v


def test_no_text_warning_glyph_in_view():
    assert "⚠" not in _evolve_view()


def test_lab_mode_pills_clickable():
    v = _evolve_view()
    assert 'class="evolve-lab-modes"' in v
    assert "drlMode='factor';drlPanelOpen=true" in v


def test_kpi_breakpoints():
    assert ".evolve-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}" in HTML
    assert ".evolve-kpis{grid-template-columns:1fr}" in HTML
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 新增 4 个 FAIL

- [ ] **Step 3: 模型卡旗舰高亮（模板）**

定位 `<div v-for="m in EVOLVE_MODELS" :key="m.key" class="mac-card mb-3 evolve-model-card" :style="{borderColor:m.border}">`，替换为：

```html
          <div v-for="m in EVOLVE_MODELS" :key="m.key" class="mac-card mb-3 evolve-model-card"
            :class="m.key===evolveAccentModel ? 'evolve-accent' : ''"
            :style="{borderColor: m.key===evolveAccentModel ? 'rgba(125,59,237,.5)' : m.border}">
```

- [ ] **Step 4: `⚠` 文本清除（模板）**

evolve 视图内定位 `<div class="evolve-model-warning">` 的 `<span>⚠</span>`，替换为：

```html
              <svg width="11" height="11" style="flex:none;margin-top:2px"><use href="#i-alert"/></svg>
```

用 `grep -n "⚠" web/static/index.html` 检查 evolve 视图范围（3428-3976 的旧区间）内其余出现处：模板文本一律换 `#i-alert` SVG；纯注释里的 `⚠` 改写为「警告：」（测试按整个视图字符串断言，注释也算）。

- [ ] **Step 5: 实验台 titlebar 加模式 pills（模板）**

定位 lab titlebar：`<span class="mac-title"><svg><use href="#i-dna"/></svg>手动 DRL 实验台</span>` 与紧随的展开按钮之间插入：

```html
          <span class="evolve-lab-modes">
            <button type="button" class="pill" :class="drlMode==='factor'?'on':''" @click="drlMode='factor';drlPanelOpen=true">RL 因子挖掘</button>
            <button type="button" class="pill" :class="drlMode==='strategy'?'on':''" @click="drlMode='strategy';drlPanelOpen=true">策略训练</button>
            <button type="button" class="pill" :class="drlMode==='models'?'on':''" @click="drlMode='models';drlPanelOpen=true">模型库</button>
            <button type="button" class="pill" :class="drlMode==='oos'?'on':''" @click="drlMode='oos';drlPanelOpen=true">跨行情 OOS</button>
          </span>
```

展开按钮去掉 `ml-auto`（改由 `.evolve-lab-modes{margin-left:auto}` 推右），按钮文案逻辑 `{{drlPanelOpen?'收起':'展开'}}` 不变。

- [ ] **Step 6: 样式：accent/lab pills/断点重写**

1. 在 `.evolve-model-card{...}` 规则后插入：

```css
  .evolve-model-card.evolve-accent{border-color:rgba(125,59,237,.5);
    box-shadow:0 0 0 1px rgba(125,59,237,.22), 0 0 16px rgba(125,59,237,.14)}
  .evolve-lab-modes{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
  .evolve-lab-modes .pill{cursor:pointer;font-family:inherit;padding:2px 9px}
  .evolve-lab-modes .pill:hover{border-color:var(--border2);color:var(--text)}
  .evolve-lab-modes .pill.on{background:var(--accent-soft);color:var(--accent-text);border-color:transparent}
  .evolve-lab .mac-titlebar{flex-wrap:wrap}
```

2. 替换断点块。删除现有 `@media (max-width:1100px){ .evolve-layout... .evolve-model-grid... }` 与 `@media (max-width:900px){...}`（若 Task 4 未删净）以及旧 `@media (max-width:700px){...}` 中 evolve 部分，统一为：

```css
  @media (max-width:1100px){
    .evolve-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
    .evolve-model-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  }
  @media (max-width:700px){
    .evolve-kpis{grid-template-columns:1fr}
    .evolve-model-grid{grid-template-columns:1fr}
    .evolve-page-head .page-actions{flex:none;flex-wrap:wrap;justify-content:flex-end}
    .evolve-page-head .page-actions .mac-btn{padding:5px 11px}
    .evolve-page-head .page-actions .mac-btn svg{margin-right:4px}
    .evolve-curve-foot .ml-auto{margin-left:0!important}
    .evolve-lab .grid.grid-cols-2,.evolve-lab .grid.grid-cols-2.lg\\:grid-cols-4{grid-template-columns:1fr}
    .evolve-lab .col-span-2{grid-column:span 1}
  }
```

（旧 700px 块里仍有效的规则必须原样保留，不能丢。）

- [ ] **Step 7: 跑测试 + 门禁**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 全部 PASS

Run: `.venv\Scripts\python.exe scripts\check_js.py`
Expected: 通过

Run: `.venv\Scripts\python.exe scripts\build_frontend.py --check-only`
Expected: 通过（静态引用检查 WARN 不阻塞）

- [ ] **Step 8: Commit**

```powershell
git add web/static/index.html tests/test_evolve_ui_contract.py
git commit -m "feat(web): 持续进化模型卡部署旗舰高亮 + 实验台模式 pills + 断点重写 + 警告图标化"
```

---

### Task 6: 全量回归与验收

**Files:**
- Test: 全套回归（无新增代码，除非发现问题回修）

**Interfaces:**
- Consumes: Task 1-5 全部产物
- Produces: 可交付的完成态（所有验收红线通过）

- [ ] **Step 1: 持续进化静态契约全量**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_ui_contract.py -v`
Expected: 全 PASS

- [ ] **Step 2: 持续进化后端回归（证明零后端改动未破坏任何契约）**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evolve_api_guards.py tests/test_evolve_keepalive.py tests/test_evolve_anchor.py tests/test_evolve_audit.py tests/test_evolve_deployment.py tests/test_evolve_runtime_reload.py tests/test_drl_val_selection.py -q`
Expected: 全 PASS

- [ ] **Step 3: 前端构建门禁 + UI 契约全家桶**

Run: `.venv\Scripts\python.exe scripts\check_js.py && .venv\Scripts\python.exe scripts\build_frontend.py --check-only && .venv\Scripts\python.exe -m pytest tests/test_backtest_ui_contract.py tests/test_ai_settings_persist.py -q`
Expected: 全 PASS

- [ ] **Step 4: 后端 diff 红线自检**

Run: `git diff --stat HEAD~5 -- web/api/evolve.py drl/evolve_engine.py`
Expected: 空输出（两个后端文件零改动）

- [ ] **Step 5: 静态自查清单（grep 验证）**

```powershell
Select-String -Path web/static/index.html -Pattern "evolveKpiTiles" | Measure-Object | Select-Object -ExpandProperty Count   # ≥3（computed/模板/return）
Select-String -Path web/static/index.html -Pattern "evolvePanelOpen" | Measure-Object | Select-Object -ExpandProperty Count   # ≥4
```

- [ ] **Step 6: 视觉验收（启动 web 服务人工过一遍，1440 / 1000 / 700 三档）**

- [ ] KPI 8 磁贴等行高、无文案扭曲/换行（重点看「运行中」大字）；运行绿光、暂停琥珀光、锚点告警红光、无状态素色
- [ ] 页头按钮区右对齐、无任何 emoji、无重复大标题（顶栏已有「持续进化」）
- [ ] 「配置 / 触发」点击弹浮层，锚定按钮下方；标的池二级菜单可打开、点浮层外关闭、点二级菜单内不关闭
- [ ] 曲线统计行独立一行（不与分段按钮挤压）、图表 260px、红点/图例正常
- [ ] 模型三卡：部署旗舰卡紫描边；查看版本弹层、回退按钮可用
- [ ] 实验台 pills 点击切模式并展开；四模式表单完整
- [ ] 页面无横向滚动条；`#evolveCurveChart` 曲线随轮询刷新
- [ ] KPI 前三张点按切换曲线模型；锚点磁贴点按开浮层

- [ ] **Step 7: 交互回归点检**

- [ ] 暂停/恢复、启用、刷新状态按钮按既有禁用条件工作
- [ ] 浮层内「保存配置」toast 正常、二级菜单多选标的池正常
- [ ] 手动触发三键在引擎未运行时禁用、训练中显示「训练中…」
- [ ] 锚点告警存在时浮层显示锚点段与「重置锚点」；无告警时整段隐藏
- [ ] 切到别的视图再切回，轮询与数据仍在（status 5s / rounds 10s 未动）

- [ ] **Step 8: 如有问题回修并重跑 Step 1-3；全绿后收尾 commit**

```powershell
git add -A
git commit -m "test(web): 持续进化页重设计回归收尾——契约测试与视觉验收修正"
```
（若 Step 8 无文件变更则跳过 commit。）

- [ ] **Step 9: 回填 spec 细化**

把「页头不放大标题 / 部署旗舰高亮规则 / ⚠ 图标化」三条细化补进 `docs/superpowers/specs/2026-09-27-continuous-evolution-kpi-dashboard-design.md` §3.1/§3.4，并 commit：

```powershell
git add docs/superpowers/specs/2026-09-27-continuous-evolution-kpi-dashboard-design.md
git commit -m "docs: 回填持续进化 UI spec 三处实现细化"
```
