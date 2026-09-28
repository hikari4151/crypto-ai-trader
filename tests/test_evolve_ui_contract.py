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
    # 约束 4：文件内不落 emoji 字面量——用码点表构造等价的禁用字符表
    blocked = "".join(map(chr, (0x1F4C8, 0x1F9EA, 0x1F9E0, 0x1F916, 0x1F9EC, 0x23F8, 0x26A1, 0x231A, 0x1F680, 0x1F3AF, 0x1F527)))
    assert not any(ch in v for ch in blocked)


def test_kpi_glow_css_present():
    assert ".evolve-kpi.on-green{border-color:rgba(13,204,130,.45)" in HTML
    assert ".evolve-kpi.on-amber" in HTML
    assert ".evolve-kpi.on-red" in HTML


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


def test_panel_pop_anchored_to_button():
    # spec §3.6：浮层必须按页头按钮位置锚定（popPos + 右对齐覆盖），不能裸 fixed
    assert "evolvePanelStyle" in _setup_return()
    assert ':style="evolvePanelStyle"' in _evolve_view()


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


def test_section_label_styled():
    # Task 4 执行期发现：计划定义了 .lab 类却没给样式，本任务补录
    assert ".evolve-page .lab::after" in HTML


def test_doc_click_closes_panel_full_selector():
    # 三类内部选择器必须完整，防回归时被掏空
    start = HTML.find("onDocClick = (e)")
    end = HTML.find("document.addEventListener('click', onDocClick)")
    body = HTML[start:end]
    assert "'.evolve-panel, .evolve-panel-pop, .evolve-symbols-pop'" in body


def test_final_review_fixes_present():
    assert ".evolve-panel-btn{border-color:rgba(125,59,237,.5)" in HTML
    assert "evolve-pop-anchor" in HTML
    assert "toggleEvolvePanel(true, 'anchor')" in HTML
    assert "_evolvePanelOffEsc" in HTML and "'Escape'" in HTML
    assert ".evolve-kpi-dot.purple" in HTML
    assert ".evolve-kpi.clickable" in HTML


def test_curve_canvas_texture():
    # 用户验收反馈：图表画布按方案稿 v4 补紫渐变底+虚线描边；事件点保留底部双轨
    assert "linear-gradient(180deg,rgba(125,59,237,.05)" in HTML
    assert "border:1px dashed var(--border);border-radius:var(--r-md)" in HTML


def test_challenger_dual_grid_split():
    # 用户验收裁定（方案 A）：挑战者曲线拆到独立 grid + 独立 y 轴，缩放共享
    start = HTML.find("function renderEvolveCurve")
    body = HTML[start:HTML.find("function setEvolveRange", start)]
    assert "gridIndex:1" in body
    assert "yAxisIndex:2" in body
    assert "const zoomX = chalLineCount ? [0,1] : 0" in body
    assert "xAxisIndex: zoomX" in body
    assert "bottom:88" in body and "height:44" in body
