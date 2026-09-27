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
