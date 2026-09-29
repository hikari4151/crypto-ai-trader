"""回测页面关键 UI 契约。"""

from pathlib import Path


HTML = Path("web/static/index.html").read_text(encoding="utf-8")


def test_backtest_timeframe_choices_include_five_minutes():
    assert "['5m','15m','30m','1h','4h','1d']" in HTML


def test_context_bar_is_transparent_and_menu_bar_is_untouched():
    ctx_start = HTML.rfind(".ctxbar{")
    ctx_rule = HTML[ctx_start:HTML.find("}", ctx_start) + 1]
    assert "background:transparent" in ctx_rule
    assert "border:0" in ctx_rule
    assert "backdrop-filter:none" in ctx_rule
    assert ".ctx-chip{background:color-mix" in HTML
    # 2026-09-23 并行 UI 重构（1292b5a）：menubar 改为毛玻璃自适应
    # （backdrop-filter blur + color-mix 背景），不再有固定 height:46px。
    # 契约更新为断言新设计特征——menubar 仍是"顶部栏"（毛玻璃分隔条），
    # 而非误回归成无样式旧版。
    assert ".menubar{backdrop-filter:blur" in HTML
    assert "background:color-mix" in HTML
    assert "var(--glass-edge)" in HTML


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
