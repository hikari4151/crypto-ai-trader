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
