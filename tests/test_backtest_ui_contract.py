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
    assert ".menubar{height:46px" in HTML
