"""第十六轮补洞：check_js 的 Vue 模板表达式检查契约测试。

背景：并行改动引入 `:class="{fr-hidden:!drlCfg.curriculum}"`——对象 key 带连字符
未加引号，Vue 编译抛 `Unexpected token '-'`，整页白屏（前端致命）。<script> 块
检查看不到 HTML 属性里的表达式。本测试锁定 check_js 必须包含 Vue 模板表达式检查，
且能拦截这类"未加引号的带连字符 key"。
"""
import scripts.check_js as check_js

BAD_TEMPLATE = """<html><body>
<div id="app" v-cloak>
  <div class="mac-field-row" :class="{fr-hidden:!drlCfg.curriculum}">x</div>
  <span>{{liveStatus.running ? 'on' : 'off'}}</span>
</div>
</body></html>
"""

GOOD_TEMPLATE = """<html><body>
<div id="app" v-cloak>
  <div class="mac-field-row" :class="{'fr-hidden':!drlCfg.curriculum}">x</div>
  <span>{{liveStatus.running ? 'on' : 'off'}}</span>
</div>
</body></html>
"""


def test_vue_expr_check_catches_unquoted_css_key():
    """未加引号的带连字符对象 key（{fr-hidden:...}）必须被拦截。"""
    errors = check_js.check_vue_template_exprs("index.html", BAD_TEMPLATE)
    assert errors, "check_js 未拦截 {fr-hidden:...} 语法错误（致命白屏 bug）"
    assert any("fr-hidden" in e for e in errors), f"报错应指明表达式: {errors}"


def test_vue_expr_check_passes_quoted_key():
    """加引号后的 {'fr-hidden':...} 必须通过。"""
    errors = check_js.check_vue_template_exprs("index.html", GOOD_TEMPLATE)
    assert errors == [], f"合法表达式不应报错: {errors}"


def test_vue_expr_check_skips_setup_var_refs():
    """引用 setup 作用域变量（liveStatus）不是语法错误，不应误报。"""
    errors = check_js.check_vue_template_exprs("index.html", GOOD_TEMPLATE)
    assert not any("liveStatus" in e for e in errors)


def test_check_file_runs_vue_expr_check():
    """check_file 对 HTML 必须触发 Vue 模板表达式检查（不只 <script> 块）。"""
    import inspect
    src = inspect.getsource(check_js.check_file)
    assert "check_vue_template_exprs" in src, "check_file 必须调用 Vue 模板表达式检查"
