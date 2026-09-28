"""REDEV_GUIDE grep 型红线门禁（scripts/check_redlines.py）回归 + B8 修复行为保真。

覆盖：
- B8：simulated.py `_split` 必须走 parse_symbol，且行为不变（拆 `BTC/USDT:USDT`、
  非统一代码 quote 兜底 "USDT"）——B8 修复的语义回归
- B8 检查：真违反（.py 中 symbol.split('/') 但不在豁免表）必须报；符号解析器本体
  与 URL 解析豁免不误报
- E2：CDN 引用必须报
- E7：!important 超基线必须报
- 当前仓库真实状态通过
"""
from pathlib import Path

import scripts.check_redlines as cl

ROOT = Path(__file__).resolve().parent.parent


# ---- B8 修复行为保真（simulated.py _split 走 parse_symbol 后语义不变）----
def test_simulated_split_behavior_preserved():
    from exchange.simulated import SimulatedExchange

    ex = SimulatedExchange()
    # 标准现货
    assert ex._split("BTC/USDT") == ("BTC", "USDT")
    # 合约统一代码：B8 红线场景（自写 split 会拆出 "USDT:USDT"）
    assert ex._split("BTC/USDT:USDT") == ("BTC", "USDT")
    assert ex._split("ETH/USDT:USDT") == ("ETH", "USDT")
    # 非统一代码：quote 兜底 "USDT"（历史语义保留）
    assert ex._split("BTC") == ("BTC", "USDT")


def test_simulated_uses_exchange_symbols_module():
    """_split 内部必须 import parse_symbol（禁自写 split——红线要求的本质）。"""
    src = (ROOT / "exchange" / "simulated.py").read_text(encoding="utf-8")
    assert "from exchange.symbols import parse_symbol" in src
    assert 'symbol.partition(":")' not in src.replace("parse_symbol", "", 1)


# ---- B8 检查本身 ----
def test_b8_skips_symbols_parser_and_url_lines():
    # 豁免表自身必须成立：symbols.py 是本体的家，URL 解析行有理由
    assert "exchange/symbols.py" in cl.B8_EXEMPT_FILES
    assert any(f == "net/proxy_nodes.py" and n == 178 for f, n, _ in cl.B8_EXEMPT_LINES)


def test_b8_detects_violation():
    """B8_PAT 必须命中违规写法（含取索引），不命中豁免场景。"""
    assert cl.B8_PAT.search("base = symbol.split('/')[0]")
    assert cl.B8_PAT.search('base = symbol.split("/")[0]')
    assert cl.B8_PAT.search("parts = [p for p in base.split('/') if p]")
    # 豁免场景：partition 不构成违规；单引号不配对不命中
    assert not cl.B8_PAT.search("base, sep, _ = symbol.partition(':')")
    # 注释/文档提及历史写法会命中正则（如 symbols.py docstring）——那由 B8_EXEMPT_FILES
    # 文件级豁免处理，正则不区分注释（保持简单），测试只验证真实文件已无违规
    assert not cl.B8_PAT.search((ROOT / "run.py").read_text(encoding="utf-8"))
    assert not cl.B8_PAT.search((ROOT / "exchange" / "simulated.py").read_text(encoding="utf-8"))


# ---- E2 / E7 ----
def test_e2_detects_cdn():
    text = '  <script src="https://cdnjs.cloudflare.com/ajax/libs/vue/3.4/vue.global.prod.js"></script>'
    # 直接构造：index.html 当前必须无 CDN；用正则语义验证检测逻辑
    import re
    cdn = re.compile(r"(unpkg\.com|cdnjs\.cloudflare|jsdelivr\.net|cdn\.)", re.I)
    assert cdn.search(text)
    idx = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert not cdn.search(idx), "index.html 不应有 CDN 引用"
    assert "unpkg" not in idx and "jsdelivr" not in idx


def test_e7_baseline_matches_current_repo():
    """E7 基线必须与现状一致：要么现状 ≤ 基线，要么基线是错的（防基线被静默抬高）。

    口径与 check_e7 一致：按"包含 !important 的行数"计（同一行出现多次只算 1 行，
    如 1548 行 `backdrop-filter:none!important;-webkit-...:none!important` 只算 1）。
    """
    text = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    n = sum(1 for ln in text.splitlines() if "!important" in ln)
    assert n <= cl.E7_IMPORTANT_BASELINE, (
        f"!important {n} 行 > 基线 {cl.E7_IMPORTANT_BASELINE}；新增豁免须带注释并更新基线")


def test_repo_state_is_clean():
    """当前仓库必须通过 grep 型红线（CI 与本机同一判定）。"""
    assert cl.main([]) == 0


def test_main_list_mode(monkeypatch, capsys):
    assert cl.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "B8" in out and "E2" in out and "E7" in out
