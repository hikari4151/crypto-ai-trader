"""TDD：GAP-01 完整验收判定（scripts/gap01_verify.py 的核心纯逻辑）。

判据（docs/LIVE_READINESS.md §3 GAP-01，单标的豁免跨品种门的完整版用于 grid 基线验证）：
1. 每标的 detect_overfit verdict == "通过"（"疑似过拟合"/"严重过拟合"→FAIL，"无法判定"→INCONCLUSIVE）；
2. OOS ret > 0 且 OOS 交易数 ≥ min_trades（不足 → INCONCLUSIVE）；
3. DSR ≥ 0.95（None → INCONCLUSIVE）；PBO ≤ 0.5（None 忽略，>0.5 → FAIL）；
4. ≥2 个标的 OOS ret > 0（方向一致）；
5. 成本鲁棒：cost_scan.sensitivity.worst_ret > 0（5× 成本仍正期望）。

合并规则：任一 FAIL → FAIL；否则任一 INCONCLUSIVE → INCONCLUSIVE；否则 PASS。
"""

from backtest.gap01_assess import assess_gap01


def _sym(symbol="BTC/USDT", verdict="通过", oos_ret=0.01, trades=40,
         dsr=0.97, pbo=0.3):
    return {"symbol": symbol, "verdict": verdict, "oos_ret": oos_ret,
            "oos_trades": trades, "dsr": dsr, "pbo": pbo, "stability": 0.8}


def _cost(worst_ret=0.01, base_ret=0.02):
    return {"sensitivity": {"worst_ret": worst_ret, "base_ret": base_ret,
                            "worst_ret_drop_pct": 50.0},
            "base": {"metrics": {"total_return": base_ret}}}


def test_gap01_pass_all_criteria():
    syms = [_sym("BTC/USDT"), _sym("ETH/USDT"), _sym("SOL/USDT")]
    out = assess_gap01(syms, _cost())
    assert out["verdict"] == "PASS"


def test_gap01_fail_on_severe_overfit():
    syms = [_sym("BTC/USDT"), _sym("ETH/USDT", verdict="严重过拟合"),
            _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost())["verdict"] == "FAIL"


def test_gap01_fail_on_suspected_overfit():
    """P0 实测 grid 正是『疑似过拟合』——必须判 FAIL，不能放行。"""
    syms = [_sym("BTC/USDT", verdict="疑似过拟合"), _sym("ETH/USDT"),
            _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost())["verdict"] == "FAIL"


def test_gap01_fail_on_negative_oos():
    syms = [_sym("BTC/USDT"), _sym("ETH/USDT", oos_ret=-0.01),
            _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost())["verdict"] == "FAIL"


def test_gap01_inconclusive_when_trades_insufficient():
    syms = [_sym("BTC/USDT", trades=12), _sym("ETH/USDT"), _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost())["verdict"] == "INCONCLUSIVE"


def test_gap01_inconclusive_when_dsr_missing():
    syms = [_sym("BTC/USDT", dsr=None), _sym("ETH/USDT"), _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost())["verdict"] == "INCONCLUSIVE"


def test_gap01_fail_when_cost_not_robust():
    syms = [_sym("BTC/USDT"), _sym("ETH/USDT"), _sym("SOL/USDT")]
    assert assess_gap01(syms, _cost(worst_ret=-0.005))["verdict"] == "FAIL"


def test_gap01_fail_when_less_than_two_symbols_positive():
    syms = [_sym("BTC/USDT"), _sym("ETH/USDT", oos_ret=-0.01),
            _sym("SOL/USDT", oos_ret=-0.02)]
    assert assess_gap01(syms, _cost())["verdict"] == "FAIL"