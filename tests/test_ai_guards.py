"""验证三套 AI 守卫：方向一致性 / 交易者方程 / 决策连续性。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
from datetime import datetime, timezone, timedelta

from ai.context_engine import validate_trade_plan, build_continuity_context, parse_prev_analysis
from ai.validator import validate_ai_output, ValidationError

passed = []


def check(name, fn):
    try:
        fn()
        passed.append(f"PASS {name}")
    except AssertionError as e:
        passed.append(f"FAIL {name}: {e}")


def base_snap(program_dir="long", program_score=4):
    return {"indicators": {"close": 50000, "ma_fast": 49800, "ma_slow": 49500},
            "program_direction": program_dir, "program_score": program_score}


# ============ 1. 交易者方程 ============
def test_tp_good():
    errs = validate_trade_plan(
        {"side": "long", "entry": 50000, "stop_loss": 49500, "take_profit": 51000, "win_rate": 0.55},
        "long", 50000)
    assert errs == [], f"合格方案不应报错: {errs}"


def test_tp_rr_too_low():
    # reward=100, risk=1000 → RR=0.1 < 1.0
    errs = validate_trade_plan(
        {"side": "long", "entry": 50000, "stop_loss": 49000, "take_profit": 50100, "win_rate": 0.9},
        "long", 50000)
    assert any("RR" in e for e in errs), f"应报 RR 不足: {errs}"


def test_tp_equation_fails():
    # 胜率0.4×reward300 - 0.6×risk500 = 120-300 = -180 < 0
    errs = validate_trade_plan(
        {"side": "long", "entry": 50000, "stop_loss": 49500, "take_profit": 50300, "win_rate": 0.4},
        "long", 50000)
    assert any("交易者方程" in e for e in errs), f"应报方程不通过: {errs}"


def test_tp_geometry_wrong():
    errs = validate_trade_plan(
        {"side": "long", "entry": 50000, "stop_loss": 51000, "take_profit": 49500, "win_rate": 0.6},
        "long", 50000)
    assert any("几何" in e for e in errs), f"应报几何错误: {errs}"


def test_tp_side_conflict():
    errs = validate_trade_plan(
        {"side": "short", "entry": 50000, "stop_loss": 50500, "take_profit": 49000, "win_rate": 0.6},
        "long", 50000)
    assert any("冲突" in e for e in errs), f"应报方向冲突: {errs}"


def test_tp_hallucination_price():
    errs = validate_trade_plan(
        {"side": "long", "entry": 50000, "stop_loss": 49500, "take_profit": 9999999, "win_rate": 0.6},
        "long", 50000)
    assert any("量级" in e for e in errs), f"应报价格量级异常: {errs}"


# ============ 2. 决策连续性 ============
def test_continuity_flip_conflict():
    prev_ts = datetime.now(timezone.utc) - timedelta(hours=1)
    ctx = build_continuity_context({"bias": "long", "ts": prev_ts.isoformat()}, "short", "BTC/USDT")
    assert ctx["flipped"] and ctx["flip_conflict"], f"应判定短时反手: {ctx}"


def test_continuity_flip_ok():
    prev_ts = datetime.now(timezone.utc) - timedelta(hours=10)
    ctx = build_continuity_context({"bias": "long", "ts": prev_ts.isoformat()}, "short", "BTC/USDT")
    assert ctx["flipped"] and not ctx["flip_conflict"], f"非短时反手不应触发冷却: {ctx}"


def test_parse_prev():
    d = parse_prev_analysis('{"bias": "long", "confidence": 0.7, "regime": "趋势"}')
    assert d and d["bias"] == "long", f"解析失败: {d}"


# ============ 3. 校验器集成（方向一致性 + 方程 + 连续性） ============
def test_validator_direction_conflict():
    data = {"regime": "趋势", "bias": "short", "confidence": 0.7,
            "summary": "下跌趋势", "warnings": [], "signals": ["反弹做空"]}
    try:
        validate_ai_output("market_analysis", data, {"snap": base_snap("long", 4)})
        raise AssertionError("方向矛盾应被拒绝")
    except ValidationError as e:
        assert any("矛盾" in x["reason"] for x in e.errors), f"应报方向矛盾: {e.errors}"


def test_validator_direction_weak_ok():
    # 程序方向弱确认（score=1）时不拦截
    data = {"regime": "震荡", "bias": "neutral", "confidence": 0.5,
            "summary": "震荡", "warnings": [], "signals": []}
    validate_ai_output("market_analysis", data, {"snap": base_snap("long", 1)})  # 不抛异常


def test_validator_trade_plan_bad():
    data = {"regime": "趋势", "bias": "long", "confidence": 0.7,
            "summary": "上行", "warnings": [], "signals": ["突破做多"],
            "trade_plan": {"side": "long", "entry": 50000, "stop_loss": 49500,
                           "take_profit": 50100, "win_rate": 0.5}}
    try:
        validate_ai_output("market_analysis", data, {"snap": base_snap("long", 4)})
        raise AssertionError("RR不足方案应被拒绝")
    except ValidationError as e:
        assert any("trade_plan" in x["field"] for x in e.errors), f"应报 trade_plan 错误: {e.errors}"


def test_validator_trade_plan_good():
    data = {"regime": "趋势", "bias": "long", "confidence": 0.7,
            "summary": "上行", "warnings": [], "signals": ["突破做多"],
            "trade_plan": {"side": "long", "entry": 50000, "stop_loss": 49500,
                           "take_profit": 51000, "win_rate": 0.55}}
    validate_ai_output("market_analysis", data, {"snap": base_snap("long", 4)})  # 通过


def test_validator_continuity_cap():
    prev_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    prev = {"bias": "long", "ts": prev_ts, "confidence": 0.7}
    data = {"regime": "拐点", "bias": "short", "confidence": 0.75,
            "summary": "反转向下", "warnings": [], "signals": ["做空"]}
    try:
        validate_ai_output("market_analysis", data, {"snap": base_snap("short", 4), "continuity": prev})
        raise AssertionError("短时反手高置信应被拒绝")
    except ValidationError as e:
        assert any("反手" in x["reason"] for x in e.errors), f"应报反手: {e.errors}"


# 运行
check("合格 trade_plan", test_tp_good)
check("RR<1.0 拒绝", test_tp_rr_too_low)
check("交易者方程不通过拒绝", test_tp_equation_fails)
check("几何错误拒绝", test_tp_geometry_wrong)
check("方向冲突拒绝", test_tp_side_conflict)
check("价格量级异常拒绝", test_tp_hallucination_price)
check("短时反手触发冷却", test_continuity_flip_conflict)
check("非短时反手不冷却", test_continuity_flip_ok)
check("上轮解析", test_parse_prev)
check("校验器-方向矛盾拒绝", test_validator_direction_conflict)
check("校验器-弱方向不拦截", test_validator_direction_weak_ok)
check("校验器-坏 trade_plan 拒绝", test_validator_trade_plan_bad)
check("校验器-好 trade_plan 通过", test_validator_trade_plan_good)
check("校验器-短时反手置信度上限", test_validator_continuity_cap)

print("\n".join(passed))
fails = [p for p in passed if p.startswith("FAIL")]
print(f"\n共 {len(passed)} 项，失败 {len(fails)} 项")
sys.exit(1 if fails else 0)
