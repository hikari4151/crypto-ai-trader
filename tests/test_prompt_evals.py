"""提示词工程优化（P1-P3）测试：评测集机制 / evidence 反幻觉 / JSON Schema / 上下文精简。"""
import pytest

from ai.evals import GOLDEN_SET, evaluate_prompt
from ai.validator import ValidationError, validate_ai_output


# ============ P3. 提示词回归评测集 ============

def test_golden_set_deterministic():
    """golden set 确定性构造：案例数固定、每案例含三种客观基准。"""
    assert len(GOLDEN_SET) >= 8
    for case in GOLDEN_SET:
        assert "snap" in case and "program_direction" in case
        assert "actual_direction" in case and case["actual_direction"] in ("long", "short", "neutral")
        assert case["snap"]["indicators"]["close"] > 0


_PROG: dict = {}


def test_evaluate_prompt_metrics_computed():
    """canned 输出也能跑通指标闭环：校验通过率=1、方向一致率=1（跟随程序方向时）。"""
    # 用 golden 案例的 program_direction 作为 produce 的 bias
    for case in GOLDEN_SET:
        _PROG[id(case["snap"])] = case["program_direction"]

    def produce(snap):
        bias = _PROG.get(id(snap), "neutral")
        if bias in ("long", "short"):
            return {"regime": "趋势", "bias": bias, "confidence": 0.65,
                    "summary": "测试输出", "signals": [], "evidence": []}
        return {"regime": "震荡", "bias": "neutral", "confidence": 0.4,
                "summary": "测试输出", "signals": [], "evidence": []}

    m = evaluate_prompt(produce)
    assert set(m) == {"cases", "validator_pass_rate", "direction_agreement_rate",
                      "directional_accuracy"}
    assert m["cases"] == len(GOLDEN_SET)
    assert m["validator_pass_rate"] == 1.0, "测试输出应全部通过校验"
    assert m["direction_agreement_rate"] >= 0.9, "跟随程序方向时一致率应接近 1"
    assert 0.0 <= m["directional_accuracy"] <= 1.0


def test_evaluate_prompt_rejects_invalid():
    """produce 输出不合规（缺 summary）→ 校验通过率 <1。"""
    def bad_produce(snap):
        return {"regime": "震荡", "bias": "neutral", "confidence": 0.4}  # 缺 summary

    m = evaluate_prompt(bad_produce)
    assert m["validator_pass_rate"] == 0.0


# ============ P2. evidence 结构化反幻觉 ============

def _snap_with(indicators: dict) -> dict:
    return {"indicators": indicators}


def test_evidence_match_passes():
    """evidence 引用快照指标且数值一致 → 通过。"""
    out = {"regime": "震荡", "bias": "neutral", "confidence": 0.4, "summary": "x",
           "evidence": [{"name": "rsi", "value": 55.0}]}
    validate_ai_output("market_analysis", out, {"snap": _snap_with({"rsi": 55.0, "close": 50000.0})})


def test_evidence_mismatch_rejected():
    """evidence 引用快照指标但数值编造 → 拒绝（疑似幻觉）。"""
    out = {"regime": "震荡", "bias": "neutral", "confidence": 0.4, "summary": "x",
           "evidence": [{"name": "rsi", "value": 99.0}]}
    with pytest.raises(ValidationError) as ei:
        validate_ai_output("market_analysis", out, {"snap": _snap_with({"rsi": 55.0, "close": 50000.0})})
    assert any("疑似编造" in e.get("reason", "") for e in ei.value.errors)


def test_evidence_non_numeric_rejected():
    """evidence 的 value 非有限数值 → 拒绝。"""
    out = {"regime": "震荡", "bias": "neutral", "confidence": 0.4, "summary": "x",
           "evidence": [{"name": "rsi", "value": float("nan")}]}
    with pytest.raises(ValidationError):
        validate_ai_output("market_analysis", out, {"snap": _snap_with({"rsi": 55.0})})


# ============ P1. JSON Schema 结构化输出 ============

def test_schemas_defined_for_features():
    from ai.schemas import FEATURE_SCHEMAS, schema_for
    for f in ("market_analysis", "param_optimize", "strategy_design",
              "strategy_iterate", "trade_review", "factor_mine"):
        assert schema_for(f) is not None, f"特征 {f} 应有 JSON Schema"
        assert FEATURE_SCHEMAS[f]["type"] == "object"
        assert FEATURE_SCHEMAS[f]["additionalProperties"] is False


def test_json_schema_capability_gate():
    from ai.client import _supports_json_schema
    assert _supports_json_schema("gpt-4o") is True
    assert _supports_json_schema("gpt-4o-mini") is True
    assert _supports_json_schema("deepseek-chat") is False      # DeepSeek 不支持
    assert _supports_json_schema("deepseek-reasoner") is False


# ============ P5. 上下文精简辅助函数 ============

def test_trades_block_capped_at_20():
    from ai.prompts import _trades_block
    trades = [{"ts": "2026-01-01T00:00:00+00:00", "side": "buy", "price": 100.0,
               "qty": 0.1, "pnl": 0.0, "strategy": "dual_ma", "reason": "r"}] * 30
    block = _trades_block(trades)
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    assert len(lines) == 20, "近期交易应只保留最近 20 笔"
    assert "暂无" not in block


def test_backtests_block_capped_at_5():
    from ai.prompts import _backtests_block
    bts = [{"strategy": "dual_ma", "symbol": "BTC/USDT", "timeframe": "1h"} for _ in range(8)]
    block = _backtests_block(bts)
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    assert len(lines) == 5, "回测历史应只保留最近 5 条"


def test_fewshot_examples_cover_features():
    from ai.client import _FEWSHOT_EXAMPLES, AI_FEATURE_PARAMS
    assert set(_FEWSHOT_EXAMPLES) == set(AI_FEATURE_PARAMS), "few-shot 应覆盖全部功能"