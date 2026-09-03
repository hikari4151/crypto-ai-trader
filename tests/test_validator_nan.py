"""AI 输出校验 NaN/Infinity 过滤回归测试（模块 C P1-7）。

覆盖 2026-08-14 修复：json.loads 接受 NaN/Infinity 字面量 → 非有限值
必须在校验层被拦截，不再穿透 min/max 范围比较（NaN 比较恒 False 的漏洞，
与 engine/risk.py 已修的非有限值问题同类）。
"""
import math

import pytest

from ai.validator import ValidationError, _is_number, _within_range, validate_ai_output


# ============ _is_number ============

def test_is_number_accepts_finite():
    assert _is_number(1.5)
    assert _is_number(3)
    assert _is_number(0)
    assert _is_number(-0.25)


def test_is_number_rejects_non_finite():
    assert not _is_number(float("nan"))
    assert not _is_number(float("inf"))
    assert not _is_number(float("-inf"))
    assert not _is_number(math.nan)
    assert not _is_number(math.inf)
    assert not _is_number(-math.inf)


def test_is_number_rejects_bool():
    # bool 是 int 子类：排除逻辑必须保持不变
    assert not _is_number(True)
    assert not _is_number(False)


def test_is_number_rejects_non_numeric():
    assert not _is_number("1.5")
    assert not _is_number(None)
    assert not _is_number([1])
    assert not _is_number({"a": 1})


# ============ _within_range ============

def test_within_range_rejects_non_finite_float():
    spec = {"type": "float", "min": 0.0, "max": 1.0}
    assert _within_range(float("nan"), spec) is not None
    assert _within_range(float("inf"), spec) is not None
    assert _within_range(float("-inf"), spec) is not None


def test_within_range_rejects_non_finite_int():
    spec = {"type": "int", "min": 1, "max": 10}
    assert _within_range(float("nan"), spec) is not None
    assert _within_range(float("inf"), spec) is not None


def test_within_range_finite_normal_path_unchanged():
    spec = {"type": "float", "min": 0.0, "max": 1.0}
    assert _within_range(0.5, spec) is None
    assert _within_range(1.0, spec) is None
    assert _within_range(0.0, spec) is None
    assert _within_range(1.5, spec) == "大于最大值 1.0"
    assert _within_range(-0.1, spec) == "小于最小值 0.0"


def test_within_range_str_bool_branches_unchanged():
    assert _within_range("BTC", {"type": "str", "choices": ["BTC", "ETH"]}) is None
    assert _within_range("XRP", {"type": "str", "choices": ["BTC", "ETH"]}) is not None
    assert _within_range(True, {"type": "bool"}) is None
    assert _within_range(1, {"type": "bool"}) is not None
    assert _within_range(float("nan"), {"type": "bool"}) is not None  # 非 bool 一律拒绝


# ============ validate_ai_output 集成 ============

def test_strategy_design_nan_params_rejected():
    """AI 输出含 NaN/Inf 参数 → ValidationError 且 errors 含对应字段（P1-7 主场景）。"""
    schema = {
        "stop_loss_pct": {"type": "float", "min": 0.0, "max": 0.1},
        "rsi_ob": {"type": "int", "min": 50, "max": 95},
    }
    data = {
        "name": "test_strategy", "title": "t", "description": "d",
        "logic": "l", "params": {"stop_loss_pct": float("nan"), "rsi_ob": float("inf")},
    }
    with pytest.raises(ValidationError) as ei:
        validate_ai_output("strategy_design", data, {"param_schema": schema})
    fields = {e["field"] for e in ei.value.errors}
    assert "params.stop_loss_pct" in fields
    assert "params.rsi_ob" in fields


def test_strategy_design_finite_params_still_pass():
    """正常有限值参数路径与修复前一致。"""
    schema = {"stop_loss_pct": {"type": "float", "min": 0.0, "max": 0.1}}
    data = {
        "name": "test_strategy", "title": "t", "description": "d",
        "logic": "l", "params": {"stop_loss_pct": 0.05},
    }
    # 不应抛异常
    validate_ai_output("strategy_design", data, {"param_schema": schema})


def test_market_analysis_nan_confidence_rejected():
    """confidence=NaN 必须报错（NaN 比较恒 False 曾使 0-1 范围检查放行前的语义模糊）。"""
    data = {
        "regime": "趋势", "bias": "long", "confidence": float("nan"),
        "summary": "s",
    }
    with pytest.raises(ValidationError) as ei:
        validate_ai_output("market_analysis", data, {"snap": {}})
    fields = {e["field"] for e in ei.value.errors}
    assert "confidence" in fields


def test_market_analysis_nan_outlook_probability_rejected():
    """outlook_24h.probability=NaN 必须报错。"""
    data = {
        "regime": "趋势", "bias": "long", "confidence": 0.8,
        "summary": "s",
        "outlook_24h": {"scenario": "s", "probability": float("nan")},
    }
    with pytest.raises(ValidationError) as ei:
        validate_ai_output("market_analysis", data, {"snap": {}})
    fields = {e["field"] for e in ei.value.errors}
    assert "outlook_24h" in fields


def test_market_analysis_normal_path_unchanged():
    """正常 market_analysis 输出仍通过（回归红线）。"""
    data = {
        "regime": "趋势", "bias": "long", "confidence": 0.8,
        "summary": "s", "signals": ["买入"],
    }
    validate_ai_output("market_analysis", data, {"snap": {}})
