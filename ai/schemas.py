"""各 AI 功能的 JSON Schema（P1 结构化输出）。

把 `ai/prompts.py` 里"自然语言描述的输出 schema"固化为机器可读的 JSON Schema，
配合 `ai/client.py` 在支持 json_schema 的模型上以 `response_format=json_schema`
下发，从根源约束字段集合与类型，显著降低解析失败/字段缺失/类型错误。

注意：
- `additionalProperties: false` 约束 AI 不得输出 schema 之外的字段。
- `required` 只列**必填**核心字段；可选字段出现在 properties 但不在 required
  （保持与现有"可选增强字段"语义一致，兼容旧模型输出）。
- 本模块只做 schema 定义；是否启用由 `settings.ai_use_json_schema` + 模型
  能力 + 400 自动回退三重控制（见 client.py），默认关闭，不破坏既有流程。
"""

MARKET_ANALYSIS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regime": {"type": "string", "enum": ["趋势", "震荡", "拐点", "未知"]},
        "bias": {"type": "string", "enum": ["long", "short", "neutral"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "signals": {"type": "array", "items": {"type": "string"}},
        "trade_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "side": {"type": "string", "enum": ["long", "short"]},
                "entry": {"type": "number"},
                "stop_loss": {"type": "number"},
                "take_profit": {"type": "number"},
                "win_rate": {"type": "number"},
            },
            "required": ["side", "entry", "stop_loss", "take_profit", "win_rate"],
        },
        "bull_strength": {"type": "number"},
        "bear_strength": {"type": "number"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "outlook_24h": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scenario": {"type": "string"},
                "probability": {"type": "number"},
                "trigger": {"type": "string"},
            },
            "required": ["scenario"],
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["name", "value"],
            },
        },
    },
    "required": ["regime", "bias", "confidence", "summary"],
}

PARAM_OPTIMIZE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "params": {"type": "object"},
        "reason": {"type": "string"},
        "focus": {"type": "string"},
    },
    "required": ["params", "reason", "focus"],
}

_STRATEGY_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "logic": {"type": "string"},
        "params": {"type": "object"},
        "risk_tips": {"type": "array", "items": {"type": "string"}},
        "pine_code": {"type": "string"},
    },
    "required": ["name", "title", "description", "logic", "params", "risk_tips"],
}

STRATEGY_DESIGN = dict(_STRATEGY_BASE)

STRATEGY_ITERATE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "logic": {"type": "string"},
        "params": {"type": "object"},
        "risk_tips": {"type": "array", "items": {"type": "string"}},
        "critique": {"type": "string"},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["name", "title", "description", "logic", "params",
                 "risk_tips", "critique", "improvements", "summary"],
}

TRADE_REVIEW = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "problems": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["score", "strengths", "problems", "action_items", "summary"],
}

FACTOR_MINE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "category": {"type": "string"},
                    "expression": {"type": "string"},
                    "logic": {"type": "string"},
                },
                "required": ["name", "title", "category", "expression", "logic"],
            },
        }
    },
    "required": ["factors"],
}

FEATURE_SCHEMAS: dict[str, dict] = {
    "market_analysis": MARKET_ANALYSIS,
    "param_optimize": PARAM_OPTIMIZE,
    "strategy_design": STRATEGY_DESIGN,
    "strategy_iterate": STRATEGY_ITERATE,
    "trade_review": TRADE_REVIEW,
    "factor_mine": FACTOR_MINE,
}


def schema_for(feature: str) -> dict | None:
    """返回某功能的 JSON Schema；无则 None。"""
    return FEATURE_SCHEMAS.get(feature)
