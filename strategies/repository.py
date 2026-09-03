"""策略仓库：内置策略模板库。

每个模板包含：
- key: 唯一标识
- name: 策略名
- title: 中文标题
- category: 策略类别（趋势/震荡/突破/均值回归/动量）
- description: 一句话描述
- logic: 逻辑说明
- suitable: 适用市场条件
- risk_tips: 风控提示
- default_params: 默认参数
- param_schema: 参数 schema（与 price_action 一致的执行器约束）
"""
from typing import Any

from .price_action import PriceActionStrategy

# price_action 执行器的参数约束（复用该执行器的仓库模板用）。
# 直接引用执行器类：曾在此手抄一份，抄本与真实执行器漂移（多出三个不被 on_candle
# 读取的死参数），AI 优化会把这些无效旋钮当可调项。
_PRICE_ACTION_SCHEMA: dict[str, dict[str, Any]] = PriceActionStrategy.param_schema

CATEGORIES = [
    {"key": "trend", "label": "趋势跟踪", "icon": "📈"},
    {"key": "breakout", "label": "突破交易", "icon": "⚡"},
    {"key": "mean_reversion", "label": "均值回归", "icon": "🎯"},
    {"key": "grid", "label": "网格震荡", "icon": "🔲"},
]

REPOSITORY: list[dict[str, Any]] = [
    {
        "key": "dual_ma",
        "name": "dual_ma",
        "title": "双均线趋势",
        "category": "trend",
        "description": "快慢双均线金叉死叉顺势交易，趋势市最经典的基础策略。",
        "logic": "快线(MA10)上穿慢线(MA30)做多，下穿做空；顺势持仓，止损止盈控制风险。",
        "suitable": "明显的上升/下降趋势行情",
        "risk_tips": ["震荡市会频繁假信号", "趋势反转时回撤较大"],
        "default_params": {"fast_period": 10, "slow_period": 30, "size_pct": 0.5, "stop_loss_pct": 0.03, "take_profit_pct": 0.06},
        "param_schema": {
            "fast_period": {"type": "int", "min": 2, "max": 120, "label": "快线周期"},
            "slow_period": {"type": "int", "min": 5, "max": 300, "label": "慢线周期"},
            "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
            "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "止损比例"},
            "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.5, "label": "止盈比例"},
        },
        "builtin": True,
    },
    {
        "key": "price_action",
        "name": "price_action",
        "title": "关键位突破",
        "category": "breakout",
        "description": "识别支撑/阻力位，突破或回调入场，量能与 RSI 过滤，止损止盈参考关键位。",
        "logic": "用摆动高低点识别 S/R，突破阻力放量做多、回调支撑企稳做多，止损止盈与关键位联动。",
        "suitable": "有明确关键位的震荡后突破行情",
        "risk_tips": ["假突破会带来快速亏损", "震荡中 S/R 频繁失效"],
        "default_params": {
            "mode": "breakout", "breakout_pct": 0.001, "volume_confirm": 1.2, "rsi_ob": 72,
            "stop_loss_pct": 0.02, "take_profit_pct": 0.04, "use_sr_stop": True, "size_pct": 0.5,
        },
        "param_schema": _PRICE_ACTION_SCHEMA,
        "builtin": True,
    },
    {
        "key": "grid",
        "name": "grid",
        "title": "网格震荡",
        "category": "grid",
        "description": "相对基准价下跌买一档、上涨卖一档，适合横盘震荡行情自动低吸高抛。",
        "logic": "设基准价与网格间距，价格下跌 x% 买入一档、上涨 x% 卖出一档，控制最大档位。",
        "suitable": "长期横盘震荡、波动率稳定的行情",
        "risk_tips": ["单边下跌时不断加仓套牢", "单边上涨时过早清仓"],
        "default_params": {"grid_pct": 0.02, "qty_per_grid": 0.001, "max_positions": 20},
        "param_schema": {
            "grid_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "网格间距"},
            "qty_per_grid": {"type": "float", "min": 0.0001, "max": 100, "label": "每档数量"},
            "max_positions": {"type": "int", "min": 1, "max": 200, "label": "最大档位数"},
        },
        "builtin": True,
    },
    {
        "key": "bb_mean_reversion",
        "name": "bb_mean_reversion",
        "title": "布林带均值回归",
        "category": "mean_reversion",
        "description": "价格触及布林下轨超卖做多、上轨超买做空，回归中轨获利了结。",
        "logic": "收盘跌破布林下轨且 RSI<30 做多，站上中轨止盈；涨破上轨且 RSI>70 做空，回落中轨止盈。",
        "suitable": "区间震荡、有界波动行情",
        "risk_tips": ["单边趋势中逆势接飞刀", "布林带扩张时假超买超卖"],
        "default_params": {
            "mode": "pullback", "breakout_pct": 0.002, "volume_confirm": 0.8, "rsi_ob": 70,
            "stop_loss_pct": 0.03, "take_profit_pct": 0.05, "use_sr_stop": True, "size_pct": 0.5,
        },
        "param_schema": _PRICE_ACTION_SCHEMA,
        "builtin": True,
    },
    {
        "key": "rsi_reversal",
        "name": "rsi_reversal",
        "title": "RSI 超买超卖反转",
        "category": "mean_reversion",
        "description": "RSI 极端区间的反转策略，超卖抄底、超买逃顶。",
        "logic": "RSI<30 超卖做多、RSI>70 超买做空，价格回归关键位或移动止损离场。",
        "suitable": "波动率高、反转频繁的行情",
        "risk_tips": ["强趋势中 RSI 会持续钝化", "反转时机难以精准把握"],
        "default_params": {
            "mode": "pullback", "breakout_pct": 0.001, "volume_confirm": 0.7, "rsi_ob": 70,
            "stop_loss_pct": 0.025, "take_profit_pct": 0.04, "use_sr_stop": True, "size_pct": 0.4,
        },
        "param_schema": _PRICE_ACTION_SCHEMA,
        "builtin": True,
    },
    {
        "key": "atr_breakout",
        "name": "atr_breakout",
        "title": "ATR 波动突破",
        "category": "breakout",
        "description": "基于关键位突破 + 波动率确认的顺势突破策略，捕捉放量突破行情。",
        "logic": "识别近期高/低点构成通道，价格放量突破上轨（量能确认+RSI 过滤）做多，跌破下轨做空。",
        "suitable": "波动率放大、即将启动的行情",
        "risk_tips": ["横盘假突破多", "突破失败需快速止损"],
        "default_params": {
            "mode": "breakout", "breakout_pct": 0.003, "volume_confirm": 1.5, "rsi_ob": 75,
            "stop_loss_pct": 0.02, "take_profit_pct": 0.06, "use_sr_stop": True, "size_pct": 0.5,
        },
        "param_schema": _PRICE_ACTION_SCHEMA,
        "builtin": True,
    },
]


def list_repository() -> list[dict]:
    """返回策略仓库完整列表（含类别信息）。"""
    return [dict(t) for t in REPOSITORY]


def get_by_key(key: str) -> dict | None:
    for t in REPOSITORY:
        if t["key"] == key:
            return dict(t)
    return None


def register_repository_strategies() -> list[str]:
    """把仓库中的模板注册为可用策略（复用 price_action 执行器或原生策略）。"""
    from . import register_dynamic, _REGISTRY
    registered = []
    for t in REPOSITORY:
        name = t["name"]
        if name in _REGISTRY:
            continue  # 原生内置策略已有
        register_dynamic(name, {
            "name": name,
            "title": t["title"],
            "description": t["description"],
            "logic": t["logic"],
            "params": dict(t["default_params"]),
            "risk_tips": t["risk_tips"],
            "created_by": "repository",
        })
        registered.append(name)
    return registered
