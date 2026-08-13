"""AI 输出量化校验体系：不带感情的规则检验引擎。

每个 AI 功能的输出都要经过本引擎的量化规则校验，防止幻觉/越界/自相矛盾：

1. **结构校验**：必填字段、类型、允许的 key 集合
2. **数值范围校验**：参数必须落在 param_schema 的 min/max/choices 内
3. **逻辑一致性校验**（每功能自定义规则）：
   - 行情解读：bias=long 必须 confidence>=0.6；signals 方向必须与 bias 一致；
     summary 必须引用实际提供的市场数据（防幻觉编造价格）
   - 策略设计/迭代：止盈>止损；RSI 超买>超卖；参数必须可执行；
     逻辑描述必须引用输入中的 S/R/量能 等真实特征
   - 复盘：score 必须与实际胜率/盈亏比匹配（胜率低不能给高分）
4. **幻觉检测**：输出中不得出现输入里不存在的价格/指标数字（关键词匹配）

校验失败返回**结构化错误列表**（每项含字段与原因），可驱动 AI 有界重试：
重试时把错误反馈给 AI 要求修正，最多 N 次；仍失败则该输出作废。
"""
import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# 允许的 AI 功能名
FEATURES = {
    "market_analysis", "param_optimize", "strategy_design",
    "strategy_iterate", "trade_review", "factor_mine",
}


class ValidationError(Exception):
    """AI 输出未通过量化校验。携带可反馈给 AI 的具体错误信息。"""

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        msg = "; ".join(f"{e.get('field','?')}: {e.get('reason','')}" for e in errors[:5])
        super().__init__(f"AI 输出未通过量化校验: {msg}")


# ============ 基础工具 ============

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _require(data: dict, field: str, types: tuple, reason: str) -> list[dict]:
    if field not in data:
        return [{"field": field, "reason": f"缺少必填字段 {field}"}]
    if not isinstance(data[field], types):
        return [{"field": field, "reason": f"{field} 类型错误：期望 {types} 之一，实际 {type(data[field]).__name__}"}]
    return []


def _within_range(v: Any, spec: dict) -> Optional[str]:
    """按 param_schema 检查数值/选择范围，返回错误原因或 None。"""
    t = spec.get("type")
    lo, hi = spec.get("min"), spec.get("max")
    if t == "float" or t == "int":
        if not _is_number(v):
            return f"应为 {t} 数值，实际 {v!r}"
        if lo is not None and v < lo:
            return f"小于最小值 {lo}"
        if hi is not None and v > hi:
            return f"大于最大值 {hi}"
    elif t == "str":
        choices = spec.get("choices")
        if choices and str(v) not in choices:
            return f"不在可选范围内 {choices}"
    elif t == "bool":
        if not isinstance(v, bool):
            return f"应为布尔值，实际 {v!r}"
    return None


def _validate_params(params: dict, param_schema: dict) -> list[dict]:
    """校验策略参数：仅允许 schema 内的 key，且值必须在范围内。"""
    errors = []
    for k, v in (params or {}).items():
        if k not in param_schema:
            errors.append({"field": f"params.{k}", "reason": f"参数 {k} 不在策略 schema 中"})
            continue
        reason = _within_range(v, param_schema[k])
        if reason:
            errors.append({"field": f"params.{k}", "reason": reason})
    return errors


# ============ 各功能校验器 ============

def _validate_market_analysis(data: dict, snap: dict, ctx: dict = {}) -> list[dict]:
    errors = []
    errors += _require(data, "regime", (str,), "必须给出市场状态")
    errors += _require(data, "bias", (str,), "必须给出方向")
    errors += _require(data, "confidence", (int, float), "必须给出置信度")
    errors += _require(data, "summary", (str,), "必须给出解读")
    if errors:
        return errors

    regime, bias = data["regime"], data["bias"]
    conf = float(data["confidence"])
    # 量化规则
    if regime not in ("趋势", "震荡", "拐点", "未知"):
        errors.append({"field": "regime", "reason": f"regime 非法: {regime}"})
    if bias not in ("long", "short", "neutral"):
        errors.append({"field": "bias", "reason": f"bias 非法: {bias}"})
    if not 0 <= conf <= 1:
        errors.append({"field": "confidence", "reason": "confidence 必须在 0-1"})
    if bias == "long" and conf < 0.6:
        errors.append({"field": "confidence", "reason": "bias=long 要求 confidence>=0.6（趋势确立门槛）"})
    if bias == "short" and conf < 0.6:
        errors.append({"field": "confidence", "reason": "bias=short 要求 confidence>=0.6"})

    # ---- 方向一致性：程序复算方向（强确认时）与 AI bias 矛盾 → 打回 ----
    program_dir = snap.get("program_direction")
    program_score = int(snap.get("program_score") or 0)
    if program_dir in ("long", "short") and bias in ("long", "short") and program_dir != bias:
        # 仅当程序方向强确认（|score|>=3）时强制对齐；弱确认只提示不拦截
        if abs(program_score) >= 3:
            errors.append({
                "field": "bias",
                "reason": f"程序从K线结构复算方向为 {program_dir}（评分 {program_score:+d}），"
                          f"你的 bias={bias} 与之矛盾。请修正 bias，或在 summary 中明确给出结构反转证据后重试。",
            })

    # ---- 决策连续性：上一轮结论短时反手 → 强制降置信度上限 ----
    prev = ctx.get("continuity") or {}
    if bias in ("long", "short"):
        from .context_engine import FLIP_CONFIDENCE_CAP, FLIP_COOLDOWN_HOURS
        prev_bias = prev.get("bias") if isinstance(prev, dict) else None
        prev_ts = prev.get("ts") if isinstance(prev, dict) else None
        if prev_bias in ("long", "short") and prev_bias != bias and prev_ts:
            from datetime import datetime, timezone
            try:
                if isinstance(prev_ts, datetime):
                    prev_dt = prev_ts
                else:
                    prev_dt = datetime.fromisoformat(str(prev_ts))
                if prev_dt.tzinfo is None:
                    prev_dt = prev_dt.replace(tzinfo=timezone.utc)
                hours = (datetime.now(timezone.utc) - prev_dt).total_seconds() / 3600.0
                if hours < FLIP_COOLDOWN_HOURS and conf > FLIP_CONFIDENCE_CAP:
                    errors.append({
                        "field": "confidence",
                        "reason": f"上一轮分析（{hours:.1f}小时前）为 {prev_bias}，本轮反手为 {bias}，"
                                  f"属短时反手（<{FLIP_COOLDOWN_HOURS:.0f}h）。置信度不得超过 "
                                  f"{FLIP_CONFIDENCE_CAP:.0%}，或保持 neutral；除非你能用结构反转证据说明。",
                    })
            except Exception:  # noqa: BLE001
                pass

    # signals 方向一致性
    signals = data.get("signals") or []
    if not isinstance(signals, list):
        errors.append({"field": "signals", "reason": "signals 必须是数组"})
    else:
        for i, s in enumerate(signals):
            if not isinstance(s, str):
                errors.append({"field": f"signals[{i}]", "reason": "signals 元素必须是字符串"})
                continue
            if bias == "long" and ("做空" in s or "卖出" in s):
                errors.append({"field": f"signals[{i}]", "reason": f"bias=long 但信号建议做空: {s[:30]}"})
            if bias == "short" and ("做多" in s or "买入" in s):
                errors.append({"field": f"signals[{i}]", "reason": f"bias=short 但信号建议做多: {s[:30]}"})

    # 幻觉检测：signals 中出现的价格数字必须与快照价格同一量级（防编造离谱价位）
    ind = snap.get("indicators") or {}
    ref_price = None
    for k in ("close", "ma_fast", "ma_slow"):
        v = ind.get(k)
        if isinstance(v, (int, float)) and v:
            ref_price = float(v)
            break
    if ref_price and isinstance(signals, list):
        import re as _re
        for i, s in enumerate(signals):
            if not isinstance(s, str):
                continue
            # 提取信号里所有数字价格（整数/带小数）
            for num in _re.findall(r"\d{3,}(?:\.\d+)?", s):
                try:
                    v = float(num)
                except ValueError:
                    continue
                # 价格量级检测：与参考价格相差超过 50 倍视为幻觉
                if ref_price > 1 and (v > ref_price * 50 or v < ref_price / 50):
                    errors.append({
                        "field": f"signals[{i}]",
                        "reason": f"价格 {v:.0f} 与当前市场价 {ref_price:.0f} 不在同一量级，疑似编造",
                    })
                    break

    # ---- 交易者方程校验：trade_plan 三价几何 + RR≥1.0 + 胜率×回报 ----
    trade_plan = data.get("trade_plan")
    if trade_plan is not None:
        from .context_engine import validate_trade_plan
        ref = ctx.get("trade_plan_ref_price") or ref_price
        for msg in validate_trade_plan(trade_plan, bias, ref):
            errors.append({"field": "trade_plan", "reason": msg})
    return errors


def _validate_strategy(data: dict, param_schema: dict, feature: str) -> list[dict]:
    errors = []
    for f in ("name", "title", "description", "logic", "params"):
        errors += _require(data, f, (str,) if f != "params" else (dict,), f"缺少 {f}")
    if errors:
        return errors
    # 参数范围
    errors += _validate_params(data.get("params"), param_schema)
    # 风控逻辑一致性
    p = data.get("params") or {}
    sl, tp = p.get("stop_loss_pct"), p.get("take_profit_pct")
    if sl is not None and tp is not None and tp <= sl:
        errors.append({"field": "params.take_profit_pct",
                       "reason": "止盈必须大于止损（否则无法形成正盈亏比）"})
    rsi_ob, rsi_os = p.get("rsi_ob"), p.get("rsi_os")
    if rsi_ob is not None and rsi_os is not None and rsi_ob <= rsi_os:
        errors.append({"field": "params.rsi_ob", "reason": "RSI 超买阈值必须大于超卖阈值"})
    # 名称规范性（括号明确优先级：空名 或 非"字母/数字/下划线组合"即报错）
    name = data.get("name", "")
    if not name or not all(c.isalnum() or c == "_" for c in name):
        errors.append({"field": "name", "reason": "策略名必须为小写字母/数字/下划线"})
    return errors


def _validate_review(data: dict, summary: dict) -> list[dict]:
    errors = []
    errors += _require(data, "score", (int, float), "缺少评分")
    errors += _require(data, "strengths", (list,), "缺少优点列表")
    errors += _require(data, "problems", (list,), "缺少问题列表")
    errors += _require(data, "action_items", (list,), "缺少行动项列表")
    if errors:
        return errors
    # 量化规则：评分必须与实际胜率/盈亏比匹配
    score = float(data["score"])
    win_rate = float(summary.get("win_rate") or 0)
    total_pnl = float(summary.get("total_pnl") or 0)
    if not 0 <= score <= 100:
        errors.append({"field": "score", "reason": "score 必须在 0-100"})
    # 亏损且胜率低时不能给高分（防 AI 谄媚）
    if total_pnl < 0 and win_rate < 0.4 and score > 60:
        errors.append({"field": "score", "reason":
                       f"实际亏损({total_pnl:.0f})且胜率({win_rate:.0%})低，score={score:.0f} 明显偏高（禁止谄媚打分）"})
    return errors


def _validate_factor_mine(data: dict) -> list[dict]:
    errors = []
    factors = data.get("factors")
    if not isinstance(factors, list) or not factors:
        return [{"field": "factors", "reason": "缺少候选因子数组"}]
    for i, f in enumerate(factors[:10]):
        if not isinstance(f, dict):
            errors.append({"field": f"factors[{i}]", "reason": "因子项必须是对象"})
            continue
        for k in ("name", "title", "category", "expression", "logic"):
            if not f.get(k):
                errors.append({"field": f"factors[{i}].{k}", "reason": f"缺少 {k}"})
    return errors


# ============ 引擎 ============

_VALIDATORS: dict[str, Callable] = {
    "market_analysis": lambda d, ctx: _validate_market_analysis(d, (ctx or {}).get("snap", {}), ctx or {}),
    "param_optimize": lambda d, ctx: _validate_strategy_params_optimize(d, ctx),
    "strategy_design": lambda d, ctx: _validate_strategy(d, ctx["param_schema"], "strategy_design"),
    "strategy_iterate": lambda d, ctx: _validate_strategy(d, ctx["param_schema"], "strategy_iterate"),
    "trade_review": lambda d, ctx: _validate_review(d, ctx.get("summary", {})),
    "factor_mine": lambda d, ctx: _validate_factor_mine(d),
}


def _validate_strategy_params_optimize(data: dict, ctx: dict) -> list[dict]:
    """参数优化：只输出 params 和 reason，校验参数范围与合理性。"""
    errors = []
    params = data.get("params")
    if not isinstance(params, dict):
        return [{"field": "params", "reason": "缺少参数对象"}]
    param_schema = ctx.get("param_schema", {})
    errors += _validate_params(params, param_schema)
    if not data.get("reason"):
        errors.append({"field": "reason", "reason": "缺少修改理由"})
    # 参数不能为空
    if not params:
        errors.append({"field": "params", "reason": "参数为空（无把握时可以给出最小调整或明确不改）"})
    return errors


def validate_ai_output(feature: str, data: dict, ctx: dict = {}) -> None:
    """校验 AI 输出。通过则返回；失败抛 ValidationError（携带可反馈的错误）。

    ctx 需提供校验所需的上下文，如:
    - param_schema: 策略参数 schema（strategy_design / strategy_iterate / param_optimize）
    - summary: 交易汇总（trade_review）
    - indicators 快照（market_analysis，经 snap）
    """
    validator = _VALIDATORS.get(feature)
    if validator is None:
        raise ValidationError([{"field": "_", "reason": f"未知 AI 功能: {feature}"}])
    errors = validator(data, ctx or {})
    if errors:
        raise ValidationError(errors)


def feedback_message(feature: str, data: dict, ctx: dict = {}) -> str:
    """把校验错误转成可喂回 AI 的修正指令。"""
    try:
        validate_ai_output(feature, data, ctx)
        return ""
    except ValidationError as e:
        lines = ["你的输出未通过量化规则校验，请修正后重新输出严格 JSON："]
        for err in e.errors[:6]:
            lines.append(f"- {err['field']}: {err['reason']}")
        return "\n".join(lines)
