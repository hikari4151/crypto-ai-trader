"""AI 市场解读的确定性程序上下文：方向复算 + 决策连续性 + 交易者方程。

设计动机（借鉴 PA Agent 的"AI 提出、程序裁决"思想）：
AI 解读是可验证的——方向/方案都应与 K 线结构、历史结论一致。本模块在
解读前程序独立计算三类可验证事实，供两处使用：
  1. 提示词层：注入引导，让 AI 一开始就对齐程序结论（减少无效重试）
  2. 校验层：validator 强制检查一致性（矛盾/反手/方程不通过即打回）

三个能力：
  - 方向复算：K 线五信号投票 → program_direction
  - 决策连续性：上一轮结论 vs 本轮，短时反手 → 强制降置信度
  - 交易者方程：RR≥1.0 + 胜率×回报>败率×风险 → 校验 AI 结构化方案
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

# 反手冷却窗口（小时）：上一轮与本轮方向相反且间隔短于该值 → 程序干预
FLIP_COOLDOWN_HOURS = 2.0
# 短时反手时 AI confidence 上限
FLIP_CONFIDENCE_CAP = 0.45
# 交易者方程参数
MIN_RISK_REWARD_RATIO = 1.0


# ============ 交易者方程 ============

def validate_trade_plan(plan: dict, bias: str, ref_price: Optional[float]) -> list[str]:
    """校验 AI 结构化交易方案（side/entry/stop_loss/take_profit/win_rate）。

    返回错误列表（空=通过）。规则：
      - side 必须与 bias 一致
      - 几何：long 时 stop < entry < tp；short 时 stop > entry > tp
      - RR = reward / risk >= 1.0
      - 交易者方程：win_rate*reward > (1-win_rate)*risk
      - 价格与参考价同量级（反幻觉）
    """
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["trade_plan 必须是对象"]

    side = str(plan.get("side", ""))
    entry = _to_float(plan.get("entry"))
    stop = _to_float(plan.get("stop_loss"))
    tp = _to_float(plan.get("take_profit"))
    wr = _to_float(plan.get("win_rate"))

    # 必填
    if side not in ("long", "short"):
        errors.append("trade_plan.side 必须是 long/short")
    if entry is None:
        errors.append("trade_plan.entry 必须为数值")
    if stop is None:
        errors.append("trade_plan.stop_loss 必须为数值")
    if tp is None:
        errors.append("trade_plan.take_profit 必须为数值")
    if wr is None or not (0 < wr < 1):
        errors.append("trade_plan.win_rate 必须在 (0,1)")
    if errors:
        return errors

    # 方向一致性：side 必须与 bias 一致
    if bias in ("long", "short") and side != bias:
        errors.append(f"trade_plan.side={side} 与 bias={bias} 冲突（方案方向必须与整体判断一致）")

    # 几何关系
    if side == "long":
        if not (stop < entry < tp):
            errors.append(f"做多几何错误：要求 stop({stop}) < entry({entry}) < take_profit({tp})")
    elif side == "short":
        if not (stop > entry > tp):
            errors.append(f"做空几何错误：要求 stop({stop}) > entry({entry}) > take_profit({tp})")

    if errors:
        return errors

    # 盈亏比
    risk = abs(entry - stop)
    reward = abs(tp - entry)
    if risk < 1e-12:
        errors.append("风险距离为 0（stop==entry）")
        return errors
    rr = reward / risk
    if rr < MIN_RISK_REWARD_RATIO:
        errors.append(f"盈亏比 RR={rr:.2f} < {MIN_RISK_REWARD_RATIO}（止盈过近，期望为负）")

    # 交易者方程：胜率×回报 > 败率×风险
    if wr is not None:
        expected = wr * reward - (1 - wr) * risk
        if expected <= 0:
            errors.append(
                f"交易者方程不通过：{wr:.0%}×{reward:.2f} - {1-wr:.0%}×{risk:.2f} = {expected:.2f} ≤ 0"
            )

    # 价格量级（反幻觉）
    if ref_price and ref_price > 1:
        for name, v in (("entry", entry), ("stop_loss", stop), ("take_profit", tp)):
            if v > ref_price * 50 or v < ref_price / 50:
                errors.append(f"trade_plan.{name}={v:.0f} 与当前价 {ref_price:.0f} 不在同一量级，疑似编造")
                break
    return errors


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ============ 决策连续性 ============

def build_continuity_context(prev_analysis: Optional[dict], current_bias: str,
                             symbol: str) -> dict:
    """根据上一轮分析结论构建决策连续性上下文。

    prev_analysis: 上一轮 Analysis 解析结果（含 bias/confidence/ts）
    返回 {has_prev, prev_bias, prev_confidence, hours_ago, flipped, flip_conflict,
          program_confirmed, note}
    """
    ctx: dict[str, Any] = {
        "has_prev": False,
        "prev_bias": None,
        "prev_confidence": None,
        "hours_ago": None,
        "flipped": False,
        "flip_conflict": False,
        "note": "",
    }
    if not prev_analysis:
        return ctx
    prev_bias = str(prev_analysis.get("bias") or "neutral")
    ctx.update({"has_prev": True, "prev_bias": prev_bias,
                "prev_confidence": prev_analysis.get("confidence")})

    # 计算时间间隔
    prev_ts = prev_analysis.get("ts")
    hours_ago = None
    if prev_ts:
        try:
            if isinstance(prev_ts, datetime):
                prev_dt = prev_ts
            else:
                prev_dt = datetime.fromisoformat(str(prev_ts))
            if prev_dt.tzinfo is None:
                prev_dt = prev_dt.replace(tzinfo=timezone.utc)
            hours_ago = (datetime.now(timezone.utc) - prev_dt).total_seconds() / 3600.0
            ctx["hours_ago"] = round(hours_ago, 2)
        except Exception:  # noqa: BLE001
            log.warning("[context] 时间解析失败: %r", prev_ts)

    # 反手判定：上一轮 long↔short 相反
    _L = {"long", "short"}
    if prev_bias in _L and current_bias in _L and prev_bias != current_bias:
        ctx["flipped"] = True
        if hours_ago is not None and hours_ago < FLIP_COOLDOWN_HOURS:
            ctx["flip_conflict"] = True
            ctx["note"] = (
                f"上一轮分析（{hours_ago:.1f}小时前）判断 {prev_bias}，本轮反手为 {current_bias}，"
                f"属短时反手（<{FLIP_COOLDOWN_HOURS:.0f}h）。除非结构已明确反转，否则置信度不得超过 "
                f"{FLIP_CONFIDENCE_CAP:.0%}，或保持 neutral。"
            )
        else:
            ctx["note"] = (
                f"上一轮判断 {prev_bias}，本轮 {current_bias}（方向反转，请确认结构反转证据）"
            )
    return ctx


def parse_prev_analysis(content: str) -> Optional[dict]:
    """解析上一轮 Analysis.content（JSON 字符串）为可读字段。"""
    if not content:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "bias": data.get("bias"),
        "confidence": data.get("confidence"),
        "regime": data.get("regime"),
        "ts": data.get("ts"),
    }
