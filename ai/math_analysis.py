"""程序化数学行情分析（2026-09-22 从 web/api/ai.py 抽出复用）。

纯指标计算、不依赖 AI 猜测：方向五信号投票 + 摆动点 S/R 聚类 + ATR 波动 +
入场/止损/止盈 + 盈亏比 + 胜率估计 + 期望收益%。被 /api/ai/analyze 与引擎
数学期望信号门（ai.math_gate）共同使用。
"""
import logging

log = logging.getLogger("ai.math_analysis")


def math_analysis(snap: dict) -> dict:
    """程序化数学行情分析：不依赖 AI 猜测，纯指标计算给出可复现的结论。

    输出：趋势判定(上涨/下跌/震荡)、支撑/阻力位置、入场/止损/止盈、
    盈亏比、胜率估计(基于价格在 S/R 区间位置 + 方向信号一致性)、
    期望收益%（量化到当前价 1 单位仓位）。数据不足时返回 {}，前端隐藏。
    """
    try:
        ind = snap.get("indicators") or {}
        closes = snap.get("closes") or []
        candles = snap.get("candles") or []
        if not closes or not candles:
            return {}
        last = float(closes[-1])
        if last <= 0:
            return {}
        # 趋势判定：五信号投票（复用方向引擎）
        from ai.direction_engine import compute_direction
        prog = compute_direction(candles)
        direction = prog.get("direction") or "neutral"
        score = int(prog.get("score") or 0)
        conf = float(prog.get("confidence") or 0.0)
        trend_label = {"long": "上涨趋势", "short": "下跌趋势", "neutral": "震荡行情"}.get(direction, "震荡行情")
        # 波动率（ATR% → 价格）
        atr_pct = float(ind.get("atr_pct") or 0.0)
        atr = last * atr_pct / 100.0 if atr_pct > 0 else last * 0.01
        # 支撑/阻力（数学聚类位，非猜测）
        sr = ind.get("sr") or {}
        support = sr.get("support")
        resistance = sr.get("resistance")
        support_levels = sr.get("support_levels") or []
        resistance_levels = sr.get("resistance_levels") or []
        # ---- 入场方案（数学规则，不做行情猜测）----
        entry = stop = tp = None
        rr = 0.0
        if direction == "long":
            # 做多：贴支撑入场（若距支撑 <1.5ATR），否则现价；止损 1.5ATR 下方；止盈取阻力或 2.5ATR
            entry = support if (support and last - support < 1.5 * atr) else last
            stop = entry - 1.5 * atr
            tp = resistance if (resistance and resistance > entry) else entry + 2.5 * atr
        elif direction == "short":
            entry = resistance if (resistance and resistance - last < 1.5 * atr) else last
            stop = entry + 1.5 * atr
            tp = support if (support and support < entry) else entry - 2.5 * atr
        else:
            # 震荡：区间策略——贴近下沿做多 / 上沿做空（同时给两个方案）
            pass  # 区间边界由前端基于 S/R 呈现，这里不强制给单点方案
        if direction in ("long", "short") and entry and stop and tp:
            risk = abs(entry - stop)
            reward = abs(tp - entry)
            rr = round(reward / risk, 2) if risk > 0 else 0.0
        # ---- 胜率估计（数学口径，非玄学）----
        span = None
        if support is not None and resistance is not None and resistance > support:
            span = (last - support) / (resistance - support)
        base = 0.5
        if span is not None:
            if direction == "long":
                base += (1.0 - span) * 0.25      # 贴支撑(+0.25)~贴阻力(0.0)
            elif direction == "short":
                base += span * 0.25               # 贴阻力(+0.25)~贴支撑(0.0)
            else:
                base += 0.02                       # 震荡中性微偏
        win_rate = min(0.85, max(0.35, base + conf * 0.10))
        # ---- 期望收益%（按 1 单位价格计算，含盈亏比）----
        ev_pct = None
        if direction in ("long", "short") and entry and stop and tp:
            ev_pct = round(
                win_rate * (tp - entry) / entry
                - (1 - win_rate) * (entry - stop) / entry, 5)
        return {
            "direction": direction,
            "trend_label": trend_label,
            "score": score,
            "confidence": round(conf, 3),
            "last_price": round(last, 6),
            "atr": round(atr, 6),
            "support": support, "resistance": resistance,
            "support_levels": support_levels, "resistance_levels": resistance_levels,
            "entry": round(entry, 6) if entry else None,
            "stop_loss": round(stop, 6) if stop else None,
            "take_profit": round(tp, 6) if tp else None,
            "risk_reward": rr,
            "win_rate": round(win_rate, 3),
            "expected_value_pct": ev_pct,
            "interval_low": round(last - 2.0 * atr, 6) if direction == "neutral" else None,
            "interval_high": round(last + 2.0 * atr, 6) if direction == "neutral" else None,
            "basis": "基于摆动点 S/R 聚类 + ATR 波动 + 方向五信号投票的纯数学计算（不依赖 AI 预测）",
        }
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 数学分析计算失败: %s", e)
        return {}
