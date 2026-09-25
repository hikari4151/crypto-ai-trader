"""AI 数学期望信号门（2026-09-22，§19 落地）。

`_math.expected_value_pct`（纯数学、可复现、无 LLM）是行情分析里最可信的
量化护栏：实测 ETH/5m 期望 -0.16% 时，即使 LLM 方向解读明确也不应入场。
本门只拦"新开多/加仓"（buy）且期望低于阈值的情况；sell/止损/减仓与数据
缺失一律放行（保守原则：不阻断风控路径，也不因数据缺失阻断交易）。
"""
from typing import Optional, Tuple


def math_ev_gate_decision(side: str,
                          expected_value_pct: Optional[float],
                          enabled: bool = True,
                          min_ev: float = 0.0) -> Tuple[bool, str]:
    """数学期望信号门判定。

    Args:
        side: 信号方向（"buy"/"sell"）。
        expected_value_pct: AI 数学分析的期望收益%（可 None=无结论）。
        enabled: 门开关（False=全部放行）。
        min_ev: 期望阈值（低于该值且为 buy 时拦截；默认 0.0=负期望才拦）。

    Returns:
        (allow, reason)：allow=False 时 reason 为拦截原因。
    """
    if not enabled:
        return True, ""
    if side != "buy":
        return True, ""
    if expected_value_pct is None:
        return True, ""
    if float(expected_value_pct) < float(min_ev or 0.0):
        return False, (f"AI数学期望 {float(expected_value_pct):.4f} < "
                       f"{float(min_ev):.4f}（负期望抑制买入）")
    return True, ""


MATH_EV_CACHE_MAX = 128


def math_ev_cached(snap: dict, compute, cache: dict,
                   maxsize: int = MATH_EV_CACHE_MAX) -> Tuple[Optional[float], bool]:
    """信号门数学期望缓存（2026-09-22，P1-①）：同一根 K 线（candles[-1][5] 时间戳）
    内的多个 buy 信号复用一次 math_analysis 结果；新 K 线自动失效；无 candles
    （无时间戳）不缓存（每次计算）；缓存超限清空（简单策略）。

    Returns:
        (expected_value_pct, hit)：hit=True 表示来自缓存。
    """
    try:
        key = (snap or {}).get("candles")[-1][5]
    except (IndexError, TypeError, KeyError, AttributeError):
        return (compute(snap).get("expected_value_pct"), False)
    if key in cache:
        return cache[key], True
    ev = compute(snap).get("expected_value_pct")
    if len(cache) >= maxsize:
        cache.clear()
    cache[key] = ev
    return ev, False
