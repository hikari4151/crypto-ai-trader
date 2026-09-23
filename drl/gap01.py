"""GAP-01 实盘验收判定（同品种单标的口径）。

口径依据（docs/LIVE_READINESS.md）：
- GAP-01 要求真实数据上、扣成本后、样本外正期望且稳健（walk-forward + 成本敏感性）；
- E7：目标为**单标的**时"无要求"（不强制多标的/跨品种）——本程序现役 evolve_symbols
  为单标的 ETH/USDT，因此跨品种不做硬门，判定聚焦同品种多折正期望 + 成本鲁棒性。

summarize_gap01 把两档 walk-forward 汇总（基准成本 / 成本翻倍）合并为最终判定：
- PASS        ：两档均 PASS（≥75% 折为正、中位数>0、每折成交≥min_trades）；
- INCONCLUSIVE：任一档样本不足（无折 / 成交样本不足），不误判 PASS/FAIL；
- FAIL        ：其余（基准不过，或成本翻倍后失去正期望）。
"""


def summarize_gap01(base: dict, cost2: dict) -> dict:
    """合并基准成本档与成本翻倍档的 walk-forward 汇总为 GAP-01 判定。

    Args:
        base: summarize_folds 输出（基准 fee/slippage）。
        cost2: summarize_folds 输出（成本 × 倍数，如 fee×2、slippage×2）。
    Returns:
        {"verdict": "PASS|INCONCLUSIVE|FAIL", "reason": ..., "base": base, "cost2": cost2}
    """
    bv = str((base or {}).get("verdict") or "INCONCLUSIVE")
    cv = str((cost2 or {}).get("verdict") or "INCONCLUSIVE")
    parts: list[str] = []
    if bv == "PASS" and cv == "PASS":
        verdict = "PASS"
    elif bv == "INCONCLUSIVE" or cv == "INCONCLUSIVE":
        verdict = "INCONCLUSIVE"
        if bv == "INCONCLUSIVE":
            parts.append("基准成本档样本不足")
        if cv == "INCONCLUSIVE":
            parts.append("成本翻倍档样本不足")
    elif bv == "PASS" and cv != "PASS":
        verdict = "FAIL"
        parts.append("成本翻倍后不满足多折正期望（成本鲁棒性不足）")
    else:
        verdict = "FAIL"
        parts.append("基准成本档未通过多折正期望")
    return {
        "verdict": verdict,
        "reason": "；".join(parts) if parts else "两档均通过（同品种多折正期望 + 成本鲁棒）",
        "base": base,
        "cost2": cost2,
    }