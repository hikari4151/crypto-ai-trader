"""GAP-01 完整验收判定（grid 基线验证用，见 docs/LIVE_READINESS.md §3）。

assess_gap01 汇总多标的 detect_overfit 报告 + 成本扫描，按 GAP-01 四条给出
PASS / INCONCLUSIVE / FAIL 与逐项检查清单。

注意口径分工：
- 本模块用于"先让任一简单策略（grid）过 GAP-01"的基线验证（诊断报告建议第6条(a)），
  **包含跨品种方向一致性（≥2 标的 OOS ret>0）与 DSR/PBO 统计检验**；
- 进化引擎里的单标的观测 `drl/gap01.py` 依据 LIVE_READINESS E7 豁免跨品种，
  两者用途不同，不要混用。
"""

_PASS_VERDICTS = {"通过"}

# P0 实测校准（2026-09-18）：grid 在 BTC/SOL 上 OOS 微正但被完整检测判"疑似过拟合"
#（DSR 0.37~0.68、PBO 0.20~0.45），ETH 严重过拟合——所以"疑似过拟合"必须判 FAIL。
_FAIL_VERDICTS = {"严重过拟合", "疑似过拟合"}
_INCONCLUSIVE_VERDICTS = {"无法判定", ""}


def assess_gap01(symbol_reports, cost_scan=None, *,
                 min_trades: int = 30, dsr_min: float = 0.95,
                 pbo_max: float = 0.5) -> dict:
    """按 GAP-01 四条验收标准汇总判定。

    Args:
        symbol_reports: [{symbol, verdict, oos_ret, oos_trades, dsr, pbo, ...}]
        cost_scan: run_cost_scan 输出（或 None）。
    Returns:
        {"verdict": "PASS|INCONCLUSIVE|FAIL", "reason", "checks",
         "positive_symbols", "total_symbols"}
    """
    checks: list[dict] = []
    fails = 0
    inconclusive = 0
    positive_syms = 0

    for sr in symbol_reports or []:
        sym = sr.get("symbol", "?")
        verdict = str(sr.get("verdict") or "")
        if verdict in _PASS_VERDICTS:
            checks.append({"key": f"{sym}.verdict", "ok": True, "note": f"verdict={verdict}"})
        elif verdict in _FAIL_VERDICTS:
            checks.append({"key": f"{sym}.verdict", "ok": False,
                           "note": f"verdict={verdict}（不满足通过标准）"})
            fails += 1
        else:
            checks.append({"key": f"{sym}.verdict", "ok": None,
                           "note": f"verdict={verdict or '缺失'}（证据不足）"})
            inconclusive += 1

        oos = float(sr.get("oos_ret") or 0.0)
        trades = int(sr.get("oos_trades") or 0)
        if oos <= 0:
            checks.append({"key": f"{sym}.oos_ret", "ok": False,
                           "note": f"OOS收益 {oos:+.4f} ≤ 0"})
            fails += 1
        else:
            positive_syms += 1
            if trades < min_trades:
                checks.append({"key": f"{sym}.oos_trades", "ok": None,
                               "note": f"OOS交易 {trades} < {min_trades}（证据不足）"})
                inconclusive += 1
            else:
                checks.append({"key": f"{sym}.oos_trades", "ok": True,
                               "note": f"OOS交易 {trades} ≥ {min_trades}"})

        dsr = sr.get("dsr")
        if dsr is None:
            checks.append({"key": f"{sym}.dsr", "ok": None,
                           "note": "DSR 缺失（证据不足）"})
            inconclusive += 1
        elif dsr >= dsr_min:
            checks.append({"key": f"{sym}.dsr", "ok": True,
                           "note": f"DSR={dsr:.2f} ≥ {dsr_min}"})
        else:
            checks.append({"key": f"{sym}.dsr", "ok": False,
                           "note": f"DSR={dsr:.2f} < {dsr_min}"})
            fails += 1

        pbo = sr.get("pbo")
        if pbo is not None and pbo > pbo_max:
            checks.append({"key": f"{sym}.pbo", "ok": False,
                           "note": f"PBO={pbo:.2f} > {pbo_max}"})
            fails += 1

    # 方向一致性：≥2 标的 OOS 为正
    if positive_syms >= 2:
        checks.append({"key": "direction", "ok": True,
                       "note": f"{positive_syms} 个标的 OOS 为正（≥2）"})
    else:
        checks.append({"key": "direction", "ok": False,
                       "note": f"仅 {positive_syms} 个标的 OOS 为正（需 ≥2）"})
        fails += 1

    # 成本鲁棒：5× 成本档仍正
    if cost_scan:
        worst = float((cost_scan.get("sensitivity") or {}).get("worst_ret") or 0.0)
        base = float(((cost_scan.get("base") or {}).get("metrics") or {}).get("total_return") or 0.0)
        if worst > 0:
            checks.append({"key": "cost.worst_ret", "ok": True,
                           "note": f"5×成本仍正（base={base:+.4f}，worst={worst:+.4f}）"})
        else:
            checks.append({"key": "cost.worst_ret", "ok": False,
                           "note": f"5×成本收益 {worst:+.4f} ≤ 0（成本不鲁棒）"})
            fails += 1
    else:
        checks.append({"key": "cost", "ok": None,
                       "note": "未提供成本扫描（证据不足）"})
        inconclusive += 1

    if fails:
        verdict = "FAIL"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    bad = [c["note"] for c in checks if c["ok"] is False]
    reason = "；".join(bad) if bad else (
        "全部通过（多标的 OOS 正期望 + DSR/PBO 统计检验 + 成本鲁棒 + 方向一致）"
        if verdict == "PASS" else "存在证据不足项（未判通过/失败）")
    return {"verdict": verdict, "reason": reason, "checks": checks,
            "positive_symbols": positive_syms,
            "total_symbols": len(symbol_reports or [])}