"""进化引擎纯判定规则（2026-09-22，P2-④ 拆分第一步）。

从 drl/evolve_engine.py 抽出的无状态纯函数（无 self/引擎依赖）：
- _should_rollback：回退判据（P0-4）
- _should_from_scratch_reset：连续拒绝重训判据
- _oos_gate_passed：因子挖掘 OOS 安检门
- _longwindow_verdict：30/90 天长窗回放判定

evolve_engine.py 通过模块级 import 保留这些名字（外部/测试引用不变）。
"""
import math
from typing import Optional


def _should_rollback(new_fit: float, old_fit: Optional[float], threshold: float) -> bool:
    """回退判据（P0-4）：旧 fitness 为正时按比例阈值比较；旧 fitness 非正时
    “×threshold”会让阈值更宽松（方向反转），改为新值更低即回退。

    run24 修复：NaN 旧/新 fitness 必须视为不可比（NaN 与任何数比较恒 False，
    会让回退保护永久失效——NaN 锚点污染后永不回退/永不拦截）。
    """
    if old_fit is None or not math.isfinite(old_fit) or not math.isfinite(new_fit):
        return False
    if old_fit <= 0:
        return new_fit < old_fit
    return new_fit < old_fit * threshold


def _should_from_scratch_reset(reject_streak: int, threshold: int) -> bool:
    """连续拒绝达到阈值时，下一轮强制 from-scratch 重训（0=关闭）。

    对齐 factor_miner 的 evolve_factor_miner_reset_after_rejects 先例：续训会
    困在老化基线上（"永久连拒"），周期重置可跳出局部区域重新探索。
    """
    threshold = int(threshold or 0)
    return threshold > 0 and int(reject_streak or 0) >= threshold


def _longwindow_verdict(ret_30d: Optional[float], ret_90d: Optional[float],
                        data_ok: bool) -> dict:
    """长窗回放判定（2026-09-21，§16.5 升级）：30 天与 90 天真实数据引擎撮合
    回放均正收益才算 PASS（杜绝 v44 式"近期窗口效应"冒充实盘候选）。

    数据不足 → INCONCLUSIVE；任一窗口缺失/收益 ≤0 → FAIL。
    """
    if not data_ok or ret_30d is None or ret_90d is None:
        return {"verdict": "INCONCLUSIVE",
                "reason": "长窗数据不足（30/90 天回放缺失）"}
    if float(ret_30d) > 0 and float(ret_90d) > 0:
        return {"verdict": "PASS",
                "reason": f"30d={ret_30d:+.4f} 90d={ret_90d:+.4f} 均正"}
    return {"verdict": "FAIL",
            "reason": f"30d={ret_30d:+.4f} 90d={ret_90d:+.4f} 未全正"}


def _oos_gate_passed(report: Optional[dict]) -> tuple[bool, str]:
    """因子挖掘的 OOS 安检门：判据与 /api/drl/mine-factor 注册前完全一致。

    进化引擎此前把组合因子无条件写进 _cascade_*，策略 DRL 下一轮就直接继承，
    等于绕过了 API 路径上的 valid 检查。
    """
    rep = report or {}
    if not rep.get("enabled"):
        return False, "OOS 段样本不足，无样本外证据"
    if not rep.get("valid"):
        return False, str(rep.get("reason") or "OOS 安检未通过")
    # E-EXP（2026-09 修复）：factor_quality_gate 用 |rank_ic| 判门，会放行
    # OOS 方向反转的因子——组合因子权重按决策时点 IC 定向，OOS rank_ic 为负 =
    # 实时"反着做"。生产实锤：round 727 发布因子 rank_ic=-0.0416 却 valid=True
    # （.optim/exp_training/）。组合因子必须正向 rank_ic 才可发布/级联。
    rank_ic = rep.get("rank_ic")
    if rank_ic is not None:
        try:
            if float(rank_ic) <= 0.0:
                return False, f"OOS 秩IC非正（{float(rank_ic):.4f}），因子方向反转，拒绝发布"
        except (TypeError, ValueError):
            return False, f"OOS 秩IC无效（{rank_ic}），拒绝发布"
    return True, ""