"""AI 提示词回归评测集（P3）：让"提示词改动"可度量。

目标：改一句提示词，如何知道解读质量变好还是变坏？这里提供数据驱动闭环：
- 用历史K线构造 N 个固定快照（golden set，确定性、可复现）
- 每个快照预计算两项客观基准：
  - program_direction：程序方向复算结果（direction_engine）
  - actual_direction：该时点之后未来 N 根K线的真实涨跌方向
- produce(snap) -> AI 输出 dict；对其跑 validator 并统计三类指标：
  ① validator 一次通过率（格式/规则合规性）
  ② 方向与程序方向一致率（与"AI 提出、程序裁决"的信任度）
  ③ 方向与真实未来涨跌命中率（方向预测能力）

真实用法：produce 用 AIClient.chat_json_validated 调真实模型，把 (输出, 是否
通过) 交给本模块统计；无 API 时用 canned 输出即可验证评测机制本身。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional


from .validator import ValidationError, validate_ai_output

log = logging.getLogger(__name__)

_HORIZON = 8          # 未来 8 根K线判断真实方向
_TREND_BAND = 0.002   # 涨跌幅低于该带视为 neutral


def _build_golden(n_cases: int = 12, n_candles: int = 60, seed: int = 7) -> list[dict]:
    """确定性构造 golden set（用 generate_demo 合成行情，避免网络依赖）。"""
    from backtest.data_loader import generate_demo
    from .direction_engine import compute_direction

    df = generate_demo(timeframe="1h", n=n_candles + n_cases + _HORIZON + 2)
    closes = df["close"].to_numpy(float)
    golden = []
    for k in range(n_cases):
        i = n_candles + k
        window = df.iloc[i - n_candles: i]
        candles = [[int(ts.timestamp() * 1000), float(o), float(h), float(l), float(c), float(v)]
                   for ts, o, h, l, c, v in zip(window.index, window["open"], window["high"],
                                                window["low"], window["close"], window["volume"])]
        closes_w = list(closes[i - n_candles:i])
        snap = {
            "symbol": "BTC/USDT", "timeframe": "1h",
            "candles": candles, "closes": closes_w,
            "indicators": {"close": float(closes[i - 1])},
        }
        prog = compute_direction(candles)
        fut = closes[i + _HORIZON] / closes[i - 1] - 1.0
        actual = "long" if fut > _TREND_BAND else ("short" if fut < -_TREND_BAND else "neutral")
        golden.append({
            "snap": snap,
            "program_direction": prog.get("direction"),
            "future_ret": float(fut),
            "actual_direction": actual,
        })
    return golden


# 模块级 golden set：一次构建、多处复用（确定性 seed，可复现）
GOLDEN_SET: list[dict] = _build_golden()


def _one_pass(out: dict, snap: dict) -> bool:
    try:
        validate_ai_output("market_analysis", out, {"snap": snap})
        return True
    except ValidationError:
        return False


def evaluate_prompt(produce: Callable[[dict], dict],
                    golden: Optional[list[dict]] = None) -> dict:
    """对 produce 函数在每个 golden 案例上评估，返回三类指标。

    produce(snap) 需返回一个 market_analysis 输出 dict（bias/confidence/summary/...）。
    返回 {cases, validator_pass_rate, direction_agreement_rate, directional_accuracy}。
    """
    golden = golden or GOLDEN_SET
    total = ok = 0
    agree_den = agree_num = 0
    hit_den = hit_num = 0
    for case in golden:
        try:
            out = produce(case["snap"])
        except Exception as e:  # noqa: BLE001
            log.warning("[evals] produce 异常，跳过该案例: %s", e)
            continue
        if not isinstance(out, dict):
            continue
        total += 1
        valid = _one_pass(out, case["snap"])
        ok += int(valid)
        bias = out.get("bias")
        prog = case["program_direction"]
        actual = case["actual_direction"]
        if bias in ("long", "short") and prog in ("long", "short"):
            agree_den += 1
            agree_num += int(bias == prog)
        if bias in ("long", "short") and actual in ("long", "short"):
            hit_den += 1
            hit_num += int(bias == actual)
    return {
        "cases": total,
        "validator_pass_rate": round(ok / total, 4) if total else 0.0,
        "direction_agreement_rate": round(agree_num / agree_den, 4) if agree_den else 0.0,
        "directional_accuracy": round(hit_num / hit_den, 4) if hit_den else 0.0,
    }


# ============ 设计 / 迭代提示词的回归评测（结构级） ============
# 目的：让"改一句设计/迭代提示词"同样可度量——golden set 复用行情快照，
# produce 返回对应通道的输出 dict，统计校验一次通过率与 schema 覆盖度。
# 方向命中率对设计/迭代不适用（策略无"未来方向"单一答案），故不计算。

def _structural_pass(feature: str, out: dict, ctx: dict) -> tuple[bool, int, int]:
    """跑 validator + params schema 覆盖率，返回 (通过?, 覆盖键数, schema键数)。"""
    try:
        validate_ai_output(feature, out, ctx)
        valid = True
    except ValidationError:
        valid = False
    params = out.get("params") or {}
    sch = ctx.get("param_schema") or {}
    covered = sum(1 for k in params if k in sch)
    return valid, covered, len(sch)


def evaluate_design_prompt(produce: Callable[[dict], dict],
                           golden: Optional[list[dict]] = None) -> dict:
    """对 design 提示词做结构评测。

    produce(snap) -> strategy_design 输出 dict（自由选执行器）。
    返回 {cases, validator_pass_rate, param_schema_coverage, pine_provision_rate}。
    """
    from .strategy_designer import design_executor_catalog
    catalog = design_executor_catalog()
    schemas = {n: catalog[n]["param_schema"] for n in catalog}
    ctx = {"param_schema": schemas["price_action"],
           "allowed_executors": tuple(catalog), "executor_schemas": schemas}
    golden = golden or GOLDEN_SET
    total = ok = 0
    cover_sum = cover_den = 0
    pine_count = 0
    for case in golden:
        try:
            out = produce(case["snap"])
        except Exception as e:  # noqa: BLE001
            log.warning("[evals] design produce 异常，跳过: %s", e)
            continue
        if not isinstance(out, dict):
            continue
        total += 1
        valid, covered, sch_len = _structural_pass("strategy_design", out, ctx)
        ok += int(valid)
        if sch_len:  # 覆盖率只统计"有 schema 可对照"的产出
            cover_sum += covered / sch_len
            cover_den += 1
        if out.get("pine_code"):
            pine_count += 1
    return {
        "cases": total,
        "validator_pass_rate": round(ok / total, 4) if total else 0.0,
        "param_schema_coverage": round(cover_sum / cover_den, 4) if cover_den else 0.0,
        "pine_provision_rate": round(pine_count / total, 4) if total else 0.0,
    }


def evaluate_iterate_prompt(produce: Callable[[dict], dict],
                            param_schema: Optional[dict] = None,
                            golden: Optional[list[dict]] = None) -> dict:
    """对 iterate 提示词做结构评测（默认用 dual_ma 的 schema 作对照）。

    produce(snap) -> strategy_iterate 输出 dict。
    返回 {cases, validator_pass_rate, param_schema_coverage}。
    """
    from strategies import _REGISTRY
    sch = param_schema or _REGISTRY["dual_ma"].param_schema
    ctx = {"param_schema": sch}
    golden = golden or GOLDEN_SET
    total = ok = 0
    cover_sum = cover_den = 0
    for case in golden:
        try:
            out = produce(case["snap"])
        except Exception as e:  # noqa: BLE001
            log.warning("[evals] iterate produce 异常，跳过: %s", e)
            continue
        if not isinstance(out, dict):
            continue
        total += 1
        valid, covered, sch_len = _structural_pass("strategy_iterate", out, ctx)
        ok += int(valid)
        if sch_len:
            cover_sum += covered / sch_len
            cover_den += 1
    return {
        "cases": total,
        "validator_pass_rate": round(ok / total, 4) if total else 0.0,
        "param_schema_coverage": round(cover_sum / cover_den, 4) if cover_den else 0.0,
    }
