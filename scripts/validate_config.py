"""配置 schema 校验：启动前/CI 检查 config/config.yaml（不侵入 settings.py）。

目的：配置写错时**快速失败**而不是跑起来才发现（如 default_timeframe 写成
"1H"、proxy_url 填了不可解析地址、数值写成字符串）。

用法：
  python scripts/validate_config.py                  # 校验，0=通过 1=有问题
  python scripts/validate_config.py --strict         # 未知键也报错（默认仅警告）
  python scripts/validate_config.py --show-schema    # 打印内置 schema

设计：独立于 config/settings.py（该文件可能被并行工作流编辑），schema 在此
内置。校验点：类型 / 数值范围 / 枚举 / 常见格式。
"""
import argparse
import json
import sys
from pathlib import Path

import yaml  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "config.yaml"

# ---- 内置 schema：字段 -> (期望类型, 校验规则 dict) ----
# 规则: min/max 数值范围；enum 允许值；pattern 正则（字符串）
SCHEMA: dict[str, tuple[type, dict]] = {
    "default_exchange": (str, {"enum": ["binance", "okx", "bybit", "bitget"]}),
    "default_symbol": (str, {"pattern": r"^[A-Z0-9]+/[A-Z0-9]+$"}),
    "default_timeframe": (str, {"enum": ["1m", "5m", "15m", "1h", "4h", "1d"]}),
    "default_strategy": (str, {}),
    "default_start_cash": (float, {"min": 1.0}),
    "paper_slippage": (float, {"min": 0.0, "max": 0.1}),
    "trade_on_open": (bool, {}),
    "trading_auto_start": (bool, {}),
    "protective_stop_enabled": (bool, {}),
    "protective_stop_default_pct": (float, {"min": 0.0, "max": 0.5}),
    "protective_stop_buffer_pct": (float, {"min": 0.0, "max": 0.1}),
    "max_stale_seconds": (int, {"min": 0}),
    "ai_market_analysis_interval": (int, {"min": 1}),
    "ai_optimize_interval": (int, {"min": 1}),
    "ai_review_interval": (int, {"min": 1}),
    "ai_optimize_validate": (bool, {}),
    "ai_optimize_validation_bars": (int, {"min": 100}),
    "proxy_url": (str, {}),
    "web_host": (str, {}),
    "web_port": (int, {"min": 1, "max": 65535}),
    "log_level": (str, {"enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}),
    "evolve_enabled": (bool, {}),
    "evolve_rolling_window": (int, {"min": 100}),
    "evolve_factor_miner_interval": (int, {"min": 1}),
    "evolve_strategy_drl_interval": (int, {"min": 1}),
    "evolve_strategy_drl_episodes": (int, {"min": 1}),
    "evolve_strategy_drl_n_episodes": (int, {"min": 1}),
    "evolve_strategy_drl_ppo_epochs": (int, {"min": 1}),
    "evolve_factor_miner_episodes": (int, {"min": 1}),
    "evolve_gap01_folds": (int, {"min": 1, "max": 10}),
    "evolve_gap01_bars": (int, {"min": 100}),
    "evolve_gap01_cost_multiplier": (float, {"min": 1.0}),
    "evolve_gap01_min_trades": (int, {"min": 1}),
    "evolve_gap01_fail_reset_after": (int, {"min": 1}),
    "evolve_gap01_stable_pass_required": (int, {"min": 1}),
    "evolve_strategy_drl_walkforward_folds": (int, {"min": 1}),
    "evolve_strategy_drl_walkforward_bars": (int, {"min": 100}),
    "evolve_walkforward_min_trades": (int, {"min": 1}),
    # ---- 2026-09-21~24 本会话新增键 ----
    "evolve_strategy_drl_longwindow_obs": (bool, {}),
    "evolve_strategy_drl_longwindow_30d_bars": (int, {"min": 100}),
    "evolve_strategy_drl_longwindow_90d_bars": (int, {"min": 100}),
    "evolve_strategy_drl_longwindow_every_n": (int, {"min": 1}),
    "evolve_strategy_drl_deploy_requires_longwindow": (bool, {}),
    "evolve_strategy_drl_deploy_longwindow_90d": (bool, {}),
    "evolve_strategy_drl_min_trade_zone": (float, {"min": 0.0, "max": 1.0}),
    "evolve_strategy_drl_state_window": (int, {"min": 1, "max": 16}),
    "evolve_strategy_drl_replay_capacity": (int, {"min": 0}),
    "evolve_strategy_drl_replay_mix_ratio": (float, {"min": 0.0, "max": 1.0}),
    "evolve_strategy_drl_multi_symbol": (bool, {}),
    "evolve_strategy_drl_multi_mix_ratio": (float, {"min": 0.0, "max": 1.0}),
    "signal_math_ev_gate": (bool, {}),
    "signal_math_ev_min": (float, {}),
    # ---- 既有键补录（消除 WARN）----
    "evolve_symbols": (list, {}),
    "evolve_timeframe": (str, {"enum": ["1m", "5m", "15m", "1h", "4h", "1d"]}),
    "evolve_max_versions": (int, {"min": 1}),
    "evolve_gap01_obs": (bool, {}),
    "evolve_rollback_threshold": (float, {}),
    "evolve_strategy_drl_hidden": (list, {}),
    "evolve_strategy_drl_reset_after_rejects": (int, {"min": 1}),
    "evolve_strategy_drl_walkforward_obs": (bool, {}),
}


def _check_value(field: str, value, expected: type, rules: dict) -> list[str]:
    """校验单个值，返回错误列表。"""
    errs = []
    if value is None:
        return errs  # 缺省/空值由 settings 默认兜底，不强制
    # 类型：bool 是 int 子类，须先判 bool
    if expected is bool:
        if not isinstance(value, bool):
            errs.append(f"{field}: 期望 bool，实际 {type(value).__name__} ({value!r})")
            return errs
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            errs.append(f"{field}: 期望 int，实际 {type(value).__name__} ({value!r})")
            return errs
    elif not isinstance(value, expected):
        errs.append(f"{field}: 期望 {expected.__name__}，实际 {type(value).__name__} ({value!r})")
        return errs
    # 数值范围
    if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
        errs.append(f"{field}: 低于最小值 {rules['min']}（实际 {value}）")
    if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
        errs.append(f"{field}: 超过最大值 {rules['max']}（实际 {value}）")
    # 枚举
    if "enum" in rules and value not in rules["enum"]:
        errs.append(f"{field}: 不在允许值 {rules['enum']}（实际 {value!r}）")
    # 正则
    if "pattern" in rules and isinstance(value, str):
        import re
        if not re.match(rules["pattern"], value):
            errs.append(f"{field}: 格式不符 {rules['pattern']}（实际 {value!r}）")
    return errs


def validate(cfg_path: Path, strict: bool = False) -> list[str]:
    """校验 config.yaml，返回错误列表（空=通过）。"""
    if not cfg_path.exists():
        return [f"配置文件不存在: {cfg_path}"]
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        return [f"config.yaml 解析失败: {e}"]
    if not isinstance(data, dict):
        return ["config.yaml 顶层应为映射（dict）"]

    errs = []
    for field, (expected, rules) in SCHEMA.items():
        if field in data:
            errs += _check_value(field, data[field], expected, rules)
    # 未知键：strict 报错，否则警告（提示但不阻断，向前兼容新增键）
    unknown = [k for k in data if k not in SCHEMA]
    for k in unknown:
        msg = f"{k}: 未知键（不在内置 schema，请确认拼写）"
        errs.append(msg) if strict else print(f"[WARN] {msg}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="未知键也报错（默认仅警告）")
    ap.add_argument("--show-schema", action="store_true", help="打印内置 schema")
    args = ap.parse_args()
    if args.show_schema:
        print(json.dumps({k: [t.__name__, r] for k, (t, r) in SCHEMA.items()},
                         ensure_ascii=False, indent=1))
        return 0
    errs = validate(CONFIG, strict=args.strict)
    if errs:
        for e in errs:
            print(f"[FAIL] {e}")
        print(f"配置校验失败：{len(errs)} 处问题（{CONFIG}）")
        return 1
    print(f"[OK] 配置校验通过（{CONFIG}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
