"""AI 策略设计师：在既有执行器上生成可执行策略，注册前过信号密度硬门 + 过拟合检测。

过拟合检测照常判定（通过/疑似/严重过拟合/无法判定），但"严重过拟合"不再一票
否决注册——检测结论仅作为"禁止 AI 自动接管实盘"的依据（引擎参数验证门按
verdict 拦截），策略仍注册供用户查看检测报告后手动启用（P4-D7）。
"""
import asyncio
import json
import logging
import re
from typing import Any, Optional

from sqlalchemy import select

from core.bus import EventBus
from core.database import Database
from .client import AIClient, AICallError, AINotConfigured
from .prompts import design_strategy_messages

log = logging.getLogger(__name__)

# AI 设计可选的执行器目录。rl_adaptive / meta_controller 需要已训练的 model_path，
# AI 现场设计不出可用权重，放进目录只会产出跑不起来的策略。
DESIGN_EXECUTORS: tuple[str, ...] = ("price_action", "dual_ma", "factor_signal", "grid")

# 用户指定策略类型 → 执行器。此前所有类型都压到 price_action：趋势/均值回归想法被
# 强行翻译成"关键位突破/回调"，再叠多层 AND 过滤，实测近 1000 根只出 7 笔信号
# （dual_ma 18 笔、factor_signal 36 笔），这是"AI 策略没有信号"的主因。
STRATEGY_TYPE_EXECUTOR: dict[str, str] = {
    "trend": "dual_ma",
    "breakout": "price_action",
    "mean_reversion": "factor_signal",
    "grid": "grid",
}

_EXECUTOR_HINT = {
    "price_action": "价格行为：关键位(S/R)突破/回调，量能倍数与 RSI 超买过滤确认",
    "dual_ma": "双均线交叉：快线上穿慢线买入、下穿卖出，可配 ATR 止损与单笔风险预算仓位",
    "factor_signal": "因子信号：动量/MACD/RSI/波动突破/布林位置等因子（支持自定义表达式与 RL 组合因子）",
    "grid": "网格：相对基准价下跌买一档、上涨卖一档，适合区间震荡",
}


def design_executor_catalog() -> dict[str, dict]:
    """执行器 → {param_schema, description}：prompt 渲染与输出校验共用同一份事实源。"""
    from strategies import _REGISTRY
    return {name: {"param_schema": _REGISTRY[name].param_schema,
                   "description": _EXECUTOR_HINT.get(name, _REGISTRY[name].description)}
            for name in DESIGN_EXECUTORS}


def resolve_design_executor(strategy_type: str, ai_choice: str = "") -> str:
    """用户指定的策略类型固定执行器；未指定/自定义时采用 AI 的选择，兜底 price_action。"""
    pinned = STRATEGY_TYPE_EXECUTOR.get((strategy_type or "").strip())
    if pinned:
        return pinned
    return ai_choice if ai_choice in DESIGN_EXECUTORS else "price_action"


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", (name or "ai_strategy").lower().strip())
    return name[:48] or "ai_strategy"


# ============ 注册前信号密度硬门 ============
# AI 写"层层 AND 过滤"的策略，回测漂亮却长期不出信号（实测 price_action 默认参数
# 在最近 1000 根真实K线上 0 笔成交）。这类设计此前会直接注册、进策略库并被后续优化采纳，
# 相当于把"没有策略"当成"策略"传下去。注册前先跑一次真实回测数成交笔数。
_DENSITY_CANDLES = 1000
_DENSITY_MIN_TRADES = 5
_DENSITY_MAX_RETRIES = 2

# 每个执行器能提高信号频率的真实旋钮（只列 param_schema 里存在的键，
# 否则又是在教 AI 调一个不影响回测的死参数）
_DENSITY_KNOBS = {
    "price_action": "调小 breakout_pct（突破确认幅度）、把 volume_confirm 往下限 0.5 调、"
                    "mode 改用 pullback、必要时放宽 rsi_ob",
    "dual_ma": "缩短 fast_period 与 slow_period（均线越短、交叉越频繁）",
    "factor_signal": "放宽 buy_threshold / sell_threshold 的绝对值、换更灵敏的因子"
                     "（mom_ma10 比 mom_ma30 出信号多）、或切换 mode（trend/reversal）",
    "grid": "调小 grid_pct（网格越密、成交越多）",
}


def _candles_to_df(candles: list):
    """[[ts_ms, o, h, l, c, v], ...] → 回测引擎要的 OHLCV DataFrame（utc 索引）。"""
    import pandas as pd
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)


def _measure_signal_density(executor: str, params: dict, candles: list,
                            symbol: str, timeframe: str) -> dict:
    """在最近 _DENSITY_CANDLES 根K线上用执行器真实回测，数成交笔数（同步、CPU 密集）。

    数据不足时 available=False：没资格据此判定，也不该因此拒绝一个正常设计。
    """
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    window = list(candles[-_DENSITY_CANDLES:])
    base = {"candles": len(window), "required_candles": _DENSITY_CANDLES,
            "min_trades": _DENSITY_MIN_TRADES, "symbol": symbol, "timeframe": timeframe}
    if len(window) < _DENSITY_CANDLES:
        return {**base, "available": False, "passed": True, "trades": None,
                "reason": f"历史K线仅 {len(window)} 根（需 {_DENSITY_CANDLES} 根），跳过信号密度检测"}
    cfg = BacktestConfig(symbol=symbol, timeframe=timeframe, strategy_name=executor,
                         strategy_params=dict(params))
    metrics = (run_backtest_fast(_candles_to_df(window), cfg) or {}).get("metrics") or {}
    trades = int(metrics.get("total_trades") or 0)
    return {**base, "available": True, "passed": trades >= _DENSITY_MIN_TRADES,
            "trades": trades, "reason": ""}


async def _check_signal_density(executor: str, params: dict, snap: dict,
                                guard_candles: Optional[list] = None) -> dict:
    """信号密度门（异步壳）：回测放后台线程，任何异常都退化为"不拦截"。"""
    candles = guard_candles or snap.get("candles") or []
    symbol = snap.get("symbol", "BTC/USDT")
    timeframe = snap.get("timeframe", "1h")
    try:
        return await asyncio.to_thread(_measure_signal_density, executor, params,
                                       candles, symbol, timeframe)
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 信号密度检测异常（不拦截）: %s", e)
        return {"available": False, "passed": True, "trades": None,
                "candles": len(candles), "required_candles": _DENSITY_CANDLES,
                "min_trades": _DENSITY_MIN_TRADES, "reason": f"检测异常：{e}"}


def _density_feedback(executor: str, params: dict, density: dict) -> str:
    """密度门不达标 → 喂回 AI 的具体修正指令（带实测笔数与该执行器的真实旋钮）。"""
    return (
        f"上一版在执行器 {executor} 上，"
        f"最近 {density.get('candles')} 根 {density.get('timeframe')} K线真实回测"
        f"只成交 {density.get('trades')} 笔，低于注册门槛 {density.get('min_trades')} 笔，"
        "已被系统拒绝注册。\n"
        f"请针对该执行器提高信号频率：{_DENSITY_KNOBS.get(executor, '')}。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}\n"
        "减少 AND 叠加的过滤条件比新增条件更有效；保持逻辑与风控完整。")


# ============ 过拟合回炉（P4-D8）：检测报告喂回 AI 重设计 ============
# 过拟合检测照常判定（verdict/score/PBO 全不动），但"严重过拟合"不再直接放行入库，
# 而是把检测指标与修正方向作为 feedback 喂回 AI 重新设计（降自由度/收敛参数/简化逻辑），
# 最多重试 _OVERFIT_MAX_RETRIES 次；重试用尽仍过拟合才注册打标（P4-D7 兜底，人工可启用）。
_OVERFIT_MAX_RETRIES = 2


def _overfit_feedback(executor: str, params: dict, report: dict) -> str:
    """过拟合守卫不达标 → 喂回 AI 的具体修正指令（带检测指标，指向降自由度/收敛参数）。"""
    flags = (report.get("flags") or [])[:3]
    flag_txt = "；".join(str(f.get("msg", "")) for f in flags) or \
        "样本内收益明显好于样本外（衰减过大），参数过度贴合历史行情"
    return (
        f"上一版在执行器 {executor} 上过拟合检测未通过："
        f"样本内收益 {report.get('is_ret')} vs 样本外 {report.get('oos_ret')}，"
        f"衰减 {report.get('decay')}，PBO 过拟合概率 {report.get('pbo')}，"
        f"评分 {report.get('score')}（{report.get('verdict')}）。\n"
        f"检测提示：{flag_txt}\n"
        "请针对性修正（这是过拟合检测问题，不是信号密度问题）：\n"
        "1) 大幅降低参数自由度：参数取整、取常规值，不要在 schema 允许范围内挑极端值；\n"
        "2) 收敛参数范围到常见稳健区间（周期、阈值取中间值，避免贴边）；\n"
        "3) 简化逻辑：减少 AND 叠加条件，条件越少越难拟合噪声；\n"
        "4) 以样本外稳健性优先：宁可信号少一点、收益平一点，也要样本外不衰减。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")


async def _guard_ai_strategy(name: str, params: dict, snap: dict,
                             executor: str = "price_action",
                             guard_candles: Optional[list] = None) -> Optional[dict]:
    """AI 策略过拟合守卫：用历史数据跑前推验证 + PBO。

    - 数据源：guard_candles（引擎主动拉的足量真实K线）或快照K线。
      合成数据无法让 price_action 产生可靠交易，会导致守卫误判，
      故数据不足时返回"无法判定"而非拦截。
    - executor：AI 策略在执行器上回测（AI 策略本身尚未注册，直接用执行器类）
    返回过拟合报告 dict；无法判定时返回 None（不拦截）。
    """
    from backtest.overfit import OverfitConfig, detect_overfit

    try:
        candles = guard_candles or snap.get("candles") or []
        if len(candles) < 600:
            return {
                "verdict": "无法判定",
                "score": None,
                "data_source": "数据不足",
                "flags": [{"level": "info",
                           "msg": f"历史K线仅 {len(candles)} 根，不足 600 根，过拟合检测跳过。"
                                  "策略落地前建议先用足量历史数据跑过拟合检测。"}],
            }
        df = _candles_to_df(candles)
        timeframe = snap.get("timeframe", "1h")
        cfg = OverfitConfig(symbol=snap.get("symbol", "BTC/USDT"),
                            timeframe=timeframe, n_folds=4, fold_ratio=0.2, min_is_len=200)
        # 用执行器名回测（AI 策略尚未注册，无法用 AI 名）。
        # CSCV/PBO 计算量大（数百次回测），放后台线程避免阻塞主事件循环
        report = await asyncio.to_thread(detect_overfit, df, executor, params, cfg)
        return {
            "verdict": report.verdict, "score": report.score,
            "is_ret": report.is_ret, "oos_ret": report.oos_ret,
            "decay": report.decay, "pbo": report.pbo,
            "oos_sharpe": report.oos_sharpe, "stability": report.stability,
            "flags": report.flags, "n_folds": report.n_folds,
            "data_source": "真实K线",
        }
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 过拟合守卫跳过: %s", e)
        return None


class StrategyDesigner:
    def __init__(self, client: AIClient, db: Database, bus: EventBus) -> None:
        self._client = client
        self._db = db
        self._bus = bus

    async def _log_design_gate(self, name: str, spec: dict, density: dict) -> None:
        """未过门的设计落一条 OptimizationLog：被拒绝也要可追溯，不当作没发生过。"""
        from core.database import OptimizationLog
        try:
            async with self._db.session() as s:
                s.add(OptimizationLog(
                    kind="design_gate",
                    summary=f"[信号密度门拒绝] {name}（executor={spec.get('executor')}）"
                            f"近 {density.get('candles')} 根K线仅 {density.get('trades')} 笔，"
                            f"门槛 {_DENSITY_MIN_TRADES} 笔，已重试 {_DENSITY_MAX_RETRIES} 次",
                    suggestion=_DENSITY_KNOBS.get(spec.get("executor"), ""),
                    params_json=json.dumps(spec.get("params") or {}, ensure_ascii=False)))
                await s.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("[ai] 设计门日志写入失败 %s: %s", name, e)

    async def design(self, snap: dict, recent_trades: Optional[list] = None,
                     backtests: Optional[list] = None,
                     guard_candles: Optional[list] = None,
                     strategy_type: str = "",
                     custom_requirement: str = "") -> dict:
        """让 AI 基于行情快照（含 S/R 与价格行为）设计策略。返回策略规格 dict。

        backtests: 历史回测结果摘要，供 AI 参考（避免重蹈覆辙）。
        guard_candles: 用于过拟合守卫的足量历史K线（引擎传入）。
        strategy_type: 用户指定策略类型（trend/breakout/mean_reversion/grid/custom）。
        custom_requirement: 用户自定义设计要求（自由打字）。

        注册前两道门，都带"回炉重试"（P4-D8）：
        1) 信号密度门（最近 1000 根K线 ≥5 笔成交）：不达标把实测结果喂回 AI 重设计，
           最多 _DENSITY_MAX_RETRIES 次；仍不达标不注册（blocked_by_signal_density）。
        2) 过拟合守卫：verdict/score 照判（检测本身不动），但"严重过拟合"不再直接
           放行或一票否决——把检测指标（IS/OOS 收益、衰减、PBO、flags）喂回 AI
           重新设计（降参数自由度、收敛范围、简化逻辑），最多 _OVERFIT_MAX_RETRIES
           次；重试用尽仍过拟合才注册并打标 blocked_by_overfit（P4-D7 兜底：用户可
           手动启用，引擎自动接管门按 verdict 拦截）。
        """
        from strategies import register_dynamic, _REGISTRY

        # 策略类型固定执行器；未指定/custom 时让 AI 在目录内自选（前端选项即"默认（AI 自选）"）
        pinned = STRATEGY_TYPE_EXECUTOR.get((strategy_type or "").strip())
        allowed = (pinned,) if pinned else DESIGN_EXECUTORS
        catalog = design_executor_catalog()
        ctx = {
            "param_schema": _REGISTRY[allowed[0]].param_schema,
            "allowed_executors": tuple(allowed),
            "executor_schemas": {e: catalog[e]["param_schema"] for e in allowed},
        }
        spec: dict = {}
        density: dict = {}
        feedback = ""
        density_retries = 0
        overfit_retries = 0
        max_attempts = _DENSITY_MAX_RETRIES + _OVERFIT_MAX_RETRIES + 1
        for attempt in range(max_attempts):
            result = await self._client.chat_json_validated(
                design_strategy_messages(snap, recent_trades or [], backtests or [],
                                         strategy_type=strategy_type,
                                         custom_requirement=custom_requirement,
                                         executor_catalog={e: catalog[e] for e in allowed},
                                         feedback=feedback),
                feature="strategy_design",
                ctx=ctx)
            name = sanitize_name(result.get("name"))
            if name in _REGISTRY:
                name = "ai_" + name

            executor = resolve_design_executor(strategy_type, str(result.get("executor") or ""))
            strategy_cls = _REGISTRY[executor]
            # 按执行器 schema 校验/限制参数范围，保证可直接执行
            applied = strategy_cls().update_params(result.get("params") or {})

            # 提取 AI 生成的 Pine Script v5 代码（若存在，作为"训练好的交易代码"）
            pine_code = result.get("pine_code") or ""
            pine_note = ""
            if pine_code:
                # 关键要求：贴到图表必须画出买卖点。AI 常见产出只有下单语句、零标记，
                # 这里统一兜底补成交标记块（已自带双向标记的代码原样返回）
                from strategies.pine_utils import ensure_trade_markers
                pine_code = ensure_trade_markers(pine_code)
            else:
                pine_note = "AI 未生成 Pine 代码，本次策略无图表自动交易代码"
            # 仅 price_action 有与执行器逐参数同源的 input 模板：此时以 Pine 为源回读参数。
            # 其他执行器没有该模板，解析出的键属于 price_action，合并会污染本次参数。
            if pine_code and executor == "price_action":
                from strategies.pine_utils import parse_pine_params
                pine_params = parse_pine_params(pine_code)
                if pine_params:
                    applied = strategy_cls().update_params({**applied, **pine_params})  # Pine 优先

            spec = {
                "name": name,
                "title": result.get("title", "AI 设计策略"),
                "description": result.get("description", ""),
                "logic": result.get("logic", ""),
                "executor": executor,
                "params": applied,
                "risk_tips": result.get("risk_tips", []),
                "created_by": "ai",
                "pine_code": pine_code,  # 完整 Pine 代码，供前端直接展示
                "pine_note": pine_note,
                "design_attempts": attempt + 1,  # 两道门回炉了几版，界面据此说明
            }
            # ---- 门 1：注册前信号密度门（不出信号的策略等于没有策略）----
            density = await _check_signal_density(executor, applied, snap,
                                                  guard_candles=guard_candles)
            spec["signal_density"] = density
            if not density.get("passed"):
                if density_retries >= _DENSITY_MAX_RETRIES:
                    spec["blocked_by_signal_density"] = True
                    await self._log_design_gate(spec.get("name", ""), spec, density)
                    log.warning("[ai] 策略 %s 连续 %s 版未达信号密度门槛（近 %s 根K线需 ≥%s 笔），拒绝注册",
                                spec.get("name", ""), _DENSITY_MAX_RETRIES + 1,
                                density.get("candles"), _DENSITY_MIN_TRADES)
                    return spec
                density_retries += 1
                feedback = _density_feedback(executor, applied, density)
                log.warning("[ai] 策略 %s 信号密度不足（近 %s 根 %s 笔），回炉重设计（第 %s 次）",
                            name, density.get("candles"), density.get("trades"), density_retries)
                continue

            # ---- 门 2：过拟合守卫（P4-D8 回炉：检测照判，报告喂回 AI 重设计）----
            overfit_report = await _guard_ai_strategy(name, applied, snap, executor=executor,
                                                      guard_candles=guard_candles)
            if overfit_report:
                spec["overfit"] = overfit_report
            if overfit_report and overfit_report.get("verdict") == "无法判定":
                # 注册照准（证据不足不该算到策略头上），但要显式标出"未证明可用"：
                # 曾把"无法判定"当静默放行，AI 参数热更新据此接管实盘
                spec["overfit_inconclusive"] = True
                log.warning("[ai] 策略 %s 过拟合守卫无法判定（%s），允许注册但禁止自动接管实盘",
                            name, overfit_report.get("data_source", "证据不足"))
                break
            if overfit_report and overfit_report.get("verdict") == "严重过拟合":
                if overfit_retries < _OVERFIT_MAX_RETRIES:
                    # 回炉：把检测指标喂回 AI，让它降自由度/收敛参数/简化逻辑后重新设计
                    overfit_retries += 1
                    feedback = _overfit_feedback(executor, applied, overfit_report)
                    log.warning("[ai] 策略 %s 过拟合未通过（%s 分），回炉重设计（过拟合第 %s/%s 次）",
                                name, overfit_report.get("score"),
                                overfit_retries, _OVERFIT_MAX_RETRIES)
                    continue
                # 重试用尽 → P4-D7 兜底：仍注册，但打标（前端警告 + 引擎自动接管门拦截）
                spec["blocked_by_overfit"] = True
                log.warning("[ai] 策略 %s 过拟合回炉 %s 次仍未通过（%s 分），注册供人工启用（自动接管被拦）",
                            name, _OVERFIT_MAX_RETRIES, overfit_report.get("score"))
            break

        register_dynamic(name, spec)

        from core.database import AiStrategy
        async with self._db.session() as s:
            existing = (await s.execute(select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
            if existing:
                existing.spec_json = json.dumps(spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=name, spec_json=json.dumps(spec, ensure_ascii=False)))
            await s.commit()
        log.info("[ai] 新策略已设计并注册: %s", name)
        return spec