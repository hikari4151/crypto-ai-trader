"""AI 策略设计师：在既有执行器上生成可执行策略，注册前过信号密度硬门 + 过拟合检测。

过拟合检测照常判定（通过/疑似/严重过拟合/无法判定），但"严重过拟合"不再一票
否决注册——检测结论仅作为"禁止 AI 自动接管实盘"的依据（引擎参数验证门按
verdict 拦截），策略仍注册供用户查看检测报告后手动启用（P4-D7）。
"""
import asyncio
import json
import logging
import math
import re
from typing import Optional

from sqlalchemy import select

from core.bus import EventBus
from core.database import Database, _run_on_main
from .client import AIClient
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


# ============ L2：草稿（未证明）状态 ============
# 证据不足或检测出过拟合的产物，此前是"注册 + 打个标记"就完事——等于它们仍被当成
# 正常策略传给下一代：
#   · 会成为迭代链的父代（在它身上继续迭代，低交易/过拟合特征被继承放大）
#   · 会出现在可应用的策略列表里，和过了门的策略长得一样
# 现在改为独立状态 status="draft"（未证明）：
#   · 不进迭代链的父代选择（web/api/ai.py::iterate_strategy 直接拒绝）
#   · 不进引擎的 AI 参数热更新通道（engine/trading_engine.py::_validate_param_update）
#   · 策略列表单独输出 status/draft 供前端分区显示（strategies/__init__.py）
# 仍然注册入库：用户要能看到检测报告并手动启用（这是人工判断，不是系统背书）。
_DRAFT_STATUS = "draft"
_ACTIVE_STATUS = "active"


def draft_reason_for(spec: dict) -> str:
    """按 spec 上的检测标记推断草稿原因；返回空串表示"不是草稿"（= 过了门）。

    规则本体在 `strategies._draft_reason_from_flags`（策略注册层必须用同一条规则，
    而 strategies 是最底层模块、不能反向 import 本模块）——委托过去，避免两份规则漂移。
    """
    from strategies import _draft_reason_from_flags
    return _draft_reason_from_flags(spec)


def mark_status(spec: dict) -> dict:
    """给 spec 就地写入 status(=active/draft) 与 draft_reason，返回同一个 dict。"""
    reason = draft_reason_for(spec)
    spec["status"] = _DRAFT_STATUS if reason else _ACTIVE_STATUS
    if reason:
        spec["draft_reason"] = reason
    return spec


# ============ 注册前信号密度硬门 ============
# AI 写"层层 AND 过滤"的策略，回测漂亮却长期不出信号（实测 price_action 默认参数
# 在最近 1000 根真实K线上 0 笔成交）。这类设计此前会直接注册、进策略库并被后续优化采纳，
# 相当于把"没有策略"当成"策略"传下去。注册前先跑一次真实回测数成交笔数。
_DENSITY_CANDLES = 1000
# ---- L2（2026-09-12）：门槛从「绝对笔数」改为「归一化频率（笔/1000 根）」----
# 原判据是绝对 5 笔，与窗口长度脱钩：主窗口 1000 根 = 5 笔/1000 根（0.5%），
# 而备选长窗口 3000 根**仍只要 5 笔** → 同一条策略被稀释到 1.67 笔/1000 根照样放行。
# "交易次数很少"因此被制度化了：越长的窗口越容易蒙混过关。
#
# 实测（.optim/bughunt/probe_l2_density.py，BTC/USDT 1h 真实数据，各执行器默认参数）：
#   executor          1000根  笔/1000  |  3000根  笔/1000   频率是否稳定
#   dual_ma              18     18.0   |      53     17.7   稳定
#   factor_signal        36     36.0   |     112     37.3   稳定
#   price_action          7      7.0   |      11      3.7   不稳且极低
#   grid                  5      5.0   |      37     12.3   不稳
# 健康策略的笔/1000 跨窗口基本一致（dual_ma 18.0 vs 17.7）→ 归一化频率才是可比的尺子，
# 绝对笔数不是。阈值取 15（建议区间 15~20 的下沿）：dual_ma / factor_signal 默认参数
# 达标，price_action / grid 默认参数不达标——后者正是"回测漂亮、长期不出信号"那一类。
_DENSITY_MIN_TRADES_PER_1000 = 15
# 兼容旧引用：主窗口（_DENSITY_CANDLES 根）折算出的绝对笔数
_DENSITY_MIN_TRADES = max(1, math.ceil(_DENSITY_MIN_TRADES_PER_1000 * _DENSITY_CANDLES / 1000))
_DENSITY_MAX_RETRIES = 2
# 备选长窗口：主窗口（最近 1000 根）0 成交可能是窗口恰好横盘/波动不足（如网格
# grid_pct 超出窗口波动），同参数在更长历史上重测一次，避免误杀区间类设计
# （镜像迭代验证门的 validation_df_alt 思路，取值与引擎 GUARD_CANDLES 上限一致）。
# ⚠️ 该窗口必须用**同一个归一化门槛**判定，否则又会稀释成"长窗口只要 5 笔"。
_DENSITY_ALT_CANDLES = 3000


def _min_trades_for(n_candles: int) -> int:
    """窗口长度 n_candles 下，归一化密度门槛折算出的绝对笔数（向上取整）。

    `_min_trades_for(1000) == 15`、`_min_trades_for(3000) == 45`：同一策略在
    长短窗口上的达标判定等价。
    """
    return max(1, math.ceil(_DENSITY_MIN_TRADES_PER_1000 * max(0, n_candles) / 1000))


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


def _judge_density(trades: int, n_candles: int) -> dict:
    """在 n_candles 根窗口上按**归一化频率**判定信号密度。

    达标门槛 = `_min_trades_for(n_candles)`（1000 根 → 15 笔，3000 根 → 45 笔）。
    同时回传 `trades_per_1000`（实测频率）与 `min_trades_per_1000`（门槛频率），
    让反馈能直接说"你的频率是多少、要求多少"，而不是给一个随窗口漂移的绝对数。
    """
    need = _min_trades_for(n_candles)
    return {"trades": int(trades), "candles": int(n_candles), "min_trades": need,
            "trades_per_1000": round(trades / n_candles * 1000.0, 2) if n_candles else 0.0,
            "min_trades_per_1000": _DENSITY_MIN_TRADES_PER_1000,
            "passed": int(trades) >= need}


def _measure_signal_density(executor: str, params: dict, candles: list,
                            symbol: str, timeframe: str) -> dict:
    """在最近 _DENSITY_CANDLES 根K线上用执行器真实回测，数成交笔数（同步、CPU 密集）。

    判定用**归一化频率**（笔/1000 根），阈值 `_DENSITY_MIN_TRADES_PER_1000`：
    绝对笔数会随窗口长度漂移，曾让"3000 根只要 5 笔"（=1.67 笔/1000 根）也能过关。

    主窗口 0/少成交时，若有更长历史（>1000 根），用同参数在备选长窗口
    （最多 _DENSITY_ALT_CANDLES 根）重测一次：窗口恰好横盘（如 grid_pct 超出
    该窗口波动导致网格永不触发）不是策略没信号，不该据此误杀区间类设计；
    ⚠️ 备选窗口用**同一个归一化门槛**判定，否则又会退化成"长窗口稀释"。
    备选窗口仍不达标才判不过；其实测频率会留在 alt_* 字段里供反馈引用。

    数据不足时 available=False：没资格据此判定，也不该因此拒绝一个正常设计。
    """
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    window = list(candles[-_DENSITY_CANDLES:])
    n_win = len(window)
    base = {"required_candles": _DENSITY_CANDLES, "symbol": symbol, "timeframe": timeframe,
            "min_trades_per_1000": _DENSITY_MIN_TRADES_PER_1000}
    if n_win < _DENSITY_CANDLES:
        return {**base, "available": False, "passed": True, "trades": None,
                "candles": n_win, "min_trades": _min_trades_for(n_win),
                "trades_per_1000": None,
                "reason": f"历史K线仅 {n_win} 根（需 {_DENSITY_CANDLES} 根），跳过信号密度检测"}
    cfg = BacktestConfig(symbol=symbol, timeframe=timeframe, strategy_name=executor,
                         strategy_params=dict(params))
    metrics = (run_backtest_fast(_candles_to_df(window), cfg) or {}).get("metrics") or {}
    trades = int(metrics.get("total_trades") or 0)
    result = {**base, **_judge_density(trades, n_win), "available": True, "reason": ""}

    # 备选长窗口重测（仅主窗口未达标且有更多历史时；回测昂贵，正常达标不白跑）
    if not result["passed"] and len(candles) > _DENSITY_CANDLES:
        alt = list(candles[-_DENSITY_ALT_CANDLES:])
        if len(alt) > n_win:
            try:
                alt_metrics = (run_backtest_fast(_candles_to_df(alt), cfg)
                               or {}).get("metrics") or {}
                alt_j = _judge_density(int(alt_metrics.get("total_trades") or 0), len(alt))
                if alt_j["passed"]:
                    result = {**result, **alt_j, "passed": True,
                              "reason": (f"主窗口({n_win}根)成交 {trades} 笔"
                                         f"（{result['trades_per_1000']} 笔/1000 根）不达标，"
                                         f"备选长窗口({len(alt)}根)成交 {alt_j['trades']} 笔"
                                         f"（{alt_j['trades_per_1000']} 笔/1000 根）达标")}
                else:
                    # 备选窗口也没信号：如实记录实测频率，反馈里给定量依据
                    result = {**result, "alt_candles": len(alt),
                              "alt_trades": alt_j["trades"],
                              "alt_min_trades": alt_j["min_trades"],
                              "alt_trades_per_1000": alt_j["trades_per_1000"]}
            except Exception as e:  # noqa: BLE001
                log.warning("[ai] 密度门备选长窗口重测异常（沿用主窗口判定）: %s", e)
    return result


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
    """密度门不达标 → 喂回 AI 的具体修正指令（带实测频率与该执行器的真实旋钮）。

    用**归一化频率**（笔/1000 根）表述：绝对值会随窗口漂移，AI 无法据此判断
    自己离门槛还差多远（"1000 根 7 笔"和"3000 根 11 笔"哪个更差？7/1000 更差）。
    """
    need_rate = density.get("min_trades_per_1000", _DENSITY_MIN_TRADES_PER_1000)
    alt_txt = ""
    if density.get("alt_candles"):
        alt_txt = (f"；备选长窗口 {density.get('alt_candles')} 根上也只有 "
                   f"{density.get('alt_trades')} 笔 = "
                   f"{density.get('alt_trades_per_1000')} 笔/1000 根"
                   "，说明不是「窗口恰好横盘」，而是策略本身几乎不出信号")
    return (
        f"上一版在执行器 {executor} 上，"
        f"最近 {density.get('candles')} 根 {density.get('timeframe')} K线真实回测"
        f"只成交 {density.get('trades')} 笔（{density.get('trades_per_1000')} 笔/1000 根），"
        f"低于注册门槛 {need_rate} 笔/1000 根{alt_txt}，已被系统拒绝注册。\n"
        f"请针对该执行器提高信号频率：{_DENSITY_KNOBS.get(executor, '')}。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}\n"
        "减少 AND 叠加的过滤条件比新增条件更有效；保持逻辑与风控完整。")


# 过拟合回炉（P4-D8）：检测报告喂回 AI 重设计 ============
# 过拟合检测照常判定（verdict/score/PBO 全不动），但"严重过拟合"不再直接放行入库，
# 而是把检测指标与修正方向作为 feedback 喂回 AI 重新设计（降自由度/收敛参数/简化逻辑），
# 最多重试 _OVERFIT_MAX_RETRIES 次；重试用尽仍过拟合才注册打标（P4-D7 兜底，人工可启用）。
_OVERFIT_MAX_RETRIES = 2
# "无法判定"（真实K线上样本外交易 < _OVERFIT_MIN_OOS_TRADES 笔）也回炉一次：
# 统计证据不足不是"放行了事"的理由——让 AI 放宽过滤/降低触发门槛产生足够样本外
# 交易，使过拟合检测有据可判；重试仍不足才打标 overfit_inconclusive 放行（自动接管仍被拦）。
_OVERFIT_MIN_OOS_TRADES = 3   # 与 backtest.overfit._MIN_TRADES_FOR_VERDICT 对齐
_OVERFIT_INCONCLUSIVE_MAX_RETRIES = 1


def _overfit_feedback(executor: str, params: dict, report: dict) -> str:
    """过拟合守卫不达标 → 喂回 AI 的具体修正指令（带检测指标，指向降自由度/收敛参数）。

    区分两类失败：样本外成交偏少（avg<8 笔/折）时的负收益是成交噪声/行情不适配，
    应先提高成交频率（与信号密度门同向）；成交充足仍为负才是真参数过拟合，需收敛自由度。
    """
    flags = (report.get("flags") or [])[:3]
    flag_txt = "；".join(str(f.get("msg", "")) for f in flags) or \
        "样本内收益明显好于样本外（衰减过大），参数过度贴合历史行情"
    avg = report.get("avg_oos_trades")
    if isinstance(avg, (int, float)):
        if avg < 8:
            evidence_txt = (f"样本外平均成交仅 {avg:.1f} 笔/折——成交偏少时负收益多为"
                            "成交噪声/行情不适配，请同时放宽过滤提高每折成交频率"
                            "（与信号密度门同向），成交充足后再谈过拟合")
        else:
            evidence_txt = (f"样本外平均成交 {avg:.1f} 笔/折，成交充足仍为负收益——"
                            "参数过度贴合样本内行情，必须大幅收敛参数自由度")
    else:
        evidence_txt = "样本外成交情况未知"
    return (
        f"上一版在执行器 {executor} 上过拟合检测未通过："
        f"样本内收益 {report.get('is_ret')} vs 样本外 {report.get('oos_ret')}，"
        f"衰减 {report.get('decay')}，PBO 过拟合概率 {report.get('pbo')}，"
        f"评分 {report.get('score')}（{report.get('verdict')}）。\n"
        f"检测提示：{flag_txt}\n"
        f"{evidence_txt}。\n"
        "请针对性修正（这是过拟合检测问题，不是信号密度问题）：\n"
        "1) 大幅降低参数自由度：参数取整、取常规值，不要在 schema 允许范围内挑极端值；\n"
        "2) 收敛参数范围到常见稳健区间（周期、阈值取中间值，避免贴边）；\n"
        "3) 简化逻辑：减少 AND 叠加条件，条件越少越难拟合噪声；\n"
        "4) 以样本外稳健性优先：宁可信号少一点、收益平一点，也要样本外不衰减。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")


def _inconclusive_feedback(executor: str, params: dict, report: dict) -> str:
    """过拟合守卫"无法判定"（真实K线上样本外交易不足）→ 喂回 AI 的具体修正指令。

    统计证据不足不是放行的理由：让 AI 放宽过滤/降低触发门槛，使样本外能产生
    足够交易，过拟合检测才有据可判；否则策略只能打标"未证明可用"、禁止自动接管。
    """
    from backtest.overfit import _TRADES_EVIDENCE_OK
    return (
        f"上一版在执行器 {executor} 上过拟合检测无法判定："
        f"样本外平均交易仅 {report.get('avg_oos_trades', 0)} 笔"
        f"（低于 {_TRADES_EVIDENCE_OK} 笔证据线，统计证据不足），"
        "策略只能被标为「未证明可用」、禁止自动接管实盘。\n"
        f"请放宽过滤条件或降低触发门槛，让样本外能产生足够成交："
        f"{_DENSITY_KNOBS.get(executor, '')}。\n"
        "宁可信号多一点、收益平一点，也要让检测有据可判；"
        "这是证据充分性问题，不是要求你追求高收益。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")


async def _guard_ai_strategy(name: str, params: dict, snap: dict,
                             executor: str = "price_action",
                             guard_candles: Optional[list] = None,
                             prior_trials: int = 0) -> Optional[dict]:
    """AI 策略过拟合守卫：用历史数据跑前推验证 + PBO。

    - 数据源：guard_candles（引擎主动拉的足量真实K线）或快照K线。
      合成数据无法让 price_action 产生可靠交易，会导致守卫误判，
      故数据不足时返回"无法判定"而非拦截。
    - executor：AI 策略在执行器上回测（AI 策略本身尚未注册，直接用执行器类）
    - prior_trials：该策略此前已经"试过"的配置数（AI 回炉尝试次数 × 候选数、
      迭代链累积代次等）。进 DSR 的多重检验校正基数——原实现把它固定成候选数(≈9)，
      与真实搜索规模脱钩，会让 DSR 系统性偏乐观。给不出确数时给保守下限。
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
        # CSCV/PBO 计算量大（数百次回测），放后台线程避免阻塞主事件循环。
        # neighbor_params=None → 检测器内部按 param_schema 全范围生成固定网格候选。
        report = await asyncio.to_thread(detect_overfit, df, executor, params, cfg,
                                         prior_trials=prior_trials)
        avg_oos_trades = (round(sum(f.oos_trades for f in report.folds) / len(report.folds), 2)
                          if report.folds else 0.0)
        return {
            "verdict": report.verdict, "score": report.score,
            "is_ret": report.is_ret, "oos_ret": report.oos_ret,
            "decay": report.decay, "pbo": report.pbo,
            "oos_sharpe": report.oos_sharpe, "stability": report.stability,
            "flags": report.flags, "n_folds": report.n_folds,
            "avg_oos_trades": avg_oos_trades,
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

        async def _persist() -> None:
            async with self._db.session() as s:
                s.add(OptimizationLog(
                    kind="design_gate",
                    summary=f"[信号密度门拒绝] {name}（executor={spec.get('executor')}）"
                            f"近 {density.get('candles')} 根K线仅 {density.get('trades')} 笔"
                            f"（{density.get('trades_per_1000')} 笔/1000 根），"
                            f"门槛 "
                            f"{density.get('min_trades_per_1000', _DENSITY_MIN_TRADES_PER_1000)}"
                            f" 笔/1000 根（该窗口折算 {density.get('min_trades')} 笔），"
                            f"已重试 {_DENSITY_MAX_RETRIES} 次",
                    suggestion=_DENSITY_KNOBS.get(spec.get("executor"), ""),
                    params_json=json.dumps(spec.get("params") or {}, ensure_ascii=False)))
                await s.commit()
        try:
            # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
            await _run_on_main(self._db, _persist)
        except Exception as e:  # noqa: BLE001
            log.warning("[ai] 设计门日志写入失败 %s: %s", name, e)

    async def design(self, snap: dict, recent_trades: Optional[list] = None,
                     backtests: Optional[list] = None,
                     guard_candles: Optional[list] = None,
                     strategy_type: str = "",
                     custom_requirement: str = "",
                     baseline_strategy: Optional[str] = None,
                     baseline_params: Optional[dict] = None) -> dict:
        """让 AI 基于行情快照（含 S/R 与价格行为）设计策略。返回策略规格 dict。

        backtests: 历史回测结果摘要，供 AI 参考（避免重蹈覆辙）。
        guard_candles: 用于过拟合守卫的足量历史K线（引擎传入）。
        strategy_type: 用户指定策略类型（trend/breakout/mean_reversion/grid/custom）。
        custom_requirement: 用户自定义设计要求（自由打字）。
        baseline_strategy/baseline_params: 引擎当前活动策略（名+参数），注册前
            在 guard 数据上跑一次同数据对比回测，附加 baseline_comparison，
            让用户看到"新设计 vs 当前正在跑的策略"谁更好，而不是只看孤立的门。

        注册前两道门，都带"回炉重试"（P4-D8），且两门的反馈各自累积注入
        （此前共用单个 feedback 字符串：修好密度又触发过拟合时，过拟合反馈会
        覆盖密度反馈，AI 来回震荡、烧掉全部尝试预算）：
        1) 信号密度门（**归一化频率** ≥15 笔/1000 根；主窗口不达标时用同一门槛在
           备选长窗口重测）：不达标把实测频率喂回 AI 重设计，最多 _DENSITY_MAX_RETRIES
           次；仍不达标不注册（blocked_by_signal_density）。
        2) 过拟合守卫：verdict/score 照判（检测本身不动），但"严重过拟合"不再直接
           放行或一票否决——把检测指标（IS/OOS 收益、衰减、PBO、flags）喂回 AI
           重新设计（降参数自由度、收敛范围、简化逻辑），最多 _OVERFIT_MAX_RETRIES
           次；重试用尽仍过拟合才注册并打标 blocked_by_overfit（P4-D7 兜底：用户可
           手动启用，引擎自动接管门按 verdict 拦截）。
           另：真实K线上样本外交易不足（< _OVERFIT_MIN_OOS_TRADES 笔）导致的
           "无法判定"也回炉一次（让 AI 放宽过滤产生足够样本外交易），重试仍不足
           才打标 overfit_inconclusive 放行（自动接管仍被拦）。
        L2（2026-09-12）：上述两种"未证明"的产物登录时带 status="draft"，
        不参与迭代链父代选择、不进引擎 AI 热更新通道，只在策略库里供人工审阅启用。
        """
        from strategies import register_dynamic, _REGISTRY, _DYNAMIC

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
        feedback_parts: list[str] = []   # 密度门 + 过拟合门 + 无法判定门各自累积，互不覆盖
        density_retries = 0
        overfit_retries = 0
        inconclusive_retries = 0
        max_attempts = (_DENSITY_MAX_RETRIES + _OVERFIT_MAX_RETRIES
                        + _OVERFIT_INCONCLUSIVE_MAX_RETRIES + 1)
        for attempt in range(max_attempts):
            result = await self._client.chat_json_validated(
                design_strategy_messages(snap, recent_trades or [], backtests or [],
                                         strategy_type=strategy_type,
                                         custom_requirement=custom_requirement,
                                         executor_catalog={e: catalog[e] for e in allowed},
                                         feedback="\n\n".join(feedback_parts)),
                feature="strategy_design",
                ctx=ctx)
            # 重名护栏：内置名与已注册动态策略名都避让（此前只查 _REGISTRY，
            # 与 _DYNAMIC 重名时 register_dynamic 会静默覆盖既有 AI 策略）
            name = sanitize_name(result.get("name"))
            candidate = name
            clash_n = 0
            while candidate in _REGISTRY or candidate in _DYNAMIC:
                clash_n += 1
                suffix = f"_v{clash_n}"
                candidate = name[:48 - len(suffix)] + suffix
            name = candidate

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

            # Pine ↔ params 一致性预检（只报告不阻塞）：AI 手写 Pine 常漏 input 声明
            # 或套用别的执行器模板，图上买卖点与本地参数脱节——标记出来供前端提示
            pine_consistency = None
            if pine_code:
                from strategies.pine_utils import check_pine_consistency
                pine_consistency = check_pine_consistency(pine_code, executor, applied)
                if not pine_consistency["ok"] and not pine_note:
                    pine_note = ("Pine 一致性缺口："
                                 + "；".join(pine_consistency["missing"][:4])
                                 + "——图表代码可能未忠实反映全部参数")

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
                "pine_consistency": pine_consistency,
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
                    log.warning("[ai] 策略 %s 连续 %s 版未达信号密度门槛（近 %s 根K线实测 %s 笔"
                                "= %s 笔/1000 根，需 ≥%s 笔/1000 根），拒绝注册",
                                spec.get("name", ""), _DENSITY_MAX_RETRIES + 1,
                                density.get("candles"), density.get("trades"),
                                density.get("trades_per_1000"),
                                density.get("min_trades_per_1000",
                                             _DENSITY_MIN_TRADES_PER_1000))
                    return spec
                density_retries += 1
                feedback_parts.append(_density_feedback(executor, applied, density))
                log.warning("[ai] 策略 %s 信号密度不足（近 %s 根 %s 笔 = %s 笔/1000 根），"
                            "回炉重设计（第 %s 次）",
                            name, density.get("candles"), density.get("trades"),
                            density.get("trades_per_1000"), density_retries)
                continue

            # ---- 门 2：过拟合守卫（P4-D8 回炉：检测照判，报告喂回 AI 重设计）----
            overfit_report = await _guard_ai_strategy(name, applied, snap, executor=executor,
                                                      guard_candles=guard_candles,
                                                      # 多重检验校正：本次设计已消耗的 AI 尝试数
                                                      prior_trials=attempt)
            if overfit_report:
                spec["overfit"] = overfit_report
            if overfit_report and overfit_report.get("verdict") == "无法判定":
                # 真实K线上样本外交易不足：证据不足不该判策略有罪，但也不是放行了事——
                # 回炉一次让 AI 放宽过滤产生足够样本外交易，检测才有据可判
                data_src = str(overfit_report.get("data_source") or "")
                avg_trades = float(overfit_report.get("avg_oos_trades") or 0)
                if (data_src == "真实K线" and avg_trades < _OVERFIT_MIN_OOS_TRADES
                        and inconclusive_retries < _OVERFIT_INCONCLUSIVE_MAX_RETRIES):
                    inconclusive_retries += 1
                    feedback_parts.append(_inconclusive_feedback(executor, applied, overfit_report))
                    log.warning("[ai] 策略 %s 过拟合无法判定（样本外 %s 笔不足），回炉放宽触发门槛（第 %s 次）",
                                name, avg_trades, inconclusive_retries)
                    continue
                # 仍不足/数据不足（非策略责任）→ 注册照准，但显式标出"未证明可用"：
                # 曾把"无法判定"当静默放行，AI 参数热更新据此接管实盘
                spec["overfit_inconclusive"] = True
                log.warning("[ai] 策略 %s 过拟合守卫无法判定（%s），允许注册但禁止自动接管实盘",
                            name, overfit_report.get("data_source", "证据不足"))
                break
            if overfit_report and overfit_report.get("verdict") == "严重过拟合":
                if overfit_retries < _OVERFIT_MAX_RETRIES:
                    # 回炉：把检测指标喂回 AI，让它降自由度/收敛参数/简化逻辑后重新设计
                    overfit_retries += 1
                    feedback_parts.append(_overfit_feedback(executor, applied, overfit_report))
                    log.warning("[ai] 策略 %s 过拟合未通过（%s 分），回炉重设计（过拟合第 %s/%s 次）",
                                name, overfit_report.get("score"),
                                overfit_retries, _OVERFIT_MAX_RETRIES)
                    continue
                # 重试用尽 → P4-D7 兜底：仍注册，但打标（前端警告 + 引擎自动接管门拦截）
                spec["blocked_by_overfit"] = True
                log.warning("[ai] 策略 %s 过拟合回炉 %s 次仍未通过（%s 分），注册供人工启用（自动接管被拦）",
                            name, _OVERFIT_MAX_RETRIES, overfit_report.get("score"))
            break

        # ---- 与引擎当前策略的同数据对比（让"新设计 vs 正在跑的"可比较）----
        if (baseline_strategy and baseline_params and guard_candles
                and len(guard_candles) >= 200):
            try:
                from backtest.engine import BacktestConfig
                from backtest.fast_engine import run_backtest_fast
                df = _candles_to_df(list(guard_candles))
                new_bt = (run_backtest_fast(df, BacktestConfig(
                    symbol=snap.get("symbol", "BTC/USDT"), timeframe=snap.get("timeframe", "1h"),
                    strategy_name=executor, strategy_params=applied)) or {}).get("metrics") or {}
                base_bt = (run_backtest_fast(df, BacktestConfig(
                    symbol=snap.get("symbol", "BTC/USDT"), timeframe=snap.get("timeframe", "1h"),
                    strategy_name=baseline_strategy, strategy_params=baseline_params))
                    or {}).get("metrics") or {}
                spec["baseline_comparison"] = {
                    "base": {"total_return": base_bt.get("total_return"),
                             "sharpe": base_bt.get("sharpe"),
                             "max_drawdown": base_bt.get("max_drawdown"),
                             "win_rate": base_bt.get("win_rate"),
                             "total_trades": int(base_bt.get("total_trades") or 0)},
                    "new": {"total_return": new_bt.get("total_return"),
                            "sharpe": new_bt.get("sharpe"),
                            "max_drawdown": new_bt.get("max_drawdown"),
                            "win_rate": new_bt.get("win_rate"),
                            "total_trades": int(new_bt.get("total_trades") or 0)},
                    "note": f"同数据（{len(guard_candles)} 根）对比当前引擎策略 {baseline_strategy}",
                }
            except Exception as e:  # noqa: BLE001
                log.warning("[ai] 设计基线对比回测失败（不影响注册）: %s", e)

        # L2：按门的结果定 status（draft = 未证明：过拟合未过 / 证据不足）。
        # 注册仍然发生（用户可见可手动启用），但草稿不再参与迭代链父代选择，
        # 也不再被引擎的 AI 热更新通道采用。
        mark_status(spec)
        register_dynamic(name, spec)

        async def _persist_design() -> None:
            from core.database import AiStrategy
            async with self._db.session() as s:
                existing = (await s.execute(select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
                if existing:
                    existing.spec_json = json.dumps(spec, ensure_ascii=False)
                else:
                    s.add(AiStrategy(name=name, spec_json=json.dumps(spec, ensure_ascii=False)))
                await s.commit()
        # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
        await _run_on_main(self._db, _persist_design)
        log.info("[ai] 新策略已设计并注册: %s", name)
        return spec