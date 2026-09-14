"""AI 策略迭代：重新思考现有策略，批判性反思并给出改进版本。

与"参数优化"的区别：
- 参数优化：在原有框架内微调参数
- 策略迭代：重新审视策略逻辑本身（入场/出场/风控假设），可能调整参数+逻辑思路，
  产出改进版说明，供用户选择是否应用
"""
import json
import logging
import math
import threading
from typing import Optional

from core.database import Database, _run_on_main
from strategies.base import Strategy
from .client import AIClient, AICallError, AINotConfigured
from .prompts import iteration_messages
from .strategy_designer import (
    _DENSITY_KNOBS, _OVERFIT_MAX_RETRIES, _guard_ai_strategy, _overfit_feedback,
    _inconclusive_feedback, _OVERFIT_MIN_OOS_TRADES, _OVERFIT_INCONCLUSIVE_MAX_RETRIES,
    mark_status,
)

log = logging.getLogger(__name__)

# run25 R2：迭代序号分配锁——_next_version 的"读 KV → 自增 → 冲突避让 → 写回"
# 非原子，_AI_SEM=2 允许两个迭代任务并发（各自独立事件循环，asyncio.Lock 不
# 共享），曾并发重号并静默覆盖一代完整迭代产物。用 threading.Lock 串行化
# 该临界区（kv 操作毫秒级，阻塞可接受）。
_ITER_VERSION_LOCK = threading.Lock()

# 验证门回炉（与设计侧两道门对称）：同数据回测未超越上一代时，把实测新旧指标
# 喂回 AI 重迭代，最多 _VALIDATION_MAX_RETRIES 次；仍不达标才拒绝注册。
_VALIDATION_MAX_RETRIES = 2
# 验证门判定（二.1 多指标门槛）：收益必须实打实提升 ≥1 个百分点（绝对值）、
# 回撤不显著恶化、夏普不被拉垮、且有足够的真实成交（防单笔运气单过关）。
_ITER_MIN_GAIN = 0.01          # 总收益绝对值提升下限（在**选优段**上衡量）
# L2（2026-09-12）：留出段最少成交笔数 5 → 10。
# 留出段约 1200~1600 根，按设计侧新的信号密度门槛（15 笔/1000 根）折算，
# 一个"够密"的策略在这一段本该有 ~20 笔成交；只要 5 笔时，
# "5 笔里 1 笔运气单"就足以把 total_return 抬过 +1pp 的门槛。
# ⚠️ 探针实测（.optim/bughunt/probe_l2_density.py）：对**成交充足**的候选，
# 这条从 5 提到 10/15 完全没有过滤效果（24/24 全过，见 out_l2_density.txt 拟议④⑤）
# ——它只在成交稀少的候选上起作用，而那正是它要防的场景。
# 所以这是一条**便宜的保险**，不要指望它提升选优质量。
_ITER_MIN_TRADES = 10          # 新策略最少成交笔数（在**留出段**上衡量）
_ITER_SHARPE_TOLERANCE = 0.1   # 夏普允许下降幅度

# ---- L1：留出集（OOS）判定，取代「同窗口链式超越」----
# 校验窗口按时间切两段：前 _ITER_IS_FRACTION 为「选优段」(IS)，其后为「留出段」(OOS)。
# IS 用于选优与反馈（可以告诉 AI），OOS 只用于判定、**绝不进 prompt**。
#
# 为什么必须改（2026-09-12 实测，见 .optim/bughunt/REPORT_overfit_source.md）：
# 原判据是「新收益 ≥ 上一代收益 + 1pp」且新旧都跑在同一个窗口上，配合链式
# v1→v2→…→v18，等价于**对该窗口做梯度上升**。在真实 BTC/USDT 1h 上复现：
#   v0  W1(门可见) -2.06% / W2(门不可见) -3.21%
#   v1  W1 +10.81% / W2 -10.33%   ← 门"接受"
#   v2  W1 +17.96% / W2 -10.82%   ← 门"接受"
# 门的优化目标 +20.0pp，真实泛化 -7.6pp（兑现率 -38%）——门在批量制造假赢家。
# 更糟的是它无法分辨：同构实验在 factor_signal 上恰好同向（兑现率 122%），
# 因为门从来看不到留出数据。改成留出集判定 + 锚定固定基线后，
# 链不再沿"上一代"逐级抬升，判定也变成真正的泛化测试。
_ITER_IS_FRACTION = 0.6        # 选优段占比（其余为留出段）
_ITER_OOS_TOLERANCE = 0.005    # 留出段允许的最大退化（锚：不劣于固定基线与上一代）
_ITER_DD_TOLERANCE = 1.3       # 留出段回撤容忍倍数
_ITER_MIN_SEGMENT = 200        # 任一段不足此根数则降级（不判定）

# ---- L2：为什么**没有**把「每笔收益 t 统计量」加成硬门槛（负结果，录以备查）----
# 原计划（L2 第 6 条）是"improved 加入样本量感知指标（每笔收益 t > 1）"。
# 动手前先做了决策级实验（.optim/bughunt/probe_l2_density.py E2）：真实 BTC/USDT 1h
# 数据、v0 邻域 24 个候选、以**两个门都看不到的第三段 holdout** 当裁判：
#   · 与 holdout 的秩相关：留出段裸收益差 ρ=+0.337(dual_ma) / -0.596(factor_signal)；
#     每笔 t 统计量 ρ=+0.305 / -0.506 —— **t 并不比裸收益更会挑泛化好的候选**；
#   · 阈值 t≥0 或 t≥1：24/24 全被挡（该留出段上每笔 t 全为负），门直接变成一堵墙；
#   · "t 相对基线差 ≥ -0.5"虽有鉴别力，但通过 20/24，比现判据（通过 11/24）更松。
# 结论：加 t 硬门槛无鉴别力增益、却让门不可达 → **只作为诊断量输出**
# （diagnostics.oos_new_t_stat / oos_base_t_stat），用于解释"这笔收益为什么不可信"，
# 不参与 improved 判定。留出段判定要真正有分辨力，靠更长历史（≥1 年）而非再叠判据。


def _validation_feedback(executor: str, params: dict, comparison: dict,
                         no_trade: bool, oos_rejected: bool = False) -> str:
    """验证门不达标 → 喂回 AI 的具体修正指令（带实测新旧指标，指向参数间距/逻辑）。

    ⚠️ 反馈纪律：只允许引用 comparison["old"]/["new"]（= **选优段 IS** 指标）。
    留出段（OOS）的数字绝不能出现在这里——一旦 AI 看到 OOS，它就会转去拟合 OOS，
    留出集退化成第二个训练集，问题只是换了个位置。
    oos_rejected=True 表示"在 IS 上确实提升了、但被留出段否决"，
    此时必须据实说明：拿 IS 数字说"未超越上一代"是假话，会让 AI 朝错误方向调参。
    """
    if oos_rejected:
        return (
            "上一版迭代在**可见样本**上确实提升了，但被留出段（你没见过的数据）否决——"
            "说明这次改动拟合的是可见段的噪声，不是可复现的规律。\n"
            "按以下方向重做（这是稳健性问题，不是收益不够）：\n"
            "1) 降自由度：参数取整、取常规值，不要挑 schema 边界上的极端值；\n"
            "2) 减少 AND 叠加的过滤条件——条件越多越容易拟合噪声；\n"
            "3) 只保留有明确市场含义的改动（支撑/阻力结构、量能确认），"
            "删掉纯粹为了把回测数字做高的微调；\n"
            "4) 宁可收益平一点，也要在不熟悉的数据上不衰减。\n"
            f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")
    if no_trade:
        old_t = comparison["old"]["total_trades"]
        new_t = comparison["new"]["total_trades"]
        return (
            f"上一版迭代在验证窗口无成交触发（旧 {old_t} 笔 / 新 {new_t} 笔）——"
            "疑似参数间距（如 grid_pct/触发阈值）超出窗口实际波动或窗口恰好横盘，"
            "并非绩效下降。\n"
            f"请调整触发间距使验证窗口能产生成交：{_DENSITY_KNOBS.get(executor, '')}。\n"
            f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")
    new_r = comparison["new"]["total_return"]
    old_r = comparison["old"]["total_return"]
    new_dd = comparison["new"]["max_drawdown"]
    old_dd = comparison["old"]["max_drawdown"]
    new_sh = round(float(comparison["new"].get("sharpe") or 0), 3)
    old_sh = round(float(comparison["old"].get("sharpe") or 0), 3)
    return (
        f"上一版迭代在同数据回测中未超越上一代：新收益 {new_r:.4f} vs 旧 {old_r:.4f}，"
        f"回撤 {new_dd:.4f} vs {old_dd:.4f}，夏普 {new_sh} vs {old_sh}。\n"
        "请针对性调整入场条件/出场逻辑/过滤强度（参数间距过紧过松都会失败），"
        "必须让同数据回测的收益、回撤、夏普同时改善；不要只改名字和文案。\n"
        f"上一版参数（需要改动的就是这些）：{json.dumps(params, ensure_ascii=False)}")


def _split_is_oos(df):
    """按时间把校验窗口切成 (选优段 IS, 留出段 OOS)。

    任一段不足 _ITER_MIN_SEGMENT 根 → (None, None)（调用方降级为不判定）。
    时间顺序切分是刻意选择：留出段必须是**更晚**的行情，才能检验
    "在没见过的新数据上还成不成立"。
    """
    if df is None or len(df) < _ITER_MIN_SEGMENT * 2:
        return None, None
    n_is = int(len(df) * _ITER_IS_FRACTION)
    if n_is < _ITER_MIN_SEGMENT or (len(df) - n_is) < _ITER_MIN_SEGMENT:
        return None, None
    return df.iloc[:n_is], df.iloc[n_is:]


def _pnl_stats(pnls: list) -> dict:
    """逐笔已平仓盈亏的统计量（手算，不引 numpy）。

    `t_stat` = 每笔平均收益 / (标准差/√n)：样本量感知的收益质量指标。
    3 笔赚 2 笔、其中 1 笔特别大 → t 很小；50 笔稳定小赚 → t 大。
    目前仅作**诊断**（原因见文件头部 L2 负结果注释），不参与判定。
    """
    n = len(pnls)
    if n == 0:
        return {"t_stat": None, "avg_pnl": None, "pnl_std": None}
    mean = sum(pnls) / n
    if n < 2:
        return {"t_stat": None, "avg_pnl": round(mean, 6), "pnl_std": None}
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
    return {"t_stat": (round(mean / (sd / math.sqrt(n)), 4) if sd > 0 else None),
            "avg_pnl": round(mean, 6), "pnl_std": round(sd, 6)}


def _segment_metrics(df, symbol: str, timeframe: str, name: str, params: dict) -> dict:
    """在给定数据段上跑一次向量化回测，返回统一口径的关键指标。

    除收益/夏普/回撤/成交笔数外，还回传逐笔收益的 `t_stat`（见 `_pnl_stats`）——
    它是"这笔收益有多少样本量在支撑"的诊断量，用来识别"5 笔里 1 笔运气单"。
    """
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    res = run_backtest_fast(df, BacktestConfig(
        symbol=symbol, timeframe=timeframe,
        strategy_name=name, strategy_params=dict(params or {})), bootstrap=False) or {}
    m = res.get("metrics") or {}
    # metrics.total_trades 统计的正是已平仓（side=sell）的笔数，两边口径一致
    pnls = [float(t.get("pnl") or 0.0)
            for t in (res.get("trades") or []) if t.get("side") == "sell"]
    return {"total_return": float(m.get("total_return") or 0.0),
            "sharpe": float(m.get("sharpe") or 0.0),
            "max_drawdown": float(m.get("max_drawdown") or 0.0),
            "win_rate": float(m.get("win_rate") or 0.0),
            "total_trades": int(m.get("total_trades") or 0),
            **_pnl_stats(pnls)}


def evaluate_iteration_gate(df, *, symbol: str, timeframe: str,
                            base: tuple, parent: tuple, new: tuple) -> Optional[dict]:
    """迭代验证门（留出集版）。纯函数，便于单测与离线校准。

    base   = 固定基线（本迭代链的 v0）—— 锚点，防链式逐级抬升
    parent = 上一代（strategy 自身当前参数）
    new    = 本次候选
    三个参数均为 (strategy_name, params) 元组。

    判定（除 (1) 在选优段，其余全部在**留出段 OOS**）：
      1. is_improved：选优段上 new 比 base 多赚 ≥ _ITER_MIN_GAIN（AI 的努力方向，可见）
      2. oos_anchor ：留出段 new 不劣于 base、也不劣于 parent（各容忍 _ITER_OOS_TOLERANCE）
      3. dd_ok      ：留出段回撤不超 base/parent 的 _ITER_DD_TOLERANCE 倍
      4. sharpe_ok  ：留出段夏普不被拉垮
      5. trades_ok  ：留出段成交 ≥ _ITER_MIN_TRADES（防单笔运气单过关，L2 由 5 提到 10）
    improved = 1∧2∧3∧4∧5

    ⚠️ 留出段每笔收益的 t 统计量只放在返回值的 ``diagnostics`` 里，**不参与判定**
    ——原计划把它当第 6 条硬门槛，但实测既无鉴别力增益又会让门不可达
    （详见本文件头部「L2 负结果」注释与 .optim/bughunt/probe_l2_density.py）。

    返回 dict：``old``/``new``/``base`` = **选优段**指标（可安全喂给 AI），
    ``oos`` = 留出段指标（**只用于判定，禁止进 prompt**）。
    df 太短无法切分时返回 None（调用方降级）。
    """
    is_df, oos_df = _split_is_oos(df)
    if is_df is None:
        return None
    (base_name, base_params), (par_name, par_params), (new_name, new_params) = base, parent, new

    is_base = _segment_metrics(is_df, symbol, timeframe, base_name, base_params)
    is_par = _segment_metrics(is_df, symbol, timeframe, par_name, par_params)
    is_new = _segment_metrics(is_df, symbol, timeframe, new_name, new_params)
    oos_base = _segment_metrics(oos_df, symbol, timeframe, base_name, base_params)
    oos_par = _segment_metrics(oos_df, symbol, timeframe, par_name, par_params)
    oos_new = _segment_metrics(oos_df, symbol, timeframe, new_name, new_params)

    is_improved = is_new["total_return"] >= is_base["total_return"] + _ITER_MIN_GAIN
    oos_anchor_ok = (oos_new["total_return"] >= oos_base["total_return"] - _ITER_OOS_TOLERANCE
                     and oos_new["total_return"] >= oos_par["total_return"] - _ITER_OOS_TOLERANCE)
    # 回撤上限：取 base/parent 的较差者再放宽倍数，并给固定余量
    # （base 与 parent 都没交易时 dd=0，不给余量会导致"新策略必须零回撤"这种不可能的门槛）
    dd_bound = max(max(oos_base["max_drawdown"], oos_par["max_drawdown"]) * _ITER_DD_TOLERANCE,
                   0.05)
    dd_ok = oos_new["max_drawdown"] <= dd_bound
    sharpe_ok = oos_new["sharpe"] >= (min(oos_base["sharpe"], oos_par["sharpe"])
                                      - _ITER_SHARPE_TOLERANCE)
    trades_ok = oos_new["total_trades"] >= _ITER_MIN_TRADES
    no_trade = oos_new["total_trades"] == 0

    return {
        # 兼容既有前端/prompt/测试语义：old/new/base = 选优段（IS）指标
        "old": is_par, "new": is_new, "base": is_base,
        # 留出段：仅判定用。禁止进入任何喂给 AI 的文本
        "oos": {"old": oos_par, "new": oos_new, "base": oos_base},
        "is_improved": bool(is_improved),
        "oos_ok": bool(oos_anchor_ok and dd_ok and sharpe_ok and trades_ok),
        "improved": bool(is_improved and oos_anchor_ok and dd_ok and sharpe_ok and trades_ok),
        "no_trade": bool(no_trade),
        "checks": {"is_improved": bool(is_improved), "oos_anchor": bool(oos_anchor_ok),
                   "dd": bool(dd_ok), "sharpe": bool(sharpe_ok), "trades": bool(trades_ok),
                   "dd_bound": round(dd_bound, 6)},
        "criteria": {"min_gain": _ITER_MIN_GAIN, "min_trades": _ITER_MIN_TRADES,
                     "sharpe_tolerance": _ITER_SHARPE_TOLERANCE,
                     "is_fraction": _ITER_IS_FRACTION,
                     "oos_tolerance": _ITER_OOS_TOLERANCE,
                     "dd_tolerance": _ITER_DD_TOLERANCE},
        # 诊断量（**不参与判定**）：留出段每笔收益的 t 统计量。
        # 用途是解释"这笔收益有多少样本量在支撑"，而不是再加一道门——
        # 实测它并不比裸收益更能挑出泛化好的候选（见文件头部 L2 负结果注释）。
        "diagnostics": {"oos_new_t_stat": oos_new.get("t_stat"),
                        "oos_base_t_stat": oos_base.get("t_stat"),
                        "oos_par_t_stat": oos_par.get("t_stat"),
                        "oos_new_avg_pnl": oos_new.get("avg_pnl")},
        "note": (f"选优段(IS) 前 {len(is_df)} 根用于提升/反馈；"
                 f"留出段(OOS) 后 {len(oos_df)} 根仅用于判定，不反馈给 AI"),
    }


def _prompt_snap_truncated(snap: dict, full_df, is_df) -> dict:
    """把喂给 AI 的快照截断到选优段：candles 截断 + 指标按截断后重算。

    不截断的话 _snapshot_block/_sr_pa_block 会给出**留出段末尾**的收盘价、
    最近K线明细、指标读数与 S/R —— AI 等于直接对着留出集设计，留出段就形同虚设。
    对不齐（长度不等）时保持原样并告警：宁可漏截，也不截错位置。
    """
    candles = snap.get("candles") or []
    if is_df is None or full_df is None or not candles:
        return snap
    if len(candles) != len(full_df) or len(is_df) > len(candles):
        log.warning("[ai] 快照K线(%s)与校验窗口(%s)长度不一致，跳过留出集截断"
                    "（本次 AI 可见范围会包含留出段）", len(candles), len(full_df))
        return snap
    cut = [list(c) for c in candles[:len(is_df)]]
    out = {**snap, "candles": cut}
    try:
        from indicators.technical import compute_latest
        out["indicators"] = compute_latest(cut)
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 截断后重算指标失败（沿用原指标，可能含留出段信息）: %s", e)
    return out


class StrategyIteration:
    """AI 重新思考迭代现有策略。"""

    def __init__(self, client: AIClient, db: Database) -> None:
        self._client = client
        self._db = db

    @staticmethod
    def _base_name(name: str) -> str:
        """剥离迭代后缀：dual_ma_v2 -> dual_ma。用于基于同一基础策略累加序号。"""
        import re
        return re.sub(r"_v\d+$", "", name)

    async def _next_version(self, based_on: str) -> int:
        """返回下一迭代序号（基于同一基础策略）：1 → 2 → 3 …（对应 _v1/_v2/_v3）。

        冲突避让：若 _v{n} 已被内置/动态策略占用（用户手工设计过同名策略、
        或 DB 重置后 KV 计数回退），递增跳过，绝不静默覆盖既有策略
        （register_dynamic 对同名是覆盖式写入，get_strategy 又优先取动态版）。
        """
        try:
            from strategies import _REGISTRY, _DYNAMIC
            key = f"ai_iter_count_{based_on}"
            with _ITER_VERSION_LOCK:
                # run25 R2：读-自增-避让-写回整体持锁（防并发重号静默覆盖）
                count = int(await self._db.kv_get(key) or 0)
                new_count = count + 1
                while f"{based_on}_v{new_count}" in _REGISTRY or f"{based_on}_v{new_count}" in _DYNAMIC:
                    new_count += 1
                await self._db.kv_set(key, str(new_count))
            return new_count
        except Exception:  # noqa: BLE001
            return 1

    async def iterate(self, strategy: Strategy, performance: dict, snap: dict,
                      backtests: Optional[list] = None,
                      previous: Optional[list] = None,
                      validation_df=None, validation_symbol: str = "BTC/USDT",
                      validation_timeframe: str = "1h",
                      validation_df_alt=None,
                      validation_alt_symbol: Optional[str] = None,
                      validation_alt_timeframe: Optional[str] = None,
                      require_improve: bool = True,
                      goal: str = "") -> Optional[dict]:
        """对现有策略做一次深度反思迭代（参考历史回测与前代迭代记录）。失败时返回 None。

        validation_df：同数据回测验证门（先回测后注册，绩效差拒绝）——
        由调用方传入快照 K 线构造的 DataFrame；None 时跳过验证直接注册。
        validation_df_alt：备选验证窗口（如更大周期/更长历史）——主窗口 0 成交时
        自动用备选窗口重验同一版参数，避免"验证窗口恰好横盘→网格 0 成交→
        四指标全 0→误判绩效下降"的死锁（grid_pct 超出窗口波动时窗口触发不了）。
        """
        # 保留原策略的执行器与参数 schema：用原策略类校验 AI 给出的参数，
        # 避免迭代 dual_ma/grid 后 executor 被替换成 price_action（逻辑脱节）。
        from strategies import get_strategy as _get_strategy
        from strategies import get_dynamic as _get_dynamic
        _dyn = _get_dynamic(strategy.name) or {}
        executor = _dyn.get("executor") or strategy.name

        # 计算迭代序号：基于同一基础策略的累计迭代次数 → 1、2、3…（对应 _v1/_v2/_v3）
        base_name = self._base_name(strategy.name)
        iter_no = await self._next_version(base_name)
        new_name = f"{base_name}_v{iter_no}"

        # ---- L1：固定基线 + 留出集切分 ----
        # 基线 = 本条迭代链的 v0（基础策略）。链式对比"新 vs 上一代"会沿同一个窗口
        # 逐级抬升（实测 +20pp 假收益 / -7.6pp 真实泛化），锚定固定基线后链不再漂移。
        _base_dyn = _get_dynamic(base_name) or {}
        if base_name == strategy.name:
            # 迭代的就是基础策略本身：基线 = 它自己（锚点 = 当前实盘/在跑参数）
            base_executor, base_params = executor, dict(strategy.params or {})
        else:
            base_executor = _base_dyn.get("executor") or base_name
            base_params = dict(_base_dyn.get("params") or {}) or dict(strategy.params or {})

        # 喂给 AI 的快照截断到选优段：不截断则最近K线/指标/S-R 全落在留出段里，
        # 留出集形同虚设（详见 _prompt_snap_truncated 注释）
        _is_df, _ = _split_is_oos(validation_df)
        prompt_snap = _prompt_snap_truncated(snap, validation_df, _is_df)
        if _is_df is None and validation_df is not None:
            log.warning("[ai] 迭代 %s 校验窗口仅 %s 根，无法切分选优/留出段，"
                        "本次验证门降级为不判定", strategy.name, len(validation_df))

        # ---- 过拟合回炉 + 验证门回炉（与设计侧对称）：反馈各自累积，互不覆盖 ----
        # 严重过拟合时把检测指标（IS/OOS、衰减、PBO、flags）作为 feedback 注入下一次
        # 迭代 prompt，让 AI 降自由度/收敛参数/简化逻辑；最多 _OVERFIT_MAX_RETRIES 次，
        # 重试用尽仍过拟合才注册打标（P4-D7 兜底：人工可启用，自动接管被拦）。
        # 真实K线上样本外交易不足的"无法判定"也回炉一次（放宽过滤产生足够样本交易）。
        guard_candles = snap.get("candles", [])
        spec: dict = {}
        applied: dict = {}
        overfit_report = None
        feedback_parts: list[str] = []    # 过拟合/无法判定反馈累积（互不覆盖）
        validation_feedback = ""          # 验证门反馈（单条，每轮覆盖）
        overfit_retries = 0
        inconclusive_retries = 0
        validation_retries = 0
        max_attempts = (_OVERFIT_MAX_RETRIES + _OVERFIT_INCONCLUSIVE_MAX_RETRIES
                        + _VALIDATION_MAX_RETRIES + 1)
        # 循环前初始化：overfit_inconclusive（无法判定）路径在 comparison 赋值前
        # break 跳出循环，注册段仍引用它——曾 UnboundLocalError 500（L2 修复）
        comparison = None
        no_trade = False
        is_up = False
        oos_up = False
        for attempt in range(max_attempts):
            try:
                result = await self._client.chat_json_validated(
                    # prompt_snap（= 截断到选优段的快照）而非 snap：留出段不进 prompt
                    iteration_messages(strategy, performance, prompt_snap, backtests or [],
                                       previous or [], goal=goal,
                                       overfit_feedback="\n\n".join(feedback_parts),
                                       validation_feedback=validation_feedback),
                    feature="strategy_iterate",
                    ctx={"param_schema": strategy.param_schema})
            except (AINotConfigured, AICallError) as e:
                log.warning("[ai] 策略迭代跳过: %s", e)
                return None

            stub = _get_strategy(strategy.name)  # 同执行器类的新实例，参数仅保留 schema 内合法项
            applied = stub.update_params(result.get("params") or {})

            # 迭代生成新策略名（基础名_v序号），不显示 v1.0 版本徽标，改由策略名体现代次
            spec = {
                "name": new_name,
                "title": result.get("title", "迭代后策略"),
                "description": result.get("description", ""),
                "logic": result.get("logic", ""),
                "params": applied,
                "risk_tips": result.get("risk_tips", []),
                "critique": result.get("critique", ""),
                "improvements": result.get("improvements", []),
                "summary": result.get("summary", ""),
                "created_by": "ai_iteration",
                "based_on": base_name,
                "version": "",
                "iter_no": iter_no,
            }
            # 过拟合守卫：迭代出的策略同样要过前推验证，防止"把样本内噪声当规律"。
            # ⚠️ 守卫只看**选优段**（与 prompt 同源）：它的 walk-forward 折会横跨整个窗口，
            # 若窗口含留出段，_overfit_feedback 回给 AI 的 oos_ret/decay 就构成对留出段的
            # 信息泄漏。守卫负责"选优段内是否稳健"，留出段由验证门单独裁决。
            guard_for_check = prompt_snap.get("candles") or guard_candles
            try:
                overfit_report = await _guard_ai_strategy(
                    spec["name"], applied, prompt_snap, executor=executor,
                    guard_candles=guard_for_check,
                    # 多重检验校正：本条链上已试过的 AI 提案数（代次 + 本轮尝试）
                    prior_trials=iter_no + attempt)
            except Exception:  # noqa: BLE001
                overfit_report = None
                log.warning("[ai] 迭代过拟合守卫跳过", exc_info=True)
            if overfit_report:
                spec["overfit"] = overfit_report
            if overfit_report and overfit_report.get("verdict") == "无法判定":
                # 真实K线上样本外交易不足：回炉一次让 AI 放宽过滤产生足够样本，
                # 使检测有据可判（与设计侧一致）
                data_src = str(overfit_report.get("data_source") or "")
                avg_trades = float(overfit_report.get("avg_oos_trades") or 0)
                if (data_src == "真实K线" and avg_trades < _OVERFIT_MIN_OOS_TRADES
                        and inconclusive_retries < _OVERFIT_INCONCLUSIVE_MAX_RETRIES):
                    inconclusive_retries += 1
                    feedback_parts.append(_inconclusive_feedback(executor, applied, overfit_report))
                    log.warning("[ai] 迭代策略 %s 过拟合无法判定（样本外 %s 笔不足），回炉放宽触发门槛（第 %s 次）",
                                spec["name"], avg_trades, inconclusive_retries)
                    continue
                # 仍不足/数据不足（非策略责任）→ 不拦注册，但必须标出"未证明可用"：
                # 自动接管实盘的通道据此拒绝
                spec["overfit_inconclusive"] = True
                log.warning("[ai] 迭代策略 %s 过拟合守卫无法判定，允许注册但禁止自动接管实盘",
                            spec["name"])
                break
            if overfit_report and overfit_report.get("verdict") == "严重过拟合":
                if overfit_retries < _OVERFIT_MAX_RETRIES:
                    overfit_retries += 1
                    feedback_parts.append(_overfit_feedback(executor, applied, overfit_report))
                    log.warning("[ai] 迭代策略 %s 过拟合未通过（%s 分），回炉重迭代（第 %s/%s 次）",
                                spec["name"], overfit_report.get("score"),
                                overfit_retries, _OVERFIT_MAX_RETRIES)
                    continue
                # 重试用尽 → P4-D7 兜底：仍注册但打标（前端警告、自动接管门拦截）
                log.warning("[ai] 迭代策略 %s 过拟合回炉 %s 次仍未通过（%s 分），注册供人工启用（自动接管被拦）",
                            spec["name"], _OVERFIT_MAX_RETRIES, overfit_report.get("score"))
                spec["blocked_by_overfit"] = True

            # ---- 回测验证门（先回测后注册）：**留出段判定 + 锚定固定基线** ----
            # 曾对比回测在注册后执行、仅展示——劣质迭代照样入库污染策略库。
            # 现判据见 evaluate_iteration_gate：选优段(IS) 占前 _ITER_IS_FRACTION 用于
            # 提升与反馈，留出段(OOS) 用于裁决；且锚定**固定基线**（链 v0）而非上一代。
            # 原判据（同窗口比上一代 +1pp）实测会把策略改造成"窗口专精"：
            # W1 +20pp / W2 -7.6pp。拒绝时先回炉（_VALIDATION_MAX_RETRIES 次），
            # 只把 IS 指标喂回 AI，OOS 数字绝不外泄；仍不达标才拒绝并记 iteration_blocked。
            # （comparison/no_trade/is_up/oos_up 已在循环前初始化，此处重置）
            comparison = None
            no_trade = False
            is_up = False
            oos_up = False
            if validation_df is not None and len(validation_df) >= _ITER_MIN_SEGMENT * 2:
                _gate_symbol = validation_symbol
                _gate_timeframe = validation_timeframe
                _base_spec = (base_executor, base_params)
                _parent_spec = (strategy.name, dict(strategy.params or {}))

                def _gate_on(df, symbol: str, timeframe: str) -> Optional[dict]:
                    return evaluate_iteration_gate(
                        df, symbol=symbol, timeframe=timeframe,
                        base=_base_spec, parent=_parent_spec,
                        new=(executor, applied))

                try:
                    comparison = _gate_on(validation_df, _gate_symbol, _gate_timeframe)
                    if comparison is not None:
                        spec["comparison"] = comparison
                    else:
                        log.warning("[ai] 迭代 %s 校验窗口（%s 根）无法切分选优/留出段，"
                                    "本次验证门降级为不判定", spec["name"], len(validation_df))

                    # 留出段 0 成交识别：留出段没有任何成交 → 无法比较绩效。
                    # 此时若有备选窗口（更大周期/更长历史），用同一版参数自动重验一次，
                    # 避免因窗口选择而误杀合理迭代。
                    if (comparison is not None and comparison["no_trade"]
                            and validation_df_alt is not None
                            and len(validation_df_alt) >= _ITER_MIN_SEGMENT * 2):
                        try:
                            alt = _gate_on(validation_df_alt,
                                           validation_alt_symbol or _gate_symbol,
                                           validation_alt_timeframe or _gate_timeframe)
                            if alt is not None and not alt["no_trade"]:
                                comparison = alt
                                spec["comparison"] = comparison
                                log.info("[ai] 迭代 %s 留出段无成交，改用备选窗口(%s)重验判定",
                                         spec["name"],
                                         validation_alt_timeframe or _gate_timeframe)
                        except Exception:  # noqa: BLE001
                            log.warning("[ai] 迭代备选窗口重验失败，沿用主窗口判定: %s",
                                        exc_info=True)

                    if comparison is not None:
                        no_trade = bool(comparison["no_trade"])
                        checks = comparison.get("checks") or {}
                        is_up = bool(checks.get("is_improved"))
                        oos_up = bool(comparison.get("oos_ok"))
                    if (comparison is not None and require_improve
                            and not comparison["improved"]):
                        # "IS 上确实更好、但被留出段否决"必须单独说明：拿 IS 数字说
                        # "未超越上一代"是假话，会把 AI 引向错误方向。
                        oos_rejected = bool(is_up and not oos_up)
                        # 回炉：只喂 IS 指标 + 未过项（OOS 数字绝不出现），限次
                        if validation_retries < _VALIDATION_MAX_RETRIES:
                            validation_retries += 1
                            validation_feedback = _validation_feedback(
                                executor, applied, comparison, no_trade,
                                oos_rejected=oos_rejected)
                            log.warning("[ai] 迭代策略 %s 验证门未通过（IS提升=%s OOS达标=%s "
                                        "未过项=%s），回炉重迭代（验证门第 %s/%s 次）",
                                        spec["name"], is_up, oos_up,
                                        [k for k, v in (comparison.get("checks") or {}).items()
                                         if v is False],
                                        validation_retries, _VALIDATION_MAX_RETRIES)
                            continue
                        spec["registered"] = False
                        if no_trade:
                            # 明确区别"留出段无成交"与"绩效下降"，不给 0%→0% 的误导结论
                            log.warning("[ai] 迭代策略 %s 留出段无成交触发，拒绝注册", spec["name"])
                            spec["no_trade"] = True
                            spec["rejected_reason"] = (
                                "留出段（未参与选优的后段数据）无成交触发——"
                                "疑似参数间距（如 grid_pct）超出该段实际波动或恰好横盘，"
                                "非绩效下降；请调整参数间距后重试，或等待行情波动放大")
                            summary = (f"策略[{strategy.name}] 迭代 [{spec['name']}] "
                                       f"留出段无成交被拒绝")
                        else:
                            failed = [k for k, v in (comparison.get("checks") or {}).items()
                                      if v is False]
                            log.warning("[ai] 迭代策略 %s 验证门未通过（未过项=%s），拒绝注册",
                                        spec["name"], failed)
                            spec["rejected_reason"] = (
                                f"留出段验证未通过（未过项：{'、'.join(failed)}）；"
                                f"选优段新收益 {comparison['new']['total_return']:.2%} vs "
                                f"基线 {comparison['base']['total_return']:.2%}，保留旧策略。"
                                "（留出段具体数值不外泄，避免后续迭代转去拟合留出段）")
                            summary = (f"策略[{strategy.name}] 迭代 [{spec['name']}] "
                                       f"因留出段验证未通过被拒绝（{','.join(failed)}）")
                        async def _persist_blocked() -> None:
                            from core.database import OptimizationLog
                            async with self._db.session() as s:
                                s.add(OptimizationLog(
                                    kind="iteration_blocked",
                                    summary=summary,
                                    suggestion=json.dumps({"comparison": comparison,
                                                           "no_trade": no_trade},
                                                          ensure_ascii=False),
                                    params_json=json.dumps(applied, ensure_ascii=False),
                                ))
                                await s.commit()
                        # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
                        await _run_on_main(self._db, _persist_blocked)
                        return spec
                except Exception:  # noqa: BLE001
                    log.warning("[ai] 迭代回测验证失败（降级为直接注册）: %s", exc_info=True)
                    break

        # 注册为可用策略（新策略名 = 基础名_v序号，如 dual_ma_v1）
        from strategies import register_dynamic
        iter_spec = {
            "name": new_name,
            "title": spec.get("title", "迭代策略"),
            "description": spec.get("description", ""),
            "logic": spec.get("logic", ""),
            "params": applied,
            "risk_tips": spec.get("risk_tips", []),
            "created_by": "ai_iteration",
            "based_on": base_name,
            "version": "",
            "iter_no": iter_no,
            "critique": spec.get("critique", ""),
            "improvements": spec.get("improvements", []),
            "executor": executor,          # 保留原策略执行器，注册时不再默认 price_action
            "backtest": comparison,        # 注册时的回测验证指标（下一代迭代可见）
            # 绝对指标（new 侧原值）：链式对比（v3 vs v2）会掩盖长期整体退化，
            # 持久化绝对值让前端/后续迭代能看出 v10 相对 v1 的累计漂移
            "metrics_abs": (comparison or {}).get("new"),
            # P4-D7：过拟合照判但注册——标记与检测报告随 spec 落库，供前端显示警告、
            # 引擎自动接管门拦截（verdict 判定不变，只是不再一票否决注册）
            "overfit": spec.get("overfit"),
            "blocked_by_overfit": spec.get("blocked_by_overfit", False),
            # L2：此前这一步把 overfit_inconclusive 丢掉了——它只写在返回给前端的
            # spec 上，落库/进 _DYNAMIC 的 iter_spec 里没有。后果：重启或换页后
            # 看不出"这份策略证据不足"，而它照样能被选作迭代父代继续繁殖。
            "overfit_inconclusive": spec.get("overfit_inconclusive", False),
        }
        # L2：定 status（draft = 未证明：过拟合未过 / 证据不足）。
        # 注册仍然发生（用户可查看检测报告后手动启用），但草稿不再被当成可用策略。
        mark_status(iter_spec)
        # C3：迭代策略也生成 Pine Script 代码（与前端 buildPine 同源模板），
        # 保证"最新迭代"页签展示的是可运行 Pine 而非仅参数 JSON。
        # 但模板只对 price_action 执行器成立：曾无条件套用，rl_adaptive / dual_ma
        # 迭代出来的策略在图上显示的买卖点与其真实逻辑无关（假代码）。
        from strategies.pine_utils import PINE_TEMPLATE_EXECUTORS
        if executor in PINE_TEMPLATE_EXECUTORS:
            try:
                from strategies.pine_utils import build_price_action_pine
                iter_spec["pine_code"] = build_price_action_pine(iter_spec)
            except Exception as e:  # noqa: BLE001
                log.warning("[ai] 迭代 Pine 代码生成失败（不影响注册）: %s", e)
        else:
            iter_spec["pine_note"] = (
                f"执行器 {executor} 无等价 Pine 模板，未生成图表代码"
                f"（不套用其他策略模板，避免图上买卖点与真实逻辑脱节）")
        register_dynamic(new_name, iter_spec)
        spec["registered"] = True
        spec["comparison"] = comparison
        mark_status(spec)          # L2：返回给前端的 spec 也带 status/draft_reason
        # 持久化到 AiStrategy（重启后仍可见，全部策略列表能显示迭代策略）
        async def _persist_registered() -> None:
            from core.database import AiStrategy, OptimizationLog
            from sqlalchemy import select
            async with self._db.session() as s:
                s.add(OptimizationLog(
                    kind="iteration",
                    summary=f"策略[{strategy.name}] AI 深度反思迭代 -> [{new_name}]",
                    suggestion=json.dumps({
                        "critique": spec["critique"],
                        "improvements": spec["improvements"],
                        "summary": spec["summary"],
                        "overfit": overfit_report if overfit_report else None,
                    }, ensure_ascii=False),
                    params_json=json.dumps(applied, ensure_ascii=False),
                ))
                existing = (await s.execute(select(AiStrategy).where(AiStrategy.name == new_name))).scalar_one_or_none()
                if existing:
                    existing.spec_json = json.dumps(iter_spec, ensure_ascii=False)
                else:
                    s.add(AiStrategy(name=new_name, spec_json=json.dumps(iter_spec, ensure_ascii=False)))
                await s.commit()
        # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
        await _run_on_main(self._db, _persist_registered)
        log.info("[ai] 策略迭代完成: %s -> %s", strategy.name, new_name)
        return spec
