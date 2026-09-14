"""回测过拟合检测：识别 AI/参数搜索在历史数据上"作弊"的策略。

方法（多角度交叉验证）：
1. **时间序列前推验证（Walk-Forward）**：把数据切成训练段/验证段，滚动前推多次。
   只报告验证段（OOS）绩效——训练段过拟合无法在验证段延续。
2. **样本内外绩效衰减**：`1 - OOS_total_return / IS_total_return`，衰减>阈值判过拟合。
3. **PBO 过拟合概率**（Bailey & López de Prado）：对最优参数邻域做 CDF 分析，
   估计"在样本内选出的最优参数在样本外跑进后 50% 分布"的概率。PBO 越高越危险。
4. **多折稳定性**：每折验证段收益方向一致性 + 波动。

设计目标：任何 AI 设计/迭代/仓库策略在落地前先跑一次本检测，
如果 OOS 收益为负或 PBO 过高，给出"疑似过拟合"红旗并建议放弃或减少参数。
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 判定所需的最小样本外交易样本数：低于此值认为统计证据不足，
# 一律判“无法判定”而非“严重过拟合”（避免误拦低频 AI 设计策略）。
_MIN_TRADES_FOR_VERDICT = 3

# 判定"严重过拟合"的证据档位：样本外平均成交低于此值且收益为负时，
# 负收益更可能是成交样本不足/行情不适配的噪声，而非"参数过拟合"。
# 校准依据（真实数据探针）：内置默认策略（dual_ma/factor_signal/grid）
# 在近 7 个月真实 1h K线上 OOS 收益为负时，avg_oos_trades 仅 3~12 笔，
# 此前全部被判"严重过拟合"→ 设计通道"经常严重过拟合"的相当一部分是
# 判定误标（负收益=噪声）而非策略真过拟合。avg>=8（每折约 2 笔以上）
# 才认为收益符号有统计意义，维持"OOS 转负 → 严重过拟合"的既有语义。
# （回归用例将 avg=8 视为"交易充分"，证据线取严格小于 8。）
_TRADES_EVIDENCE_OK = 8

# PBO/DSR 的候选参数个数。原实现在 CSCV 内层只用 8 个"提交点 ±10% 抖动"候选
# （+当前参数共 9 个），候选都贴着 AI 已选中的点 → 度量不到参数选择偏差。
# 改为按 param_schema 全范围采样 32 个候选，使 PBO 真正反映"从候选集里挑最优"。
# 代价：CSCV 组合数 × 候选数 的回测次数上升（splits=6 时 20×33×2 ≈ 1320 次），
# 数据分块后单次回测很快；若嫌慢可下调本常量而不是回到抖动式候选。
_PBO_CANDIDATES = 32


@dataclass
class OverfitConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_cash: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    n_folds: int = 5          # 前推验证折数
    fold_ratio: float = 0.2   # 每折验证段占比（前推，非随机）
    min_is_len: int = 200     # 每折训练段最少K线数
    cscv_splits: int = 6      # CSCV 分块数（PBO 计算，Bailey-López de Prado；0=禁用回退邻域法）


@dataclass
class FoldResult:
    fold: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_ret: float
    oos_ret: float
    oos_sharpe: float
    oos_drawdown: float
    oos_trades: int


@dataclass
class OverfitReport:
    """过拟合检测报告。"""
    verdict: str = "未判定"          # "通过" / "疑似过拟合" / "严重过拟合"
    score: float = 0.0              # 0-100，越高越健康
    flags: list[dict] = field(default_factory=list)
    folds: list[FoldResult] = field(default_factory=list)
    is_ret: float = 0.0
    oos_ret: float = 0.0
    decay: float = 0.0              # 样本外衰减率
    pbo: float = 0.0                # 过拟合概率（0-1）
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    stability: float = 0.0          # 各折 OOS 收益同向比例
    n_folds: int = 0
    # 统计显著性（多重测试校正，Bailey & López de Prado）
    dsr: float | None = None        # 缩水夏普比率（0-1，≥0.95 显著）
    dsr_trials: int = 1             # DSR 校正用的搜索配置数
    sharpe_ci: list[float] | None = None  # Bootstrap 年化夏普 95% CI [low, high]
    ret_ci: list[float] | None = None     # Bootstrap 总收益 95% CI
    p_sharpe_pos: float | None = None     # Bootstrap P(夏普>0)


def _run_backtest_on(df: pd.DataFrame, cfg: OverfitConfig,
                     strategy_name: str, strategy_params: dict,
                     _external_clean: Optional[pd.DataFrame] = None) -> dict:
    """在指定数据段上跑向量化回测，返回 metrics。

    _external_clean: 已清洗的同一 df（CSCV 多候选共用一段数据时只清洗一次，
    见 _cscv_pbo 的 _rets_for 缓存；clean 幂等，输出与独立调用逐位一致）。
    """
    from .fast_engine import run_backtest_fast
    from .engine import BacktestConfig
    bc = BacktestConfig(symbol=cfg.symbol, timeframe=cfg.timeframe,
                        strategy_name=strategy_name, strategy_params=strategy_params,
                        start_cash=cfg.start_cash, fee_rate=cfg.fee_rate,
                        slippage=cfg.slippage)
    result = run_backtest_fast(df, bc, bootstrap=False,
                               _external_clean=_external_clean)  # P2-13 统计显著性由 detect_overfit 自身计算
    return result["metrics"]


def detect_overfit(df: pd.DataFrame, strategy_name: str, strategy_params: dict,
                   cfg: Optional[OverfitConfig] = None,
                   neighbor_params: Optional[list[dict]] = None,
                   prior_trials: int = 0) -> OverfitReport:
    """对策略做完整过拟合检测。

    neighbor_params: 参数候选集（供 PBO/DSR 分析）。若为 None 则按 param_schema
        **全范围**生成固定网格候选（不再是"提交点 ±10% 抖动"——后者只检验单点
        稳定性，度量不到"从众多候选里挑最优"的选择偏差）。
    prior_trials: 该参数在本策略上**已经试过的配置数**（AI 回炉次数 × 候选数、
        迭代链累积代次等）。它进 DSR 的多重检验校正基数：原实现把 trials 固定成
        候选数（≈9），与真实搜索规模脱钩，会让 DSR 系统性偏乐观。
        调用方给不出确切值时给保守下限（宁大勿小），0 = 未知/不校正。
    """
    cfg = cfg or OverfitConfig()
    if df.empty or len(df) < cfg.min_is_len * 2:
        raise ValueError("数据量不足，过拟合检测至少需要 "
                         f"{cfg.min_is_len * 2} 根K线（训练+验证）")

    report = OverfitReport(n_folds=cfg.n_folds)

    # ---- 1. Walk-Forward 前推验证（扩展窗口） ----
    # 每折：训练段 = [0, oos_start)（扩展窗口，数据越来越充分），
    # 验证段 = [oos_start, oos_start+fold_len)（依次前移，永不重叠、永不回看）。
    # 这种设计彻底杜绝未来函数泄漏：验证段数据从未参与参数选择。
    n = len(df)
    fold_len = max(50, int(n * cfg.fold_ratio))
    avail_folds = (n - cfg.min_is_len) // fold_len
    n_folds = min(cfg.n_folds, max(1, avail_folds))
    oos_rets: list[float] = []
    for f in range(n_folds):
        oos_start = cfg.min_is_len + f * fold_len
        oos_end = min(oos_start + fold_len, n)
        is_df = df.iloc[:oos_start]
        oos_df = df.iloc[oos_start:oos_end]
        if len(is_df) < 60 or len(oos_df) < 20:
            continue

        is_metrics = _run_backtest_on(is_df, cfg, strategy_name, strategy_params)
        oos_metrics = _run_backtest_on(oos_df, cfg, strategy_name, strategy_params)

        fr = FoldResult(
            fold=f + 1,
            is_start=str(is_df.index[0]),
            is_end=str(is_df.index[-1]),
            oos_start=str(oos_df.index[0]),
            oos_end=str(oos_df.index[-1]),
            is_ret=float(is_metrics.get("total_return", 0.0)),
            oos_ret=float(oos_metrics.get("total_return", 0.0)),
            oos_sharpe=float(oos_metrics.get("sharpe", 0.0) or 0.0),
            oos_drawdown=float(oos_metrics.get("max_drawdown", 0.0)),
            oos_trades=int(oos_metrics.get("total_trades", 0)),
        )
        report.folds.append(fr)
        oos_rets.append(fr.oos_ret)
    report.n_folds = len(report.folds)

    if not report.folds:
        report.verdict = "无法判定"
        report.score = 0.0
        report.flags.append({"level": "error", "msg": "前推验证折叠数不足，无法判定"})
        return report

    # ---- 2. 样本内外绩效汇总 ----
    report.is_ret = round(float(np.mean([f.is_ret for f in report.folds])), 6)
    report.oos_ret = round(float(np.mean(oos_rets)), 6)
    report.oos_sharpe = round(float(np.mean([f.oos_sharpe for f in report.folds])), 4)
    report.oos_win_rate = round(float(sum(1 for r in oos_rets if r > 0) / len(oos_rets)), 4)
    report.stability = round(report.oos_win_rate, 4)
    report.decay = round(1.0 - report.oos_ret / report.is_ret, 4) if report.is_ret > 0 else round(report.oos_ret, 4)

    # ---- 3. PBO 过拟合概率（CSCV 组合对称交叉验证，Bailey & López de Prado） ----
    # 候选集按 param_schema 全范围生成（取不到 schema 时退化回 ±10% 抖动）
    schema = _param_schema_for(strategy_name)
    if cfg.cscv_splits >= 4:
        _pbo = _cscv_pbo(df, cfg, strategy_name, strategy_params,
                         neighbor_params, schema)
        # run24 修复：_cscv_pbo 返回 None（数据不足）与 0.0（零过拟合）必须区分——
        # 原 `or 0.5` 把合法的 PBO=0.0 也吞成 0.5（0.0 为 falsy），健康度少 20 分
        report.pbo = round(0.5 if _pbo is None else _pbo, 4)
    else:
        # 兼容旧路径：walk-forward 邻域法
        report.pbo = round(_estimate_pbo(df, cfg, strategy_name, strategy_params,
                                         neighbor_params, report.folds, schema), 4)

    # ---- 3.5 统计显著性：DSR（多重测试校正）+ Bootstrap 夏普 CI ----
    # DSR 校正的"试验数"口径（2026-09-12 校正）：
    #   = 本次评估的候选集大小（schema 全范围网格）
    #   + prior_trials（调用方告知的历史搜索规模：AI 回炉次数 × 候选数、
    #     迭代链累积代次、被拒设计数……）
    # 原实现把 trials 固定成"候选数"（≈9），与真实搜索规模完全脱钩：
    # 实测同一策略同一数据，候选集定义一变 DSR 的 trials 就从 9 变 31
    # （见 .optim/bughunt/REPORT_overfit_source.md）。校正基数偏小 → DSR 偏乐观，
    # 使"扣掉运气后并不显著"的策略也能拿到"通过"。
    _cand_params = ([dict(strategy_params)]
                    + [dict(p) for p in (neighbor_params or _generate_neighbors(
                        strategy_params, count=_PBO_CANDIDATES, schema=schema))])
    _cand_params = _cand_params[: _PBO_CANDIDATES + 1]
    _n_candidates = max(1, len(_cand_params) + max(0, int(prior_trials)))
    try:
        from .fast_engine import run_backtest_fast
        from .engine import BacktestConfig
        from .metrics import PERIODS_PER_YEAR, bootstrap_sharpe_ci, deflated_sharpe_ratio, equity_returns

        def _cfg_for(params: dict) -> "BacktestConfig":
            return BacktestConfig(symbol=cfg.symbol, timeframe=cfg.timeframe,
                                  strategy_name=strategy_name, strategy_params=params,
                                  start_cash=cfg.start_cash, fee_rate=cfg.fee_rate,
                                  slippage=cfg.slippage)

        full = run_backtest_fast(df, _cfg_for(strategy_params), bootstrap=False)  # P2-13 同 OOS 口径
        rets = equity_returns(full.get("equity_curve") or [])
        # 候选集各自的**单周期**样本夏普 → DSR 据此估计 SR 跨试验方差（比 H0 近似更贴合实际）。
        # 必须是单周期口径（deflated_sharpe_ratio 内部 sr=mu/sigma 不年化），
        # 传年化夏普会把 sr0 放大 sqrt(periods_per_year) 倍、DIRECTION 完全错掉。
        trial_sharpes: list[float] = []
        for _p in _cand_params:
            try:
                _r = equity_returns(run_backtest_fast(df, _cfg_for(_p),
                                                     bootstrap=False).get("equity_curve") or [])
                _r = _r[np.isfinite(_r)]
                if len(_r) < 20:
                    continue
                _sd = float(_r.std(ddof=0))
                if _sd > 1e-12:
                    trial_sharpes.append(float(_r.mean() / _sd))
            except Exception:  # noqa: BLE001
                continue
        d = deflated_sharpe_ratio(
            rets, trials=max(1, _n_candidates),
            trial_sharpes=trial_sharpes if len(trial_sharpes) >= 2 else None)
        report.dsr = d["dsr"]
        report.dsr_trials = d["trials"]
        ppy = PERIODS_PER_YEAR.get(cfg.timeframe, 8760)
        b = bootstrap_sharpe_ci(rets, n_boot=1000, periods_per_year=ppy)
        report.sharpe_ci = b["sharpe_ci"]
        report.ret_ci = b["ret_ci"]
        report.p_sharpe_pos = b["p_sharpe_pos"]
    except Exception as e:  # noqa: BLE001
        log.warning("[overfit] DSR/Bootstrap 计算失败（跳过）: %s", e)

    # ---- 4. 综合判定 ----
    report = _score_and_verdict(report)
    return report


def _cscv_pbo(df: pd.DataFrame, cfg: OverfitConfig, strategy_name: str,
              strategy_params: dict, neighbor_params: Optional[list[dict]],
              schema: Optional[dict] = None) -> Optional[float]:
    """CSCV 组合对称交叉验证 PBO（Bailey, Borwein, López de Prado & Zhu 2015）。

    方法：把样本按时间顺序分成 S 块，枚举全部"取 S/2 块训练、其余测试"的组合：
    - 每个组合在训练块上选 IS 最优参数（候选集 = 当前参数 + 扰动邻域）
    - 记录该参数在测试块上的表现排名（1=最好）
    - PBO = IS 最优参数在测试块上掉进后 50% 的组合比例

    相比单路径 walk-forward 邻域分析，CSCV 枚举了全部训练/测试划分，
    对"参数选择是否只是运气"的检验更全面（López de Prado AFML 第 12 章）。
    返回 None 表示数据/候选不足，由调用方回退。
    """
    import itertools

    cand = ([dict(strategy_params)]
            + [dict(p) for p in (neighbor_params
                                 or _generate_neighbors(strategy_params,
                                                        count=_PBO_CANDIDATES,
                                                        schema=schema))])
    cand = cand[: _PBO_CANDIDATES + 1]
    if len(cand) < 4:
        return None
    n = len(df)
    splits = max(4, int(cfg.cscv_splits))
    if n < splits * 60:
        return None
    # 时间顺序分块（CSCV 的分块顺序本身不破坏时间连续性——每块内部保持时序）
    block_len = n // splits
    blocks = [df.iloc[i * block_len:(i + 1) * block_len] for i in range(splits)]
    blocks = [b for b in blocks if len(b) >= 30]
    if len(blocks) < 4:
        return None
    half = len(blocks) // 2

    ranks: list[float] = []
    # 记忆化（4× 加速、不改统计量）：C(S, S/2) 枚举出的每个组合，其测试集
    # 恰好是互补组合的训练集 —— 同一段数据会被反复回测两次。
    # 按"块集合"缓存每候选的收益，S=6 时唯一分段从 20×2 降到 10，
    # 回测次数从 20×2×N 降到 10×N。
    # 清洗复用：每唯一分段只 OHLCVSanitizer 清洗一次（clean 与策略参数无关且
    # 幂等，probe_sanitize_idem PASS），全部候选共享——CSCV 是验证门主要耗时，
    # 清洗约占总回测 25% 的重复开销。
    ret_cache: dict[frozenset, list[float]] = {}
    clean_cache: dict[frozenset, "pd.DataFrame"] = {}

    def _rets_for(idx) -> list[float]:
        key = frozenset(idx)
        cached = ret_cache.get(key)
        if cached is not None:
            return cached
        sub = pd.concat([blocks[i] for i in sorted(idx)])
        cleaned = clean_cache.get(key)
        if cleaned is None:
            from .data_loader import OHLCVSanitizer
            cleaned = OHLCVSanitizer().clean(sub.copy(), mode="mark")
            clean_cache[key] = cleaned
        out: list[float] = []
        for p in cand:
            try:
                m = _run_backtest_on(sub, cfg, strategy_name, p,
                                     _external_clean=cleaned)
                out.append(float(m.get("total_return", 0.0)))
            except Exception as e:  # noqa: BLE001
                log.warning("[overfit] CSCV 分段回测失败: %s", e)
                out.append(float("-inf"))
        ret_cache[key] = out
        return out

    for train_idx in itertools.combinations(range(len(blocks)), half):
        test_idx = tuple(i for i in range(len(blocks)) if i not in train_idx)
        # 训练块上选 IS 最优参数
        is_rets = _rets_for(train_idx)
        best = int(np.argmax(is_rets))
        # 最优参数在测试块上的排名（1=最好）
        oos_rets = _rets_for(test_idx)
        rank = int((np.asarray(oos_rets) > oos_rets[best]).sum()) + 1
        ranks.append(rank / len(cand))

    if not ranks:
        return None
    # PBO = IS 最优参数在 OOS 掉到后 50% 的比例
    return float(np.mean([r > 0.5 for r in ranks]))


def _estimate_pbo(df: pd.DataFrame, cfg: OverfitConfig, strategy_name: str,
                  strategy_params: dict, neighbor_params: Optional[list[dict]],
                  folds: list[FoldResult],
                  schema: Optional[dict] = None) -> float:
    """估计 PBO（过拟合概率）。

    方法：对每个前推折，若参数 A 在训练段最优、但在验证段排名掉到后 50%，
    记为一次"过拟合选择"。统计比例即为 PBO 的近似。需要参数邻域样本。
    若没提供邻域，基于最优参数自动生成扰动邻域。
    """
    if not neighbor_params or len(neighbor_params) < 5:
        neighbor_params = _generate_neighbors(strategy_params, count=_PBO_CANDIDATES,
                                             schema=schema)
    if not neighbor_params:
        return 0.5  # 无法评估时给中性值

    # 每折：训练段选最优参数（IS 收益最高），看它在验证段的排名
    oos_ranks: list[float] = []
    for fr in folds:
        is_start = pd.Timestamp(fr.is_start, tz="UTC")
        is_end = pd.Timestamp(fr.is_end, tz="UTC")
        oos_start = pd.Timestamp(fr.oos_start, tz="UTC")
        oos_end = pd.Timestamp(fr.oos_end, tz="UTC")
        is_df = df.loc[is_start:is_end]
        # 与主流程切片语义一致（iloc 不含端点）：此前 loc 含端点，
        # 相邻折 OOS 段重叠 1 根，PBO 评估数据被重复使用
        oos_df = df.iloc[df.index.get_loc(oos_start):df.index.get_loc(oos_end)]

        # 候选参数集合 = 最优 + 邻域
        cand = [dict(strategy_params)] + [dict(p) for p in neighbor_params]
        is_ret = []
        oos_ret = []
        for p in cand[:8]:  # 限制数量控制耗时
            try:
                m1 = _run_backtest_on(is_df, cfg, strategy_name, p)
                m2 = _run_backtest_on(oos_df, cfg, strategy_name, p)
                is_ret.append(float(m1.get("total_return", 0.0)))
                oos_ret.append(float(m2.get("total_return", 0.0)))
            except Exception as e:  # noqa: BLE001
                log.warning("[overfit] 邻域参数回测失败: %s", e)
                is_ret.append(float("-inf"))
                oos_ret.append(float("-inf"))
        if len(is_ret) < 5:
            continue
        is_ret = np.asarray(is_ret)
        oos_ret = np.asarray(oos_ret)
        # 最优参数 = IS 排名第 1
        best_is_idx = int(np.argmax(is_ret))
        # 它在 OOS 中的排名（1=最好）
        oos_rank = int((oos_ret > oos_ret[best_is_idx]).sum()) + 1
        oos_ranks.append(oos_rank / len(oos_ret))
    if not oos_ranks:
        return 0.5
    # PBO = IS 最优参数在 OOS 掉到后 50% 的比例
    pbo = float(np.mean([r > 0.5 for r in oos_ranks]))
    return pbo


def _generate_neighbors(params: dict, count: int = 12,
                        schema: Optional[dict] = None) -> list[dict]:
    """生成 PBO/DSR 用的参数候选集。

    ⚠️ 语义要点（2026-09-12 校正）：PBO 要回答的是
    **"从一个候选集里挑出的样本内最优，在样本外是否掉到后 50%"**——
    它度量的是**参数选择偏差**。原实现只用"提交点 ±10% 抖动"，
    候选全部贴着 AI 已经选中的那个点，于是它实际只回答了
    "这一个点稳不稳"，而不是"你是不是在众多候选里过拟合地挑了最优"。
    实测（BTC/USDT 1h，同一策略同一数据）：候选集定义一变，
    PBO 就在 0.00 ↔ 0.30 之间跳（见 .optim/bughunt/REPORT_overfit_source.md）。

    因此：schema 给了 min/max 的键 → 在**全 schema 范围**内采样（真正的候选搜索空间，
    与提交点无关、可复现）；schema 缺失或键不在 schema 内 → 退回 ±10% 抖动（兼容旧行为）。
    """
    if not params:
        return []
    numeric_keys = [k for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric_keys:
        return []
    schema = schema or {}
    rng = np.random.default_rng(42)
    neighbors = []
    for _ in range(count):
        p = dict(params)
        for k in numeric_keys:
            v = params[k]
            spec = schema.get(k) or {}
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and hi is not None and float(hi) > float(lo):
                if isinstance(v, int) and not isinstance(v, bool):
                    p[k] = int(rng.integers(int(lo), int(hi) + 1))
                else:
                    p[k] = round(float(rng.uniform(float(lo), float(hi))), 6)
            else:
                delta = rng.uniform(-0.1, 0.1) * abs(v) + 1e-9
                if isinstance(v, int):
                    p[k] = max(1, int(round(v + delta)))
                else:
                    p[k] = round(v + delta, 6)
        neighbors.append(p)
    return neighbors


def _param_schema_for(strategy_name: str) -> dict:
    """取某执行器的 param_schema；取不到返回 {}（调用方退回 ±10% 抖动）。"""
    try:
        from strategies import _REGISTRY
        cls = _REGISTRY.get(strategy_name)
        return dict(getattr(cls, "param_schema", {}) or {}) if cls is not None else {}
    except Exception:  # noqa: BLE001
        return {}


def _is_statistically_significant(report: OverfitReport) -> bool:
    """"通过"的必要条件：扣除多重检验的运气成分后仍显著。

    - Bootstrap 年化夏普 95% CI 下界 > 0 → 显著（表现不可能由抽样噪声解释）
    - 缩水夏普 DSR ≥ 0.95（Bailey & López de Prado 惯例）→ 显著
    - 两者都拿不到（样本太短、方差为零）→ 无法证明，返回 False（证据不足不等于通过）
    """
    ci = report.sharpe_ci
    if ci is not None and len(ci) == 2 and float(ci[0]) > 0:
        return True
    if report.dsr is not None:
        return float(report.dsr) >= 0.95
    return False


def _score_and_verdict(report: OverfitReport) -> OverfitReport:
    """按多指标综合评分并给出判定。

    核心原则：**只有"样本外转负"才是过拟合**。OOS 仍为正的稳健策略
    即使样本内赚得更多也不判过拟合（衰减大只是信息提示）。

    健康度 score 0-100（越高越可信），权重：
    - OOS 收益为正且同向稳定（50 分）
    - PBO 过拟合概率（30 分）
    - OOS 交易数 / 稳定性（20 分）
    """
    score = 0.0
    flags: list[dict] = []

    # OOS 收益（核心：样本外是否真的赚）
    if report.oos_ret > 0:
        score += 30 + 20 * min(1.0, report.oos_ret / 0.05)   # 正收益给分，5%以上满分
        if report.stability >= 0.6:
            score += 10  # 多数折叠为正 → 稳定
        # 衰减大但 OOS 仍正 → 信息提示，不否决
        if report.decay > 0.6:
            flags.append({"level": "info",
                          "msg": f"样本外相对样本内衰减 {report.decay:.0%}，但样本外仍为正，"
                                 "提示参数可能偏热（样本内收益不可复制）"})
    else:
        flags.append({"level": "high", "msg": f"样本外平均收益为负 ({report.oos_ret:.2%})，过拟合信号强"})
        score -= 15

    # PBO（参数选择的鲁棒性）
    if report.pbo <= 0.25:
        score += 30 - report.pbo * 30
    elif report.pbo <= 0.5:
        score += 10
        flags.append({"level": "warn", "msg": f"PBO 过拟合概率 {report.pbo:.0%}，所选参数有一定过拟合风险"})
    else:
        flags.append({"level": "high", "msg": f"PBO 过拟合概率 {report.pbo:.0%}，所选参数很可能只在样本内有效"})

    # 交易数（低频策略不因此否决；样本外完全无交易 → 无法判定而非过拟合证据）
    avg_trades = float(np.mean([f.oos_trades for f in report.folds])) if report.folds else 0.0
    if avg_trades >= 3:
        score += 10
    elif avg_trades > 0:
        score += max(0, 10 * avg_trades / 3)
        flags.append({"level": "info", "msg": f"样本外平均交易数仅 {avg_trades:.1f} 笔，统计意义有限"})
    else:
        # 无交易：策略可能不适应验证段的数据形态，但不是"过拟合"的直接证据
        flags.append({"level": "info", "msg": "样本外没有产生交易，检测证据不足，请补充不同行情数据后重测"})

    # 统计显著性（DSR + Bootstrap；仅作 ±10 分微调，OOS/PBO 仍主导判定）
    if report.dsr is not None:
        if report.dsr >= 0.95:
            score += 8
            flags.append({"level": "info",
                          "msg": f"缩水夏普 DSR={report.dsr:.2f}（{report.dsr_trials} 个候选多重测试校正后仍显著）"})
        elif report.dsr < 0.5:
            score -= 6
            flags.append({"level": "warn",
                          "msg": f"缩水夏普 DSR={report.dsr:.2f}，扣除 {report.dsr_trials} 次搜索的运气成分后收益不显著"})
    if report.sharpe_ci is not None:
        if report.sharpe_ci[0] > 0:
            score += 2  # Bootstrap 夏普 95% CI 下界为正
        elif report.sharpe_ci[1] < 0:
            score -= 4
            flags.append({"level": "warn",
                          "msg": f"Bootstrap 夏普 95% CI [{report.sharpe_ci[0]:.2f}, {report.sharpe_ci[1]:.2f}] 全为负"})

    report.score = round(max(0.0, min(100.0, score)), 1)
    # 判定（含证据充足性门）：只有当样本外确实产生足够交易时才谈“过拟合”。
    # 曾仅当 avg_trades==0 且 oos_ret==0.0 时判“无法判定”；但 AI 设计策略在
    # 验证段常只产生零星几笔交易，任一折落一笔微小负收益就让 oos_ret 略非 0，
    # 从而被误判“严重过拟合”→ 拒绝注册 → 无法应用、不入库。
    # 交易样本不足（< _MIN_TRADES_FOR_VERDICT）时，OOS 收益是噪声而非证据，
    # 统一降级为“无法判定”。它不拦注册（数据不够不该怪策略），
    # 但也不等于通过：自动接管实盘的通道把它当作"未证明可用"直接拒绝。
    # 交易充分且 OOS 转负 / PBO 高 → 仍判“严重过拟合”拦截。
    # 证据档位（_TRADES_EVIDENCE_OK）：3~7 笔成交仍不足以支撑"参数过拟合"
    # 结论——此时 OOS 为负只能说明"这段行情没赚到钱"（噪声/行情不适配），
    # 判"无法判定"而非"严重过拟合"，同样禁止自动接管（安全语义不变）。
    if avg_trades < _MIN_TRADES_FOR_VERDICT:
        report.verdict = "无法判定"
        flags.append({"level": "warn",
                      "msg": f"样本外平均交易仅 {avg_trades:.1f} 笔，证据不足，判定为无法判定"
                             "（可注册，但禁止自动接管实盘）"})
    elif report.oos_ret <= 0 and avg_trades < _TRADES_EVIDENCE_OK:
        report.verdict = "无法判定"
        flags.append({"level": "warn",
                      "msg": f"样本外平均成交仅 {avg_trades:.1f} 笔且收益为负（{report.oos_ret:.2%}）——"
                             "负收益更可能是成交样本不足/行情不适配的噪声，不足以判定参数过拟合"
                             "（可注册，但禁止自动接管实盘）"})
    elif report.score >= 60:
        # "通过"的必要条件（2026-09-12 校正）：扣除多重检验的运气成分后仍显著。
        # 此前 DSR/Bootstrap 只作 ±10 分的微调，于是 score 靠"OOS 正收益 + PBO 低 +
        # 有成交"就能堆到 90 并判"通过"，而 DSR≈0（与运气不可区分）照样放行。
        # 实测：BTC/USDT 1h 近 3000 根上，四个内置执行器默认参数的 DSR 全部落在
        # 0.0002~0.13、bootstrap 夏普 95% CI 下界全为负 —— 即"检测不出任何技术优势"。
        # 这类结果不该被标成"通过"，故降级为"疑似过拟合"并显式写明原因。
        if _is_statistically_significant(report):
            report.verdict = "通过"
        else:
            report.verdict = "疑似过拟合"
            _ci = (None if report.sharpe_ci is None
                   else [round(float(x), 2) for x in report.sharpe_ci])
            flags.append({"level": "warn",
                          "msg": f"综合评分达标，但统计显著性不足（DSR={report.dsr}，"
                                 f"夏普 95% CI={_ci}）：扣除 {report.dsr_trials} 次搜索的"
                                 "运气成分后与随机不可区分，或样本期太短无法判定 —— 不判通过"})
    elif report.score >= 40:
        report.verdict = "疑似过拟合"
    else:
        report.verdict = "严重过拟合"
    report.flags = flags
    return report
