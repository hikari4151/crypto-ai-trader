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
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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


def _run_backtest_on(df: pd.DataFrame, cfg: OverfitConfig,
                     strategy_name: str, strategy_params: dict) -> dict:
    """在指定数据段上跑向量化回测，返回 metrics。"""
    from .fast_engine import run_backtest_fast
    from .engine import BacktestConfig
    bc = BacktestConfig(symbol=cfg.symbol, timeframe=cfg.timeframe,
                        strategy_name=strategy_name, strategy_params=strategy_params,
                        start_cash=cfg.start_cash, fee_rate=cfg.fee_rate,
                        slippage=cfg.slippage)
    result = run_backtest_fast(df, bc)
    return result["metrics"]


def detect_overfit(df: pd.DataFrame, strategy_name: str, strategy_params: dict,
                   cfg: Optional[OverfitConfig] = None,
                   neighbor_params: Optional[list[dict]] = None) -> OverfitReport:
    """对策略做完整过拟合检测。

    neighbor_params: 参数邻域候选（供 PBO 分析）。若为 None 则自动基于
        strategy_params 生成小幅扰动邻域（按参数 schema 采样）。
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
    if cfg.cscv_splits >= 4:
        report.pbo = round(_cscv_pbo(df, cfg, strategy_name, strategy_params,
                                     neighbor_params) or 0.5, 4)
    else:
        # 兼容旧路径：walk-forward 邻域法
        report.pbo = round(_estimate_pbo(df, cfg, strategy_name, strategy_params,
                                         neighbor_params, report.folds), 4)

    # ---- 4. 综合判定 ----
    report = _score_and_verdict(report)
    return report


def _cscv_pbo(df: pd.DataFrame, cfg: OverfitConfig, strategy_name: str,
              strategy_params: dict, neighbor_params: Optional[list[dict]]) -> Optional[float]:
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

    cand = [dict(strategy_params)] + [dict(p) for p in (neighbor_params or _generate_neighbors(strategy_params, count=8))]
    cand = cand[:9]
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
    for train_idx in itertools.combinations(range(len(blocks)), half):
        test_idx = [i for i in range(len(blocks)) if i not in train_idx]
        train_df = pd.concat([blocks[i] for i in sorted(train_idx)])
        test_df = pd.concat([blocks[i] for i in sorted(test_idx)])

        # 训练块上选 IS 最优参数
        is_rets: list[float] = []
        for p in cand:
            try:
                m = _run_backtest_on(train_df, cfg, strategy_name, p)
                is_rets.append(float(m.get("total_return", 0.0)))
            except Exception as e:  # noqa: BLE001
                log.warning("[overfit] CSCV 训练段回测失败: %s", e)
                is_rets.append(float("-inf"))
        best = int(np.argmax(is_rets))

        # 最优参数在测试块上的排名（1=最好）
        oos_rets: list[float] = []
        for p in cand:
            try:
                m = _run_backtest_on(test_df, cfg, strategy_name, p)
                oos_rets.append(float(m.get("total_return", 0.0)))
            except Exception:  # noqa: BLE001
                oos_rets.append(float("-inf"))
        rank = int((np.asarray(oos_rets) > oos_rets[best]).sum()) + 1
        ranks.append(rank / len(cand))

    if not ranks:
        return None
    # PBO = IS 最优参数在 OOS 掉到后 50% 的比例
    return float(np.mean([r > 0.5 for r in ranks]))


def _estimate_pbo(df: pd.DataFrame, cfg: OverfitConfig, strategy_name: str,
                  strategy_params: dict, neighbor_params: Optional[list[dict]],
                  folds: list[FoldResult]) -> float:
    """估计 PBO（过拟合概率）。

    方法：对每个前推折，若参数 A 在训练段最优、但在验证段排名掉到后 50%，
    记为一次"过拟合选择"。统计比例即为 PBO 的近似。需要参数邻域样本。
    若没提供邻域，基于最优参数自动生成扰动邻域。
    """
    if not neighbor_params or len(neighbor_params) < 5:
        neighbor_params = _generate_neighbors(strategy_params, count=12)
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


def _generate_neighbors(params: dict, count: int = 12) -> list[dict]:
    """基于参数 schema 生成小幅扰动邻域（±5%/±10%），数值类型才扰动。"""
    if not params:
        return []
    neighbors = []
    numeric_keys = [k for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric_keys:
        return []
    rng = np.random.default_rng(42)
    for _ in range(count):
        p = dict(params)
        for k in numeric_keys:
            v = params[k]
            delta = rng.uniform(-0.1, 0.1) * abs(v) + 1e-9
            if isinstance(v, int):
                p[k] = max(1, int(round(v + delta)))
            else:
                p[k] = round(v + delta, 6)
        neighbors.append(p)
    return neighbors


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
    avg_trades = float(np.mean([f.oos_trades for f in report.folds]))
    if avg_trades >= 3:
        score += 10
    elif avg_trades > 0:
        score += max(0, 10 * avg_trades / 3)
        flags.append({"level": "info", "msg": f"样本外平均交易数仅 {avg_trades:.1f} 笔，统计意义有限"})
    else:
        # 无交易：策略可能不适应验证段的数据形态，但不是"过拟合"的直接证据
        flags.append({"level": "info", "msg": "样本外没有产生交易，检测证据不足，请补充不同行情数据后重测"})

    report.score = round(max(0.0, min(100.0, score)), 1)
    if avg_trades == 0 and report.oos_ret == 0.0:
        # 无交易且 OOS 收益为 0（未发生任何成交）→ 无法判定，不妄下结论
        report.verdict = "无法判定"
    elif report.score >= 60:
        report.verdict = "通过"
    elif report.score >= 40:
        report.verdict = "疑似过拟合"
    else:
        report.verdict = "严重过拟合"
    report.flags = flags
    return report


def ai_strategy_guard(df: pd.DataFrame, strategy_name: str, strategy_params: dict,
                      cfg: Optional[OverfitConfig] = None) -> OverfitReport:
    """AI 策略落地守卫：AI 设计/迭代出的策略在注册前强制过拟合检测。

    与 detect_overfit 等价，但语义强调"防 AI 作弊"。调用方可据此拒绝
    verdict != "通过" 的策略。
    """
    return detect_overfit(df, strategy_name, strategy_params, cfg)
