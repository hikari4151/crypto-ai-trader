"""多标的混合训练采样器：为 DRL 训练提供"经验广度"。

背景（2026-10，P0 多标的混合训练）：单标的训练只见过一段行情的一种
价格行为，agent 学不到"币种无关的交易规律"，val/OOS 与训练同分布，
"样本外验证"退化为"验证记住的这段行情的模式"。多标的混合训练让每个
episode 从"标的池 × 时间窗"随机采样一段，网络共享，把经验广度补上。

设计要点：
- 主标的（main_df）仍是训练/验证/OOS 切分与部署语义的锚点；
  extra_dfs 只作为额外经验来源，不参与 val/OOS 判据（避免改部署门语义）。
- 特征矩阵对每个标的**整段**预计算一次（rolling 窗口需要前序历史），
  episode 采样时按行切片，切片内每一行都自带完整历史信息（无前视）。
- TradingEnv 从 t=warmup 开始采样，切片起点无需额外留 warmup 余量；
  切片长度只要 ≥ warmup + 最小可学步数即可。
- 级联组合因子（factor_composite 配方）可在任意标的 df 上复算（expanding z，
  无前视，与部署端同口径），故混合模式可保持因子列维度一致；无配方的
  factor_values 无法跨标的重建 → 降级单标的训练并警告。
- 无 extra 标的（single-asset 模式）时，pool 退化为主标的整段采样，
  行为与旧 train_drl 完全一致（不改变既有模型/测试）。
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .env import TradingEnv, _precompute_features

# 采样器内使用的最小K线数（低于此长度无法跑出有意义的 episode）
_MIN_BARS = 300
_WARMUP = 60


class _Asset:
    """单个标的的可训练资产：df + 预计算特征 + 可选因子列。"""

    __slots__ = ("name", "df", "features", "factors", "difficulty", "_diff_win")

    def __init__(self, name: str, df: pd.DataFrame, features: np.ndarray,
                 factors: Optional[np.ndarray]) -> None:
        self.name = name
        self.df = df
        self.features = features            # (n, feat_dim) 整段预计算
        self.factors = factors              # (n,) 或 None（长度 == n）
        # 课程学习（P1）：各采样起点的窗口难度分位（[0,1]，0=最易/最低波动）。
        # 惰性填充：sample() 需要时按 window_len 计算（成本 O(n log n) 一次）。
        self.difficulty: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.df)

    def difficulty_ranks(self, window_len: int, rng_seed: int = 0) -> np.ndarray:
        """各采样起点的窗口难度分位（长度 = n - win + 1，与起点一一对应）。

        难度 = 窗口内 vol_5（特征矩阵第 4 列，5期已实现波动率，tanh 压缩）的
        均值——低波动段（趋势清晰、奖励信号强）= 简单，高波动段 = 难。
        用分位（argsort rank / max(1,n-1)）归一化到 [0,1]，跨标的可比
        （"该窗口在本标的历史中属于前百分之几的难度"）。

        window_len: 当前采样窗口长度。缓存按 window_len 区分（配置不变则
        只需算一次；多线程下重复计算无害——纯只读，结果确定）。
        """
        win = max(1, int(window_len))
        cached = self.difficulty
        if cached is not None and cached.shape[0] == max(0, self.n - win + 1) \
                and getattr(self, "_diff_win", None) == win:
            return cached
        n = self.n
        if n < win:
            arr: np.ndarray = np.zeros(1, dtype=float)
            self.difficulty = arr
            self._diff_win = win
            return arr
        vol = self.features[:, 4]  # _VOL_IDX=4：vol_5（tanh 压缩，单调→分位等价）
        view = np.lib.stride_tricks.sliding_window_view(vol, win)
        diff = view.mean(axis=1)                    # (n-win+1,)
        order = np.argsort(diff, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(order), dtype=float)
        ranks /= max(1, len(ranks) - 1)             # [0,1]
        self.difficulty = ranks
        self._diff_win = win
        return ranks


class MultiAssetPool:
    """标的池：构造时预计算各标的特征，episode 时随机采样"标的 × 时间窗"。

    sample() 返回的 dict 含：df_slice / features_slice / factors_slice /
    name / start / window_len——调用方用这些构造独立 TradingEnv。
    线程安全：sample 接受 per-worker rng（numpy Generator 非线程安全，
    绝不共享实例级 rng），构造与采样均在 worker 内串行执行。
    """

    def __init__(self, main_df: pd.DataFrame,
                 main_features: np.ndarray,
                 main_factors: Optional[np.ndarray] = None,
                 extra_dfs: Optional[dict] = None,
                 extra_assets: Optional[list["_Asset"]] = None,
                 state_window: int = 1,
                 backend: str = "numpy",
                 factor_composite: Optional[dict] = None,
                 factor_builder: Optional[Callable[[pd.DataFrame], Optional[np.ndarray]]] = None,
                 window_len: int = 0,
                 extra_mix_ratio: float = 0.5,
                 curriculum: bool = False,
                 curriculum_start: float = 0.3,
                 log=None) -> None:
        """构造标的池。

        Args:
            main_df: 主标的训练段 DataFrame（train_df）
            main_features: 主标的训练段预计算特征（train 段全段，与旧路径同源）
            main_factors: 主标的训练段因子列（(n,) 或 None，长度==len(main_df)）
            extra_dfs: {name: df} 额外标的（按 key 排序保证可复现）
            extra_assets: 预构造好的额外标的列表（evolve 复用已拉取数据时用，
                优先级高于 extra_dfs）
            state_window: 状态窗口（透传 TradingEnv）
            backend: 特征预计算后端（numpy/cupy）
            factor_composite: 级联组合因子复算配方（{weights: {...}}）；
                非空时对每个额外标的复算因子列，保持状态维度一致
            factor_builder: 跨标因子构建器 callable(df)->(n,)；expr/composite
                可跨标复算时由 train_drl 注入，None 表示主标的因子列不可跨标
                （此时带因子的主标的将降级为单标的训练，见 train_drl）
            window_len: episode 切片长度（0=自动取主标的训练段长度）
            extra_mix_ratio: 轨迹来自额外标的的概率（0=只用主标的，1=只用额外）
            curriculum: P1 课程学习——训练早期只从低难度（低波动）窗口采样，
                随进度（sample 的 frac 参数）逐步放开高波动段。False=关闭
                （采样行为与旧路径完全一致）。
            curriculum_start: 课程起始难度分位（0=只采最易窗口，1=不限制）。
                训练进度 frac 从 0→1 时，允许难度 = start + (1-start)*frac。
            log: 日志器（缺省静默）
        """
        self._log = log
        self.state_window = max(1, int(state_window))
        self.backend = backend
        self.factor_composite = dict(factor_composite or {}) or None
        self.factor_builder = factor_builder
        # P1 课程学习：钳到 [0,1]；curriculum_start>=1 视为关闭（不限制难度）
        self.curriculum = bool(curriculum)
        self.curriculum_start = float(min(1.0, max(0.0, curriculum_start)))
        self._curriculum_on = bool(self.curriculum and self.curriculum_start < 1.0)
        # episode 窗口长度：主标的训练段长度（与旧单标的行为一致——主标的跑整段）
        self._main_len = len(main_df)
        if window_len and window_len > 0:
            self.window_len = int(window_len)
        else:
            self.window_len = self._main_len
        # 额外标的概率：钳到 [0,1]
        self.extra_mix_ratio = float(min(1.0, max(0.0, extra_mix_ratio)))
        self._assets: list[_Asset] = []
        self._names: list[str] = []

        # 主标的（名字固定为 "main"，训练日志里用 "main" 展示）
        self._assets.append(_Asset("main", main_df, main_features, main_factors))
        self._names.append("main")

        # 额外标的：extra_assets 优先（evolve 已拉好数据/已算好特征）
        if extra_assets:
            for a in extra_assets:
                self._add(a)
        elif extra_dfs:
            for name in sorted(extra_dfs.keys()):  # 排序保证随机性可复现
                df_i = extra_dfs[name]
                if df_i is None or not isinstance(df_i, pd.DataFrame) or len(df_i) == 0:
                    self._warn("[multi_asset] 忽略无效额外标的 %r（空数据）", name)
                    continue
                feat_i = _precompute_features(df_i, backend=backend)
                factors_i = None
                if self.factor_builder is not None:
                    try:
                        factors_i = self.factor_builder(df_i)
                    except Exception:  # noqa: BLE001
                        factors_i = None
                    if factors_i is None:
                        self._warn("[multi_asset] 标的 %s 因子列构建失败，"
                                   "若主标的有因子列该标的将被排除", name)
                self._add(_Asset(str(name), df_i, feat_i, factors_i))

        self._usable = [a for a in self._assets
                        if a.n >= max(_MIN_BARS, _WARMUP + 50)]
        if len(self._usable) < len(self._assets):
            skipped = [a.name for a in self._assets if a not in self._usable]
            self._warn("[multi_asset] 以下标的K线过短(<%d根)，已排除出采样池: %s",
                       max(_MIN_BARS, _WARMUP + 50), skipped)
        if not self._usable:
            raise ValueError("[multi_asset] 采样池为空：无任何标的满足最小K线数要求")

    def _add(self, asset: _Asset) -> None:
        """加入一个额外标的，校验特征/因子维度与主标的一致（否则无法共享网络）。"""
        if asset.name == "main":
            return
        main_asset = self._assets[0]
        main_dim = main_asset.features.shape[1]
        if asset.features.shape[1] != main_dim:
            self._warn("[multi_asset] 标的 %s 特征维度 %d 与主标的 %d 不一致，"
                       "跳过（无法共享网络）", asset.name,
                       asset.features.shape[1], main_dim)
            return
        if len(main_asset.df.columns) != len(asset.df.columns):
            self._warn("[multi_asset] 标的 %s 列结构 %r 与主标的 %r 不一致，跳过",
                       asset.name, list(asset.df.columns),
                       list(main_asset.df.columns))
            return
        # 因子列维度守卫：主标的有因子、额外标的没有 → 无法共享网络，排除；
        # 主标的无因子、额外标的有 → 丢弃额外因子列（保持同维度）。
        main_has_factor = main_asset.factors is not None
        if main_has_factor and asset.factors is None:
            self._warn("[multi_asset] 主标的有因子列而标的 %s 无因子列，"
                       "跳过（维度不一致）", asset.name)
            return
        if not main_has_factor and asset.factors is not None:
            self._warn("[multi_asset] 主标的无因子列，丢弃标的 %s 的因子列"
                       "（保持状态维度一致）", asset.name)
            asset.factors = None
        if (asset.factors is not None
                and asset.factors.shape[0] != asset.n):
            self._warn("[multi_asset] 标的 %s 因子列长度 %d != K线数 %d，"
                       "丢弃因子列（保持与主标的同维度）", asset.name,
                       asset.factors.shape[0], asset.n)
            asset.factors = None
        self._assets.append(asset)
        self._names.append(asset.name)

    def _warn(self, fmt: str, *args) -> None:
        if self._log is not None:
            self._log.warning(fmt, *args)

    # ---- 采样 ----

    def _allowed_difficulty(self, frac: float) -> float:
        """课程当前允许的最大窗口难度分位（frac∈[0,1] 为训练进度）。

        线性课程：允许难度 = curriculum_start + (1 - curriculum_start) * frac。
        - frac=0（训练刚开始）：只允许最低 curriculum_start 分位的窗口
        - frac=1（训练尾声）：放开全部（难度 1.0 = 不限）
        curriculum 关闭时恒返回 1.0（不限制，与旧路径一致）。
        """
        if not self._curriculum_on:
            return 1.0
        frac = float(min(1.0, max(0.0, frac)))
        return self.curriculum_start + (1.0 - self.curriculum_start) * frac

    def sample(self, rng: np.random.Generator, frac: float = 1.0) -> dict:
        """随机采样一个 episode 的训练切片。

        分布：
        - 以 extra_mix_ratio 概率从额外标的里随机选一个，否则主标的
          （单标的模式：额外标的列表为空 → 恒主标的，与旧行为一致）
        - 在所选标的的数据范围内随机起点取 window_len 长的切片
          （标的数据不足 window_len 时取整段，起点为 0）
        - P1 课程学习：curriculum 开启且未到全放开阶段时，只从难度分位
          ≤ _allowed_difficulty(frac) 的起点里采样（拒绝采样，最多重试
          20 次，超限退回最低难度窗口——保证始终返回合法切片）。

        rng: per-worker 独立随机源（保证 seed 可复现、线程安全）。
        frac: 训练进度 [0,1]（课程难度放开用；关闭时忽略）。
        返回 dict: {df_slice, features_slice, factors_slice, name, start, window_len}
        """
        usable = self._usable  # 已通过最小K线数守卫的标的
        if self.extra_mix_ratio > 0 and len(usable) > 1:
            extras = [a for a in usable if a is not usable[0]] or usable[1:]
            if rng.random() < self.extra_mix_ratio:
                asset = extras[rng.integers(0, len(extras))]
            else:
                asset = usable[0]
        else:
            asset = usable[0]

        n = asset.n
        win = min(self.window_len, n)
        # 数据不足最小可学长度时用整段（起点 0）；否则随机起点
        max_start = max(0, n - win)
        allowed = self._allowed_difficulty(frac)

        start = 0
        if max_start > 0:
            if self._curriculum_on and allowed < 1.0:
                # 课程拒绝采样：在难度分位 ≤ allowed 的起点里随机挑
                ranks = asset.difficulty_ranks(win)
                # 候选起点索引（rank 数组与起点一一对应）
                feasible = np.nonzero(ranks[:max_start + 1] <= allowed)[0]
                for _ in range(20):
                    if feasible.size > 0:
                        start = int(feasible[rng.integers(0, feasible.size)])
                        break
                    # 重试期窗口全被拒（罕见：数据整段高波动）→ 取最低难度
                    start = int(np.argmin(ranks[:max_start + 1]))
                    break
                else:
                    start = int(np.argmin(ranks[:max_start + 1]))
            else:
                start = int(rng.integers(0, max_start + 1))

        df_slice = asset.df.iloc[start:start + win]
        features_slice = asset.features[start:start + win]
        factors_slice = (asset.factors[start:start + win]
                         if asset.factors is not None else None)
        return {
            "df_slice": df_slice,
            "features_slice": features_slice,
            "factors_slice": factors_slice,
            "name": asset.name,
            "start": start,
            "window_len": win,
            "n": n,
        }

    @property
    def n_assets(self) -> int:
        """采样池中可用标的数（含主标的）。"""
        return len(self._usable)

    @property
    def extra_names(self) -> list[str]:
        """可用额外标的名单（不含主标的，按加入顺序）。"""
        return [a.name for a in self._usable if a.name != "main"]


def build_pool(main_df: pd.DataFrame, main_features: np.ndarray,
               extra_dfs: Optional[dict] = None,
               extra_assets: Optional[list] = None,
               state_window: int = 1,
               backend: str = "numpy",
               factor_composite: Optional[dict] = None,
               window_len: int = 0,
               extra_mix_ratio: float = 0.5,
               curriculum: bool = False,
               curriculum_start: float = 0.3,
               log=None) -> MultiAssetPool:
    """便捷构造：main_features 可由调用方预计算（主标的复用旧路径特征）。"""
    return MultiAssetPool(
        main_df, main_features, extra_dfs=extra_dfs, extra_assets=extra_assets,
        state_window=state_window, backend=backend,
        factor_composite=factor_composite, window_len=window_len,
        extra_mix_ratio=extra_mix_ratio, curriculum=curriculum,
        curriculum_start=curriculum_start, log=log)


def _build_composite_factor(df: pd.DataFrame, weights: dict) -> Optional[np.ndarray]:
    """按配方在任意标的 df 上复算级联组合因子列（长度 == len(df)）。

    与 walkforward.build_factor_column 同口径：expanding z（无前视，只用
    截至当期统计量）+ 按权重绝对值和归一；NaN 填 0。无有效权重返回 None。
    """
    w = {str(k): float(v) for k, v in (weights or {}).items() if v}
    if not w:
        return None
    try:
        from factors.engine import compute_factor_matrix
        from factors.mining import composite_factor
        mat = compute_factor_matrix(df, keys=list(w))
        combo = composite_factor(mat, w, method="ic", z_mode="expanding")
        vals = np.asarray(combo.to_numpy(float), dtype=float).reshape(-1)
        if len(vals) != len(df):
            return None
        return np.nan_to_num(vals, nan=0.0)
    except Exception:  # noqa: BLE001 —— 因子复算失败不应阻断训练
        return None


def make_worker_env(sample: dict, cfg: dict, backend: str = "numpy") -> TradingEnv:
    """用 pool.sample 的切片构造独立 TradingEnv（worker 内调用）。

    与旧 _collect_one 中构造环境的口径一致：滑点/fee/波动惩罚等全部
    透传 cfg。precomputed_features 用切片特征矩阵（不再重复计算）。
    """
    from .env import TradingEnv as _T
    from .env import policy_env_kwargs as _policy_env_kwargs
    return _T(
        sample["df_slice"],
        start_cash=float(cfg.get("start_cash", 10000.0)),
        fee_rate=float(cfg.get("fee_rate", 0.001)),
        vol_penalty=float(cfg.get("vol_penalty", 20.0)),
        cost_scale=float(cfg.get("cost_scale", 1.0)),
        slippage=float(cfg.get("slippage", 0.0005)),
        min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
        extra_factors=sample["factors_slice"],
        state_window=int(cfg.get("state_window", 1)),
        backend=backend,
        reward_dd_penalty=float(cfg.get("reward_dd_penalty", 0.0)),
        reward_losing_penalty=float(cfg.get("reward_losing_penalty", 0.0)),
        reward_trend_align=float(cfg.get("reward_trend_align", 0.0)),
        # 2026-10 做空奖惩设计：爆仓追加惩罚透传（多标采样 env 同源）
        reward_bankrupt_penalty=float(cfg.get("reward_bankrupt_penalty", 0.0)),
        tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
        participation_rate=float(cfg.get("participation_rate", 0.0)),
        funding_rate=float(cfg.get("funding_rate", 0.0)),
        precomputed_features=sample["features_slice"],
        # 奖励口径 / 决策间隔与主训练路径同源（见 env.policy_env_kwargs）
        **_policy_env_kwargs(cfg),
    )
