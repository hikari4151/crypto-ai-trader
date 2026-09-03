"""RL 因子组合挖掘环境（AlphaForge 范式的 RL 版）。

把"因子池 → 组合因子"的决策建模为序列决策 MDP：
- **状态 s_t**：候选因子池的表现特征（每个因子：滚动 IC / ICIR / 换手 / 与已选集合的最大相关 / 是否已选）
  + 已选数量 + 当前组合在验证段的 fitness
- **动作 a_t**：从候选池选择 1 个因子加入组合（n_factors 个动作），或"结束"（动作 n）
- **奖励 r_t**：选择后组合 fitness 的增量（放大）+ 冗余惩罚（与已选因子高度相关）+ 重复选择惩罚

关键设计（防过拟合 / 无前视）：
1. **时间切分**：因子表现特征取自训练段（决策信息），组合 fitness 在验证段评估
   （agent 从未在验证段上"见过"奖励以外的东西——训练/验证分离）
2. **滚动 IC 滞后**：IC 用非重叠窗口计算并 shift(h) 延迟到"已实现"才可用（与
   dynamic_composite 同一套无前视处理，杜绝用未来收益做决策）
3. **多样性内建奖励**：选择与已选集合高相关的因子得负奖励 → 学会选互补因子
4. **OOS 只报告**：训练完成后用 OOS 段（agent 从未见过）做最终安检，与
   factor_quality_gate 门槛衔接

输出：选出因子组合 + IC 加权权重 → 合成组合因子，接入现有因子体系。
"""
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from factors.analysis import factor_quality_gate
from factors.mining import composite_factor

log = logging.getLogger(__name__)

# 每因子状态特征维度：滚动IC, |滚动IC|, ICIR, 换手, 与已选最大相关, 是否已选
_FEAT_PER_FACTOR = 6


class FactorMiningEnv:
    """RL 因子组合挖掘环境（gym 风格：reset/step）。"""

    def __init__(self, mat: pd.DataFrame, close: pd.Series, h: int = 1,
                 ic_window: int = 120, max_steps: int = 6,
                 train_ratio: float = 0.6, val_ratio: float = 0.2,
                 corr_threshold: float = 0.85,
                 reward_scale: float = 100.0,
                 duplicate_penalty: float = 0.05,
                 seed: Optional[int] = None,
                 precomputed_ics: Optional[pd.DataFrame] = None,
                 precomputed_icir: Optional[pd.DataFrame] = None) -> None:
        self.mat = mat
        self.close = close
        self.h = max(1, int(h))
        self.ic_window = ic_window
        self.max_steps = max_steps
        self.corr_threshold = corr_threshold
        self.reward_scale = reward_scale
        self.duplicate_penalty = duplicate_penalty
        self.cols = list(mat.columns)
        self.n_factors = len(self.cols)
        n = len(mat)
        self.n_train = max(200, int(n * train_ratio))
        self.n_val = min(n - 30, max(self.n_train + 100, int(n * (train_ratio + val_ratio))))
        if self.n_val >= n - 30:
            raise ValueError("数据量不足：验证段过短，请增加数据")
        self._rng = np.random.default_rng(seed)

        # ---- 无前视的因子表现特征（预计算） ----
        # 支持传入预计算的 IC/ICIR（避免并行收集时 8 路重复计算 _rolling_ic）
        if precomputed_ics is not None:
            self._ics = precomputed_ics.iloc[:n]
            if precomputed_icir is not None:
                self._icir = precomputed_icir.iloc[:n]
            else:
                # ICIR 需从预计算 ICS 重新算（依赖 self.ic_window）
                self._icir = self._ics.rolling(ic_window * 5, min_periods=ic_window * 2).mean() / (
                    self._ics.rolling(ic_window * 5, min_periods=ic_window * 2).std() + 1e-12)
        else:
            from factors.analysis import _rolling_ic
            ics = {c: _rolling_ic(mat[c], close, h=self.h, method="rank", window=ic_window)
                   for c in self.cols}
            self._ics = pd.DataFrame(ics, index=mat.index).shift(self.h).ffill()
            self._icir = self._ics.rolling(ic_window * 5, min_periods=ic_window * 2).mean() / (
                self._ics.rolling(ic_window * 5, min_periods=ic_window * 2).std() + 1e-12)
        # 换手率（全量，静态）
        from factors.analysis import factor_turnover
        self._turnover = np.asarray([factor_turnover(mat[c]) for c in self.cols])
        # 因子相关矩阵（训练段，冗余惩罚用）
        self._corr = mat.iloc[:self.n_train].corr().to_numpy()

        # P2-12：组合因子 z-score 统计量预计算一次（expanding 口径与 composite_factor
        # 逐位一致：只用截至当期的 mean/std，无前视）。fitness 每步直接从预计算 z
        # 加权求和，不再每步全量重算各列 expanding 统计量（O(n×k) → O(k)）。
        self._z_exp: dict[str, pd.Series] = {}
        for c in self.cols:
            s = mat[c]
            self._z_exp[c] = ((s - s.expanding().mean()) / (s.expanding().std() + 1e-12)).fillna(0.0)

        # ---- 账户状态 ----
        self._t = 0            # 决策时点（训练段内的随机位置）
        self._steps = 0        # 已执行步数（重复选择也推进，保证 episode 必终止）
        self._selected: list[int] = []  # 已选因子索引
        self._done = False
        # P2-12：fitness 缓存 {(t, tuple(sorted(selected))): fitness}——
        # 同一步内 old/new fitness 与状态构造会重复计算同一集合，直接命中
        self._fitness_cache: dict[tuple, float] = {}

    # ---- 兼容 ACAgent.collect_episode（max_steps 取环境上限） ----
    @property
    def n(self) -> int:
        return self.max_steps + 1

    @property
    def warmup(self) -> int:
        return 0

    @property
    def state_dim(self) -> int:
        return self.n_factors * _FEAT_PER_FACTOR + 2

    @property
    def n_actions(self) -> int:
        # 无"结束"动作：episode 强制选满 max_steps 个因子（组合大小固定，
        # RL 只学"选哪些"——避免智能体学会"立即结束"的退化策略）
        return self.n_factors

    def _state_feats(self) -> np.ndarray:
        """每因子特征（t 时点已实现信息）。"""
        ic_row = self._ics.iloc[self._t].to_numpy(float)
        icir_row = self._icir.iloc[self._t].to_numpy(float)
        feats = np.zeros((self.n_factors, _FEAT_PER_FACTOR))
        for i in range(self.n_factors):
            ic = ic_row[i]
            ic = 0.0 if ic != ic else ic  # NaN→0（无已实现 IC）
            icir = icir_row[i]
            icir = 0.0 if icir != icir else icir
            # 与已选集合的最大相关
            max_corr = 0.0
            if self._selected:
                max_corr = float(np.max(np.abs(self._corr[i, self._selected])))
            feats[i] = [ic, abs(ic), icir, self._turnover[i],
                        max_corr, 1.0 if i in self._selected else 0.0]
        return feats

    def _composite_fitness(self) -> float:
        """当前已选组合在验证段的 fitness（0=未选）。

        P2-12：结果按 (决策时点 t, 已选因子集合) 缓存——同一 t 内集合单调增长，
        old/new fitness 与状态构造重复计算同一集合时直接命中（奖励路径只对
        真正的新集合做一次合成 + 安检门）；z-score 统计量已在 __init__ 预计算
        （expanding 口径，无前视），合成从 O(n×k) 降到 O(k)。
        """
        if not self._selected:
            return 0.0
        key = (self._t, tuple(sorted(self._selected)))
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        weights = {}
        for i in self._selected:
            ic = self._ics.iloc[self._t, i]
            ic = 0.0 if ic != ic else ic
            weights[self.cols[i]] = ic if abs(ic) > 1e-6 else 0.01  # 未实现 IC 给极小权重
        combo = composite_factor(self.mat, weights, method="ic", z_mode="expanding",
                                 z_map=self._z_exp)
        gate = factor_quality_gate(combo.iloc[self.n_train:self.n_val],
                                   self.close.iloc[self.n_train:self.n_val],
                                   h=self.h)
        fitness = gate["fitness"]
        self._fitness_cache[key] = fitness
        return fitness

    def reset(self) -> np.ndarray:
        """回到起点：随机决策时点 + 清空选择，返回初始状态。"""
        # 决策时点须在训练段内且滚动 IC 已实现（至少 2*ic_window 根）
        lo = min(2 * self.ic_window, self.n_train // 2)
        hi = max(lo + 1, self.n_train - 1)
        self._t = int(self._rng.integers(lo, hi))
        self._steps = 0
        self._selected = []
        self._done = False
        # P2-12：缓存键含 t，reset 后旧键全部失效，整体清空防内存膨胀
        self._fitness_cache = {}
        return self._build_state()

    def _build_state(self) -> np.ndarray:
        feats = self._state_feats().reshape(-1)
        global_feats = np.asarray([len(self._selected) / self.max_steps,
                                   self._composite_fitness()], dtype=float)
        return np.concatenate([feats, global_feats])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行动作：选择因子（或结束），返回 (next_state, reward, done, info)。"""
        assert not self._done, "环境已终止，请先 reset"
        assert 0 <= action < self.n_actions, f"非法动作 {action}"

        old_fitness = self._composite_fitness()
        info: dict[str, Any] = {"action": action, "done": False}
        reward = 0.0

        if action in self._selected:
            # 重复选择：轻微负奖励（浪费步数，学会不重复）
            reward = -self.duplicate_penalty
            info["reason"] = "duplicate"
        else:
            # 冗余惩罚：与已选集合高度相关
            penalty = 0.0
            if self._selected:
                max_corr = float(np.max(np.abs(self._corr[action, self._selected])))
                if max_corr > self.corr_threshold:
                    penalty = self.duplicate_penalty * 4
            self._selected.append(action)
            new_fitness = self._composite_fitness()
            # 奖励 = fitness 增量（放大）+ 冗余惩罚
            reward = self.reward_scale * (new_fitness - old_fitness) - penalty
            info["reason"] = "select"
            info["fitness"] = new_fitness

        # 选满 max_steps 个因子或步数耗尽后 episode 结束（组合大小固定，必终止）
        self._steps += 1
        if len(self._selected) >= self.max_steps or self._steps >= self.max_steps * 3:
            self._done = True
            info["done"] = True
            info["selected"] = [self.cols[i] for i in self._selected]
            info["final_fitness"] = self._composite_fitness()
        return self._build_state(), float(reward), self._done, info
