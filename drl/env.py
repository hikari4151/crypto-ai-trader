"""MDP 交易环境：将单标的交易建模为马尔可夫决策过程。

- **状态 s_t**：归一化的价量 / 波动率 / 趋势与风险 regime / 当前持仓与账户状态
- **动作 a_t**：离散目标仓位档位（0 / 0.25 / 0.5 / 0.75 / 1.0），表示目标持仓比例
  （买入→目标位上调、卖出→目标位下调、持仓→保持）
- **奖励 r_t**：组合收益率 - 交易成本 - 波动率风险惩罚
  （该惩罚正是"波动率上升时降低仓位"的行为来源：高波动环境下满仓会得到负奖励）

环境完全向量化预计算特征序列，训练时按索引 O(1) 取状态，可高速反复试错。
"""
from typing import Optional

import numpy as np
import pandas as pd

# 离散动作 = 目标仓位档位
# 长仓档（旧行为，默认）：只能做多/空仓
LONG_ONLY_BUCKETS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
LONG_ONLY_NAMES = ["清仓", "轻仓", "半仓", "重仓", "满仓"]
# 对称档（short_enabled=True）：−1~+1，含做空与精确空仓。
# 为什么需要它：动作空间只能做多/空仓时，熊市样本外"最优多头策略"就是全程空仓，
# 相对基准的超额全靠"没参与"（见 docs/DRL成本标定-决策频率与滑点复核.md §4）——
# 做空是多头市场之外唯一能真正创造正收益的路径。
SHORT_BUCKETS = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
SHORT_NAMES = ["满空", "重空", "半空", "轻空", "清仓", "轻仓", "半仓", "重仓", "满仓"]

# 兼容旧引用：模块级常量 = 默认（长仓）动作空间
ACTION_BUCKETS = LONG_ONLY_BUCKETS
ACTION_NAMES = LONG_ONLY_NAMES


def action_buckets_for(short_enabled: bool = False) -> np.ndarray:
    """动作空间（单一定义处）。

    short_enabled=True → 对称 9 档（−1~+1）；False → 旧的 5 档长仓空间。
    动作空间必须随**模型**走（写进模型文件、部署端按模型的档位执行），
    否则"训练会做空、部署按长仓档位解释同一个 argmax"会静默漂移。
    """
    return SHORT_BUCKETS if short_enabled else LONG_ONLY_BUCKETS


def action_names_for(short_enabled: bool = False) -> list:
    """与动作空间配套的可读名（日志/前端用）。"""
    return list(SHORT_NAMES if short_enabled else LONG_ONLY_NAMES)


def buckets_require_short(buckets) -> bool:
    """该动作空间是否含做空档（任一档位为负）。

    部署端能力判定用：撮合内核（backtest/_matching.py）当前是现货语义
    （`sell` 必须 position>0 且 qty≤position），**无法建立负仓位**——
    含做空档的模型在执行层支持之前必须显式阻断部署，绝不允许静默按长仓执行。
    """
    try:
        return bool(np.any(np.asarray(buckets, dtype=float) < -1e-12))
    except Exception:  # noqa: BLE001 —— 形状异常按"不涉及做空"处理（保守回退旧语义）
        return False

# 奖励口径 / 决策间隔的配置键（训练·验证·OOS·长窗观测必须同源，否则口径漂移：
# 训练时每 4 根决策一次、评估时每根决策一次，模型行为与收益就完全不是一回事）
def policy_env_kwargs(cfg: dict) -> dict:
    """从训练配置里取出环境口径参数（reward_mode/benchmark_position/risk_coef/decision_interval）。

    单一定义处：train_drl 的主环境、并行 worker、val/val 子段、OOS 段、蒸馏取样、
    持续进化的长窗观测全部用它，保证"训练时学的东西"和"评估时量的东西"同一口径。
    缺键时返回旧行为（absolute / k=1），向后兼容。
    """
    cfg = cfg or {}
    return {
        "reward_mode": str(cfg.get("reward_mode", "absolute") or "absolute"),
        "benchmark_position": float(cfg.get("benchmark_position", 1.0)),
        "risk_coef": float(cfg.get("risk_coef", 0.5)),
        "decision_interval": max(1, int(cfg.get("decision_interval", 1) or 1)),
        "short_enabled": bool(cfg.get("short_enabled", False)),
    }


def policy_env_kwargs_from_settings(prefix: str = "evolve_strategy_drl") -> dict:
    """同上，但直接读 settings（供不经过 train_drl 的观测/回放路径用）。"""
    from config.settings import settings
    return policy_env_kwargs({
        "reward_mode": getattr(settings, f"{prefix}_reward_mode", "absolute"),
        "benchmark_position": getattr(settings, f"{prefix}_benchmark_position", 1.0),
        "risk_coef": getattr(settings, f"{prefix}_risk_coef", 0.5),
        "decision_interval": getattr(settings, f"{prefix}_decision_interval", 1),
        "short_enabled": getattr(settings, f"{prefix}_short_enabled", False),
    })



class TradingEnv:
    """基于历史K线的交易环境（可重置、可 step）。"""

    def __init__(self, df: pd.DataFrame,
                 start_cash: float = 10000.0,
                 fee_rate: float = 0.001,
                 vol_penalty: float = 0.5,
                 cost_scale: float = 1.0,
                 slippage: float = 0.0,
                 min_trade_zone: float = 0.05,
                 warmup: int = 60,
                 backend: str = "numpy",
                 extra_factors: Optional[np.ndarray] = None,
                 state_window: int = 1,
                 # 可选奖励塑形（系数=0 即关闭，默认关；A/B 验证后开启）
                 reward_dd_penalty: float = 0.0,      # 回撤加深惩罚（含 dd>12% 硬约束）
                 reward_losing_penalty: float = 0.0,  # 连亏惩罚（3 连亏起二次斜坡）
                 reward_trend_align: float = 0.0,     # 趋势一致性奖励（低波动趋势持仓同向加分）
                 tail_risk_penalty: float = 0.0,      # 尾部风险惩罚（单根超大幅回撤额外罚）
                 # ---- P2-12 成交/成本约束（与回测/纸面同口径，默认关保持旧行为） ----
                 participation_rate: float = 0.0,     # 调仓数量上限 = 该K线成交量×参与率（0=不限）
                 funding_rate: float = 0.0,           # 每根K线按持仓价值收取的资金费率（0=关闭）
                 precomputed_features: Optional[np.ndarray] = None,  # 预计算特征，避免并行 worker 重复计算
                 # ---- 2026-10 根本性逻辑错误修复（详见 docs/DRL训练根本性逻辑错误-诊断报告.md）----
                 reward_mode: str = "absolute",       # "absolute"=旧行为 | "excess"=基准相对（推荐）
                 benchmark_position: float = 1.0,     # excess 模式的基准仓位（1.0=买入持有）
                 risk_coef: float = 0.5,              # excess 模式跟踪误差惩罚系数（bp 量级，见 _bar_reward）
                 decision_interval: int = 1,          # 每 k 根K线决策一次（k=1=旧行为，逐根决策）
                 short_enabled: bool = False,         # 动作空间是否含做空档（−1~+1）；False=旧的长仓 5 档
                 ) -> None:
        self.df = df
        self.start_cash = start_cash
        self.fee_rate = fee_rate
        self.vol_penalty = vol_penalty      # 波动率惩罚系数（absolute 模式）
        # ============ 奖励口径（F1：奖励最优解不能再是"永不交易"） ============
        # 旧口径（absolute）的问题：奖励 = 组合收益(bp) - vol_penalty·(仓位·vol%)²，
        # 风险项是无条件"持仓税"——只要持仓就扣分，与这笔仓位对不对无关；而空仓奖励
        # 恒等于 0。随机游走下任何非零仓位的期望奖励 = -E[惩罚] < 0 ⇒ 全局最优解就是
        # "永不交易"（实测：行情涨 8.6%~26.9%、账户收益为正，持仓策略累计奖励却全是
        # 大幅负值，躺平恰好 0 为最大值）。RL 于是把全部算力学成了"不做任何事"。
        # excess 口径：奖励 = 相对基准的超额收益 - 跟踪误差惩罚。
        # 空仓在上涨行情里同样被扣分（机会成本），"躺平"不再免费；对称的跟踪惩罚
        # 也让"高波动降仓"的意图重新成立（偏离基准才受罚）。
        self.reward_mode = str(reward_mode or "absolute").lower()
        if self.reward_mode not in ("absolute", "excess"):
            self.reward_mode = "absolute"
        self.benchmark_position = float(min(1.0, max(0.0, benchmark_position)))
        # risk_coef：单位与收益项一致（bp / (%波动率)²）。量级标定说明：
        # 旧 vol_penalty=40 在 bp 口径下的惩罚是 40·(pos·vol%)²，1h/vol≈0.6% 时满仓
        # 每根约 14bp —— 是行情自身漂移（零点几 bp/根）的数十倍，等于"持仓必亏"。
        # 0.5 时同样条件下约 0.18bp，只做"高波动别满仓"的平局裁决，不吞掉收益信号。
        self.risk_coef = max(0.0, float(risk_coef))
        # ============ 决策间隔（F2：探索成本与样本效率） ============
        # 旧环境每根K线都重新决策，而动作是"目标仓位"：初始策略近均匀 ⇒ 平均 |Δ仓位|
        # ≈0.4 ⇒ 每根 0.06% 摩擦 ⇒ 2939 根 ≈ -83%（实测 -82%）。也就是说"随机试错"
        # 本身就是自杀，优势函数里"别交易"的信号压过一切真实行情信号。
        # decision_interval=k 时：每 k 根K线只决策一次、其间保持仓位，成本收一次，
        # 奖励按 k 根累计。探索成本降 k 倍，且每个决策覆盖 k 根行情 ⇒ 同样墙钟内
        # 见过的行情多 k 倍（样本效率 = 用户要的"几个月顶几年"）。
        self.decision_interval = max(1, int(decision_interval or 1))
        # ============ 动作空间（F3：放开做空） ============
        # 只能做多/空仓时，熊市样本外"最优多头策略"= 全程空仓，超额全靠"没参与"；
        # 做空是唯一能在那类窗口真正创造正收益的路径。
        # 幂等语义（与多头一致）：目标仓位是**权益比例**，价格漂移会被下一次决策
        # 拉回目标（恒定比例再平衡）；做空仓位的保证金/爆仓由 equity≤0 守卫兜底。
        self.short_enabled = bool(short_enabled)
        self.action_buckets = action_buckets_for(self.short_enabled)
        self.action_names = action_names_for(self.short_enabled)
        self.n_actions = int(len(self.action_buckets))
        self.cost_scale = cost_scale        # 交易成本缩放
        # D2：滑点（比例）。默认 0 向后兼容；train_drl 按 cfg.slippage（默认 0.0005，
        # 与回测/纸面 paper_slippage 同口径）传入，消除"训练能赚、实盘亏在摩擦"偏差。
        self.slippage = float(slippage)
        self.min_trade_zone = min_trade_zone  # 调仓死区：与目标仓位差异小于此比例则不调仓
        # P2-12：成交量参与率上限与资金费率（0=关闭，向后兼容）
        self.participation_rate = float(participation_rate or 0.0)
        self.funding_rate = float(funding_rate or 0.0)
        self.reward_dd_penalty = reward_dd_penalty
        self.reward_losing_penalty = reward_losing_penalty
        self.reward_trend_align = reward_trend_align
        self.tail_risk_penalty = tail_risk_penalty
        # 奖励塑形启用日志（观测点，不改变塑形计算逻辑或训练结果）：
        # 任一塑形系数 >0 时打印启用信息，供部署排查"奖励分布为何变了"
        if reward_dd_penalty > 0 or reward_losing_penalty > 0 or reward_trend_align > 0 or tail_risk_penalty > 0:
            import logging
            logging.getLogger(__name__).info(
                "[drl] 奖励塑形已启用: dd=%s losing=%s trend=%s",
                reward_dd_penalty, reward_losing_penalty, reward_trend_align)
        self.warmup = warmup
        self.backend = backend
        # 状态窗口：堆叠最近 N 根K线的特征，给 MLP 时序记忆（单点特征看不到形态变化）。
        # 默认 1=向后兼容旧模型；>1 时 state_dim 放大 N 倍
        self.state_window = max(1, int(state_window))
        closes = df["close"].to_numpy(float)
        self.n = len(closes)
        self._closes = closes
        # P2：预计算最近 5 根已实现波动率序列（ddof=1，与特征列 vol_5 的
        # pandas rolling(5).std() 同口径——此前逐 step 用 np.std(ddof=0) 重算
        # 且与特征口径不一致）。首 3 根不足窗口期为 0（与原实现 len<3→0 一致）
        _rets = np.zeros_like(closes)
        if self.n > 1:
            _rets[1:] = np.diff(closes) / closes[:-1]
        _vol5 = np.zeros(self.n)
        # 自治优化 E1：_vol5 预计算向量化（sliding_window_view 批量 np.std）。
        # 原实现为逐元素循环（每根K线一次 np.std+切片，环境构造在每 worker×每轮
        # 轨迹发生一次，占训练 wall 约 10%）；批量版 138x 加速且与逐元素版
        # bitwise 一致（.optim/verify_e1_vol5_batch.py 全数据集验证，零浮点尾差）。
        if self.n >= 3:
            if self.n < 5:
                for _i in range(2, self.n):
                    _vol5[_i] = float(np.std(_rets[max(0, _i - 4):_i + 1], ddof=1) * 100.0)
            else:
                for _i in (2, 3):
                    _vol5[_i] = float(np.std(_rets[:_i + 1], ddof=1) * 100.0)
                _view = np.lib.stride_tricks.sliding_window_view(_rets, 5)
                _vol5[4:] = np.std(_view, axis=1, ddof=1) * 100.0
        self._vol5_series = _vol5
        # P2-12：成交量序列（参与率约束用；缺 volume 列则视为无限流动性）
        if "volume" in df.columns:
            self._volumes = df["volume"].to_numpy(float)
        else:
            self._volumes = np.full(self.n, 1e12)
        # 使用预计算特征（避免在并行 worker 中重复计算 _precompute_features）
        if precomputed_features is not None:
            self._features = precomputed_features
        else:
            self._features = _precompute_features(df, backend=backend)  # (n, feat_dim)
        # P2：state_window>1 时预计算堆叠状态矩阵——_state_feats 每步切片+拼接+
        # reshape 是训练热路径（每 env×每 step 一次），改为一次性查表。
        # 头部零填充语义与原实现完全一致（t<w-1 时补零）
        if self.state_window > 1:
            _w = self.state_window
            _pad = np.zeros((_w - 1, self._features.shape[1]))
            _padded = np.vstack([_pad, self._features])
            _win = np.lib.stride_tricks.sliding_window_view(_padded, _w, axis=0)
            self._state_stack = np.ascontiguousarray(_win).reshape(self.n, -1)
        else:
            self._state_stack = None
        # 外部因子信号列（如模型因子/自定义表达式因子），追加到状态末端：
        # 训练与部署使用同一表达式计算（见 train_drl / rl_adaptive），保证状态口径一致
        if extra_factors is not None:
            self._extra = np.asarray(extra_factors, dtype=float)
            if self._extra.ndim == 1:
                self._extra = self._extra.reshape(-1, 1)
            if len(self._extra) != self.n:
                raise ValueError(f"extra_factors 长度 {len(self._extra)} 与K线数 {self.n} 不一致")
        else:
            self._extra = None
        self._state_dim = (self._features.shape[1] * self.state_window
                           + 2 + (self._extra.shape[1] if self._extra is not None else 0))
        # 账户状态
        self._t = 0
        self._cash = start_cash
        self._qty = 0.0
        self._entry_price: Optional[float] = None
        self._prev_equity = start_cash
        self._done = False
        # 奖励塑形状态（回撤峰值 / 连亏计数）
        self._peak_equity = start_cash
        self._last_dd = 0.0
        self._losing_streak = 0
        # 成本可观测性（只记账，不影响任何行为/数值）：累计被收费的换手基数与现金成本
        self._turnover_units = 0.0
        self._cost_paid = 0.0
        self._bankrupt = False

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def reset(self, start_offset: int = 0) -> np.ndarray:
        """回到起点，返回初始状态。

        start_offset>0：从 warmup 后随机偏移处起跑（2026-10 审计修复 LOW——
        蒸馏多次 rollout 用随机起点制造状态多样性；缺省 0 保持旧行为逐位兼容）。
        """
        self._t = self.warmup + max(0, int(start_offset or 0))
        if self._t > self.n - 3:
            self._t = max(self.warmup, self.n - 3)
        self._cash = self.start_cash
        self._qty = 0.0
        self._entry_price = None
        self._prev_equity = self.start_cash
        self._done = False
        # P0-5：奖励塑形状态必须随 episode 重置——曾遗留上一段的峰值权益/回撤/
        # 连亏计数，新 episode 首根K线就带上旧分布 → 塑形奖励失真（首步被多罚/漏罚）
        self._peak_equity = self.start_cash
        self._last_dd = 0.0
        self._losing_streak = 0
        # 成本记账随 episode 重置（跨 episode 累计会让换手/成本口径失真）
        self._turnover_units = 0.0
        self._cost_paid = 0.0
        self._bankrupt = False
        # P4-E1：最终权益/仓位快照（episode 收尾后由 step 写入最后一次 equity_next）。
        # 曾直接读 _closes[_t]，而 done 分支把 _t 钳回 n-2——最后一根K线的收益
        # 计入 total_ret 却不在 final_equity 里（两指标互相矛盾的 off-by-one）。
        self._final_equity = self.start_cash
        self._final_position_ratio = 0.0
        return self._build_state()

    def _state_feats(self) -> np.ndarray:
        """当前时点的特征向量：state_window>1 时堆叠最近 N 根（早期零填充）。

        P2：state_window>1 时直接查 __init__ 预计算的堆叠矩阵（行=逐时点展开）。
        """
        if self.state_window <= 1:
            return self._features[self._t]
        if self._state_stack is not None:
            return self._state_stack[self._t]
        # 兜底（理论不可达）：与原实现一致的逐次拼接
        t = self._t
        w = self.state_window
        start = max(0, t - w + 1)
        seg = self._features[start:t + 1]
        if len(seg) < w:
            pad = np.zeros((w - len(seg), self._features.shape[1]))
            seg = np.concatenate([pad, seg], axis=0)
        return seg.reshape(-1)

    def _build_state(self) -> np.ndarray:
        st = np.concatenate([
            self._state_feats(),
            [float(self._pos_ratio()), float(self._pnl_ratio())],
        ])
        if self._extra is not None:
            st = np.concatenate([st, self._extra[self._t]])
        return st

    def _pos_ratio(self) -> float:
        if self._prev_equity <= 0:
            return 0.0
        return (self._qty * self._closes[self._t]) / self._prev_equity

    def _pnl_ratio(self) -> float:
        """持仓浮动盈亏（状态特征）：做空时符号相反。

        做空的"浮盈"是价格下跌，用 `(entry − price)/entry`；多头仍是
        `(price − entry)/entry`。纯多头（qty>0）逐位不变。
        """
        if not self._entry_price or self._entry_price <= 0:
            return 0.0
        if self._qty < 0:
            return (self._entry_price - self._closes[self._t]) / self._entry_price
        return (self._closes[self._t] - self._entry_price) / self._entry_price

    def _equity(self, price: float) -> float:
        return self._cash + self._qty * price

    def _recent_vol_5(self) -> float:
        """最近 5 根K线的已实现波动率（%），快速反映波动率 regime。

        P2：改为查询 __init__ 预计算的序列（值不变，ddof=1 已与特征列 vol_5
        对齐），逐 step 重算 np.diff+np.std 被移除（奖励路径每步调用）。
        """
        if self.n == 0:
            return 0.0
        return float(self._vol5_series[self._t])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行动作，推进一根K线。

        action: 动作索引（目标仓位档位，见 self.action_buckets / 模块 ACTION_BUCKETS）。
        返回 (next_state, reward, done, info)。
        """
        assert not self._done, "环境已终止，请先 reset"
        assert 0 <= action < self.n_actions, f"非法动作 {action}"

        price_now = self._closes[self._t]
        price_next = self._closes[self._t + 1] if self._t + 1 < self.n else price_now

        # ---- 1. 按目标仓位在【当前收盘】调仓（下一根K线才产生收益，无前视） ----
        target_ratio = float(self.action_buckets[action])
        equity = self._equity(price_now)
        cur_value = self._qty * price_now
        cur_ratio = cur_value / max(equity, 1e-9)
        diff = target_ratio - cur_ratio
        # 调仓死区：与目标仓位差异过小则不交易，减少摩擦损耗
        if abs(diff) > self.min_trade_zone and equity > 0:
            qty_delta = (diff * equity) / price_now
            # P2-12：成交量参与率约束——调仓数量不得超过决策期内累计成交量×参与率
            # （流动性限制，与回测 P0-2 participation_rate 同口径；0=不限，向后兼容）。
            # 2026-10 审计修复（LOW）：k>1 时一次调仓覆盖 k 根，按**单根**成交量算
            # 上限会被系统性低估 k 倍（成交本可分摊到决策期各根）——用决策期累计。
            if self.participation_rate > 0:
                _k_end = min(self._t + self.decision_interval, self.n)
                vol_cap = float(np.sum(self._volumes[self._t:_k_end])) * self.participation_rate
                if vol_cap > 0:
                    max_delta = (abs(diff) * equity) / price_now
                    if qty_delta > 0:
                        qty_delta = min(qty_delta, max_delta, vol_cap)
                    else:
                        qty_delta = max(qty_delta, -max_delta, -vol_cap)
            # D2：成本率 = 手续费 + 滑点（与回测/纸面 paper_slippage 同口径）。
            # 滑点按成交额比例计（买卖对称，近似实际滑点成本），默认 0 向后兼容。
            unit_cost = self.fee_rate * self.cost_scale + self.slippage
            # P0-BUGFIX（成本被退还）：现货语义下买入数量必须夹到现金可支付范围
            # （成交额 + 费用 ≤ 现金），与回测 _execute_fill 的
            # `qty = min(qty, cash / (fill_price * (1 + fee_rate)))` 同构。
            # 原实现在扣款后用 `self._cash = max(self._cash, 0.0)` 抹平负现金，
            # 而满仓买入时现金恰好被扣成 -cost，抹平 = 把买入腿的手续费+滑点
            # 全额退还（实测恒价 1 次往返末权益 9990.00 vs 解析解 9980.01，
            # 20 次往返退还 49.5%）。⇒ 训练学的是"交易半价"世界，
            # 且成本瓶颈的幅度被低估约一半。
            if qty_delta > 0 and not self.short_enabled:
                # 现货语义的现金约束（只有旧的长仓模式适用）：买入数量夹到现金可支付
                # 范围。做空模式是合约/保证金语义（目标仓位已封在 ±1 内，保证金
                # 充足性由 equity≤0 守卫兜底），现金夹取会让"回补空头"失败并留下
                # 残仓——那才是真正的失真。
                max_affordable = self._cash / (price_now * (1.0 + unit_cost))
                qty_delta = min(qty_delta, max(0.0, max_affordable))
            if abs(qty_delta) > 1e-12:
                cost = abs(qty_delta) * price_now * unit_cost
                self._cash -= cost
                # 成本口径可观测性（2026-10 成本标定）：累计"实际成交的名义/权益"，
                # 即成本基数。被收的现金 = 该值 × unit_cost × equity（严格恒等）。
                # 为什么必须由环境自己记：外部用"相邻仓位差"求和会把**价格漂移**
                # 造成的被动仓位变化也算成换手——实测 1h k=4 用外部口径得 159.6，
                # 而真实被收费的换手只有约 1/3，breakeven 与 margin 全部失真。
                _traded = abs(qty_delta) * price_now / max(equity, 1e-9)
                self._turnover_units += _traded
                self._cost_paid += cost
                if diff > 0:  # 买入
                    self._cash -= qty_delta * price_now
                    self._qty += qty_delta
                else:         # 卖出
                    self._cash += abs(qty_delta) * price_now
                    self._qty += qty_delta
                # 2026-10 审计修复（LOW）：持仓均价按部分平/加仓重加权——
                # 曾只在首次建仓记价，部分平仓后 _pnl_ratio（浮动盈亏特征）仍按原始
                # 入场价算，与实际未实现盈亏脱节（做空模式同样失真）。
                _old_qty = self._qty - qty_delta   # 成交前持仓
                if abs(self._qty) <= 1e-12:
                    self._qty = 0.0
                    self._entry_price = None
                elif abs(_old_qty) < 1e-12 or self._entry_price is None:
                    self._entry_price = price_now
                elif abs(qty_delta) > 1e-12:
                    self._entry_price = (
                        _old_qty * self._entry_price + qty_delta * price_now
                    ) / self._qty
                # 现货账户不得透支：夹取后现金 ≥0 数学上成立，此处仅为浮点兜底
                # （只归零 |cash|<1e-9 的尾差，不会退还任何真实成本）
                # 做空模式现金可为负（保证金语义），不做该兜底。
                if not self.short_enabled and -1e-9 < self._cash < 0.0:
                    self._cash = 0.0
        # 若目标清仓则重置入场价（做空模式下 ±0 都算清仓）
        if abs(target_ratio) <= 1e-9 and abs(self._qty) <= 1e-12:
            self._entry_price = None

        # ---- 2. 按决策间隔推进 k 根K线（k=1 时与旧行为逐位一致） ----
        # 每根都累计奖励（semi-MDP：一个决策 = k 根行情的总收益总奖励），
        # 但调仓成本只在决策点收一次。k>1 时：
        #   - 探索的摩擦成本降 k 倍（旧行为下随机试错每回合亏 46~82%）；
        #   - 一个决策覆盖 k 根行情 ⇒ 同样墙钟内见过 k 倍行情（样本效率）；
        #   - 有效决策频率贴近人类交易节奏，而不是每 5 分钟翻一次仓。
        reward = 0.0
        # 决策期起点权益：info.ret_pct 必须是【整个决策期】的收益（k 根K线合计），
        # 而不是最后一根K线的收益——run_episode/collect_episode 用
        # total_ret *= (1+ret_pct/100) 连乘求总收益，若只上报末根收益会
        # 严重低报（实测 k=4 时把它当总收益会得到 2.79% 而真实是 8.60%），
        # 进而污染 best_train_ret / val 选模 / OOS 判据。
        _entry_equity = self._prev_equity
        for _ in range(self.decision_interval):
            p_cur = self._closes[self._t]
            price_next = self._closes[self._t + 1] if self._t + 1 < self.n else p_cur
            self._t += 1
            if self._t >= self.n - 1:
                self._t = self.n - 2
                self._done = True
            # P2-12：资金费率（永续合约持仓成本，可选）——按穿过的下一根K线持仓价值收取。
            # 与回测 P0-3 funding_rate 同口径（每根K线按持仓价值×费率），0=关闭。
            # 2026-10 做空支持：改用**带符号**持仓（qty 而非 abs(qty)）——正费率下
            # 多头付费、空头收钱，这是永续合约的定义。对纯多头（qty>0）逐位等价。
            if self.funding_rate != 0.0 and abs(self._qty) > 1e-12:
                self._cash -= self._qty * price_next * self.funding_rate
            equity_next = self._equity(price_next)
            ret = (equity_next / max(self._prev_equity, 1e-9)) - 1.0
            # 标的本身这根K线的收益（未乘仓位）——excess 模式算"相对基准的超额"要用
            asset_ret = (price_next / p_cur - 1.0) if p_cur > 0 else 0.0
            reward += self._bar_reward(ret, asset_ret, equity_next)
            self._prev_equity = equity_next
            # P4-E1：记录最终权益/仓位快照（step 用 price_next=最后一根K线收盘价，
            # 而 _t 在 done 分支被钳回 n-2——若 run_episode 事后读 _closes[_t] 会
            # 漏掉最后一根K线的 PnL）。由 run_episode/collect_episode 直接取快照。
            self._final_equity = equity_next
            self._final_position_ratio = (self._qty * price_next) / max(equity_next, 1e-9)
            # 爆仓守卫（仅做空模式）：空头在上涨中亏损无上限，权益击穿 0 即强平终止。
            # 多仓模式权益理论上界为 0（价格跌到 0），不加该守卫以保持逐位兼容。
            if self.short_enabled and equity_next <= 0.0:
                self._bankrupt = True
                self._done = True
            if self._done:
                break

        info = {
            "equity": equity_next,
            "ret_pct": (equity_next / max(_entry_equity, 1e-9) - 1.0) * 100.0,
            "position_ratio": self._pos_ratio(), "vol": self._recent_vol_5(),
            "price": price_next, "action": self.action_names[action],
            "bankrupt": bool(getattr(self, "_bankrupt", False)),
        }
        return self._build_state(), reward, self._done, info

    def _bar_reward(self, ret: float, asset_ret: float, equity_next: float) -> float:
        """单根K线的奖励（F1 口径分离）。

        absolute（旧行为，逐位一致）：reward = tanh 软裁剪的组合收益(bp) - vol_penalty·(仓位·vol%)²
        excess（修复口径）：        reward = tanh 软裁剪的超额收益(bp) - risk_coef·(偏离基准·vol%)²

        为什么必须区分（实测证据见诊断报告 §1）：
        - absolute 的风险项与收益项**单位不一致**（bp vs 百分比平方，差 10⁴），
          且是对持仓的**无条件**扣分 ⇒ "永不交易"成为全局最优；
        - excess 把基准（默认买入持有）作为零点：空仓在上涨行情里要付机会成本，
          惩罚只作用于"偏离基准"的部分，量级也以 bp 标定（risk_coef），
          于是最优解从"别动"变成"择时"。
        """
        vol = self._recent_vol_5()
        vol_ref = 1.0  # 正常波动率基准（%）
        pos = self._pos_ratio()
        # P0-3 优化：奖励软裁剪（原为 clip ±150 硬截断）。tanh 压缩保留极端行情的
        # 方向梯度（硬截断在边界饱和，智能体学不到"越极端越该调整"），
        # 且 |tanh|<=1 天然有界（无需 clip，数值更稳）。
        # 标定：ret*10000=±150bp 时 tanh(±1)=±0.7619，即软裁剪略缩中间值，
        # 但尾部仍保留 0.76~1.0 的梯度信号（原实现尾部直接饱和为 1）。
        if self.reward_mode == "excess":
            excess_bp = (ret - self.benchmark_position * asset_ret) * 10000.0
            reward_bp = float(150.0 * np.tanh(excess_bp / 150.0))
            active = pos - self.benchmark_position
            risk_penalty = self.risk_coef * ((active * vol / vol_ref) ** 2)
        else:
            reward_bp = float(150.0 * np.tanh(ret * 10000.0 / 150.0))
            risk_penalty = self.vol_penalty * ((pos * vol / vol_ref) ** 2)
        reward = reward_bp - risk_penalty

        # ---- 可选奖励塑形（系数=0 即关闭；回撤惩罚对齐真实目标、连亏抑制
        #      "抄底死扛"、趋势一致性强化"趋势加仓"方向） ----
        if self.reward_dd_penalty > 0 or self.reward_losing_penalty > 0 or self.reward_trend_align > 0 or self.tail_risk_penalty > 0:
            self._peak_equity = max(self._peak_equity, equity_next)
            dd_t = (self._peak_equity - equity_next) / max(self._peak_equity, 1e-9)
            if self.reward_dd_penalty > 0:
                # 回撤加深才罚（增量），噪声小；dd>12% 加重罚（接近风险管理条）
                if dd_t > self._last_dd:
                    reward -= self.reward_dd_penalty * (dd_t - self._last_dd) * 100.0
                if dd_t > 0.12:
                    reward -= 3.0 * self.reward_dd_penalty
            self._last_dd = dd_t
            if self.reward_losing_penalty > 0:
                self._losing_streak = self._losing_streak + 1 if ret < 0 else 0
                if self._losing_streak > 2:
                    reward -= self.reward_losing_penalty * ((self._losing_streak - 2) ** 2)
            if self.reward_trend_align > 0 and self._t >= 20 and vol > 0:
                # 低波动趋势中持仓与方向一致才加分；高波动自动失效，不与风险项冲突
                ret20 = (self._closes[self._t] / self._closes[self._t - 20]) - 1.0
                vol_factor = max(0.0, 1.0 - min(1.0, vol / 3.0))
                reward += self.reward_trend_align * pos * (1.0 if ret20 > 0 else -1.0) * vol_factor
            if self.tail_risk_penalty > 0:
                # 尾部风险惩罚：单根超大幅回撤（<-2%）额外加重罚（CVaR 思想），
                # 训练时抑制"高波动满仓赌单根大K线"的极端行为。
                if ret < -0.02:
                    reward -= self.tail_risk_penalty * (abs(ret) * 100.0) * pos
        return float(reward)


# 波动率特征在特征矩阵中的索引（由 _precompute_features 固定顺序）
_VOL_IDX = 4


def _precompute_features(df: pd.DataFrame, backend: str = "numpy") -> np.ndarray:
    """向量化预计算状态特征矩阵 (n, 特征数)。

    backend: "numpy" 或 "cupy"（GPU 加速特征预计算，需 cupy 可用；不可用时回退 numpy）。
    特征预计算是大K线数据下的主成本，GPU 化可显著加速；训练循环本身
    （逐K线模拟）有状态依赖无法向量化，仍走 CPU。

    特征顺序（供 _VOL_IDX 引用）：
    0  close 对数收益率(1期)
    1  ret_3（3期收益）
    2  ret_5
    3  ret_10
    4  5期已实现波动率(%)   ← _VOL_IDX（与奖励惩罚一致，智能体可直接观测）
    5  RSI(14) 归一化 0-1
    6  MACD柱 归一化
    7  量比（当前/5期均量）
    8  MA10偏离
    9  MA30偏离
    10 布林带位置 0-1
    11 趋势regime（长周期动量方向）
    12 风险regime（波动率分位）
    """
    xp = _get_xp(backend)
    c = df["close"]
    # 2026-10 审计修复（MED）：volume 缺列兜底与 TradingEnv.__init__ 的守卫同源
    #（曾 KeyError 崩掉整个环境构造——init 对缺列做了 np.full 兜底，特征预计算没有）。
    v = df["volume"] if "volume" in df.columns else pd.Series(1e12, index=df.index)
    ret = c.pct_change() * 100.0
    if xp is not np:
        # GPU 后端：核心窗口特征在 GPU 上向量化计算（cumsum 技巧），
        # 完成后转回 numpy 供训练循环无缝使用（env.step 逐K线有状态依赖，无法 GPU）。
        return _precompute_features_cupy(c.to_numpy(float), v.to_numpy(float), xp)
    # P4-B4：ret/vol 列用 tanh 压缩到 (-1,1) 有界——此前 ret_N 是原始百分比
    # （异常行情可达 ±20+）、vol_5 是原始 0~15+，与已归一化的 RSI/MACD/regime 混拼
    # 入网，大数值主导 MLP 输入、ReLU 激活分布不稳。tanh 单调且有界，方向信息
    # 保留、极端值收敛；训练/部署共用本函数，口径天然一致。
    _TANH_RET = 5.0    # ret ±5% → ±0.76（常见 ±2% → ±0.38）
    _TANH_VOL = 3.0    # vol 3% → 0.76（高波动 10%+ 收敛到 ~1）
    feat = pd.DataFrame({
        "ret_1": np.tanh(ret / _TANH_RET),
        "ret_3": np.tanh(c.pct_change(3) * 100.0 / _TANH_RET),
        "ret_5": np.tanh(c.pct_change(5) * 100.0 / _TANH_RET),
        "ret_10": np.tanh(c.pct_change(10) * 100.0 / _TANH_RET),
        "vol_5": np.tanh(ret.rolling(5).std() / _TANH_VOL),
        "rsi": (_rsi(c, 14) / 100.0).clip(0, 1),
        "macd": (_macd_hist(c) / c * 100.0).clip(-10, 10) / 10.0,
        "vol_ratio": (v / (v.rolling(5).mean() + 1e-12)).clip(0, 5) / 5.0,
        "ma_dist10": (c / c.rolling(10).mean() - 1.0) * 100.0 / 10.0,
        "ma_dist30": (c / c.rolling(30).mean() - 1.0) * 100.0 / 10.0,
        "bb_pos": _bb_pos(c),
        "trend_regime": np.sign(c.pct_change(60)),
        # P0-2（前视修复）：risk_regime 改为滚动分位——rolling(40).rank(pct=True)
        # 会用到整列全序列排名（含未来值），部署端 ta.percentrank 只能看到历史，
        # 训练/部署口径不一致。改为逐窗口分位：每根K线用最近 40 个样本算当前值的百分位。
        "risk_regime": _rolling_percentile(ret.rolling(20).std(), 40),
    }).fillna(0.0)
    return feat.to_numpy(float)


def _precompute_features_cupy(closes: np.ndarray, volumes: np.ndarray, xp) -> np.ndarray:
    """GPU 版特征预计算：核心窗口特征用 cupy 向量化（cumsum），返回 numpy。"""
    c = xp.asarray(closes, dtype=xp.float64)
    v = xp.asarray(volumes, dtype=xp.float64)
    n = len(c)

    def sma(a, period):
        out = xp.full(n, xp.nan)
        if n < period:
            return out
        # 零前缀 cumsum（与 indicators.vectorized 对齐），避免下标差 1
        prefix = xp.zeros(1, dtype=a.dtype)
        cs = xp.cumsum(xp.concatenate((prefix, a)))
        out[period - 1:] = (cs[period:] - cs[:-period]) / period
        return out

    # M2-T3：cupyx.scipy.signal.lfilter 用于向量化 EMA（见 ema()）；
    # 该子模块并非所有 cupy 安装都包含，缺失时回退 for 循环版本
    try:
        from cupyx.scipy.signal import lfilter as _lfilter
    except ImportError:
        _lfilter = None

    def ema(a, period):
        """cupy 向量化 EMA：优先用 lfilter（GPU 原生扫描），不可用时回退 Python for 循环。

        lfilter 通过 y[n] = k*x[n] + (1-k)*y[n-1] 一步算出全量序列，
        初始条件 zi=a[0] 使 y[0]=a[0]（与原始 for 循环 y[0]=a[0] 逐位一致）。
        """
        k = 2.0 / (period + 1)
        if _lfilter is not None:
            b = xp.array([k], dtype=xp.float64)
            a_coeff = xp.array([1.0, -(1.0 - k)], dtype=xp.float64)
            zi = xp.array([a[0]], dtype=xp.float64)
            out, _ = _lfilter(b, a_coeff, a, zi=zi)
            return out
        # 回退：Python for 循环（与 RSI 循环的处理方式一致，非热点路径）
        out = xp.empty_like(a)
        out[0] = a[0]
        km1 = 1.0 - k
        for i in range(1, n):
            out[i] = a[i] * k + out[i - 1] * km1
        return out

    def pct_chg(a, period):
        out = xp.full(n, xp.nan)
        out[period:] = a[period:] / a[:-period] - 1.0
        return out * 100.0

    def fill_nan0(a):
        """首元素 NaN 填 0（仅影响 warmup 区，训练从 warmup 后开始）。"""
        out = xp.where(xp.isnan(a), 0.0, a)
        return out

    def roll_std(a, period):
        out = xp.full(n, xp.nan)
        if n < period:
            return out
        prefix = xp.zeros(1, dtype=a.dtype)
        cs = xp.cumsum(xp.concatenate((prefix, a)))
        cs2 = xp.cumsum(xp.concatenate((prefix, a * a)))
        sums = cs[period:] - cs[:-period]
        sqs = cs2[period:] - cs2[:-period]
        # 样本标准差（ddof=1），与 numpy rolling().std() 一致
        var = (sqs - sums * sums / period) / (period - 1)
        out[period - 1:] = xp.sqrt(xp.maximum(var, 0))
        return out

    ret = pct_chg(c, 1)
    # 首元素 NaN 填 0（避免 cumsum 传播），warmup 区不影响训练
    ret0 = fill_nan0(ret)
    vol5 = roll_std(ret0, 5)
    # 风险 regime：与 numpy 版完全一致用原始 ret（保留首元素 NaN）做
    # rolling(20).std().rolling(40).rank(pct=True)。rank 难 GPU 化，此列
    # 量小走 numpy，保证训练/部署特征严格一致。
    import pandas as pd
    _ret_series = pd.Series(xp.asnumpy(ret) if hasattr(xp, "asnumpy") else ret)
    _risk = _ret_series.rolling(20).std().rolling(40).rank(pct=True).to_numpy()
    risk_regime = xp.asarray(_risk, dtype=xp.float64)

    ma10 = sma(c, 10)
    ma30 = sma(c, 30)
    mid20 = sma(c, 20)
    sd20 = roll_std(c, 20)
    # RSI(14) ewm：与 pandas ewm(alpha, adjust=False).mean() 语义一致——
    # 从第一个有效值开始累积，NaN 跳过。前 14 根 warmup 期近似（不影响训练）。
    # M2-T3：此循环本质也是线性递推（ag[i]=ag[i-1]*(1-alpha)+gain[i]*alpha），
    # 理论上可用 lfilter 向量化，但初始条件 ag[1]=gain[1] 而非 ag[0]=gain[0]，
    # 需特殊处理 diff 数组的偏移（gain[0]=NaN）。保持现状：_precompute_features_cupy
    # 仅训练开始时调用一次，非热点路径；numpy 版已用 pd.ewm 向量化最优。
    diff_arr = xp.diff(c)
    delta = xp.empty(n, dtype=xp.float64)
    delta[0] = xp.nan
    delta[1:] = diff_arr
    gain = xp.clip(delta, 0, None)
    loss = xp.clip(-delta, 0, None)
    ag = xp.full(n, xp.nan)
    al = xp.full(n, xp.nan)
    alpha = 1.0 / 14
    # 第一个有效 delta 索引 = 1
    ag[1] = gain[1]
    al[1] = loss[1]
    for i in range(2, n):
        ag[i] = ag[i - 1] * (1 - alpha) + gain[i] * alpha
        al[i] = al[i - 1] * (1 - alpha) + loss[i] * alpha
    rs = ag / (al + 1e-12)
    rsi = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # MACD
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(xp.nan_to_num(dif, nan=0.0), 9)
    macd = xp.clip((dif - dea) * 2.0 / c * 100.0, -10, 10) / 10.0

    vol_ratio = xp.clip(v / (sma(v, 5) + 1e-12), 0, 5) / 5.0
    # 与 numpy 版一致：(偏离率 * 100) / 10
    ma_dist10 = (c / ma10 - 1.0) * 100.0 / 10.0
    ma_dist30 = (c / ma30 - 1.0) * 100.0 / 10.0
    bb_pos = xp.clip((c - (mid20 - 2 * sd20)) / (4 * sd20 + 1e-12), 0, 1)
    trend_regime = xp.sign(pct_chg(c, 60))

    # P4-B4：与 numpy 版一致的 tanh 压缩（ret/vol 列），保证 GPU/CPU 训练与
    # 部署特征严格同口径
    _TANH_RET = 5.0
    _TANH_VOL = 3.0
    feat = xp.column_stack([
        xp.tanh(ret / _TANH_RET),
        xp.tanh(pct_chg(c, 3) / _TANH_RET),
        xp.tanh(pct_chg(c, 5) / _TANH_RET),
        xp.tanh(pct_chg(c, 10) / _TANH_RET),
        xp.tanh(vol5 / _TANH_VOL),
        rsi, macd, vol_ratio, ma_dist10, ma_dist30, bb_pos,
        trend_regime, risk_regime,
    ])
    feat = xp.nan_to_num(feat, nan=0.0)
    return xp.asnumpy(feat) if hasattr(xp, "asnumpy") else np.asarray(feat.get())


def _get_xp(backend: str):
    """返回数组库：numpy 或 cupy（不可用回退 numpy）。与 indicators.vectorized 对齐。"""
    if backend == "cupy":
        try:
            import cupy  # type: ignore
            return cupy
        except ImportError:
            pass
    return np


_GPU_STATE: Optional[bool] = None


def gpu_available() -> bool:
    """是否真正可用 GPU 后端（import 成功 + CUDA 设备存在 + 实际能跑一次运算）。与回测引擎对齐。

    Bug 修复：曾只测 import cupy + 一次 cumsum 运算——CPU-only 环境装了 cupy 时
    cumsum 也能跑（慢+告警），误报 True。用户勾选 GPU 后训练实际走 CPU 且每次
    运算都尝试加载 CUDA（慢/告警），表现为"GPU 训练坏掉"。现在显式检查
    cupy.cuda.runtime 是否有可用设备（无 CUDA 时抛错→回退 numpy）。
    """
    global _GPU_STATE
    if _GPU_STATE is not None:
        return _GPU_STATE
    try:
        import cupy  # type: ignore
        # 关键：确认真实 CUDA 设备存在（CPU-only cupy 安装 import 成功但无设备）
        dev = cupy.cuda.runtime.getDeviceCount()
        if dev <= 0:
            _GPU_STATE = False
            return False
        a = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        _ = cupy.asnumpy(cupy.cumsum(a))  # 触发编译，缺头文件会抛错
        _GPU_STATE = True
        return True
    except Exception:  # noqa: BLE001
        _GPU_STATE = False
        return False


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    gain = d.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-d.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=signal, adjust=False).mean()
    return (dif - dea) * 2.0


def _bb_pos(c: pd.Series, n: int = 20) -> pd.Series:
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std()
    return ((c - (mid - 2 * sd)) / (4 * sd + 1e-12)).clip(0, 1)


def _rolling_percentile(s: pd.Series, window: int = 40) -> pd.Series:
    """滚动百分位：每根K线用最近 window 个样本计算当前值的百分位（0~1），
    与 Pine ta.percentrank 同口径（无前视）。

    P0-1 优化：从 rolling.apply(逐窗口 Python 回调) 改为 rolling.rank(pct=True)
    （pandas C 实现）。rolling.rank(pct=True) 输出窗口内每个元素的百分位，
    最后一行即当前值的百分位，与旧实现 (x <= x[-1]).mean() 逐位一致。
    """
    r = s.rolling(window).rank(pct=True)
    # 预热区（窗口不足）+ NaN 归零（滚动 rank 在窗口前为 NaN）
    return r.fillna(0.0)


# ============ 完整 episode 采样（供训练） ============

def run_episode(env: TradingEnv, act_fn, max_steps: Optional[int] = None,
                return_trajectory: bool = True) -> dict:
    """跑完一个 episode。act_fn(state)->action。返回轨迹或汇总。

    返回: {total_ret, final_equity, steps, (可选)states, actions, rewards, infos}
    """
    state = env.reset()
    states, actions, rewards, infos = [], [], [], []
    total_ret = 1.0
    max_steps = max_steps or (env.n - env.warmup - 1)
    for _ in range(max_steps):
        action = act_fn(state)
        nxt, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        total_ret *= (1.0 + info["ret_pct"] / 100.0)
        state = nxt
        if done:
            break
    out = {
        "total_ret": total_ret - 1.0,
        # P4-E1：用 env 保存的最终快照（含最后一根K线收益），不再读 _closes[_t]
        #（done 分支把 _t 钳回 n-2，直接读会漏最后一段 PnL，两指标互相矛盾）
        "final_equity": env._final_equity,
        "steps": len(rewards),
        "final_position_ratio": env._final_position_ratio,
        # 成本标定（2026-10）：被收费的换手基数与现金成本（严格：cost ≈ turnover×unit_cost×equity）
        "turnover": float(getattr(env, "_turnover_units", 0.0)),
        "cost_paid": float(getattr(env, "_cost_paid", 0.0)),
        # 做空模式：权益击穿 0（空头被逼空强平）——评估时必须让它可见，
        # 否则"平均收益很好看但中途爆过仓"的模型会被当成可用
        "bankrupt": bool(getattr(env, "_bankrupt", False)),
    }
    if return_trajectory:
        out.update({
            "states": np.asarray(states, dtype=float),
            "actions": np.asarray(actions, dtype=int),
            "rewards": np.asarray(rewards, dtype=float),
            "infos": infos,
        })
    return out
