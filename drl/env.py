"""MDP 交易环境：将单标的交易建模为马尔可夫决策过程。

- **状态 s_t**：归一化的价量 / 波动率 / 趋势与风险 regime / 当前持仓与账户状态
- **动作 a_t**：离散目标仓位档位（0 / 0.25 / 0.5 / 0.75 / 1.0），表示目标持仓比例
  （买入→目标位上调、卖出→目标位下调、持仓→保持）
- **奖励 r_t**：组合收益率 - 交易成本 - 波动率风险惩罚
  （该惩罚正是"波动率上升时降低仓位"的行为来源：高波动环境下满仓会得到负奖励）

环境完全向量化预计算特征序列，训练时按索引 O(1) 取状态，可高速反复试错。
"""
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

# 离散动作 = 目标仓位档位
ACTION_BUCKETS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
ACTION_NAMES = ["清仓", "轻仓", "半仓", "重仓", "满仓"]


class TradingEnv:
    """基于历史K线的交易环境（可重置、可 step）。"""

    def __init__(self, df: pd.DataFrame,
                 start_cash: float = 10000.0,
                 fee_rate: float = 0.001,
                 vol_penalty: float = 0.5,
                 cost_scale: float = 1.0,
                 min_trade_zone: float = 0.05,
                 warmup: int = 60,
                 backend: str = "numpy",
                 extra_factors: Optional[np.ndarray] = None,
                 state_window: int = 1,
                 # 可选奖励塑形（系数=0 即关闭，默认关；A/B 验证后开启）
                 reward_dd_penalty: float = 0.0,      # 回撤加深惩罚（含 dd>12% 硬约束）
                 reward_losing_penalty: float = 0.0,  # 连亏惩罚（3 连亏起二次斜坡）
                 reward_trend_align: float = 0.0) -> None:  # 趋势一致性奖励（低波动趋势持仓同向加分）
        self.df = df
        self.start_cash = start_cash
        self.fee_rate = fee_rate
        self.vol_penalty = vol_penalty      # 波动率惩罚系数
        self.cost_scale = cost_scale        # 交易成本缩放
        self.min_trade_zone = min_trade_zone  # 调仓死区：与目标仓位差异小于此比例则不调仓
        self.reward_dd_penalty = reward_dd_penalty
        self.reward_losing_penalty = reward_losing_penalty
        self.reward_trend_align = reward_trend_align
        self.warmup = warmup
        self.backend = backend
        # 状态窗口：堆叠最近 N 根K线的特征，给 MLP 时序记忆（单点特征看不到形态变化）。
        # 默认 1=向后兼容旧模型；>1 时 state_dim 放大 N 倍
        self.state_window = max(1, int(state_window))
        closes = df["close"].to_numpy(float)
        self.n = len(closes)
        self._closes = closes
        self._features = _precompute_features(df, backend=backend)  # (n, feat_dim)
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

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def reset(self) -> np.ndarray:
        """回到起点，返回初始状态。"""
        self._t = self.warmup
        self._cash = self.start_cash
        self._qty = 0.0
        self._entry_price = None
        self._prev_equity = self.start_cash
        self._done = False
        return self._build_state()

    def _state_feats(self) -> np.ndarray:
        """当前时点的特征向量：state_window>1 时堆叠最近 N 根（早期零填充）。"""
        if self.state_window <= 1:
            return self._features[self._t]
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
        if not self._entry_price or self._entry_price <= 0:
            return 0.0
        return (self._closes[self._t] - self._entry_price) / self._entry_price

    def _equity(self, price: float) -> float:
        return self._cash + self._qty * price

    def _recent_vol_5(self) -> float:
        """最近 5 根K线的已实现波动率（%），快速反映波动率 regime。"""
        lo = max(0, self._t - 4)
        seg = self._closes[lo:self._t + 1]
        if len(seg) < 3:
            return 0.0
        rets = np.diff(seg) / seg[:-1]
        return float(np.std(rets) * 100.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行动作，推进一根K线。

        action: ACTION_BUCKETS 的索引（目标仓位档位）。
        返回 (next_state, reward, done, info)。
        """
        assert not self._done, "环境已终止，请先 reset"
        assert 0 <= action < len(ACTION_BUCKETS), f"非法动作 {action}"

        price_now = self._closes[self._t]
        price_next = self._closes[self._t + 1] if self._t + 1 < self.n else price_now

        # ---- 1. 按目标仓位在【当前收盘】调仓（下一根K线才产生收益，无前视） ----
        target_ratio = ACTION_BUCKETS[action]
        equity = self._equity(price_now)
        target_value = equity * target_ratio
        cur_value = self._qty * price_now
        cur_ratio = cur_value / max(equity, 1e-9)
        diff = target_ratio - cur_ratio
        # 调仓死区：与目标仓位差异过小则不交易，减少摩擦损耗
        if abs(diff) > self.min_trade_zone and equity > 0:
            qty_delta = (diff * equity) / price_now
            cost = abs(diff * equity) * self.fee_rate * self.cost_scale
            self._cash -= cost
            if diff > 0:  # 买入
                self._cash -= abs(diff * equity)
                self._qty += qty_delta
            else:         # 卖出
                self._cash += abs(diff * equity)
                self._qty += qty_delta
            if self._qty <= 1e-12:
                self._qty = 0.0
                self._entry_price = None
            elif self._entry_price is None:
                self._entry_price = price_now
            self._cash = max(self._cash, 0.0)
        # 若目标清仓则重置入场价
        if target_ratio <= 1e-9 and self._qty <= 1e-12:
            self._entry_price = None

        # ---- 2. 按下一根收盘价计算收益（穿越到 t+1） ----
        self._t += 1
        if self._t >= self.n - 1:
            self._t = self.n - 2
            self._done = True
        equity_next = self._equity(price_next)
        ret = (equity_next / max(self._prev_equity, 1e-9)) - 1.0

        # ---- 3. 奖励 = 收益 - 波动率风险惩罚（高波动满仓受罚） ----
        # 风险调整奖励设计（关键：信号必须清晰可学）：
        # - 收益用 bp（基点）但裁剪到 ±150bp，压低高波动段的噪声方差（否则 ±300bp
        #   的噪声会淹没惩罚信号，REINFORCE 学不动）
        # - 惩罚 = vol_penalty * (仓位 * 波动率/基准)²，随波动率平方放大，且用短周期
        #   波动率（5根）快速响应 regime 切换
        #   低波动趋势(vol≈0.8%)：+40 - 惩罚小 → 净正 → 学"趋势明朗加仓"
        #   高波动(vol≈3%)：裁剪后均值≈0 - 惩罚大 → 净负 → 学"波动率上升降仓"
        vol = self._recent_vol_5()
        vol_ref = 1.0  # 正常波动率基准（%）
        pos = self._pos_ratio()
        risk_penalty = self.vol_penalty * ((pos * vol / vol_ref) ** 2)
        reward_bp = float(np.clip(ret * 10000.0, -150.0, 150.0))
        reward = reward_bp - risk_penalty

        # ---- 4. 可选奖励塑形（系数=0 即关闭；回撤惩罚对齐真实目标、连亏抑制
        #      "抄底死扛"、趋势一致性强化"趋势加仓"方向） ----
        if self.reward_dd_penalty > 0 or self.reward_losing_penalty > 0 or self.reward_trend_align > 0:
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
                # 低波动趋势中持仓与方向一致才加分；高波动自动失效，不与 vol_penalty 冲突
                ret20 = (self._closes[self._t] / self._closes[self._t - 20]) - 1.0
                vol_factor = max(0.0, 1.0 - min(1.0, vol / 3.0))
                reward += self.reward_trend_align * pos * (1.0 if ret20 > 0 else -1.0) * vol_factor

        info = {
            "equity": equity_next, "ret_pct": ret * 100.0,
            "position_ratio": self._pos_ratio(), "vol": vol,
            "price": price_next, "action": ACTION_NAMES[action],
        }
        self._prev_equity = equity_next
        return self._build_state(), reward, self._done, info


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
    v = df["volume"]
    ret = c.pct_change() * 100.0
    if xp is not np:
        # GPU 后端：核心窗口特征在 GPU 上向量化计算（cumsum 技巧），
        # 完成后转回 numpy 供训练循环无缝使用（env.step 逐K线有状态依赖，无法 GPU）。
        return _precompute_features_cupy(c.to_numpy(float), v.to_numpy(float), xp)
    feat = pd.DataFrame({
        "ret_1": ret,
        "ret_3": c.pct_change(3) * 100.0,
        "ret_5": c.pct_change(5) * 100.0,
        "ret_10": c.pct_change(10) * 100.0,
        "vol_5": ret.rolling(5).std(),
        "rsi": (_rsi(c, 14) / 100.0).clip(0, 1),
        "macd": (_macd_hist(c) / c * 100.0).clip(-10, 10) / 10.0,
        "vol_ratio": (v / (v.rolling(5).mean() + 1e-12)).clip(0, 5) / 5.0,
        "ma_dist10": (c / c.rolling(10).mean() - 1.0) * 100.0 / 10.0,
        "ma_dist30": (c / c.rolling(30).mean() - 1.0) * 100.0 / 10.0,
        "bb_pos": _bb_pos(c),
        "trend_regime": np.sign(c.pct_change(60)),
        "risk_regime": ret.rolling(20).std().rolling(40).rank(pct=True),
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

    def ema(a, period):
        out = xp.empty_like(a)
        k = 2.0 / (period + 1)
        out[0] = a[0]
        for i in range(1, n):
            out[i] = a[i] * k + out[i - 1] * (1 - k)
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
    vol20 = roll_std(ret0, 20)
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

    feat = xp.column_stack([
        ret, pct_chg(c, 3), pct_chg(c, 5), pct_chg(c, 10), vol5,
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
    """是否真正可用 GPU 后端（import 成功 + 实际能跑一次运算）。与回测引擎对齐。"""
    global _GPU_STATE
    if _GPU_STATE is not None:
        return _GPU_STATE
    try:
        import cupy  # type: ignore
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
        "final_equity": env._equity(env._closes[env._t]),
        "steps": len(rewards),
        "final_position_ratio": env._pos_ratio(),
    }
    if return_trajectory:
        out.update({
            "states": np.asarray(states, dtype=float),
            "actions": np.asarray(actions, dtype=int),
            "rewards": np.asarray(rewards, dtype=float),
            "infos": infos,
        })
    return out
