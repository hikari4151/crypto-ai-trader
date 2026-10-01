"""深度强化学习智能体：Actor-Critic 策略梯度（REINFORCE with baseline）。

- Actor 网络：state → 各动作概率（softmax），负责学习"何时买入/卖出/调整仓位"
- Critic 网络：state → 价值估计，作为策略梯度的基线降低方差
- 训练：在回测环境反复跑 episode，用蒙特卡洛回报计算优势 (G - V)，策略梯度更新
- 探索：epsilon-greedy（随训练衰减）保证"反复试错"
- 无 torch 依赖，纯 numpy（MLP + Adam）

学习目标示例（奖励塑造已内置在 env）：
- 波动率上升且满仓 → 负奖励 → 学会降仓位
- 趋势明朗（trend_regime/ma 特征走强）→ 保持/上调仓位
"""
import copy
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

from .env import ACTION_BUCKETS, ACTION_NAMES, TradingEnv, run_episode, _precompute_features
from .nnet import MLP, _log_softmax


def _resolve_collect_workers(n_episodes: int, cfg_workers: int = 0) -> int:
    """解析并行收集线程数（P0 并行采样上量）。

    背景（2026-10）：并行收集的 worker 上限此前硬编码 min(n_episodes, 4)，
    是经验规模的吞吐瓶颈——n_episodes 提到 16/32 时仍只有 4 条同时在跑。
    本函数按配置 + CPU 核数解析实际线程数：
    - cfg_workers > 0：显式指定（夹到 [1, n_episodes]）
    - cfg_workers <= 0：自动——min(n_episodes, max(4, cpu_count))，
      至少 4 个（短数据上也有并行收益），不超 n_episodes（不造空线程），
      不超物理核数（超线程并行收益递减且抢内存带宽）。
    """
    n_episodes = max(1, int(n_episodes))
    if cfg_workers is not None and int(cfg_workers) > 0:
        return max(1, min(n_episodes, int(cfg_workers)))
    n_cpu = os.cpu_count() or 4
    return max(1, min(n_episodes, max(4, n_cpu)))

log = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区：存储历史轨迹，支持按优先级采样，与 PPO 兼容。

    传统 PPO 只用当前批量轨迹做多轮 mini-batch 更新，丢弃后不用。
    ReplayBuffer 保留最近 N 条轨迹，训练时混合采样——旧轨迹通过
    重要性采样比率自然衰减（ratio = exp(logp_new - logp_old)，
    策略偏移越远 ratio 越小，clip 越容易触发，影响自然降低）。

    用法：
        buf = ReplayBuffer(capacity=200)  # 保留最近 200 条轨迹
        for ep in range(episodes):
            trajs = collect_episodes(...)
            # 存入缓冲区
            for t in trajs:
                buf.push(t["states"], t["actions"], t["rewards"],
                         t["old_log_probs"])
            # 训练：当前批量 + 缓冲区混合采样
            batch = buf.sample(batch_size, mix_ratio=0.3)
            agent.train_batch(...)
    """

    def __init__(self, capacity: int = 200):
        self.capacity = max(1, int(capacity))
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.ndarray] = []
        self._old_log_probs: list[np.ndarray] = []
        self._segment_lengths: list[int] = []
        # P2：实例级 rng（sample 每调用新建 default_rng 无法复现且无必要）
        self._rng = np.random.default_rng()

    def push(self, states: np.ndarray, actions: np.ndarray,
             rewards: np.ndarray, old_log_probs: np.ndarray) -> None:
        """推入一条轨迹。"""
        self._states.append(states)
        self._actions.append(actions)
        self._rewards.append(rewards)
        self._old_log_probs.append(old_log_probs)
        self._segment_lengths.append(len(rewards))
        # 超过容量时移除最旧轨迹
        while len(self._states) > self.capacity:
            self._states.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._old_log_probs.pop(0)
            self._segment_lengths.pop(0)

    def sample(self, batch_size: int, mix_ratio: float = 0.3) -> dict:
        """从缓冲区采样混合批次。

        mix_ratio: 来自缓冲区的样本比例（剩余来自最新轨迹）。
        返回 {states, actions, rewards, old_log_probs, segment_lengths}。
        """
        if not self._states:
            return {"states": np.empty((0, 0)), "actions": np.empty(0, dtype=int),
                    "rewards": np.empty(0), "old_log_probs": np.empty(0),
                    "segment_lengths": []}

        n_buf = len(self._states)
        n_from_buf = min(n_buf, max(1, int(batch_size * mix_ratio)))
        n_from_new = max(1, batch_size - n_from_buf)

        # 从缓冲区均匀采样（P2：实例级 rng——每次新建 default_rng 无法复现）
        rng = self._rng
        idx = rng.choice(n_buf, size=min(n_from_buf, n_buf), replace=False)

        states = [self._states[i] for i in idx]
        actions = [self._actions[i] for i in idx]
        rewards = [self._rewards[i] for i in idx]
        log_probs = [self._old_log_probs[i] for i in idx]
        seg_lens = [self._segment_lengths[i] for i in idx]

        # 加入最新轨迹（队尾 n_from_new 条）——P2：`i not in idx` 是 O(len(idx))，
        # 逐 i 判断累计 O(n²)；转 set 后 O(1) 判定
        idx_set = set(idx.tolist())
        start = max(0, n_buf - n_from_new)
        for i in range(start, n_buf):
            if i not in idx_set:
                states.append(self._states[i])
                actions.append(self._actions[i])
                rewards.append(self._rewards[i])
                log_probs.append(self._old_log_probs[i])
                seg_lens.append(self._segment_lengths[i])

        return {
            "states": np.concatenate(states) if states else np.empty((0, 1)),
            "actions": np.concatenate(actions) if actions else np.empty(0, dtype=int),
            "rewards": np.concatenate(rewards) if rewards else np.empty(0),
            "old_log_probs": np.concatenate(log_probs) if log_probs else np.empty(0),
            "segment_lengths": seg_lens,
        }

    def __len__(self) -> int:
        return len(self._states)

    def feature_dim(self) -> Optional[int]:
        """已存轨迹的状态维度；空缓冲返回 None。

        供跨训练轮复用缓冲区时做维度守卫：配置变更（state_window / 因子列）会
        改变状态维度，此时旧轨迹与当前网络不符，必须在拼接前清空重建，否则
        会在 np.concatenate 处维度崩溃。
        """
        if not self._states:
            return None
        arr = self._states[0]
        return int(arr.shape[1]) if getattr(arr, "ndim", 0) == 2 else None

    def clear(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._old_log_probs.clear()
        self._segment_lengths.clear()


def _oos_decay(train_ref: float, oos_ret: float) -> float:
    """OOS 收益相对训练段收益的衰减（D3，独立成函数便于单测）。

    仅在"训练段为正收益"时有意义：
    - train>0, oos>=0：decay = 1 - oos/train
    - train>0, oos<0 ：decay = 1.0（训练盈利、样本外亏损 = 最大衰减）
    - train<=0       ：decay = 0.0（不可比，不判定，避免"OOS 更好反而判过拟合"的病态）
    """
    if train_ref > 0 and oos_ret >= 0:
        return 1.0 - oos_ret / train_ref
    if train_ref > 0 and oos_ret < 0:
        return 1.0
    return 0.0


def _clip_grad_norm(gW: list, gb: list, max_norm: float) -> tuple[list, list]:
    """全局梯度范数裁剪（D5）。max_norm<=0 时不裁剪（向后兼容）。

    Actor 此前无任何梯度裁剪，极端奖励下策略一步可能过冲；PPO 标配
    全局范数裁剪（SB3 clip_range_vf≈0.5-1.0 的同款思想）。
    """
    if max_norm is None or max_norm <= 0:
        return gW, gb
    total_sq = 0.0
    for g in gW:
        total_sq += float(np.sum(np.square(g)))
    for g in gb:
        total_sq += float(np.sum(np.square(g)))
    norm = float(np.sqrt(total_sq))
    if norm > max_norm and norm > 1e-12:
        scale = max_norm / norm
        return [g * scale for g in gW], [b * scale for b in gb]
    return gW, gb


class ACAgent:
    """Actor-Critic 智能体（REINFORCE with baseline）。"""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: tuple[int, ...] = (32, 32),
                 lr_actor: float = 1e-3, lr_critic: float = 3e-3,
                 gamma: float = 0.99, seed: Optional[int] = None,
                 entropy_coef: float = 0.01,
                 state_window: int = 1,
                 backbone: str = "mlp",
                 gru_hidden: int = 32) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        # 状态窗口：训练时环境的特征堆叠数（部署时 rl_adaptive 必须用同一窗口重建状态）
        self.state_window = max(1, int(state_window))
        self.backbone = backbone if backbone in ("mlp", "gru") else "mlp"
        # P1 时序网络升级：backbone="gru" 时 actor 用 GRU 序列编码（GRUActor），
        # 突破 MLP 的"只看最近 N 根各自孤立"——学到形态/惯性等跨帧模式。
        # GRU 只替换 actor 输入编码（输出 5 档 softmax 不变）；critic 保持
        # MLP（价值预测无时序强需求，且保持简单）。GRU actor 无法直接导出
        # Pine，但蒸馏路径（P2）把 GRU teacher 压成 MLP student 供 TradingView。
        if self.backbone == "gru":
            from .nnet_temporal import GRUActor
            self.actor = GRUActor(state_dim, n_actions,
                                  state_window=self.state_window,
                                  gru_hidden=int(gru_hidden),
                                  head_hidden=tuple(hidden),
                                  seed=seed, lr=lr_actor)
            self._gru_hidden = int(gru_hidden)
        else:
            dims = [state_dim, *hidden, n_actions]
            self.actor = MLP(dims, seed=seed, lr=lr_actor)
            self._gru_hidden = 0
        self.critic = MLP([state_dim, *hidden, 1], seed=(seed or 0) + 1, lr=lr_critic)
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # 熵正则：鼓励探索，防止过早收敛到单一动作
        self.epsilon = 1.0            # 初始探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999    # 乘法衰减（无 schedule 时兜底）
        self._rng = np.random.default_rng(seed)
        # P2 蒸馏导出（Pine 体积门 × 网络容量）：可选的蒸馏小网络（student）。
        # 大网络（self）用于软件内实盘；distill 装得进 Pine，供 TradingView
        # 自动交易代码导出。None = 未蒸馏（teacher 本就可导出，向后兼容）。
        self.distill: Optional["ACAgent"] = None
        self.distill_meta: Optional[dict] = None
        # 蒸馏标记：本 agent 是否为蒸馏版（student）。teacher 恒 False。
        self.is_distilled = False

    def set_epsilon_schedule(self, episodes: int) -> None:
        """线性退火：从 1.0 → epsilon_min 在前 70% 训练轮内完成。

        高探索期充分试错（学习"波动率上升满仓会亏"），后期收敛到确定性策略。
        训练循环每轮调用 set_episode_epsilon(ep) 推进。
        """
        self._eps_schedule = [max(self.epsilon_min, 1.0 - (1.0 - self.epsilon_min) * (i / max(1, int(episodes * 0.7))))
                              for i in range(episodes)]

    def set_episode_epsilon(self, episode_index: int) -> None:
        """设置本轮训练使用的探索率（0 基索引）。"""
        sched = getattr(self, "_eps_schedule", None)
        if sched and 0 <= episode_index < len(sched):
            self.epsilon = sched[episode_index]
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sample_action_with_logp(self, state: np.ndarray, rng=None,
                                mask: Optional[np.ndarray] = None) -> tuple[int, float]:
        """纯策略采样并返回 (action, log_prob)（PPO 收集旧策略概率用）。

        关键：不叠加 epsilon 噪声——否则 log_prob 与真实行为策略不一致，
        重要性采样比会失真（PPO 的探索由熵正则承担）。
        单次前向即取概率与采样，避免重复计算（曾每次算两次 predict_proba）。

        2026-08：改用 predict_log_proba（log_softmax 直接算 logp，消除下溢风险）。
        rng：可传入 per-worker 的独立随机源。多线程并行收集轨迹时**必须**传独立
        rng，否则共享 self._rng（numpy Generator 非线程安全）会造成非确定性/
        潜在竞态，破坏 seed 可复现（D1 修复）。

        mask：可选动作掩码（合法动作索引数组）。传入时把已选/非法动作概率置零
        并重归一化（动作掩码标准做法：FactorMiningEnv 用它强制每步选新因子，
        杜绝 argmax=首动作的重复选择浪费步数）。logp 按掩码后的分布计算，
        exp(logp) 与采样概率严格一致（重要性采样比率不失真）。
        """
        rng = rng or self._rng
        # P2：collect 路径前向不写 _ln_cache（多线程并行收集时共享 MLP 的
        # 缓存写入竞态是潜在地雷；backward 需要的是 train_batch 内前向的缓存）
        logp_all = self.actor.predict_log_proba(state.reshape(1, -1),
                                                cache_for_backward=False)[0]
        proba = np.exp(logp_all)  # 从 log_softmax 恢复概率（exp 无下溢，因为 logp >= -inf）
        if mask is not None:
            valid = np.zeros_like(proba, dtype=bool)
            valid[np.asarray(mask, dtype=int)] = True
            proba[~valid] = 0.0
            total = float(proba.sum())
            if total <= 0 or not np.isfinite(total):
                # 防御：掩码后无正概率（理论上不会发生）→ 合法动作均匀分布
                proba = np.zeros_like(proba)
                proba[valid] = 1.0
                proba /= float(valid.sum())
            else:
                proba /= total
            # logp 同步到掩码后的分布（exp(logp)==proba 严格成立，
            # 保证 old_log_probs 与真实行为策略一致）。近确定策略下
            # 个别动作概率可下溢到 0（log=-inf），但被采样动作概率恒>0，
            # 其 logp 不受影响；用 errstate 压掉无害告警。
            with np.errstate(divide="ignore"):
                logp_all = np.log(proba)
        # run6 E1：手写 inverse-CDF 采样替代 rng.choice(n, p)——
        # Generator.choice 对 p 恰好消耗 1 个随机数且等价于
        # searchsorted(cumsum(p), u)（u 在同一序列同位置取）；
        # 位级一致验证见 .optim/verify_sdrl_e1_choice.py（2000 步
        # action/logp 全等 + rng 尾序列一致）。采样热路径 94k 步/回合
        # 省 numpy choice 调度（~7µs/步 → ~2µs）。
        u = rng.random()
        action = int(np.searchsorted(np.cumsum(proba), u, side="left"))
        action = min(action, len(proba) - 1)  # 浮点累计误差防御
        logp = float(logp_all[action])
        return action, logp

    def collect_episode(self, env, rng=None) -> dict:
        """跑完一个 episode 并收集轨迹（含旧策略 log_prob，供 PPO 使用）。

        env 需提供 reset/step；episode 长度上限优先取 env.max_steps
        （FactorMiningEnv 等序列决策环境），否则用 n - warmup - 1（TradingEnv）。
        rng：可选，透传给采样；多线程收集时传入 per-worker 独立 rng（见 D1）。
        env.use_action_mask=True 时每步先取 env.valid_actions() 作为动作掩码
        （强制选新动作；FactorMiningEnv 防重复选择浪费步数）。
        """
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        total_ret = 1.0
        use_mask = bool(getattr(env, "use_action_mask", False))
        max_steps = getattr(env, "max_steps", None) or (env.n - env.warmup - 1)
        for _ in range(max_steps):
            mask = env.valid_actions() if use_mask else None
            action, logp = self.sample_action_with_logp(state, rng=rng, mask=mask)
            nxt, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(logp)
            # ret_pct 是 TradingEnv 专属；FactorMiningEnv 无收益信息（total_ret 仅统计用）
            total_ret *= (1.0 + info.get("ret_pct", 0.0) / 100.0)
            state = nxt
            if done:
                break
        return {
            "states": np.asarray(states, dtype=float),
            "actions": np.asarray(actions, dtype=int),
            "rewards": np.asarray(rewards, dtype=float),
            "old_log_probs": np.asarray(log_probs, dtype=float),
            "total_ret": total_ret - 1.0,
            # FactorMiningEnv 无权益概念（final_equity 仅 TradingEnv 统计用）。
            # P4-E1：TradingEnv 用 step 末尾保存的最终快照（含最后一根K线 PnL），
            # 不再读 _closes[_t]（done 后 _t 被钳回 n-2，直接读会漏最后一段收益）。
            "final_equity": getattr(env, "_final_equity",
                                    env._equity(env._closes[env._t]) if hasattr(env, "_equity") else 0.0),
            "steps": len(rewards),
        }

    def greedy_action(self, state: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> int:
        """确定性动作（评估/部署用）。

        mask：可选动作掩码（合法动作索引数组），非法动作概率压到 -inf 再取
        argmax（与采样端同口径：FactorMiningEnv 用它保证贪心决策也强制选新因子）。
        """
        proba = self.actor.predict_proba(state.reshape(1, -1))[0]
        if mask is not None:
            invalid = np.ones_like(proba, dtype=bool)
            invalid[np.asarray(mask, dtype=int)] = False
            proba[invalid] = -1e18
        return int(np.argmax(proba))

    # ---------- 训练（真 PPO：GAE + 多轮 mini-batch 裁剪 + 熵正则） ----------
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     lam: float = 0.95,
                     segment_lengths: Optional[list[int]] = None) -> tuple[np.ndarray, np.ndarray]:
        """GAE 广义优势估计：比蒙特卡洛回报方差更低、偏差可控。

        advantage_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        segment_lengths: 拼接批次中各 episode 的步数（P2-8）。多条轨迹拼接批量
        训练时，若把整批当单条轨迹算 GAE，末步 next_val 会取下一 episode 首状态、
        gae 跨轨迹串扰（末段优势被污染）——按段独立计算：段末 next_val=0、gae 归零。
        """
        T = len(rewards)
        advantages = np.zeros(T)
        if segment_lengths:
            start = 0
            for seg_len in segment_lengths:
                end = min(start + seg_len, T)
                if end > start:
                    gae = 0.0
                    for t in range(end - 1, start - 1, -1):
                        next_val = values[t + 1] if t + 1 < end else 0.0
                        delta = rewards[t] + self.gamma * next_val - values[t]
                        gae = delta + self.gamma * lam * gae
                        advantages[t] = gae
                start = end
            # 防御：段长之和不足 T 时把余下部分按一段处理（正常路径不会发生）
            if start < T:
                gae = 0.0
                for t in range(T - 1, start - 1, -1):
                    next_val = values[t + 1] if t + 1 < T else 0.0
                    delta = rewards[t] + self.gamma * next_val - values[t]
                    gae = delta + self.gamma * lam * gae
                    advantages[t] = gae
        else:
            gae = 0.0
            for t in range(T - 1, -1, -1):
                next_val = values[t + 1] if t + 1 < T else 0.0
                delta = rewards[t] + self.gamma * next_val - values[t]
                gae = delta + self.gamma * lam * gae
                advantages[t] = gae
        returns = advantages + values  # G_t 估计
        return advantages, returns

    def train_batch(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, old_log_probs: np.ndarray,
                    ppo_epochs: int = 4, mini_batch_size: int = 128,
                    clip_eps: float = 0.2, lam: float = 0.95,
                    segment_lengths: Optional[list[int]] = None,
                    advantage_scale: float = 1.0,
                    actor_grad_clip: float = 0.0) -> dict:
        """PPO 更新（借鉴 Stable Baselines3 / FinRL 的 PPO 核心）：

        - GAE 估计优势（低方差）+ 优势归一化
        - 多轮（ppo_epochs）× mini-batch 更新：旧策略 log_prob 在收集时保存，
          因此 ratio = exp(logp_new - logp_old) 在多轮更新中真正 ≠1，
          clipped surrogate 约束才实际生效（此前单次更新 ratio≡1，clip 恒不触发）
        - 熵正则鼓励探索，防止过早收敛到单一动作
        - 每批 actor 只前向一次（消除重复前向）
        """
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        old_log_probs = np.asarray(old_log_probs, dtype=float)
        n = len(states)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "mean_advantage": 0.0, "epsilon": round(self.epsilon, 4)}

        V = self.critic.predict_value(states)
        # P4-E2：价值基线尺度统一。critic 训练目标是批内标准化的 G_norm（P4-B2，
        # 目标分布稳定），但 GAE 的 δ_t = r + γV' - V 必须用与奖励同尺度的基线——
        # 否则 critic 输出 ~N(0,1) 而 r 量级可达 ±10²，V 项被稀释到可忽略，
        # 方差削减失效、策略梯度退化为带噪声的原始回报加权。
        # 用原始折扣回报（零基线 GAE）的统计量把 V 还原到原始尺度后再算 GAE，
        # 两全：critic 目标稳定 + 基线有效（SB3 同款口径）。
        _, G = self._compute_gae(rewards, np.zeros_like(rewards), lam=lam,
                                 segment_lengths=segment_lengths)
        G_std = G.std() + 1e-8
        G_mean = G.mean()
        V_raw = V * G_std + G_mean
        advantages, _ = self._compute_gae(rewards, V_raw, lam=lam,
                                          segment_lengths=segment_lengths)
        # 归一化优势降低方差
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # P4-B2：价值目标 G 批内标准化。G 是原始累计折扣回报，量级随 episode 长度
        # 与 reward 尺度剧烈波动（可达数百~数千），critic 目标分布不稳 → 价值函数
        # 难收敛、优势更噪。标准化为标准正态后 MSE 目标稳定，价值网络学得更准。
        G_norm = (G - G_mean) / G_std
        # 过拟合衰减：当训练收益显著高于验证收益时衰减优势权重，
        # 让 PPO 更新更保守（避免模型记忆噪声模式）
        if advantage_scale != 1.0:
            adv *= advantage_scale

        idx = np.arange(n)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        for _ in range(ppo_epochs):
            self._rng.shuffle(idx)
            for start in range(0, n, mini_batch_size):
                b = idx[start:start + mini_batch_size]
                mb = len(b)

                # ---- Actor：单次前向 → log_softmax → 重要性采样比 → clipped surrogate ----
                # 2026-08：用 predict_log_proba（log_softmax 直接算 logp），
                # 消除 softmax+log 数值下溢（当 proba→0 时 log 会坍缩到 -inf）
                acts = self.actor.forward(states[b])   # 保留激活值供 backward 使用
                logp_all = _log_softmax(acts[-1])
                act = actions[b]
                logp = logp_all[np.arange(mb), act]
                proba = np.exp(logp_all)  # 从 log_softmax 恢复概率（exp 无下溢）
                # 重要性采样比率 r(θ) = exp(log π_new - log π_old)
                ratio = np.exp(logp - old_log_probs[b])
                adv_b = adv[b]
                surr1 = ratio * adv_b
                surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                # 策略损失 = -min(surr1, surr2)；梯度仅在 surr1 更小时生效
                use_surr1 = surr1 <= surr2
                # softmax 交叉熵梯度 = (proba - onehot) * adv * ratio（PPO 重要性采样）
                # 注意：必须乘 ratio——clip 目标对未裁剪样本的梯度含 r(θ) 缩放因子，
                # 遗漏会让策略偏移越远时梯度方向/幅度失真（此前版本缺少 ratio）
                dlogits = proba.copy()
                dlogits[np.arange(mb), act] -= 1.0
                dout_actor = (ratio[:, None] * dlogits) * (adv_b * use_surr1)[:, None]
                # 熵正则梯度：鼓励探索防止过早收敛到单一动作
                # 优化：直接用 log_softmax 输出 logp_all 计算熵，
                # 避免额外 np.log(proba+1e-12) 的昂贵对数运算
                if self.entropy_coef > 0:
                    log_proba = logp_all  # 已是对数概率（数值稳定）
                    entropy = -np.sum(proba * log_proba, axis=-1, keepdims=True)
                    # P0-4 符号修复：网络是梯度下降（apply_grad 做 W -= lr*g），
                    # 熵 H 对 logits 的梯度为 -p*(H+log p)，故这里取 +p*(H+log p)
                    # 才等价于对熵做梯度上升（探索被鼓励）。曾用负号把方向取反，
                    # 实际在【最小化】熵 → 训练主动压制探索、策略过早坍缩。
                    d_entropy = +proba * (entropy + log_proba)
                    dout_actor = dout_actor + self.entropy_coef * d_entropy
                gW_a, gb_a = self.actor.backward(acts, dout_actor)
                # D5：Actor 全局梯度范数裁剪（防极端奖励一步过冲）
                gW_a, gb_a = _clip_grad_norm(gW_a, gb_a, actor_grad_clip)
                self.actor.apply_grad(gW_a, gb_a)

                # ---- Critic：MSE(V, G_norm) + 梯度裁剪（防止价值函数一步过冲） ----
                # P4-B2：G 已在上方批内标准化为 G_norm，MSE 目标尺度稳定
                cacts = self.critic.forward(states[b])
                G_b = G_norm[b, None] if G_norm.ndim == 1 else G_norm[b]
                G_b = G_b.reshape(-1, 1)
                dout_critic = 2.0 * (cacts[-1] - G_b) / max(1, mb)
                gW_c, gb_c = self.critic.backward(cacts, dout_critic)
                # P4-B1：Critic 梯度改用全局范数裁剪（与 Actor 同口径）。
                # 此前逐元素 clip 到 [-1,1]：reward_bp 达 ±150、G 更大，梯度长期远超 1，
                # clip 恒饱和——每个元素独立硬截断会扭曲梯度方向，批判性学习率被压碎，
                # 优势估计噪声大 → 整个 actor-critic 不稳。改为全局范数（裁剪后方向不变）。
                gW_c, gb_c = _clip_grad_norm(gW_c, gb_c, max_norm=max(0.1, actor_grad_clip))
                self.critic.apply_grad(gW_c, gb_c)

                policy_losses.append(float(np.mean(-np.minimum(surr1, surr2))))
                value_losses.append(float(np.mean((cacts[-1] - G_b) ** 2)))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "mean_advantage": float(np.mean(adv)),
            "epsilon": round(self.epsilon, 4),
        }

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        d = {
            "state_dim": self.state_dim, "n_actions": self.n_actions,
            "gamma": self.gamma, "epsilon": self.epsilon,
            "entropy_coef": self.entropy_coef,
            "state_window": self.state_window,
            "backbone": self.backbone,
            "gru_hidden": int(getattr(self, "_gru_hidden", 0)),
            "actor": self.actor.to_dict(), "critic": self.critic.to_dict(),
            # P2 蒸馏导出标记：该 agent 是否为蒸馏版（student）
            "is_distilled": bool(getattr(self, "is_distilled", False)),
        }
        # P2 蒸馏导出：student 随 teacher 一同序列化（distill 键）。
        # 旧模型无该键 → from_dict 兜底 None，向后兼容。
        if self.distill is not None:
            d["distill_agent"] = self.distill.to_dict()
            d["distill_meta"] = self.distill_meta or {}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ACAgent":
        backbone = str(data.get("backbone", "mlp"))
        gru_hidden = int(data.get("gru_hidden", 0) or 32)
        agent = cls(data["state_dim"], data["n_actions"], gamma=data.get("gamma", 0.99),
                    entropy_coef=float(data.get("entropy_coef", 0.01)),
                    state_window=int(data.get("state_window", 1)),
                    backbone=backbone, gru_hidden=gru_hidden)
        if backbone == "gru" and data.get("actor", {}).get("type") == "gru":
            from .nnet_temporal import GRUActor
            agent.actor = GRUActor.from_dict(data["actor"], lr=agent.actor.lr)
        else:
            agent.actor = MLP.from_dict(data["actor"], lr=agent.actor.lr)
        agent.critic = MLP.from_dict(data["critic"], lr=agent.critic.lr)
        agent.epsilon = float(data.get("epsilon", 0.05))
        # P2 蒸馏导出：恢复蒸馏标记与内嵌的 student（若存在）
        if data.get("is_distilled"):
            agent.is_distilled = True
        if isinstance(data.get("distill_agent"), dict):
            try:
                agent.distill = cls.from_dict(data["distill_agent"])
                agent.distill_meta = dict(data.get("distill_meta") or {})
            except Exception:  # noqa: BLE001——蒸馏权重损坏不影响主模型加载
                agent.distill = None
                agent.distill_meta = None
        return agent

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ACAgent":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ============ 训练器 ============

def train_drl(df, cfg: dict, on_progress: Optional[Callable[[dict], None]] = None,
              extra_dfs: Optional[dict] = None) -> dict:
    """完整训练循环：在回测环境反复试错，产出训练好的智能体。

    cfg:
        episodes: 训练轮数（默认 60）
        hidden: (int, int) 隐藏层维度
        lr_actor / lr_critic: 学习率
        gamma: 折扣因子
        vol_penalty: 波动率风险惩罚系数
        cost_scale: 交易成本缩放
        start_cash / fee_rate
        seed
        多标的混合训练（P0 经验广度，可选）：
          extra_mix_ratio: 每条轨迹来自额外标的的概率（默认 0.5）
          episode_window: 采样切片长度（0=主标的训练段长度）
    extra_dfs: {name: df} 额外标的数据（可选）。传入后每个 episode 以
        extra_mix_ratio 概率随机选一个额外标的、在其数据内随机时间窗切片
        训练；否则（默认）只跑主标的训练段——行为与未传时完全一致。
        额外标的只作为经验来源，不参与 val/OOS 判据（部署语义仍由主标的承担）。
    on_progress(info): 每轮回调 {episode, total_ret, final_equity, best_ret, epsilon, ...}
    返回 {agent, history, best_ret, cfg}
    """
    t0 = time.time()
    # PPO 默认超参（真 PPO：批量收集 + 多轮 mini-batch，比单轨迹更新更稳）
    episodes = int(cfg.get("episodes", 300))
    # 单轮训练时间预算（秒）：>0 时训练到预算即提前结束（连续训练控成本用，
    # 续训场景下"学一点就停"优于"从头重训"，且超时后 OOS 门照常把关防过拟合）。
    # 0 = 不限时（手动训练全量跑满 episodes 用）。
    time_budget = float(cfg.get("time_budget", 0.0) or 0.0)
    # C2：隐藏层默认 32 而非 64——15 维输入下 64 维隐层过参数化（过拟合风险高），
    # 且 Pine 导出权重规模与隐层平方相关（64×64=4096 vs 32×32=1024，缩小 4 倍）
    hidden = tuple(cfg.get("hidden", [32, 32]))
    # 每轮收集的轨迹数（批大小，A2C/PPO 标配：批量平均降方差）
    n_episodes = max(1, int(cfg.get("n_episodes", 4)))
    # P0 并行采样上量：并行收集线程数（0=自动按 CPU 核数，见 _resolve_collect_workers）
    collect_workers = int(cfg.get("collect_workers", 0) or 0)
    n_collect_workers = _resolve_collect_workers(n_episodes, collect_workers)
    # PPO 多轮更新轮数（SB3 默认 10；小数据建议 3-5）
    ppo_epochs = max(1, int(cfg.get("ppo_epochs", 4)))
    mini_batch_size = max(8, int(cfg.get("mini_batch_size", 128)))
    # 波动率惩罚需足够大以压过高波动段的收益噪声，才能学到"波动上升→降仓"
    # （收益为 bp 标度，惩罚系数默认 20：低波动趋势净正、高波动满仓净负）
    vol_penalty = float(cfg.get("vol_penalty", 20.0))
    cost_scale = float(cfg.get("cost_scale", 1.0))
    # D2：训练/验证/OOS 环境统一滑点（默认与回测/纸面 paper_slippage 同口径 0.0005）
    slippage = float(cfg.get("slippage", 0.0005))
    # D5：Actor 全局梯度范数裁剪（默认 1.0，0=禁用）
    actor_grad_clip = float(cfg.get("actor_grad_clip", 1.0))
    seed_cfg = cfg.get("seed", 42)
    seed = int(seed_cfg) if seed_cfg is not None else random.randint(0, 99999)

    # ============ 训练/验证/OOS 三区切分（防过拟合核心） ============
    # - 训练段 train_ratio：模型在其中反复试错学习
    # - 验证段 val_ratio：训练期间周期性评估，选最优模型（模型从未见过这段）
    # - OOS 段 oos_ratio：训练完全结束后的独立样本外评估（换一段未见过行情）
    # 三者不重叠，杜绝"训练=验证=评估同一段数据"的前视偏差。
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    # P1-5：OOS 最小样本数可配置（默认 80 保持兼容；持续进化引擎传 settings.evolve_oos_min_bars）
    oos_min_bars = max(80, int(cfg.get("oos_min_bars", 80)))
    n_total = len(df)
    n_train = max(300, int(n_total * train_ratio))
    n_val = max(100, int(n_total * val_ratio))
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    oos_df = df.iloc[n_train + n_val:]
    # 数据不足时退化为单段（不切分），保证小数据集仍可训练
    val_skipped = False  # P1-10：数据不足回退后 val 评估整体停用（防静默维度失配）
    if len(oos_df) < oos_min_bars or len(val_df) < 80:
        train_df = val_df = df
        oos_df = df.iloc[:0]
        val_skipped = True
        # P1-10：显式跳过而非静默失败——回退后 val_df 与 train_df 同一段，
        # val_env 的因子列/维度必然与训练 env 不一致（曾抛维度异常被 :462-463
        # 吞掉 → 早停/选优全部静默失效，best_agent 永不更新）
        log.warning("[drl] 数据不足(仅%d根)，跳过训练/验证/OOS 切分，val 评估停用（防止静默维度失配）", n_total)
    # 提前记录 val_skipped，供后续早停警告用（早停参数尚未定义，在定义处检查）

    # 外部因子信号列（可选）：训练/验证/OOS 用同一表达式在各自数据段上计算，
    # 与部署（rl_adaptive 滚动缓冲）口径一致。因子值标准化在训练段拟合。
    #
    # 两种来源：
    # 1. factor_expression：表达式字符串，训练/部署端都用 FactorExecutor 实时计算
    # 2. factor_values（新增，级联注入）：预计算好的因子值数组（长度 == len(df)），
    #    来自 factor_miner 产出的组合因子。部署端 rl_adaptive 无法实时重建，故
    #    同时要求 factor_values 写入模型文件，rl_adaptive 按索引取用（见其 _factor_value）。
    extra_factors_all: Optional[np.ndarray] = None
    extra_factors_val: Optional[np.ndarray] = None
    extra_factors_oos: Optional[np.ndarray] = None
    factor_expr = str(cfg.get("factor_expression", "") or "").strip()
    # 级联组合因子复算配方（{weights: {因子key: 权重}}，见 CompositeFactorEvaluator）
    factor_composite: Optional[dict] = None
    # 标准化统计量（训练段拟合）写入模型文件，部署端用同一口径标准化——
    # 曾只存表达式不存 mu/sd，部署喂原始值，状态分布与训练完全不同
    factor_mu: float = 0.0
    factor_sd: float = 1.0
    # P1：级联因子列（来自 factor_miner 组合因子）——预计算值直接注入
    factor_values: Optional[np.ndarray] = cfg.get("factor_values")
    if factor_values is not None:
        fvals = np.asarray(factor_values, dtype=float).reshape(-1)
        if len(fvals) != n_total:
            log.warning("[drl] factor_values 长度 %d 与K线数 %d 不一致，忽略级联因子",
                        len(fvals), n_total)
            factor_values = None
    if factor_values is not None:
        # 级联因子：训练段拟合 mu/sd 标准化（与表达式路径同口径）
        train_fv = factor_values[:n_train]
        valid = train_fv[~np.isnan(train_fv)]
        mu = float(valid.mean()) if len(valid) else 0.0
        sd = float(valid.std()) + 1e-8
        factor_mu, factor_sd = mu, sd
        vals = (factor_values - mu) / sd
        vals = np.nan_to_num(vals, nan=0.0)
        extra_factors_all = vals.reshape(-1, 1)
        extra_factors_val = (vals[n_train:n_train + n_val].reshape(-1, 1)
                             if (not val_skipped and len(val_df) == n_val) else None)
        extra_factors_oos = vals[n_train + n_val:].reshape(-1, 1) if len(oos_df) > 0 else None
        factor_expr = ""  # 级联因子不再用表达式（值已给定）
        log.info("[drl] 已注入级联因子列（factor_miner 组合因子, mu=%.4f sd=%.4f）", factor_mu, factor_sd)
        # 级联因子的复算配方（weights 映射）随结果透传，部署端写入模型文件：
        # 实盘按配方在滚动缓冲上复算组合因子（CompositeFactorEvaluator，与训练
        # expanding z + mu/sd 二次标准化同式）——曾只保存预计算数组，新K线
        # 无值可取、因子列恒 None，带级联因子的模型实盘每根K线都"跳过本根"。
        factor_composite = dict(cfg.get("factor_composite") or {}) or None
    elif factor_expr:
        from factors.mining import FactorExecutor
        try:
            fx = FactorExecutor(factor_expr)
            vals = fx.eval(df).astype(float).to_numpy(float)
            # 标准化统计量只在训练段【有效值】上拟合：
            # 曾先 nan_to_num(0) 再拟合，训练段大量 NaN 时 mu 被 0 污染；
            # 缺失值标准化后按 0（均值）填充，避免 NaN 进入状态向量
            train_vals = vals[:n_train]
            valid = train_vals[~np.isnan(train_vals)]
            mu = float(valid.mean()) if len(valid) else 0.0
            sd = float(valid.std()) + 1e-8
            factor_mu, factor_sd = mu, sd
            vals = (vals - mu) / sd
            vals = np.nan_to_num(vals, nan=0.0)
            extra_factors_all = vals.reshape(-1, 1)
            # 退化回退（val_skipped）时 val_df 与 train_df 同一段，不再切 val 切片
            # （曾产生空切片/维度失配 → val_env 构造即崩）
            extra_factors_val = (vals[n_train:n_train + n_val].reshape(-1, 1)
                                 if (not val_skipped and len(val_df) == n_val) else None)
            extra_factors_oos = vals[n_train + n_val:].reshape(-1, 1) if len(oos_df) > 0 else None
            log.info("[drl] 已注入因子信号列: %s（标准化后加入状态, mu=%.4f sd=%.4f）", factor_expr[:60], factor_mu, factor_sd)
        except (ValueError, TypeError, ImportError) as e:  # 因子表达式计算：值/类型/导入错误属于已知异常类型
            log.warning("[drl] 因子表达式计算失败，忽略因子列: %s", e)

    # GPU 后端：特征预计算走 cupy（可用时），训练循环仍 CPU（逐K线有状态依赖）
    from .env import gpu_available
    backend = "cupy" if (cfg.get("use_gpu") and gpu_available()) else "numpy"
    # 状态窗口（时序记忆）：默认 1=单点特征（兼容旧模型）；>1 时堆叠最近 N 根
    state_window = max(1, int(cfg.get("state_window", 1)))
    # P4-B5：预计算特征矩阵一次，主 env 与并行 worker 共用——曾主 env 内部
    # 算一遍（未传 precomputed_features）、worker 又算一遍，大 DataFrame 下
    # 启动即重复一次滚动窗口计算（cupy 后端更是双份 GPU 编译+拷贝）
    _precomputed_features = _precompute_features(train_df, backend=backend)
    # ---- P0 多标的混合训练：构建标的池（经验广度） ----
    # 单标的模式（未传 extra_dfs）：pool 只有主标的，sample() 恒返回主标的
    # 完整训练段——行为与旧路径完全一致（start=0、window=len(train_df)）。
    # 多标的模式：每 episode 以 extra_mix_ratio 概率随机选额外标的并在其
    # 数据内随机时间窗切片，网络共享，把"见过的局面"扩到多个标的多年行情。
    # 额外标的只作为经验来源，不参与 val/OOS 判据（部署语义仍由主标的承担）。
    extra_dfs = extra_dfs or {}
    extra_mix_ratio = float(cfg.get("extra_mix_ratio", 0.5))
    episode_window = int(cfg.get("episode_window", 0) or 0)
    # P1 课程学习：训练早期只从低难度（低波动）窗口采样，随进度逐步放开。
    # 默认关闭——不启用时采样行为与旧路径完全一致（向后兼容）。
    curriculum = bool(cfg.get("curriculum", False))
    curriculum_start = float(cfg.get("curriculum_start", 0.3))
    factor_builder = None
    if extra_dfs:
        # 跨标因子列构建器：主标的有因子列时，额外标的必须能重建同维度因子列
        # 才能共享网络。来源二选一：
        # 1. factor_composite 配方（factor_miner 级联组合因子）：配方可在任意
        #    标的 df 上复算（expanding z 无前视，与部署端同口径）
        # 2. factor_expression 表达式：FactorExecutor 对任意 df 计算同一表达式
        # 仅有预计算数组 factor_values（无配方无表达式）无法跨标重建 → 不传
        # factor_builder，pool 会因主标的带因子列而排除全部额外标的（降级单标）。
        if factor_composite:
            _weights = (factor_composite.get("weights") or {})
            if _weights:
                from .multi_asset import _build_composite_factor
                _comp_mu, _comp_sd = factor_mu, factor_sd

                def _fb(df, _w=_weights, _mu=_comp_mu, _sd=_comp_sd):
                    vals = _build_composite_factor(df, _w)
                    if vals is None:
                        return None
                    vals = (vals - _mu) / (_sd if _sd and _sd > 0 else 1.0)
                    return np.nan_to_num(vals, nan=0.0)
                factor_builder = _fb
        elif factor_expr:
            _expr = factor_expr
            _expr_mu, _expr_sd = factor_mu, factor_sd

            def _fb(df, _e=_expr, _mu=_expr_mu, _sd=_expr_sd):
                try:
                    from factors.mining import FactorExecutor
                    vals = FactorExecutor(_e).eval(df).astype(float).to_numpy(float)
                    vals = (vals - _mu) / (_sd if _sd and _sd > 0 else 1.0)
                    return np.nan_to_num(vals, nan=0.0)
                except Exception:  # noqa: BLE001
                    return None
            factor_builder = _fb
        elif factor_values is not None:
            log.warning("[drl] 多标的训练：主标的因子为预计算数组（无复算配方），"
                        "额外标的无法重建同维度因子列，本轮降级为单标的训练")
    if extra_dfs or curriculum:
        from .multi_asset import MultiAssetPool
        main_factors = (extra_factors_all[:len(train_df)].reshape(-1)
                        if extra_factors_all is not None else None)
        try:
            asset_pool = MultiAssetPool(
                train_df, _precomputed_features, main_factors=main_factors,
                extra_dfs=extra_dfs, state_window=state_window, backend=backend,
                factor_builder=factor_builder, window_len=episode_window,
                extra_mix_ratio=extra_mix_ratio,
                curriculum=curriculum, curriculum_start=curriculum_start,
                log=log)
            if asset_pool.extra_names:
                log.info("[drl] 多标的混合训练已启用: 额外标的 %s (mix_ratio=%.2f, window=%d)",
                         asset_pool.extra_names, extra_mix_ratio, asset_pool.window_len)
            elif extra_dfs:
                log.warning("[drl] 多标的混合训练：所有额外标的均被排除"
                            "（维度/长度不符），降级为单标的训练")
            if curriculum:
                log.info("[drl] 课程学习已启用: 起始难度分位=%.2f（训练早期只采样低波动段，"
                         "随进度逐步放开）", curriculum_start)
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] 多标的池构建失败，降级为单标的训练: %s", e)
            asset_pool = None
    else:
        asset_pool = None
    env = TradingEnv(train_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                     fee_rate=float(cfg.get("fee_rate", 0.001)),
                     vol_penalty=vol_penalty, cost_scale=cost_scale,
                     slippage=slippage,
                     min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                     backend=backend,
                     extra_factors=extra_factors_all[:len(train_df)] if extra_factors_all is not None else None,
                     state_window=state_window,
                     precomputed_features=_precomputed_features,
                     reward_dd_penalty=float(cfg.get("reward_dd_penalty", 0.0)),
                     reward_losing_penalty=float(cfg.get("reward_losing_penalty", 0.0)),
                     reward_trend_align=float(cfg.get("reward_trend_align", 0.0)),
                     tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
                     participation_rate=float(cfg.get("participation_rate", 0.0)),
                     funding_rate=float(cfg.get("funding_rate", 0.0)))
    agent = ACAgent(env.state_dim, len(ACTION_BUCKETS), hidden=hidden,
                    lr_actor=float(cfg.get("lr_actor", 2e-3)),
                    lr_critic=float(cfg.get("lr_critic", 5e-3)),
                    gamma=float(cfg.get("gamma", 0.99)), seed=seed,
                    entropy_coef=float(cfg.get("entropy_coef", 0.05)),
                    state_window=state_window,
                    backbone=str(cfg.get("backbone", "mlp")),
                    gru_hidden=int(cfg.get("gru_hidden", 32)))
    # 续训支持：两种起点（优先 base_agent，均为"在旧模型上继续学"）——
    # 1) base_agent：内存中的 ACAgent 对象（持续进化引擎用，已通过 zoo.load_agent
    #    加载，无需磁盘往返；必须 deepcopy 权重，否则训练会原地修改 best_agent，
    #    回退保护就形同虚设）
    # 2) base_model：模型文件路径（手动训练 API 用，兼容旧调用）
    # 注意：续训只是起点不同，训练/验证/OOS 切分、OOS 硬门、过拟合衰减全部照常生效。
    base_agent = cfg.get("base_agent")
    base_model = cfg.get("base_model")
    if base_agent is not None:
        try:
            if base_agent.state_dim == agent.state_dim and base_agent.n_actions == agent.n_actions:
                agent.actor = copy.deepcopy(base_agent.actor)
                agent.critic = copy.deepcopy(base_agent.critic)
                log.info("[drl] 已加载内存基础模型作为续训起点（state_dim=%d）", agent.state_dim)
                # 与 base_model 路径同口径：重设 cfg 的学习率（退火起点 L689-690 之前）
                agent.actor.lr = float(cfg.get("lr_actor", 2e-3))
                agent.critic.lr = float(cfg.get("lr_critic", 5e-3))
                # 续训时保留少量探索，避免刚接手就过度确定
                agent.epsilon = 0.4
            else:
                log.warning("[drl] 基础模型维度不匹配(%s vs %s)，忽略，从零训练",
                            getattr(base_agent, "state_dim", "?"), agent.state_dim)
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] 内存基础模型接入失败，从零训练: %s", e)
    elif base_model:
        try:
            from .agent import ACAgent as _ACA
            base = _ACA.load(base_model)
            if base.state_dim == agent.state_dim and base.n_actions == agent.n_actions:
                agent.actor = base.actor
                agent.critic = base.critic
                log.info("[drl] 已加载基础模型 %s 作为续训起点", base_model)
                # P4-E1：替换网络后显式恢复 cfg 的学习率——base 的 to_dict/from_dict
                # 不持久化 lr，加载往返后 actor/critic 的 lr 恒为默认 1e-3/3e-3，
                # 此前 lr_actor_init 取到的是默认值，cfg 的 lr_actor/lr_critic
                # 被静默丢弃、退火起点全错。这里在退火初始化（L689-690）前重设。
                agent.actor.lr = float(cfg.get("lr_actor", 2e-3))
                agent.critic.lr = float(cfg.get("lr_critic", 5e-3))
                # 续训时保留少量探索，避免刚接手就过度确定
                agent.epsilon = 0.4
            else:
                log.warning("[drl] 基础模型维度不匹配(%s)，忽略，从零训练",
                            getattr(base, "state_dim", "?"))
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:  # 基础模型加载：文件不存在/JSON解析/键缺失属于已知异常类型
            log.warning("[drl] 基础模型加载失败(%s)，从零训练: %s", base_model, e)
    agent.set_epsilon_schedule(episodes)

    # 经验回放缓冲区（可选）：capacity>0 时启用，混合旧轨迹提升样本效率。
    # 旧轨迹经重要性采样比率自然衰减，PPO 兼容（ratio=exp(logp_new-logp_old)）。
    #
    # cfg["replay_buffer"]：调用方可传入既有缓冲区以**跨训练轮存活**——持续进化
    # 引擎据此让上一轮（甚至更早窗口）的轨迹参与本轮更新，这是对抗灾难性遗忘的
    # 关键：每轮只训 8ep 时，若缓冲每轮重建，则旧行情经验在轮间被完全丢弃。
    # 未传入时维持旧行为（每次调用内新建，只在同轮内混合）。
    replay_mix_ratio = float(cfg.get("replay_mix_ratio", 0.3))
    replay_buf = cfg.get("replay_buffer")
    if replay_buf is not None:
        _prev_dim = replay_buf.feature_dim()
        if _prev_dim is not None and _prev_dim != agent.state_dim:
            # 维度守卫：配置变更后旧轨迹维度不符，复用会在 concatenate 处崩溃
            log.warning("[drl] 回放缓冲区维度不符（缓冲 %s vs 当前 %s），已清空重建",
                        _prev_dim, agent.state_dim)
            replay_buf.clear()
        replay_capacity = int(replay_buf.capacity)
    else:
        replay_capacity = int(cfg.get("replay_capacity", 0))  # 0=禁用
        replay_buf = ReplayBuffer(capacity=replay_capacity) if replay_capacity > 0 else None
    if replay_buf is not None:
        log.info("[drl] 经验回放已启用: capacity=%d mix_ratio=%.2f 复用轨迹=%d",
                 replay_capacity, replay_mix_ratio, len(replay_buf))

    history: list[dict] = []
    best_train_ret = -1e9
    best_val_ret = -1e9
    best_agent: Optional[ACAgent] = None
    # 验证评估间隔：默认10轮。2026-09-20（诊断 §15）：持续进化 episodes=8 时
    # 默认10会导致 val 只在最后一轮（ep==episodes 兜底）评估一次，"周期性按
    # val 选最优"失效。钳到 episodes//3（至少1），保证训练期间≥3次 val 评估。
    _val_interval_cfg = max(1, int(cfg.get("val_eval_interval", 10)))
    val_eval_interval = max(1, min(_val_interval_cfg, max(1, episodes // 3)))
    # 2026-09-20（diagnosis §15 / regime 诊断）：val 段单一时序可能只有单一
    # regime（如下行震荡），选出的"保守模型"对上行的 OOS 段不交易（position_ratio=0）。
    # val_segments>1 时把 val 段切成多段，选模收益取各段中位数（与 oos_segments 同构），
    # 让模型必须在多个子段都表现好才被选，不偏向单一 regime。默认 1 保持兼容。
    val_segments = max(1, int(cfg.get("val_segments", 1)))
    # 早停保护：验证收益连续无改善次数达到阈值则提前停止（借鉴 SB3 EvalCallback）
    no_improve_streak = 0
    early_stop_patience = int(cfg.get("early_stop_patience", 0))  # 0=禁用，>0 启用
    # 数据不足 + 用户要求早停 → 早停实际不会触发（val 被跳过），显式警告
    if val_skipped and early_stop_patience > 0:
        log.warning("[drl] 数据不足，早停保护已禁用（early_stop_patience=%d 但 val 评估被跳过），"
                    "请增加数据量或关闭 early_stop_patience=0", early_stop_patience)
    # 强制自动训练：开启后即使触发早停也不中断，重置早停计数继续训练，
    # 直到达到用户设置的轮数（进度条不重置，连续显示总进度）
    force_train = bool(cfg.get("force_train", False))
    force_rounds = 0  # 强制模式下早停被越过的次数
    # 过拟合衰减系数：训练收益 > 验证收益时衰减下一步优势权重，限制噪声记忆
    _adv_scale = 1.0
    _last_val_gap = 0.0
    reward_val_gap_penalty = float(cfg.get("reward_val_gap_penalty", 0.5))

    # 验证用无探索环境（确定性策略，评估真实策略质量）
    # 关键：验证段用独立的 val_df（与训练段不重叠），杜绝"训练=验证同一段数据"的前视偏差
    # P1-10：退化回退（val_skipped）时 val_df 与 train_df 同一段且因子列切片为空，
    # 构造 val_env 只会维度失配/语义失真 → 直接停用（不构造）
    val_env = None
    if not val_skipped:
        val_env = TradingEnv(val_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                             fee_rate=float(cfg.get("fee_rate", 0.001)),
                             vol_penalty=vol_penalty, cost_scale=1.0,
                             slippage=slippage,
                             min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                             extra_factors=extra_factors_val,
                             state_window=state_window,
                             tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
                             participation_rate=float(cfg.get("participation_rate", 0.0)),
                             funding_rate=float(cfg.get("funding_rate", 0.0)))

    # 学习率退火（PPO 标配技巧）：随训练进度从初始值线性降到 lr_final_ratio 倍
    # 注：曾尝试加入 warmup，A/B 多 seed 验证无正面提升（均值 2.85 vs 3.14），已回退。
    lr_actor_init = agent.actor.lr
    lr_critic_init = agent.critic.lr
    lr_final_ratio = float(cfg.get("lr_final_ratio", 0.2))
    # 注：熵正则退火 A/B 验证无正面提升（均值 2.16 vs 3.39），保持固定熵。

    # 批量收集轨迹（n_episodes 条）→ 拼接 → 真 PPO 更新（多轮 mini-batch）
    # 2026-08：并行收集（每个轨迹独立 env 实例，线程安全），
    # n_episodes 较大时加速显著（环境计算含指标，CPU 密集但可并行）
    # P2-8 优化：线程池在循环外创建一次，避免每轮重建线程的开销
    # P2-9 优化：预计算特征矩阵一次，所有 worker 共享（避免每个 worker 重复调用 _precompute_features）

    # 特征矩阵已在主 env 构造前预计算一次（P4-B5），worker 直接共享
    def _collect_one(args: tuple) -> dict:
        """每个 worker 用独立 env 实例，避免共享状态竞争。
        特征矩阵已预计算并传入，不重复计算 _precompute_features。
        D1：每个 worker 用独立 rng（seed 派生，可复现）——绝不共享 agent._rng，
        numpy Generator 非线程安全，共享会导致采样非确定/竞态。"""
        ep_i, worker_i = args
        local_rng = np.random.default_rng(seed + ep_i * 1000 + worker_i)
        if asset_pool is not None:
            # 多标的/课程学习：随机标的 × 随机时间窗切片（pool 内部线程安全，
            # rng 独立）。课程模式把训练进度 frac 传给采样器，早期只采低难度段。
            _frac = (ep_i / episodes) if curriculum else 1.0
            sample = asset_pool.sample(local_rng, frac=_frac)
            from .multi_asset import make_worker_env
            local_env = make_worker_env(sample, cfg, backend=backend)
        else:
            # 单标的：主标的完整训练段（旧路径，行为不变）
            extra_for_worker = (extra_factors_all[:len(train_df)]
                                if extra_factors_all is not None else None)
            local_env = TradingEnv(train_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                                   fee_rate=float(cfg.get("fee_rate", 0.001)),
                                   vol_penalty=vol_penalty, cost_scale=cost_scale,
                                   slippage=slippage,
                                   min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                                   backend=backend,
                                   extra_factors=extra_for_worker,
                                   state_window=state_window,
                                   reward_dd_penalty=float(cfg.get("reward_dd_penalty", 0.0)),
                                   reward_losing_penalty=float(cfg.get("reward_losing_penalty", 0.0)),
                                   reward_trend_align=float(cfg.get("reward_trend_align", 0.0)),
                                   tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
                                   participation_rate=float(cfg.get("participation_rate", 0.0)),
                                   funding_rate=float(cfg.get("funding_rate", 0.0)),
                                   precomputed_features=_precomputed_features)
        return agent.collect_episode(local_env, rng=local_rng)

    pool = ThreadPoolExecutor(max_workers=n_collect_workers)
    if n_collect_workers > 4:
        log.info("[drl] 并行收集上量: n_episodes=%d → %d 线程并行（自动按核数）",
                 n_episodes, n_collect_workers)
    try:
        for ep in range(1, episodes + 1):
            # 时间预算：至少完成 1 轮后超时即提前收尾（保证 history 非空、best_agent 有值）
            if time_budget > 0 and ep > 1 and (time.time() - t0) >= time_budget:
                log.info("[drl] 单轮训练达到时间预算 %.0fs（已完成 %d 轮），提前结束",
                         time_budget, ep - 1)
                break
            agent.set_episode_epsilon(ep - 1)
            # 学习率退火：训练后期降低更新幅度，收敛更稳
            frac = ep / episodes
            lr_scale = lr_final_ratio + (1 - lr_final_ratio) * (1 - frac)
            agent.actor.lr = lr_actor_init * lr_scale
            agent.critic.lr = lr_critic_init * lr_scale

            # 并行收集轨迹（每个 worker 独立 rng，见 D1）
            trajs = list(pool.map(_collect_one, [(ep, i) for i in range(n_episodes)]))
            # 推入经验回放缓冲区（启用时）
            if replay_buf is not None:
                for t in trajs:
                    replay_buf.push(t["states"], t["actions"], t["rewards"], t["old_log_probs"])
            # 从缓冲区混合采样（启用时）或直接用当前批量
            if replay_buf is not None and len(replay_buf) > 0:
                mixed = replay_buf.sample(
                    batch_size=max(len(t["states"]) for t in trajs),  # 目标批量大小 ≈ 当前最大轨迹
                    mix_ratio=replay_mix_ratio)
                states = mixed["states"]
                actions = mixed["actions"]
                rewards = mixed["rewards"]
                old_log_probs = mixed["old_log_probs"]
                seg_lens = mixed["segment_lengths"]
            else:
                states = np.concatenate([t["states"] for t in trajs])
                actions = np.concatenate([t["actions"] for t in trajs])
                rewards = np.concatenate([t["rewards"] for t in trajs])
                old_log_probs = np.concatenate([t["old_log_probs"] for t in trajs])
                seg_lens = [t["steps"] for t in trajs]
            stats = agent.train_batch(states, actions, rewards, old_log_probs,
                                  ppo_epochs=ppo_epochs,
                                  mini_batch_size=min(mini_batch_size, len(states)),
                                  segment_lengths=seg_lens,
                                  advantage_scale=_adv_scale,
                                  actor_grad_clip=actor_grad_clip)
            # 批量收益：取各轨迹收益的均值（比单条轨迹更稳）
            total_ret = float(np.mean([t["total_ret"] for t in trajs]))
            row = {
                "episode": ep, "total_ret": round(total_ret, 6),
                "final_equity": round(float(np.mean([t["final_equity"] for t in trajs])), 2),
                "steps": int(np.mean([t["steps"] for t in trajs])),
                "policy_loss": round(stats["policy_loss"], 6),
                "value_loss": round(stats["value_loss"], 6),
                "epsilon": stats["epsilon"],
            }
            if total_ret > best_train_ret:
                best_train_ret = total_ret
            if val_skipped:
                # P1-10：退化分支显式标注（前端/日志可识别，val 评估整体停用）
                row["val_skipped"] = True

            # 周期性用确定性策略验证，按验证收益选最优模型（避免训练期随机轨迹误导）
            val_ret = None
            if (ep % val_eval_interval == 0 or ep == episodes) and not val_skipped:
                try:
                    vtraj = None
                    v_equities: list = []
                    # 多段 val（val_segments>1）：val 段切成多个子段逐段确定性 greedy，
                    # 取收益中位数作为本轮 val_ret——避免单一 regime 选出"只在某段行情
                    # 表现好"的模型（regime 诊断：val 下行震荡 → OOS 上行不交易）。
                    if val_segments > 1 and len(val_df) >= val_segments * 80:
                        _vseg = len(val_df) // val_segments
                        _vrets: list[float] = []
                        for _j in range(val_segments):
                            _s = val_df.iloc[_j * _vseg:(len(val_df) if _j == val_segments - 1 else (_j + 1) * _vseg)]
                            _se_extra = None
                            if extra_factors_val is not None:
                                _se_extra = extra_factors_val[_j * _vseg:(len(val_df) if _j == val_segments - 1 else (_j + 1) * _vseg)]
                            _venv = TradingEnv(_s, start_cash=float(cfg.get("start_cash", 10000.0)),
                                               fee_rate=float(cfg.get("fee_rate", 0.001)),
                                               vol_penalty=vol_penalty, cost_scale=1.0,
                                               slippage=slippage,
                                               min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                                               extra_factors=_se_extra,
                                               state_window=state_window,
                                               tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
                                               participation_rate=float(cfg.get("participation_rate", 0.0)),
                                               funding_rate=float(cfg.get("funding_rate", 0.0)))
                            _vt = run_episode(_venv, lambda s: agent.greedy_action(s))
                            _vrets.append(float(_vt["total_ret"]))
                            v_equities += [i.get("equity") for i in _vt.get("infos", [])
                                           if i.get("equity") is not None]
                        val_ret = float(np.median(_vrets))
                    else:
                        vtraj = run_episode(val_env, lambda s: agent.greedy_action(s))
                        val_ret = float(vtraj["total_ret"])
                        v_equities = [info.get("equity") for info in vtraj.get("infos", [])
                                      if info.get("equity") is not None]
                    row["val_ret"] = round(val_ret, 6)
                    row["val_segments"] = val_segments
                    # ---- 验证评估指标增强 ----
                    val_sharpe = _equity_sharpe(v_equities) if len(v_equities) > 2 else 0.0
                    val_max_dd = _equity_drawdown(v_equities) if len(v_equities) > 2 else 0.0
                    row["val_sharpe"] = round(val_sharpe, 4)
                    row["val_max_dd"] = round(val_max_dd, 4)
                    # ---- 过拟合衰减：训练收益 > 验证收益时衰减下一步优势权重 ----
                    # Pine 代码权重固化后无法持续适应，过拟合直接导致 TradingView 亏损。
                    # 此扣分项让 PPO 更新在泛化差距大时更保守，抑制噪声记忆。
                    if reward_val_gap_penalty > 0:
                        _val_gap = max(0.0, total_ret - val_ret)
                        if _val_gap > 0.001:
                            _adv_scale = 1.0 / (1.0 + reward_val_gap_penalty * _val_gap)
                            row["val_gap"] = round(_val_gap, 6)
                            row["adv_scale"] = round(_adv_scale, 4)
                            log.info("[drl] 过拟合衰减: gap=%.4f adv_scale=%.4f", _val_gap, _adv_scale)
                        else:
                            _adv_scale = 1.0
                    if val_ret > best_val_ret:
                        best_val_ret = val_ret
                        # P2：JSON 往返（to_dict→from_dict 全量序列化）改为 deepcopy
                        best_agent = copy.deepcopy(agent)
                        no_improve_streak = 0
                    else:
                        no_improve_streak = no_improve_streak + 1
                        # 早停保护（借鉴 SB3 EvalCallback）：验证收益连续 N 次无改善则提前停止，
                        # 避免训练后期退化浪费算力（大规模训练尤其重要）
                        # 早停仅在 patience>0 时启用（0=禁用，跑满 episodes）
                        if early_stop_patience > 0 and no_improve_streak >= early_stop_patience:
                            if force_train:
                                # 强制自动训练：早停触发但继续训练，重置早停计数，
                                # 训练进入新阶段（进度条不重置，模型保留已学知识）
                                force_rounds += 1
                                no_improve_streak = 0
                                row["early_stop_overridden"] = True
                                row["force_round"] = force_rounds
                                log.info("[drl] 强制训练模式：第%s轮早停被越过，继续训练", force_rounds)
                            else:
                                log.info("[drl] 验证收益 %s 轮无改善，提前停止于第 %s 轮", early_stop_patience, ep)
                                row["early_stopped"] = True
                                history.append(row)
                                break
                except Exception as e:  # noqa: BLE001
                    log.warning("[drl] 验证失败: %s", e)
            # 过拟合衰减：验证后更新下一批的 advantage 缩放系数。
            # 当训练收益 > 验证收益时（泛化差距），按比例衰减优势权重，
            # 让 PPO 更新更保守（避免模型记忆噪声模式）。
            # 仅当 reward_val_gap_penalty>0 且本轮有验证结果时计算。
            if val_ret is not None and reward_val_gap_penalty > 0:
                gap = total_ret - val_ret
                _last_val_gap = max(gap, 0.0)
                if gap > 0.001:
                    _adv_scale = 1.0 / (1.0 + reward_val_gap_penalty * max(gap, 0.0))
                else:
                    _adv_scale = 1.0
                row["adv_scale"] = round(_adv_scale, 4)
                row["val_gap"] = round(_last_val_gap, 6)
            # 无验证轮次：保留上一轮的 _adv_scale（不重置）
            history.append(row)

            if on_progress:
                try:
                    on_progress({
                        "episode": ep, "episodes": episodes,
                        "total_ret": round(total_ret, 6),
                        "best_ret": round(max(best_train_ret, best_val_ret), 6),
                        "val_ret": None if val_ret is None else round(val_ret, 6),
                        "final_equity": round(row["final_equity"], 2),
                        "epsilon": stats["epsilon"],
                        "policy_loss": stats["policy_loss"],
                        "elapsed_sec": round(time.time() - t0, 1),
                    })
                except Exception as e:  # noqa: BLE001
                    log.warning("[drl] 进度回调异常: %s", e)

    # 若期间从未验证成功（异常），退回当前 agent
    finally:
        pool.shutdown(wait=True)
    final_agent = best_agent or agent
    final_agent.epsilon = final_agent.epsilon_min

    # ============ OOS 独立评估（防过拟合核心） ============
    # 训练结束后用【完全未见过】的 oos_df 评估最优模型，与训练段对比：
    # 若训练段收益高但 OOS 大幅缩水 → 过拟合信号。OOS 段在训练/验证中从未被使用。
    oos_report = {"enabled": False}
    deployment_blocked = False  # D3：OOS 硬门触发后置 True（拦截 Pine 自动产出）
    if len(oos_df) >= oos_min_bars:
        try:
            # P1-5：多段 OOS 评估——把 OOS 段切成连续子段，每段独立跑 greedy 取中位数，
            # 降低单段行情运气导致的评估噪声；段数=1 或数据不足时退化为单段（原逻辑）。
            oos_segments = max(1, int(cfg.get("oos_segments", 1)))
            seg_len = len(oos_df) // oos_segments
            oos_rets: list = []
            oos_pos_ratios: list = []
            oos_equities_all: list = []
            oos_segment_equities: list[list[float]] = []
            oos_final_equities: list = []
            if oos_segments > 1 and seg_len >= oos_min_bars:
                for i in range(oos_segments):
                    start = i * seg_len
                    end = len(oos_df) if i == oos_segments - 1 else (i + 1) * seg_len
                    seg = oos_df.iloc[start:end]
                    seg_extra = (extra_factors_oos[start:end]
                                 if extra_factors_oos is not None else None)
                    seg_env = TradingEnv(seg, start_cash=float(cfg.get("start_cash", 10000.0)),
                                         fee_rate=float(cfg.get("fee_rate", 0.001)),
                                         vol_penalty=vol_penalty, cost_scale=1.0,
                                         slippage=slippage,
                                         min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                                         extra_factors=seg_extra,
                                         state_window=state_window,
                                         participation_rate=float(cfg.get("participation_rate", 0.0)),
                                         funding_rate=float(cfg.get("funding_rate", 0.0)))
                    seg_traj = run_episode(seg_env, lambda s: final_agent.greedy_action(s))
                    oos_rets.append(float(seg_traj["total_ret"]))
                    oos_pos_ratios.append(float(seg_traj["final_position_ratio"]))
                    oos_final_equities.append(float(seg_traj["final_equity"]))
                    seg_equities = [info.get("equity") for info in seg_traj.get("infos", [])
                                    if info.get("equity") is not None]
                    oos_segment_equities.append(seg_equities)
                oos_ret = float(np.median(oos_rets))  # 中位数收益：对单段运气更稳健
                oos_position_ratio = float(np.median(oos_pos_ratios))
                oos_equity_final = float(np.median(oos_final_equities))
            else:
                oos_env = TradingEnv(oos_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                                     fee_rate=float(cfg.get("fee_rate", 0.001)),
                                     vol_penalty=vol_penalty, cost_scale=1.0,
                                     slippage=slippage,
                                     min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                                     extra_factors=extra_factors_oos,
                                     state_window=state_window,
                                     participation_rate=float(cfg.get("participation_rate", 0.0)),
                                     funding_rate=float(cfg.get("funding_rate", 0.0)))
                oos_traj = run_episode(oos_env, lambda s: final_agent.greedy_action(s))
                oos_ret = float(oos_traj["total_ret"])
                oos_position_ratio = float(oos_traj["final_position_ratio"])
                oos_equity_final = float(oos_traj["final_equity"])
                oos_equities_all = [info.get("equity") for info in oos_traj.get("infos", [])
                                    if info.get("equity") is not None]
                oos_rets = [oos_ret]
                oos_pos_ratios = [oos_position_ratio]
            if oos_segment_equities:
                oos_equity_metrics = _aggregate_oos_segment_metrics(oos_segment_equities)
            else:
                oos_equity_metrics = {
                    "sharpe": _equity_sharpe(oos_equities_all)
                    if len(oos_equities_all) > 2 else 0.0,
                    "max_drawdown": _equity_drawdown(oos_equities_all)
                    if len(oos_equities_all) > 2 else 0.0,
                }
            # 训练段代表收益（用验证段 best 或训练段末收益）
            train_ref = best_train_ret if best_train_ret > -1e8 else float(total_ret)
            # 过拟合判断：OOS 收益相对训练段收益的衰减（D3，语义见 _oos_decay）
            decay = _oos_decay(train_ref, oos_ret)
            # P1-7：市场状态（regime）对比——训练段 vs OOS 段波动率/趋势方向差异。
            # 训练段趋势市 + OOS 震荡市会让任何模型显得"过拟合"，反之假通过；
            # 此指标帮助判断 OOS 收益差异是否由行情状态剧变造成（而非模型能力）。
            _tr = train_df["close"].pct_change().dropna()
            _or = oos_df["close"].pct_change().dropna()
            _train_vol = float(_tr.std()) if len(_tr) > 2 else 0.0
            _oos_vol = float(_or.std()) if len(_or) > 2 else 0.0
            _vol_ratio = _oos_vol / (_train_vol + 1e-12) if _train_vol > 0 else 0.0
            _train_trend = float(train_df["close"].iloc[-1] / train_df["close"].iloc[0] - 1.0)
            _oos_trend = float(oos_df["close"].iloc[-1] / oos_df["close"].iloc[0] - 1.0)
            # 波动率剧变（>2.5 倍或 <0.4 倍）或趋势方向相反且都较显著 → 判定状态剧变
            _regime_shift = bool(_vol_ratio > 2.5 or _vol_ratio < 0.4
                                 or (_train_trend * _oos_trend < 0
                                     and abs(_train_trend) > 0.05 and abs(_oos_trend) > 0.05))
            oos_report = {
                "enabled": True,
                "oos_ret": round(oos_ret, 6),
                "oos_rets": [round(r, 6) for r in oos_rets],  # P1-5：多段收益列表（单段时同旧口径）
                "oos_segments": len(oos_rets),
                "oos_ret_std": round(float(np.std(oos_rets)), 6) if len(oos_rets) > 1 else 0.0,
                "train_ret": round(train_ref, 6),
                "decay": round(decay, 4),
                "overfit_likely": bool(decay > 0.5),  # OOS 收益衰减过半 → 疑似过拟合
                "oos_equity_final": round(oos_equity_final, 2),
                "oos_position_ratio": round(oos_position_ratio, 4),
                "oos_sharpe": round(oos_equity_metrics["sharpe"], 4),
                "oos_max_drawdown": round(oos_equity_metrics["max_drawdown"], 4),
                # P1-7：regime 对比
                "train_vol": round(_train_vol, 6),
                "oos_vol": round(_oos_vol, 6),
                "vol_ratio": round(_vol_ratio, 3),
                "train_trend": round(_train_trend, 4),
                "oos_trend": round(_oos_trend, 4),
                "regime_shift": _regime_shift,
            }
            # D3：OOS 硬门——样本外衰减过半视为过拟合，默认拒绝自动部署（Pine 产出）。
            # 与 AI 策略过拟合守卫（严重过拟合即拦截）对齐，可 cfg.oos_hard_gate=False 关闭。
            oos_hard_gate = bool(cfg.get("oos_hard_gate", True))
            if oos_hard_gate and oos_report["overfit_likely"]:
                oos_report["hard_rejected"] = True
                oos_report["reason"] = (
                    f"OOS 收益 {oos_ret:.4f} 较训练段 {train_ref:.4f} 衰减 {decay:.1%}，"
                    f"疑似过拟合，已拦截自动部署（可关闭 oos_hard_gate 强制产出）")
                deployment_blocked = True
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] OOS 评估失败: %s", e)
            oos_report = {"enabled": False}

    # DRL → Pine：与训练端同一口径（含因子 mu/sd 标准化），不可导出时给出原因说明
    _metrics = {"train_ret": round(best_train_ret, 4),
                "oos_ret": round(oos_report.get("oos_ret", 0.0), 4) if oos_report.get("enabled") else "-",
                "sharpe": round(oos_report.get("oos_sharpe", 0.0), 3) if oos_report.get("enabled") else "-"}
    # ---- 蒸馏导出（Pine 体积门 × 网络容量的长期解） ----
    # teacher（final_agent）负责软件内实盘；若它装不进 Pine（大网络/级联因子），
    # 蒸馏一个可导出的小网络（student）专供 TradingView 自动交易代码。
    # 配置：
    #   distill: 是否启用蒸馏（默认 True——体积门拦截时自动蒸馏，不静默失败）
    #   distill_hidden: student 网络（默认 [16,16]，实测可装进 Pine）
    #   distill_epochs / distill_lr: 蒸馏训练超参
    # teacher 本身可导出时跳过蒸馏（向后兼容，行为不变）。
    distill_enabled = bool(cfg.get("distill", True))
    distill_hidden = [int(h) for h in (cfg.get("distill_hidden") or [16, 16])]
    distill_epochs = max(10, int(cfg.get("distill_epochs", 300)))
    distill_lr = float(cfg.get("distill_lr", 5e-3))
    distill_agent_obj = None   # student（可导出的小网络）
    distill_meta = None
    if deployment_blocked:
        pine_code, pine_note = "", (oos_report.get("reason") or "OOS 硬门判定疑似过拟合，已拦截自动部署")
    else:
        pine_code, pine_note = _build_drl_pine(
            final_agent, str(cfg.get("name") or "rl_agent"),
            factor_expression=factor_expr, factor_mu=factor_mu, factor_sd=factor_sd,
            train_metrics=_metrics)
        # teacher 装不进 Pine 且启用了蒸馏 → 蒸馏小网络再试导出
        if not pine_code and distill_enabled:
            try:
                from .distill import distill_agent as _distill
                from .pine_export import estimate_pine_bytes, PINE_MAX_BYTES
                if estimate_pine_bytes(env.state_dim, distill_hidden,
                                       len(ACTION_BUCKETS)) <= PINE_MAX_BYTES:
                    log.info("[drl] teacher 装不进 Pine（%s），蒸馏小网络导出（hidden=%s）",
                             pine_note[:60], distill_hidden)
                    distill_agent_obj = _distill(
                        final_agent, train_df, cfg,
                        student_hidden=distill_hidden,
                        epochs=distill_epochs, lr=distill_lr,
                        seed=((int(seed) + 7) if seed is not None else 42))
                    distill_meta = dict(getattr(distill_agent_obj, "distill_meta", {}))
                    pine_code, pine_note = _build_drl_pine(
                        distill_agent_obj, str(cfg.get("name") or "rl_agent"),
                        factor_expression=factor_expr, factor_mu=factor_mu,
                        factor_sd=factor_sd, train_metrics=_metrics)
                    if pine_code:
                        # 蒸馏版标记由 build_drl_pine 依据 is_distilled 自动写入
                        # 头部（此处不再手动 replace——与模型文件重导路径同源）
                        final_agent.distill = distill_agent_obj
                        final_agent.distill_meta = distill_meta
                        log.info("[drl] 蒸馏导出成功: Pine %d B（KL 收敛见日志）", len(pine_code))
                    else:
                        log.warning("[drl] 蒸馏后仍无法导出 Pine: %s", pine_note)
            except Exception as e:  # noqa: BLE001
                log.warning("[drl] 蒸馏失败（保持原不可导出状态）: %s", e)

    result = {
        "agent": final_agent,
        "history": history,
        "best_ret": round(max(best_train_ret, best_val_ret), 6),
        "best_val_ret": round(best_val_ret, 6),
        "val_segments": val_segments,
        "final_ret": round(total_ret, 6),
        "episodes": episodes,
        "state_dim": env.state_dim,
        "n_actions": len(ACTION_BUCKETS),
        "action_buckets": ACTION_BUCKETS.tolist(),
        "action_names": ACTION_NAMES,
        "algo": "ppo",
        "ppo_cfg": {"n_episodes": n_episodes, "ppo_epochs": ppo_epochs,
                    "mini_batch_size": mini_batch_size, "clip_eps": 0.2,
                    "factor_expression": factor_expr,
                    "replay": {"capacity": replay_capacity, "mix_ratio": replay_mix_ratio}},
        # P0 并行采样上量：记录实际使用的并行线程数（诊断/审计可见）
        "collect_workers": n_collect_workers,
        # P1 课程学习：难度分层采样开关与起始难度分位（关闭时 False/0）
        "curriculum": bool(curriculum),
        "curriculum_start": round(curriculum_start, 4),
        # P0 多标的混合训练元数据（单标的训练时 extra_names=[] / multi_symbol=False）
        "multi_symbol": bool(asset_pool is not None
                             and asset_pool.extra_names),
        "multi_extra_names": (list(asset_pool.extra_names)
                              if asset_pool is not None else []),
        "multi_mix_ratio": (asset_pool.extra_mix_ratio
                            if asset_pool is not None else 0.0),
        "time_budget": time_budget,
        "factor_expression": factor_expr,
        "factor_mu": factor_mu,
        "factor_sd": factor_sd,
        # 级联因子列（factor_miner 组合因子）的原始值，供部署端 rl_adaptive 用
        "factor_values": (factor_values.tolist() if factor_values is not None else None),
        # 级联因子复算配方（weights 映射）：部署端写入模型文件，实盘按此在
        # 滚动缓冲上复算组合因子（详见 factors.mining.CompositeFactorEvaluator）
        "factor_composite": factor_composite,
        # DRL 神经引擎 → Pine Script v5 自动交易代码（可直接粘贴 TradingView）
        # D3：OOS 硬门触发时不产出 Pine（deployment_blocked=True），防止过拟合模型上线
        "pine_code": pine_code,
        "pine_note": pine_note,  # 装不进 Pine / 被拦截时的显式原因（供 UI 提示）
        # 蒸馏导出（Pine 体积门解）：student 小网络（供 TradingView），
        # teacher 仍为 final_agent（软件内实盘）。未蒸馏时 None。
        # 也挂在 final_agent.distill 上随模型文件序列化（to_dict 持久化）。
        "distill_agent": distill_agent_obj,
        "distill_meta": distill_meta,
        # UI/审计：本次训练是否实际走了蒸馏导出（大网络 → 小网络）
        "distill_used": distill_agent_obj is not None,
        "deployment_blocked": deployment_blocked,  # D3：OOS 硬门状态
        "elapsed_sec": round(time.time() - t0, 2),
        "backend": backend,
        "oos_report": oos_report,
        "cfg": cfg,
    }
    if val_skipped:
        # P1-10：退化分支标记（返回 dict 键只增不减，其余键不变）
        result["val_skipped"] = True
    return result


def _build_drl_pine(agent, model_name: str, factor_expression: str = "",
                     factor_mu: float = 0.0, factor_sd: float = 1.0,
                     train_metrics: dict | None = None) -> tuple[str, str]:
    """DRL 神经引擎 → Pine Script v5 自动交易代码（权重固化前向）。

    返回 (代码, 不可导出原因)：模型装不进 Pine 时不能只留空串——UI 会据此
    显示一段与训练模型无关的模板代码，原因必须显式带到用户面前。
    """
    from .pine_export import try_build_drl_pine
    code, note = try_build_drl_pine(agent, model_name=model_name or "rl_agent",
                                    factor_expression=factor_expression,
                                    factor_mu=factor_mu, factor_sd=factor_sd,
                                    train_metrics=train_metrics)
    if note:
        log.warning("[drl] Pine 代码未产出（不影响模型保存）: %s", note)
    return code, note


def _equity_sharpe(equities: list) -> float:
    """从权益序列算年化夏普（简化为周期内夏普）。"""
    import numpy as _np
    eq = _np.asarray(equities, dtype=float)
    if len(eq) < 3:
        return 0.0
    rets = _np.diff(eq) / (eq[:-1] + 1e-9)
    std = rets.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(rets.mean() / std * _np.sqrt(len(rets)))


def _equity_drawdown(equities: list) -> float:
    """最大回撤。"""
    import numpy as _np
    eq = _np.asarray(equities, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = _np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-9)
    return float(dd.max())


def _aggregate_oos_segment_metrics(segments: list[list[float]]) -> dict[str, float]:
    """按独立 OOS 段聚合权益指标，避免段间重置造成虚假跳点。"""
    import numpy as _np
    sharpes = [_equity_sharpe(eq) for eq in segments if len(eq) > 2]
    drawdowns = [_equity_drawdown(eq) for eq in segments if len(eq) > 1]
    return {
        "sharpe": float(_np.median(sharpes)) if sharpes else 0.0,
        "max_drawdown": float(_np.median(drawdowns)) if drawdowns else 0.0,
    }


def evaluate_agent(agent: ACAgent, df, cfg=None) -> dict:
    """用确定性策略评估智能体在数据上的表现（回测）。

    cfg: 可选模型元数据（state_window/factor_expression/factor_mu/factor_sd/
         min_trade_zone），与 train_drl 同口径重建环境（P1-9）：
         - state_window 缺省取 agent.state_window（旧模型 from_dict 已缺省为 1）
         - factor_expression 为空则无因子列（旧模型兼容）
    返回 dict 键（total_ret/final_equity/final_position_ratio/steps）不变。
    """
    cfg = cfg or {}
    # state_window 优先级：cfg 显式给定（模型文件元数据）> agent 自身（旧模型
    # from_dict 已缺省为 1）；曾只读 agent.state_window，cfg 里的键成为死配置
    state_window = max(1, int(cfg.get("state_window", getattr(agent, "state_window", 1) or 1)))
    expr = str(cfg.get("factor_expression", "") or "").strip()
    extra = None
    if expr:
        # 与 train_drl 同口径：同一表达式 + 模型文件里的训练段拟合 mu/sd 标准化
        from factors.mining import FactorExecutor
        vals = FactorExecutor(expr).eval(df).astype(float).to_numpy(float)
        mu, sd = float(cfg.get("factor_mu", 0.0)), float(cfg.get("factor_sd", 1.0))
        vals = (vals - mu) / (sd if sd and sd > 0 else 1.0)
        extra = np.nan_to_num(vals, nan=0.0).reshape(-1, 1)
        # 长度不符由 TradingEnv 抛 ValueError（端点层转 400 友好错误）
    env = TradingEnv(df, start_cash=float(cfg.get("start_cash", 10000.0)),
                     fee_rate=float(cfg.get("fee_rate", 0.001)),
                     vol_penalty=float(cfg.get("vol_penalty", 0.0)), cost_scale=1.0,
                     slippage=float(cfg.get("slippage", 0.0005)),  # 与训练同口径
                     min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                     extra_factors=extra, state_window=state_window,
                     participation_rate=float(cfg.get("participation_rate", 0.0)),
                     funding_rate=float(cfg.get("funding_rate", 0.0)))
    traj = run_episode(env, lambda s: agent.greedy_action(s))
    return {
        "total_ret": round(float(traj["total_ret"]), 6),
        "final_equity": round(traj["final_equity"], 2),
        "final_position_ratio": round(traj["final_position_ratio"], 4),
        "steps": traj["steps"],
    }


# ============ 奖励系数自动搜索 ============

# 奖励系数搜索空间：vol_penalty / reward_dd_penalty / entropy_coef 的候选值
# （Optuna 风格短周期评估：用少量 episode 快速对比各组合，选验证收益最高者）
_REWARD_SEARCH_SPACE = {
    "vol_penalty": [5.0, 10.0, 20.0, 40.0],
    "reward_dd_penalty": [0.0, 0.5, 1.0, 2.0],
    "entropy_coef": [0.01, 0.05, 0.1, 0.2],
}


def search_reward_coefficients(df, cfg: dict,
                               n_trials: int = 6,
                               quick_episodes: int = 3,
                               seed: int = 42,
                               on_trial: Optional[Callable[[dict], None]] = None) -> dict:
    """奖励系数自动搜索：短周期快速评估多组系数，返回最优组合。

    对非平稳金融数据，vol_penalty/奖励塑形系数严重依赖手工调参，且对收益
    影响大。本函数用小批量 episode 快速评估各候选组合（粗筛），选验证段
    收益最高者，作为完整训练的起点。

    Args:
        df: 训练数据
        cfg: 基础训练配置（不含被搜索的系数）
        n_trials: 评估的组合数（默认 6，采样自搜索空间）
        quick_episodes: 每次粗筛的训练轮数（小值，快速）
        seed: 随机种子（保证可复现）
        on_trial: 每组合完成后的回调

    Returns:
        {"best_cfg": {...最优系数...}, "best_val_ret": float,
         "results": [{cfg, val_ret}], "search_space": {...}}
    """
    rng = random.Random(seed)
    base_cfg = dict(cfg)
    base_cfg["seed"] = base_cfg.get("seed", seed)
    base_cfg["episodes"] = quick_episodes
    base_cfg["val_eval_interval"] = 1
    # 禁用可能干扰粗筛的早停/回放（快速评估）
    base_cfg["early_stop_patience"] = 0
    base_cfg["replay_capacity"] = 0

    space = _REWARD_SEARCH_SPACE
    # 生成候选组合：固定 n_trials 个（含搜索空间首值，保证覆盖）
    candidates = []
    # 第一个候选用搜索空间默认中间值
    candidates.append({"vol_penalty": 20.0, "reward_dd_penalty": 0.5, "entropy_coef": 0.05})
    seen = {tuple(sorted(c[0] for c in candidates[0].items()))}
    for _ in range(max(1, n_trials - 1)):
        for _attempt in range(50):
            cand = {
                "vol_penalty": rng.choice(space["vol_penalty"]),
                "reward_dd_penalty": rng.choice(space["reward_dd_penalty"]),
                "entropy_coef": rng.choice(space["entropy_coef"]),
            }
            key = tuple(sorted(cand.items()))
            if key not in seen:
                seen.add(key)
                candidates.append(cand)
                break

    results = []
    best_cfg: Optional[dict] = None
    best_val_ret = -1e9
    for cand in candidates:
        trial_cfg = dict(base_cfg)
        trial_cfg.update(cand)
        try:
            res = train_drl(df, trial_cfg)
            history = res.get("history", [])
            # 用历史中最后一次 val_ret 作为粗筛依据（best_val_ret 可能来自不同轮）
            val_ret = 0.0
            for row in reversed(history):
                if "val_ret" in row and row["val_ret"] is not None:
                    val_ret = float(row["val_ret"])
                    break
            entry = {"cfg": cand, "val_ret": round(val_ret, 6)}
            results.append(entry)
            if val_ret > best_val_ret:
                best_val_ret = val_ret
                best_cfg = dict(cand)
            if on_trial:
                try:
                    on_trial(entry)
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] 系数搜索候选 %s 评估失败: %s", cand, e)

    return {
        "best_cfg": best_cfg or {"vol_penalty": 20.0, "reward_dd_penalty": 0.0, "entropy_coef": 0.05},
        "best_val_ret": round(best_val_ret, 6),
        "results": results,
        "search_space": space,
    }
