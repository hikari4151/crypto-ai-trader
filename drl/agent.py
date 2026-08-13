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
import json
import logging
import math
import os
import random
import time
from typing import Any, Callable, Optional

import numpy as np

from .env import ACTION_BUCKETS, ACTION_NAMES, TradingEnv, run_episode
from .nnet import MLP

log = logging.getLogger(__name__)


class ACAgent:
    """Actor-Critic 智能体（REINFORCE with baseline）。"""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: tuple[int, int] = (32, 32),
                 lr_actor: float = 1e-3, lr_critic: float = 3e-3,
                 gamma: float = 0.99, seed: Optional[int] = None,
                 entropy_coef: float = 0.01,
                 state_window: int = 1) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        # 状态窗口：训练时环境的特征堆叠数（部署时 rl_adaptive 必须用同一窗口重建状态）
        self.state_window = max(1, int(state_window))
        dims = [state_dim, *hidden, n_actions]
        self.actor = MLP(dims, seed=seed, lr=lr_actor)
        self.critic = MLP([state_dim, *hidden, 1], seed=(seed or 0) + 1, lr=lr_critic)
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # 熵正则：鼓励探索，防止过早收敛到单一动作
        self.epsilon = 1.0            # 初始探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999    # 乘法衰减（无 schedule 时兜底）
        self._rng = np.random.default_rng(seed)

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

    def sample_action_with_logp(self, state: np.ndarray) -> tuple[int, float]:
        """纯策略采样并返回 (action, log_prob)（PPO 收集旧策略概率用）。

        关键：不叠加 epsilon 噪声——否则 log_prob 与真实行为策略不一致，
        重要性采样比会失真（PPO 的探索由熵正则承担）。
        单次前向即取概率与采样，避免重复计算（曾每次算两次 predict_proba）。
        """
        proba = self.actor.predict_proba(state.reshape(1, -1))[0]
        action = int(self._rng.choice(self.n_actions, p=proba))
        logp = float(np.log(proba[action] + 1e-12))
        return action, logp

    def collect_episode(self, env) -> dict:
        """跑完一个 episode 并收集轨迹（含旧策略 log_prob，供 PPO 使用）。

        env 需提供 reset/step；episode 长度上限优先取 env.max_steps
        （FactorMiningEnv 等序列决策环境），否则用 n - warmup - 1（TradingEnv）。
        """
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        total_ret = 1.0
        max_steps = getattr(env, "max_steps", None) or (env.n - env.warmup - 1)
        for _ in range(max_steps):
            action, logp = self.sample_action_with_logp(state)
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
            # FactorMiningEnv 无权益概念（final_equity 仅 TradingEnv 统计用）
            "final_equity": env._equity(env._closes[env._t]) if hasattr(env, "_equity") else 0.0,
            "steps": len(rewards),
        }

    def greedy_action(self, state: np.ndarray) -> int:
        """确定性动作（评估/部署用）。"""
        proba = self.actor.predict_proba(state.reshape(1, -1))[0]
        return int(np.argmax(proba))

    # ---------- 训练（真 PPO：GAE + 多轮 mini-batch 裁剪 + 熵正则） ----------
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """GAE 广义优势估计：比蒙特卡洛回报方差更低、偏差可控。

        advantage_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        gae = 0.0
        advantages = np.zeros(T)
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
                    clip_eps: float = 0.2, lam: float = 0.95) -> dict:
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
        advantages, G = self._compute_gae(rewards, V, lam=lam)
        # 归一化优势降低方差
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.arange(n)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        for _ in range(ppo_epochs):
            self._rng.shuffle(idx)
            for start in range(0, n, mini_batch_size):
                b = idx[start:start + mini_batch_size]
                mb = len(b)

                # ---- Actor：单次前向 → softmax → logp → clipped surrogate ----
                acts = self.actor.forward(states[b])
                logits = acts[-1]
                logits = logits - logits.max(axis=-1, keepdims=True)
                e = np.exp(logits)
                proba = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
                act = actions[b]
                logp = np.log(proba[np.arange(mb), act] + 1e-12)
                # 重要性采样比率 r(θ)
                ratio = np.exp(logp - old_log_probs[b])
                adv_b = adv[b]
                surr1 = ratio * adv_b
                surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                # 策略损失 = -min(surr1, surr2)；梯度仅在 surr1 更小时生效
                use_surr1 = surr1 <= surr2
                # softmax 交叉熵梯度 = (proba - onehot) * adv，未裁剪样本才传播
                dlogits = proba.copy()
                dlogits[np.arange(mb), act] -= 1.0
                dout_actor = dlogits * (adv_b * use_surr1)[:, None]
                # 熵正则梯度：鼓励探索防止过早收敛到单一动作
                if self.entropy_coef > 0:
                    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=-1, keepdims=True)
                    d_entropy = -proba * (entropy + np.log(proba + 1e-12))
                    dout_actor = dout_actor + self.entropy_coef * d_entropy
                gW_a, gb_a = self.actor.backward(acts, dout_actor)
                self.actor.apply_grad(gW_a, gb_a)

                # ---- Critic：MSE(V, G) ----
                cacts = self.critic.forward(states[b])
                G_b = G[b, None] if G.ndim == 1 else G[b]
                G_b = G_b.reshape(-1, 1)
                dout_critic = 2.0 * (cacts[-1] - G_b) / max(1, mb)
                gW_c, gb_c = self.critic.backward(cacts, dout_critic)
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
        return {
            "state_dim": self.state_dim, "n_actions": self.n_actions,
            "gamma": self.gamma, "epsilon": self.epsilon,
            "entropy_coef": self.entropy_coef,
            "state_window": self.state_window,
            "actor": self.actor.to_dict(), "critic": self.critic.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ACAgent":
        agent = cls(data["state_dim"], data["n_actions"], gamma=data.get("gamma", 0.99),
                    entropy_coef=float(data.get("entropy_coef", 0.01)),
                    state_window=int(data.get("state_window", 1)))
        agent.actor = MLP.from_dict(data["actor"], lr=agent.actor.lr)
        agent.critic = MLP.from_dict(data["critic"], lr=agent.critic.lr)
        agent.epsilon = float(data.get("epsilon", 0.05))
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

def train_drl(df, cfg: dict, on_progress: Optional[Callable[[dict], None]] = None) -> dict:
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
    on_progress(info): 每轮回调 {episode, total_ret, final_equity, best_ret, epsilon, ...}
    返回 {agent, history, best_ret, cfg}
    """
    t0 = time.time()
    # PPO 默认超参（真 PPO：批量收集 + 多轮 mini-batch，比单轨迹更新更稳）
    episodes = int(cfg.get("episodes", 300))
    hidden = tuple(cfg.get("hidden", [64, 64]))
    # 每轮收集的轨迹数（批大小，A2C/PPO 标配：批量平均降方差）
    n_episodes = max(1, int(cfg.get("n_episodes", 4)))
    # PPO 多轮更新轮数（SB3 默认 10；小数据建议 3-5）
    ppo_epochs = max(1, int(cfg.get("ppo_epochs", 4)))
    mini_batch_size = max(8, int(cfg.get("mini_batch_size", 128)))
    # 波动率惩罚需足够大以压过高波动段的收益噪声，才能学到"波动上升→降仓"
    # （收益为 bp 标度，惩罚系数默认 20：低波动趋势净正、高波动满仓净负）
    vol_penalty = float(cfg.get("vol_penalty", 20.0))
    cost_scale = float(cfg.get("cost_scale", 1.0))
    seed = int(cfg.get("seed", 42))

    # ============ 训练/验证/OOS 三区切分（防过拟合核心） ============
    # - 训练段 train_ratio：模型在其中反复试错学习
    # - 验证段 val_ratio：训练期间周期性评估，选最优模型（模型从未见过这段）
    # - OOS 段 oos_ratio：训练完全结束后的独立样本外评估（换一段未见过行情）
    # 三者不重叠，杜绝"训练=验证=评估同一段数据"的前视偏差。
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    n_total = len(df)
    n_train = max(300, int(n_total * train_ratio))
    n_val = max(100, int(n_total * val_ratio))
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    oos_df = df.iloc[n_train + n_val:]
    # 数据不足时退化为单段（不切分），保证小数据集仍可训练
    if len(oos_df) < 80 or len(val_df) < 80:
        train_df = val_df = df
        oos_df = df.iloc[:0]
        log.warning("[drl] 数据量不足(仅%d根)，跳过训练/验证/OOS切分", n_total)

    # 外部因子信号列（可选）：训练/验证/OOS 用同一表达式在各自数据段上计算，
    # 与部署（rl_adaptive 滚动缓冲）口径一致。因子值标准化在训练段拟合。
    extra_factors_all: Optional[np.ndarray] = None
    extra_factors_val: Optional[np.ndarray] = None
    extra_factors_oos: Optional[np.ndarray] = None
    factor_expr = str(cfg.get("factor_expression", "") or "").strip()
    # 标准化统计量（训练段拟合）写入模型文件，部署端用同一口径标准化——
    # 曾只存表达式不存 mu/sd，部署喂原始值，状态分布与训练完全不同
    factor_mu: float = 0.0
    factor_sd: float = 1.0
    if factor_expr:
        from factors.mining import FactorExecutor
        try:
            fx = FactorExecutor(factor_expr)
            vals = fx.eval(df).astype(float).to_numpy(float)
            vals = np.nan_to_num(vals, nan=0.0)
            # 标准化（训练段拟合均值/方差，防统计泄漏）
            mu = np.nanmean(vals[:n_train]) if n_train > 0 else 0.0
            sd = np.nanstd(vals[:n_train]) + 1e-8
            factor_mu, factor_sd = float(mu), float(sd)
            vals = (vals - mu) / sd
            extra_factors_all = vals.reshape(-1, 1)
            extra_factors_val = vals[n_train:n_train + n_val].reshape(-1, 1) if len(val_df) == n_val else None
            extra_factors_oos = vals[n_train + n_val:].reshape(-1, 1) if len(oos_df) > 0 else None
            log.info("[drl] 已注入因子信号列: %s（标准化后加入状态, mu=%.4f sd=%.4f）", factor_expr[:60], factor_mu, factor_sd)
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] 因子表达式计算失败，忽略因子列: %s", e)

    # GPU 后端：特征预计算走 cupy（可用时），训练循环仍 CPU（逐K线有状态依赖）
    from .env import gpu_available
    backend = "cupy" if (cfg.get("use_gpu") and gpu_available()) else "numpy"
    # 状态窗口（时序记忆）：默认 1=单点特征（兼容旧模型）；>1 时堆叠最近 N 根
    state_window = max(1, int(cfg.get("state_window", 1)))
    env = TradingEnv(train_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                     fee_rate=float(cfg.get("fee_rate", 0.001)),
                     vol_penalty=vol_penalty, cost_scale=cost_scale,
                     min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                     backend=backend,
                     extra_factors=extra_factors_all[:len(train_df)] if extra_factors_all is not None else None,
                     state_window=state_window)
    agent = ACAgent(env.state_dim, len(ACTION_BUCKETS), hidden=hidden,
                    lr_actor=float(cfg.get("lr_actor", 2e-3)),
                    lr_critic=float(cfg.get("lr_critic", 5e-3)),
                    gamma=float(cfg.get("gamma", 0.99)), seed=seed,
                    entropy_coef=float(cfg.get("entropy_coef", 0.05)),
                    state_window=state_window)
    # 续训支持：从已训练模型加载权重作为起点（保留已学知识继续增强）
    base_model = cfg.get("base_model")
    if base_model:
        try:
            from .agent import ACAgent as _ACA
            base = _ACA.load(base_model)
            if base.state_dim == agent.state_dim and base.n_actions == agent.n_actions:
                agent.actor = base.actor
                agent.critic = base.critic
                log.info("[drl] 已加载基础模型 %s 作为续训起点", base_model)
                # 续训时保留少量探索，避免刚接手就过度确定
                agent.epsilon = 0.4
            else:
                log.warning("[drl] 基础模型维度不匹配(%s)，忽略，从零训练",
                            getattr(base, "state_dim", "?"))
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] 基础模型加载失败(%s)，从零训练: %s", base_model, e)
    agent.set_epsilon_schedule(episodes)

    history: list[dict] = []
    best_train_ret = -1e9
    best_val_ret = -1e9
    best_agent: Optional[ACAgent] = None
    val_eval_interval = max(1, int(cfg.get("val_eval_interval", 10)))
    # 早停保护：验证收益连续无改善次数达到阈值则提前停止（借鉴 SB3 EvalCallback）
    no_improve_streak = 0
    early_stop_patience = int(cfg.get("early_stop_patience", 0))  # 0=禁用，>0 启用
    # 强制自动训练：开启后即使触发早停也不中断，重置早停计数继续训练，
    # 直到达到用户设置的轮数（进度条不重置，连续显示总进度）
    force_train = bool(cfg.get("force_train", False))
    force_rounds = 0  # 强制模式下早停被越过的次数

    # 验证用无探索环境（确定性策略，评估真实策略质量）
    # 关键：验证段用独立的 val_df（与训练段不重叠），杜绝"训练=验证同一段数据"的前视偏差
    val_env = TradingEnv(val_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                         fee_rate=float(cfg.get("fee_rate", 0.001)),
                         vol_penalty=0.0, cost_scale=1.0,
                         min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                         extra_factors=extra_factors_val,
                         state_window=state_window)

    # 学习率退火（PPO 标配技巧）：随训练进度从初始值线性降到 lr_final_ratio 倍
    # 注：曾尝试加入 warmup，A/B 多 seed 验证无正面提升（均值 2.85 vs 3.14），已回退。
    lr_actor_init = agent.actor.lr
    lr_critic_init = agent.critic.lr
    lr_final_ratio = float(cfg.get("lr_final_ratio", 0.2))
    # 注：熵正则退火 A/B 验证无正面提升（均值 2.16 vs 3.39），保持固定熵。

    for ep in range(1, episodes + 1):
        agent.set_episode_epsilon(ep - 1)
        # 学习率退火：训练后期降低更新幅度，收敛更稳
        frac = ep / episodes
        lr_scale = lr_final_ratio + (1 - lr_final_ratio) * (1 - frac)
        agent.actor.lr = lr_actor_init * lr_scale
        agent.critic.lr = lr_critic_init * lr_scale

        # 批量收集轨迹（n_episodes 条）→ 拼接 → 真 PPO 更新（多轮 mini-batch）
        trajs = [agent.collect_episode(env) for _ in range(n_episodes)]
        states = np.concatenate([t["states"] for t in trajs])
        actions = np.concatenate([t["actions"] for t in trajs])
        rewards = np.concatenate([t["rewards"] for t in trajs])
        old_log_probs = np.concatenate([t["old_log_probs"] for t in trajs])
        stats = agent.train_batch(states, actions, rewards, old_log_probs,
                                  ppo_epochs=ppo_epochs,
                                  mini_batch_size=min(mini_batch_size, len(states)))
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

        # 周期性用确定性策略验证，按验证收益选最优模型（避免训练期随机轨迹误导）
        val_ret = None
        if ep % val_eval_interval == 0 or ep == episodes:
            try:
                vtraj = run_episode(val_env, lambda s: agent.greedy_action(s))
                val_ret = float(vtraj["total_ret"])
                row["val_ret"] = round(val_ret, 6)
                if val_ret > best_val_ret:
                    best_val_ret = val_ret
                    best_agent = ACAgent.from_dict(agent.to_dict())
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
    final_agent = best_agent or agent
    final_agent.epsilon = final_agent.epsilon_min

    # ============ OOS 独立评估（防过拟合核心） ============
    # 训练结束后用【完全未见过】的 oos_df 评估最优模型，与训练段对比：
    # 若训练段收益高但 OOS 大幅缩水 → 过拟合信号。OOS 段在训练/验证中从未被使用。
    oos_report = {"enabled": False}
    if len(oos_df) >= 80:
        try:
            oos_env = TradingEnv(oos_df, start_cash=float(cfg.get("start_cash", 10000.0)),
                                 fee_rate=float(cfg.get("fee_rate", 0.001)),
                                 vol_penalty=0.0, cost_scale=1.0,
                                 min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                                 extra_factors=extra_factors_oos,
                                 state_window=state_window)
            oos_traj = run_episode(oos_env, lambda s: final_agent.greedy_action(s))
            oos_ret = float(oos_traj["total_ret"])
            oos_equities = [info.get("equity") for info in oos_traj.get("infos", [])
                            if info.get("equity") is not None]
            # 训练段代表收益（用验证段 best 或训练段末收益）
            train_ref = best_train_ret if best_train_ret > -1e8 else float(total_ret)
            # 过拟合判断：OOS 收益相对训练段收益的衰减
            if abs(train_ref) > 1e-6:
                decay = 1.0 - oos_ret / train_ref
            else:
                decay = 0.0
            oos_report = {
                "enabled": True,
                "oos_ret": round(oos_ret, 6),
                "train_ret": round(train_ref, 6),
                "decay": round(decay, 4),
                "overfit_likely": bool(decay > 0.5),  # OOS 收益衰减过半 → 疑似过拟合
                "oos_equity_final": round(oos_traj["final_equity"], 2),
                "oos_position_ratio": round(oos_traj["final_position_ratio"], 4),
                "oos_sharpe": round(_equity_sharpe(oos_equities), 4) if len(oos_equities) > 2 else 0.0,
                "oos_max_drawdown": round(_equity_drawdown(oos_equities), 4) if len(oos_equities) > 2 else 0.0,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[drl] OOS 评估失败: %s", e)
            oos_report = {"enabled": False}

    return {
        "agent": final_agent,
        "history": history,
        "best_ret": round(max(best_train_ret, best_val_ret), 6),
        "best_val_ret": round(best_val_ret, 6),
        "final_ret": round(total_ret, 6),
        "episodes": episodes,
        "state_dim": env.state_dim,
        "n_actions": len(ACTION_BUCKETS),
        "action_buckets": ACTION_BUCKETS.tolist(),
        "action_names": ACTION_NAMES,
        "algo": "ppo",
        "ppo_cfg": {"n_episodes": n_episodes, "ppo_epochs": ppo_epochs,
                    "mini_batch_size": mini_batch_size, "clip_eps": 0.2,
                    "factor_expression": factor_expr},
        "factor_expression": factor_expr,
        "factor_mu": factor_mu,
        "factor_sd": factor_sd,
        "elapsed_sec": round(time.time() - t0, 2),
        "backend": backend,
        "oos_report": oos_report,
        "cfg": cfg,
    }


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


def evaluate_agent(agent: ACAgent, df, cfg: dict = {}) -> dict:
    """用确定性策略评估智能体在数据上的表现（回测）。"""
    env = TradingEnv(df, start_cash=float(cfg.get("start_cash", 10000.0)),
                     fee_rate=float(cfg.get("fee_rate", 0.001)),
                     vol_penalty=0.0, cost_scale=1.0)
    traj = run_episode(env, lambda s: agent.greedy_action(s))
    return {
        "total_ret": round(float(traj["total_ret"]), 6),
        "final_equity": round(traj["final_equity"], 2),
        "final_position_ratio": round(traj["final_position_ratio"], 4),
        "steps": traj["steps"],
    }
