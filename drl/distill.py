"""离线行为蒸馏：把大网络（装不进 Pine）压成可导出的小网络。

背景（2026-10）：Pine 体积门（64KB，实测仅约 [16,16] 网络可装下）与 DRL
网络容量存在长期矛盾——大网络学习能力强（也是后续时序网络升级的必需），
但权重整段内联进 Pine 脚本必然超限。本模块实现「训练大网络 + 蒸馏小网络
导出」：teacher 保留在软件内实盘（rl_adaptive），student（[16,16]）用于
TradingView Pine 自动交易代码，两者都从同一份权重出发、行为尽可能一致。

方法：离线 soft-target 行为克隆。
- 在训练数据上跑 teacher 的确定性 rollout，收集一组状态（覆盖训练段分布）
- teacher 对每个状态输出 softmax 动作概率（软标签，保留不确定性，比硬标签
  argmax 保留更多信息——低置信区间也告诉 student「这个状态下别乱动」）
- student 用交叉熵拟合这些概率（梯度下降，纯 numpy，复用 MLP 前向/反向）
- 迭代多轮至 KL 散度收敛或达步数上限

设计要点：
- 蒸馏只改「导出用的小网络」，不影响 teacher 的 val/OOS/回退/部署语义
- student 维度 = teacher 输入维度 + 指定 hidden（默认 [16,16] 可装进 Pine）
- 蒸馏后小网络的 Pine 代码标注「蒸馏自大网络」，供用户对照
- 无蒸馏需求（teacher 本就可导出）时完全不触发，向后兼容
"""
import logging
import time
from typing import Optional

import numpy as np

from .agent import ACAgent
from .env import ACTION_BUCKETS, TradingEnv, run_episode

log = logging.getLogger(__name__)

# 蒸馏采样：每个 rollout 收集的状态数（覆盖训练段的时间分布）
_DEFAULT_SAMPLES_PER_ROLLOUT = 400
# 蒸馏迭代默认参数
_DEFAULT_EPOCHS = 400
_DEFAULT_BATCH = 256
_DEFAULT_LR = 5e-3


def _sample_states(teacher: ACAgent, df, cfg: dict, n_rollouts: int = 6,
                   samples_per_rollout: int = _DEFAULT_SAMPLES_PER_ROLLOUT,
                   seed: int = 42) -> np.ndarray:
    """在训练数据上跑 teacher 确定性 rollout，采样一批状态。

    状态直接取 env 的原始状态向量（含价量特征 + 账户状态 + 可选因子），
    与训练/部署端同一口径（部署时 rl_adaptive 用同一特征序列喂网络）。

    返回 (n, state_dim) 的 float 数组。
    """
    # 用与训练同口径的 env 收集状态（vol_penalty 不影响状态，只影响奖励）
    # 口径走 policy_env_kwargs（reward_mode/decision_interval/short_enabled 等）：
    # 蒸馏也要在"训练时那套环境"里采样状态，否则学生网络见到的状态分布与老师不同。
    from .env import policy_env_kwargs as _policy_env_kwargs
    env = TradingEnv(
        df, start_cash=float(cfg.get("start_cash", 10000.0)),
        fee_rate=float(cfg.get("fee_rate", 0.001)),
        vol_penalty=float(cfg.get("vol_penalty", 20.0)),
        cost_scale=float(cfg.get("cost_scale", 1.0)),
        slippage=float(cfg.get("slippage", 0.0005)),
        min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
        state_window=int(cfg.get("state_window", 1)),
        reward_dd_penalty=float(cfg.get("reward_dd_penalty", 0.0)),
        reward_losing_penalty=float(cfg.get("reward_losing_penalty", 0.0)),
        reward_trend_align=float(cfg.get("reward_trend_align", 0.0)),
        tail_risk_penalty=float(cfg.get("tail_risk_penalty", 0.0)),
        participation_rate=float(cfg.get("participation_rate", 0.0)),
        funding_rate=float(cfg.get("funding_rate", 0.0)),
        **_policy_env_kwargs(cfg),
    )
    states: list[np.ndarray] = []
    # 2026-10 审计修复（LOW）：多次 rollout 用**随机起点**制造状态多样性——
    # 曾每次 env.reset() 恒回 warmup，n_rollouts 次 rollout 是同一轨迹的重复
    #（学生网络只见过单条轨迹的状态分布，蒸馏覆盖度被高估）。
    _rng = np.random.default_rng(int(cfg.get("seed", 0) or 0) + 3)
    _span = max(0, env.n - env.warmup - 200)
    for _ in range(n_rollouts):
        # 用 teacher 的确定性策略 rollout（随机起点 → 覆盖不同行情窗口）
        env.reset(start_offset=int(_rng.integers(0, _span + 1)) if _span > 0 else 0)
        traj = run_episode(env, lambda s: teacher.greedy_action(s))
        arr = traj.get("states")
        if arr is None or len(arr) == 0:
            continue
        if len(arr) > samples_per_rollout:
            # 均匀抽样：覆盖整条轨迹的时间分布，避免只取尾部
            idx: np.ndarray = np.linspace(0, len(arr) - 1,
                                          samples_per_rollout).astype(int)
            states.append(np.asarray(arr)[idx])
        else:
            states.append(np.asarray(arr))
    if not states:
        raise ValueError("[distill] 未能收集到任何状态，无法蒸馏")
    out = np.concatenate(states, axis=0)
    return np.ascontiguousarray(out, dtype=float)


def _softmax_proba(teacher: ACAgent, states: np.ndarray) -> np.ndarray:
    """teacher 对一批状态的 softmax 动作概率（软标签）。

    批量前向（teacher.actor.predict_proba 支持 (batch, in)），一次算完。
    """
    proba = teacher.actor.predict_proba(states, cache_for_backward=False)
    return np.asarray(proba, dtype=float)


def _distill_student_actor(teacher: ACAgent, states: np.ndarray,
                           soft_targets: np.ndarray,
                           student_hidden: list[int],
                           epochs: int = _DEFAULT_EPOCHS,
                           batch_size: int = _DEFAULT_BATCH,
                           lr: float = _DEFAULT_LR,
                           seed: int = 42) -> "ACAgent":
    """用软标签训练 student 网络（交叉熵 = KL + H(teacher) 单调等价）。

    student 与 teacher 同维度接口（state_dim/n_actions/state_window），
    但 hidden 更小。梯度下降复用 MLP 的 forward/backward/apply_grad。

    返回训练好的 student ACAgent（仅 actor 有意义，critic 冷启动占位）。
    """
    from .nnet import MLP, _log_softmax

    state_dim = int(states.shape[1])
    n_actions = int(soft_targets.shape[1])
    student = ACAgent(state_dim, n_actions, hidden=tuple(student_hidden),
                      lr_actor=lr, lr_critic=lr,
                      seed=seed,
                      state_window=int(getattr(teacher, "state_window", 1) or 1))
    student.actor = MLP([state_dim, *[int(h) for h in student_hidden], n_actions],
                        seed=seed + 1, lr=lr)
    student.epsilon = 0.0  # 蒸馏是确定性拟合，无探索

    n = len(states)
    rng = np.random.default_rng(seed + 2)
    idx = np.arange(n)
    t0 = time.time()
    last_kl: Optional[float] = None
    for ep in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            mb = len(b)
            acts = student.actor.forward(states[b])
            logp = _log_softmax(acts[-1])
            target = soft_targets[b]
            # 交叉熵梯度：dLoss/dlogits = (softmax(logits) - target)
            dout_actor = (np.exp(logp) - target) / mb
            gW, gb = student.actor.backward(acts, dout_actor)
            student.actor.apply_grad(gW, gb)
        # 收敛监控：全量 KL 散度（每 50 轮算一次，避免 O(n·params) 拖慢）
        if ep % 50 == 0 or ep == epochs - 1:
            proba = student.actor.predict_proba(states, cache_for_backward=False)
            proba = np.clip(proba, 1e-12, 1.0)
            tgt = np.clip(soft_targets, 1e-12, 1.0)
            kl = float(np.mean(np.sum(tgt * (np.log(tgt) - np.log(proba)), axis=-1)))
            last_kl = kl
            if ep % 100 == 0:
                log.info("[distill] epoch %d KL=%.5f", ep, kl)
    if last_kl is not None:
        log.info("[distill] 蒸馏完成: %d 状态, %d epoch, 终 KL=%.5f, 耗时 %.1fs",
                 n, epochs, last_kl, time.time() - t0)
    return student


def distill_agent(teacher: ACAgent, df, cfg: dict,
                  student_hidden: Optional[list[int]] = None,
                  n_rollouts: int = 6,
                  samples_per_rollout: int = _DEFAULT_SAMPLES_PER_ROLLOUT,
                  epochs: int = _DEFAULT_EPOCHS,
                  batch_size: int = _DEFAULT_BATCH,
                  lr: float = _DEFAULT_LR,
                  seed: int = 42,
                  log_info: bool = True) -> "ACAgent":
    """完整蒸馏：采样状态 → soft 标签 → 训练 student。

    参数:
        teacher: 训练好的大网络（装不进 Pine）
        df: 训练数据（训练段，与 train_drl 的 train_df 同源）
        cfg: 训练配置（与 train_drl 同口径重建 env）
        student_hidden: student 隐藏层（默认 [16,16]——实测可装进 Pine）
        n_rollouts/samples_per_rollout: 状态采样规模
        epochs/batch_size/lr: 蒸馏训练超参
        seed: 可复现

    返回: student ACAgent（state_dim/n_actions/state_window 与 teacher 相同）。
    """
    hidden = [int(h) for h in (student_hidden or [16, 16])]
    if not hidden or any(h < 4 for h in hidden):
        raise ValueError(f"[distill] student_hidden 非法: {hidden}（每层须 ≥4）")
    t0 = time.time()
    states = _sample_states(teacher, df, cfg, n_rollouts=n_rollouts,
                            samples_per_rollout=samples_per_rollout, seed=seed)
    soft = _softmax_proba(teacher, states)
    if log_info:
        log.info("[distill] 采样 %d 状态（%d rollout），开始蒸馏 → hidden=%s",
                 len(states), n_rollouts, hidden)
    student = _distill_student_actor(
        teacher, states, soft, hidden, epochs=epochs,
        batch_size=batch_size, lr=lr, seed=seed)
    # 标记：这是蒸馏版（供 Pine 导出头部注明来源）
    student.is_distilled = True
    # teacher 结构描述：兼容 MLP（actor.W）与 GRU（gru_hidden + head，P1 时序骨干）
    _t_actor = teacher.actor
    if hasattr(_t_actor, "W") and _t_actor.W:
        _t_hidden = [int(w.shape[1]) for w in _t_actor.W[:-1]]
        _t_desc = {"type": "mlp", "hidden": _t_hidden}
    else:
        _hh = [int(h) for h in getattr(_t_actor, "head_hidden", ())]
        _t_hidden = [int(getattr(_t_actor, "gru_hidden", 0)), *_hh] if _hh else \
            [int(getattr(_t_actor, "gru_hidden", 0))]
        _t_desc = {"type": "gru", "gru_hidden": int(getattr(_t_actor, "gru_hidden", 0))}
    student.distill_meta = {
        "teacher_state_dim": teacher.state_dim,
        "teacher_hidden": _t_hidden,
        "teacher_arch": _t_desc,
        "student_hidden": hidden,
        "n_samples": int(len(states)),
        "epochs": int(epochs),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return student


def estimate_student_exportable(teacher: ACAgent,
                                student_hidden: Optional[list[int]] = None) -> bool:
    """预判：给定 teacher 的维度，student 网络能否装进 Pine（无需真跑蒸馏）。

    用于 train_drl 在训练结束后决定「是否需要蒸馏」——teacher 本身可导出则
    直接用 teacher，否则蒸馏到 student_hidden 再试。
    """
    from .pine_export import estimate_pine_bytes, PINE_MAX_BYTES
    hidden = [int(h) for h in (student_hidden or [16, 16])]
    in_dim = teacher.state_dim
    return estimate_pine_bytes(in_dim, hidden, len(ACTION_BUCKETS)) <= PINE_MAX_BYTES
