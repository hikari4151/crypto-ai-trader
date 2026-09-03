"""RL 因子组合挖掘训练器（AlphaForge 范式的 RL 版）。

用 PPO 训练一个"因子选择智能体"：在因子池上学习"选哪些因子 + 怎么加权"，
产出组合因子（接入现有因子体系），而不是直接产出交易策略。

训练流程（与 train_drl 同构的防过拟合设计）：
- 因子表现特征取自训练段（决策信息），组合 fitness 在验证段评估（奖励）
- 训练/验证/OOS 三段时间切分，OOS 只做最终报告
- 每轮收集 n_episodes 条轨迹（不同决策时点）→ 真 PPO 多轮 mini-batch 更新
- 训练完成后 greedy 决策 → 选中因子组合 → IC 加权合成 → OOS 段安检门

输出：
- selected_factors: 选中的因子（按加入顺序）
- weights: 各因子权重（决策时点的已实现滚动 IC 方向，负 IC 反向暴露）
- composite: 组合因子序列（全量，供回测/实盘消费）
- report: OOS 段安检报告（valid/rank_ic/icir/turnover/fitness）
"""
import logging
import random
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .agent import ACAgent
from .factor_env import FactorMiningEnv
from .env import ACTION_BUCKETS  # noqa: F401（保持与 drl 包导出习惯一致）
from factors.analysis import factor_quality_gate
from factors.mining import composite_factor

log = logging.getLogger(__name__)


def train_factor_miner(df: pd.DataFrame, mat: Optional[pd.DataFrame] = None,
                       cfg: Optional[dict] = None,
                       on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """PPO 训练因子组合挖掘智能体。

    df: 标准 OHLCV（因子矩阵从其计算；也可直接传 mat 复用已算因子）
    cfg:
        episodes: 训练轮数（默认 60）
        n_episodes: 每轮收集轨迹数（默认 8，因子环境轨迹短，可多收集）
        ppo_epochs / mini_batch_size: PPO 更新参数
        hidden: 网络隐藏层
        ic_window: 滚动 IC 窗口（默认 120）
        max_steps: 最多选几个因子（默认 6）
        h: 预测周期
        train_ratio / val_ratio: 时间切分
        corr_threshold: 冗余惩罚相关阈值
        seed
    on_progress(info): 每轮回调
    返回 {agent, history, selected_factors, weights, composite, report, meta}
    """
    from factors.engine import compute_factor_matrix

    t0 = time.time()
    cfg = cfg or {}
    h = max(1, int(cfg.get("h", 1)))
    ic_window = max(30, int(cfg.get("ic_window", 120)))
    max_steps = max(1, min(int(cfg.get("max_steps", 6)), 12))
    episodes = int(cfg.get("episodes", 60))
    n_episodes = max(1, int(cfg.get("n_episodes", 8)))
    ppo_epochs = max(1, int(cfg.get("ppo_epochs", 4)))
    mini_batch_size = max(8, int(cfg.get("mini_batch_size", 64)))
    hidden = tuple(cfg.get("hidden", [64, 64]))
    seed_cfg = cfg.get("seed", 42)
    seed = int(seed_cfg) if seed_cfg is not None else random.randint(0, 99999)
    corr_threshold = float(cfg.get("corr_threshold", 0.85))
    train_ratio = float(cfg.get("train_ratio", 0.6))
    val_ratio = float(cfg.get("val_ratio", 0.2))

    # 因子矩阵（若未传入则从 OHLCV 计算全部内置因子）
    if mat is None:
        mat = compute_factor_matrix(df)
    close = df["close"]

    # 时间切分（与 TradingEnv 的 train/val/OOS 同构）
    n = len(df)
    n_train = max(200, int(n * train_ratio))
    n_val = max(n_train + 100, int(n * (train_ratio + val_ratio)))
    n_val = min(n_val, n - 30)
    oos_df = df.iloc[n_val:]
    oos_mat = mat.iloc[n_val:] if len(oos_df) >= 60 else None

    env = FactorMiningEnv(mat, close, h=h, ic_window=ic_window,
                          max_steps=max_steps,
                          train_ratio=train_ratio, val_ratio=val_ratio,
                          corr_threshold=corr_threshold, seed=seed)
    agent = ACAgent(env.state_dim, env.n_actions, hidden=hidden,
                    lr_actor=float(cfg.get("lr_actor", 3e-3)),
                    lr_critic=float(cfg.get("lr_critic", 6e-3)),
                    gamma=float(cfg.get("gamma", 0.95)), seed=seed,
                    entropy_coef=float(cfg.get("entropy_coef", 0.05)))

    # 训练循环（PPO 批量收集 + 多轮 mini-batch）
    history: list[dict] = []
    best_fitness = -1e9
    best_agent: Optional[ACAgent] = None
    # 并行收集：预计算一次 IC/ICIR 矩阵（避免每个 worker 重复 84ms 的 _rolling_ic）
    from factors.analysis import _rolling_ic
    _ics = pd.DataFrame({c: _rolling_ic(mat[c], close, h=h, method="rank", window=ic_window)
                          for c in mat.columns}, index=mat.index).shift(h).ffill()
    _icir = _ics.rolling(ic_window * 5, min_periods=ic_window * 2).mean() / (
        _ics.rolling(ic_window * 5, min_periods=ic_window * 2).std() + 1e-12)
    from concurrent.futures import ThreadPoolExecutor

    def _collect_one(args: tuple) -> dict:
        ep_i, worker_i = args
        local_env = FactorMiningEnv(mat, close, h=h, ic_window=ic_window,
                                    max_steps=max_steps,
                                    train_ratio=train_ratio, val_ratio=val_ratio,
                                    corr_threshold=corr_threshold,
                                    seed=seed + ep_i * 1000 + worker_i,
                                    precomputed_ics=_ics, precomputed_icir=_icir)
        # D1：独立 rng 采样（绝不共享 agent._rng，保证多线程可复现）
        local_rng = np.random.default_rng(seed + ep_i * 2000 + worker_i)
        return agent.collect_episode(local_env, rng=local_rng)

    with ThreadPoolExecutor(max_workers=min(n_episodes, 4)) as pool:
        for ep in range(1, episodes + 1):
            trajs = list(pool.map(_collect_one, [(ep, i) for i in range(n_episodes)]))
            states = np.concatenate([t["states"] for t in trajs])
            actions = np.concatenate([t["actions"] for t in trajs])
            rewards = np.concatenate([t["rewards"] for t in trajs])
            old_log_probs = np.concatenate([t["old_log_probs"] for t in trajs])
            stats = agent.train_batch(states, actions, rewards, old_log_probs,
                                      ppo_epochs=ppo_epochs,
                                      mini_batch_size=min(mini_batch_size, len(states)),
                                      # P2-8：GAE 按 episode 分段（段末 next_val=0、gae 归零），
                                      # 消除拼接轨迹边界处的优势串扰
                                      segment_lengths=[t["steps"] for t in trajs],
                                      actor_grad_clip=float(cfg.get("actor_grad_clip", 1.0)))
            # 本批平均 fitness（从 info 汇总：collect_episode 不返回 info，
            # 用验证段确定性评估当前策略的选择质量）
            fit = _greedy_fitness(agent, env)
            if fit > best_fitness:
                best_fitness = fit
                best_agent = ACAgent.from_dict(agent.to_dict())
            row = {
                "episode": ep,
                "fitness": round(fit, 5),
                "best_fitness": round(max(best_fitness, 0.0), 5),
                "policy_loss": round(stats["policy_loss"], 6),
                "value_loss": round(stats["value_loss"], 6),
                "steps": int(np.mean([t["steps"] for t in trajs])),
            }
            history.append(row)
        if on_progress:
            try:
                on_progress({
                    "episode": ep, "episodes": episodes,
                    "fitness": row["fitness"],
                    "best_fitness": row["best_fitness"],
                    "policy_loss": stats["policy_loss"],
                    "elapsed_sec": round(time.time() - t0, 1),
                })
            except Exception as e:  # noqa: BLE001
                log.warning("[factor_miner] 进度回调异常: %s", e)

    final_agent = best_agent or agent

    # ---- 最终决策：greedy 跑一个完整 episode ----
    decision = _greedy_select(final_agent, env)
    selected = decision["selected"]
    if not selected:
        raise ValueError("RL 未选出任何因子（奖励未学出来），请调整参数或检查数据")

    # 组合权重：决策时点已实现的滚动 IC 方向（负 IC 反向暴露）
    weights: dict[str, float] = {}
    for i in selected:
        ic = env._ics.iloc[env._t, env.cols.index(i)]
        ic = 0.0 if ic != ic else ic
        weights[i] = ic if abs(ic) > 1e-6 else 0.01
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")

    # ---- OOS 段安检（agent 从未见过，只报告） ----
    report: dict[str, Any] = {"enabled": False}
    if oos_mat is not None:
        gate = factor_quality_gate(combo.iloc[n_val:], close.iloc[n_val:], h=h)
        report = {
            "enabled": True,
            "valid": gate["valid"],
            "reason": gate["reason"],
            "rank_ic": gate["rank_ic"], "icir": gate["icir"],
            "turnover": gate["turnover"], "fitness": gate["fitness"],
            "samples": gate["samples"],
        }

    meta = {
        "algo": "ppo",
        "h": h, "ic_window": ic_window, "max_steps": max_steps,
        "n_factors": env.n_factors,
        "episodes": episodes, "n_episodes": n_episodes,
        "ppo_epochs": ppo_epochs, "mini_batch_size": mini_batch_size,
        "corr_threshold": corr_threshold,
        "train_bars": n_train, "val_bars": n_val - n_train,
        "oos_bars": n - n_val,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return {
        "agent": final_agent,
        "history": history,
        "selected_factors": selected,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "composite": combo,
        "report": report,
        "meta": meta,
    }


def _greedy_fitness(agent: ACAgent, env: FactorMiningEnv) -> float:
    """确定性策略跑一个 episode 的最终组合 fitness（验证段评估）。"""
    state = env.reset()
    for _ in range(env.max_steps * 3 + 1):
        action = agent.greedy_action(state)
        state, _r, done, info = env.step(action)
        if done:
            return float(info.get("final_fitness", 0.0))
    return 0.0


def _greedy_select(agent: ACAgent, env: FactorMiningEnv) -> dict:
    """确定性策略完整决策：返回选中因子列表（按加入顺序）。"""
    state = env.reset()
    for _ in range(env.max_steps * 3 + 1):
        action = agent.greedy_action(state)
        state, _r, done, info = env.step(action)
        if done:
            return info
    return {"selected": []}
