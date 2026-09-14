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
import copy
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
    episodes = max(1, int(cfg.get("episodes", 60)))  # run24：下界钳制（0/负 → 空 history 崩溃）
    n_episodes = max(1, int(cfg.get("n_episodes", 8)))
    ppo_epochs = max(1, int(cfg.get("ppo_epochs", 4)))
    mini_batch_size = max(8, int(cfg.get("mini_batch_size", 64)))
    hidden = tuple(cfg.get("hidden", [64, 64]))
    seed_cfg = cfg.get("seed", 42)
    seed = int(seed_cfg) if seed_cfg is not None else random.randint(0, 99999)
    corr_threshold = float(cfg.get("corr_threshold", 0.85))
    train_ratio = float(cfg.get("train_ratio", 0.6))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    # E-EXP：动作掩码（强制每步选新因子，组合大小恒为 max_steps）。
    # 实验证据（.optim/exp_training/sweep_factor.py + verify_factor_win.py，真实
    # BTC/ETH 1h、3 seed）：无掩码时近均匀策略贪心恒选 argmax=首动作，重复选择
    # 浪费步数、最终只选出 1~2 个因子；掩码后 OOS 秩IC 0.0088→0.0356、
    # 续训 OOS 安检门通过率 0/3→3/3，ETH 同样更优、SOL 持平。默认开启，
    # 显式 cfg action_masking=False 可关闭（旧行为）。
    action_masking = bool(cfg.get("action_masking", True))
    # E-EXP：选模时贪心 fitness 的评估点数（>1 平均多个随机决策时点，降低选模噪声）
    greedy_evals = max(1, int(cfg.get("greedy_evals", 1)))
    # E-EXP：学习率退火（与 train_drl 同口径）；默认关
    lr_anneal = bool(cfg.get("lr_anneal", False))
    lr_final_ratio = float(cfg.get("lr_final_ratio", 0.2))

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

    # 并行收集：预计算一次 IC/ICIR 矩阵（避免每个 worker 重复 84ms 的 _rolling_ic）。
    # P2：提前到共享 env 构造之前——共享 env 此前漏传 precomputed_*，_greedy_fitness
    # /_greedy_select 每轮会重算 0.27s 的全列 IC 矩阵；传入后与 worker 同源同值。
    from factors.analysis import _rolling_ic
    _ics = pd.DataFrame({c: _rolling_ic(mat[c], close, h=h, method="rank", window=ic_window)
                          for c in mat.columns}, index=mat.index).shift(h).ffill()
    _icir = _ics.rolling(ic_window * 5, min_periods=ic_window * 2).mean() / (
        _ics.rolling(ic_window * 5, min_periods=ic_window * 2).std() + 1e-12)

    # run5 E3：组合因子 expanding z 只预计算一次，全部 env 共享（此前每个
    # worker env 构造时重算 26 列 expanding mean/std——profile 中 expanding.std
    # 1692 次 cumtime 1.9s；同公式同输入，结果逐位一致）。
    _z_exp = {c: ((mat[c] - mat[c].expanding().mean())
                  / (mat[c].expanding().std() + 1e-12)).fillna(0.0)
              for c in mat.columns}

    env = FactorMiningEnv(mat, close, h=h, ic_window=ic_window,
                          max_steps=max_steps,
                          train_ratio=train_ratio, val_ratio=val_ratio,
                          corr_threshold=corr_threshold, seed=seed,
                          precomputed_ics=_ics, precomputed_icir=_icir,
                          precomputed_z=_z_exp,
                          use_action_mask=action_masking)
    agent = ACAgent(env.state_dim, env.n_actions, hidden=hidden,
                    lr_actor=float(cfg.get("lr_actor", 3e-3)),
                    lr_critic=float(cfg.get("lr_critic", 6e-3)),
                    gamma=float(cfg.get("gamma", 0.95)), seed=seed,
                    entropy_coef=float(cfg.get("entropy_coef", 0.05)))

    # 续训支持（与 train_drl 同口径）：base_agent 为内存 ACAgent 对象
    # （持续进化引擎用），权重必须 deepcopy——否则训练原地修改 best_agent，
    # 回退保护失效。续训只换起点，OOS 安检/回退比较等防线照常生效。
    base_agent = cfg.get("base_agent")
    if base_agent is not None:
        try:
            if base_agent.state_dim == agent.state_dim and base_agent.n_actions == agent.n_actions:
                agent.actor = copy.deepcopy(base_agent.actor)
                agent.critic = copy.deepcopy(base_agent.critic)
                log.info("[factor_miner] 已加载内存基础模型作为续训起点（state_dim=%d）",
                         agent.state_dim)
                agent.actor.lr = float(cfg.get("lr_actor", 3e-3))
                agent.critic.lr = float(cfg.get("lr_critic", 6e-3))
                agent.epsilon = 0.4
            else:
                log.warning("[factor_miner] 基础模型维度不匹配(%s vs %s)，忽略，从零训练",
                            getattr(base_agent, "state_dim", "?"), agent.state_dim)
        except Exception as e:  # noqa: BLE001
            log.warning("[factor_miner] 内存基础模型接入失败，从零训练: %s", e)
    # 单轮训练时间预算（秒）：>0 时超时提前结束（连续训练控成本）
    time_budget = float(cfg.get("time_budget", 0.0) or 0.0)

    # 训练循环（PPO 批量收集 + 多轮 mini-batch）
    history: list[dict] = []
    best_fitness = -1e9
    best_agent: Optional[ACAgent] = None
    from concurrent.futures import ThreadPoolExecutor

    def _collect_one(args: tuple) -> dict:
        ep_i, worker_i = args
        local_env = FactorMiningEnv(mat, close, h=h, ic_window=ic_window,
                                    max_steps=max_steps,
                                    train_ratio=train_ratio, val_ratio=val_ratio,
                                    corr_threshold=corr_threshold,
                                    seed=seed + ep_i * 1000 + worker_i,
                                    precomputed_ics=_ics, precomputed_icir=_icir,
                                    precomputed_z=_z_exp,  # E3：共享预计算 z
                                    use_action_mask=action_masking)
        # D1：独立 rng 采样（绝不共享 agent._rng，保证多线程可复现）
        local_rng = np.random.default_rng(seed + ep_i * 2000 + worker_i)
        return agent.collect_episode(local_env, rng=local_rng)

    # E-EXP：并行收集线程数（默认 4 与旧行为一致；16 核机器上 n_episodes=8
    # 只需 1 波收集，墙钟约减半——经 .optim/exp_training/bench_parallel.py 实测）
    max_workers = min(n_episodes, int(cfg.get("max_workers", 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        lr_actor_init = agent.actor.lr
        lr_critic_init = agent.critic.lr
        for ep in range(1, episodes + 1):
            # 时间预算：至少完成 1 轮后超时即提前收尾
            if time_budget > 0 and ep > 1 and (time.time() - t0) >= time_budget:
                log.info("[factor_miner] 单轮训练达到时间预算 %.0fs（已完成 %d 轮），提前结束",
                         time_budget, ep - 1)
                break
            # E-EXP：学习率退火（训练后期降低更新幅度，收敛更稳，与 train_drl 同口径）
            if lr_anneal:
                frac = ep / episodes
                lr_scale = lr_final_ratio + (1 - lr_final_ratio) * (1 - frac)
                agent.actor.lr = lr_actor_init * lr_scale
                agent.critic.lr = lr_critic_init * lr_scale
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
            # E-EXP：greedy_evals>1 时对多个随机决策时点取平均——单点时点噪声
            # 大，选最优模型可能被"运气好的时点"误导（同样策略不同 t 的
            # fitness 可差数倍），平均后选模更稳。
            fits = [_greedy_fitness(agent, env, action_masking) for _ in range(greedy_evals)]
            fit = float(np.mean(fits))
            if fit > best_fitness:
                best_fitness = fit
                # P2：JSON 往返改为 deepcopy（to_dict→from_dict 要序列化全部权重）
                best_agent = copy.deepcopy(agent)
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

    # run24：空 history 守卫（episodes=0/时间预算首轮即超时/异常时 history 可为空，
    # 调用方 evolve_engine L1313 直接 history[-1] 会 IndexError；strategy/meta 路径
    # 均有守卫，factor 路径补上）
    if not history:
        raise ValueError("RL 挖掘未产出任何训练轮次（history 为空），无法生成因子组合")

    # ---- 最终决策：greedy 跑一个完整 episode ----
    decision = _greedy_select(final_agent, env, action_masking)
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
        "action_masking": action_masking, "greedy_evals": greedy_evals,
        "lr_anneal": lr_anneal,
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


def _greedy_fitness(agent: ACAgent, env: FactorMiningEnv,
                    action_masking: bool = False) -> float:
    """确定性策略跑一个 episode 的最终组合 fitness（验证段评估）。"""
    state = env.reset()
    for _ in range(env.max_steps * 3 + 1):
        mask = env.valid_actions() if action_masking else None
        action = agent.greedy_action(state, mask=mask)
        state, _r, done, info = env.step(action)
        if done:
            return float(info.get("final_fitness", 0.0))
    return 0.0


def _greedy_select(agent: ACAgent, env: FactorMiningEnv,
                   action_masking: bool = False) -> dict:
    """确定性策略完整决策：返回选中因子列表（按加入顺序）。"""
    state = env.reset()
    for _ in range(env.max_steps * 3 + 1):
        mask = env.valid_actions() if action_masking else None
        action = agent.greedy_action(state, mask=mask)
        state, _r, done, info = env.step(action)
        if done:
            return info
    return {"selected": []}
