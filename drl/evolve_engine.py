"""持续进化引擎：管理因子挖掘和策略训练的持续循环，实现"越训练越强"。

核心流程：
1. 后台循环按间隔检查是否有新数据
2. 加载上次最佳模型继续训练（而非从头开始）
3. 滚动窗口训练（最近 N 根 K 线）
4. 验证段评估新模型 vs 最佳模型，退化时自动回退
5. 保存最佳模型到 ModelZoo

使用方式：
    evolve = EvolveEngine(db, bus, settings.data_dir / "models")
    await evolve.start()
    # ... 运行中 ...
    await evolve.pause()
    await evolve.resume()
    await evolve.stop()
"""
import asyncio
import concurrent.futures
import json
import logging
import math
import random
import threading
import time
from datetime import datetime, timezone
from collections.abc import Awaitable
from typing import Any, Callable, Optional

from backtest.data_loader import generate_demo
from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from drl.agent import ReplayBuffer, train_drl
from drl.factor_miner import train_factor_miner
from factors.engine import compute_factor_matrix
from strategies import get_dynamic, register_dynamic
from strategies.rl_adaptive import RLAdaptiveStrategy

from .model_zoo import ModelZoo

log = logging.getLogger(__name__)

# P4-E1：K线数据缓存键上限（symbol×timeframe 组合只增不减，超限淘汰最旧）
_DATA_CACHE_MAX_KEYS = 16


class InsufficientDataError(RuntimeError):
    """数据不足的业务分支（非崩溃）：_run_loop 用 warning 记日志、不带全栈。

    此前以裸 RuntimeError 表达，落在通用 except 里 log.exception 每 5 分钟
    刷一屏 traceback，且把「数据不足」与真正的训练崩溃混在同一语义。
    """


def _strategy_drl_train_cfg(episodes: int) -> dict:
    """持续进化「策略 DRL」的训练配置（单一定义处：训练与 Pine 导出口径同源）。

    state_window / hidden 默认取「装得进 Pine」的组合：神经引擎导出自动交易代码时
    权重整段内联，旧配置 state_window=20 + hidden=[64,64] 有 21184 个权重
    （脚本约 700KB），导出器只能显式拒绝——进化产物于是永远没有 Pine 代码，
    图上也就永远没有买卖点（本次关键要求的症结）。窗口从 20 降到 4 牺牲部分
    时序记忆，同时也在压低过拟合风险；要恢复旧规模改 config.yaml 两项即可。
    """
    return {
        "episodes": episodes,
        "n_episodes": max(1, int(getattr(settings, "evolve_strategy_drl_n_episodes", 16) or 0) or 8),
        "collect_workers": max(0, int(getattr(settings, "evolve_strategy_drl_collect_workers", 0) or 0)),
        "ppo_epochs": max(1, int(getattr(settings, "evolve_strategy_drl_ppo_epochs", 8) or 0) or 4),
        "mini_batch_size": 128,
        "hidden": [int(h) for h in getattr(settings, "evolve_strategy_drl_hidden", [16, 16])],
        # 固定种子（2026-10，可复现）：旧值 None=每轮随机，单 seed 下训练结果不可复现、
        # 跨重启不可比；配合 evolve_strategy_drl_seeds>1 做多 seed 中位数选模时，
        # 种子按 base+i 派生（见 _run_multi_seed）。
        "seed": int(getattr(settings, "evolve_strategy_drl_seed", 42) or 42),
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "fee_rate": 0.001,
        "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 40.0)),
        "cost_scale": 1.0,
        # 训练环境最小交易区（2026-09-22）：0.05→0.10 让模型学会少动、降换手
        #（v44/v47 长窗亏损主因之一是高换手成本侵蚀；buy_zone 部署侧扫描无效，
        # 改在训练侧解决）。
        "min_trade_zone": float(getattr(settings, "evolve_strategy_drl_min_trade_zone", 0.10)),
        "state_window": int(getattr(settings, "evolve_strategy_drl_state_window", 4)),
        "lr_actor": 3e-3,
        "lr_critic": 6e-3,
        "gamma": 0.95,
        "entropy_coef": 0.05,
        # 2026-10 根本性逻辑错误修复：奖励口径 / 决策间隔（详见 config/settings.py 注释
        # 与 docs/DRL训练根本性逻辑错误-诊断报告.md）。训练、验证、OOS、长窗观测
        # 全部由 env.policy_env_kwargs 同源取值，防止"训练每4根决策、评估每根决策"的口径漂移。
        "reward_mode": str(getattr(settings, "evolve_strategy_drl_reward_mode", "excess")),
        "benchmark_position": float(getattr(settings, "evolve_strategy_drl_benchmark_position", 1.0)),
        "risk_coef": float(getattr(settings, "evolve_strategy_drl_risk_coef", 0.5)),
        "decision_interval": max(1, int(getattr(
            settings, "evolve_strategy_drl_decision_interval", 1) or 1)),
        # 经验量：单条 episode 的时间窗长度（0=整段训练段）。>0 时走标的池采样，
        # 每条轨迹随机取一个年代切片（配合 history_bars 拉满多年历史）。
        "episode_window": max(0, int(getattr(settings, "evolve_strategy_drl_episode_window", 0) or 0)),
        # 做空（2026-10）：动作空间 −1~+1。默认关闭——执行内核（撮合/纸面/实盘）
        # 当前为现货语义、无法建立负仓位，开启只用于**离线实验/评估**；
        # 含做空档的模型会被部署门显式阻断（见 _strategy_deploy_gate）。
        "short_enabled": bool(getattr(settings, "evolve_strategy_drl_short_enabled", False)),
        "val_eval_interval": 10,
        # 2026-09-20（regime 诊断）：val 段切成 3 段取收益中位数选模——避免单一
        # regime（如下行震荡）选出"只会在该段表现好"的保守模型（OOS 上行不交易）。
        "val_segments": 3,
        "early_stop_patience": 0,
        "reward_dd_penalty": 0.0,
        # E-EXP（2026-09 实验落地）：连亏抑制塑形。sweep_drl_rewards.py 30ep 双标的：
        # baseline（vol20、无塑形）在 BTC OOS 为负（-0.006/sharpe-0.48）；vol40+此 0.5
        # 后 OOS 转正（BTC +0.0005、ETH +0.0057/sharpe+0.42）。趋势塑形/尾部罚/回撤罚
        # 均经实验判定无益或有害，保持 0。
        "reward_losing_penalty": 0.5,
        "reward_trend_align": 0.0,
        # P1 课程学习（能力层次）：早期只采低难度窗口，随进度放开。
        # 不进入 _training_identity（同 episodes 语义，改动不触发锚点不可比）。
        "curriculum": bool(getattr(settings, "evolve_strategy_drl_curriculum", False)),
        "curriculum_start": float(getattr(settings, "evolve_strategy_drl_curriculum_start", 0.3)),
        # P1 时序网络升级：actor 骨干。默认 mlp（向后兼容，持续进化行为不变）；
        # "gru" 用 GRU 序列编码，能力更强但无法直接导出 Pine——依赖蒸馏（distill=True
        # 默认开）把 GRU teacher 压成 MLP student 供 TradingView。
        "backbone": str(getattr(settings, "evolve_strategy_drl_backbone", "mlp")),
        "gru_hidden": int(getattr(settings, "evolve_strategy_drl_gru_hidden", 32)),
        # P1-5：OOS 安检门槛（settings.evolve_oos_min_bars，默认 250）
        # 与多段评估（3 段取中位数，降单段运气噪声）
        "oos_min_bars": getattr(settings, "evolve_oos_min_bars", 250),
        "oos_segments": 3,
    }


from drl.evolve_observations import EvolveObservationsMixin  # noqa: E402
from drl.evolve_rules import (  # noqa: E402
    _should_rollback, _should_from_scratch_reset,
    _longwindow_verdict,  # noqa: F401  # 兼容导出（实际使用在 evolve_observations）
    _oos_gate_passed,
)


def _strategy_drl_identity_config(train_cfg: dict, multi_names: list) -> dict:
    """策略 DRL 训练身份指纹的配置项（单一定义处：引擎与测试同源）。

    身份指纹必须覆盖所有"影响状态/奖励/切分/部署门"的训练口径，否则不同口径的
    产出会被拿去互相回退比较。2026-10 新增奖励口径/决策间隔/风险系数/时间窗：
    旧锚点是在"躺平即最优"的目标下训出来的（fitness 大量恰为 0.0），
    与新目标下的产出不可比，必须触发锚点重建。
    """
    return {
        "episodes": int(getattr(settings, "evolve_strategy_drl_episodes", 8)),
        "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
        "oos_min_bars": int(getattr(settings, "evolve_oos_min_bars", 250)),
        # 2026-10 审计修复（HIGH-3）：**不能**把"本轮实际取到数据的额外标的名单"
        # 混入身份指纹——任一额外标的瞬时 fetch 失败（429/网络抖动）都会让同一配置
        # 的指纹逐轮漂移 ⇒ 回退保护被旁路/挑战者永不晋升（factor_miner 端 2026-09
        # 已修过同类 bug："把训练输出纳入身份 → 回退保护永久旁路"）。
        # 多标的**开关**（设置值，稳定）保留——开/关是真实的训练语义差异；
        # 不能再取运行时名单 bool(multi_names)（瞬时 fetch 失败会让它漂移）。
        "multi_symbol": bool(getattr(settings, "evolve_strategy_drl_multi_symbol", True)),
        # P1 时序网络升级：actor 骨干是训练口径——GRU 与 MLP 架构不同，
        # 混入身份指纹，防止"GRU 训练产出"与"MLP 锚点"互相回退比较。
        "backbone": str(train_cfg.get("backbone", "mlp")),
        # 2026-10 根本性逻辑错误修复：奖励口径 / 决策间隔 / 风险系数 / 时间窗
        "reward_mode": str(train_cfg.get("reward_mode", "absolute")),
        "decision_interval": int(train_cfg.get("decision_interval", 1)),
        "risk_coef": float(train_cfg.get("risk_coef", 0.5)),
        "episode_window": int(train_cfg.get("episode_window", 0) or 0),
        # 2026-10 验收/ fitness 口径：绝对收益 vs 基准相对超额。锚点 fitness 的
        # **语义**变了（同一数字含义不同），必须混入身份指纹，否则会拿"绝对值时代"
        # 的锚点去回退"超额时代"的候选。
        "fitness_excess": bool(getattr(
            settings, "evolve_strategy_drl_fitness_excess", True)),
        "deploy_requires_benchmark": bool(getattr(
            settings, "evolve_strategy_drl_deploy_requires_benchmark", True)),
        # 分段稳健性入闸同样属于**部署门口径**：验收语义变了 → 旧锚点不能用于回退比较
        "deploy_min_positive_fraction": float(getattr(
            settings, "evolve_strategy_drl_deploy_min_positive_fraction", 0.0) or 0.0),
        # 动作空间（做空开关）改变了策略能做什么，与长仓产出不可互比
        "short_enabled": bool(train_cfg.get("short_enabled", False)),
    }


_TRAINING_CFG_KEYS = (
    # 2026-10 审计修复（MED）：增量门失效白名单——这些配置直接影响 strategy_drl
    # 训练口径，热更新任一键都必须让 _last_tail_ts 失效（立即重训），否则面板显示
    # "已应用"而后台仍按旧口径 gating 直到有新 K 线。
    "evolve_strategy_drl_seed", "evolve_strategy_drl_seeds",
    "evolve_strategy_drl_episodes", "evolve_strategy_drl_n_episodes",
    "evolve_strategy_drl_ppo_epochs", "evolve_strategy_drl_collect_workers",
    "evolve_strategy_drl_hidden", "evolve_strategy_drl_backbone",
    "evolve_strategy_drl_gru_hidden", "evolve_strategy_drl_curriculum",
    "evolve_strategy_drl_curriculum_start",
    "evolve_strategy_drl_decision_interval", "evolve_strategy_drl_reward_mode",
    "evolve_strategy_drl_benchmark_position", "evolve_strategy_drl_risk_coef",
    "evolve_strategy_drl_episode_window", "evolve_strategy_drl_history_bars",
    "evolve_strategy_drl_min_trade_zone", "evolve_strategy_drl_short_enabled",
    "evolve_strategy_drl_multi_symbol", "evolve_strategy_drl_multi_mix_ratio",
    "evolve_strategy_drl_replay_capacity", "evolve_strategy_drl_replay_mix_ratio",
    "evolve_strategy_drl_fitness_excess", "evolve_train_time_budget",
)


def _cfg_epoch_hash(snapshot: dict) -> str:
    """训练口径配置快照的稳定指纹（供审计测试锁"白名单键变更 ⇒ 指纹变化"）。"""
    import hashlib
    return hashlib.sha256(
        json.dumps({k: v for k, v in snapshot.items() if k in _TRAINING_CFG_KEYS},
                   sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _seed_score(res: Optional[dict]) -> float:
    """多 seed 选模的评分：验证段超额（模型没见过的段，跨轮可比）。

    OOS **不参与选模**——OOS 是选完模之后的独立安检（部署门），
    用它选模等于把样本外泄漏进选择。缺超额字段（老结果）回退验证段绝对收益。
    """
    if res is None:
        return -1e18
    vx = res.get("best_val_excess")
    if vx is not None:
        try:
            return float(vx)
        except (TypeError, ValueError):
            pass
    try:
        return float(res.get("best_val_ret", -1e18) or -1e18)
    except (TypeError, ValueError):
        return -1e18


def _seed_meta(res: dict, seed: int) -> dict:
    """单个训练结果的可见摘要（审计/前端展示用，不做判据）。"""
    oos = res.get("oos_report") or {}
    return {
        "seed": int(seed),
        "val_excess": res.get("best_val_excess"),
        "val_ret": res.get("best_val_ret"),
        "oos_ret": oos.get("oos_ret"),
        "oos_excess": oos.get("oos_excess"),
        "elapsed_sec": res.get("elapsed_sec"),
    }


def _run_multi_seed(train_drl_fn, train_cfg: dict, n_seeds: int,
                    base_seed: int) -> tuple[dict, dict]:
    """多 seed 训练 + 中位数选模（2026-10 解决"单轮判据像抽奖"）。

    实测（生产配置 1h/k=8/128 轮，3 个 seed 取自 `_strategy_drl_train_cfg(seed=None)`）：
    验证段超额 13.1% / 6.4% / 16.0%，OOS 超额 +2.4% / −1.5% / −5.3% ——
    单次训练的成败接近抛硬币。多 seed 的意义不在"多训几次"，
    而在**选模判据从"单个样本"变成"中位数样本"**：最佳/最差都是运气，
    中位数更能代表该配置的真实能力。

    - 选模标准 = 验证段超额的中位数（OOS 不参与，避免泄漏进选择）；
    - 偶数个 seed 取排序后下半的中位数（保守）；
    - 所有 seed 从同一续训起点（base_agent）出发，即"一个父代 → N 个子女 → 选中间那个"；
    - train_drl_fn(cfg) → result；返回 (选中的 result, meta)。

    meta: {"seed_runs": [...], "seed_chosen": seed, "n_seeds": N,
           "replay_buffer": 选中 seed 的回放缓冲副本（供引擎接回主缓冲）}

    2026-10 审计修复（HIGH）：seed 间回放缓冲必须**独立**。旧实现共享同一个
    replay_buffer 实例：seed i 的轨迹 push 进缓冲后，seed i+1 的 PPO 更新会采样到它
    ⇒"中位数=同一配置的独立样本"的前提被破坏、选模结果依赖 seed 执行顺序。
    现在每 seed 从主缓冲的**深拷贝副本**起步（seed 内自由推进互不污染），
    选中 seed 的最终副本经 meta 交回（跨轮记忆 = 上一轮选中 seed 的缓冲）。
    """
    import copy as _copy
    n = max(1, int(n_seeds or 1))
    base = max(0, int(base_seed if base_seed is not None else 42))
    main_buf = train_cfg.get("replay_buffer")
    runs: list[tuple[int, dict, object]] = []
    for i in range(n):
        cfg = dict(train_cfg)
        cfg["seed"] = base + i
        # 每 seed 独立回放副本（仅多 seed 需要；单 seed 直接共享主缓冲，行为不变）
        if main_buf is not None and n > 1:
            cfg["replay_buffer"] = _copy.deepcopy(main_buf)
        try:
            r = train_drl_fn(cfg)
        except Exception:  # noqa: BLE001 —— 单个 seed 失败不拖垮整轮，但必须显式可见
            log.exception("[evolve] 多 seed 训练 seed=%d 失败（跳过，改用其余 seed）", base + i)
            r = None
        if r is not None and isinstance(r, dict) and r.get("history"):
            runs.append((i, r, cfg.get("replay_buffer")))
    if not runs:
        raise RuntimeError("多 seed 训练全部失败，无结果可选")
    scored = sorted(runs, key=lambda p: _seed_score(p[1]))
    # 2026-10 审计修复（MED）：偶数个 seed 取**下半**中位数（与文档一致、保守）。
    # 旧实现 `len//2` 在 n=2 时取较高分那个 → 退化成最大值挑选（正是注释要消灭的
    # "抽奖"行为）；`(len-1)//2`：n=2→0（保守）、n=3→1（中位）、n=4→1（下半）。
    chosen_idx, chosen_res, chosen_buf = scored[(len(scored) - 1) // 2]
    meta = {
        "n_seeds": len(runs),
        "seed_runs": [_seed_meta(r, base + i) for i, r, _ in runs],
        "seed_chosen": base + chosen_idx,
    }
    if chosen_buf is not None:
        meta["replay_buffer"] = chosen_buf
    chosen_res["seed_runs"] = meta["seed_runs"]
    chosen_res["seed_chosen"] = meta["seed_chosen"]
    log.info("[evolve] 多 seed 中位数选模：%d 个 seed（超额 %s），选中 seed=%d（超额=%s）",
             len(runs),
             [round(_seed_score(r), 4) for _, r, _ in runs],
             base + chosen_idx, round(_seed_score(chosen_res), 4))
    return chosen_res, meta


def _oos_inactive_reason(oos: dict, min_turnover: float) -> Optional[str]:
    """样本外"一手没做"的判据（活跃度守卫），返回原因或 None。

    为什么必须独立于收益判据（2026-10 验收口径修复）：改成基准相对验收后，
    "全程不交易"在下跌窗口里会白拿正超额（oos_ret=0 而基准 −27% → 超额 +27%），
    那是**没参与**而不是**会择时**，不该被当成能力部署。
    判据优先级（新→旧，兼容老报告）：
      1. oos_turnover（实收换手基数，最准：直接来自环境成本记账）
      2. oos_trades（真实交易笔数）
      3. oos_position_ratio（段末持仓比例，最弱：交易后已平仓会被误判）
    """
    raw_turn = oos.get("oos_turnover")
    if raw_turn is not None:
        try:
            turn = float(raw_turn)
        except (TypeError, ValueError):
            turn = None
        if turn is not None:
            if turn < float(min_turnover):
                return (f"OOS 段换手仅 {turn:.4f}（<{min_turnover}），"
                        f"样本外几乎没有交易，不构成择时能力")
            return None
    raw_trades = oos.get("oos_trades")
    if raw_trades is not None:
        try:
            n_trades = int(raw_trades)
        except (TypeError, ValueError):
            n_trades = None
        if n_trades is not None:
            return (f"OOS 段模型从未发出交易指令(oos_trades={n_trades})，"
                    f"样本外未交易" if n_trades <= 0 else None)
    raw_pos = oos.get("oos_position_ratio")
    if raw_pos is not None:
        try:
            pos_ratio = float(raw_pos)
        except (TypeError, ValueError):
            pos_ratio = None
        if pos_ratio is not None and pos_ratio < 0.05:
            return f"OOS 段几乎无信号(position_ratio={pos_ratio:.3f})，样本外未交易"
    return None


def _strategy_deploy_gate(result: dict) -> tuple[bool, str]:
    """新模型能否接管实盘（策略 DRL / 元控制器共用）：先过 OOS 硬门，再按**基准相对**验收。

    train_drl 的硬门只在「训练段盈利且 OOS 衰减过半」时判过拟合，训练段本身亏损时
    decay=0（不可比、不判定）；而进化引擎旧逻辑连这个硬门都没读，只要数据源不是
    demo 就注册 rl_evolve 并用新权重覆盖实盘模型文件——未通过样本外检验的模型会被
    自动上线。

    P1-6：判据统一为 OOS 收益（与回退保护同口径）。
    P1-7：regime 剧变时把原因写清楚，供用户判断是否行情状态切换造成的假阳性。

    2026-10 验收口径修复（docs/DRL成本标定-决策频率与滑点复核.md §4）：
    旧判据 `oos_ret > 0` 与**训练目标（基准相对超额）**、与**动作空间（只能做多/空仓）**
    互相矛盾——OOS 窗口是熊市时，多头策略的最优解是全程空仓 = 0%，而 0 被 `>0` 拒掉，
    **结构上不可能有任何模型过门**（产线实测 452/501 轮被拒）。
    新判据：`oos_ret - 同段买入持有 > 0`（超额为正）——熊市里保护住资本算合格，
    牛市里"只会满仓躺赢"（超额 0）被拒，只有真择时能过。
    同时加**活跃度守卫**：靠"从不交易"在熊市里白拿超额的不算能力。
    报告缺基准字段（老模型/精简报告）或开关关闭时，自动退回旧的绝对收益判据。
    """
    oos = result.get("oos_report") or {}
    if result.get("deployment_blocked") or oos.get("hard_rejected"):
        return False, str(oos.get("reason") or "OOS 硬门拦截（疑似过拟合）")
    if not oos.get("enabled"):
        return False, "OOS 段样本不足（<最小样本数），无样本外证据不得自动接管实盘"
    # 执行层能力门（2026-10 做空）：含做空档的模型在**执行内核支持负仓位之前**
    # 一律不可部署。撮合内核 backtest/_matching.py 是现货语义
    #（sell 要求 position>0 且 qty≤position），纸面/实盘同样无做空路径——
    # 若放行，部署端会把负档位当"卖出/空仓"执行，训练与实盘彻底不是一回事
    #（这正是本项目一路在消灭的口径漂移）。宁可显式拒绝并说清原因。
    if result.get("requires_short") or bool(result.get("short_enabled")):
        return False, ("该模型动作空间含做空档（−1~+1），而执行内核（撮合/纸面/实盘）"
                       "当前为现货语义、无法建立负仓位：训练会做空但实盘执行不了，"
                       "不得部署（需先完成撮合内核做空支持）")
    try:
        oos_ret = float(oos.get("oos_ret", 0.0))
    except (TypeError, ValueError):
        oos_ret = 0.0
    if not math.isfinite(oos_ret):
        # run24 修复：NaN/Inf oos_ret 曾绕过硬门（float("nan") <= 0 为 False → 部署）。
        # 行情数据缺口导致 NaN 指标时不得上线："无有效 OOS 收益"按非正拦截。
        return False, f"OOS 段收益无效（{oos_ret}，非有限数值），数据缺口，暂不部署"
    _regime_note = ("；且训练/样本外行情状态剧变(regime_shift)"
                    if oos.get("regime_shift") else "")

    # ---- 活跃度守卫（先于收益判据：不参与的"超额"不是能力） ----
    _min_turn = float(getattr(settings, "evolve_strategy_drl_deploy_min_oos_turnover", 0.5) or 0.0)
    _inactive = _oos_inactive_reason(oos, _min_turn)
    if _inactive:
        return False, f"{_inactive}{_regime_note}"

    # ---- 基准相对验收（默认开；报告缺基准或开关关闭时退回旧绝对判据） ----
    _need_bh = bool(getattr(settings, "evolve_strategy_drl_deploy_requires_benchmark", True))
    _bh_raw = oos.get("oos_bh_ret")
    if _need_bh and _bh_raw is not None:
        try:
            oos_bh = float(_bh_raw)
        except (TypeError, ValueError):
            oos_bh = None
        if oos_bh is not None and math.isfinite(oos_bh):
            excess = oos_ret - oos_bh
            if excess <= 0:
                return False, (f"样本外未跑赢买入持有：模型 {oos_ret:.4f} vs 基准 {oos_bh:.4f}"
                               f"（超额 {excess:.4f} ≤ 0）{_regime_note}")
            # ---- 分段稳健性（2026-10）：正超额子段 ≥ ceil(n×比例)，防"一段灾难掩盖" ----
            # 中位超额为正但多数子段为负（如 [-0.45, +0.02, +0.03]，中位 +0.02 旧门可过）
            # = 择时能力并不存在，只是运气在两个小段上微赚。要求正超额子段达比例。
            # 缺逐段字段（老模型/精简报告）或 fraction<=0（关闭）时跳过，向前兼容。
            _frac = float(getattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                                  0.0) or 0.0)
            if _frac > 0.0:
                _segs = oos.get("oos_excess_segments") or []
                _n = int(oos.get("oos_segments", 0) or 0) or len(_segs)
                if _n >= 1 and (_segs or oos.get("oos_excess_positive_segments") is not None):
                    _pos_raw = oos.get("oos_excess_positive_segments")
                    _pos = (int(_pos_raw) if _pos_raw is not None
                            else int(sum(1 for x in _segs if x > 0)))
                    # −1e-9 防浮点边界：配置 0.666667×3=2.000001 会被 ceil 成 3，
                    # 把"≥2/3"悄悄变成"必须3/3"（实测 E 配置被误拒）；settings 默认
                    # 2.0/3.0 的积在 fp 里是 1.999…，本就不需要，但用户手填小数时会踩中。
                    _need = max(1, int(math.ceil(_n * _frac - 1e-9)))
                    if _pos < _need:
                        return False, (f"分段稳健性不足：{_n} 段中仅 {_pos} 段超额为正"
                                       f"（需 ≥{_need} 段）{_regime_note}")
            return True, ""

    if oos_ret <= 0:
        # 旧绝对判据（无基准字段/开关关闭）：P2-14 区分"样本外无信号"与"真亏损"，
        # 活跃度已在上方守卫里判过，这里只说明收益非正。
        return False, f"OOS 段收益 {oos_ret:.4f} 非正，样本外不具备盈利能力{_regime_note}"
    return True, ""


def _reject_downshift_retry(base_interval: int, reject_streak: int, floor: int) -> int:
    """连续 OOS 被拒降频（第2项完善）：被拒轮数达到阈值后训练间隔放大 5 倍。

    floor<=0 表示关闭（保持固定间隔）；未触发则原样返回。纯函数，便于单测。
    """
    if floor <= 0 or reject_streak < floor:
        return base_interval
    return max(base_interval * 5, 300)


def _training_identity(model: str, symbol: str, timeframe: str, *,
                       state_window: int = 1, factor_signature: str = "",
                       config: Optional[dict] = None,
                       window_tail_ts: Optional[int] = None) -> dict:
    """当前训练轮的可比较身份：回退判据只在身份一致时生效。

    不同 symbol/timeframe/state_window/训练配置产出的模型互相比较没有意义
    （BTC 锚点不该压住 ETH 候选；1h 锚点不该与 15m 候选比）。config_fingerprint
    用稳定排序后的结构化 JSON 做 SHA-256，只覆盖影响状态/奖励/切分/部署门的口径。
    """
    stable = config or {}
    import hashlib
    canonical = json.dumps(
        stable, sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "model": model,
        "symbol": symbol,
        "timeframe": timeframe,
        "state_window": int(state_window),
        "factor_signature": str(factor_signature or ""),
        "config_fingerprint": fingerprint,
        "window_tail_ts": window_tail_ts,
    }


def _fitness_mode_conv(v) -> str:
    """因子训练目标口径校验：仅允许 ic / net（2026-09-29 成本对齐修复）。"""
    s = str(v).strip().lower()
    if s not in ("ic", "net"):
        raise ValueError(f"fitness_mode 只能是 ic 或 net，收到: {v!r}")
    return s


def _identity_mismatch(anchor_meta: dict, identity: dict) -> Optional[str]:
    """锚点与候选的身份差异。返回 None 表示可比；否则给出不可比原因。

    锚点缺身份字段（旧版本）视为 unknown：不进行跨口径回退比较，
    但保留其权重作为可运行部署（comparable 仍由 demo 口径控制）。
    """
    for key in ("model", "symbol", "timeframe", "state_window",
                "factor_signature", "config_fingerprint"):
        a = anchor_meta.get(key)
        b = identity.get(key)
        if a is None:
            return f"锚点缺少身份字段 {key}（旧版本，不可比）"
        if str(a) != str(b):
            return f"{key} 不一致（锚点 {a} vs 候选 {b}）"
    return None


def _clear_gap01_streak_on_version(eng, new_version, *, prev_version) -> None:
    """部署版本**更换**时清空 GAP-01 的 FAIL 计数（2026-10 审计修复 MED）。

    旧实现只在版本变化时重置 PASS 计数（_gap01_pass_*），FAIL 计数（_gap01_streak）
    不清——旧模型积累的 FAIL 会传导到新模型，触发冠军 from-scratch。

    语义边界：版本**首次确定**（prev_version 为 None，即第一个部署模型的评估开始）
    不属于"换模型"，不清 FAIL——否则第一轮的 FAIL 被吞，连续 FAIL 的判定失真。
    """
    try:
        if prev_version is None or prev_version == new_version:
            return
        if eng._gap01_streak.get("strategy_drl", 0) != 0:
            log.info("[evolve] 部署版本更换(%s→%s)：清空 GAP-01 FAIL 计数（旧模型 FAIL 不传导）",
                     prev_version, new_version)
        eng._gap01_streak["strategy_drl"] = 0
        eng._strategy_drl_status["gap01_streak"] = 0
    except Exception:  # noqa: BLE001
        pass


def _bump_reject_streak(eng, model: str, *, lineage: str = "") -> None:
    """连续拒绝计数递增（2026-10 审计修复 MED）：challenger 谱系不递增。

    _log_round 的旧逻辑对任何非 ok/demo_blocked 状态 +1——challenger_kept/degraded
    （"晋级失败被冠军压住"的健康交替）也累积，约每 5 个挑战者轮就会把选模合格的
    冠军强制 from-scratch 重训（丢经验）。挑战者谱系不计入；main 谱系照旧。
    """
    if lineage == "challenger":
        return
    eng._reject_streak[model] = int(eng._reject_streak.get(model, 0)) + 1


def _make_cross_oos_env(df, *, state_window: int, decision_interval: int,
                        min_trade_zone: float, short_enabled: bool = False,
                        reward_mode: str = "absolute", benchmark_position: float = 1.0,
                        risk_coef: float = 0.5) -> object:
    """跨标 OOS 评估环境（2026-10 审计修复 HIGH-1c）。

    此前硬编码 k=1/absolute/zone=0.05 评估 k=8/excess 训练的模型——验收门量的是
    错误语义（更高频交易、更小死区），主 OOS 门与跨标 OOS 门对同一模型会自相矛盾。
    现在必须与训练/主 OOS 段同口径（policy_env_kwargs 语义）。
    """
    from drl.env import TradingEnv
    return TradingEnv(
        df,
        start_cash=float(getattr(settings, "default_start_cash", 10000.0)),
        fee_rate=0.001,
        vol_penalty=float(getattr(settings, "evolve_vol_penalty", 20.0)),
        slippage=0.0005,
        min_trade_zone=float(min_trade_zone),
        extra_factors=None,
        state_window=max(1, int(state_window)),
        reward_mode=str(reward_mode),
        benchmark_position=float(benchmark_position),
        risk_coef=float(risk_coef),
        decision_interval=max(1, int(decision_interval or 1)),
        short_enabled=bool(short_enabled),
    )


class EvolveEngine(EvolveObservationsMixin):
    """持续进化引擎：管理因子挖掘和策略训练的持续循环。"""

    def __init__(self, db: Database, bus: EventBus, models_dir) -> None:
        self.db = db
        self.bus = bus
                # P4-A5：版本保留数从 settings 传入（此前 ModelZoo 硬编码 _MAX_VERSIONS=10，
        # settings.evolve_max_versions 是死配置，用户修改无效）
        self.zoo = ModelZoo(models_dir,
                            max_versions=int(getattr(settings, "evolve_max_versions", 10)))
        self._running = False
        self._paused = False
        self._tasks: list[asyncio.Task] = []
        # 2026-10 审计修复（留档项）：优雅停止事件。stop()/set_enabled(False) 置位后，
        # 训练循环在**当前轮完整结束（含写盘/注册/落库）**后自然退出，而非硬 cancel——
        # 硬 cancel 只中断协程 await，executor 线程跑完训练但后处理段（save_agent/
        # register/_log_round）可能丢失（"训练+写盘必然完整执行"的旧注释不成立）。
        self._stop_event = asyncio.Event()
        # 保活（watchdog）：_run_loop 只能兜住"逐轮训练"里的异常。一旦异常从
        # 循环体之外逸出，那条 asyncio 任务会直接结束，而 _running/_enabled 仍是
        # True——引擎对外依然显示"运行中"，实际已永久停训，只能重启进程或手动
        # 关开一次开关。下面这组状态供 _supervise_loop 周期巡检并重建死循环。
        self._loop_names = ("evolve_factor_miner", "evolve_strategy_drl")
        self._restarts: dict[str, int] = {n: 0 for n in self._loop_names}
        self._last_restart: dict[str, float] = {n: 0.0 for n in self._loop_names}
        self._supervise_interval = 30.0
        # 同一管线两次自愈的最小间隔：防"一重建就崩"演变成重启风暴烧 CPU
        self._restart_min_gap = 20.0
        self._supervise_status: dict[str, Any] = {
            "alive": False, "last_check": 0.0, "checks": 0, "restarts": 0,
            "interval": self._supervise_interval, "last_restart_at": 0.0,
            "last_restart_name": "",
        }
        # P2-10：全局训练锁——两条训练管线串行执行（因子挖掘/策略DRL
        # 同时训练会吃满 CPU 拖垮实时引擎），手动触发与周期循环共用同一把锁。
        self._train_lock = asyncio.Lock()
        # 部署后后台回测摘要任务引用（防 GC；done 回调里弹出）
        self._backtest_tasks: dict[str, asyncio.Task] = {}
        # P2-11：数据增量门——记录各管线最近一次训练时的数据末根时间戳；
        # 数据无新增 K 线时跳过本轮（避免 60s 间隔 + 5 分钟数据缓存下的空转重训）
        self._last_tail_ts: dict[str, float] = {}
        # 数据缓存：5 分钟内不重复拉取交易所（避免频繁限流）
        self._data_cache: dict[str, tuple[float, Any]] = {}  # key -> (timestamp, df)
        self._data_cache_ttl = 300  # 5 分钟
        # P4-E1：缓存键上限（symbol×timeframe 组合只增不减，超限淘汰最旧）
        # 训练状态（_demo_streak 为内部字段，不对外展示：记录连续 demo 训练次数，
        # 供 _run_loop 在交易所持续不可达时自动降频，避免 60s 一次全量烧 CPU）
        self._factor_miner_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                     "last_error": "", "next_run": 0, "last_success": False,
                                     "oos_rejected": "", "_demo_streak": 0,
                                     "_reject_streak": 0,   # E-EXP：连续 OOS 被拒计数（自适应重置）
                                     "loop_beat": 0.0, "restarts": 0,
                                     # 进化组合因子部署状态（Task：进化因子使用场景；combo_ 前缀
                                     # 避开 _pipeline_view/_mark_deployment_status 的 deployed_* 键覆盖）
                                     "combo_strategy": "", "combo_version": "",
                                     "combo_deployed_at": 0, "combo_deploy_error": "",
                                     "combo_backtest": None, "combo_backtest_error": ""}
        self._strategy_drl_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                     "last_error": "", "next_run": 0, "last_success": False,
                                     "oos_rejected": "", "_demo_streak": 0,
                                     "loop_beat": 0.0, "restarts": 0,
                                     # P0-族群：冠军/挑战者轮次计数与当前谱系（展示用）
                                     "_pop_round": 0, "lineage": "main",
                                     # P0 多标的混合训练状态（[]=本轮单标的）
                                     "multi_symbol": []}
        # 策略 DRL 经验回放缓冲区（跨训练轮存活）：每轮只训 8ep，若缓冲每轮重建，
        # 旧行情经验在轮间被完全丢弃（灾难性遗忘的加速器）。持有实例后，上一轮
        # 乃至更早窗口的轨迹会参与本轮 PPO 更新（旧轨迹经重要性比率自然衰减）。
        # capacity 由 settings.evolve_strategy_drl_replay_capacity 决定（0=禁用）。
        self._strategy_drl_replay: Optional[ReplayBuffer] = None
        self._strategy_drl_replay_capacity = 0
        self._strategy_drl_replay_symbol = ""   # 缓冲区当前服务的训练标的（轮换时重建）
        # 锁死自检：连续多少轮没有任何模型被接受。进化引擎只看 episode/fitness 时
        # "一直在训"和"训了但全被锚点拒绝"长得一模一样（实测 72/72 全 rollback，
        # 前端毫无异常提示），所以这里按管线记连续拒绝数，供 status 判 stalled。
        # 注：元策略训练管线已移除（2026-09-28），现存两管线。
        self._models = ("factor_miner", "strategy_drl")
        self._reject_streak: dict[str, int] = {m: 0 for m in self._models}
        # GAP-01 连续 FAIL 计数（2026-09-20，软信号）：连续 N 轮 FAIL 触发
        # 冠军轮 from-scratch 重学（evolve_gap01_fail_reset_after，0=关闭）。
        self._gap01_streak: dict[str, int] = {}
        # 多窗口连续 PASS 判定（2026-09-21，§16.5 遗留1）：同一部署模型连续 N 次
        # GAP-01 观测 PASS 才算稳定（gap01_stable）；版本变化/非 PASS 即清零。
        self._gap01_pass_version: Optional[int] = None
        self._gap01_pass_streak: int = 0
        self._rounds_total: dict[str, int] = {m: 0 for m in self._models}
        self._stall_rounds = max(2, int(getattr(settings, "evolve_stall_rounds", 10)))
        # 从配置读取间隔
        self._factor_miner_interval = getattr(settings, "evolve_factor_miner_interval", 60)
        self._strategy_drl_interval = getattr(settings, "evolve_strategy_drl_interval", 60)
        self._rolling_window = getattr(settings, "evolve_rolling_window", 5000)
        self._rollback_threshold = getattr(settings, "evolve_rollback_threshold", 0.95)
        self._enabled = getattr(settings, "evolve_enabled", True)
        # 训练标的池（用户指定：仅 BTC/ETH/SOL），训练时轮换使用
        self._symbols = list(getattr(settings, "evolve_symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]))
        if not self._symbols:
            self._symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        # P4-A1：两管线共享同一个"当前训练标的"（不再各自轮换）。
        # 此前 factor_miner/strategy_drl 各自调用 _next_symbol 独立递增，
        # 导致级联因子（BTC 上挖出）被注入 ETH 训练——长度校验通过但语义错配，
        # 整个"因子→策略DRL"链路断裂。现在：两管线同标训练，
        # 全部完成一个 cycle 后才切换到下一标的。
        self._symbol_idx = 0
        # 当前标的已训练完成的管线集合（两管线齐全后切换标的）
        self._cycle_pipelines_done: set[str] = set()
        # 专用训练线程池（单 worker）：训练本体串行执行；stop/set_enabled(False)
        # 通过 _training_done 事件等待在跑的训练跑完——取消协程只会中断 await，
        # executor 线程会继续烧 CPU，等待后"禁用/停止"语义才名副其实
        self._train_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="evolve-train")
        self._training_done = threading.Event()
        self._training_done.set()
        # 拉数串行锁：两管线 + 跨标 OOS 并发打交易所会放大 429，统一串行化
        self._fetch_lock = asyncio.Lock()
        # 部署编排：可选的运行时 reload 回调（由 TradingEngine 注入）。
        # 未配置时部署仍完成文件与注册表层，runtime_reload 标记 skipped。
        self._deployment_reload: Optional[Callable[[str, int, dict], Awaitable[Optional[dict]]]] = None
        # 部署/候选状态：candidate 是本轮训练结果，deployed 是当前 best 快照。
        # 回退轮 candidate_fitness 与 deployed_fitness 分离后，UI 不再误把
        # 被回退的候选值当成"当前模型收益"。
        self._deploy_status: dict[str, dict] = {
            m: {
                "candidate_fitness": None, "deployed_fitness": None,
                "deployed_version": None, "deployed_source": "",
                "last_outcome": "", "anchor_mismatch": "",
                "runtime_reload": "skipped",
            } for m in ("factor_miner", "strategy_drl")
        }

    def set_deployment_reload(
        self,
        callback: Optional[Callable[[str, int, dict], Awaitable[Optional[dict]]]],
    ) -> None:
        """注入运行时模型热加载回调（TradingEngine 在构造完成后调用）。"""
        self._deployment_reload = callback

    async def start(self) -> None:
        """启动后台训练循环。"""
        if self._running:
            return
        self._running = True
        self._paused = False
        # 启动恢复：只要训练产物（flat 模型文件）存在就注册对应策略，
        # 保证「由持续进化训练的策略」在重启/多轮回退后依然在其他模块可选可用
        try:
            await self._restore_evolve_strategies()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 启动恢复策略注册失败: %s", e)
        # 重启后恢复进化组合因子的部署状态键（combo_strategy 等），否则面板
        # 会把已部署策略误显示为「尚未部署」（bug-fix 自实验发现）
        try:
            await self._restore_combo_deploy_status()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 启动恢复组合因子部署状态失败: %s", e)
        # 重启后恢复各管线已完成的训练轮次（episode），避免「重新进入软件后
        # 训练轮次显示归零」——轮次是累计事实，应以数据库落库为准接续显示
        try:
            await self._restore_episode_counts()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 启动恢复训练轮次失败: %s", e)
        if self._enabled:
            # 首次训练在启动 15s 后进行（等待系统初始化完成），之后按间隔定期训练
            self._spawn_loops()
            # 保活监督：两条循环任一条意外退出，都由巡检发现并重建
            self._spawn_supervisor()
            self._supervise_status["alive"] = True
            log.info("[evolve] 持续进化引擎已启动（因子挖掘=%ds，策略DRL=%ds，"
                     "首次训练将在15s后开始；保活巡检间隔=%ds）",
                     self._factor_miner_interval, self._strategy_drl_interval,
                     int(self._supervise_interval))
        else:
            log.info("[evolve] 持续进化引擎已禁用（evolve_enabled=false）")

    async def stop(self) -> None:
        """优雅停止训练循环（2026-10 审计修复留档项）。

        语义（三层）：
        1. 置 _running=False + _stop_event：真实循环在下一轮迭代自然退出；
        2. 等**训练本体**（executor 线程，无法 kill）跑完 + 给协程后处理（写盘/
           注册/落库）约 5s 缓冲——这两段保证"当前轮完整落盘"；
        3. 超时/缓冲后**硬 cancel 兜底**（测试的 stub 循环、失控任务不配合停止时
           不会让 stop() 永久挂起）。旧实现只硬 cancel：训练本体在 executor 线程
           照跑，但协程后处理段（save_agent/register/_log_round）被丢——这正是
           "训练+写盘必然完整执行"旧注释不成立之处。
        """
        self._running = False
        self._stop_event.set()
        # 只在实际有训练在跑（_training_done 处于 cleared）时等训练本体 + 写盘缓冲；
        # 无训练时立即收尾（曾固定 sleep(3) —— startup/无训练的 stop 被拖慢 3 秒，
        # perf_baseline：53ms → 3053ms 全是这个固定缓冲）。
        if not self._training_done.is_set():
            await self._wait_training_done(timeout=float(getattr(
                settings, "evolve_stop_training_grace", 120.0) or 120.0))
            # 写盘缓冲：训练结果已返回，给 save_agent/register/_log_round 短暂收尾时间
            await asyncio.sleep(3.0)
        if self._tasks:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._supervise_status["alive"] = False
        log.info("[evolve] 持续进化引擎已停止")

    async def _wait_training_done(self, timeout: float = 180.0) -> None:
        """等待专用训练线程池里当前训练结束（无训练在跑时立即返回）。"""
        try:
            await asyncio.to_thread(self._training_done.wait, timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            log.warning("[evolve] 等待训练线程结束超时（>%.0fs），可能仍在后台训练", timeout)

    # ---- 策略注册（训练成功 / 启动恢复共用同一套规格，保证各模块看到的口径一致） ----

    async def _upsert_ai_strategy(self, name: str, spec: dict) -> None:
        """动态策略注册结果落库（AiStrategy 表），重启后由 dynamic_store.restore_from_db 恢复。"""
        import json as _json
        from core.database import AiStrategy
        from sqlalchemy import select
        async with self.db.session() as s:
            row = (await s.execute(select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
            if row:
                row.spec_json = _json.dumps(spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=name, spec_json=_json.dumps(spec, ensure_ascii=False)))
            await s.commit()

    def _export_rl_evolve_pine(self) -> tuple[str, str]:
        """从磁盘上的 strategy_drl 最佳模型现场导出 Pine，返回 (代码, 不可导出原因)。

        为什么注册时读文件重导，而不是复用训练返回值里的 pine_code：
        1. 回退分支在写 pine_code 之前就 return 了，注册的却是历史最佳模型；
        2. save_agent_best 会重写 flat 文件，训练返回值可能已不是当前最佳；
        3. 启动恢复路径（_restore_evolve_strategies）根本没有训练返回值。
        flat 文件才是"部署端与回测端实际跑的那个模型"，从它导出才能保证
        图上买卖点与实盘行为同源。装不进 Pine 时留空 + 写明原因，
        由前端显式提示——绝不回落到别的策略模板（那是假代码）。
        """
        from .pine_export import export_pine_from_model
        return export_pine_from_model(self.zoo._flat_path("strategy_drl"), "rl_evolve")

    async def _register_rl_evolve(self) -> None:
        """把当前 strategy_drl 最佳（flat）模型注册为 rl_evolve 策略并落库。"""
        episodes = getattr(settings, "evolve_strategy_drl_episodes", 8)
        # Pine 导出是同步文件读 + 字符串拼装（潜在几百 KB），放线程池防阻塞事件循环
        pine_code, pine_note = await asyncio.to_thread(self._export_rl_evolve_pine)
        if pine_note:
            log.warning("[evolve] rl_evolve 无法导出 Pine：%s", pine_note)
        spec = {
            "name": "rl_evolve",
            "title": f"RL持续进化·{settings.default_symbol}",
            "description": f"持续进化引擎训练的强化学习策略（{episodes}轮/轮）",
            "logic": "MDP建模+策略梯度训练，持续进化自动续训",
            "executor": "rl_adaptive",
            "param_schema": RLAdaptiveStrategy.param_schema,
            "params": {"model_path": str(self.zoo._flat_path("strategy_drl")), "buy_zone": 0.02},
            "risk_tips": ["持续进化模型，自动回退保护"],
            "created_by": "evolve_engine",
            "version": f"v{self.zoo.best_version('strategy_drl') or 0}",
            "base_symbol": settings.default_symbol,
            "base_timeframe": self._effective_timeframe(),
            "pine_code": pine_code,
            "pine_note": pine_note,
            # 级联因子模型无法导出 Pine → 显式标记"仅软件内运行"，前端据此显示徽标
            #（诊断报告建议第5条(b)：不假装可以上图，也不回落假模板）。
            "internal_only": bool(pine_note),
        }
        register_dynamic("rl_evolve", spec)
        await self._upsert_ai_strategy("rl_evolve", spec)
        log.info("[evolve] 策略 %s 已注册（pine %s）",
                 "rl_evolve", f"{len(pine_code)}B" if pine_code else "不可导出")

    # ---- 进化组合因子策略部署（Task：进化因子使用场景）----

    def _combo_strategy_name(self, symbol: str) -> str:
        """进化组合因子策略名：evolve_combo_<symbol>，/ 替换为 _，与级联文件命名一致）。"""
        return f"evolve_combo_{symbol.replace('/', '_')}"

    async def _deploy_factor_strategy(self, symbol: str, weights: dict,
                                      report: Optional[dict],
                                      meta: Optional[dict] = None) -> dict:
        """把进化产出的组合因子权重部署为 factor_signal 组合策略（幂等覆盖）。
        spec 落 AiStrategy 表（重启后由 dynamic_store.restore_from_db 恢复）；
        同标的重复部署 = 覆盖更新同名 spec，version 递增。返回 {name, version, spec}。
        """
        from strategies.factor_signal import FactorSignalStrategy
        name = self._combo_strategy_name(symbol)
        prev = get_dynamic(name)
        prev_version = 0
        if prev and prev.get("version"):
            try:
                prev_version = int(str(prev["version"]).lstrip("v"))
            except ValueError:
                prev_version = 0
        version = f"v{prev_version + 1}"
        spec = {
            "name": name,
            "title": f"进化组合因子·{symbol}",
            "description": f"持续进化引擎因子挖掘（factor_miner）产出的组合因子策略（{symbol}）",
            "logic": "RL因子挖掘选出因子组合，按IC方向加权合成，z-score后与阈值比较产生买卖信号",
            "executor": "factor_signal",
            "param_schema": FactorSignalStrategy.param_schema,
            "params": {
                "factor": "combo",
                "combo_spec": json.dumps(weights, ensure_ascii=False),
                "mode": "trend",
                "buy_threshold": 0.0,
                "sell_threshold": 0.0,
            },
            "risk_tips": ["因子IC可能衰减，因子库下线后该因子自动跳过"],
            "created_by": "evolve_engine",
            "version": version,
            "base_symbol": symbol,
            "base_timeframe": self._effective_timeframe(),
            "evolve_meta": {
                "fitness": (meta or {}).get("fitness"),
                "oos_report": report,
                "selected_factors": (meta or {}).get("selected_factors", []),
                "deployed_at": time.time(),
                "round_no": (meta or {}).get("round_no"),
                "backtest": None,
            },
        }
        register_dynamic(name, spec)
        await self._upsert_ai_strategy(name, spec)
        log.info("[evolve] 进化组合因子策略 %s %s 已部署（%d 个因子）",
                 name, version, len(weights))
        return {"name": name, "version": version, "spec": spec}

    async def _refresh_factor_strategy_backtest(self, name: str, symbol: str,
                                                df, version: str) -> None:
        """后台跑一次部署策略的回测，把摘要写回 factor_miner 状态（仅展示用）。

        用训练轮同一份 df（rolling_window 根K线）+ 部署策略参数回测。
        - 优化（自测实验，实测数据）：摘要回测耗时随窗口线性增长（5m 上 1000 根
          1.9s / 3000 根 5.9s / 5000 根 10s），纯展示用途可截断窗口——settings
          evolve_factor_summary_bars>0 时只回测最近 N 根（默认 0=全窗口，语义不变）。
        - bootstrap 关掉（实测仅 77ms/2% 开销，摘要无需置信区间，省一轮重采样）。
        跨轮竞态守卫：写回时校验 combo_strategy/combo_version 仍是本轮部署
        （防旧轮慢任务清掉新一轮部署错误、用旧摘要覆盖新摘要）；不归属则丢弃。
        异常自吞，只记独立键 combo_backtest_error（与部署语义 combo_deploy_error 分离）。
        """

        def _owned_now() -> bool:
            # 状态必须在写回时刻读取：任务可能跨过多轮部署，开头读一次不够
            cur_name = self._factor_miner_status.get("combo_strategy")
            cur_ver = self._factor_miner_status.get("combo_version")
            return cur_name == name and cur_ver == version

        try:
            from backtest.engine import BacktestConfig, run_backtest
            _summary_bars = int(getattr(settings, "evolve_factor_summary_bars", 0) or 0)
            if _summary_bars > 0 and df is not None and len(df) > _summary_bars:
                df = df.iloc[-_summary_bars:]
            spec = get_dynamic(name) or {}
            cfg = BacktestConfig(
                symbol=symbol,
                timeframe=self._effective_timeframe(),
                strategy_name=name,
                strategy_params=dict(spec.get("params", {})),
                start_cash=10000.0, fee_rate=0.001, slippage=0.0005,
            )
            res = await asyncio.to_thread(run_backtest, df, cfg, bootstrap=False)
            m = res.get("metrics") or {}
            summary = {
                "total_ret": float(m.get("total_return", 0.0)),
                "max_drawdown": float(m.get("max_drawdown", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "trades": int(m.get("total_trades", 0)),
                "benchmark_ret": float((res.get("benchmark") or {}).get("buy_hold_ret", 0.0)),
                "at": time.time(),
            }
            if _owned_now():
                self._factor_miner_status["combo_backtest"] = summary
                self._factor_miner_status["combo_backtest_error"] = ""
                log.info("[evolve] %s 回测摘要已刷新: %s", name, summary)
            else:
                log.debug("[evolve] %s %s 回测摘要过期（当前部署非本轮），丢弃", name, version)
        except Exception as e:  # noqa: BLE001
            if _owned_now():
                self._factor_miner_status["combo_backtest_error"] = f"回测摘要失败: {e}"
                log.warning("[evolve] %s 回测摘要失败: %s", name, e)
            else:
                log.debug("[evolve] %s %s 回测失败但部署已换代，丢弃错误", name, version)

    def _compute_combo_factor_health(self, df, weights: dict) -> dict:
        """已部署组合因子的近期 IC 健康度（第3项完善：IC 衰变监控/上线质检）。

        对每个权重因子在最近非重叠 60 根窗口上算滚动 rank IC 并取最新值；
        组合 IC = 按权重方向加权的因子 IC 均值（仅评估/展示用，非回测指标）。
        decayed：|组合 IC| < 0.02（与 OOS 安检 |rank_ic|>=0.03 同量级）时标记，
        提示该组合在当前数据窗口信号已弱化、应关注或重新训练。
        """
        from factors.analysis import _rolling_ic
        from factors.library import get_factor
        close = df["close"]
        rows = []
        total = 0.0
        n_used = 0
        for key, w in (weights or {}).items():
            f = get_factor(str(key))
            if f is None:
                continue
            try:
                s = f.series(df).astype(float)
            except Exception:  # noqa: BLE001
                continue
            ic_series = _rolling_ic(s, close, h=1, method="rank", window=60)
            ic_val = None
            for v in reversed(ic_series.tolist()):
                if v == v:  # 取最近一个非 NaN 窗口的 IC
                    ic_val = float(v)
                    break
            rows.append({"factor": str(key), "weight": round(float(w), 4), "ic": ic_val})
            if ic_val is not None:
                total += float(w) * ic_val
                n_used += 1
        combo_ic = (total / max(1, n_used)) if n_used else None
        return {
            "factors": rows,
            "combo_ic": combo_ic,
            "n_used": n_used,
            "decayed": bool(combo_ic is not None and abs(combo_ic) < 0.02),
            "at": time.time(),
        }

    def _refresh_deployed_combo_health(self, df) -> None:
        """用最新数据刷新当前已部署组合因子的健康度（第3项完善）。

        训练轮各出口（通过/被拒/回退）都应调用——IC 衰变监控要覆盖
        "连续被拒期间健康度持续下降"这一核心场景，而非只在部署成功时更新。
        异常自吞，不影响训练流程。
        """
        _deployed = self._factor_miner_status.get("combo_strategy") or ""
        if not _deployed:
            return
        try:
            from strategies import get_dynamic as _gd
            _cs = ((_gd(_deployed) or {}).get("params") or {}).get("combo_spec") or ""
            if _cs:
                import json as _json
                self._factor_miner_status["combo_health"] = self._compute_combo_factor_health(
                    df, _json.loads(_cs))
        except Exception as e:  # noqa: BLE001
            log.debug("[evolve] 组合因子健康度刷新跳过: %s", e)

    async def _restore_evolve_strategies(self) -> None:
        """启动时按已训练模型补齐动态策略注册（幂等）。

        此前 rl_evolve 只在「训练成功且未被回退/拦截」时注册——一旦多轮回退
        （新模型 fitness 低于历史最佳×阈值即回退，是常态），训练产物永远不出现在
        策略库，其他模块（回测/实盘/统一策略）无法使用持续进化成果。
        """
        from strategies import get_dynamic
        if self.zoo._flat_path("strategy_drl").exists() and get_dynamic("rl_evolve") is None:
            await self._register_rl_evolve()

    async def _restore_combo_deploy_status(self) -> None:
        """启动时恢复进化组合因子的部署状态键（combo_strategy 等）。

        已部署策略持久化在 AiStrategy 表（启动时由 dynamic_store.restore_from_db
        先恢复进注册表），但 combo_* 状态键是进程内存态——重启后为空，进化面板会
        误显示「尚未部署」、无「去回测」按钮，直到下一轮安检通过的训练才纠正。
        这里从注册表的最新部署（evolve_meta.deployed_at 最大者）恢复，幂等只填空。
        """
        from strategies import dynamic_names, get_dynamic
        try:
            candidates = []
            for name in dynamic_names():
                if not name.startswith("evolve_combo_"):
                    continue
                spec = get_dynamic(name) or {}
                if spec.get("created_by") != "evolve_engine":
                    continue
                em = spec.get("evolve_meta") or {}
                candidates.append((name, str(spec.get("version", "")),
                                   em.get("deployed_at"), name))
            if not candidates:
                return
            candidates.sort(key=lambda c: (c[2] or 0.0), reverse=True)
            name, version, deployed_at, _ = candidates[0]
            if not self._factor_miner_status.get("combo_strategy"):
                self._factor_miner_status["combo_strategy"] = name
                self._factor_miner_status["combo_version"] = version
                if deployed_at:
                    self._factor_miner_status["combo_deployed_at"] = float(deployed_at)
                log.info("[evolve] 已恢复组合因子部署状态：%s %s", name, version)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 恢复组合因子部署状态失败（不影响启动）: %s", e)

    async def _restore_episode_counts(self) -> None:
        """重启后从 evolve_rounds 恢复各管线已完成轮次（episode 计数）。

        训练轮次是累计事实：持续进化跨会话/跨重启保持训练时，轮次显示应接续
        而不是归零。数据库 EvolveRound 表以 round_no 顺序落库每轮结果，这里取
        各模型的最大 round_no 作为已完成的轮次数（无记录则保持 0）。
        """
        from sqlalchemy import func, select
        from core.database import EvolveRound
        try:
            async with self.db.session() as s:
                rows = (await s.execute(
                    select(EvolveRound.model, func.max(EvolveRound.round_no))
                    .group_by(EvolveRound.model))).all()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 读取历史轮次失败（轮次从 0 开始）: %s", e)
            return
        counts = {model: (max_round or 0) for model, max_round in rows}
        self._factor_miner_status["episode"] = counts.get("factor_miner", 0)
        self._strategy_drl_status["episode"] = counts.get("strategy_drl", 0)
        # 连续拒绝数同样要跨重启接续：锁死是"上一轮还没解开"的状态，
        # 重启不该让它看起来像刚恢复正常
        try:
            # 锚点重置是账界线：重置前的连续拒绝针对的是已作废的基线，重启后不该把它
            # 们算回来——否则横幅会显示一个"没人能追平"的 streak（现场实测 35/10）。
            # EvolveRound.ts 是 SQLite CURRENT_TIMESTAMP（naive UTC），故这里转成同基准。
            cutoffs: dict[str, Any] = {}
            for m in self._models:
                rst = (self.zoo.best_info(m).get("reset") or {})
                if rst.get("timestamp"):
                    cutoffs[m] = datetime.fromtimestamp(
                        rst["timestamp"], tz=timezone.utc).replace(tzinfo=None)
            async with self.db.session() as s:
                # P2-9：按模型分查 + LIMIT（连续 streak 在第 1 条 ok/锚点重置处即终止，
                # 每模型最多需要 stall_rounds+1 条；此前无 WHERE/LIMIT 全表载入，长期
                # 运行后重启耗时会随 evolve_rounds 表增长）
                hist: list[Any] = []
                for m in self._models:
                    rows_m = (await s.execute(
                        select(EvolveRound.model, EvolveRound.status, EvolveRound.round_no,
                               EvolveRound.ts)
                        .where(EvolveRound.model == m)
                        .order_by(EvolveRound.round_no.desc())
                        .limit(int(self._stall_rounds) + 1))).all()
                    hist.extend(rows_m)
            seen: set[str] = set()
            streaks: dict[str, int] = {m: 0 for m in self._models}
            for model, st, _rn, ts in hist:  # 已按 round_no 倒序
                if model not in streaks or model in seen:
                    continue
                cutoff = cutoffs.get(model)
                if cutoff is not None and ts is not None and ts < cutoff:
                    seen.add(model)
                    continue
                if st == "ok":
                    seen.add(model)
                    continue
                if st == "demo_blocked":
                    # run24 修复：与运行时 _log_round 口径一致——demo_blocked
                    # 行不计入连续拒绝（P1-3 修复只覆盖了运行时路径，重启恢复
                    # 曾把交易所故障期计入 streak，误报 stalled/诱导重置锚点）
                    seen.add(model)
                    continue
                streaks[model] += 1
            self._reject_streak.update(streaks)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 读取连续拒绝历史失败: %s", e)
        log.info("[evolve] 已恢复训练轮次: 因子挖掘=%d 策略DRL=%d"
                 "（连续未接受: %s）",
                 self._factor_miner_status["episode"],
                 self._strategy_drl_status["episode"],
                 "/".join(str(self._reject_streak.get(m, 0)) for m in self._models))

    async def set_enabled(self, enabled: bool) -> None:
        """启动/禁用持续进化（运行时热切换）。"""
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled and self._running:
            # 从禁用状态切换为启用：补齐缺失的训练循环。
            # P4-A2：两条管线都要恢复（此前遗漏一条 → 热切换后
            # 该管线永久停训，直到进程重启）。现在改为"缺哪条补哪条"，
            # 判定不再依赖 `not self._tasks`——保活监督自身也占一个任务槽，
            # 拿列表空否当开关会让热切换漏建整组循环。
            # 2026-10 审计修复（留档项）：重建停止事件，让新循环可运行。
            self._stop_event = asyncio.Event()
            created = self._spawn_loops()
            self._spawn_supervisor()
            self._supervise_status["alive"] = True
            if created:
                log.info("[evolve] 持续进化已启用（补齐 %d 条训练循环，保活监督已就位）", created)
        elif not enabled:
            # 禁用（2026-10 审计修复留档项）：**优雅停止**——置停止事件让循环在
            # 当前轮完整结束（含写盘/注册/落库）后自然退出；等训练本体 + 写盘缓冲
            # 后硬 cancel 兜底（stub/失控任务不配合时不挂起）。曾只硬 cancel：只
            # 中断协程，executor 线程跑完训练但后处理段写入可能丢失。
            self._enabled = False
            self._stop_event.set()
            # 仅实际有训练在跑时才等训练本体 + 写盘缓冲（无训练立即收尾，见 stop()）
            if not self._training_done.is_set():
                await self._wait_training_done(timeout=float(getattr(
                    settings, "evolve_stop_training_grace", 120.0) or 120.0))
                await asyncio.sleep(3.0)
            if self._tasks:
                for t in self._tasks:
                    t.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            self._supervise_status["alive"] = False
            log.info("[evolve] 持续进化已禁用")

    async def pause(self) -> None:
        """暂停训练（当前轮完成后停止新轮）。"""
        self._paused = True
        log.info("[evolve] 持续训练已暂停")

    async def resume(self) -> None:
        """恢复训练。"""
        self._paused = False
        log.info("[evolve] 持续训练已恢复")

    async def trigger_factor_miner(self) -> dict:
        """手动触发一次因子挖掘训练（同步执行，不阻塞太久需容忍）。"""
        if not self._running:
            return {"ok": False, "reason": "持续进化引擎未运行"}
        # 2026-10 审计修复（留档项）：停止/禁用状态下手动触发拒绝（同 trigger_strategy_drl）
        if self._stop_event.is_set():
            return {"ok": False, "reason": "持续进化已停止/禁用，无法手动触发"}
        # P2：忙态守卫——另一条训练正在进行时直接拒绝（此前连点会在
        # _train_lock 上排队，连续执行数轮完整训练）
        if self._train_lock.locked():
            return {"ok": False, "busy": True, "reason": "另一条训练正在进行中，请稍后再试"}
        async with self._train_lock:
            self._factor_miner_status["active"] = True
            try:
                self._factor_miner_status["last_error"] = ""
                _res = await self._train_factor_miner_once(force=True)
                self._factor_miner_status["last_run"] = time.time()
                if _res == "demo":
                    # 演示数据轮：降级不算成功（last_error 已写明故障）
                    self._factor_miner_status["last_success"] = False
                    return {"ok": True, "model": "factor_miner", "demo": True}
                self._factor_miner_status["last_success"] = True
                return {"ok": True, "model": "factor_miner"}
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)[:200]
                self._factor_miner_status["last_error"] = err
                self._factor_miner_status["last_success"] = False
                return {"ok": False, "error": err}
            finally:
                self._factor_miner_status["active"] = False

    async def trigger_strategy_drl(self) -> dict:
        """手动触发一次策略 DRL 训练（同步执行）。"""
        if not self._running:
            return {"ok": False, "reason": "持续进化引擎未运行"}
        # 2026-10 审计修复（留档项）：停止/禁用状态下手动触发拒绝——
        # 曾 stop() 不取消手动触发任务，禁用返回后手动轮仍可能继续部署写盘。
        if self._stop_event.is_set():
            return {"ok": False, "reason": "持续进化已停止/禁用，无法手动触发"}
        # P2：忙态守卫——另一条训练正在进行时直接拒绝（防连点排队执行多轮）
        if self._train_lock.locked():
            return {"ok": False, "busy": True, "reason": "另一条训练正在进行中，请稍后再试"}
        async with self._train_lock:
            self._strategy_drl_status["active"] = True
            try:
                self._strategy_drl_status["last_error"] = ""
                _res = await self._train_strategy_drl_once(force=True)
                # 2026-10 审计修复（LOW）：手动触发也推进"全管线完成才切换标的"记账
                # （曾只有 _run_loop 调用 _mark_pipeline_done——频繁手动触发时当前标的
                #  的 cycle 记账滞后，标的轮换被卡住）。
                self._mark_pipeline_done("strategy_drl")
                self._strategy_drl_status["last_run"] = time.time()
                if _res == "demo":
                    self._strategy_drl_status["last_success"] = False
                    return {"ok": True, "model": "strategy_drl", "demo": True}
                self._strategy_drl_status["last_success"] = True
                return {"ok": True, "model": "strategy_drl"}
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)[:200]
                self._strategy_drl_status["last_error"] = err
                self._strategy_drl_status["last_success"] = False
                return {"ok": False, "error": err}
            finally:
                self._strategy_drl_status["active"] = False

    def _pipeline_view(self, model: str, base: dict, anchor: dict) -> dict:
        """管线状态 + 锁死自检（episode 只说明"训了几轮"，不说明"有没有一轮被接受"）。

        P4-E1：anchor 由 status() 预取传入——每个模型每轮 status() 只调一次
        best_info，此前 _pipeline_view 内又读一次（前端秒级轮询时重复磁盘 I/O）。
        """
        out = dict(base)
        streak = int(self._reject_streak.get(model, 0))
        out["reject_streak"] = streak
        out["rounds_total"] = int(self._rounds_total.get(model, 0))
        stalled = streak >= self._stall_rounds
        out["stalled"] = stalled
        # 真锁死 = 连续未接受 且 被一份"不可能追上"的基线压住。
        # 只有这种情况该给「重置锚点」；同口径下的连续拒绝是正常保护，
        # 而无锚点时回退守卫根本不生效（此时 streak 只是重置前的历史）。
        out["anchor_locked"] = bool(stalled and anchor.get("fitness") is not None
                                    and not anchor.get("comparable"))
        out["stall_reason"] = ""
        # 部署/候选画像：candidate vs deployed 分离（Task 2 契约）
        _ds = self._deploy_status.get(model, {})
        out["candidate_fitness"] = _ds.get("candidate_fitness")
        out["deployed_fitness"] = _ds.get("deployed_fitness")
        out["deployed_version"] = _ds.get("deployed_version")
        out["deployed_source"] = _ds.get("deployed_source") or ""
        out["last_outcome"] = _ds.get("last_outcome") or ""
        out["anchor_mismatch"] = _ds.get("anchor_mismatch") or ""
        out["runtime_reload"] = _ds.get("runtime_reload") or "skipped"
        if stalled:
            _anchor_txt = ("无锚点" if anchor.get("fitness") is None
                           else f"fitness={anchor['fitness']}"
                                f"({'demo 合成数据' if anchor.get('data_source') == 'demo' else '真实数据'})")
            if anchor.get("fitness") is None:
                out["stall_reason"] = (f"当前无回退基线（已重置或尚无模型通过检验），"
                                       f"此前连续 {streak} 轮未接受不计入锁死判定；"
                                       f"下一轮起重新建立锚点")
            elif not anchor.get("comparable"):
                out["stall_reason"] = (f"连续 {streak} 轮无模型被接受：{anchor.get('reason') or '锚点不可比'}"
                                       f"（当前锚点 {_anchor_txt}）")
            else:
                out["stall_reason"] = (f"连续 {streak} 轮无模型被接受，均低于当前锚点"
                                       f"（{_anchor_txt}）或被 OOS 硬门拦截")
        return out

    def _mark_deployment_status(self, model: str, *, outcome: str = "",
                                candidate_fitness: Optional[float] = None,
                                runtime_reload: str = "skipped",
                                anchor_mismatch: str = "") -> None:
        """同步管线部署画像：deployed_* 一律取自 best 快照（zoo 为准），
        candidate_fitness 由训练轮单独写入，二者分离防止回退轮状态污染。"""
        status = self._deploy_status.setdefault(model, {
            "candidate_fitness": None, "deployed_fitness": None,
            "deployed_version": None, "deployed_source": "",
            "last_outcome": "", "anchor_mismatch": "",
            "runtime_reload": "skipped",
        })
        if outcome:
            status["last_outcome"] = outcome
        if candidate_fitness is not None:
            status["candidate_fitness"] = candidate_fitness
        if runtime_reload != "skipped":
            status["runtime_reload"] = runtime_reload
        if anchor_mismatch:
            status["anchor_mismatch"] = anchor_mismatch
        info = self.zoo.best_info(model)
        status["deployed_version"] = info.get("version")
        status["deployed_fitness"] = info.get("fitness")
        status["deployed_source"] = info.get("data_source") or ""

    async def _register_deployed_model(self, name: str) -> None:
        """注册当前部署模型为动态策略（与部署快照同源）。

        注：元策略训练/部署分支已移除（2026-09-28）——meta_controller 不再是
        进化模型（deploy_model 按 _models 拦截），既有的已注册策略与模型文件
        不受影响，运行时仍可按 model_path 加载使用。
        """
        if name == "strategy_drl":
            await self._register_rl_evolve()

    async def _audit_manual_rollback(self, name: str, version: int, *, reason: str,
                                     runtime_reload: dict, registered: bool) -> None:
        """手动回退的审计轮次：目标/源版本、原因、运行时结果（不阻断主流程）。"""
        try:
            await self._log_round(
                name, "", self._effective_timeframe(), "manual", 0.0,
                status="manual_rollback",
                audit={"target_version": version,
                       "reason": reason or "",
                       "runtime_reload": runtime_reload.get("status", "skipped"),
                       "registered": registered})
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 手动回退审计落库失败(%s v%d): %s", name, version, e)

    async def deploy_model(self, name: str, version: int, *,
                           outcome: str, symbol: str = "",
                           timeframe: str = "", reason: str = "") -> dict:
        """统一部署入口：磁盘快照恢复 → 动态策略注册 → 运行时 reload。

        自动接受、自动回退、手动回退共用本入口。任何一层失败都不伪报成功：
        reload 失败时尝试恢复部署前的 best 快照，仍失败则保留现场并返回错误。
        _train_lock 由调用方在必要时持有；本方法不重复加锁。
        """
        if name not in self._models:
            return {"ok": False, "error": f"未知模型 {name}，可选：{'/'.join(self._models)}",
                    "version": version, "meta": {}, "runtime_reload": {},
                    "registered": False}
        zoo = self.zoo
        old_best = zoo.best_version(name)
        try:
            restored = await asyncio.to_thread(
                zoo.restore_deployment, name, version,
                reason=reason or outcome, outcome=outcome)
        except Exception as e:  # noqa: BLE001
            self._mark_deployment_status(name, outcome="failed", runtime_reload="failed")
            log.warning("[evolve] 部署快照恢复失败(%s v%d): %s", name, version, e)
            return {"ok": False, "version": version, "meta": {},
                    "runtime_reload": {"status": "failed", "error": str(e)},
                    "registered": False, "error": str(e)}
        deployed_meta = restored.get("meta") or {}
        registered = True
        try:
            await self._register_deployed_model(name)
        except Exception as e:  # noqa: BLE001
            registered = False
            log.warning("[evolve] 部署后策略注册失败(%s): %s", name, e)

        if self._deployment_reload is None:
            self._mark_deployment_status(name, outcome=outcome,
                                         runtime_reload="skipped")
            if outcome == "manual_rollback":
                await self._audit_manual_rollback(
                    name, version, reason=reason,
                    runtime_reload={"status": "skipped"}, registered=registered)
            log.info("[evolve] 部署完成(%s v%d, 无运行时回调, outcome=%s)",
                     name, version, outcome)
            return {"ok": True, "version": version, "meta": deployed_meta,
                    "runtime_reload": {"status": "skipped"},
                    "registered": registered, "error": None}
        try:
            result = await self._deployment_reload(name, version, deployed_meta)
        except Exception as e:  # noqa: BLE001
            result = {"status": "failed", "error": str(e)}
        if result and result.get("status") == "reloaded":
            self._mark_deployment_status(name, outcome=outcome,
                                         runtime_reload="reloaded")
            if outcome == "manual_rollback":
                await self._audit_manual_rollback(
                    name, version, reason=reason,
                    runtime_reload=result, registered=registered)
            log.info("[evolve] 部署完成(%s v%d, 运行实例已热加载, outcome=%s)",
                     name, version, outcome)
            return {"ok": True, "version": version, "meta": deployed_meta,
                    "runtime_reload": result, "registered": registered,
                    "error": None}
        # reload 失败：恢复到部署前的版本，不留下"文件已回退、运行时未回退"
        # 的半成功状态。恢复失败时保留现场并如实报告。
        runtime_status = dict(result or {"status": "failed"})
        if old_best and old_best != version:
            try:
                await asyncio.to_thread(
                    zoo.restore_deployment, name, old_best,
                    reason=f"reload失败恢复({runtime_status.get('error', '')})",
                    outcome=outcome)
            except Exception as e:  # noqa: BLE001
                log.error("[evolve] reload 失败后恢复 %s v%d 也失败: %s",
                          name, old_best, e)
        self._mark_deployment_status(name, outcome="failed",
                                     runtime_reload="failed")
        if outcome == "manual_rollback":
            await self._audit_manual_rollback(
                name, version, reason=reason,
                runtime_reload=runtime_status, registered=registered)
        log.warning("[evolve] 部署失败(%s v%d): reload 未成功 %s",
                    name, version, runtime_status)
        return {"ok": False, "version": version, "meta": deployed_meta,
                "runtime_reload": runtime_status, "registered": registered,
                "error": runtime_status.get("error") or "运行时模型热加载未完成"}

    def status(self) -> dict:
        """返回当前训练状态。"""
        tasks_alive = {t.get_name(): not t.done() for t in self._tasks}
        # P4-E1：锚点画像只读一次（_pipeline_view 与 anchor 字段共用），
        # 消除秒级轮询下的重复 meta.json 磁盘读
        anchors = {m: self.zoo.best_info(m) for m in self._models}
        # 保活视角：应当存活的训练循环里，有哪几条已经不在了。
        # 「引擎开着但已停训」此前完全不可观测（tasks_alive 算了却无人消费），
        # 这里把它升级为带判定的健康信号，前端/AI 都能据此判断是否真的在跑。
        _now = time.time()
        loops_alive: dict[str, bool] = {}
        for _n in self._loop_names:
            _t = self._find_task(_n)
            loops_alive[_n] = bool(_t is not None and not _t.done())
        _expected = bool(self._running and self._enabled)
        _last_check = float(self._supervise_status.get("last_check") or 0.0)
        supervisor = {
            **self._supervise_status,
            "loops_alive": loops_alive,
            "restarts_by_loop": dict(self._restarts),
            "heartbeat_age": (_now - _last_check) if _last_check else None,
            "expected_running": _expected,
            "degraded": bool(_expected and not all(loops_alive.values())),
        }
        return {
            "running": self._running,
            "paused": self._paused,
            "enabled": self._enabled,
            "symbols": self._symbols,
            "current_symbol": self._symbols[self._symbol_idx % len(self._symbols)],
            "timeframe": self._effective_timeframe(),
            "factor_miner": self._pipeline_view("factor_miner", self._factor_miner_status, anchors["factor_miner"]),
            "strategy_drl": self._pipeline_view("strategy_drl", self._strategy_drl_status, anchors["strategy_drl"]),
            # 锚点（回退基线）画像：fitness 数值 + 口径凭证，前端据此解释"为什么不接受"
            "anchor": anchors,
            "stall_rounds": self._stall_rounds,
            "factor_miner_interval": self._factor_miner_interval,
            "strategy_drl_interval": self._strategy_drl_interval,
            "rolling_window": self._rolling_window,
            "rollback_threshold": self._rollback_threshold,
            "tasks_alive": tasks_alive,
            # 保活状态：巡检心跳、各循环存活情况、自愈次数与降级判定
            "supervisor": supervisor,
        }

    async def reset_anchor(self, name: str, reason: str = "") -> dict:
        """重置锚点（归档式，不删权重）：解锁被不可比基线锁死的进化循环。

        P4-E1：与训练写路径互斥（_train_lock）——否则 reset_anchor 读 meta 后
        后台训练又写回 meta（追加版本记录），旧快照 replace 覆盖导致版本记录
        丢失更新；zoo.reset_anchor 含 gzip/复制写盘，放 to_thread 防阻塞事件循环。
        """
        if name not in self._models:
            return {"ok": False, "error": f"未知模型 {name}，可选：{'/'.join(self._models)}"}
        async with self._train_lock:
            try:
                out = await asyncio.to_thread(self.zoo.reset_anchor, name, reason=reason)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 锚点重置失败(%s): %s", name, e)
                return {"ok": False, "error": str(e)}
        self._reject_streak[name] = 0
        try:
            from core.notify import notify
            cleared = out.get("cleared") or {}
            await notify(self.db, "进化引擎：锚点已重置",
                         f"{name} 的回退基线已清除（原 fitness={cleared.get('fitness')} "
                         f"来源={cleared.get('data_source') or '未知'}），"
                         f"下一条通过 OOS 检验的模型将重建基线。留档：{out.get('stamp')}")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, **out, "anchor": self.zoo.best_info(name)}

    @property
    def config_keys(self) -> dict:
        """当前可配置项（P2-12 超参面板 + 「指定品种/周期」落地。训练循环读的是
        self 缓存属性与 settings 字段，这里统一收口作为面板编辑的数据源）。"""
        return {
            "symbols": list(self._symbols),
            "timeframe": getattr(settings, "evolve_timeframe", "") or settings.default_timeframe,
            "factor_miner_interval": int(self._factor_miner_interval),
            "strategy_drl_interval": int(self._strategy_drl_interval),
            "rolling_window": int(self._rolling_window),
            "rollback_threshold": float(self._rollback_threshold),
            "factor_miner_episodes": int(getattr(settings, "evolve_factor_miner_episodes", 8)),
            # 因子挖掘训练目标口径（2026-09-29 成本对齐修复）：net=扣成本净收益
            "factor_miner_fitness_mode": str(getattr(
                settings, "evolve_factor_miner_fitness_mode", "net") or "ic"),
            "factor_miner_cost_per_turn": float(getattr(
                settings, "evolve_factor_miner_cost_per_turn", 0.002) or 0.0),
            "factor_miner_net_fitness_scale": float(getattr(
                settings, "evolve_factor_miner_net_fitness_scale", 500.0) or 1.0),
            "strategy_drl_episodes": int(getattr(settings, "evolve_strategy_drl_episodes", 8)),
            "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
            "oos_min_bars": int(getattr(settings, "evolve_oos_min_bars", 250)),
            "cross_symbol_oos": bool(getattr(settings, "evolve_cross_symbol_oos", True)),
            "min_new_bars": int(getattr(settings, "evolve_min_new_bars", 2)),
            "train_time_budget": float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0),
            "min_train_gap_sec": int(getattr(settings, "evolve_min_train_gap_sec", 0) or 0),
            # 第2项完善：连续 OOS 被拒降频阈值（可热调，0=关闭）
            "reject_downshift_after": int(getattr(settings, "evolve_reject_downshift_after", 6) or 0),
            "train_on_demo": bool(getattr(settings, "evolve_train_on_demo", False)),
            "anchor_max_fitness": float(getattr(settings, "evolve_anchor_max_fitness", 50.0) or 0.0),
            "strategy_drl_population": bool(getattr(settings, "evolve_strategy_drl_population", True)),
            # 训练能力参数（n_episodes/ppo_epochs，4→8→16 实验；collect_workers 并行上量）
            "strategy_drl_n_episodes": int(getattr(settings, "evolve_strategy_drl_n_episodes", 16) or 0),
            "strategy_drl_collect_workers": int(getattr(settings, "evolve_strategy_drl_collect_workers", 0) or 0),
            "strategy_drl_ppo_epochs": int(getattr(settings, "evolve_strategy_drl_ppo_epochs", 8) or 0),
            # P1 课程学习（能力层次）：先易后难的难度分层采样
            "strategy_drl_curriculum": bool(getattr(settings, "evolve_strategy_drl_curriculum", False)),
            "strategy_drl_curriculum_start": float(getattr(
                settings, "evolve_strategy_drl_curriculum_start", 0.3) or 0.0),
            "strategy_drl_min_trade_zone": float(getattr(settings, "evolve_strategy_drl_min_trade_zone", 0.10)),
            # 经验回放（跨轮存活，对抗遗忘）
            "strategy_drl_replay_capacity": int(getattr(
                settings, "evolve_strategy_drl_replay_capacity", 0) or 0),
            "strategy_drl_replay_mix_ratio": float(getattr(
                settings, "evolve_strategy_drl_replay_mix_ratio", 0.3) or 0.0),
            # walk-forward 多折观测（同品种，不改接受逻辑）
            "strategy_drl_walkforward_obs": bool(getattr(settings, "evolve_strategy_drl_walkforward_obs", True)),
            "strategy_drl_walkforward_folds": int(getattr(settings, "evolve_strategy_drl_walkforward_folds", 2)),
            "strategy_drl_walkforward_bars": int(getattr(settings, "evolve_strategy_drl_walkforward_bars", 6000)),
            "walkforward_min_trades": int(getattr(settings, "evolve_walkforward_min_trades", 8)),
            # 连续拒绝 from-scratch 重置（0=关闭）
            "strategy_drl_reset_after_rejects": int(getattr(
                settings, "evolve_strategy_drl_reset_after_rejects", 0) or 0),
            # GAP-01 实盘验收观测（同品种单标的，两档成本）
            "gap01_obs": bool(getattr(settings, "evolve_gap01_obs", True)),
            "gap01_folds": int(getattr(settings, "evolve_gap01_folds", 2)),
            "gap01_bars": int(getattr(settings, "evolve_gap01_bars", 6000)),
            "gap01_min_trades": int(getattr(settings, "evolve_gap01_min_trades", 8)),
            "gap01_cost_multiplier": float(getattr(settings, "evolve_gap01_cost_multiplier", 2.0)),
            "gap01_fail_reset_after": int(getattr(settings, "evolve_gap01_fail_reset_after", 0) or 0),
            "gap01_stable_pass_required": int(getattr(settings, "evolve_gap01_stable_pass_required", 3) or 0),
            # 长窗回放观测（30/90 天）
            "strategy_drl_longwindow_obs": bool(getattr(settings, "evolve_strategy_drl_longwindow_obs", True)),
            "strategy_drl_longwindow_30d_bars": int(getattr(settings, "evolve_strategy_drl_longwindow_30d_bars", 8640) or 0),
            "strategy_drl_longwindow_90d_bars": int(getattr(settings, "evolve_strategy_drl_longwindow_90d_bars", 25920) or 0),
            "strategy_drl_longwindow_every_n": int(getattr(settings, "evolve_strategy_drl_longwindow_every_n", 5) or 1),
        }

    async def apply_config(self, changes: dict) -> dict:
        """热更新可配置项（不重启生效）：写引擎缓存属性 + settings 字段，
        并持久化回 config.yaml（用户在面板上改的配置重启后保留）。

        'symbols'：允许单元素列表 → 固定品种训练（替代默认轮换）。
        'timeframe'：训练时间周期（空/缺省时回落 settings.default_timeframe）。
        返回 {ok, applied: {...}, persisted: bool}。

        事务性（P1-1/P1-2 修复）：先对全部变更做校验（含 timeframe 白名单），
        全部通过后才一次性写入 settings/引擎属性并持久化；任何一项失败返回
        错误且不产生副作用——此前 symbols/timeframe 先赋值、后续字段校验失败
        不回滚，造成「前端提示失败但引擎已在用新配置」的状态分裂。
        """
        import shutil
        from backtest.data_loader import _BINANCE_TF  # 合法周期白名单（唯一权威来源）
        applied: dict = {}
        allowed = set(self.config_keys)
        unknown = sorted(set(changes) - allowed)
        if unknown:
            return {"ok": False, "error": f"未知配置项: {', '.join(unknown)}",
                    "applied": applied}
        # 1) 全部校验 → pending 清单（不产生任何副作用）
        pending: list[tuple[str, Any, Optional[str] | None]] = []  # (settings_attr, value, engine_attr)

        # 训练标的池：支持"单元素 = 固定品种训练"。
        # P4-E1：空数组视为非法输入（清空标的池会让训练管线无从取数），
        # 明确报错而不是静默 no-op；此前 `if changes["symbols"]:` 把空池整体跳过，
        # 前端却提示"保存成功"。
        if "symbols" in changes:
            raw_symbols = changes["symbols"]
            if not isinstance(raw_symbols, list) or not raw_symbols or len(raw_symbols) > 50:
                return {"ok": False, "error": "symbols 必须是 1-50 个交易对的列表",
                        "applied": applied}
            syms = []
            for raw_symbol in raw_symbols:
                if not isinstance(raw_symbol, str):
                    return {"ok": False, "error": "symbols 必须全部是字符串",
                            "applied": applied}
                symbol = raw_symbol.strip().upper()
                if not symbol or "/" not in symbol or len(symbol) > 40:
                    return {"ok": False, "error": f"非法交易对: {raw_symbol!r}",
                            "applied": applied}
                syms.append(symbol)
            pending.append(("evolve_symbols", syms, "_symbols"))
        # 训练时间周期：independent of default_timeframe（P2-12 用户主诉求落地）。
        # P4-E1：timeframe 为 null/空串 = 前端「跟随默认」——清除独立周期、回落
        # default_timeframe。此前 str(None)="None" 恒真值，把 "none" 写进 settings，
        # 非法周期导致拉数失败退化 demo 且面板永久显示错误周期。
        # P1-2：非法周期在这里用 _BINANCE_TF 白名单拦截（此前注释声称"持久化前
        # 兜底"但全仓库根本不存在该白名单，非法值被静默持久化，重启后整个进化
        # 链路持续 demo 空转）。
        if "timeframe" in changes:
            tf_raw = changes.get("timeframe")
            tf = str(tf_raw).strip().lower() if tf_raw not in (None, "") else ""
            if tf:
                if tf not in _BINANCE_TF:
                    return {"ok": False,
                            "error": f"不支持的训练时间周期 {tf!r}，可选：{'/'.join(sorted(_BINANCE_TF))}",
                            "applied": applied}
                pending.append(("evolve_timeframe", tf, None))
            else:
                # 显式清除：回落默认周期
                pending.append(("evolve_timeframe", "", None))

        def _check(name: str, settings_attr: str, engine_attr: str = None,
                   conv=None, min_v: float = None, max_v: float = None) -> None:
            if name not in changes or changes[name] is None:
                return
            val = changes[name]
            if isinstance(val, bool) and conv is not None:
                raise ValueError(f"{name} 必须是数字，收到: {changes[name]!r}")
            if conv is not None:
                try:
                    val = conv(val)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} 必须是数字，收到: {changes[name]!r}")
            if min_v is not None and val < min_v:
                raise ValueError(f"{name} 不能小于 {min_v}，收到: {val}")
            if max_v is not None and val > max_v:
                raise ValueError(f"{name} 不能大于 {max_v}，收到: {val}")
            pending.append((settings_attr, val, engine_attr))

        try:
            _check("factor_miner_interval", "evolve_factor_miner_interval", "_factor_miner_interval",
                   conv=int, min_v=1)
            # 因子训练目标口径（net/ic 二选一；net 的 cost/scale 数值范围）
            _check("factor_miner_fitness_mode", "evolve_factor_miner_fitness_mode",
                   conv=_fitness_mode_conv)
            _check("factor_miner_cost_per_turn", "evolve_factor_miner_cost_per_turn",
                   conv=float, min_v=0.0, max_v=0.1)
            _check("factor_miner_net_fitness_scale", "evolve_factor_miner_net_fitness_scale",
                   conv=float, min_v=1.0, max_v=100000.0)
            _check("strategy_drl_interval", "evolve_strategy_drl_interval", "_strategy_drl_interval",
                   conv=int, min_v=1)
            _check("rolling_window", "evolve_rolling_window", "_rolling_window",
                   conv=lambda v: max(1000, int(v)), min_v=1000)
            _check("rollback_threshold", "evolve_rollback_threshold", "_rollback_threshold",
                   conv=float, min_v=0.0, max_v=1.0)
            _check("factor_miner_episodes", "evolve_factor_miner_episodes", conv=int, min_v=1, max_v=500)
            _check("strategy_drl_episodes", "evolve_strategy_drl_episodes", conv=int, min_v=1, max_v=500)
            _check("vol_penalty", "evolve_vol_penalty", conv=float, min_v=0.0, max_v=1000.0)
            _check("oos_min_bars", "evolve_oos_min_bars", conv=int, min_v=50, max_v=100000)
            _check("min_new_bars", "evolve_min_new_bars", conv=int, min_v=0, max_v=100000)
            _check("train_time_budget", "evolve_train_time_budget", conv=float, min_v=0.0, max_v=86400.0)
            _check("min_train_gap_sec", "evolve_min_train_gap_sec", conv=int, min_v=0, max_v=604800)
            # 第2项完善：连续 OOS 被拒降频阈值（0=关闭）
            _check("reject_downshift_after", "evolve_reject_downshift_after",
                   conv=int, min_v=0, max_v=100000)
            _check("anchor_max_fitness", "evolve_anchor_max_fitness", conv=float, min_v=0.0, max_v=1e9)
            if "train_on_demo" in changes and changes["train_on_demo"] is not None:
                if not isinstance(changes["train_on_demo"], bool):
                    raise ValueError("train_on_demo 必须是布尔值")
                pending.append(("evolve_train_on_demo", changes["train_on_demo"], None))
            if "strategy_drl_population" in changes and changes["strategy_drl_population"] is not None:
                if not isinstance(changes["strategy_drl_population"], bool):
                    raise ValueError("strategy_drl_population 必须是布尔值")
                pending.append(("evolve_strategy_drl_population", changes["strategy_drl_population"], None))
            _check("strategy_drl_n_episodes", "evolve_strategy_drl_n_episodes",
                   conv=int, min_v=1, max_v=256)
            _check("strategy_drl_collect_workers", "evolve_strategy_drl_collect_workers",
                   conv=int, min_v=0, max_v=64)
            if "strategy_drl_curriculum" in changes and changes["strategy_drl_curriculum"] is not None:
                if not isinstance(changes["strategy_drl_curriculum"], bool):
                    raise ValueError("strategy_drl_curriculum 必须是布尔值")
                pending.append(("evolve_strategy_drl_curriculum",
                                changes["strategy_drl_curriculum"], None))
            _check("strategy_drl_curriculum_start", "evolve_strategy_drl_curriculum_start",
                   conv=float, min_v=0.0, max_v=1.0)
            _check("strategy_drl_ppo_epochs", "evolve_strategy_drl_ppo_epochs",
                   conv=int, min_v=1, max_v=64)
            _check("strategy_drl_min_trade_zone", "evolve_strategy_drl_min_trade_zone",
                   conv=float, min_v=0.0, max_v=0.5)
            _check("strategy_drl_replay_capacity", "evolve_strategy_drl_replay_capacity",
                   conv=int, min_v=0, max_v=100000)
            _check("strategy_drl_replay_mix_ratio", "evolve_strategy_drl_replay_mix_ratio",
                   conv=float, min_v=0.0, max_v=1.0)
            if "cross_symbol_oos" in changes and changes["cross_symbol_oos"] is not None:
                if not isinstance(changes["cross_symbol_oos"], bool):
                    raise ValueError("cross_symbol_oos 必须是布尔值")
                pending.append(("evolve_cross_symbol_oos", changes["cross_symbol_oos"], None))
            # walk-forward 多折观测配置（同品种，不改接受逻辑）
            _check("strategy_drl_walkforward_folds", "evolve_strategy_drl_walkforward_folds",
                   conv=int, min_v=1, max_v=10)
            _check("strategy_drl_walkforward_bars", "evolve_strategy_drl_walkforward_bars",
                   conv=lambda v: max(2000, int(v)), min_v=2000, max_v=100000)
            _check("walkforward_min_trades", "evolve_walkforward_min_trades",
                   conv=int, min_v=0, max_v=1000)
            _check("strategy_drl_reset_after_rejects",
                   "evolve_strategy_drl_reset_after_rejects",
                   conv=int, min_v=0, max_v=100000)
            # GAP-01 实盘验收观测配置
            _check("gap01_folds", "evolve_gap01_folds", conv=int, min_v=1, max_v=10)
            _check("gap01_bars", "evolve_gap01_bars",
                   conv=lambda v: max(2000, int(v)), min_v=2000, max_v=100000)
            _check("gap01_min_trades", "evolve_gap01_min_trades",
                   conv=int, min_v=0, max_v=1000)
            _check("gap01_cost_multiplier", "evolve_gap01_cost_multiplier",
                   conv=float, min_v=1.0, max_v=10.0)
            _check("gap01_fail_reset_after", "evolve_gap01_fail_reset_after",
                   conv=int, min_v=0, max_v=100000)
            _check("gap01_stable_pass_required", "evolve_gap01_stable_pass_required",
                   conv=int, min_v=0, max_v=100000)
            if "strategy_drl_longwindow_obs" in changes and changes["strategy_drl_longwindow_obs"] is not None:
                if not isinstance(changes["strategy_drl_longwindow_obs"], bool):
                    raise ValueError("strategy_drl_longwindow_obs 必须是布尔值")
                pending.append(("evolve_strategy_drl_longwindow_obs",
                                changes["strategy_drl_longwindow_obs"], None))
            _check("strategy_drl_longwindow_30d_bars", "evolve_strategy_drl_longwindow_30d_bars",
                   conv=int, min_v=1000, max_v=200000)
            _check("strategy_drl_longwindow_90d_bars", "evolve_strategy_drl_longwindow_90d_bars",
                   conv=int, min_v=1000, max_v=400000)
            _check("strategy_drl_longwindow_every_n", "evolve_strategy_drl_longwindow_every_n",
                   conv=int, min_v=1, max_v=100000)
            if "gap01_obs" in changes and changes["gap01_obs"] is not None:
                if not isinstance(changes["gap01_obs"], bool):
                    raise ValueError("gap01_obs 必须是布尔值")
                pending.append(("evolve_gap01_obs", changes["gap01_obs"], None))
            if "strategy_drl_walkforward_obs" in changes and changes["strategy_drl_walkforward_obs"] is not None:
                if not isinstance(changes["strategy_drl_walkforward_obs"], bool):
                    raise ValueError("strategy_drl_walkforward_obs 必须是布尔值")
                pending.append(("evolve_strategy_drl_walkforward_obs",
                                changes["strategy_drl_walkforward_obs"], None))
        except ValueError as e:
            log.warning("[evolve] 配置校验拒绝: %s", e)
            return {"ok": False, "error": str(e), "applied": applied}

        # 2) 全部通过后一次性生效
        for settings_attr, val, engine_attr in pending:
            setattr(settings, settings_attr, val)
            if engine_attr is not None:
                setattr(self, engine_attr, val)
            applied[settings_attr] = val
        if "evolve_symbols" in applied:
            # 标的池变更：重置轮换起点与 cycle 记账（P4-A1）
            self._symbol_idx = 0
            self._cycle_pipelines_done.clear()
            self._last_tail_ts.clear()  # 旧键对新池无意义，防跨池误判
        if "evolve_timeframe" in applied:
            self._last_tail_ts.clear()  # 增量门键含周期，切周期后旧键作废
        # 2026-10 审计修复（MED）：影响**训练口径**的配置变更同样必须让增量门失效，
        # 否则改 seed/backbone/决策间隔/历史深度后，后台轮仍按旧口径 gating（数据尾戳
        # 未变就静默跳过，新配置要等新 K 线才生效，面板却显示"已应用"）。
        if any(k in applied for k in _TRAINING_CFG_KEYS):
            self._last_tail_ts.clear()
            log.info("[evolve] 训练口径配置变更(%s)：增量门失效，下一轮强制重训",
                     sorted(k for k in applied if k in _TRAINING_CFG_KEYS))

        # 3) 持久化回 config.yaml（扁平结构，键名与 Settings 字段同名）：
        # 逐键更新顶层文档，备份后原子写入（失败不致命，仅内存生效）。
        # P1-1：此前 open("w") 直写非原子，写一半崩溃会损坏配置；改 tmp+rename。
        persisted = False
        from config.settings import ROOT as _SETTINGS_ROOT
        yaml_cfg_path = _SETTINGS_ROOT / "config" / "config.yaml"
        if yaml_cfg_path.exists():
            import yaml
            try:
                with open(str(yaml_cfg_path), "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                doc.update(applied)
                try:
                    shutil.copy2(str(yaml_cfg_path), str(yaml_cfg_path) + ".bak")
                except Exception:  # noqa: BLE001
                    pass
                tmp_path = yaml_cfg_path.with_name(yaml_cfg_path.name + ".tmp")
                with open(str(tmp_path), "w", encoding="utf-8") as f:
                    yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
                tmp_path.replace(yaml_cfg_path)
                persisted = True
                log.info("[evolve] 配置已持久化到 config.yaml: %s", applied)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 配置持久化失败（仅内存生效）: %s", e)
        else:
            log.warning("[evolve] 未找到 config.yaml，配置仅内存生效")

        log.info("[evolve] 应用配置变更: %s", applied)
        return {"ok": True, "applied": applied, "persisted": persisted}

    def _effective_timeframe(self) -> str:
        """训练时间周期：面板独立配置优先，缺省回落 default_timeframe。"""
        return getattr(settings, "evolve_timeframe", "") or settings.default_timeframe

    # ---- 通用训练循环模板 ----

    # 训练失败后的快速重试间隔（秒）：避免失败后等 7 天/1 天才重试
    _RETRY_INTERVAL = 300  # 5 分钟

    def _run_train(self, train_callable: Callable) -> asyncio.Future:
        """在专用单线程 executor 里执行训练本体，并置位训练完成事件。

        - 单 worker 保证即便手动触发与周期循环并发，训练也绝不并行（双保险，
          与 _train_lock 同一目的）；executor 线程无法被 kill，训练本体会跑到底，
          事件标志在 finally 里置位。
        - 2026-10 审计修复（留档项）：**停止/禁用改为优雅停止**——stop()/
          set_enabled(False) 不再 cancel 协程，而是置 _stop_event 让循环在当前轮
          完整结束后自然退出。因此协程后处理段（写盘/注册/落库）与训练本体一并
          保证完整执行（旧注释"取消协程不会中断 executor 线程…写盘必然完整执行"
          不成立——取消只中断协程 await，后处理段在协程里会被丢）。
        """
        self._training_done.clear()

        def _wrap():
            try:
                return train_callable()
            finally:
                self._training_done.set()

        return asyncio.get_running_loop().run_in_executor(self._train_executor, _wrap)

    async def _run_loop(self, status: dict, train_fn: Callable, interval: int, name: str,
                        start_delay: int = 15, pipeline: str = "",
                        yield_to: tuple = ()) -> None:
        """通用训练循环：启动 start_delay 秒后首次训练，之后按间隔（失败时 5 分钟重试）。

        pipeline: 管线标识（factor_miner/strategy_drl），传入后
        在每次真实训练尝试后调用 _mark_pipeline_done——全部管线同标训练完成一轮才
        切换标的（P4-A1 同步修复）。增量门跳过（train_fn 返回 False，无训练发生）
        时不推进；异常也算一次尝试（防单管线长期失败导致标的永卡）。

        yield_to: 本管线应向哪些更高优先级管线让出锁（P4-C3）。拿锁前若任一
        更高优先级管线 active（正在训练），先让出重试——保证链路源头（因子挖掘）
        不被下游挤占，避免训练耗时波动时低优先级管线挨饿。
        """
        # 首次训练前等待：等待引擎初始化、WebSocket 建立连接、基础数据就位
        # 两个训练循环错峰启动，避免同时开始吃满 CPU
        try:
            await asyncio.sleep(start_delay)
        except asyncio.CancelledError:
            raise
        status["next_run"] = time.time() + start_delay
        status["loop_beat"] = time.time()
        while self._running and not self._stop_event.is_set():
            # 每轮先给 retry_interval 兜底初值：此前它只在 try 内赋值，若异常在
            # 赋值之前逸出，循环尾部的 `time.time() + retry_interval` 会抛
            # UnboundLocalError——而异常处理器已在 try 内，兜不住尾部，
            # 整条任务就此终结。这是"训练循环静默死亡"的一条真实路径。
            retry_interval = interval
            try:
                if self._paused:
                    await asyncio.sleep(10)
                    if not self._running:
                        break
                    continue
                # P4-C3：让出给更高优先级管线（因子挖掘 > 策略DRL）。
                # 锁本身无优先级，若不主动让出，间隔相同 + 训练耗时波动时
                # 可能长时间拿不到锁。
                _yielded = False
                if yield_to:
                    for _prio in range(12):  # 最多让 12×5s=60s
                        if not self._running:
                            break
                        higher_active = any(
                            getattr(self, f"_{p}_status", {}).get("active", False)
                            for p in yield_to)
                        if not higher_active:
                            break
                        _yielded = True
                        await asyncio.sleep(5)
                # P2-10：全局训练锁——同一时刻仅一条训练管线在跑
                async with self._train_lock:
                    status["active"] = True
                    status["last_error"] = ""
                    # P2-11：train_fn 返回 False 表示"数据无新增 K 线，本轮跳过"
                    # （增量门），不更新上次训练时间/成功标志，保持等待。
                    # "demo" = 交易所故障期的演示数据轮：训练发生了但属降级，
                    # 不记 last_success（与 last_error 并存自相矛盾），也不把
                    # demo 轮算进 reject_streak（_log_round 对 demo_blocked 已特殊处理）。
                    _res = await train_fn()
                    if _res is False:
                        log.debug("[evolve] %s：无新增K线，跳过本轮", name)
                    else:
                        status["last_run"] = time.time()
                        if _res == "demo":
                            status["last_success"] = False
                            log.warning("[evolve] %s：交易所故障，本轮为演示数据训练（不记成功）", name)
                        else:
                            status["last_success"] = True
                        # 真实训练尝试完成 → 标记当前标的该管线已训练（P4-A1）
                        if pipeline:
                            self._mark_pipeline_done(pipeline)
                # 训练轮正常结束：恢复常规间隔（失败分支在上面各自改写）
                # P4-C1：交易所持续不可达（连续 demo 训练）时自动降频——
                # 故障 1 小时若仍每 60s 全量训练合成数据，纯烧 CPU 拖垮实盘。
                # 连续 demo ≥2 次后放大到 10 分钟；恢复正常（streak 清零）即恢复。
                _demo_streak = int(status.get("_demo_streak", 0) or 0)
                if _demo_streak >= 2:
                    retry_interval = max(interval * 10, 600)
                    log.info("[evolve] %s：交易所不可达连续 %d 次，训练降频至 %ds",
                             name, _demo_streak, retry_interval)
                # 第2项完善：连续 OOS 被拒（训练正常但安检门连续拦截）自动降频——
                # 被拒说明当前数据窗口暂时供不出有效因子，60s 一轮全量重训纯烧 CPU。
                # 复用引擎级 reject_streak 计数（_log_round：ok 清零/被拒递增），
                # 阈值 evolve_reject_downshift_after 可配（0=关）。
                _reject_streak = int(self._reject_streak.get(pipeline, 0) or 0) if pipeline else 0
                _reject_floor = int(getattr(settings, "evolve_reject_downshift_after", 6) or 0)
                _downshifted = _reject_downshift_retry(retry_interval, _reject_streak, _reject_floor)
                if _downshifted != retry_interval:
                    log.info("[evolve] %s 连续 %d 轮未通过安检，训练降频至 %ds",
                             name, _reject_streak, _downshifted)
                retry_interval = _downshifted
                # P4-C4：连续失败指数退避（防"长期故障下每 5 分钟空转一次"）——
                # 失败重试随连续失败次数增长：5min → 10min → 20min → 封顶 30min；
                # 成功或跳过即清零。
                status["_fail_streak"] = 0
            except asyncio.CancelledError:
                raise
            except InsufficientDataError as e:
                # 数据不足是预期业务分支：不带全栈刷屏，仍按失败退避节奏重试
                err = str(e)[:200]
                log.warning("[evolve] %s 数据不足，进入失败退避: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                _fs = int(status.get("_fail_streak", 0) or 0) + 1
                status["_fail_streak"] = _fs
                retry_interval = min(self._RETRY_INTERVAL * (2 ** min(_fs - 1, 3)), 1800)
                log.info("[evolve] %s 连续数据不足 %d 次，%d 秒后重试", name, _fs, retry_interval)
                # 异常也算一次训练尝试（防单管线长期失败导致标的永卡，P4-A1）
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                log.exception("[evolve] %s 训练异常: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                # P4-C4：连续失败退避（封顶 30 分钟），避免长期故障下高频空转
                _fs = int(status.get("_fail_streak", 0) or 0) + 1
                status["_fail_streak"] = _fs
                retry_interval = min(self._RETRY_INTERVAL * (2 ** min(_fs - 1, 3)), 1800)
                log.info("[evolve] %s 连续失败 %d 次，%d 秒后重试", name, _fs, retry_interval)
                # 异常也算一次训练尝试（防某管线长期失败导致标的永卡，P4-A1）
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, (SystemExit, KeyboardInterrupt, GeneratorExit)):
                    raise  # 真正的退出请求不该被吞掉
                # 兜底：训练本体可能抛出非 Exception 基类的错误。此前这类异常会
                # 直接终结整条循环任务，而 _running 仍是 True——外部完全看不出
                # 停训，只能靠保活监督事后重建。这里先降级为"一轮失败"，
                # 多数情况不必动用监督。
                err = f"{type(e).__name__}: {e}"[:200]
                log.error("[evolve] %s 循环遭遇非 Exception 异常，已兜底保活: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                retry_interval = max(interval, self._RETRY_INTERVAL)
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            finally:
                status["active"] = False
            status["next_run"] = time.time() + retry_interval
            status["loop_beat"] = time.time()
            # 2026-10 审计修复（留档项）：间隔 sleep 可被停止事件提前唤醒——
            # 否则 stop() 后循环要等完整个 retry_interval（60s~30min）才退出，
            # "停止/禁用"看起来毫无反应。
            await self._sleep_or_stop(retry_interval)

    async def _sleep_or_stop(self, seconds: float) -> None:
        """sleep 与停止事件谁先到听谁（优雅停止的唤醒点；无停止时等价 asyncio.sleep）。"""
        try:
            sleep_t = asyncio.ensure_future(asyncio.sleep(max(0.0, float(seconds))))
            stop_t = asyncio.ensure_future(self._stop_event.wait())
            done, pending = await asyncio.wait({sleep_t, stop_t},
                                               return_when=asyncio.FIRST_COMPLETED)
            # 取消**所有**未完成任务（曾只 cancel done 里的任务——done 里是刚完成的
            # stop_t，而 sleep_t 留在 pending，gather 会等满整个 interval 才返回，
            # "停止即唤醒"失效）。
            for t in done:
                t.cancel()
            for t in pending:
                t.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        except asyncio.CancelledError:
            raise

    async def _factor_miner_loop(self) -> None:
        """因子挖掘持续训练循环（最高优先级：链路源头，下游依赖其产出）。"""
        await self._run_loop(self._factor_miner_status, self._train_factor_miner_once,
                             self._factor_miner_interval, "因子挖掘", pipeline="factor_miner")

    async def _strategy_drl_loop(self) -> None:
        """策略 DRL 持续训练循环（首个训练延迟 30s；因子挖掘优先，见 yield_to）。"""
        await self._run_loop(self._strategy_drl_status, self._train_strategy_drl_once,
                             self._strategy_drl_interval, "策略DRL",
                             start_delay=30, pipeline="strategy_drl",
                             yield_to=("factor_miner",))

    # ---- 保活（watchdog）：训练循环的统一创建与自愈 ----
    # 背景：训练循环是 asyncio 任务。任务一旦结束就再也不会自己回来，而
    # _running/_enabled 仍是 True，status() 照样报 running=true —— 「看起来在跑、
    # 其实已经停训」是最难发现的一类故障。此前 status() 虽然算了 tasks_alive，
    # 但全仓无人消费它（只读诊断），也没有任何重建逻辑，只能重启进程恢复。

    def _loop_factories(self) -> dict:
        """训练循环名 → 协程工厂。清单单一来源，避免多处硬编码走偏。"""
        return {
            "evolve_factor_miner": self._factor_miner_loop,
            "evolve_strategy_drl": self._strategy_drl_loop,
        }

    def _status_for(self, name: str) -> Optional[dict]:
        """训练循环名 → 对应管线的状态字典。"""
        return {
            "evolve_factor_miner": self._factor_miner_status,
            "evolve_strategy_drl": self._strategy_drl_status,
        }.get(name)

    def _find_task(self, name: str) -> Optional[asyncio.Task]:
        """按任务名查任务；_tasks 里可能同时残留已终结的旧任务。"""
        for t in self._tasks:
            if t.get_name() == name:
                return t
        return None

    def _spawn_loops(self) -> int:
        """补齐缺失的训练循环任务（幂等），返回新建条数。

        start() / set_enabled(True) / 保活监督三条路径共用同一入口（"三条"指这三条调用路径），消除
        "哪条路径漏建了哪条管线"这类只在热切换时才暴露的漏配。
        """
        factories = self._loop_factories()
        created = 0
        for name in self._loop_names:
            t = self._find_task(name)
            if t is not None:
                if not t.done():
                    continue
                # 清掉已终结的旧任务，让 _tasks 与现实保持一致
                self._tasks.remove(t)
            self._tasks.append(asyncio.create_task(factories[name](), name=name))
            created += 1
        return created

    def _spawn_supervisor(self) -> None:
        """启动保活监督任务（幂等）。"""
        t = self._find_task("evolve_supervisor")
        if t is not None:
            if not t.done():
                return
            self._tasks.remove(t)
        self._tasks.append(asyncio.create_task(self._supervise_loop(),
                                               name="evolve_supervisor"))

    async def _supervise_loop(self) -> None:
        """保活监督：周期巡检各训练循环，发现已退出的就地重建。"""
        log.info("[evolve] 保活监督已启动（巡检间隔 %.0fs，自愈最小间隔 %.0fs）",
                 self._supervise_interval, self._restart_min_gap)
        while True:
            try:
                await asyncio.sleep(self._supervise_interval)
            except asyncio.CancelledError:
                raise
            self._supervise_status["last_check"] = time.time()
            self._supervise_status["checks"] = int(self._supervise_status["checks"]) + 1
            self._supervise_status["alive"] = True
            if not self._running:
                break
            if not self._enabled:
                continue  # 用户主动禁用：不重建
            try:
                for name in self._loop_names:
                    t = self._find_task(name)
                    if t is not None and not t.done():
                        continue
                    now = time.time()
                    if now - self._last_restart.get(name, 0.0) < self._restart_min_gap:
                        # 自愈最小间隔未到：留给下一轮，防"一重建就崩"变成重启风暴
                        continue
                    await self._restart_loop(name, now)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("[evolve] 保活巡检异常（下一轮继续）")
        log.info("[evolve] 保活监督已停止")

    async def _restart_loop(self, name: str, now: float) -> None:
        """重建一条已退出的训练循环，记录自愈次数并通知（通知已去重）。"""
        dead = self._find_task(name)
        detail = "任务已丢失"
        if dead is not None:
            self._tasks.remove(dead)
            if dead.cancelled():
                detail = "任务被取消"
            elif dead.done():
                try:
                    exc = dead.exception()
                except asyncio.CancelledError:
                    exc = None
                detail = f"异常退出：{exc!r}" if exc is not None else "任务正常结束"
        try:
            self._tasks.append(
                asyncio.create_task(self._loop_factories()[name](), name=name))
        except Exception as e:  # noqa: BLE001
            log.exception("[evolve] 保活：重建 %s 失败: %s", name, e)
            return
        self._restarts[name] = int(self._restarts.get(name, 0)) + 1
        self._last_restart[name] = now
        self._supervise_status["restarts"] = sum(self._restarts.values())
        self._supervise_status["last_restart_at"] = now
        self._supervise_status["last_restart_name"] = name
        status = self._status_for(name)
        if status is not None:
            status["active"] = False
            status["restarts"] = self._restarts[name]
        log.error("[evolve] 保活：训练循环 %s 已自动重建（第 %d 次，%s）",
                  name, self._restarts[name], detail)
        try:
            from core.notify import notify
            await notify(self.db, f"进化引擎自愈：{name}",
                         f"训练循环曾意外退出（{detail}），已自动重建，"
                         f"累计自愈 {self._restarts[name]} 次。")
        except Exception:  # noqa: BLE001
            pass  # 通知不可用不阻断保活主流程

    async def _train_factor_miner_once(self, force: bool = False) -> Optional[bool]:
        """执行一次因子挖掘训练（轮换训练标的：BTC/ETH/SOL）。"""
        # 追踪实际使用的数据源（真实交易所 or 演示兜底）
        data_source_used = "exchange"
        # 轮换选择下一个训练标的
        symbol = self._next_symbol()
        self._factor_miner_status["symbol"] = symbol

        # 1. 获取最新数据（交易所不可达时按配置决定：跳过本轮 或 用演示数据兜底）
        df = await self._fetch_latest_data(symbol=symbol)
        if df is None or len(df) < self._rolling_window * 0.1:
            if df is None:
                # P4-C1：递增连续 demo 计数（供 _run_loop 降频）
                self._factor_miner_status["_demo_streak"] = int(self._factor_miner_status.get("_demo_streak", 0)) + 1
                # P1：evolve_train_on_demo=False（默认）时故障期直接跳过——
                # demo 模型永远不会被部署，故障期每 60s 全量重训纯烧 CPU，
                # 合成数据的 OOS 还必为负、徒增回退噪声。跳过同样返回 False
                # （与增量门同语义），_run_loop 的降频逻辑照常生效。
                if not getattr(settings, "evolve_train_on_demo", False):
                    self._factor_miner_status["last_error"] = f"交易所数据不可达，已暂停训练（{symbol}）"
                    log.warning("[evolve] 因子挖掘（%s）：交易所数据不可达，跳过本轮（演示训练已禁用）", symbol)
                    return False
                data_source_used = "demo"
                self._factor_miner_status["last_error"] = f"交易所数据不可达，已用合成数据兜底（{symbol}）"
                log.warning("[evolve] 因子挖掘（%s）：交易所数据不可达，使用合成数据兜底", symbol)
                df = self._generate_symbol_demo(symbol, n=self._rolling_window, timeframe=self._effective_timeframe())
            else:
                self._factor_miner_status["last_error"] = f"数据不足（{len(df)} < {int(self._rolling_window * 0.1)}），5分钟后重试"
                log.info("[evolve] 因子挖掘（%s）：数据不足（%d < %d），5分钟后重试", symbol, len(df), int(self._rolling_window * 0.1))
                raise InsufficientDataError(f"数据不足: {len(df)} < {int(self._rolling_window * 0.1)}")
        else:
            # 真实交易所数据：清零连续 demo 计数（恢复正常训练节奏）
            self._factor_miner_status["_demo_streak"] = 0

        # 2. 截取滚动窗口
        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]

        # P2-11 增量门：数据无新增 K 线（末根时间戳未变化）则跳过本轮，
        # 避免 60s 间隔 + 5 分钟数据缓存下的空转重训。demo 轮直接放行且不写 prev。
        if not force and not self._data_incremented("factor_miner", df, symbol=symbol,
                                                    is_demo=(data_source_used == "demo")):
            return False

        # 3. 加载上次最佳模型
        agent = self.zoo.load_agent("factor_miner")
        best_agent = agent  # 训练前保存当前最佳
        # E-EXP（2026-09 实验落地）：自适应 from-scratch 重置——连续 OOS 被拒达到
        # settings.evolve_factor_miner_reset_after_rejects 时，本轮不从旧模型续训
        # （base_agent=None），强制从头训练打破"永久连拒"。实验依据：
        # .optim/exp_training/replicate_prod_windows.py——生产 9/13-9/14 连续 15+ 轮
        # 被拒（|rank_ic|0.004~0.028<0.03）而同窗口 masked from-scratch 过门槛率 44%，
        # 重置后约 2-3 轮可重新产出一版通过安检的组合因子。
        _reject_streak = int(self._factor_miner_status.get("_reject_streak", 0))
        _reset_after = int(getattr(settings, "evolve_factor_miner_reset_after_rejects", 0) or 0)
        if _reset_after > 0 and _reject_streak >= _reset_after:
            best_agent = None
            log.warning("[evolve] 因子挖掘连续 %d 轮 OOS 被拒，本轮 from-scratch 重训"
                        "（打破连拒，阈值=%d）", _reject_streak, _reset_after)
            self._factor_miner_status["last_error"] = (
                f"连续 {_reject_streak} 轮被拒，已触发 from-scratch 重训")

        # 4. 计算因子矩阵（P4-E1：包 to_thread——对最多 rolling_window 根K线
        # 计算全部内置因子，纯 pandas 向量化数百毫秒~秒级，留在主协程会阻塞事件循环）
        mat = await asyncio.to_thread(compute_factor_matrix, df)

        # 5. 训练（P0：续训——把已加载的 best_agent 传给训练器作为起点。
        # 此前 load_agent 只用于回退对照，训练本体每次随机初始化重训，
        # 8 轮 PPO 从零起步学不到东西 → 每轮都被锚点回退，形成死循环。
        # 现在每轮在旧模型上小步增量，成本大幅下降、模型真正进化。
        # 续训只换起点：OOS 安检、回退保护、demo 轮短路等防线全部照常生效。)
        def _train():
            return train_factor_miner(
                df, mat=mat,
                cfg={
                    "episodes": getattr(settings, "evolve_factor_miner_episodes", 8),
                    "n_episodes": 8,
                    "ppo_epochs": 4,
                    "mini_batch_size": 64,
                    "hidden": [64, 64],
                    "seed": None,  # 随机种子，每次不同
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "h": 1,
                    "ic_window": 120,
                    "max_steps": 6,
                    "corr_threshold": 0.85,
                    # E-EXP：动作掩码（强制每步选新因子，组合恒为 max_steps 个）。
                    # .optim/exp_training/sweep_factor.py + verify_factor_win.py
                    # 实测：OOS 秩IC 0.0088→0.0356、续训 OOS 门 0/3→3/3（3 seed）。
                    "action_masking": True,
                    # 训练目标口径（2026-09-29 成本对齐修复）：net = 验证段可交易
                    # 净收益（方向收益 − 换手成本）× scale。旧 ic 口径的 fitness
                    # 会被低换手奖励虚高（0.35 vs 真实 IC 0.07），与回测收益脱节。
                    "fitness_mode": str(getattr(settings, "evolve_factor_miner_fitness_mode", "net") or "ic"),
                    "cost_per_turn": float(getattr(settings, "evolve_factor_miner_cost_per_turn", 0.002) or 0.0),
                    "net_fitness_scale": float(getattr(settings, "evolve_factor_miner_net_fitness_scale", 500.0) or 1.0),
                    "base_agent": best_agent,   # 续训起点（None=首轮从零）
                    "time_budget": float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0),
                },
                # 不传 on_progress 避免回调跨线程
            )

        result = await self._run_train(_train)
        new_fitness = result["history"][-1]["best_fitness"]
        self._factor_miner_status["episode"] += 1
        self._factor_miner_status["fitness"] = round(new_fitness, 5)

        # 6.0 P4-E1：demo 合成数据轮短路——不回退比较、不写 best/版本、不触发通知。
        # 此前 demo 轮走 save_agent(is_best=False) 占版本名额（高失败率下版本历史
        # 被合成模型污染、还可能裁剪掉 best 锚点快照），且 _log_round("ok") 会把
        # _reject_streak 清零掩盖真实锁死状态；通知文本还引用未绑定的 old_fitness
        # （锚点不可比路径 NameError 被吞，通知永远发不出）。
        if data_source_used == "demo":
            log.warning("[evolve] 因子挖掘（%s）demo 数据轮：跳过回退/落库，存档留证", symbol)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], "factor_miner",
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": "demo", "demo_blocked": True})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 因子挖掘 demo 存档失败: %s", e)
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked")
            # demo 轮不发布组合因子（防污染下游），状态里标记
            self._factor_miner_status["oos_rejected"] = "演示数据训练，禁止级联发布"
            return "demo"

        # 6. 最佳模型回退保护（P0-4：旧 fitness 非正时“×阈值”方向反转，统一走 _should_rollback）
        # 锚点口径不可比时（如 demo 合成数据留下的 3.9 分）跳过：真实数据永远够不着
        # 合成数据的量级，继续比较等于让这条管线永久只回退不前进。
        # Task 4：同符号/周期/配置才可比；旧锚点缺身份字段按 unknown 不压新模型。
        anchor = self.zoo.best_info("factor_miner")
        # 自实验观察项解决：factor_signature 不再取"选中因子列表"。RL 因子选择是
        # 训练**输出**（每轮探索必变），把它纳入身份指纹会让新旧锚点每轮都
        # factor_signature 不一致 → _identity_mismatch 永不通过 → 回退保护被永久
        # 旁路（日志对每轮 "锚点身份不可比，跳过回退比较"，实际退化模型一直上线）。
        # 身份口径回到 symbol/timeframe/config（config 指纹已含 episodes/ic_window/
        # corr/action_masking，训练口径改了照样判不可比重建基线）。
        _factor_identity = _training_identity(
            "factor_miner", symbol, self._effective_timeframe(),
            state_window=1,
            config={
                    "episodes": int(getattr(settings, "evolve_factor_miner_episodes", 8)),
                    "ic_window": 120, "corr_threshold": 0.85,
                    "train_ratio": 0.6, "val_ratio": 0.2,
                    # E-EXP：纳入身份——旧锚点（无掩码训练）自动判不可比，
                    # 重建基线而非与不同训练口径比较回退
                    "action_masking": True,
                    # 训练目标口径（2026-09-29）：fitness 量纲变了就不该与旧锚点
                    # 直接比——旧 ic 口径 fitness≈0.35 vs 新 net 口径不同量级，
                    # 强制判不可比、重建基线（缺失=旧模型，同样判不可比）。
                    "fitness_mode": str(getattr(settings, "evolve_factor_miner_fitness_mode", "net") or "ic"),
                    "cost_per_turn": float(getattr(settings, "evolve_factor_miner_cost_per_turn", 0.002) or 0.0),
                })
        if best_agent is not None and anchor.get("comparable"):
            _factor_mismatch = _identity_mismatch(anchor, _factor_identity)
            if _factor_mismatch:
                self._mark_deployment_status("factor_miner", anchor_mismatch=_factor_mismatch)
                log.info("[evolve] 因子挖掘锚点身份不可比（%s），跳过回退比较", _factor_mismatch)
            else:
                old_fitness = anchor.get("fitness") or 0.0
                if _should_rollback(new_fitness, old_fitness, self._rollback_threshold):
                    log.warning("[evolve] 因子挖掘新模型退化 (new=%.4f < old=%.4f*%.2f)，回退到最佳版本",
                                new_fitness, old_fitness, self._rollback_threshold)
                    try:
                        from core.notify import notify
                        await notify(self.db, "进化引擎：模型回退",
                                     f"因子挖掘新模型退化（new={new_fitness:.4f} < "
                                     f"old={old_fitness:.4f}×{self._rollback_threshold}），已回退最佳版本")
                    except Exception:  # noqa: BLE001
                        pass
                    # 统一部署入口回退到当前锚点版本（候选 fitness 与部署分离）
                    _target_version = anchor.get("version") or self.zoo.best_version("factor_miner")
                    await self.deploy_model(
                        "factor_miner", _target_version,
                        outcome="rollback", symbol=symbol, timeframe=self._effective_timeframe(),
                        reason=f"新模型退化({new_fitness:.4f})")
                    self._mark_deployment_status(
                        "factor_miner", outcome="rollback",
                        candidate_fitness=new_fitness)
                    # P2-13：回退轮次也要落库（此前回退路径漏记，曲线因此长期空白）
                    await self._log_round(
                        "factor_miner", symbol, self._effective_timeframe(),
                        data_source_used, new_fitness, status="rollback")
                    # P4-E1：回退轮补发 evolve_train 事件——此前各条 rollback 分支
                    # 不发事件，前端若依赖事件流刷新进度，回退轮完全看不到更新
                    await self.bus.publish(Event(EventType.SYSTEM, {
                        "kind": "evolve_train",
                        "model": "factor_miner",
                        "fitness": float(new_fitness),
                        "status": "rollback",
                    }, source="evolve_engine"))
                    return
        elif best_agent is not None:
            log.warning("[evolve] 因子挖掘锚点不可比（%s），跳过回退保护，本轮模型重建基线",
                        anchor.get("reason") or "未知")

        # 7.0 OOS 安检门前置（Task 4）：未通过安检的因子不得成为 factor best
        # 基线，也不得发布级联因子——此前安检在 save_agent(is_best=True) 之后，
        # 被拦模型照样覆盖 best（研究最佳与部署基线混为一谈）。
        gate_ok, gate_reason = _oos_gate_passed(result.get("report"))
        if data_source_used == "demo":
            self._factor_miner_status["oos_rejected"] = "演示数据训练，禁止级联发布"
            log.warning("[evolve] 因子挖掘使用演示数据训练，跳过最佳保存与级联发布")
            try:
                await asyncio.to_thread(
                    self.zoo.save_agent_archive, result["agent"], "factor_miner",
                    meta={"fitness": new_fitness, "timestamp": time.time(),
                          "data_source": "demo", "demo_blocked": True})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 因子挖掘 demo 存档失败: %s", e)
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked")
            return "demo"
        if not gate_ok:
            self._factor_miner_status["oos_rejected"] = gate_reason
            # E-EXP：拒绝连击计数（自适应重置的触发依据；通过/回退时清零）
            self._factor_miner_status["_reject_streak"] = _reject_streak + 1
            log.warning("[evolve] 组合因子未通过 OOS 安检（%s），仅存档不设为 best"
                        "（连续 %d 轮被拒）", gate_reason, _reject_streak + 1)
            try:
                await asyncio.to_thread(
                    self.zoo.save_agent_archive, result["agent"], "factor_miner",
                    meta={"fitness": new_fitness, "timestamp": time.time(),
                          "data_source": data_source_used,
                          "oos_rejected": gate_reason})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 被拦因子模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：因子 OOS 安检拦截",
                             f"因子挖掘新模型未通过样本外安检（{gate_reason}），"
                             f"已保留原因子最佳基线")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  data_source_used, new_fitness, status="oos_rejected",
                                  selected_factors=result.get("selected_factors", []))
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "factor_miner",
                "fitness": float(new_fitness),
                "oos_rejected": gate_reason,
            }, source="evolve_engine"))
            # 第3项完善：被拒轮也刷新已部署策略健康度（衰变监控核心场景）
            self._refresh_deployed_combo_health(df)
            return "oos_rejected"

        # 7. 保存最佳模型（OOS 安检通过的 exchange 模型才写 best）
        # E-EXP：通过安检 → 清空拒绝连击计数
        self._factor_miner_status["_reject_streak"] = 0
        is_best = True
        await asyncio.to_thread(
            self.zoo.save_agent, result["agent"], "factor_miner",
            meta={"fitness": new_fitness, "timestamp": time.time(),
                  "data_source": data_source_used, **_factor_identity},
            is_best=is_best)
        self._mark_deployment_status("factor_miner", outcome="accepted",
                                     candidate_fitness=new_fitness)
        try:
            from core.notify import notify
            # P4-E1：old_fitness 只在上方 comparable 分支内绑定；锚点不可比路径
            # 直接引用会 NameError（被外层 except 吞掉，通知静默丢失）。
            _old_fit = anchor.get("fitness") or 0.0
            await notify(self.db, "进化引擎：因子挖掘模型更新",
                         f"fitness {_old_fit:.4f} → {new_fitness:.4f}"
                         f"（{data_source_used} 数据）")
        except Exception:  # noqa: BLE001
            pass
        log.info("[evolve] 因子挖掘训练完成，fitness=%.4f，已保存", new_fitness)

        # 7.1 保存组合因子供策略 DRL 级联使用（已过 7.0 安检）
        import numpy as np
        composite_series = result.get("composite")
        self._factor_miner_status["oos_rejected"] = ""
        # 组合因子权重（combo_spec）落盘不依赖 composite：手动重新部署只读权重
        # 文件，composite 为 None（跳过级联 npy 保存）时权重仍需持久化（M-5）。
        weights = result.get("weights", {})
        if weights:
            import json as _json
            cascade_weights_path = self.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
            with open(str(cascade_weights_path), "w", encoding="utf-8") as f:
                _json.dump(weights, f, ensure_ascii=False)
            log.info("[evolve] 组合因子权重已保存到 %s", cascade_weights_path)
            # 伴随报告元数据：手动重新部署时还原权重对应的 OOS 报告/fitness/选中因子
            # （bug-fix 自实验发现：手动部署此前只能写入 status 最新轮次值，且 report
            # 传 None 会覆盖掉上一版已部署策略的 OOS 报告）。
            try:
                _rep_path = self.zoo.models_dir / f"_cascade_report_{symbol.replace('/', '_')}.json"
                _rep_path.write_text(_json.dumps({
                    "fitness": float(new_fitness),
                    "oos_report": result.get("report") or {},
                    "selected_factors": result.get("selected_factors", []),
                    "round_no": int(self._factor_miner_status.get("episode", 0)),
                    "ts": time.time(),
                }, ensure_ascii=False), encoding="utf-8")
                log.info("[evolve] 组合因子报告元数据已保存到 %s", _rep_path)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 组合因子报告元数据写入失败（不影响权重落盘）: %s", e)
        # 清理旧格式单文件（兼容升级前的遗留），防止被误加载
        legacy = self.zoo.models_dir / "_cascade_factor_values.npy"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:  # noqa: BLE001
                pass
        if composite_series is not None:
            import pandas as pd
            # 对齐到原始 df 的索引（factor_miner 内部可能截断或排序）
            factor_values = pd.Series(composite_series.to_numpy(), index=composite_series.index)
            aligned = factor_values.reindex(df.index).to_numpy(float)
            # 按标的分文件保存（P4-A1）：level-2 防御。尽管各管线现已同标训练，
            # 分文件 + 加载时按标的匹配仍防止任何异常路径下的跨标错配。
            cascade_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.npy"
            np.save(str(cascade_path), aligned)
            # P4-E1：与 npy 同存窗口尾时间戳 companion JSON——加载端只校验长度
            # 无法发现"窗口右移后的静默错位"（长度同为 rolling_window），
            # 存尾时间戳让加载端能按窗口身份校验，错位即拒绝注入。
            try:
                import json as _json
                tail_ts = None
                try:
                    tail_ts = int(df.index[-1].timestamp())
                except Exception:  # noqa: BLE001
                    tail_ts = int(df.index[-1]) if df.index[-1] is not None else None
                if tail_ts is not None:
                    meta_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.json"
                    meta_path.write_text(_json.dumps({"tail_ts": tail_ts, "n": int(len(aligned))}),
                                         encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 级联因子时间戳元数据写入失败（不影响 npy 保存）: %s", e)
            log.info("[evolve] 组合因子已保存到 %s，供策略 DRL 级联注入", cascade_path)

        # 8. 发布系统事件
        # P2-13：训练轮次落库（fitness/选中因子/数据源）
        await self._log_round(
            "factor_miner", symbol, self._effective_timeframe(),
            data_source_used, new_fitness,
            status="ok", selected_factors=result.get("selected_factors", []))
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "evolve_train",
            "model": "factor_miner",
            "fitness": new_fitness,
            "selected_factors": result.get("selected_factors", []),
        }, source="evolve_engine"))

        # 9. 自动部署：OOS 安检已过（gate_ok），把组合权重部署为按标的的组合因子策略。
        # 部署失败只记 combo_deploy_error，不阻断训练流程（训练主体已完成并落库）。
        try:
            _weights = result.get("weights") or {}
            if not _weights:
                # 权重缺失：显式记错并跳过部署，不更新已有部署状态
                # （注册表里已部署的策略仍有效，组合策略相关键保持上一轮值不动）
                self._factor_miner_status["combo_deploy_error"] = "权重缺失，跳过自动部署"
                log.warning("[evolve] 组合因子策略自动部署跳过(%s): 权重缺失", symbol)
            else:
                _meta = {
                    "fitness": float(new_fitness),
                    "selected_factors": result.get("selected_factors", []),
                    "round_no": self._factor_miner_status.get("episode", 0),
                }
                _dep = await self._deploy_factor_strategy(
                    symbol, _weights, result.get("report"), _meta)
                self._factor_miner_status["combo_strategy"] = _dep["name"]
                self._factor_miner_status["combo_version"] = _dep["version"]
                self._factor_miner_status["combo_deployed_at"] = time.time()
                self._factor_miner_status["combo_deploy_error"] = ""
                # 后台回测摘要（fire-and-forget，不阻塞训练循环；持有任务引用防 GC）
                _task = asyncio.create_task(
                    self._refresh_factor_strategy_backtest(_dep["name"], symbol, df, _dep["version"]))
                self._backtest_tasks[_dep["name"]] = _task
                _task.add_done_callback(
                    lambda t, n=_dep["name"]: self._backtest_tasks.pop(n, None))
        except Exception as e:  # noqa: BLE001
            self._factor_miner_status["combo_deploy_error"] = f"自动部署失败: {e}"
            log.warning("[evolve] 组合因子策略自动部署失败(%s): %s", symbol, e)

        # 10. 组合因子健康度（第3项完善：IC 衰变监控/上线质检）——每轮用最新
        # 数据对当前已部署策略的权重因子算近期滚动 rank IC（通过轮刚更新
        # combo_strategy → 算的就是新部署；被拒轮已在安检分支刷新）。
        self._refresh_deployed_combo_health(df)

    # ---- 策略 DRL 持续训练循环 ----

    async def _train_strategy_drl_once(self, force: bool = False) -> Optional[bool]:
        """执行一次策略 DRL 训练，并注册为可用策略（轮换训练标的：BTC/ETH/SOL）。"""
        # 1. 获取最新数据（交易所不可达时按配置决定：跳过本轮 或 用演示数据兜底）
        data_source_used = "exchange"
        symbol = self._next_symbol()
        self._strategy_drl_status["symbol"] = symbol
        # 经验量修复（2026-10）：策略 DRL 按 evolve_strategy_drl_history_bars 拉取多年
        # 历史（0=沿用 evolve_rolling_window 的旧行为）。旧行为只取最近 5000 根
        # （5m≈17 天）并在同一段上每 60s 重训——"人类几年的学习量"在数据维度就被挡死。
        _sdrl_history = max(0, int(getattr(settings, "evolve_strategy_drl_history_bars", 0) or 0))
        _keep_bars = _sdrl_history if _sdrl_history > 0 else int(self._rolling_window)
        df = await self._fetch_latest_data(
            symbol=symbol, bars=_sdrl_history if _sdrl_history > 0 else 0)
        if df is None or len(df) < self._rolling_window * 0.1:
            if df is None:
                # P4-C1：递增连续 demo 计数（供 _run_loop 降频）
                self._strategy_drl_status["_demo_streak"] = int(self._strategy_drl_status.get("_demo_streak", 0)) + 1
                # P1：evolve_train_on_demo=False（默认）时故障期直接跳过——
                # demo 模型永远不会被部署，故障期全量重训纯烧 CPU（与因子挖掘同口径）。
                if not getattr(settings, "evolve_train_on_demo", False):
                    self._strategy_drl_status["last_error"] = f"交易所数据不可达，已暂停训练（{symbol}）"
                    log.warning("[evolve] 策略DRL（%s）：交易所数据不可达，跳过本轮（演示训练已禁用）", symbol)
                    return False
                data_source_used = "demo"
                self._strategy_drl_status["last_error"] = f"交易所数据不可达，已用合成数据兜底（{symbol}）"
                log.warning("[evolve] 策略DRL（%s）：交易所数据不可达，使用合成数据兜底", symbol)
                df = self._generate_symbol_demo(symbol, n=_keep_bars, timeframe=self._effective_timeframe())
            else:
                self._strategy_drl_status["last_error"] = f"数据不足（{len(df)} < {int(self._rolling_window * 0.1)}），5分钟后重试"
                log.info("[evolve] 策略DRL（%s）：数据不足（%d < %d），5分钟后重试", symbol, len(df), int(self._rolling_window * 0.1))
                raise InsufficientDataError(f"数据不足: {len(df)} < {int(self._rolling_window * 0.1)}")
        else:
            # 真实交易所数据：清零连续 demo 计数（恢复正常训练节奏）
            self._strategy_drl_status["_demo_streak"] = 0

        # 2. 截取训练窗口（经验量修复：默认取 history_bars 而非 rolling_window）
        if len(df) > _keep_bars:
            df = df.iloc[-_keep_bars:]

        # P2-11 增量门：数据无新增 K 线则跳过本轮（仅连续训练生效，手动触发总执行）。
        # demo 轮直接放行且不写 prev（合成数据尾戳 ≈now 会污染真实数据增量判定）。
        if not force and not self._data_incremented("strategy_drl", df, symbol=symbol,
                                                    is_demo=(data_source_used == "demo")):
            return False

        # 2026-10 审计修复（MED-2）：demo（交易所故障合成数据）轮不做任何观测——
        # 合成数据评估部署模型会污染 gap01_streak/_gap01_pass_*/longwindow_verdict，
        # 故障期累积的假 FAIL 会在恢复后触发冠军 from-scratch；观测只在真实数据上有意义。
        _demo_round = (data_source_used == "demo")
        if not _demo_round:
            # walk-forward 多折观测（2026-09-19，用户选"1"）：本轮确认真实训练后，
            # 用部署中的最佳模型做同品种多折评估（不改接受逻辑），结果写
            # models/walkforward_obs.json + 日志。异常不影响训练主流程。
            try:
                await self._observe_strategy_drl_walkforward(df=df)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] walk-forward观测失败（不影响训练）: %s", e)

            # GAP-01 实盘验收观测（2026-09-19，建议第6条(b)）：同品种多折 × 两档成本，
            # 输出 PASS/FAIL/INCONCLUSIVE（models/gap01_obs.json）。软信号（2026-09-20）：
            # FAIL 与 INCONCLUSIVE 均累计 gap01_streak、PASS 清零；连续 N 轮不过
            # 触发冠军轮 from-scratch 重学（evolve_gap01_fail_reset_after）。异常自吞。
            # 注：INCONCLUSIVE=样本外证据不足/不交易，同样不能上线——若不计入 streak，
            # from-scratch 重训后的模型不交易会永远判 INCONCLUSIVE，软信号被卡死
            # （运行态实录 2026-09-20：gap01_streak 停在 2 到不了阈值）。
            try:
                _gap01 = await self._observe_gap01(df=df)
                if _gap01 is not None:
                    _v = str(_gap01.get("verdict") or "")
                    if _v == "PASS":
                        self._gap01_streak["strategy_drl"] = 0
                    elif _v in ("FAIL", "INCONCLUSIVE"):
                        self._gap01_streak["strategy_drl"] = (
                            int(self._gap01_streak.get("strategy_drl", 0)) + 1)
                    self._strategy_drl_status["gap01_verdict"] = _v
                    self._strategy_drl_status["gap01_streak"] = self._gap01_streak.get("strategy_drl", 0)
                    # 多窗口连续 PASS 判定（2026-09-21，§16.5 遗留1）：同一部署模型
                    # 连续 N 次观测 PASS 才算稳定（gap01_stable=True，实盘候选依据）。
                    # 仅 PASS 累计；FAIL/INCONCLUSIVE 或部署版本变化即清零（不交易/
                    # 证据不足不算稳定 PASS，杜绝"窗口滑动运气"——v36/v41/v42 教训）。
                    try:
                        _deploy_ver = (self.zoo.best_info("strategy_drl") or {}).get("version")
                    except Exception:  # noqa: BLE001
                        _deploy_ver = None
                    if _deploy_ver != self._gap01_pass_version:
                        _prev_ver = self._gap01_pass_version
                        self._gap01_pass_version = _deploy_ver
                        self._gap01_pass_streak = 0
                        # 2026-10 审计修复（MED）：部署版本**更换**时 FAIL 计数同样清零
                        #（旧模型的 FAIL 不传导到新模型，否则新模型一上线就背着旧账
                        #  触冠军 from-scratch；首次确定不算换模型）。
                        _clear_gap01_streak_on_version(self, _deploy_ver,
                                                       prev_version=_prev_ver)
                    if _v == "PASS":
                        self._gap01_pass_streak += 1
                    else:
                        self._gap01_pass_streak = 0
                    _stable_req = int(getattr(settings, "evolve_gap01_stable_pass_required", 3) or 0)
                    self._strategy_drl_status["gap01_pass_streak"] = self._gap01_pass_streak
                    self._strategy_drl_status["gap01_stable"] = bool(
                        _stable_req > 0 and self._gap01_pass_streak >= _stable_req)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] GAP-01观测失败（不影响训练）: %s", e)

            # 长窗回放观测（2026-09-21，§16.5 升级）：30/90 天真实数据引擎撮合回放，
            # 输出 longwindow_verdict（两窗均正才 PASS）；防"近期窗口效应"冒充实盘候选
            #（v44 教训：短窗 stable PASS 但 30 天 -26.6% / 90 天 -67.8%）。异常自吞。
            # 2026-09-22（训练速度研究）：训练本身仅 ~9.5s/轮，瓶颈在 30/90 天回放
            #（每轮 +数分钟）——默认每 5 轮跑一次（longwindow_every_n），进化节奏回升。
            _lw_every = max(1, int(getattr(settings, "evolve_strategy_drl_longwindow_every_n", 5) or 1))
            if _lw_every == 1 or int(self._strategy_drl_status.get("_pop_round", 0)) % _lw_every == 0:
                try:
                    await self._observe_strategy_drl_longwindow()
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 长窗回放观测失败（不影响训练）: %s", e)

        # P0-族群：冠军/挑战者 K=2 交替。_pop_round 每完成一轮真实训练递增一次
        # （增量门跳过/故障跳过不计），偶数为冠军轮（走原路径，部署+注册），
        # 奇数为挑战者轮（独立谱系 strategy_drl_alt，须显著优于冠军才晋升）。
        # 每轮仍只训一个模型 → 墙钟成本不变；多样性来自两条谱系独立进化。
        _pop_on = bool(getattr(settings, "evolve_strategy_drl_population", True))
        # force（手动触发）恒走冠军谱系：手动训练语义是"更新部署模型"，
        # 挑战者轮只由后台连续训练节奏驱动。
        _round_no = int(self._strategy_drl_status.get("_pop_round", 0))
        is_challenger = bool(_pop_on and not force and (_round_no % 2 == 1))
        # 2026-10 审计修复（MED-2/3）：只有**真实训练轮**推进谱系交替：
        # demo 轮（合成数据）与异常轮（训练无产出）不占冠军/挑战者槽位——
        # 否则故障期每次 demo 都翻转奇偶、恢复后首个真实轮变成挑战者轮（不能部署）。
        if not _demo_round:
            self._strategy_drl_status["_pop_round"] = _round_no + 1
        self._strategy_drl_status["lineage"] = "challenger" if is_challenger else "main"
        lineage_name = "strategy_drl_alt" if is_challenger else "strategy_drl"

        # 3. 加载本轮谱系的上次最佳模型（续训）。
        # 族群根因修复（2026-09-19）：挑战者谱系若 best 为空（OOS 门失败从未更新），
        # 回退到独立「训练基线」槽位续训，避免每轮从零训练（fitness 恒 0 的死循环）。
        agent = self.zoo.load_agent(lineage_name)
        if agent is None and is_challenger:
            agent = self.zoo.load_agent_base("strategy_drl_alt")
        best_agent = agent

        # 连续拒绝 / GAP-01 FAIL from-scratch 重置（仅冠军轮）：
        # 冠军谱系被老化锚点压住、连续未接受（或被 GAP-01 连续判 FAIL）时，
        # 本轮从零重训跳出局部区域；挑战者轮有独立训练基线机制，不参与此重置。
        if not is_challenger:
            _reset_by_rejects = _should_from_scratch_reset(
                self._reject_streak.get("strategy_drl", 0),
                getattr(settings, "evolve_strategy_drl_reset_after_rejects", 0))
            _reset_by_gap01 = _should_from_scratch_reset(
                self._gap01_streak.get("strategy_drl", 0),
                getattr(settings, "evolve_gap01_fail_reset_after", 0))
            if _reset_by_rejects or _reset_by_gap01:
                best_agent = None
                if _reset_by_gap01:
                    log.warning("[evolve] 策略DRL 连续 %d 轮 GAP-01 未通过"
                                "（FAIL/INCONCLUSIVE），本轮 from-scratch 重训（阈值=%d）",
                                self._gap01_streak.get("strategy_drl", 0),
                                int(getattr(settings, "evolve_gap01_fail_reset_after", 0)))
                    self._strategy_drl_status["last_error"] = (
                        f"连续 {self._gap01_streak.get('strategy_drl', 0)} 轮 "
                        "GAP-01 未通过，已触发 from-scratch 重训")
                    # 触发后清零：给新起点 N 轮续训窗口，避免"每轮都 from-scratch"
                    # 空转（运行态实录 2026-09-20：11:00:43 触发后 11:00:47 又触发）
                    self._gap01_streak["strategy_drl"] = 0
                    self._strategy_drl_status["gap01_streak"] = 0
                if _reset_by_rejects:
                    log.warning("[evolve] 策略DRL 连续 %d 轮未接受，本轮 from-scratch 重训"
                                "（打破连拒，阈值=%d）",
                                self._reject_streak.get("strategy_drl", 0),
                                int(getattr(settings, "evolve_strategy_drl_reset_after_rejects", 0)))
                    self._strategy_drl_status["last_error"] = (
                        f"连续 {self._reject_streak.get('strategy_drl', 0)} 轮被拒，"
                        "已触发 from-scratch 重训")
                    # 与 gap01 分支同理（见上）：触发后清零，给新起点 N 轮续训窗口。
                    # 此前只清 gap01、不清 reject，而 _reject_streak 仅在 status=="ok"
                    # 或重置锚点时归零——strategy_drl 的 from-scratch 过门率为 0
                    # （REPORT §22：生产窗口复刻 16 次全不过门），于是连拒一旦达阈值
                    # 就再也不会回落，**每一轮都从零重训**：累积学习被反复丢弃，
                    # 表现为模型始终学不动（现场实测 reject_streak=19 > 阈值 5）。
                    self._reject_streak["strategy_drl"] = 0

        # 级联因子值：按当前训练标的加载，保证注入的因子与训练标的一致（P4-A1）
        cascade_factor_values = self._load_cascade_factor_values(df, symbol=symbol)
        # 级联因子复算配方（选中因子权重）：与 npy 同标保存，供部署端实盘复算
        cascade_recipe = (self._load_cascade_factor_recipe(symbol)
                          if cascade_factor_values is not None else None)

        # 训练配置提升到函数作用域：跨标传导 OOS（下方）需要复用 state_window，
        # 曾定义在 _train 闭包内导致成功路径必然 NameError
        train_cfg = _strategy_drl_train_cfg(
            getattr(settings, "evolve_strategy_drl_episodes", 8))

        # 经验回放缓冲区（跨轮存活）：capacity>0 时持有实例并每轮透传，
        # 让上一轮/更早窗口的轨迹参与本轮更新。
        # 重建条件：(a) capacity 变更；(b) 训练标的变更——符号轮换（ETH→SOL）不会
        # 改变状态维度，维度守卫生效不了，若不按标的隔离，上一标的的轨迹会混入本轮
        # 策略更新（每轮只 8ep 时，64 条跨标轨迹足以主导梯度）。按符号重建更保守：
        # 回放只覆盖"同一标的的最近若干轮"，符合"对抗遗忘"的本意而非引入跨标噪声。
        # P0 多标的混合训练例外：训练本体就是跨标的混合采样，缓冲里的轨迹天然
        # 含多标的经验——按主标的重建会每轮清空缓冲（主标轮换），反而丢掉经验，
        # 与"多标的混合"的目的相悖。多标模式开启时跳过 (b) 条件，只在容量变更时重建。
        _multi_on = bool(getattr(settings, "evolve_strategy_drl_multi_symbol", True))
        _replay_cap = max(0, int(getattr(
            settings, "evolve_strategy_drl_replay_capacity", 0) or 0))
        if _replay_cap > 0:
            if (self._strategy_drl_replay is None
                    or int(self._strategy_drl_replay.capacity) != _replay_cap
                    or (not _multi_on and self._strategy_drl_replay_symbol != symbol)):
                self._strategy_drl_replay = ReplayBuffer(
                    capacity=_replay_cap,
                    # 2026-10 审计修复（MED-2）：采样 rng 派生自训练 seed（可复现）
                    seed=int(getattr(settings, "evolve_strategy_drl_seed", 42) or 42))
                log.info("[evolve] 策略DRL 经验回放缓冲区已建立: capacity=%d 标的=%s%s",
                         _replay_cap, symbol, "（多标模式跨标共享）" if _multi_on else "")
            self._strategy_drl_replay_capacity = _replay_cap
            self._strategy_drl_replay_symbol = symbol
        else:
            self._strategy_drl_replay = None
            self._strategy_drl_replay_capacity = 0
            self._strategy_drl_replay_symbol = ""

        # 4. 训练（P0：续训——best_agent 传入作为起点。此前 load_agent 只用于
        # 回退对照，训练每次随机初始化重训，8 轮 PPO 从零起步学不到东西 →
        # 每轮都被锚点回退，形成死循环。现在每轮在旧模型上小步增量，
        # 成本大幅下降、模型真正进化。续训只换起点：OOS 硬门、回退保护、
        # 训练身份隔离、demo 短路全部照常生效。）
        # P0 多标的混合训练（经验广度）：默认把 evolve_symbols 池中除当前主标的外
        # 的全部标的拉取进来作为额外经验来源（尽力而为：单个标的拉取失败不阻断）。
        # 主标的仍是 val/OOS/部署语义锚点；额外标的只作为训练经验。
        extra_dfs_pool: dict = {}
        if getattr(settings, "evolve_strategy_drl_multi_symbol", True):
            for _sym in self._symbols:
                if _sym == symbol:
                    continue
                try:
                    _sdf = await self._fetch_latest_data(
                        symbol=_sym, bars=_sdrl_history if _sdrl_history > 0 else 0)
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 多标的数据获取失败(%s)，跳过: %s", _sym, e)
                    continue
                if _sdf is None or len(_sdf) < max(300, int(self._rolling_window * 0.1)):
                    continue
                # 额外标的同样按训练历史深度切片（与主标的经验量对齐）
                if len(_sdf) > _keep_bars:
                    _sdf = _sdf.iloc[-_keep_bars:]
                extra_dfs_pool[_sym] = _sdf
            if extra_dfs_pool:
                log.info("[evolve] 策略DRL 多标的混合训练: 额外经验标的 %s",
                         list(extra_dfs_pool))
                self._strategy_drl_status["multi_symbol"] = list(extra_dfs_pool)
            else:
                self._strategy_drl_status["multi_symbol"] = []

        def _train():
            # 级联因子注入：加载 factor_miner 产出的组合因子值（若有），
            # 作为策略 DRL 的额外状态特征（与训练/部署口径一致）
            if cascade_factor_values is not None:
                train_cfg["factor_values"] = cascade_factor_values
                if cascade_recipe is not None:
                    train_cfg["factor_composite"] = cascade_recipe
                log.info("[evolve] 已注入级联组合因子到策略 DRL 训练（含复算配方 %s）",
                         "有" if cascade_recipe is not None else "无")
            train_cfg["base_agent"] = best_agent   # 续训起点（None=首轮从零）
            train_cfg["time_budget"] = float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0)
            # 回放缓冲区透传：跨轮复用（本轮 push 的轨迹留给下一轮）。
            # from-scratch 轮（best_agent=None）同样透传——此时回放正是唯一的
            # 历史经验来源，丢弃它会让"重置"退化为彻底的信息清零。
            if self._strategy_drl_replay is not None:
                train_cfg["replay_buffer"] = self._strategy_drl_replay
                train_cfg["replay_mix_ratio"] = float(getattr(
                    settings, "evolve_strategy_drl_replay_mix_ratio", 0.3) or 0.3)
            train_cfg["extra_mix_ratio"] = float(getattr(
                settings, "evolve_strategy_drl_multi_mix_ratio", 0.5) or 0.5)
            # 多 seed 中位数选模（2026-10，解决"单轮判据像抽奖"）：
            # 固定 seed 是基线（可复现），seeds>1 时本轮训 N 个 seed、
            # 按验证段超额中位数选中一个。总墙钟 ≈ N × 单 seed 时长，
            # 每个 seed 仍受 time_budget 约束。见 _run_multi_seed。
            _n_seeds = max(1, int(getattr(settings, "evolve_strategy_drl_seeds", 1) or 1))
            _seed_base = int(getattr(settings, "evolve_strategy_drl_seed", 42) or 42)
            _chosen, _seed_meta = _run_multi_seed(
                lambda _cfg: train_drl(df, _cfg, on_progress=None,
                                       extra_dfs=extra_dfs_pool or None),
                train_cfg, _n_seeds, _seed_base)
            # 2026-10 审计修复（HIGH）：多 seed 各自用独立回放副本，
            # 选中 seed 的最终缓冲经 meta 交回引擎主缓冲（跨轮记忆 = 上轮选中 seed）。
            _buf = _seed_meta.get("replay_buffer")
            if _buf is not None:
                self._strategy_drl_replay = _buf
            return _chosen

        try:
            result = await self._run_train(_train)
        except BaseException:
            # 2026-10 审计修复（MED-3）：训练抛异常 = 本轮无产出，谱系槽位回滚
            #（不占冠军/挑战者槽位，避免"失败轮翻转奇偶"）。
            if not _demo_round:
                self._strategy_drl_status["_pop_round"] = _round_no
            raise
        # 多 seed 选模摘要 → 状态面板（前端可读：本轮几个 seed、选中哪个、各自超额）
        if result.get("seed_runs"):
            self._strategy_drl_status["seed_runs"] = result["seed_runs"]
            self._strategy_drl_status["seed_chosen"] = result.get("seed_chosen")
        # 取最后一条历史记录的收益
        history = result.get("history", [])
        if not history:
            _err = "策略DRL训练无历史记录（训练未产生结果）"
            log.warning("[evolve] %s", _err)
            # 2026-10 审计修复（MED-3）：训练无产出不算一轮真实训练 → 谱系槽位回滚
            if not _demo_round:
                self._strategy_drl_status["_pop_round"] = _round_no
            raise RuntimeError(_err)
        # fitness 口径与 /api/drl/train 一致：取 train_drl 汇总的 best_ret。
        # 旧写法从 history 末行现取，缺 best_ret 时落到单轮 total_ret，
        # 与跨版本回退比较不同量级。
        # P4-E1：不能用 `or` 链——best_ret=0.0 是 falsy，会被 history 末行
        # 的数值静默替换，锚点 fitness 记错并传导到回退比较。显式判 None。
        _vr = result.get("best_val_ret")
        _vex = result.get("best_val_excess")
        _br = result.get("best_ret")
        # 2026-09-20（训练侧证据 §15）：best_ret=max(训练段,验证段) 被训练段污染——
        # 训练段收益与 OOS 负相关（Pearson -0.486、训练正但OOS≤0占100%）。
        # fitness 口径优先用验证段最优收益（模型从未见过该段），缺失时回退 best_ret
        # （旧模型/手动训练兼容）。OOS 部署门不变，仍按样本外收益把关。
        # 2026-10 基准相对口径：优先用**验证段超额**（val_ret − 同段买入持有）。
        # 原因：val 窗口每轮前移，不同轮的绝对收益基准不同（牛市 +20% 与熊市 −20%
        # 的"0%"完全不是一回事），拿绝对值跨轮比较锚点/回退会误判；同一轮内
        # 基准是常数、排序等价，所以选模行为不变，只是跨轮可比性被修正。
        _use_excess_fitness = bool(getattr(
            settings, "evolve_strategy_drl_fitness_excess", True))
        if _use_excess_fitness and _vex is not None:
            new_fitness = float(_vex)
        elif _vr is not None:
            new_fitness = float(_vr)
        else:
            new_fitness = float(_br if _br is not None
                                else history[-1].get("best_ret") or 0.0)
        anchor = self.zoo.best_info("strategy_drl")
        prev_fitness = anchor.get("fitness") or 0.0
        self._strategy_drl_status["episode"] += 1
        self._strategy_drl_status["fitness"] = round(new_fitness, 5)

        # 训练身份：与锚点同口径才允许回退比较（不同符号/周期/状态维度/配置不可比）。
        # P0 多标的：训练模式（单标/多标）是不可比的配置维度——混入身份指纹，
        # 防止"单标的训练产出"与"多标的混合训练产出"互相回退比较（语义不同）。
        _multi_names = sorted(result.get("multi_extra_names") or [])
        _id_cfg = _strategy_drl_identity_config(train_cfg, _multi_names)
        # 2026-10 审计修复（MED）：级联因子注入与否改变 state_dim 与状态语义——
        # 注入轮与未注入轮（级联因子失效/复算失败时返回 None）的产出被误当同口径
        # 回退比较会错配。仅在**注入**时加 factor_injected 键（未注入轮保持原格式，
        # 与旧锚点/测试手工构造的身份指纹一致）；键名进 config_fingerprint
        #（_identity_mismatch 比较该项）。
        if cascade_factor_values is not None:
            _id_cfg["factor_injected"] = True
        _identity = _training_identity(
            "strategy_drl", symbol, self._effective_timeframe(),
            state_window=int(train_cfg.get("state_window", 1)),
            factor_signature=str(result.get("factor_expression", "") or ""),
            config=_id_cfg)

        # 5.0 P4-E1：demo 合成数据轮短路——不回退比较、不跑 OOS 部署门。
        # 交易所故障时合成数据的 OOS 几乎必为负，会触发无意义的
        # save_agent_best（覆盖 flat）+ 重新导出 Pine + _log_round("rollback")，
        # 且 _reject_streak 递增可能误触发 stalled 横幅。demo 模型只存档留证，
        # 记为 demo_blocked（与 factor_miner 口径一致），best/flat 保持真实模型。
        if data_source_used == "demo":
            log.warning("[evolve] 策略DRL（%s）demo 数据轮：跳过回退/OOS 门，存档留证", symbol)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": "demo", "demo_blocked": True,
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 策略DRL demo 存档失败: %s", e)
            await self._log_round("strategy_drl", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked",
                                  lineage=("challenger" if is_challenger else "main"))
            return "demo"

        # 族群根因修复（2026-09-19）：挑战者轮无论后续 OOS 门/回退结果如何，
        # 先把候选保存为独立「训练基线」，下一轮从它续训（demo 轮已短路跳过）。
        # 基线不进 best/flat、不占版本名额，部署保护语义不变——只保证谱系不再
        # "每轮从零训练→学不到东西→fitness 恒 0→曲线 0/有数据交替"。
        if is_challenger:
            try:
                await asyncio.to_thread(self.zoo.save_agent_base,
                                        result["agent"], "strategy_drl_alt")
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 挑战者训练基线保存失败: %s", e)

        # 5. 最佳模型回退保护（P0-4 负 fitness 方向反转修复；
        # P1-6 回退判据统一为 OOS 收益——与部署门同口径，不再用训练段收益比较）
        # 锚点口径不可比时跳过比较：demo 合成数据留下的高分基线（实测 3.9167）真实
        # 行情一辈子也够不到，只会把每一轮都判成"退化"，进化循环事实上停摆。
        # P0-族群：冠军轮走"退化即回退"；挑战者轮走"显著优于冠军才晋升"——
        # 挑战者永不动部署端（best/flat/注册全保留），只有晋升那一轮才接管冠军槽位。
        # 挑战者轮的晋升只依赖冠军锚点（best_info("strategy_drl")），与挑战者自身
        # 起点无关——首轮 alt 槽位为空（best_agent=None）也应能与冠军比较。
        promote = False  # 挑战者轮晋升标志（冠军轮恒 False，不用）
        if is_challenger:
            if anchor.get("comparable") and not _identity_mismatch(anchor, _identity):
                oos_report = result.get("oos_report") or {}
                new_oos = float(oos_report.get("oos_ret") or 0.0) if oos_report.get("enabled") else None
                prev_oos = anchor.get("oos_ret")
                if new_oos is not None and prev_oos is not None:
                    # 挑战者须显著优于冠军才晋升（> 冠军×(1/threshold)，防噪音替换）
                    promote = new_oos > float(prev_oos) * (1.0 / max(self._rollback_threshold, 1e-6))
                    cmp_desc = f"挑战者OOS {new_oos:.4f} vs 冠军 {float(prev_oos):.4f}×(1/{self._rollback_threshold})"
                else:
                    # 旧模型缺 OOS 记录（升级前版本）：退化为训练段收益比较
                    promote = new_fitness > prev_fitness * (1.0 / max(self._rollback_threshold, 1e-6))
                    cmp_desc = f"挑战者fitness {new_fitness:.4f} vs 冠军 {prev_fitness:.4f}×(1/{self._rollback_threshold})"
                if promote:
                    log.info("[evolve] 策略DRL挑战者显著优于冠军（%s），晋升接管冠军槽位", cmp_desc)
                else:
                    log.info("[evolve] 策略DRL挑战者未显著优于冠军（%s），保留冠军，挑战者更新自身谱系", cmp_desc)
            else:
                log.info("[evolve] 策略DRL挑战者轮：冠军锚点不可比（%s），不晋升",
                         anchor.get("reason") or "身份不匹配")
        elif best_agent is not None and anchor.get("comparable"):
            identity_mismatch = _identity_mismatch(anchor, _identity)
            if identity_mismatch:
                self._mark_deployment_status("strategy_drl", anchor_mismatch=identity_mismatch)
                log.info("[evolve] 策略DRL 锚点身份不可比（%s），跳过回退比较", identity_mismatch)
            oos_report = result.get("oos_report") or {}
            # 2026-10 审计修复（HIGH）：回退比较用**超额**口径（oos_excess）——
            # 不同滚动窗口的绝对 oos_ret 不可比（下跌窗口真候选必更低 → 被误回退；
            # 上涨窗口差候选逃过回退），与 fitness/部署门已统一的 excess 语义一致。
            # 缺超额记录（升级前版本/无基准的老锚点）时退化为 fitness（已是超额口径）。
            new_ex = (float(oos_report.get("oos_excess") or 0.0)
                      if oos_report.get("enabled") else None)
            prev_ex = anchor.get("oos_excess")
            # 2026-10（MED-3）：仅当新旧锚点同为 segment_median 口径才直比超额；
            # 缺 basis 的旧锚点（median_diff/升级前）与新口径不可比 → 退化为 fitness。
            _ex_comparable = (prev_ex is not None
                              and anchor.get("oos_excess_basis") == "segment_median"
                              and oos_report.get("oos_excess_basis") == "segment_median")
            if new_ex is not None and _ex_comparable:
                rollback_hit = _should_rollback(new_ex, float(prev_ex), self._rollback_threshold)
                cmp_desc = f"OOS超额 {new_ex:.4f} < {float(prev_ex):.4f}×{self._rollback_threshold}"
            else:
                # 旧模型缺 OOS 超额记录（升级前版本/不同 basis）：退化为训练段收益比较
                rollback_hit = _should_rollback(new_fitness, prev_fitness, self._rollback_threshold)
                cmp_desc = f"fitness {new_fitness:.4f} < {prev_fitness:.4f}×{self._rollback_threshold}"
            if identity_mismatch:
                rollback_hit = False
            if rollback_hit:
                log.warning("[evolve] 策略DRL新模型退化 (%s)，回退到最佳版本", cmp_desc)
                try:
                    from core.notify import notify
                    await notify(self.db, "进化引擎：模型回退",
                                 f"策略DRL新模型退化（{cmp_desc}），已回退最佳版本")
                except Exception:  # noqa: BLE001
                    pass
                await self.deploy_model(
                    "strategy_drl", anchor.get("version") or self.zoo.best_version("strategy_drl"),
                    outcome="rollback", symbol=symbol, timeframe=self._effective_timeframe(),
                    reason=f"新模型退化({cmp_desc})")
                self._mark_deployment_status("strategy_drl", outcome="rollback",
                                             candidate_fitness=new_fitness)
                # P2-13：回退轮次也要落库（此前回退路径漏记，曲线因此长期空白）
                _oos_report_rb = result.get("oos_report") or {}
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness,
                    status="rollback",
                    oos_ret=_oos_report_rb.get("oos_ret", 0.0) or 0.0,
                    decay=_oos_report_rb.get("decay", 0.0) or 0.0,
                    position_ratio=_oos_report_rb.get("oos_position_ratio", 0.0) or 0.0,
                    lineage=("challenger" if is_challenger else "main"))
                # P4-E1：回退轮补发 evolve_train 事件（与因子挖掘一致）
                await self.bus.publish(Event(EventType.SYSTEM, {
                    "kind": "evolve_train",
                    "model": "strategy_drl",
                    "fitness": float(new_fitness),
                    "status": "rollback",
                    "deployed_version": self.zoo.best_version("strategy_drl"),
                }, source="evolve_engine"))
                return
        elif best_agent is not None:
            log.warning("[evolve] 策略DRL 锚点不可比（%s），跳过回退保护，本轮模型重建基线",
                        anchor.get("reason") or "未知")

        # 5.5 OOS 硬门：未过样本外安检的模型不得接管实盘模型、不得注册策略
        gate_ok, gate_reason = _strategy_deploy_gate(result)
        self._strategy_drl_status["oos_rejected"] = "" if gate_ok else gate_reason
        if not gate_ok:
            # P4-C2：只存档不占版本名额——best/flat 文件保持原最佳模型，
            # 已注册的 rl_evolve 继续用旧权重，不会被未验证的模型替换。
            # 此前 is_best=False 的 save_agent 也会递增版本号占名额，被拦截
            # 模型多了会把真改进版挤出版本历史。
            log.warning("[evolve] 策略DRL 未通过 OOS 硬门，保留原最佳模型：%s", gate_reason)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_rejected": gate_reason,
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 被拦截模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：OOS 硬门拦截",
                             f"策略DRL 新模型未通过样本外安检，已保留原最佳模型：{gate_reason}")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="oos_rejected",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0,
                lineage=("challenger" if is_challenger else "main"))
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "oos_rejected": gate_reason,
            }, source="evolve_engine"))
            return

        # 5.7 P1-8：跨标的传导 OOS（低权重附加门）——模型在另一训练标的上
        # 的样本外表现。跨标收益非正时：仅存档不设为最佳（保持旧 best 与 flat 文件），
        # 防止"单标的特有模式"被当成通用能力上线。
        # 级联因子注入时 agent 状态含因子列，跨标数据无法重建同维状态，跳过（不拦）。
        # P0 多标的混合训练例外：训练本体已见全部池内标的（主标+额外标），
        # 跨标 OOS 不再构成"换一个没见过的标的验证泛化"，反而会因模型已学该标
        # 的本地模式而产生误导性高/低分，直接跳过（主标的本地 OOS 门照常把关）。
        _multi_used = bool(result.get("multi_symbol"))
        if cascade_factor_values is not None:
            self._strategy_drl_status["cross_oos"] = None
            log.info("[evolve] 级联因子注入中，跳过跨标传导 OOS")
            cross_oos = None
        elif _multi_used:
            self._strategy_drl_status["cross_oos"] = None
            log.info("[evolve] 多标的混合训练中（额外标的 %s），"
                     "跨标 OOS 不构成样本外验证，跳过",
                     result.get("multi_extra_names") or [])
            cross_oos = None
        else:
            cross_oos = await self._cross_symbol_oos(result["agent"], symbol,
                                                     int(train_cfg["state_window"]),
                                                     train_cfg=train_cfg)
        self._strategy_drl_status["cross_oos"] = cross_oos
        if cross_oos is not None and cross_oos.get("ret", 0.0) <= 0:
            log.warning("[evolve] 策略DRL 跨标 OOS 未通过（%s: %.4f），仅存档不设为最佳",
                        cross_oos.get("symbol"), cross_oos.get("ret", 0.0))
            try:
                # P4-C2：仅存档不占版本名额（跨标被拦模型非有效改进）
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "cross_oos_rejected": True,
                                              "cross_symbol": cross_oos.get("symbol"),
                                              "cross_oos_ret": cross_oos.get("ret", 0.0),
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 跨标被拦模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：跨标安检拦截",
                             f"策略DRL 新模型跨标收益 {cross_oos.get('ret', 0.0):.4f} 非正"
                             f"（{cross_oos.get('symbol')}），已保留原最佳模型")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="cross_rejected",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0,
                lineage=("challenger" if is_challenger else "main"))
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "cross_oos_rejected": True,
            }, source="evolve_engine"))
            return

        # 5.9 部署长窗硬门（2026-09-22，§20 落地）：冠军轮/晋升轮候选模型须过
        # 30 天长窗回放（30d>0）才允许部署；FAIL 仅存档不部署（保留旧 best），
        # 防高换手短窗模型（v50/52/53/55 类长窗 -10%~-56%）顶替长窗稳健模型。
        # 挑战者未晋升（只更新谱系不部署）跳过本门。
        _champion_round = bool(not is_challenger or promote)
        if _champion_round and getattr(settings, "evolve_strategy_drl_deploy_requires_longwindow", False):
            _lw_ok, _lw_reason = await self._candidate_longwindow_gate(
                result["agent"], symbol, train_cfg,
                factor_composite=(result or {}).get("factor_composite"),
                factor_mu=float((result or {}).get("factor_mu") or 0.0),
                factor_sd=float((result or {}).get("factor_sd") or 1.0))
            self._strategy_drl_status["longwindow_deploy_gate"] = _lw_reason
            if not _lw_ok:
                log.warning("[evolve] 策略DRL 未过部署长窗门，仅存档不部署：%s", _lw_reason)
                try:
                    await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                            meta={"fitness": new_fitness, "timestamp": time.time(),
                                                  "data_source": data_source_used,
                                                  "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                                  "longwindow_rejected": True,
                                                  "longwindow_reason": _lw_reason,
                                                  "lineage": ("challenger" if is_challenger else "main")})
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 长窗被拦模型存档失败: %s", e)
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness,
                    status="longwindow_rejected",
                    oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                    decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                    position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0,
                    lineage=("challenger" if is_challenger else "main"))
                await self.bus.publish(Event(EventType.SYSTEM, {
                    "kind": "evolve_train",
                    "model": "strategy_drl",
                    "fitness": new_fitness,
                    "longwindow_rejected": True,
                }, source="evolve_engine"))
                return

        # 6. 保存最佳模型（含训练元数据）
        # P0-2：demo 数据训练只存档不写 best（is_best=False），
        # 且跳过兼容格式 flat 写入（rl_adaptive 部署端读取的正是 flat 文件，
        # 若被合成数据模型覆盖，实盘会被未经验证的权重接管）。
        # P0-族群：挑战者轮未晋升 → 只更新挑战者谱系槽位（strategy_drl_alt，
        # 独立 best/flat，不触碰部署端），记 challenger_kept 后收尾；
        # 晋升 → 与冠军轮同样落盘部署（下方 else 分支）。
        if is_challenger and not promote:
            # 挑战者谱系自保护：新模型若相对挑战者自身锚点退化（< 自身×threshold），
            # 只存档不覆盖 alt 槽位——谱系质量优先，退化模型不进族群。
            alt_anchor = self.zoo.best_info("strategy_drl_alt")
            # 2026-10 审计修复（MED）：挑战者自回退须有身份守卫——配置变更（奖励口径/
            # 决策间隔/因子注入等）后，新候选与旧 alt 锚点的 fitness 口径可能不同量级，
            # 硬比较会"新口径更小的 exponent"误判退化并丢谱系。冠军路径已有守卫，
            # 挑战者此前没有。
            _alt_mismatch = _identity_mismatch(alt_anchor, _identity)
            alt_prev = alt_anchor.get("fitness")
            if (_alt_mismatch is None and alt_prev is not None
                    and _should_rollback(new_fitness, float(alt_prev), self._rollback_threshold)):
                log.info("[evolve] 策略DRL挑战者相对自身退化（%.4f < %.4f×%.2f），只存档不更新谱系",
                         new_fitness, float(alt_prev), self._rollback_threshold)
                try:
                    await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], "strategy_drl_alt",
                                            meta={"fitness": new_fitness, "timestamp": time.time(),
                                                  "data_source": data_source_used,
                                                  "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                                  "lineage": "challenger", "challenger_degraded": True})
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 挑战者退化存档失败: %s", e)
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness,
                    status="challenger_degraded",
                    oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                    decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                    position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0,
                    lineage="challenger")
                await self.bus.publish(Event(EventType.SYSTEM, {
                    "kind": "evolve_train",
                    "model": "strategy_drl",
                    "fitness": new_fitness,
                    "lineage": "challenger",
                    "status": "challenger_degraded",
                }, source="evolve_engine"))
                return
            log.info("[evolve] 策略DRL挑战者更新自身谱系（不晋升，冠军部署不动）")
            try:
                await asyncio.to_thread(self.zoo.save_agent, result["agent"], "strategy_drl_alt",
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "base_oos_ret": anchor.get("oos_ret"),
                                              "base_fitness": anchor.get("fitness"),
                                              "lineage": "challenger", **_identity})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 挑战者谱系保存失败: %s", e)
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="challenger_kept",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0,
                lineage="challenger")
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "lineage": "challenger",
                "status": "challenger_kept",
            }, source="evolve_engine"))
            return

        if data_source_used == "demo":
            try:
                # P4-C2：demo 模型仅存档不占版本名额（合成数据模型非真实改进）
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "demo_blocked": True, "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] demo 模型存档失败: %s", e)
            log.warning("[evolve] 策略DRL使用演示数据训练，仅存档不部署（防实盘被未验证模型接管）")
        else:
            # P0-族群加强：挑战者晋升时角色互换——旧冠军不丢弃，转入挑战者
            # 槽位继续进化（PBT top-k 保留思想）。A/B 实验（scripts/ab_population3.py）
            # 显示互换比"晋升即丢弃旧冠军" final 提升约 +92%（0.0161→0.0310）：
            # 两条谱系都保持"曾当过冠军"的强度，多样性不减、信息不丢。
            # 必须在本轮覆盖冠军槽位**之前**先读旧冠军权重，否则读到新模型。
            if is_challenger:
                try:
                    old_champ = self.zoo.load_agent("strategy_drl")
                    if old_champ is not None:
                        await asyncio.to_thread(self.zoo.save_agent, old_champ, "strategy_drl_alt",
                                                meta={"fitness": anchor.get("fitness"),
                                                      "timestamp": time.time(),
                                                      "data_source": anchor.get("data_source") or data_source_used,
                                                      "oos_ret": anchor.get("oos_ret"),
                                                      "lineage": "challenger",
                                                      "swapped_from_champion": True,
                                                      **_identity})
                        log.info("[evolve] 策略DRL挑战者晋升：旧冠军已转入挑战者槽位（角色互换）")
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 旧冠军转入挑战者槽位失败: %s", e)
            _accept_meta = {
                                        "fitness": new_fitness, "timestamp": time.time(),
                                        "data_source": data_source_used,
                                        "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                        # 2026-10 审计修复（HIGH）：回退/晋升必须用**超额**
                                        # 口径（oos_excess）而非绝对收益——不同滚动窗口的
                                        # 绝对值不可比（下跌窗口真候选必更低→被误回退；
                                        # 上涨窗口差候选逃过回退）。fitness 已是 excess 口径，
                                        # OOS 侧同样存超额供 _should_rollback 比较。
                                        "oos_excess": (result.get("oos_report") or {}).get("oos_excess"),
                                        # 2026-10 审计修复（MED-3）：excess 统一为**逐段中位数**
                                        # 口径。锚点记录 basis——旧锚点（median差/无标记）与新口径
                                        # 不可比，回退比较会 fallback 到 fitness。
                                        "oos_excess_basis": (result.get("oos_report") or {}).get(
                                            "oos_excess_basis", "median_diff"),
                                        # P2 续训护栏审计：记录起点（旧模型）的 OOS/收益，
                                        # 让"续训后是否真进步"可追溯。接受路径已保证
                                        # new_oos ≥ prev_oos×threshold（rollback 守卫），
                                        # 这里把对比写进 meta 供前端/审计展示。
                                        "base_oos_ret": anchor.get("oos_ret"),
                                        "base_fitness": anchor.get("fitness"),
                                        "base_version": anchor.get("version"),
                                        "lineage": ("challenger_promoted" if is_challenger else "main"),
                                        # 级联因子部署元数据随版本持久化（ModelZoo
                                        # _DEPLOYMENT_META_KEYS 提升到 flat）：
                                        # 否则 restore_deployment 重建 flat 时因子配方/mu/sd
                                        # 丢失，rl_adaptive 与评估脚本无法重建 state_dim 状态。
                                        "factor_composite": (result.get("factor_composite")
                                                             or None),
                                        "factor_expression": str(result.get("factor_expression", "") or "").strip() or None,
                                        "factor_mu": float(result.get("factor_mu", 0.0) or 0.0),
                                        "factor_sd": float(result.get("factor_sd", 1.0) or 1.0),
                                        # 2026-10 审计修复（HIGH-1）：决策节奏/死区随版本
                                        # 载荷持久化（ModelZoo _DEPLOYMENT_META_KEYS 已含
                                        # 这两键，restore_deployment 重建 flat 时自动提升）。
                                        # 此前只有 flat 运行时补丁写它们，回退后丢失 →
                                        # rl_adaptive 按 k=1/zone=0.02 跑 k=8 训练的模型。
                                        "decision_interval": max(1, int(train_cfg.get("decision_interval", 1) or 1)),
                                        "min_trade_zone": float(train_cfg.get("min_trade_zone", 0.05)),
                                        "short_enabled": bool(train_cfg.get("short_enabled", False)),
                                        # 2026-10（MED-1）：奖励/基准口径随版本持久化，
                                        # 手动评估（evaluate_agent/_model_eval_cfg）据此重建
                                        "reward_mode": str(train_cfg.get("reward_mode", "absolute") or "absolute"),
                                        "benchmark_position": float(train_cfg.get("benchmark_position", 1.0)),
                                        "risk_coef": float(train_cfg.get("risk_coef", 0.5)),
                                        "slippage": float(train_cfg.get("slippage", 0.0005)),
                                        **_identity}
            # 2026-10 审计修复（LOW）：接受路径 save_agent 加 try——此前异常直接逸出
            # 本轮（mark_deployment/_log_round 全跳过），ModelZoo 却已先写 meta
            # （next_version 递增 + versions 追加）→ 版本历史留无文件幽灵条目。
            try:
                await asyncio.to_thread(self.zoo.save_agent, result["agent"],
                                        "strategy_drl", meta=_accept_meta)
            except Exception as e:  # noqa: BLE001
                log.exception("[evolve] 策略DRL接受路径存档失败: %s", e)
                self._strategy_drl_status["last_error"] = f"模型存档失败: {e}"
                _err_oos = result.get("oos_report") or {}
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness, status="save_error",
                    oos_ret=_err_oos.get("oos_ret", 0.0) or 0.0,
                    decay=_err_oos.get("decay", 0.0) or 0.0,
                    position_ratio=_err_oos.get("oos_position_ratio", 0.0) or 0.0,
                    lineage=("challenger" if is_challenger else "main"))
                return
            self._mark_deployment_status("strategy_drl", outcome="accepted",
                                         candidate_fitness=new_fitness)
        try:
            from core.notify import notify
            _oos = (result.get("oos_report") or {}).get("oos_ret")
            await notify(self.db, "进化引擎：策略DRL模型更新",
                         f"fitness {prev_fitness:.4f} → {new_fitness:.4f}"
                         f"，OOS 收益 {_oos if _oos is not None else '无'}（{data_source_used} 数据）")
        except Exception:  # noqa: BLE001
            pass
        if data_source_used == "demo":
            log.warning("[evolve] 策略DRL使用演示数据训练完成（仅供测试，勿用于实盘），收益=%.4f", new_fitness)
            # P0-2：demo 模型不写兼容格式、不注册，直接结束本轮
            log.warning("[evolve] 策略DRL使用演示数据训练，跳过元数据写入与策略注册（防止误用实盘）")
            # P2-13：demo 轮次落库（标记 demo_blocked）
            _oos_report_d = result.get("oos_report") or {}
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                "demo", new_fitness,
                status="demo_blocked",
                oos_ret=_oos_report_d.get("oos_ret", 0.0) or 0.0,
                decay=_oos_report_d.get("decay", 0.0) or 0.0,
                position_ratio=_oos_report_d.get("oos_position_ratio", 0.0) or 0.0,
                lineage=("challenger" if is_challenger else "main"))
            return
        else:
            # 6.1 在兼容格式 JSON 中追加训练元数据（与 web/api/drl.py 同格式）
            try:
                flat_path = self.zoo._flat_path("strategy_drl")
                if flat_path.exists():
                    import json

                    def _write_flat_meta() -> None:
                        with open(flat_path, "r", encoding="utf-8") as f:
                            _md = json.load(f)
                        _md["train_meta"] = {
                            "data_source": data_source_used,
                            "symbol": symbol,
                            "timeframe": self._effective_timeframe(),
                            # 2026-10 审计修复（LOW）：记录实际训练深度（_keep_bars≈
                            # history_bars 30000）而非滚动窗（5000）——元数据曾与真实
                            # 训练窗口口径不符，评估/审计据此误解模型见过的数据量。
                            "limit": _keep_bars,
                            "episodes": getattr(settings, "evolve_strategy_drl_episodes", 8),
                            "best_ret": result.get("best_ret"),
                            "best_val_ret": result.get("best_val_ret"),
                            "oos_report": result.get("oos_report"),
                        }
                        _md["factor_expression"] = result.get("factor_expression", "")
                        _md["factor_mu"] = float(result.get("factor_mu", 0.0))
                        _md["factor_sd"] = float(result.get("factor_sd", 1.0))
                        if result.get("factor_composite"):
                            # 级联因子复算配方：实盘 rl_adaptive 据此在滚动缓冲
                            # 上复算组合因子（曾只读写死数组，实盘因子列恒 None）
                            _md["factor_composite"] = result["factor_composite"]
                        _md["min_trade_zone"] = float(train_cfg.get("min_trade_zone", 0.05))
                        # 决策间隔（2026-10 根本性逻辑错误修复）：部署端 rl_adaptive 必须
                        # 按同一节奏决策，否则"训练每 k 根决策、实盘每根决策"，
                        # 换手被放大 k 倍、成本吞掉训练学到的收益。
                        _md["decision_interval"] = int(train_cfg.get("decision_interval", 1) or 1)
                        _md["pine_code"] = result.get("pine_code", "")
                        _md["pine_note"] = result.get("pine_note", "")
                        # run25 R5 修复：原 open("w")+json.dump 非原子重写，
                        # 写盘窗口内读取方（rl_adaptive/回测/load_agent flat 回退）
                        # 可能读到截断 JSON；改走 ModelZoo 原子写（tmp+fsync+replace）
                        ModelZoo._atomic_write_bytes(flat_path,
                                                     ModelZoo._json_bytes(_md))

                    await asyncio.to_thread(_write_flat_meta)
            except Exception as e:
                log.warning("[evolve] 模型元数据写入失败: %s", e)

            # 7. 注册为可用策略（演示数据训练的模型不注册，防止误用实盘）
            try:
                await self._register_rl_evolve()
            except Exception as e:
                log.warning("[evolve] 策略注册失败: %s", e)

        log.info("[evolve] 策略DRL训练完成，收益=%.4f，已保存并注册", new_fitness)

        # 8. 发布系统事件
        # P2-13：训练轮次落库
        _oos_report = result.get("oos_report") or {}
        await self._log_round(
            "strategy_drl", symbol, self._effective_timeframe(),
            data_source_used, new_fitness,
            status="ok",
            oos_ret=_oos_report.get("oos_ret", 0.0) or 0.0,
            decay=_oos_report.get("decay", 0.0) or 0.0,
            position_ratio=_oos_report.get("oos_position_ratio", 0.0) or 0.0,
            lineage=("challenger_promoted" if is_challenger else "main"))
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "evolve_train",
            "model": "strategy_drl",
            "fitness": new_fitness,
            "lineage": ("challenger_promoted" if is_challenger else "main"),
        }, source="evolve_engine"))

    # ---- 数据获取 ----

    def _data_incremented(self, name: str, df, symbol: str = "",
                          is_demo: bool = False) -> bool:
        """P2-11 增量门：比较数据末尾时间戳是否较上次训练有新变化。

        返回 False 表示"数据无足够新增 K 线，应跳过本轮训练"（连续训练场景下
        60s 间隔 + 5 分钟数据缓存时数据往往没更新，空转重训纯属浪费）。
        - 首次训练（无记录）恒为 True。
        - 有新增 K 线但不足 evolve_min_new_bars 根时仍跳过（等待积累，
          1h 周期默认 2 根 ≈ 2 小时；面板可调大以降低训练频率）。
        - P4-E1：键含 symbol——此前只按管线名键控，标的轮换后拿上一标的的
          tail 比较，min_new_bars 阈值被跨标时间戳绕开（阈值形同虚设）。
        - P4-E1：尾时间戳倒退（数据源清空/截断/跨源切换）视为数据源重置——
          更新记录并放行本轮，避免管线因 prev 永不回落而无限跳过。
        - is_demo：演示数据轮直接放行且**不更新** prev（合成数据的尾时间戳
          ≈now 会污染 prev，导致故障恢复后的首个真实训练被推迟最多
          min_new_bars 根；同周期连续 demo 轮尾戳相同还会触发自身限频）。
        """
        if is_demo:
            return True  # demo 轮不参与增量门、不写 prev
        try:
            idx = df.index
            # 按索引自身单位换算成 epoch 秒：pandas 3 起 DatetimeIndex 可以是
            # s/ms/us/ns 精度（to_datetime(unit="ms") 不再升频到 ns），
            # 用 asi8 // 10**9 假定纳秒会让 ms 索引下的秒数缩水 1000 倍。
            if idx.dtype.kind != "M":
                return True  # 非时间戳索引（demo 兜底数据等），不拦截
            sec = idx.values.astype("datetime64[s]").astype("int64")
            tail_ts = float(sec[-1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return True  # 无法取时间戳，不拦截
        tf = self._effective_timeframe()
        # 2026-10 审计修复（LOW）：键含交易所——_fetch_latest_data 缓存键已含 exchange，
        # 切换数据源后若新旧源尾时间戳恰好相同（同时刻收盘），`tail_ts==prev` 会判定
        # "无新增"而永久跳过，直到新源来新根。加 exchange 后换源即换键（视为首次放行）。
        key = f"{name}:{symbol}:{tf}:{getattr(settings, 'default_exchange', 'binance')}" if symbol else name
        prev = self._last_tail_ts.get(key)
        if prev is None:
            self._last_tail_ts[key] = tail_ts
            return True
        if tail_ts == prev:
            return False  # 末尾时间戳无变化，无新根
        if tail_ts < prev:
            # 数据源已重置/缩短（本地K线库清空后回填更短历史、跨源切换）：
            # 旧 prev 永不回落会导致无限跳过，更新记录并放行本轮恢复训练
            log.info("[evolve] %s 数据尾时间戳倒退（%.0f < 上次 %.0f），"
                     "视为数据源重置，放行本轮", name, tail_ts, prev)
            self._last_tail_ts[key] = tail_ts
            return True
        try:
            # 统计自上次训练以来的新增根数（与 tail_ts 同为 epoch 秒）
            n_new = int((sec > prev).sum())
            # 增量门槛 = max(手工 min_new_bars, 按周期自动换算)。
            # P1：evolve_min_train_gap_sec>0 时按训练周期换算所需新增根数——
            # 训练间隔至少覆盖该秒数（5m 周期下 2 根=10 分钟一轮仍偏密，
            # 换算后与训练成本/行情节奏更贴合）。0 则仅用 min_new_bars。
            min_bars = max(1, int(getattr(settings, "evolve_min_new_bars", 2)))
            gap_sec = int(getattr(settings, "evolve_min_train_gap_sec", 0) or 0)
            if gap_sec > 0:
                try:
                    from backtest.data_loader import _TF_SECONDS
                    tf_sec = int(_TF_SECONDS.get(tf, 1))
                    need_by_gap = max(1, int(math.ceil(gap_sec / max(tf_sec, 1))))
                    min_bars = max(min_bars, need_by_gap)
                except Exception:  # noqa: BLE001
                    pass  # 换算失败回落 min_bars
            if n_new < min_bars:
                return False  # 有变化但根数不足，等待积累
        except Exception:  # noqa: BLE001
            # 2026-10 审计修复（MED）：统计异常时也把 tail_ts 记入 prev 再放行——
            # 否则同一批"新"K线会被下一轮用旧 prev 重复计数，门槛反复满足、
            # 同一窗口重复训练一轮。
            self._last_tail_ts[key] = tail_ts
            return True
        self._last_tail_ts[key] = tail_ts
        return True

    async def _log_round(self, model: str, symbol: str, timeframe: str,
                         data_source: str, fitness: float, status: str = "ok",
                         oos_ret: float = 0.0, decay: float = 0.0,
                         position_ratio: float = 0.0,
                         selected_factors: Optional[list] = None,
                         round_no: Optional[int] = None,
                         audit: Optional[dict] = None,
                         lineage: str = "") -> None:
        """P2-13：训练轮次落库（evolve_rounds 表）。失败仅记日志，不影响训练流程。

        Task 5：audit 为人工操作的审计载荷（manual_rollback 等），与
        selected_factors 一样以 JSON 文本落库（老行 read 端兼容两种格式）。
        lineage：族群谱系（strategy_drl 冠军/挑战者），前端曲线据此分系显示。
        """
        # 连续拒绝计数先于落库：锁死判定要在落库失败时依然有效（前端提示依赖它）
        self._rounds_total[model] = int(self._rounds_total.get(model, 0)) + 1
        if status == "ok":
            self._reject_streak[model] = 0
        elif status == "demo_blocked":
            # 交易所故障的演示数据轮是"降级训练"不是"模型被拒绝"：
            # 计入 streak 会让故障期误报锁死、诱导用户重置锚点（P1-3）
            pass
        else:
            # 2026-10 审计修复（MED）：challenger 谱系不计入（健康交替不该把冠军
            # 顶进 from-scratch）；main 谱系照旧。
            _bump_reject_streak(self, model, lineage=lineage)
        try:
            from core.database import EvolveRound
            from sqlalchemy import select, func, delete
            if round_no is None:
                async with self.db.session() as s:
                    cur_max = (await s.execute(
                        select(func.max(EvolveRound.round_no))
                        .where(EvolveRound.model == model))).scalar_one()
                    round_no = int(cur_max or 0) + 1
            async with self.db.session() as s:
                s.add(EvolveRound(
                    model=model, symbol=symbol, timeframe=timeframe,
                    data_source=data_source, round_no=round_no,
                    fitness=float(fitness), oos_ret=float(oos_ret),
                    decay=float(decay), position_ratio=float(position_ratio),
                    selected_factors=json.dumps(list(selected_factors or []),
                                                ensure_ascii=False),
                    status=status,
                    audit_json=json.dumps(audit or {}, ensure_ascii=False),
                    lineage=str(lineage or ""),
                ))
                # P2-9：每模型保留最近 N 轮（含 demo/oos_rejected/rollback），
                # 防止 evolve_rounds 表长期运行无界增长、重启全表扫描变慢
                keep = max(100, int(getattr(settings, "evolve_rounds_keep", 500)))
                # run12 E1：cutoff 查询 + delete 合并为单条 DELETE（标量子查询），
                # 省一次 aiosqlite 往返（约 -17% 全序列）；无 cutoff 时（表未超
                # keep 行）子查询返回 NULL，`id < NULL` 恒假 → 不删行，语义等价
                # （.optim/probe_evolve_merge.py：超阈/未超阈边界逐行对比 PASS）
                cutoff_subq = (select(EvolveRound.id)
                               .where(EvolveRound.model == model)
                               .order_by(EvolveRound.id.desc())
                               .offset(keep).limit(1).scalar_subquery())
                await s.execute(delete(EvolveRound)
                                .where(EvolveRound.model == model,
                                       EvolveRound.id < cutoff_subq))
                await s.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 训练轮次落库失败(%s): %s", model, e)

    async def _cross_symbol_oos(self, agent, train_symbol: str,
                                state_window: int,
                                train_cfg: Optional[dict] = None) -> Optional[dict]:
        """P1-8：跨标的传导 OOS（低权重附加门）。

        模型在另一个训练标的（非当前标的）上跑一次 greedy 样本外评估，
        检验是否学到了"币种无关"的交易规律而非单标的特有过拟合。
        仅当训练未注入级联因子（state_dim 无因子列）时可执行；
        注入因子时无法在跨标数据上重建同维状态，跳过并返回 None。

        2026-10 审计修复（HIGH-1c）：环境必须用训练同源口径（policy_env_kwargs
        语义：k=决策间隔/excess/min_trade_zone）——此前硬编码 k=1/absolute/zone=0.05
        评估 k=8/excess 训练的模型，跨标门与主 OOS 门自相矛盾。
        """
        if not getattr(settings, "evolve_cross_symbol_oos", True):
            return None
        others = [s for s in self._symbols if s != train_symbol]
        if not others:
            return None
        sym = others[0]
        try:
            df = await self._fetch_latest_data(symbol=sym)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 跨标评估数据获取失败(%s): %s", sym, e)
            return None
        if df is None or len(df) < max(300, int(self._rolling_window * 0.1)):
            return None
        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]
        from drl.env import run_episode

        def _run_cross_oos() -> Optional[dict]:
            """在工作线程中构造环境并跑 greedy rollout：
            TradingEnv 构造会做滚动特征预计算、run_episode 是纯同步 CPU 前向，
            留在主协程会阻塞事件循环（WebSocket/下单/其他管线全卡顿）。
            与 train_drl 的 asyncio.to_thread 隔离口径一致。"""
            _tc = train_cfg or {}
            env = _make_cross_oos_env(
                df,
                state_window=int(state_window),
                decision_interval=int(_tc.get("decision_interval", 1) or 1),
                min_trade_zone=float(_tc.get("min_trade_zone", 0.05)),
                short_enabled=bool(_tc.get("short_enabled", False)),
                reward_mode=str(_tc.get("reward_mode", "absolute") or "absolute"),
                benchmark_position=float(_tc.get("benchmark_position", 1.0)),
                risk_coef=float(_tc.get("risk_coef", 0.5)))
            traj = run_episode(env, lambda s: agent.greedy_action(s))
            return {"symbol": sym, "ret": float(traj["total_ret"]),
                    "position_ratio": float(traj["final_position_ratio"])}

        try:
            return await asyncio.to_thread(_run_cross_oos)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 跨标评估失败(%s): %s", sym, e)
            return None

    def _load_cascade_factor_values(self, df, symbol: str = "") -> Optional[Any]:
        """加载级联组合因子值（npy 快路径 → 已部署权重复算兜底）。

        3号修复（2026-09-19）：npy 的 companion tail_ts 严格相等校验在滚动窗口
        右移/连续回退（npy 长期不更新）场景下必然失败，导致 strategy_drl 永远
        注入不到因子特征（当日 21 拒/0 注入、32 连拒）。文件快路径失败时，
        用已部署权重在**当轮 df** 上复算——与 factor_miner 训练端同式
        （composite_factor z_mode="expanding"）、与 rl_adaptive 实盘复算同口径，
        天然按当前窗口对齐（无错位、无前视）。返回 numpy 数组（长度 == len(df)）。
        """
        values = self._load_cascade_factor_values_from_files(df, symbol=symbol)
        if values is not None:
            return values
        return self._recompute_cascade_factor_values(df, symbol=symbol)

    def _recompute_cascade_factor_values(self, df, symbol: str = "") -> Optional[Any]:
        """用已部署组合因子权重在当轮 df 上复算因子值（npy 陈旧/缺失时的兜底）。

        权重文件（_cascade_weights_<symbol>.json）是实盘 rl_adaptive 复算所用的
        同一配方；复算公式与 factor_miner 训练端 composite_factor(z_mode="expanding")
        一致：expanding z 只用截至当期统计量（无前视），权重按绝对值归一，
        结果按当轮 df 索引对齐，长度必然 == len(df)。失败返回 None（不阻断）。
        """
        try:
            import numpy as np
            from factors.engine import compute_factor_matrix
            from factors.mining import composite_factor
            weights = self._load_cascade_factor_recipe(symbol)
            if not weights or not weights.get("weights"):
                return None
            w = weights["weights"]
            mat = compute_factor_matrix(df, keys=list(w))
            combo = composite_factor(mat, w, method="ic", z_mode="expanding")
            vals = np.asarray(combo.to_numpy(float), dtype=float).reshape(-1)
            if len(vals) != len(df):
                log.warning("[evolve] 复算级联因子长度 %d != 数据 %d，放弃注入",
                            len(vals), len(df))
                return None
            log.info("[evolve] 级联因子 npy 不可用，改用已部署权重复算（%s %d 根）",
                     symbol or "?", len(df))
            return vals
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 复算级联因子失败（跳过注入）: %s", e)
            return None

    def _load_cascade_factor_values_from_files(self, df, symbol: str = "") -> Optional[Any]:
        """（旧实现）从 npy + companion 文件加载级联因子值，仅尾戳一致时返回。

        P4-A1：按标的分文件读取（_cascade_factor_values_{symbol}.npy），
        只注入与当前训练标的匹配的因子——彻底杜绝跨标错配（此前单一文件 +
        仅长度校验：BTC 因子被误当 ETH 因子注入，长度一致时静默通过）。
        P4-E1：同标但跨窗口的错位同样危险——因子值与 df 长度一致但窗口已右移
        N 根（数据 TTL 过期后取到更新的 K 线），仅长度校验会静默注入错位数据。
        保存端写了 companion JSON 记录窗口尾时间戳；加载端校验一致才注入。
        返回 numpy 数组（长度 == len(df)），无匹配级联因子时返回 None。
        """
        try:
            import numpy as np

            def _tail_ts():
                """当前 df 窗口尾时间戳（epoch 秒，兼容 DatetimeIndex/int index）。"""
                idx = df.index[-1]
                try:
                    return int(idx.timestamp())
                except Exception:  # noqa: BLE001
                    return int(idx) if idx is not None else None

            def _match_window(values_len: int) -> bool:
                if values_len != len(df):
                    log.warning("[evolve] 级联因子长度 %d 与当前数据 %d 不一致，跳过注入",
                                values_len, len(df))
                    return False
                # P4-E1：有 companion JSON 时校验窗口尾时间戳；无 companion（旧版本
                # 遗留）退化为仅长度校验（无法验证，保守放行以兼容升级前文件）。
                meta_path = cascade_path.with_suffix(".json") if symbol else None
                if meta_path is not None and meta_path.exists():
                    try:
                        import json as _json
                        rec = _json.loads(meta_path.read_text(encoding="utf-8"))
                        if rec.get("tail_ts") != _tail_ts():
                            log.warning(
                                "[evolve] 级联因子窗口尾时间戳 %s 与当前数据 %s 不一致"
                                "（窗口已右移？），拒绝注入", rec.get("tail_ts"), _tail_ts())
                            return False
                    except Exception as e:  # noqa: BLE001
                        log.warning("[evolve] 级联因子时间戳校验失败（放行）: %s", e)
                return True

            if symbol:
                cascade_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.npy"
                if not cascade_path.exists():
                    return None
                values = np.asarray(np.load(str(cascade_path)), dtype=float).reshape(-1)
                if not _match_window(len(values)):
                    return None
                return values
            # 无标的参数（旧调用路径兼容）：先试当前轮换标的，再回退旧单文件
            cur = self._next_symbol()
            cur_path = self.zoo.models_dir / f"_cascade_factor_values_{cur.replace('/', '_')}.npy"
            if cur_path.exists():
                cascade_path = cur_path
                values = np.asarray(np.load(str(cur_path)), dtype=float).reshape(-1)
                if _match_window(len(values)):
                    return values
            legacy = self.zoo.models_dir / "_cascade_factor_values.npy"
            if legacy.exists():
                cascade_path = legacy
                values = np.asarray(np.load(str(legacy)), dtype=float).reshape(-1)
                if len(values) == len(df):
                    return values
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 加载级联因子失败: %s", e)
            return None

    def _load_cascade_factor_recipe(self, symbol: str = "") -> Optional[dict]:
        """加载级联组合因子的复算配方（选中因子权重映射）。

        与 _cascade_factor_values_{symbol}.npy 同标同轮保存
        （_cascade_weights_{symbol}.json），部署端把它写进模型文件，
        实盘 rl_adaptive/meta_controller 按配方在滚动缓冲上复算组合因子。
        返回 {"weights": {因子key: 权重}}，找不到/解析失败返回 None（不阻断）。
        """
        try:
            if not symbol:
                return None
            import json as _json
            weights_path = (self.zoo.models_dir
                            / f"_cascade_weights_{symbol.replace('/', '_')}.json")
            if not weights_path.exists():
                return None
            weights = _json.loads(weights_path.read_text(encoding="utf-8"))
            if not isinstance(weights, dict) or not weights:
                return None
            clean = {str(k): float(v) for k, v in weights.items()
                     if isinstance(v, (int, float)) and v}
            if not clean:
                return None
            return {"weights": clean}
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 加载级联因子配方失败: %s", e)
            return None

    def _next_symbol(self) -> str:
        """返回当前训练标的（两管线共享，不递增）。

        P4-A1：标的轮换改由 _mark_pipeline_done 在「全部管线均完成当前标的
        训练后」统一推进，保证 factor_miner 产出的级联因子与 strategy_drl
        训练标的始终一致（此前各自递增轮换 → 级联因子跨标错配）。
        """
        return self._symbols[self._symbol_idx % len(self._symbols)]

    def _mark_pipeline_done(self, pipeline: str) -> None:
        """标记一条管线已在当前标的完成训练；全部管线齐全后切换到下一标的。

        仅在真实执行了一次训练（含失败/跳过后的完成）后调用，挂在各
        _train_*_once 的 finally 里。因子挖掘优先级最低（它是链路源头，
        需要先给下游产出因子），因此只有全部管线都完成后才推进标的。
        目标数取 len(self._models)（元策略训练管线移除后为 2），避免再硬编码。
        """
        self._cycle_pipelines_done.add(pipeline)
        if len(self._cycle_pipelines_done) >= len(self._models):
            self._cycle_pipelines_done.clear()
            self._symbol_idx += 1
            nxt = self._symbols[self._symbol_idx % len(self._symbols)]
            log.info("[evolve] 全部管线均完成 %s 训练，切换至 %s",
                     self._symbols[(self._symbol_idx - 1) % len(self._symbols)], nxt)

    def _generate_symbol_demo(self, symbol: str, n: int, timeframe: str) -> Optional[Any]:
        """生成指定标的的合成K线数据（交易所不可达时兜底），
        使用更接近真实价格水平的参数，让训练更有意义。"""
        # 各标的参数：起始价、波动率
        params = {
            "BTC/USDT": {"start_price": 60000.0, "vol": 0.015},
            "ETH/USDT": {"start_price": 3000.0, "vol": 0.02},
            "SOL/USDT": {"start_price": 150.0, "vol": 0.03},
        }
        p = params.get(symbol, {"start_price": 100.0, "vol": 0.01})
        # 每次生成不同种子，保证训练多样性
        seed = random.randint(0, 99999)
        return generate_demo(n=n, start_price=p["start_price"], vol=p["vol"],
                             timeframe=timeframe, seed=seed)

    async def _fetch_latest_data(self, symbol: str = "", bars: int = 0) -> Optional[Any]:
        """获取最新行情数据（用于训练），带 5 分钟内存缓存避免频繁拉取。
        本地K线库优先（binance 公开数据域直连，增量拉取缺失段）；
        失败返回 None（由调用方决定是否使用演示数据兜底）。

        Args:
            symbol: 标的（如 "BTC/USDT"），留空使用 settings.default_symbol。
            bars: 本次需要的K线根数（0=沿用 evolve_rolling_window）。策略 DRL 传
                evolve_strategy_drl_history_bars 拉满多年历史——旧行为只取最近
                rolling_window 根（5m 下仅 17 天）并在同一段上反复重训，
                本地 3.4 年历史（30042 根/标的）从未进入训练。
        """
        from backtest.data_loader import load_klines_cached

        symbol = symbol or settings.default_symbol
        # 训练时间周期：面板独立配置（evolve_timeframe）优先，缺省回落 default_timeframe
        timeframe = self._effective_timeframe()
        n_want = max(int(bars or 0) or int(self._rolling_window), 1000)
        # 缓存键必须含深度：否则因子挖掘先按 5000 根拉取并写缓存后，
        # 策略 DRL 按 30000 根取数据会命中那份浅缓存（静默退化成旧行为）。
        cache_key = f"{settings.default_exchange}:{symbol}:{timeframe}:{n_want}"
        now = time.time()
        # 缓存命中且在 TTL 内
        if cache_key in self._data_cache:
            cached_ts, cached_df = self._data_cache[cache_key]
            if now - cached_ts < self._data_cache_ttl:
                log.debug("[evolve] 使用缓存数据（%ds 前获取）", now - cached_ts)
                return cached_df

        # P0-1：修复窗口上限 bug——原先 min(max(rolling_window,1000),3000) 把配置的
        # 5000 根永远压到最多 3000 根。现在拉满所需深度，且 load_klines_cached
        # 支持 max_candles 历史回填（本地不足时自动翻页补足）。
        limit = n_want
        max_candles = n_want

        # P4-E1：拉数串行锁——两管线 + 跨标 OOS 并发打交易所会放大 429；
        # 统一串行化后同一时刻只有一条 fetch 在飞（429 重试逻辑见 data_loader）。
        try:
            async with self._fetch_lock:
                df = await load_klines_cached(settings.default_exchange, symbol, timeframe,
                                              limit=limit, max_candles=max_candles)
            if df is not None and not df.empty:
                log.info("[evolve] K线数据获取成功：%d 根K线 (%s %s)",
                         len(df), symbol, timeframe)
                # 写入缓存
                # P4-E1：容量上限——symbol×timeframe 组合变更后旧键永不清理，
                # 长时间运行+频繁改配置下无界增长；超过上限时淘汰最旧键。
                self._data_cache[cache_key] = (now, df)
                if len(self._data_cache) > _DATA_CACHE_MAX_KEYS:
                    oldest_key = min(self._data_cache, key=lambda k: self._data_cache[k][0])
                    self._data_cache.pop(oldest_key, None)
                return df
        except Exception as e:
            log.warning("[evolve] 拉取K线数据失败: %s", e)
            # 缓存中有旧数据则返回（即使过期，也比完全没数据好）
            if cache_key in self._data_cache:
                _, cached_df = self._data_cache[cache_key]
                log.info("[evolve] 拉取失败，返回缓存数据")
                return cached_df

        return None
