"""DRL 根本性逻辑错误的回归锁（2026-10）。

对应 docs/DRL训练根本性逻辑错误-诊断报告.md 的 F1/F2/F4：
  F1 奖励口径：旧 absolute 口径下"永不交易"的累计奖励恰为 0，而任何持仓都是
     无理扣分的负值 ⇒ 全局最优 = 躺平（产线 452/501 轮被 OOS 门拒、fitness 多为 0）。
     excess 口径把基准（默认买入持有）作为零点：上涨行情里空仓要付机会成本。
  F2 决策间隔：每根K线重新决策 + 0.15% 单边成本 ⇒ 随机试错每回合亏 46~82%，
     优势里"别交易"的信号压过真实行情信号。decision_interval=k 时成本降 k 倍。
  F4 epsilon：set_epsilon_schedule 只写 self.epsilon，动作选择从不读它 ——
     上报了一个与行为无关的"探索率"。修复后要么真正生效（ε 混合 + 同口径 logp），
     要么如实上报 0.0。
"""
import numpy as np
import pandas as pd
import pytest

from drl.env import (ACTION_BUCKETS, TradingEnv, policy_env_kwargs,
                     policy_env_kwargs_from_settings, run_episode)


def _rising_market(n: int = 900, drift: float = 0.0005, vol: float = 0.01) -> pd.DataFrame:
    """确定性上涨 + 零均值噪声的行情（模拟真实市场：漂移 ≪ 单根波动）。

    噪声用严格平衡的 ±vol 交替序列（和恰为 0）⇒ 整段涨幅 == drift·n（完全确定），
    而单根波动仍是 vol、5 根波动率 ≈ vol·100(%)。这样：
      - 行情必然上涨（不靠种子运气）→ excess 口径下空仓必然付机会成本；
      - 每根漂移 5bp 远小于旧口径的无条件风险惩罚（40·(仓位·vol%)² ≈ 10~40bp/根）
        → absolute 口径下"躺平"必然最优（复现被修复的结构性缺陷）。
    """
    n = int(n)
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    rets = drift + vol * alt
    close = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": np.full(n, 100.0)}, index=idx)


def _flat_market(n: int = 600) -> pd.DataFrame:
    """价格恒定的行情：用于隔离"调仓摩擦成本"（持仓本身不赚不亏）。"""
    close = np.full(n, 100.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": np.full(n, 100.0)}, index=idx)


def _cum_reward(df: pd.DataFrame, action: int, **kw) -> float:
    env = TradingEnv(df, **kw)
    traj = run_episode(env, lambda s, _a=action: _a)
    return float(np.sum(traj["rewards"]))


# ============ F1：奖励口径 ============

def test_legacy_absolute_mode_flat_reward_is_exactly_zero():
    """旧口径（absolute）的结构性缺陷：空仓累计奖励恒为 0，与行情无关。

    这正是"躺平"成为全局最优的机制——不管行情涨多少，什么都不做的奖励都是 0，
    而任何持仓都要付无条件风险惩罚。
    """
    df = _rising_market()
    assert _cum_reward(df, 0, vol_penalty=40.0, reward_mode="absolute") == 0.0


def test_excess_mode_flat_is_penalised_on_rising_market():
    """新口径（excess）：上涨行情里空仓必须被扣分（机会成本），不再免费。"""
    df = _rising_market()
    flat = _cum_reward(df, 0, reward_mode="excess", risk_coef=0.5)
    assert flat < 0.0, f"excess 口径下空仓仍应为负（机会成本），实际 {flat}"


def test_excess_mode_flat_is_not_reward_optimal():
    """核心不变式：修好后"永不交易"不能再是奖励最优的恒定仓位策略。"""
    df = _rising_market()
    rewards = {i: _cum_reward(df, i, reward_mode="excess", risk_coef=0.5)
               for i in range(len(ACTION_BUCKETS))}
    best_action = max(rewards, key=lambda k: rewards[k])
    assert best_action != 0, f"空仓仍是奖励最优（{rewards}）"


def test_legacy_absolute_mode_flat_is_optimal_regression():
    """反向锁：旧口径下空仓确实是最优（保持 absolute 可复现旧行为，供对照实验）。"""
    df = _rising_market()
    rewards = {i: _cum_reward(df, i, vol_penalty=40.0, reward_mode="absolute")
               for i in range(len(ACTION_BUCKETS))}
    best_action = max(rewards, key=lambda k: rewards[k])
    assert best_action == 0, f"旧口径下空仓应为最优（{rewards}）"


def test_risk_coef_scales_excess_penalty():
    """excess 的风险项按 risk_coef 标定（bp 量级），不再是无条件重税。"""
    df = _rising_market()
    low = _cum_reward(df, 4, reward_mode="excess", risk_coef=0.0)
    high = _cum_reward(df, 4, reward_mode="excess", risk_coef=5.0)
    assert high < low, "risk_coef 增大应让满仓的累计奖励下降"


# ============ F2：决策间隔 ============

def test_decision_interval_cuts_turnover_cost():
    """k=4 时同一动作序列的摩擦成本显著低于 k=1（探索成本降 k 倍）。

    用恒定价格行情隔离成本：持仓本身不赚不亏，账户变化全部来自调仓摩擦。
    """
    df = _flat_market(n=600)
    rng = np.random.default_rng(3)
    acts = rng.integers(0, 5, size=4 * len(df)).tolist()

    def _run(k: int) -> float:
        env = TradingEnv(df, fee_rate=0.001, slippage=0.0005, min_trade_zone=0.10,
                         decision_interval=k, reward_mode="absolute")
        box = {"i": 0}

        def _policy(_state):
            i = box["i"]
            box["i"] = i + 1
            return int(acts[i % len(acts)])

        traj = run_episode(env, _policy)
        return float(traj["final_equity"])

    eq_k1, eq_k4 = _run(1), _run(4)
    assert eq_k1 < 10000.0, f"每根乱调仓应亏在摩擦上，实际 {eq_k1}"
    assert eq_k4 > eq_k1, f"k=4 应显著少亏：k1={eq_k1} k4={eq_k4}"


def test_decision_interval_keeps_account_return_consistent():
    """恒定仓位下 k=1 与 k=4 的账户路径一致（决策间隔不改变持仓收益本身）。"""
    df = _rising_market(n=600)
    eqs = []
    for k in (1, 4):
        env = TradingEnv(df, fee_rate=0.001, slippage=0.0005, min_trade_zone=0.10,
                         decision_interval=k, reward_mode="absolute")
        traj = run_episode(env, lambda s: 4)
        eqs.append(float(traj["final_equity"]))
    assert eqs[0] == pytest.approx(eqs[1], rel=1e-9)


def test_decision_interval_ret_pct_is_period_return():
    """info.ret_pct 必须是整个决策期的收益：连乘 == 期末/期初权益。

    曾经只上报决策期最后一根K线的收益，导致 total_ret 严重低报
    （实测 k=4 时 2.79% vs 真实 8.60%），污染选模与 OOS 判据。
    """
    df = _rising_market(n=600)
    env = TradingEnv(df, decision_interval=4, fee_rate=0.001, min_trade_zone=0.10)
    traj = run_episode(env, lambda s: 4)
    compounded = float(np.prod([1.0 + i["ret_pct"] / 100.0 for i in traj["infos"]]) - 1.0)
    assert compounded == pytest.approx(traj["total_ret"], rel=1e-9)
    assert traj["final_equity"] == pytest.approx(10000.0 * (1.0 + traj["total_ret"]),
                                                 rel=1e-9)


def test_decision_interval_clamped_to_at_least_one():
    env = TradingEnv(_rising_market(n=200), decision_interval=0)
    assert env.decision_interval == 1


# ============ 配置打通（口径同源） ============

def test_policy_env_kwargs_defaults_are_legacy():
    """缺键时必须是旧行为（absolute / k=1 / 不做空），保证向后兼容。"""
    assert policy_env_kwargs({}) == {"reward_mode": "absolute",
                                     "benchmark_position": 1.0,
                                     "risk_coef": 0.5,
                                     "decision_interval": 1,
                                     "short_enabled": False}
    assert policy_env_kwargs(None)["reward_mode"] == "absolute"
    assert policy_env_kwargs(None)["short_enabled"] is False


def test_policy_env_kwargs_reads_settings():
    kw = policy_env_kwargs_from_settings()
    assert kw["reward_mode"] in ("absolute", "excess")
    assert kw["decision_interval"] >= 1


def test_evolve_train_cfg_carries_reward_and_decision():
    """持续进化的训练配置必须带上奖励口径/决策间隔（否则训练与观测口径漂移）。"""
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    cfg = _strategy_drl_train_cfg(8)
    for key in ("reward_mode", "benchmark_position", "risk_coef",
                "decision_interval", "episode_window"):
        assert key in cfg, f"训练配置缺少 {key}"
    assert cfg["decision_interval"] >= 1
    assert cfg["decision_interval"] == max(
        1, int(getattr(settings, "evolve_strategy_drl_decision_interval", 1) or 1))
    # 训练语义进 _training_identity：旧锚点（躺平即最优时代训出来的）不可比
    from drl.evolve_engine import _training_identity
    ident = _training_identity("strategy_drl", "ETH/USDT", "5m", state_window=4,
                               factor_signature="", config={
                                   "reward_mode": cfg["reward_mode"],
                                   "decision_interval": cfg["decision_interval"],
                                   "risk_coef": cfg["risk_coef"]})
    other = _training_identity("strategy_drl", "ETH/USDT", "5m", state_window=4,
                               factor_signature="", config={
                                   "reward_mode": "absolute",
                                   "decision_interval": 1,
                                   "risk_coef": 0.5})
    assert ident != other, "奖励口径/决策间隔变化必须改变训练身份指纹"


# ============ F4：epsilon 不再是死代码 ============

def test_epsilon_zero_means_pure_policy_sampling():
    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=5, hidden=(8, 8), seed=1)
    agent.epsilon = 0.0
    state = np.asarray([0.1, 0.2, -0.3, 0.4])
    proba = agent.actor.predict_proba(state.reshape(1, -1))[0]
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        a, logp = agent.sample_action_with_logp(state, rng=rng, use_epsilon=True)
        seen.add(a)
        assert logp == pytest.approx(float(np.log(proba[a])), abs=1e-9)
    assert len(seen) > 1


def test_epsilon_one_gives_uniform_distribution_and_consistent_logp():
    """ε=1 时行为策略是合法动作上的均匀分布，且 logp 与之一致（PPO 比值不失真）。"""
    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=5, hidden=(8, 8), seed=1)
    agent.epsilon = 1.0
    state = np.asarray([0.1, 0.2, -0.3, 0.4])
    rng = np.random.default_rng(0)
    counts = np.zeros(5)
    for _ in range(2000):
        a, logp = agent.sample_action_with_logp(state, rng=rng, use_epsilon=True)
        counts[a] += 1
        assert logp == pytest.approx(float(np.log(1.0 / 5.0)), abs=1e-6)
    assert counts.min() > 300, f"ε=1 应近似均匀，实际 {counts}"


def test_epsilon_mixing_respects_action_mask():
    """掩码 + ε 混合：非法动作概率必须仍为 0（不会被均匀混合加回来）。"""
    from drl.agent import ACAgent
    agent = ACAgent(state_dim=4, n_actions=5, hidden=(8, 8), seed=1)
    agent.epsilon = 0.9
    state = np.asarray([0.1, 0.2, -0.3, 0.4])
    rng = np.random.default_rng(0)
    mask = np.asarray([1, 3])
    for _ in range(300):
        a, logp = agent.sample_action_with_logp(state, rng=rng, mask=mask,
                                                use_epsilon=True)
        assert a in (1, 3), f"采样到非法动作 {a}"
        assert np.isfinite(logp)


def test_train_drl_reports_truthful_epsilon():
    """默认不启用 ε-greedy 时，历史里上报的 epsilon 必须是 0.0（不是假的衰减值）。"""
    from drl.agent import train_drl
    df = _rising_market(n=420)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "oos_min_bars": 80})
    assert res["history"], "应有训练历史"
    assert all(row["epsilon"] == 0.0 for row in res["history"]), \
        [row["epsilon"] for row in res["history"]]
