"""做空支持（动作空间 −1~+1）的回归锁（2026-10）。

背景：动作空间只能做多/空仓时，熊市样本外"最优多头策略"=全程空仓，
相对基准的超额全靠"没参与"（docs/DRL成本标定-决策频率与滑点复核.md §4）。
做空是那类窗口里唯一能真正创造正收益的路径。

本轮只做**训练环境**的做空（默认关闭、向后逐位兼容）；
执行内核（backtest/_matching.py，现货语义 `sell` 需 position>0）尚不支持负仓位，
因此含做空档的模型必须被**显式阻断部署**（见 evolve_engine._strategy_deploy_gate）。
"""
import numpy as np
import pandas as pd
import pytest

from drl.env import (LONG_ONLY_BUCKETS, SHORT_BUCKETS, TradingEnv,
                     action_buckets_for, action_names_for, buckets_require_short,
                     run_episode)


def _mk_df(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": np.full(len(close), 100.0)}, index=idx)


def _ramp(n: int, step: float) -> pd.DataFrame:
    return _mk_df(100.0 * np.cumprod(np.full(n, 1.0 + step)))


# ============ 动作空间 ============

def test_action_space_selection():
    assert action_buckets_for(False) is LONG_ONLY_BUCKETS
    assert action_buckets_for(True) is SHORT_BUCKETS
    assert len(SHORT_BUCKETS) == 9
    assert SHORT_BUCKETS[0] == -1.0 and SHORT_BUCKETS[-1] == 1.0
    assert 0.0 in SHORT_BUCKETS
    assert action_names_for(True)[0] == "满空"
    assert action_names_for(False) == ["清仓", "轻仓", "半仓", "重仓", "满仓"]


def test_buckets_require_short_detection():
    assert buckets_require_short(SHORT_BUCKETS) is True
    assert buckets_require_short(LONG_ONLY_BUCKETS) is False
    assert buckets_require_short(None) is False
    assert buckets_require_short([1.0, 2.0]) is False


def test_env_exposes_action_space():
    env_long = TradingEnv(_mk_df(np.full(300, 100.0)), short_enabled=False)
    env_short = TradingEnv(_mk_df(np.full(300, 100.0)), short_enabled=True)
    assert env_long.n_actions == 5 and not env_long.short_enabled
    assert env_short.n_actions == 9 and env_short.short_enabled
    assert env_short.action_names[0] == "满空"


# ============ 做空机制 ============

def test_short_profits_when_price_falls():
    """做空仓：价格下跌应盈利（多头模式下同一个动作是空仓/不赚）。"""
    df = _ramp(300, -0.002)                      # 单调下跌
    env = TradingEnv(df, short_enabled=True, fee_rate=0.0, slippage=0.0,
                     min_trade_zone=0.0)
    traj = run_episode(env, lambda s: 0)         # 0 = 满空（-1.0）
    assert traj["total_ret"] > 0.2, f"下跌行情里满空应盈利，实际 {traj['total_ret']}"


def test_short_loses_when_price_rises():
    df = _ramp(300, 0.002)
    env = TradingEnv(df, short_enabled=True, fee_rate=0.0, slippage=0.0,
                     min_trade_zone=0.0)
    traj = run_episode(env, lambda s: 0)
    assert traj["total_ret"] < -0.2, f"上涨行情里满空应亏损，实际 {traj['total_ret']}"


def test_short_mirrors_long_pnl_on_reversed_series():
    """对称性：同一段涨跌序列取反，做空收益≈做多收益的镜像（成本为 0 时）。"""
    up = _ramp(200, 0.003)["close"].to_numpy(float)
    down = _mk_df(100.0 * np.cumprod(np.full(200, 1.0 - 0.003)))
    env_long = TradingEnv(_mk_df(up), short_enabled=False, fee_rate=0.0,
                          slippage=0.0, min_trade_zone=0.0)
    env_short = TradingEnv(down, short_enabled=True, fee_rate=0.0,
                           slippage=0.0, min_trade_zone=0.0)
    long_ret = run_episode(env_long, lambda s: 4)["total_ret"]     # 满仓
    short_ret = run_episode(env_short, lambda s: 0)["total_ret"]   # 满空
    assert short_ret == pytest.approx(long_ret, rel=0.02)


def test_bankruptcy_guard_stops_short_on_gap_rally():
    """空头被跳空逼空：权益击穿 0 → 立即终止并标记 bankrupt。

    为什么必须用"跳空 + 粗决策间隔"：本环境是**恒定比例**仓位
    （目标=权益比例，价格漂移会在下一次决策被拉回），连续小涨时空头会被
    自动减仓（de-risk），权益只会渐近趋 0；只有**两次决策之间**的跳空
    超过保证金承受范围才会击穿 0 —— 而生产配置恰好是 k=8（8 小时持有），
    这类跳空真实存在，所以守卫是必需的。
    """
    close = 100.0 * np.cumprod(np.full(200, 1.25))     # 每根 +25%
    env = TradingEnv(_mk_df(close), short_enabled=True, fee_rate=0.0,
                     slippage=0.0, min_trade_zone=0.0, decision_interval=8)
    traj = run_episode(env, lambda s: 0, max_steps=3)
    assert traj["bankrupt"] is True
    assert traj["steps"] <= 8, f"应在决策期内跳空处终止，实际走了 {traj['steps']} 步"
    assert traj["final_equity"] <= 0.0


def test_short_with_frequent_rebalance_does_not_bankrupt():
    """连续小涨 + 逐根再平衡：恒定比例仓位会自动减仓（de-risk），不应爆仓。

    这是本环境做空的**安全性来源**，也是与"固定数量做空"的关键区别——
    用测试把这条性质固定下来，避免以后误以为"做空必然瞬间爆仓"。
    """
    close = 100.0 * np.cumprod(np.full(120, 1.02))
    env = TradingEnv(_mk_df(close), short_enabled=True, fee_rate=0.0,
                     slippage=0.0, min_trade_zone=0.0, decision_interval=1)
    traj = run_episode(env, lambda s: 0)
    assert traj["bankrupt"] is False
    assert 0.0 < traj["final_equity"] < 10000.0


def test_long_only_mode_never_bankrupts_or_shorts():
    """默认（长仓）模式：不产生负仓位、不触发爆仓标记、权益不为负。"""
    close = 100.0 * np.cumprod(np.full(200, 0.5))      # 每根 -50%
    env = TradingEnv(_mk_df(np.maximum(close, 1e-6)), short_enabled=False,
                     fee_rate=0.0, slippage=0.0, min_trade_zone=0.0)
    traj = run_episode(env, lambda s: 4)
    pos = np.asarray([i["position_ratio"] for i in traj["infos"]], dtype=float)
    assert pos.min() >= -1e-9, "长仓模式不应出现负仓位"
    assert traj["bankrupt"] is False


def test_short_mode_funding_is_signed():
    """资金费率：正费率下空头**收钱**（多头付钱）——这是永续合约的定义。"""
    df = _mk_df(np.full(400, 100.0))                   # 恒价：只有 funding 影响权益
    env = TradingEnv(df, short_enabled=True, fee_rate=0.0, slippage=0.0,
                     min_trade_zone=0.0, funding_rate=0.001)
    traj = run_episode(env, lambda s: 0)               # 满空
    assert traj["final_equity"] > 10000.0, "正费率下空头应收资金费"
    env_long = TradingEnv(df, fee_rate=0.0, slippage=0.0, min_trade_zone=0.0,
                          funding_rate=0.001)
    t_long = run_episode(env_long, lambda s: 4)        # 满仓
    assert t_long["final_equity"] < 10000.0, "正费率下多头应付资金费"


def test_short_mode_keeps_turnover_accounting():
    """做空模式同样精确记账（成本基数/现金成本），做空开仓=1 单位换手。"""
    df = _mk_df(np.full(300, 100.0))
    env = TradingEnv(df, short_enabled=True, fee_rate=0.001, slippage=0.0005,
                     min_trade_zone=0.05)
    traj = run_episode(env, lambda s: 0)
    assert traj["turnover"] == pytest.approx(1.0, abs=0.02)
    assert traj["cost_paid"] == pytest.approx(10000.0 * 0.0015, rel=0.05)


def test_short_reward_uses_signed_position():
    """excess 奖励的风险项用 (仓位−基准)：满空在上涨行情里应被重罚。"""
    df = _ramp(200, 0.005)
    env = TradingEnv(df, short_enabled=True, reward_mode="excess", risk_coef=0.5,
                     fee_rate=0.001, min_trade_zone=0.0)
    traj = run_episode(env, lambda s: 0)               # 满空 vs 基准满仓
    assert float(np.sum(traj["rewards"])) < 0


# ============ 模型文件携带动作空间 ============

def test_agent_roundtrip_preserves_short_action_space():
    from drl.agent import ACAgent
    agent = ACAgent(6, len(SHORT_BUCKETS), hidden=(8, 8), seed=1,
                    action_space=SHORT_BUCKETS)
    restored = ACAgent.from_dict(agent.to_dict())
    assert restored.n_actions == 9
    assert [float(x) for x in restored.action_buckets] == [float(x) for x in SHORT_BUCKETS]


def test_legacy_model_without_buckets_defaults_to_long_only():
    from drl.agent import ACAgent
    agent = ACAgent(6, 5, hidden=(8, 8), seed=1)
    data = agent.to_dict()
    data.pop("action_buckets")                          # 模拟旧模型文件
    restored = ACAgent.from_dict(data)
    assert [float(x) for x in restored.action_buckets] == [float(x) for x in LONG_ONLY_BUCKETS]


def test_train_drl_short_cfg_reports_short_action_space():
    from drl.agent import train_drl
    df = _ramp(420, 0.0005)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "short_enabled": True, "oos_min_bars": 80})
    assert res["n_actions"] == 9
    assert res["requires_short"] is True
    assert res["short_enabled"] is True
    assert min(res["action_buckets"]) < 0


def test_train_drl_default_is_long_only():
    from drl.agent import train_drl
    df = _ramp(420, 0.0005)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "oos_min_bars": 80})
    assert res["n_actions"] == 5
    assert res["requires_short"] is False
    assert min(res["action_buckets"]) >= 0


# ============ 部署阻断（训练做空/执行不能做空 = 口径漂移） ============

def test_deploy_gate_blocks_short_model():
    from drl.evolve_engine import _strategy_deploy_gate
    oos = {"enabled": True, "oos_ret": 0.30, "oos_bh_ret": -0.10,
           "oos_turnover": 10.0}
    res = {"oos_report": oos, "deployment_blocked": False,
           "requires_short": True, "short_enabled": True}
    ok, reason = _strategy_deploy_gate(res)
    assert not ok
    assert "做空" in reason and "执行内核" in reason


def test_deploy_gate_allows_long_only_model():
    from drl.evolve_engine import _strategy_deploy_gate
    oos = {"enabled": True, "oos_ret": 0.05, "oos_bh_ret": -0.10,
           "oos_turnover": 10.0}
    ok, _ = _strategy_deploy_gate({"oos_report": oos, "deployment_blocked": False,
                                   "requires_short": False, "short_enabled": False})
    assert ok


def test_pine_export_refuses_short_model():
    from drl.agent import ACAgent
    from drl.pine_export import try_build_drl_pine
    agent = ACAgent(15, len(SHORT_BUCKETS), hidden=(16, 16), seed=1,
                    action_space=SHORT_BUCKETS)
    code, note = try_build_drl_pine(agent)
    assert code == ""
    assert "做空" in note
