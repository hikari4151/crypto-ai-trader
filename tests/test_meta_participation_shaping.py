"""TDD：元策略控制器「空仓=0」懒解防呆——持仓参与塑形（reward_participation_scale）。

观察项（2026-09-19 诊断）：meta_controller 501 轮里 489 轮被拒、OOS≈-0.21、
fitness=0.0（空仓懒解）；reward_signal_align 塑形后仅出现过一次接受（v286）。
参照 TensorTrade 奖励组件化与 MetaTrader(AAAI) 泛化研究的做法，新增
「持仓参与塑形」：奖励 += scale × 当前仓位比，直接惩罚"空仓=0"的懒惰最优解。

口径保证：塑形只进训练梯度——选模/OOS 的 total_ret 按权益复利计算
（drl.env.run_episode），不受本塑形影响，验证/OOS 门禁语义不变。
"""

from backtest.data_loader import generate_demo
from strategies.meta import MetaControllerEnv


def _episode_total_reward(scale: float, action_fn) -> float:
    df = generate_demo(n=200, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    env = MetaControllerEnv(df, ["dual_ma"], reward_participation_scale=scale)
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        obs, reward, done, info = env.step(action_fn())
        total += float(reward)
    return total


def test_participation_shaping_rewards_holding_position():
    """开启持仓塑形后，同样「恒买入并持有」轨迹的累计奖励必须更高。"""
    hold = lambda: 1  # 已持仓时 action=1 为空操作 → 保持多头到结束
    with_scale = _episode_total_reward(1.0, hold)
    without_scale = _episode_total_reward(0.0, hold)
    assert with_scale > without_scale


def test_participation_shaping_flat_gets_no_bonus():
    """恒空仓（action=0）不产生持仓奖励：开/关塑形累计奖励应一致。"""
    flat = lambda: 0
    assert _episode_total_reward(1.0, flat) == _episode_total_reward(0.0, flat)


def test_env_exposes_participation_scale():
    """构造参数可读回（供训练 cfg / 面板配置校验使用）。"""
    df = generate_demo(n=200, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    env = MetaControllerEnv(df, ["dual_ma"], reward_participation_scale=1.5)
    assert env.reward_participation_scale == 1.5


def test_settings_exposes_meta_participation_scale():
    """持续进化引擎配置必须暴露 meta_participation_scale（供面板/热更新）。"""
    from config.settings import settings
    assert hasattr(settings, "evolve_meta_participation_scale")
    assert float(settings.evolve_meta_participation_scale) >= 0.0


def test_meta_env_kwargs_forwards_participation_scale():
    """train_meta_controller 的 cfg 必须把 reward_participation_scale 透传给环境。"""
    from strategies.meta import _meta_env_kwargs
    kw = _meta_env_kwargs({"reward_participation_scale": 2.5}, seed=42)
    assert kw["reward_participation_scale"] == 2.5
    # 未配置时默认关闭（0=保持旧行为）
    assert _meta_env_kwargs({}, seed=42)["reward_participation_scale"] == 0.0
