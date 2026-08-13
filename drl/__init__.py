"""深度强化学习：动态自适应策略生成。

将交易建模为马尔可夫决策过程（MDP），用 Actor-Critic 策略梯度在回测环境训练智能体：
- 状态：价量 / 波动率 / 趋势与风险 regime / 持仓与账户
- 动作：目标仓位档位（买入/卖出/持仓比例）
- 奖励：组合收益 - 交易成本 - 波动率风险惩罚
"""
from .agent import ACAgent, evaluate_agent, train_drl
from .env import ACTION_BUCKETS, ACTION_NAMES, TradingEnv, gpu_available, run_episode
from .nnet import MLP

__all__ = [
    "ACAgent", "MLP", "TradingEnv", "run_episode", "train_drl", "evaluate_agent",
    "ACTION_BUCKETS", "ACTION_NAMES", "gpu_available",
]
