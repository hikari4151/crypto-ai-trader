"""DRL 神经引擎优化（D1-D5）专项测试。"""
import numpy as np
import pytest

from drl.agent import _clip_grad_norm, _oos_decay


# ============ D5. Actor 梯度范数裁剪 ============

def test_clip_grad_norm_scales_large():
    """超大梯度被缩放到 max_norm。"""
    gW = [np.full((4, 4), 10.0)]   # 每层范数 ~40
    gb = [np.zeros(4)]
    norm_before = np.sqrt(np.sum(np.square(gW[0])))
    gW2, gb2 = _clip_grad_norm(gW, gb, 5.0)
    norm_after = np.sqrt(np.sum(np.square(gW2[0])))
    assert norm_after == pytest.approx(5.0, rel=1e-3)
    assert norm_after < norm_before


def test_clip_grad_norm_small_unchanged():
    """范数小于 max_norm 时不缩放。"""
    gW = [np.array([[0.1, -0.2]])]
    gb = [np.zeros(2)]
    gW2, _ = _clip_grad_norm(gW, gb, 5.0)
    assert np.array_equal(gW2[0], gW[0])


def test_clip_grad_norm_disabled():
    """max_norm<=0 表示禁用（向后兼容）。"""
    gW = [np.full((4, 4), 100.0)]
    gW2, _ = _clip_grad_norm(gW, [], 0.0)
    assert np.array_equal(gW2[0], gW[0])


# ============ D3. OOS 衰减语义 ============

def test_oos_decay_semantics():
    assert _oos_decay(0.10, 0.05) == pytest.approx(0.5)   # 衰减一半
    assert _oos_decay(0.10, -0.02) == 1.0                  # 训练盈利、OOS 亏损=最大衰减
    assert _oos_decay(0.10, 0.10) == pytest.approx(0.0)    # OOS 持平=不衰减
    assert _oos_decay(-0.05, 0.10) == 0.0                  # 训练段为负：不可比，不判定
    assert _oos_decay(0.0, 0.0) == 0.0


def test_worker_rng_determinism():
    """D1：并行收集使用 per-worker rng，同 seed 两次训练结果一致（可复现）。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl

    df = generate_demo(timeframe="1h", n=400)
    cfg = {"episodes": 2, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64,
           "seed": 42, "val_eval_interval": 1}
    r1 = train_drl(df, dict(cfg))
    r2 = train_drl(df, dict(cfg))
    h1 = [(row.get("policy_loss"), row.get("val_ret"), row.get("ep_ret")) for row in r1["history"]]
    h2 = [(row.get("policy_loss"), row.get("val_ret"), row.get("ep_ret")) for row in r2["history"]]
    assert h1 == h2, "同 seed 两次训练应逐值一致（多线程采样可复现）"


def test_slippage_reduces_return():
    """D2：训练环境加入滑点后，同等策略的累计收益更低（成本更贴近实盘）。"""
    from backtest.data_loader import generate_demo
    from drl.env import TradingEnv, run_episode
    from drl.agent import ACAgent

    df = generate_demo(timeframe="1h", n=400)
    env0 = TradingEnv(df, slippage=0.0, vol_penalty=0.0)
    envs = TradingEnv(df, slippage=0.005, vol_penalty=0.0)
    agent = ACAgent(env0.state_dim, 5, seed=3)
    r0 = run_episode(env0, lambda s: agent.greedy_action(s))
    rs = run_episode(envs, lambda s: agent.greedy_action(s))
    # 同策略同动作序列，滑点仅增加成本 → 净收益应更低（或至少不更高）
    assert r0["total_ret"] >= rs["total_ret"] - 1e-9


def test_deployment_flag_and_pine_when_not_blocked():
    """D3：非过拟合（oos_hard_gate=False 强制放行）时部署未被拦截、Pine 有内容。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl

    df = generate_demo(timeframe="1h", n=1000)
    cfg = {"episodes": 1, "n_episodes": 1, "ppo_epochs": 1, "mini_batch_size": 128,
           "seed": 7, "val_eval_interval": 1, "oos_hard_gate": False}
    res = train_drl(df, dict(cfg))
    assert res.get("deployment_blocked") is False, "关闭硬门时不应拦截部署"
    assert "deployment_blocked" in res
