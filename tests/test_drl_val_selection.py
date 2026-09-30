"""TDD：train/val 内部分段选模真正生效（诊断报告 §15 建议1）。

train_drl 已有 train/val/OOS 三区切分与"best_agent 按 val_ret 选优"框架，但：
1) val_eval_interval 默认 10 > 持续进化 episodes=8 → 训练期间 val 只评估最后一轮，
   "周期性按 val 选最优"失效（best_agent 只在最后一轮更新一次）；
2) 引擎 new_fitness 用 result["best_ret"]=max(训练段,验证段) → 训练段收益与 OOS
   负相关（§15：Pearson -0.486、训练正但 OOS≤0 占100%），锚点/面板 fitness 被
   训练段污染。

修复：val_eval_interval 钳到 episodes//3（保证训练期间≥3次 val 评估）；
      引擎 fitness 优先用 result["best_val_ret"]（验证段最优收益）。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


def test_train_drl_val_evaluated_multiple_times_when_episodes_small():
    """episodes=6 时 val 评估应≥2 次（val_eval_interval 钳到 2），并返回 best_val_ret。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="5m", n=600, seed=7)
    res = train_drl(df, {
        "episodes": 6, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
        "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
    })
    n_val = sum(1 for h in res["history"] if h.get("val_ret") is not None)
    assert n_val >= 2, f"episodes=6 时 val 评估应≥2 次（interval 钳到2），实际 {n_val}"
    assert res.get("best_val_ret") is not None
    # best_ret 仍兼容（max 语义），但引擎必须能用 best_val_ret
    assert "best_val_ret" in res


def test_train_drl_val_segments_median_enabled():
    """val_segments=3：val 收益取多段中位数（避免单段 regime 选出保守模型）。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="5m", n=900, seed=7)
    res = train_drl(df, {
        "episodes": 6, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
        "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
        "val_segments": 3,
    })
    assert res.get("val_segments") == 3
    # 至少一次多段 val 评估（历史里 val_ret 非 None 且标记了 segments）
    multi = [h for h in res["history"] if h.get("val_segments") == 3]
    assert multi, "val_segments=3 时历史应记录多段评估"
    assert all(h.get("val_ret") is not None for h in multi)


def test_train_drl_val_segments_one_compat():
    """val_segments=1（默认）保持原单段行为，兼容旧调用。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="5m", n=600, seed=7)
    res = train_drl(df, {
        "episodes": 6, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
        "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
        "val_segments": 1,
    })
    assert res.get("val_segments") == 1
    n_val = sum(1 for h in res["history"] if h.get("val_ret") is not None)
    assert n_val >= 2


def _make_engine(tmp_path, monkeypatch, result_extras):
    from config.settings import settings
    from backtest.data_loader import generate_demo
    import drl.evolve_engine as ev
    monkeypatch.setattr(settings, "evolve_strategy_drl_population", False,
                        raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    monkeypatch.setattr(eng, "_data_incremented", lambda *a, **k: True)
    fake_agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)

    result = {
        "agent": fake_agent,
        "history": [{"best_ret": 0.2}],
        "best_ret": 0.2,
        "oos_report": {"enabled": False},
    }
    result.update(result_extras)

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        return result
    monkeypatch.setattr(ev, "train_drl", _fake_train)
    return eng


@pytest.mark.asyncio
async def test_engine_fitness_prefers_best_val_ret(tmp_path, monkeypatch):
    """引擎本轮 fitness 优先用验证段收益（best_val_ret），而非训练段 max。"""
    eng = _make_engine(tmp_path, monkeypatch,
                       {"best_val_ret": 0.05, "best_ret": 0.2})
    await eng._train_strategy_drl_once(force=False)
    assert eng._strategy_drl_status.get("fitness") == 0.05


@pytest.mark.asyncio
async def test_engine_fitness_falls_back_to_best_ret(tmp_path, monkeypatch):
    """无 best_val_ret（旧模型/手动训练）时回退 best_ret，保持兼容。"""
    eng = _make_engine(tmp_path, monkeypatch, {})
    await eng._train_strategy_drl_once(force=False)
    assert eng._strategy_drl_status.get("fitness") == 0.2
