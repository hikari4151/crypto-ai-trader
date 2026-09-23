"""TDD：挑战者谱系独立续训基线（打破"每轮从零训练→恒0→永无基线"死循环）。

背景（2026-09-19 根因取证）：evolve_strategy_drl_population=True 时轮次冠军/挑战者
交替；挑战者谱系 strategy_drl_alt 因 OOS 门失败（P4-C2 只存档不更新 best）永远没有
可续训模型 → 每轮从零训 8 episodes → 学不到东西 → fitness=0.0 → 曲线"0/有数据"交替
的深层来源。

修复：ModelZoo 新增独立「训练基线」槽位（base.json.gz，不部署、不占版本名额）：
- 挑战者轮每次训练后（OOS 门之前，demo 轮除外）保存候选为 base；
- 下一轮挑战者加载顺序：alt best → base 兜底（base 保证续训不断链）。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from drl.model_zoo import ModelZoo
from tests.test_drl_optimizations import _MetaDB


def _agent():
    return ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)


def test_save_load_base_roundtrip(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    zoo.save_agent_base(_agent(), "strategy_drl_alt")
    loaded = zoo.load_agent_base("strategy_drl_alt")
    assert loaded is not None
    assert loaded.state_dim == 4
    assert loaded.n_actions == 3


def test_load_base_missing_returns_none(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    assert zoo.load_agent_base("strategy_drl_alt") is None


@pytest.mark.asyncio
async def test_challenger_round_saves_training_base_even_if_oos_rejected(tmp_path, monkeypatch):
    """挑战者轮即使 OOS 门失败也要保存训练基线（否则下一轮又从头训）。"""
    from backtest.data_loader import generate_demo
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    calls = []
    monkeypatch.setattr(eng.zoo, "save_agent_base",
                        lambda agent, name: calls.append((agent, name)))
    fake_agent = object()
    fake_result = {
        "agent": fake_agent,
        "history": [{"best_ret": 0.0}],
        "best_ret": 0.0,
        "oos_report": {"enabled": False, "oos_ret": 0.0},
        "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
        "factor_composite": None, "pine_code": "", "pine_note": "",
    }

    async def _fake_run_train(_train_call):
        return fake_result
    monkeypatch.setattr(eng, "_run_train", _fake_run_train)

    eng._strategy_drl_status["_pop_round"] = 1  # 挑战者轮
    await eng._train_strategy_drl_once(force=False)

    assert any(name == "strategy_drl_alt" for _, name in calls), \
        "挑战者轮 OOS 失败也应保存训练基线"


@pytest.mark.asyncio
async def test_challenger_round_loads_base_as_fallback(tmp_path, monkeypatch):
    """alt best 为空时，挑战者轮应从训练基线续训（不退回从零）。"""
    from backtest.data_loader import generate_demo
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)

    base_agent = _agent()
    loaded = []
    monkeypatch.setattr(eng.zoo, "load_agent_base",
                        lambda name: (loaded.append(name) or base_agent))
    # load_agent(alt) 返回 None → 触发 base 兜底
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: None)

    fake_result = {
        "agent": base_agent,
        "history": [{"best_ret": 0.0}],
        "best_ret": 0.0,
        "oos_report": {"enabled": False, "oos_ret": 0.0},
        "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
        "factor_composite": None, "pine_code": "", "pine_note": "",
    }

    async def _fake_run_train(_train_call):
        return fake_result
    monkeypatch.setattr(eng, "_run_train", _fake_run_train)

    eng._strategy_drl_status["_pop_round"] = 1  # 挑战者轮
    await eng._train_strategy_drl_once(force=False)
    assert "strategy_drl_alt" in loaded, "挑战者轮应从训练基线续训"