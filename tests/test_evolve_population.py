"""策略 DRL 冠军/挑战者 K=2 族群训练测试（P0-族群）。

1. 轮次交替：偶数轮训冠军谱系（strategy_drl），奇数轮训挑战者谱系
   （strategy_drl_alt），每轮只训一个模型（墙钟成本不变）；
2. 挑战者未显著优于冠军 → 只更新挑战者槽位，冠军部署/版本原样不动；
3. 挑战者显著优于冠军（> 冠军×(1/threshold)）→ 晋升接管冠军槽位并注册；
4. force 手动触发恒走冠军谱系，不受交替节奏影响。
"""
import json

import pytest

import drl.evolve_engine as ev
from backtest.data_loader import generate_demo
from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


def _agent():
    # 真实 ACAgent（极小网络）：角色互换路径要把旧冠军 save→load 往返，
    # _StubAgent 缺 state_dim 等字段无法反序列化，必须用真 agent
    return ACAgent(state_dim=2, n_actions=2, hidden=(2, 2), seed=7)


def _result(best_ret=0.30, oos_ret=0.05):
    return {
        "agent": _agent(),
        "history": [{"total_ret": -3.0}],
        "best_ret": best_ret,
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": oos_ret, "train_ret": best_ret,
                       "hard_rejected": False, "reason": ""},
        "pine_code": "", "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    # 增量门桩掉：本测试关注族群交替/晋升语义，多次调用用同一份 df
    monkeypatch.setattr(eng, "_data_incremented", lambda *a, **k: True)
    holder = {"result": None}
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None: holder["result"])
    registered: list = []
    monkeypatch.setattr(ev, "register_dynamic", lambda name, spec: registered.append(name))

    async def _no_cross_oos(*a, **k):
        return None

    async def _no_register_rl(*a, **k):
        registered.append("rl_evolve")
        return None
    monkeypatch.setattr(eng, "_cross_symbol_oos", _no_cross_oos)
    monkeypatch.setattr(eng, "_register_rl_evolve", _no_register_rl)
    return eng, registered, holder


async def test_population_alternates_lineages(engine):
    """轮次交替：第1轮冠军谱系，第2轮挑战者谱系（各自续训起点不同）。"""
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)

    await eng._train_strategy_drl_once()
    assert eng._strategy_drl_status["lineage"] == "main"
    assert eng._strategy_drl_status["_pop_round"] == 1
    # 第 1 轮（冠军轮）：部署端 best 建立 + 注册
    assert eng.zoo.best_fitness("strategy_drl") == 0.30
    n_reg_after_champion = len(registered)

    # 第 2 轮：挑战者轮——只应更新挑战者槽位，冠军不动、不新增注册
    holder["result"] = _result(best_ret=0.31, oos_ret=0.04)
    await eng._train_strategy_drl_once()
    assert eng._strategy_drl_status["lineage"] == "challenger"
    assert eng._strategy_drl_status["_pop_round"] == 2
    assert eng.zoo.best_fitness("strategy_drl_alt") == 0.31
    assert eng.zoo.best_fitness("strategy_drl") == 0.30, "挑战者未晋升不得动冠军"
    assert len(registered) == n_reg_after_champion, "挑战者未晋升不得注册新策略"


async def test_challenger_promoted_when_significantly_better(engine):
    """挑战者显著优于冠军（OOS > 冠军×1/threshold）→ 晋升接管冠军槽位并注册。"""
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)
    await eng._train_strategy_drl_once()  # 冠军轮建立基线
    n_reg_after_champion = len(registered)

    # 挑战者 OOS 0.07 > 0.05×(1/0.95)≈0.0526 → 晋升
    holder["result"] = _result(best_ret=0.32, oos_ret=0.07)
    await eng._train_strategy_drl_once()
    assert eng._strategy_drl_status["lineage"] == "challenger"
    assert eng.zoo.best_fitness("strategy_drl") == 0.32, "晋升后冠军槽位应被挑战者接管"
    assert len(registered) > n_reg_after_champion, "晋升必须注册新策略"


async def test_challenger_kept_when_only_slightly_better(engine):
    """挑战者仅略优于冠军（未达 1/threshold 显著门槛）→ 不晋升，防噪音替换。"""
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)
    await eng._train_strategy_drl_once()
    n_reg_after_champion = len(registered)

    # 0.052 < 0.0526：略好但不到显著门槛 → 不晋升
    holder["result"] = _result(best_ret=0.31, oos_ret=0.052)
    await eng._train_strategy_drl_once()
    assert eng.zoo.best_fitness("strategy_drl") == 0.30, "略优不得晋升"
    assert eng.zoo.best_fitness("strategy_drl_alt") == 0.31
    assert len(registered) == n_reg_after_champion


async def test_force_always_trains_main_lineage(engine):
    """force=True（手动触发）恒走冠军谱系，不受交替节奏影响。"""
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)

    # 直接两次 force 触发：都应是冠军轮
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_status["lineage"] == "main"
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_status["lineage"] == "main"
    assert eng.zoo.best_fitness("strategy_drl") == 0.30


async def test_promotion_swaps_old_champion_into_challenger(engine):
    """P0-族群加强：挑战者晋升时旧冠军转入挑战者槽位（角色互换），不丢弃。

    A/B 实验显示互换比"晋升即丢弃旧冠军" final 提升约 +92%：两条谱系
    都保持"曾当过冠军"的强度，多样性不减、信息不丢。
    """
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)
    await eng._train_strategy_drl_once()  # 冠军轮建立基线 fitness=0.30

    # 挑战者显著优于冠军（0.07 > 0.05×1/0.95）→ 晋升
    holder["result"] = _result(best_ret=0.32, oos_ret=0.07)
    await eng._train_strategy_drl_once()
    assert eng.zoo.best_fitness("strategy_drl") == 0.32, "晋升后冠军槽位被挑战者接管"
    # 角色互换：旧冠军（fitness=0.30）应转入挑战者槽位，而非被丢弃
    alt_anchor = eng.zoo.best_info("strategy_drl_alt")
    assert alt_anchor.get("fitness") == 0.30, "旧冠军应转入挑战者槽位"
    assert alt_anchor.get("swapped_from_champion") is True
