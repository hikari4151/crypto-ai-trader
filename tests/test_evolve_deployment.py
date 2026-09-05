"""持续进化统一部署编排测试（Task 2）。

deploy_model 是自动接受、自动回退、手动回退共用的部署入口：它必须
1. 先从版本快照原子恢复 best/flat；
2. 触发动态策略注册；
3. 等待运行时 reload 回调，成功才算部署完成；
4. 回调失败不得伪报成功，且状态标记 last_outcome=failed。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


def _agent():
    return ACAgent(state_dim=2, n_actions=2, hidden=(2, 2), seed=7)


@pytest.fixture
def engine(tmp_path):
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


@pytest.mark.asyncio
async def test_deploy_model_calls_reload_after_snapshot_restore(engine):
    eng = engine
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 0.7, "data_source": "exchange"})
    calls = []

    async def reload_model(name, version, meta):
        calls.append((name, version, meta["fitness"]))
        return {"status": "reloaded"}

    eng.set_deployment_reload(reload_model)
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")

    assert result["ok"] is True
    assert calls == [("strategy_drl", 1, 0.7)]
    assert result["runtime_reload"]["status"] == "reloaded"
    ds = eng._deploy_status["strategy_drl"]
    assert ds["deployed_version"] == 1
    assert ds["deployed_fitness"] == 0.7
    assert ds["last_outcome"] == "manual_rollback"


@pytest.mark.asyncio
async def test_deploy_model_reports_reload_failure_and_does_not_claim_success(engine):
    eng = engine
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 0.7, "data_source": "exchange"})

    async def reload_model(name, version, meta):
        return {"status": "failed", "error": "dimension mismatch"}

    eng.set_deployment_reload(reload_model)
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")

    assert result["ok"] is False
    assert result["runtime_reload"]["status"] == "failed"
    assert "dimension mismatch" in result["error"]
    assert eng._deploy_status["strategy_drl"]["last_outcome"] == "failed"


@pytest.mark.asyncio
async def test_deploy_model_restores_previous_deployment_on_reload_failure(engine):
    eng = engine
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 0.8, "data_source": "exchange"})
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 0.9, "data_source": "exchange"})

    async def reload_model(name, version, meta):
        return {"status": "failed", "error": "model mismatch"}

    eng.set_deployment_reload(reload_model)
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")

    assert result["ok"] is False
    # reload 失败应恢复"部署前的 best"（v2, fitness=0.9），而不是卡在目标 v1
    assert eng.zoo.best_fitness("strategy_drl") == 0.9, "回退失败应恢复原部署"
    assert eng.zoo.best_version("strategy_drl") == 2


@pytest.mark.asyncio
async def test_deploy_model_missing_version_reports_not_found(engine):
    eng = engine
    result = await eng.deploy_model("strategy_drl", 99, outcome="manual_rollback")
    assert result["ok"] is False
    assert "99" in result["error"]


@pytest.mark.asyncio
async def test_deploy_model_without_reload_callback_marks_skipped(engine):
    eng = engine
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 0.7, "data_source": "exchange"})
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")
    assert result["ok"] is True
    assert result["runtime_reload"]["status"] == "skipped"
