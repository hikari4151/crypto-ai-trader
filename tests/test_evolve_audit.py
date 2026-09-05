"""持续进化手动回退审计与 API 契约测试（Task 5）。

1. /api/evolve/rollback 必须走 EvolveEngine.deploy_model（outcome=manual_rollback），
   返回运行时 reload 结果，不再直调 zoo.rollback；
2. 状态区分 candidate_fitness 与 deployed_fitness/deployed_version（回退后候选
   值不得污染"当前部署"画像）；
3. /api/evolve/versions 暴露身份/部署元数据摘要。
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.api.evolve import router


class _ZooStub:
    """最小 ModelZoo 替身：供 versions/rounds API 查询。"""

    def list_versions(self, name):
        return [
            {"version": 1, "timestamp": 1.0,
             "meta": {"fitness": 0.7, "symbol": "ETH/USDT",
                      "timeframe": "5m", "config_fingerprint": "abc"}}]

    def best_version(self, name):
        return 3

    def best_fitness(self, name):
        return 0.9

    def best_info(self, name):
        return {
            "fitness": 0.9, "version": 3, "data_source": "exchange",
            "symbol": "ETH/USDT", "timeframe": "5m",
            "config_fingerprint": "abc", "comparable": True}


class _EvolveStub:
    """最小 EvolveEngine 替身：只实现 API 用到的入口与 zoo 查询。"""

    def __init__(self, calls=None, result=None):
        self.calls = calls if calls is not None else []
        self.result = result or {
            "ok": True, "version": 3, "meta": {"fitness": 0.9},
            "runtime_reload": {"status": "reloaded"},
            "registered": True, "error": None}
        self._train_lock = __import__("asyncio").Lock()
        self.zoo = _ZooStub()

    async def deploy_model(self, name, version, **kwargs):
        self.calls.append((name, version, kwargs["outcome"]))
        return self.result


def _make_app(evolve):
    app = FastAPI()
    app.state.engine = type("Engine", (), {"evolve": evolve})()
    app.include_router(router)
    return TestClient(app)


def test_manual_rollback_api_uses_engine_deployment_and_returns_runtime_status():
    calls = []
    client = _make_app(_EvolveStub(calls=calls))

    response = client.post("/api/evolve/rollback?name=strategy_drl&version=3")

    assert response.status_code == 200
    assert calls == [("strategy_drl", 3, "manual_rollback")]
    assert response.json()["runtime_reload"]["status"] == "reloaded"
    assert response.json()["registered"] is True


def test_manual_rollback_missing_version_maps_to_404():
    stub = _EvolveStub(result={
        "ok": False, "version": 99, "meta": {},
        "runtime_reload": {"status": "failed"},
        "registered": False, "error": "版本 v99 不存在"})
    client = _make_app(stub)

    response = client.post("/api/evolve/rollback?name=strategy_drl&version=99")

    assert response.status_code == 404


def test_manual_rollback_reload_failure_maps_to_409():
    stub = _EvolveStub(result={
        "ok": False, "version": 3, "meta": {},
        "runtime_reload": {"status": "failed", "error": "维度不匹配"},
        "registered": False, "error": "维度不匹配"})
    client = _make_app(stub)

    response = client.post("/api/evolve/rollback?name=strategy_drl&version=3")

    assert response.status_code == 409


def test_versions_exposes_identity_metadata():
    client = _make_app(_EvolveStub())

    response = client.get("/api/evolve/versions?name=strategy_drl")

    assert response.status_code == 200
    body = response.json()
    assert body["best_version"] == 3
    assert body["best_fitness"] == 0.9
    assert body["versions"][0]["meta"]["symbol"] == "ETH/USDT"
    assert body["best_info"]["comparable"] is True


@pytest.mark.asyncio
async def test_deploy_status_separates_candidate_from_deployed(tmp_path):
    from core.bus import EventBus
    from drl.agent import ACAgent
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng.zoo.save_agent(ACAgent(state_dim=2, n_actions=2, hidden=(2, 2), seed=7),
                       "strategy_drl", meta={"fitness": 0.9, "data_source": "exchange"})
    eng._mark_deployment_status("strategy_drl", outcome="rollback",
                                candidate_fitness=0.2)

    st = eng.status()["strategy_drl"]
    assert st["candidate_fitness"] == 0.2, "回退后候选值仍可见（研究口径）"
    assert st["deployed_fitness"] == 0.9, "deployed 必须是当前 best 快照"
    assert st["deployed_version"] == 1
    assert st["last_outcome"] == "rollback"


def test_rounds_returns_manual_rollback_audit(tmp_path):
    import asyncio
    from core.database import Database, EvolveRound

    db = Database("sqlite+aiosqlite:///:memory:")
    asyncio.run(db.init())

    async def _seed():
        async with db.session() as s:
            s.add(EvolveRound(model="strategy_drl", symbol="ETH/USDT", timeframe="5m",
                              round_no=1, status="manual_rollback",
                              audit_json=json.dumps(
                                  {"target_version": 2, "reason": "UI 手动回退",
                                   "runtime_reload": "reloaded"}),
                              selected_factors=json.dumps(["rsi(close,14)"])))
            await s.commit()
    asyncio.run(_seed())

    class _EvolveStubDB(_EvolveStub):
        def __init__(self):
            super().__init__()
            self.db = db
    client = _make_app(_EvolveStubDB())

    response = client.get("/api/evolve/rounds?model=strategy_drl")

    assert response.status_code == 200
    rows = response.json()["rounds"]
    assert rows and rows[0]["status"] == "manual_rollback"
    assert rows[0]["audit"]["target_version"] == 2
    assert rows[0]["selected_factors"] == ["rsi(close,14)"]