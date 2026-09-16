"""进化因子部署 API 端点测试（Task 4）。"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


class _EngineHolder:
    def __init__(self, evolve):
        self.evolve = evolve


@pytest.fixture
def api(tmp_path):
    # 偏差（简报逐字测试无法通过，见 task-4-report.md 自审）：strategies._DYNAMIC
    # 是进程级注册表，本文件两个用例都部署 BTC/USDT 且断言 version==v1——
    # 前序用例残留会让后序用例拿到 v2。这里在用例前后清掉 evolve_combo_* 残留
    # （只动本任务命名空间，不碰其它动态策略），使每个 API 用例从 v1 起算。
    from strategies import dynamic_names, remove_dynamic
    for n in list(dynamic_names()):
        if n.startswith("evolve_combo_"):
            remove_dynamic(n)
    from web.api.evolve import router
    app = FastAPI()
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    app.state.engine = _EngineHolder(eng)
    app.include_router(router)
    yield TestClient(app)
    for n in list(dynamic_names()):
        if n.startswith("evolve_combo_"):
            remove_dynamic(n)


def test_list_empty(api):
    r = api.get("/api/evolve/factor-strategies")
    assert r.status_code == 200
    d = r.json()
    assert d["strategies"] == [] and d["count"] == 0


def test_list_contains_deployed(api):
    # 直接调引擎方法部署一次，验证列表透出权重/OOS/版本
    eng = api.app.state.engine.evolve

    async def _deploy():
        return await eng._deploy_factor_strategy(
            "BTC/USDT", {"vol_ratio": 0.5}, {"enabled": True, "valid": True},
            {"fitness": 0.02, "selected_factors": ["vol_ratio"], "round_no": 3})
    import asyncio
    asyncio.run(_deploy())

    r = api.get("/api/evolve/factor-strategies")
    assert r.status_code == 200
    items = r.json()["strategies"]
    assert len(items) == 1
    item = items[0]
    assert item["name"] == "evolve_combo_BTC_USDT"
    assert item["symbol"] == "BTC/USDT"
    assert item["version"] == "v1"
    assert item["weights"] == {"vol_ratio": 0.5}
    assert item["oos_report"]["valid"] is True
    assert item["selected_factors"] == ["vol_ratio"]


def test_manual_deploy_404_without_weights(api):
    r = api.post("/api/evolve/factor/deploy", params={"symbol": "BTC/USDT"})
    assert r.status_code == 404
    assert "暂无进化因子权重" in r.json()["detail"]


def test_manual_deploy_ok_with_weights(api):
    eng = api.app.state.engine.evolve
    wpath = eng.zoo.models_dir / "_cascade_weights_BTC_USDT.json"
    wpath.write_text(json.dumps({"vol_ratio": 0.5, "rsi_osc": -0.3}), encoding="utf-8")

    r = api.post("/api/evolve/factor/deploy", params={"symbol": "BTC/USDT"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True
    assert d["name"] == "evolve_combo_BTC_USDT"
    assert d["version"] == "v1"


def test_manual_deploy_rejects_bad_symbol(api):
    r = api.post("/api/evolve/factor/deploy", params={"symbol": "not-a-symbol"})
    assert r.status_code == 400


def test_manual_deploy_uses_companion_report(api):
    """自实验 bug-fix：伴随报告文件存在时，手动部署还原权重对应的真实
    OOS 报告/fitness/选中因子（此前 report 恒为 null、fitness 取最新轮次值）。"""
    eng = api.app.state.engine.evolve
    eng.zoo.models_dir.joinpath("_cascade_weights_BTC_USDT.json").write_text(
        json.dumps({"vol_ratio": 0.5}), encoding="utf-8")
    eng.zoo.models_dir.joinpath("_cascade_report_BTC_USDT.json").write_text(
        json.dumps({"fitness": 0.077,
                    "oos_report": {"enabled": True, "valid": True, "rank_ic": 0.05},
                    "selected_factors": ["vol_ratio"], "round_no": 9}), encoding="utf-8")

    r = api.post("/api/evolve/factor/deploy", params={"symbol": "BTC/USDT"})
    assert r.status_code == 200, r.text
    em = r.json()["spec"]["evolve_meta"]
    assert em["oos_report"]["valid"] is True and em["oos_report"]["rank_ic"] == 0.05
    assert em["fitness"] == 0.077
    assert em["selected_factors"] == ["vol_ratio"]
    assert em["round_no"] == 9


def test_manual_deploy_carries_previous_report_without_companion(api):
    """自实验 bug-fix：无伴随文件时回退携带上一版 spec 的 OOS 报告与 fitness，
    避免手动重部署把报告覆盖成 null。"""
    import asyncio as _aio
    eng = api.app.state.engine.evolve
    _aio.run(eng._deploy_factor_strategy(
        "BTC/USDT", {"vol_ratio": 0.5},
        {"enabled": True, "valid": True, "rank_ic": 0.04},
        {"fitness": 0.02, "selected_factors": ["vol_ratio"], "round_no": 3}))
    eng.zoo.models_dir.joinpath("_cascade_weights_BTC_USDT.json").write_text(
        json.dumps({"vol_ratio": 1.5}), encoding="utf-8")
    # 不写伴随文件 → 走回退

    r = api.post("/api/evolve/factor/deploy", params={"symbol": "BTC/USDT"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["version"] == "v2"
    em = d["spec"]["evolve_meta"]
    assert em["oos_report"]["valid"] is True, "无伴随文件时应携带上一版 OOS 报告"
    assert em["fitness"] == 0.02, "无伴随文件时应携带上一版 fitness"
