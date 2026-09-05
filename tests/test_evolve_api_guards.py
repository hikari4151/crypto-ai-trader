"""进化 API 与 ModelZoo 输入边界测试。"""

import asyncio

import pytest

from drl.model_zoo import ModelZoo


def test_model_zoo_rejects_path_traversal_name(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    with pytest.raises(ValueError, match="非法模型名"):
        zoo.list_versions("../outside")


def test_model_zoo_rejects_separator_name(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    with pytest.raises(ValueError, match="非法模型名"):
        zoo.best_version("nested/model")


def test_model_zoo_rejects_non_positive_version(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    with pytest.raises(ValueError, match="非法版本"):
        zoo.load_agent("factor_miner", 0)


def test_model_zoo_rejects_boolean_version(tmp_path):
    zoo = ModelZoo(tmp_path / "models")
    with pytest.raises(ValueError, match="非法版本"):
        zoo.load_agent("factor_miner", True)


class _ConfigStub:
    async def apply_config(self, changes):
        return {"ok": True, "applied": changes, "persisted": False}


class _ConfigEngineStub:
    def __init__(self):
        self._symbols = ["BTC/USDT"]
        self._symbol_idx = 0
        self._cycle_pipelines_done = set()
        self._last_tail_ts = {}


class _EvolveStub:
    def __init__(self):
        self.zoo = ModelZoo(__import__("pathlib").Path("data/models"))
        self._train_lock = asyncio.Lock()
        self._running = False

    async def trigger_factor_miner(self):
        return {"ok": False, "reason": "持续进化引擎未运行"}


def test_apply_config_rejects_string_symbols_at_engine_boundary(tmp_path):
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    from core.bus import EventBus

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    result = asyncio.run(eng.apply_config({"symbols": "BTC/USDT"}))
    assert result["ok"] is False
    assert "列表" in result["error"]


def test_apply_config_rejects_string_false_for_bool(tmp_path):
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    from core.bus import EventBus

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    result = asyncio.run(eng.apply_config({"cross_symbol_oos": "false"}))
    assert result["ok"] is False
    assert "布尔" in result["error"]


def test_apply_config_rejects_unknown_key(tmp_path):
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    from core.bus import EventBus

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    result = asyncio.run(eng.apply_config({"unexpected": 1}))
    assert result["ok"] is False
    assert "未知配置项" in result["error"]


def test_apply_config_rejects_boolean_for_numeric_setting(tmp_path):
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    from core.bus import EventBus

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    result = asyncio.run(eng.apply_config({"meta_episodes": True}))
    assert result["ok"] is False
    assert "数字" in result["error"]


def test_trigger_not_running_uses_reason_in_api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.evolve import router

    app = FastAPI()
    app.state.engine = type("Engine", (), {"evolve": _EvolveStub()})()
    app.include_router(router)
    response = TestClient(app).post("/api/evolve/trigger/factor-miner")
    assert response.status_code == 400
    assert response.json()["detail"] == "持续进化引擎未运行"
