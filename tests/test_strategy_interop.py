"""全部策略互通回归测试（2026-08）。

覆盖四个真实断点：
A 动态策略仅在个别页面才从 DB 恢复 → 页面访问顺序决定策略可用性
B 「统一策略」构建的参数无处落地 → 需要 /strategy-repo/params 校验+持久化
C 元控制器子策略写死内置三件套
D 各端点策略列表口径不一致（内置被过滤 / 同名动态与内置重复列出）
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from strategies import (get_dynamic, get_strategy, list_strategies, register_dynamic,
                        remove_dynamic)
from strategies.dynamic_store import restore_from_db


def _names():
    return [s["name"] for s in list_strategies()]


# ---------------- D: 列表口径与去重 ----------------

def test_list_strategies_has_no_duplicate_names():
    """同名动态策略覆盖内置项后，列表不得再出现两条同名（曾致下拉重复、Vue 重复 key）。"""
    assert _names().count("grid") == 1
    register_dynamic("grid", {"params": {"grid_count": 7}, "created_by": "unittest"})
    try:
        names = _names()
        assert names.count("grid") == 1, names
        grid = [s for s in list_strategies() if s["name"] == "grid"][0]
        assert grid.get("builtin") is not True
        assert grid["default_params"]["grid_count"] == 7
    finally:
        remove_dynamic("grid")


def test_unknown_strategy_error_lists_unique_names():
    with pytest.raises(ValueError) as e:
        get_strategy("__no_such_strategy__")
    listed = str(e.value).split("可用:")[1]
    assert listed.count("meta_controller") <= 1, listed


# ---------------- A: DB 恢复单一实现 ----------------

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows, fail=False):
        self.rows, self.fail = rows, fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        if self.fail:
            raise RuntimeError("DB 不可达")
        return _FakeResult(self.rows)


class _FakeDB:
    def __init__(self, rows, fail=False):
        self._s = _FakeSession(rows, fail)

    def session(self):
        return self._s


class _Row:
    def __init__(self, name, spec):
        self.name = name
        self.spec_json = spec


def test_restore_from_db_recovers_and_survives_bad_rows():
    """单条坏记录只跳过该条；曾因整体异常吞掉全部策略。"""
    rows = [
        _Row("interop_ok", json.dumps({"params": {"fast_period": 9}, "created_by": "unittest"})),
        _Row("interop_bad", "{ 这不是 JSON"),
        _Row("interop_notdict", "[1,2,3]"),
    ]
    try:
        count = asyncio.run(restore_from_db(_FakeDB(rows)))
        assert count == 1, count
        assert get_dynamic("interop_ok")["params"]["fast_period"] == 9
    finally:
        remove_dynamic("interop_ok")


def test_restore_from_db_db_failure_is_not_fatal():
    assert asyncio.run(restore_from_db(_FakeDB([], fail=True))) == 0


def test_restore_from_db_is_idempotent():
    rows = [_Row("interop_idem", json.dumps({"params": {}}))]
    try:
        assert asyncio.run(restore_from_db(_FakeDB(rows))) == 1
        assert asyncio.run(restore_from_db(_FakeDB(rows))) == 1
        assert _names().count("interop_idem") == 1
    finally:
        remove_dynamic("interop_idem")


# ---------------- B/C: 元控制器参数变更要真正重建 ----------------

def test_meta_update_params_rebuilds_sub_strategies():
    """改了 sub_strategies 必须换掉子策略实例——曾只写 params，实际仍跑旧策略。"""
    st = get_strategy("meta_controller")
    st.reset()
    st.update_params({"sub_strategies": "dual_ma,price_action"})
    assert st._strategy_names == ["dual_ma", "price_action"]
    st.update_params({"sub_strategies": "dual_ma,grid,factor_signal"})
    assert st._strategy_names == ["dual_ma", "grid", "factor_signal"]
    assert set(st._strategy_trades) == {"dual_ma", "grid", "factor_signal"}


def test_meta_update_params_keeps_position_entry():
    """调参不得丢弃持仓跟踪（_entry 用于止损止盈）。"""
    st = get_strategy("meta_controller")
    st.reset()
    st._entry = 123.0
    st.update_params({"sub_strategies": "dual_ma,grid"})
    assert st._entry == 123.0


def test_meta_update_params_reloads_meta_agent():
    """mode/model_path 变更后必须丢弃已加载的元模型，否则重训后仍跑旧权重。"""
    st = get_strategy("meta_controller")
    st.reset()
    st._meta_agent = object()
    st.update_params({"model_path": "data/models/does_not_exist.json"})
    assert st._meta_agent is None


# ---------------- C: 训练子策略池解析 ----------------

def test_resolve_meta_sub_strategies_prefers_user_selection():
    from drl.evolve_engine import resolve_meta_sub_strategies
    # 用内置策略名（测试进程注册表中恒可用），断言用户选定优先且容忍空白
    assert resolve_meta_sub_strategies(
        {"sub_strategies": "dual_ma, price_action ,grid"}
    ) == "dual_ma,price_action,grid"


def test_resolve_meta_sub_strategies_excludes_self():
    """元控制器把自己当子策略会无限递归。"""
    from drl.evolve_engine import resolve_meta_sub_strategies
    assert resolve_meta_sub_strategies(
        {"sub_strategies": "meta_controller,dual_ma"}).split(",") == ["dual_ma"]
    # 仅选自身 → 回退默认池：结果须与空选回退一致，且绝不含 meta_controller 自身
    # （不硬编码池大小：默认池随 rl_evolve 是否已注册可能为 3 或 4 项）
    fallback = resolve_meta_sub_strategies({"sub_strategies": "  "})
    assert resolve_meta_sub_strategies({"sub_strategies": "meta_controller"}) == fallback
    assert "meta_controller" not in fallback.split(",")


def test_resolve_meta_sub_strategies_falls_back_and_dedupes():
    from strategies import list_strategies
    from drl.evolve_engine import DEFAULT_META_SUBS, resolve_meta_sub_strategies
    # 默认池按设计会过滤「当前未注册」的项（如首启时 rl_evolve 尚未注册），
    # 期望值需按同样规则从 DEFAULT_META_SUBS 推导，而非与常量直接比较
    available = {s["name"] for s in list_strategies()}
    expected = ",".join(dict.fromkeys(
        n for n in DEFAULT_META_SUBS.split(",") if n in available and n != "meta_controller")
    ) or "dual_ma,factor_signal"
    assert resolve_meta_sub_strategies(None) == expected
    assert resolve_meta_sub_strategies({"sub_strategies": "  "}) == expected
    assert resolve_meta_sub_strategies(
        {"sub_strategies": "grid,dual_ma,grid"}) == "grid,dual_ma"


# ---------------- B: POST /strategy-repo/params ----------------

class _Row2:
    def __init__(self, spec_json="{}"):
        self.spec_json = spec_json


class _Result2:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _Session2:
    def __init__(self, store, row):
        self.store, self.row = store, row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        return _Result2(self.row)

    async def commit(self):
        self.store["committed"] = True

    def add(self, obj):
        self.row = obj


class _DB2:
    def __init__(self, row=None):
        self.row, self.store = row, {}

    def session(self):
        return _Session2(self.store, self.row)


class _Engine2:
    def __init__(self, current="other"):
        self.strategy = SimpleNamespace(name=current)
        self.applied = []

    async def apply_strategy_params(self, name, params):
        self.applied.append((name, params))
        return params


def _make_client(engine, db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.strategy_repo import router
    from web.deps import get_db, get_engine

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


def test_params_endpoint_saves_and_persists():
    register_dynamic("interop_api", {"params": {"fast_period": 10},
                                     "executor": "dual_ma", "created_by": "unittest"})
    row = _Row2(json.dumps({"params": {"fast_period": 10}}))
    try:
        c = _make_client(_Engine2(current="other"), _DB2(row))
        r = c.post("/api/strategy-repo/params", json={
            "name": "interop_api", "params": {"fast_period": 21, "bogus_key": 5}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["params"]["fast_period"] == 21
        assert body["ignored_params"] == ["bogus_key"], "schema 外参数应回报而非静默丢弃"
        assert body["applied_live"] is False
        assert get_dynamic("interop_api")["params"]["fast_period"] == 21
        assert json.loads(row.spec_json)["params"]["fast_period"] == 21, "必须落库"
    finally:
        remove_dynamic("interop_api")


def test_params_endpoint_hot_applies_when_current():
    register_dynamic("interop_api2", {"params": {"fast_period": 10},
                                      "executor": "dual_ma", "created_by": "unittest"})
    eng = _Engine2(current="interop_api2")
    try:
        c = _make_client(eng, _DB2(_Row2("{}")))
        r = c.post("/api/strategy-repo/params", json={
            "name": "interop_api2", "params": {"fast_period": 33}})
        assert r.json()["applied_live"] is True
        assert eng.applied and eng.applied[0][0] == "interop_api2"
        assert eng.applied[0][1]["fast_period"] == 33
    finally:
        remove_dynamic("interop_api2")


def test_params_endpoint_rejects_builtin_and_garbage():
    c = _make_client(_Engine2(), _DB2(_Row2("{}")))
    # 内置策略没有 spec 记录，不该经此端点改参数
    assert c.post("/api/strategy-repo/params",
                  json={"name": "dual_ma", "params": {"fast_period": 3}}).status_code == 400
    register_dynamic("interop_api3", {"params": {"fast_period": 10},
                                      "executor": "dual_ma", "created_by": "unittest"})
    try:
        # 全部键不在 schema 内 → 拒绝（而非静默"保存成功"）
        r = c.post("/api/strategy-repo/params",
                   json={"name": "interop_api3", "params": {"nope": 1, "also_nope": 2}})
        assert r.status_code == 400
        assert "schema" in r.json()["detail"]
        # 空 params → 拒绝
        assert c.post("/api/strategy-repo/params",
                      json={"name": "interop_api3", "params": {}}).status_code == 400
    finally:
        remove_dynamic("interop_api3")
