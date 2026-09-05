# 持续进化回退基线优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让持续进化的自动接受、自动回退和手动回退都以完整部署快照为事实来源，并确保磁盘、动态注册表、当前运行实例和回退基线状态始终指向同一版本。

**Architecture:** 保留 `ModelZoo` 现有 `vN.json.gz`、`best.json.gz` 和 flat JSON 兼容格式，在其上增加规范化部署元数据与原子 `restore_deployment`。`EvolveEngine` 提供统一部署编排，接收 `TradingEngine` 的可选异步 reload 回调；锚点比较使用稳定的训练身份指纹，状态与 API 区分候选结果、部署结果和人工审计。

**Tech Stack:** Python 3.13、FastAPI、SQLAlchemy async、SQLite WAL、gzip JSON、pytest、pytest-asyncio、Vue 3 CDN 单体前端。

## Global Constraints

- 保留 `docs/REDEV_GUIDE.md` 的 D5/D6/D7/D9/D11 红线：OOS 门、回退落库、重启注册、flat 兼容和训练周期解耦不能回退。
- 不重写 PPO、因子搜索、OOS 统计算法、交易撮合、下单和风险规则。
- 不删除旧版本；缺少新身份字段的旧版本按未知口径处理，不进行跨口径回退比较。
- 自动接受、自动回退、手动回退必须共用同一个部署编排入口。
- 任意部署层写入或运行时 reload 失败都不能向 API 或状态报告成功。
- 编辑文件前先读取当前内容；不撤销工作区中已有的用户改动；所有新增文本默认 ASCII。
- 每个任务按“先写失败测试、确认失败、最小实现、确认通过、提交”执行；单元测试用 `.venv\\Scripts\\python -m pytest`。

---

### Task 1: ModelZoo 完整部署快照与原子恢复

**Files:**
- Modify: `drl/model_zoo.py:93-205, 373-446, 545-590`
- Test: `tests/test_evolve_anchor.py`
- Test: `tests/test_model_zoo.py`（若不存在则创建）

**Interfaces:**
- Produces `ModelZoo.restore_deployment(name: str, version: int, *, reason: str = "", outcome: str = "rollback") -> dict`。
- `restore_deployment` 返回 `{"version": int, "meta": dict, "best_path": str, "flat_path": str}`；目标不存在抛出 `FileNotFoundError`，目标损坏或原子写入失败抛出 `ModelZooError`（在 `model_zoo.py` 定义）。
- `save_agent_best` 和现有 `rollback` 最终复用完整快照恢复逻辑，不再用 `agent.to_dict()` 丢弃部署字段。

- [ ] **Step 1: 写失败测试，锁定完整元数据 round-trip 和版本裁剪不变量**

在 `tests/test_evolve_anchor.py` 增加：

```python
def test_restore_deployment_preserves_all_deployment_metadata(zoo):
    agent = _agent()
    zoo.save_agent(
        agent, "strategy_drl",
        meta={
            "fitness": 0.42, "data_source": "exchange", "oos_ret": 0.08,
            "symbol": "BTC/USDT", "timeframe": "1h", "state_window": 3,
            "factor_expression": "rsi(close, 14)", "factor_mu": 1.25,
            "factor_sd": 0.5, "min_trade_zone": 0.05,
            "pine_code": "//@version=5\nstrategy('x')",
        },
    )
    zoo.save_agent(agent, "strategy_drl", meta={"fitness": 0.10}, is_best=False)

    result = zoo.restore_deployment("strategy_drl", 1, reason="test")

    assert result["version"] == 1
    data = zoo._read_json_gz(zoo._best_path("strategy_drl"))
    assert data["_meta"]["factor_expression"] == "rsi(close, 14)"
    assert data["_meta"]["factor_mu"] == 1.25
    assert data["_meta"]["pine_code"].startswith("//@version")
    flat = json.loads(zoo._flat_path("strategy_drl").read_text(encoding="utf-8"))
    assert flat["factor_expression"] == "rsi(close, 14)"
    assert flat["min_trade_zone"] == 0.05
```

Add an injected-write failure test that monkeypatches `_write_flat_json` to raise and asserts best bytes, flat bytes, and `meta.json.best_version` remain unchanged. Add `max_versions=1` repeated-save coverage asserting `meta.best_version` always has an existing `v{best}.json.gz`.

- [ ] **Step 2: 运行新增测试确认当前实现失败**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_anchor.py -k "restore_deployment or atomic or max_versions"`
Expected: FAIL because `restore_deployment` and atomic failure semantics do not exist, and `save_agent_best` currently reconstructs the flat payload from `agent.to_dict()`.

- [ ] **Step 3: 实现完整快照和原子恢复**

In `drl/model_zoo.py`:

1. Add `ModelZooError(RuntimeError)`.
2. Add `_atomic_write_bytes(path, payload)` using a unique tempfile in the same directory, flush/close, then `os.replace`; clean only the temporary file on failure.
3. Add `_read_version_payload(name, version)` that validates and returns the complete JSON payload including `_meta` and `_version`.
4. Implement `restore_deployment` to read the target payload, normalize only missing deployment metadata to explicit `None`/`"unknown"`, create gzip best bytes and flat bytes from the same payload, atomically replace both files, then update `meta.json` last. If the second file fails, restore the original bytes for any file already replaced before raising `ModelZooError`.
5. Make `rollback` validate the target then call `restore_deployment`; set `rolled_back_at`, `rollback_reason`, and `rollback_outcome` in the target snapshot metadata without inheriting current best evaluation fields.
6. Make `save_agent_best` use the existing best version payload when available, preserving all deployment fields; for compatibility when only an `ACAgent` is supplied, merge explicit fields into its serialized payload without overwriting fields absent from the input.
7. Fix the version pruning loop so the newly saved version cannot be popped when `max_versions == 1`; prune a non-best old version before appending/reordering and assert/log when the invariant would be violated.
8. Keep `_write_meta` cache invalidation and best metadata cache behavior intact.

- [ ] **Step 4: 运行 ModelZoo 测试确认通过**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_anchor.py tests/test_model_zoo.py`
Expected: PASS, including legacy anchor/demo/reset tests and new metadata/atomicity/max-version tests.

- [ ] **Step 5: 提交独立变更**

```bash
git add drl/model_zoo.py tests/test_evolve_anchor.py tests/test_model_zoo.py
git commit -m "fix: restore evolve deployments as complete snapshots"
```

### Task 2: EvolveEngine 统一部署编排与运行时回调

**Files:**
- Modify: `drl/evolve_engine.py:183-280, 303-370, 1058-1103, 1277-1325, 1431-1512, 1602-1685`
- Test: `tests/test_evolve_anchor.py`
- Test: `tests/test_evolve_deployment.py`（创建）

**Interfaces:**
- Adds `EvolveEngine.set_deployment_reload(callback: Optional[Callable[[str, int, dict], Awaitable[dict | None]]]) -> None`.
- Adds `async EvolveEngine.deploy_model(name: str, version: int, *, outcome: str, symbol: str = "", timeframe: str = "", reason: str = "") -> dict`.
- Callback receives model name, deployed version, and complete metadata; it returns `{"status": "reloaded"|"skipped"|"failed", ...}` or `None`.
- `deploy_model` returns `{"ok": bool, "version": int, "meta": dict, "runtime_reload": dict, "registered": bool, "error": str | None}`.

- [ ] **Step 1: 写失败测试，验证统一入口和失败不伪报成功**

Create `tests/test_evolve_deployment.py` with a small `ModelZoo` fixture and an engine fixture. Add:

```python
@pytest.mark.asyncio
async def test_deploy_model_calls_reload_after_snapshot_restore(tmp_path):
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng.zoo.save_agent(_agent(), "strategy_drl", meta={"fitness": 0.7, "data_source": "exchange"})
    calls = []

    async def reload_model(name, version, meta):
        calls.append((name, version, meta["fitness"]))
        return {"status": "reloaded"}

    eng.set_deployment_reload(reload_model)
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")

    assert result["ok"] is True
    assert calls == [("strategy_drl", 1, 0.7)]
    assert result["runtime_reload"]["status"] == "reloaded"


@pytest.mark.asyncio
async def test_deploy_model_reports_reload_failure_and_does_not_claim_success(tmp_path):
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng.zoo.save_agent(_agent(), "strategy_drl", meta={"fitness": 0.7, "data_source": "exchange"})

    async def reload_model(name, version, meta):
        return {"status": "failed", "error": "dimension mismatch"}

    eng.set_deployment_reload(reload_model)
    result = await eng.deploy_model("strategy_drl", 1, outcome="manual_rollback")

    assert result["ok"] is False
    assert result["runtime_reload"]["status"] == "failed"
    assert "dimension mismatch" in result["error"]
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_deployment.py`
Expected: FAIL because the callback setter and `deploy_model` do not exist.

- [ ] **Step 3: 实现统一部署入口并接入三条管线**

In `drl/evolve_engine.py`:

1. Initialize `self._deployment_reload = None` and status defaults for `candidate_fitness`, `deployed_fitness`, `deployed_version`, `deployed_source`, `last_outcome`, `anchor_mismatch`, and `runtime_reload` for all three models.
2. Implement `set_deployment_reload` and `deploy_model`. Acquire `_train_lock` only in the public orchestration caller; do not re-acquire it inside a caller that already holds it. Capture the old best payload before restore, call `zoo.restore_deployment`, call `_register_rl_evolve` for `strategy_drl` or the existing meta-controller registration helper for `meta_controller`, then await the callback. On callback failure, call `restore_deployment` for the captured old version when available, return `ok=False`, and set `last_outcome="failed"`.
3. Add a private `_mark_deployment_status` helper that sets `candidate_fitness` separately from `deployed_fitness` and refreshes the latter from `zoo.best_info` after every accepted/rollback path.
4. Replace factor, strategy, and meta automatic `save_agent_best` rollback calls with `deploy_model` using the actual anchor version. Preserve existing `_log_round` and event behavior, but publish the deployed version/outcome in the event payload.
5. After accepted `save_agent`, call `deploy_model` only when the new version is already written, or factor the write and deployment into a helper so the new best uses the same reload/registration sequence. A candidate that fails runtime reload must not leave `last_outcome="accepted"`.
6. Preserve demo archive, OOS reject, and cross-symbol reject semantics; set their `candidate_fitness` and `last_outcome` without changing deployed fields.

- [ ] **Step 4: 运行部署与已有锚点/OOS 测试**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_deployment.py tests/test_evolve_anchor.py tests/test_evolve_oos_gate.py`
Expected: PASS; automatic rollback still preserves the anchor, while the new tests verify callback invocation and failure reporting.

- [ ] **Step 5: 提交独立变更**

```bash
git add drl/evolve_engine.py tests/test_evolve_deployment.py tests/test_evolve_anchor.py
git commit -m "feat: unify evolve deployment and rollback handling"
```

### Task 3: TradingEngine 与策略实例热加载

**Files:**
- Modify: `engine/trading_engine.py:35-90, 1288-1325`
- Modify: `strategies/rl_adaptive.py:54-103`
- Modify: `strategies/meta.py:184-226, 265-283`
- Modify: `web/main.py:75-95`
- Test: `tests/test_evolve_runtime_reload.py`（创建）

**Interfaces:**
- Adds `async TradingEngine.reload_active_strategy_model(model_name: str, version: int, meta: dict) -> dict`.
- `TradingEngine` injects `self.evolve.set_deployment_reload(self.reload_active_strategy_model)` after strategy and locks are initialized.
- Adds `RLAdaptiveStrategy.reload_model() -> bool` that clears the cached agent and model-side factor metadata, then loads the current `model_path`.
- Adds `MetaControllerStrategy.reload_model() -> bool` that clears `_meta_agent`/state dimension and reloads its current `model_path`.

- [ ] **Step 1: 写失败测试，验证实例不会继续使用旧缓存**

Create `tests/test_evolve_runtime_reload.py` with fake model paths and monkeypatched `ACAgent.load`. Assert `RLAdaptiveStrategy._agent` changes after `reload_model`, `MetaControllerStrategy._meta_agent` is cleared/reloaded, and a `TradingEngine` reload callback executes under `_strategy_lock` without replacing a non-target current strategy.

```python
@pytest.mark.asyncio
async def test_trading_engine_reload_refreshes_active_rl_instance(monkeypatch):
    engine = TradingEngine(_MetaDB(), EventBus())
    strategy = RLAdaptiveStrategy()
    strategy.params["model_path"] = "data/models/strategy_drl.json"
    old = object()
    new = object()
    strategy._agent = old
    monkeypatch.setattr(strategy, "_load_agent", lambda: setattr_and_true(strategy, "_agent", new))
    engine.strategy = strategy

    result = await engine.reload_active_strategy_model("strategy_drl", 4, {})

    assert result["status"] == "reloaded"
    assert strategy._agent is new
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_runtime_reload.py`
Expected: FAIL because no reload methods exist and the current strategy only loads an agent when `_agent is None`.

- [ ] **Step 3: 实现带策略锁的 reload**

1. In `RLAdaptiveStrategy`, factor the metadata reset/load sequence from `_load_agent` into `reload_model`; keep `_load_agent` as the lazy first-load path. On a failed reload, leave `_agent=None` and return `False` so the engine reports a runtime failure rather than using stale weights.
2. In `MetaControllerStrategy`, add `reload_model` that clears `_meta_agent`, `_meta_state_dim`, factor metadata and then calls `_load_meta_agent`; keep normal lazy loading unchanged.
3. In `TradingEngine`, implement `reload_active_strategy_model` under `_strategy_lock`. For `strategy_drl`/`rl_evolve` targets, reload the active strategy only when it is an RL adaptive instance or a dynamic strategy backed by the DRL model. For `meta_controller`, reload the active meta instance. Return `skipped` for unrelated current strategies, `reloaded` on success, and `failed` with a concrete error on load failure.
4. Inject the callback into `EvolveEngine` in `TradingEngine.__init__` after `_strategy_lock` exists. Do not call `select_strategy`, do not change the user-selected strategy, and do not clear pending signals except when the current strategy itself is reloaded; in that case clear `_pending_signal` under the same lock.
5. Update `web/main.py` only if startup ordering requires callback injection after dynamic strategy restoration; preserve the documented order in `docs/REDEV_GUIDE.md`.

- [ ] **Step 4: 运行运行时测试和交易引擎相关回归**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_runtime_reload.py tests/test_engine* tests/test_evolve_anchor.py`
Expected: PASS; existing strategy switching and lock tests remain green.

- [ ] **Step 5: 提交独立变更**

```bash
git add engine/trading_engine.py strategies/rl_adaptive.py strategies/meta.py web/main.py tests/test_evolve_runtime_reload.py
git commit -m "fix: hot reload active evolve strategy models"
```

### Task 4: 训练身份指纹与因子 OOS 基线隔离

**Files:**
- Modify: `drl/evolve_engine.py:1005-1128, 1219-1434, 1523-1685, 1813-1862`
- Modify: `drl/factor_miner.py:170-205` only if report metadata must be propagated
- Test: `tests/test_evolve_anchor.py`
- Test: `tests/test_evolve_identity.py`（创建）

**Interfaces:**
- Adds `_training_identity(model: str, symbol: str, timeframe: str, *, state_window: int = 1, factor_signature: str = "", config: Optional[dict] = None, window_tail_ts: Optional[int] = None) -> dict`.
- Adds `_identity_mismatch(anchor_meta: dict, identity: dict) -> Optional[str]`.
- Each accepted version receives `training_identity` fields in `_meta`; old versions with missing fields are `unknown` and incomparable.

- [ ] **Step 1: 写失败测试，锁定跨品种/周期/配置不可比较**

Create `tests/test_evolve_identity.py`:

```python
def test_identity_mismatch_blocks_cross_symbol_comparison():
    anchor = {"model": "strategy_drl", "symbol": "BTC/USDT", "timeframe": "1h",
              "rolling_window": 5000, "state_window": 1,
              "factor_signature": "", "config_fingerprint": "abc"}
    current = {**anchor, "symbol": "ETH/USDT"}
    assert "symbol" in ev._identity_mismatch(anchor, current)


def test_missing_identity_is_unknown_not_comparable():
    assert ev._identity_mismatch({"symbol": "BTC/USDT"}, {"symbol": "BTC/USDT"})
```

Add an async training-path test that seeds a BTC anchor, trains an ETH candidate with a passing OOS gate, and asserts the candidate can establish a new best instead of being compared against BTC. Add factor OOS rejected coverage asserting best file and cascade files remain unchanged while archive receives the candidate.

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_identity.py`
Expected: FAIL because comparison currently only checks `best_info["comparable"]`, and factor candidates can call `save_agent(... is_best=True)` before the OOS cascade gate.

- [ ] **Step 3: 实现身份指纹和因子部署基线隔离**

1. Implement canonical identity construction with sorted JSON and SHA-256 for `config_fingerprint`; include only reward, split, state, factor and deployment-gate settings that affect comparability.
2. Persist identity fields in every new version’s `_meta`, including `symbol`, `timeframe`, `rolling_window`, `state_window`, `factor_signature`, `config_fingerprint`, and `window_tail_ts`.
3. Before each `_should_rollback` call, compute `_identity_mismatch`; when non-empty, set `anchor_mismatch`, skip rollback comparison, and allow only the candidate’s own OOS/deployment gate to decide acceptance.
4. Keep demo/exchange incomparable and keep unknown legacy anchors runnable but non-comparable.
5. Reorder factor miner flow so factor OOS gate runs before any `is_best=True` save. On failed OOS/ICIR/turnover checks, call `save_agent_archive` only; do not update factor best, cascade `.npy`, cascade metadata or cascade weights.
6. Ensure strategy/meta identity is propagated through accepted, rollback, and archive metadata without changing existing OOS thresholds.

- [ ] **Step 4: 运行身份/OOS/锚点回归**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_identity.py tests/test_evolve_anchor.py tests/test_evolve_oos_gate.py tests/test_oos_split.py`
Expected: PASS; same-identity rollback remains protected, different identity skips comparison, and factor OOS rejects do not become baselines.

- [ ] **Step 5: 提交独立变更**

```bash
git add drl/evolve_engine.py drl/factor_miner.py tests/test_evolve_identity.py tests/test_evolve_anchor.py
git commit -m "fix: isolate evolve anchors by training identity"
```

### Task 5: 回退状态、人工审计与 API/UI 契约

**Files:**
- Modify: `drl/evolve_engine.py:208-224, 555-614, 1813-1862`
- Modify: `core/database.py:95-145` only for an idempotent audit metadata column if required by the implementation
- Modify: `web/api/evolve.py:147-247`
- Modify: `web/static/index.html:2478-2490, 2596-2698, 3602-3675, 3747-3767`
- Test: `tests/test_evolve_api_guards.py`
- Test: `tests/test_evolve_audit.py`（创建）

**Interfaces:**
- `/api/evolve/status` exposes `candidate_fitness`, `deployed_fitness`, `deployed_version`, `deployed_source`, `last_outcome`, `anchor_mismatch`, `runtime_reload` for each pipeline.
- `/api/evolve/versions` returns the existing version list plus normalized identity/deployment metadata.
- `/api/evolve/rollback` calls `EvolveEngine.deploy_model(..., outcome="manual_rollback")` and returns its structured result; it never calls `zoo.rollback` directly.
- `EvolveRound.status` accepts `manual_rollback`; optional `audit_json` stores target/source/reason when a schema-compatible field is added.

- [ ] **Step 1: 写失败测试，锁定手动回退审计与状态一致性**

Create `tests/test_evolve_audit.py`:

```python
@pytest.mark.asyncio
async def test_manual_rollback_api_uses_engine_deployment_and_returns_runtime_status(monkeypatch):
    calls = []

    class EvolveStub:
        async def deploy_model(self, name, version, **kwargs):
            calls.append((name, version, kwargs["outcome"]))
            return {"ok": True, "version": version,
                    "runtime_reload": {"status": "reloaded"},
                    "registered": True}

    app = FastAPI()
    app.state.engine = type("Engine", (), {"evolve": EvolveStub()})()
    app.include_router(router)
    response = TestClient(app).post("/api/evolve/rollback?name=strategy_drl&version=3")

    assert response.status_code == 200
    assert calls == [("strategy_drl", 3, "manual_rollback")]
    assert response.json()["runtime_reload"]["status"] == "reloaded"
```

Add a status test asserting rollback keeps `candidate_fitness` at the candidate value while `deployed_fitness` and `deployed_version` come from the anchor. Add versions contract coverage for identity fields and a `manual_rollback` round audit row.

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_audit.py tests/test_evolve_api_guards.py`
Expected: FAIL because the API directly invokes `zoo.rollback`, status has no separated deployed fields, and manual rollback is not logged.

- [ ] **Step 3: 实现审计状态和 API/UI 展示**

1. Extend the status initialization and `_pipeline_view` to expose the fields in the interface block. Keep `fitness` as a backward-compatible alias for the latest candidate/current display only where existing UI requires it; add explicit deployed values for new consumers.
2. Update `_log_round` to accept a compact audit payload and serialize `selected_factors` as JSON with `json.dumps(..., ensure_ascii=False)` rather than Python `str(list(...))`; preserve old rows on read by accepting both formats. Use `status="manual_rollback"` for API actions.
3. Add an idempotent SQLite `ALTER TABLE` only if a dedicated `audit_json` column is needed; default old rows to `{}` and keep PostgreSQL initialization valid. Do not make existing DB initialization fail when the column already exists.
4. Change `/api/evolve/rollback` to validate model/version, call `deploy_model`, map missing versions to 404, snapshot corruption/runtime reload failure to 409/500 with the structured error, and return the full deployment result.
5. Extend `/api/evolve/versions` with normalized metadata fields and `/api/evolve/rounds` with audit data while retaining current response keys.
6. Update the evolve page to show candidate vs deployed fitness, deployed version/source, last outcome, runtime reload status and anchor mismatch. Use existing status/banner components and existing icon/button conventions; do not add a new card-inside-card layout. Keep reset-anchor behavior intact.

- [ ] **Step 4: 运行 API/UI 契约测试**

Run: `.venv\\Scripts\\python -m pytest -q tests/test_evolve_audit.py tests/test_evolve_api_guards.py tests/test_evolve_anchor.py`
Then run: `.venv\\Scripts\\python scripts/check_js.py`
Expected: PASS; manual rollback returns runtime status and audit outcome, old API keys remain available, and frontend JavaScript parses successfully.

- [ ] **Step 5: 提交独立变更**

```bash
git add drl/evolve_engine.py core/database.py web/api/evolve.py web/static/index.html tests/test_evolve_api_guards.py tests/test_evolve_audit.py
git commit -m "feat: expose evolve deployment and rollback audit state"
```

### Task 6: 全链路验证与文档同步

**Files:**
- Modify: `docs/REDEV_GUIDE.md` only to add the accepted rollback-baseline invariants and verification commands.
- Test: existing full test suite and targeted test files from Tasks 1-5.

**Interfaces:**
- No new runtime interface; this task verifies and documents the interfaces from Tasks 1-5.

- [ ] **Step 1: 运行编译检查**

Run:

```bat
.venv\Scripts\python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py
```

Expected: exit code 0.

- [ ] **Step 2: 运行持续进化定向测试**

Run:

```bat
.venv\Scripts\python -m pytest -q tests/test_evolve_anchor.py tests/test_evolve_oos_gate.py tests/test_evolve_api_guards.py tests/test_evolve_deployment.py tests/test_evolve_runtime_reload.py tests/test_evolve_identity.py tests/test_evolve_audit.py
```

Expected: all targeted tests pass, including legacy tests in the dirty worktree.

- [ ] **Step 3: 运行全量测试和前端检查**

Run:

```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/check_js.py
```

Expected: all tests pass and JavaScript syntax check passes. If the count differs from the 2026-09-01 baseline, report the exact count and explain only changes caused by this feature.

- [ ] **Step 4: 运行必要冒烟与回测基线**

Run:

```bat
.venv\Scripts\python run.py backtest --source demo --strategy dual_ma
```

Expected: `ret=-0.206239`, `trades=37`, `final_equity=7937.6077` unless an explicitly intended unrelated worktree change already changes the baseline; in that case record the observed value and do not revert user changes.

For the web process, start on a free local port and verify:

- `GET /api/health` returns 200 with `degraded` and `bus` keys;
- unauthenticated protected API remains 401;
- `/api/evolve/status` contains candidate/deployed outcome fields;
- manual rollback of a fixture version returns `runtime_reload` and creates `manual_rollback` audit state;
- restarting restores `rl_evolve` registration and does not erase deployment metadata.

- [ ] **Step 5: 自审文档并提交验证记录**

Check `docs/REDEV_GUIDE.md` for the new rule: a rollback is successful only when ModelZoo snapshot, flat file, dynamic registration and active runtime agree. Add the targeted test names and the required compile/full-test/frontend/backtest commands. Run `git diff --check` on all touched files.

```bash
git add docs/REDEV_GUIDE.md
git commit -m "docs: record evolve rollback baseline invariants"
```

## Completion Gate

Before claiming completion, run the verification sequence from Task 6 and inspect `git diff --stat` plus `git status --short`. Do not revert or stage unrelated pre-existing worktree changes. Report any failed command with its actual output and identify whether it belongs to this feature or the pre-existing dirty worktree.
