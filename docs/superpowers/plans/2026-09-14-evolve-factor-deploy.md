# 进化因子自动部署为组合因子策略 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 持续进化引擎 factor_miner 训练通过 OOS 安检后，把组合因子权重自动部署为按标的的 factor_signal 组合因子策略（可回测、可实盘手动启用），并在进化面板展示部署状态与回测摘要。

**Architecture:** 在 `EvolveEngine` 新增部署方法（构造 factor_signal spec → `register_dynamic` + `_upsert_ai_strategy` 落库），`_train_factor_miner_once` 安检通过分支末尾触发自动部署并 fire-and-forget 跑后台回测摘要；`web/api/evolve.py` 暴露列表与手动重新部署端点；前端进化面板 factor_miner 卡片加部署状态区与按钮。部署不阻塞训练循环，失败只记 `deploy_error`。

**Tech Stack:** Python 3.11+（asyncio / pandas / numpy）、FastAPI、SQLite（AiStrategy 表）、Vue3 单页（web/static/index.html，无构建步骤）、pytest（asyncio_mode=auto）。

## Global Constraints

- 策略命名：`evolve_combo_<symbol>`，symbol 的 `/` 替换为 `_`（如 `evolve_combo_BTC_USDT`）。
- `combo_spec` 用 `json.dumps(weights, ensure_ascii=False)`，`executor="factor_signal"`，`created_by="evolve_engine"`。
- 自动部署**不得**获取 `self._train_lock`（`_train_factor_miner_once` 调用时锁已被持有，asyncio.Lock 不可重入，会死锁）；手动部署 API 端点可持锁。
- 回测摘要只写 `_factor_miner_status`（内存），不落 `backtest_results` 表、不重写 AiStrategy 行。
- 部署失败（权重缺失/异常）只记 `deploy_error`，不阻断训练流程、不抛异常。
- 与手动 RL 挖掘路径（`applyRlCombo` → 策略名 `factor_signal`）并存，命名空间互不冲突。
- 测试沿用 `tests/test_drl_optimizations.py::_MetaDB` 与 `generate_demo` 构造离线引擎；`pytest.ini` 已开 `asyncio_mode=auto`，async 测试直接写 `async def`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `drl/evolve_engine.py`（改） | 新增 `_combo_strategy_name` / `_deploy_factor_strategy` / `_refresh_factor_strategy_backtest`；`__init__` 增 `_backtest_tasks` 与 `_factor_miner_status` 5 个新键；`_train_factor_miner_once` 尾部插入自动部署钩子；import 增加 `get_dynamic` |
| `tests/test_evolve_factor_deploy.py`（新建） | 部署方法单测 + 回测摘要 + 自动部署集成测试 |
| `web/api/evolve.py`（改） | 新增 `GET /factor-strategies` 与 `POST /factor/deploy`（symbol 走 query 参数，避免路径段斜杠问题） |
| `tests/test_evolve_factor_deploy_api.py`（新建） | API 端点测试（FastAPI TestClient） |
| `web/static/index.html`（改） | 进化面板 factor_miner 卡片部署状态区 + 两个按钮 + JS 函数 |

---

### Task 1: 部署方法 `_combo_strategy_name` + `_deploy_factor_strategy`

**Files:**
- Modify: `drl/evolve_engine.py:39`（import 行改为 `from strategies import get_dynamic, register_dynamic`）
- Modify: `drl/evolve_engine.py`（新增两个方法，放在 `_register_rl_evolve` 方法之后、`_restore_evolve_strategies` 之前，约 L468 与 L470 之间）
- Create: `tests/test_evolve_factor_deploy.py`

**Interfaces:**
- Produces:
  - `EvolveEngine._combo_strategy_name(self, symbol: str) -> str`
  - `EvolveEngine._deploy_factor_strategy(self, symbol: str, weights: dict, report: Optional[dict], meta: Optional[dict] = None) -> dict`，返回 `{"name": str, "version": str, "spec": dict}`
- Consumes: `strategies.register_dynamic`（已 import）、`strategies.get_dynamic`（本任务新增 import）、`EvolveEngine._upsert_ai_strategy(name, spec)`（已存在，L415）、`FactorSignalStrategy.param_schema`（本方法内局部 import）、`self._effective_timeframe()`（已存在）、`self.db`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_evolve_factor_deploy.py`：

```python
"""进化因子自动部署为组合因子策略——部署方法单测（Task 1/3）。"""
import json

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from strategies import get_dynamic, get_strategy, remove_dynamic
from strategies.factor_signal import FactorSignalStrategy
from tests.test_drl_optimizations import _MetaDB


def _engine(tmp_path):
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


def _cleanup(*names):
    for n in names:
        remove_dynamic(n)


def test_name_flattens_symbol(tmp_path):
    eng = _engine(tmp_path)
    assert eng._combo_strategy_name("BTC/USDT") == "evolve_combo_BTC_USDT"
    assert eng._combo_strategy_name("SOL/USDT") == "evolve_combo_SOL_USDT"


async def test_deploy_creates_factor_signal_spec(tmp_path):
    eng = _engine(tmp_path)
    report = {"enabled": True, "valid": True, "rank_ic": 0.04, "icir": 1.2}
    out = await eng._deploy_factor_strategy(
        "BTC/USDT", {"vol_ratio": 0.5, "rsi_osc": -0.3}, report,
        {"fitness": 0.031, "selected_factors": ["vol_ratio", "rsi_osc"],
         "round_no": 7})
    try:
        assert out["name"] == "evolve_combo_BTC_USDT"
        assert out["version"] == "v1"
        spec = get_dynamic("evolve_combo_BTC_USDT")
        assert spec["executor"] == "factor_signal"
        assert spec["created_by"] == "evolve_engine"
        assert spec["params"]["factor"] == "combo"
        assert json.loads(spec["params"]["combo_spec"]) == {"vol_ratio": 0.5, "rsi_osc": -0.3}
        assert spec["params"]["mode"] == "trend"
        assert spec["evolve_meta"]["oos_report"] == report
        assert spec["evolve_meta"]["fitness"] == 0.031
        assert spec["evolve_meta"]["selected_factors"] == ["vol_ratio", "rsi_osc"]
        assert spec["base_symbol"] == "BTC/USDT"
    finally:
        _cleanup("evolve_combo_BTC_USDT")


async def test_deploy_overwrites_and_increments_version(tmp_path):
    eng = _engine(tmp_path)
    await eng._deploy_factor_strategy("ETH/USDT", {"vol_ratio": 1.0}, None, {})
    try:
        out2 = await eng._deploy_factor_strategy("ETH/USDT", {"vol_ratio": 2.0}, None, {})
        assert out2["version"] == "v2"
        spec = get_dynamic("evolve_combo_ETH_USDT")
        assert json.loads(spec["params"]["combo_spec"]) == {"vol_ratio": 2.0}
    finally:
        _cleanup("evolve_combo_ETH_USDT")


async def test_deployed_strategy_instantiates_as_factor_signal(tmp_path):
    eng = _engine(tmp_path)
    await eng._deploy_factor_strategy("BTC/USDT", {"vol_ratio": 0.5}, None, {})
    try:
        st = get_strategy("evolve_combo_BTC_USDT")
        assert isinstance(st, FactorSignalStrategy)
        assert st.params["factor"] == "combo"
        assert st.params["combo_spec"] == '{"vol_ratio": 0.5}'
    finally:
        _cleanup("evolve_combo_BTC_USDT")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evolve_factor_deploy.py -v`
Expected: FAIL（`AttributeError: 'EvolveEngine' object has no attribute '_combo_strategy_name'`）

- [ ] **Step 3: 实现**

修改 `drl/evolve_engine.py` L39：

```python
from strategies import get_dynamic, register_dynamic
```

在 `_register_rl_evolve` 方法（以 L468 的 `log.info("[evolve] 策略 %s 已注册...` 结尾）之后、`_restore_evolve_strategies`（L470）之前，插入：

```python
    # ---- 进化组合因子策略部署（Task：进化因子使用场景）----

    def _combo_strategy_name(self, symbol: str) -> str:
        """进化组合因子策略名：evolve_combo_<symbol>（/ 替换为 _，与级联文件命名一致）。"""
        return f"evolve_combo_{symbol.replace('/', '_')}"

    async def _deploy_factor_strategy(self, symbol: str, weights: dict,
                                      report: Optional[dict],
                                      meta: Optional[dict] = None) -> dict:
        """把进化产出的组合因子权重部署为 factor_signal 组合策略（幂等覆盖）。

        spec 落 AiStrategy 表（重启后由 dynamic_store.restore_from_db 恢复）；
        同标的重复部署 = 覆盖更新同名 spec，version 递增。返回 {name, version, spec}。
        """
        from strategies.factor_signal import FactorSignalStrategy
        name = self._combo_strategy_name(symbol)
        prev = get_dynamic(name)
        prev_version = 0
        if prev and prev.get("version"):
            try:
                prev_version = int(str(prev["version"]).lstrip("v"))
            except ValueError:
                prev_version = 0
        version = f"v{prev_version + 1}"
        spec = {
            "name": name,
            "title": f"进化组合因子·{symbol}",
            "description": f"持续进化引擎因子挖掘（factor_miner）产出的组合因子策略（{symbol}）",
            "logic": "RL因子挖掘选出因子组合，按IC方向加权合成，z-score后与阈值比较产生买卖信号",
            "executor": "factor_signal",
            "param_schema": FactorSignalStrategy.param_schema,
            "params": {
                "factor": "combo",
                "combo_spec": json.dumps(weights, ensure_ascii=False),
                "mode": "trend",
                "buy_threshold": 0.0,
                "sell_threshold": 0.0,
            },
            "risk_tips": ["因子IC可能衰变，因子库下线后该因子自动跳过"],
            "created_by": "evolve_engine",
            "version": version,
            "base_symbol": symbol,
            "base_timeframe": self._effective_timeframe(),
            "evolve_meta": {
                "fitness": (meta or {}).get("fitness"),
                "oos_report": report,
                "selected_factors": (meta or {}).get("selected_factors", []),
                "deployed_at": time.time(),
                "round_no": (meta or {}).get("round_no"),
                "backtest": None,
            },
        }
        register_dynamic(name, spec)
        await self._upsert_ai_strategy(name, spec)
        log.info("[evolve] 进化组合因子策略 %s v%s 已部署（%d 个因子）",
                 name, version, len(weights))
        return {"name": name, "version": version, "spec": spec}
```

（`json` 与 `time` 已在本文件顶部 import。）

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evolve_factor_deploy.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add drl/evolve_engine.py tests/test_evolve_factor_deploy.py
git commit -m "feat: 进化组合因子策略部署方法（按标的注册 factor_signal 组合策略，幂等覆盖）"
```

---

### Task 2: 后台回测摘要 `_refresh_factor_strategy_backtest`

**Files:**
- Modify: `drl/evolve_engine.py`（新增方法，放在 `_deploy_factor_strategy` 之后）
- Modify: `tests/test_evolve_factor_deploy.py`（追加测试）

**Interfaces:**
- Produces: `EvolveEngine._refresh_factor_strategy_backtest(self, name: str, symbol: str, df) -> None`（async，异常自吞，写 `self._factor_miner_status["backtest_summary"]`）
- Consumes: `backtest.engine.run_backtest` / `backtest.engine.BacktestConfig`（方法内局部 import）、`strategies.get_dynamic`、`self._effective_timeframe()`、`self._factor_miner_status`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_evolve_factor_deploy.py`：

```python
def test_backtest_summary_written_to_status(tmp_path, monkeypatch):
    import backtest.engine as be
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    df = generate_demo(timeframe="1h", n=400, seed=42)
    captured = {}

    def _fake_run(df_, cfg, **kw):
        captured["cfg"] = cfg
        return {"metrics": {"total_return": 0.12, "max_drawdown": 0.05,
                            "sharpe": 1.1, "total_trades": 7},
                "benchmark": {"buy_hold_ret": 0.03}}
    monkeypatch.setattr(be, "run_backtest", _fake_run)

    async def _go():
        await eng._refresh_factor_strategy_backtest(
            "evolve_combo_BTC_USDT", "BTC/USDT", df)

    asyncio.run(_go())
    summary = eng._factor_miner_status["backtest_summary"]
    assert summary["total_ret"] == 0.12
    assert summary["max_drawdown"] == 0.05
    assert summary["sharpe"] == 1.1
    assert summary["trades"] == 7
    assert summary["benchmark_ret"] == 0.03
    assert captured["cfg"].strategy_name == "evolve_combo_BTC_USDT"
    assert eng._factor_miner_status["deploy_error"] == ""
```

文件顶部补充 import：`import asyncio`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evolve_factor_deploy.py::test_backtest_summary_written_to_status -v`
Expected: FAIL（`AttributeError: 'EvolveEngine' object has no attribute '_refresh_factor_strategy_backtest'`）

- [ ] **Step 3: 实现**

在 `_deploy_factor_strategy` 方法之后插入：

```python
    async def _refresh_factor_strategy_backtest(self, name: str, symbol: str,
                                                df) -> None:
        """后台跑一次部署策略的回测，把摘要写回 factor_miner 状态（仅展示用）。

        用训练轮同一份 df（rolling_window 根K线）+ 部署策略参数回测；
        异常自吞只记 deploy_error，不影响部署状态与训练流程。
        """
        try:
            from backtest.engine import BacktestConfig, run_backtest
            spec = get_dynamic(name) or {}
            cfg = BacktestConfig(
                symbol=symbol,
                timeframe=self._effective_timeframe(),
                strategy_name=name,
                strategy_params=dict(spec.get("params", {})),
                start_cash=10000.0, fee_rate=0.001, slippage=0.0005,
            )
            res = await asyncio.to_thread(run_backtest, df, cfg)
            m = res.get("metrics") or {}
            summary = {
                "total_ret": float(m.get("total_return", 0.0)),
                "max_drawdown": float(m.get("max_drawdown", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "trades": int(m.get("total_trades", 0)),
                "benchmark_ret": float((res.get("benchmark") or {}).get("buy_hold_ret", 0.0)),
                "at": time.time(),
            }
            self._factor_miner_status["backtest_summary"] = summary
            self._factor_miner_status["deploy_error"] = ""
            log.info("[evolve] %s 回测摘要已刷新: %s", name, summary)
        except Exception as e:  # noqa: BLE001
            self._factor_miner_status["deploy_error"] = f"回测摘要失败: {e}"
            log.warning("[evolve] %s 回测摘要失败: %s", name, e)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evolve_factor_deploy.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add drl/evolve_engine.py tests/test_evolve_factor_deploy.py
git commit -m "feat: 进化组合因子策略后台回测摘要（写回 factor_miner 状态，仅展示）"
```

---

### Task 3: 自动部署钩子（`_train_factor_miner_once` 安检通过分支）

**Files:**
- Modify: `drl/evolve_engine.py:279` 附近（`__init__`：`_train_lock` 之后加 `_backtest_tasks`）
- Modify: `drl/evolve_engine.py:289-292`（`_factor_miner_status` 初始字典加 5 个键）
- Modify: `drl/evolve_engine.py:1733` 之后（`_train_factor_miner_once` 末尾，第 8 步 publish 之后）
- Modify: `tests/test_evolve_factor_deploy.py`（追加集成测试）

**Interfaces:**
- Consumes: `_deploy_factor_strategy`（Task 1）、`_refresh_factor_strategy_backtest`（Task 2）、`result`（`weights`/`report`/`selected_factors`）、`symbol`、`df`、`new_fitness`、`self._factor_miner_status`
- Produces: 状态键 `deployed_strategy` / `deployed_version` / `last_deploy_ts` / `deploy_error` / `backtest_summary`（经 `_pipeline_view` 自动透传到 `status()["factor_miner"]`）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_evolve_factor_deploy.py`：

```python
class _StubAgent:
    def to_dict(self) -> dict:
        return {"stub": True}


def _factor_result(weights=None, valid=True):
    return {
        "agent": _StubAgent(),
        "history": [{"best_fitness": 0.02}],
        "selected_factors": list((weights or {"vol_ratio": 0.5}).keys()),
        "weights": weights or {"vol_ratio": 0.5},
        "composite": None,   # 跳过级联 npy 保存，聚焦部署断言
        "report": {"enabled": True, "valid": valid, "rank_ic": 0.04,
                   "icir": 1.2, "reason": ""},
        "meta": {},
    }


async def test_passing_factor_miner_auto_deploys(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    from strategies import get_dynamic
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    monkeypatch.setattr(eng, "_fetch_latest_data", lambda symbol="": df)
    holder = {"result": _factor_result()}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    saved = []
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: saved.append(name))
    calls = []
    async def _stub_refresh(name, symbol, df_):
        calls.append((name, symbol))
    monkeypatch.setattr(eng, "_refresh_factor_strategy_backtest", _stub_refresh)

    try:
        await eng._train_factor_miner_once(force=True)

        assert saved == ["factor_miner"]
        assert eng._factor_miner_status["deployed_strategy"] == "evolve_combo_BTC_USDT"
        assert eng._factor_miner_status["deployed_version"] == "v1"
        assert eng._factor_miner_status["deploy_error"] == ""
        spec = get_dynamic("evolve_combo_BTC_USDT")
        assert spec is not None and spec["executor"] == "factor_signal"
        assert calls == [("evolve_combo_BTC_USDT", "BTC/USDT")]
        # 状态键经 _pipeline_view 透传（前端消费入口）
        view = eng.status()["factor_miner"]
        assert view["deployed_strategy"] == "evolve_combo_BTC_USDT"
    finally:
        from strategies import remove_dynamic
        remove_dynamic("evolve_combo_BTC_USDT")


async def test_rejected_factor_miner_does_not_deploy(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    from strategies import get_dynamic
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    monkeypatch.setattr(eng, "_fetch_latest_data", lambda symbol="": df)
    holder = {"result": _factor_result(valid=False)}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: None)

    await eng._train_factor_miner_once(force=True)

    assert eng._factor_miner_status["deployed_strategy"] == ""
    assert get_dynamic("evolve_combo_BTC_USDT") is None
```

注意：`_next_symbol()` 依赖 `self._symbols`（默认含 BTC/USDT）与 `_symbol_idx=0`，首轮为 `BTC/USDT`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evolve_factor_deploy.py::test_passing_factor_miner_auto_deploys tests/test_evolve_factor_deploy.py::test_rejected_factor_miner_does_not_deploy -v`
Expected: 前者 FAIL（`deployed_strategy` 为空）、后者 PASS（未实现自动部署时被拦轮本就不部署）

- [ ] **Step 3: 实现**

① `__init__` 中 `self._train_lock = asyncio.Lock()`（L279）之后插入：

```python
        # 部署后后台回测摘要任务引用（防 GC；done 回调里弹出）
        self._backtest_tasks: dict[str, asyncio.Task] = {}
```

② `_factor_miner_status` 初始字典（L289-292）改为：

```python
        self._factor_miner_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                     "last_error": "", "next_run": 0, "last_success": False,
                                     "oos_rejected": "", "_demo_streak": 0,
                                     "loop_beat": 0.0, "restarts": 0,
                                     # 进化组合因子部署状态（Task：进化因子使用场景）
                                     "deployed_strategy": "", "deployed_version": "",
                                     "last_deploy_ts": 0, "deploy_error": "",
                                     "backtest_summary": None}
```

③ `_train_factor_miner_once` 末尾（第 8 步 publish 的 `}, source="evolve_engine"))` 之后，方法结束前）插入：

```python
        # 9. 自动部署：OOS 安检已过（gate_ok），把组合权重部署为按标的的组合因子策略。
        # 部署失败只记 deploy_error，不阻断训练流程（训练主体已完成并落库）。
        try:
            _weights = result.get("weights") or {}
            if _weights:
                _meta = {
                    "fitness": float(new_fitness),
                    "selected_factors": result.get("selected_factors", []),
                    "round_no": self._factor_miner_status.get("episode", 0),
                }
                _dep = await self._deploy_factor_strategy(
                    symbol, _weights, result.get("report"), _meta)
                self._factor_miner_status["deployed_strategy"] = _dep["name"]
                self._factor_miner_status["deployed_version"] = _dep["version"]
                self._factor_miner_status["last_deploy_ts"] = time.time()
                self._factor_miner_status["deploy_error"] = ""
                # 后台回测摘要（fire-and-forget，不阻塞训练循环；持有任务引用防 GC）
                _task = asyncio.create_task(
                    self._refresh_factor_strategy_backtest(_dep["name"], symbol, df))
                self._backtest_tasks[_dep["name"]] = _task
                _task.add_done_callback(
                    lambda t, n=_dep["name"]: self._backtest_tasks.pop(n, None))
        except Exception as e:  # noqa: BLE001
            self._factor_miner_status["deploy_error"] = f"自动部署失败: {e}"
            log.warning("[evolve] 组合因子策略自动部署失败(%s): %s", symbol, e)
```

（`gate_ok` 分支已保证走到此处时 report 通过安检；`df`、`symbol`、`new_fitness` 均为本方法作用域内变量。）

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evolve_factor_deploy.py -v`
Expected: PASS（7 passed；若因 `_next_symbol` 环境差异 symbol 不是 BTC/USDT，断言改为动态取 `eng._factor_miner_status["symbol"]`）

- [ ] **Step 5: 提交**

```bash
git add drl/evolve_engine.py tests/test_evolve_factor_deploy.py
git commit -m "feat: 进化因子 OOS 安检通过后自动部署为组合因子策略（含后台回测摘要）"
```

---

### Task 4: API 端点（列表 + 手动重新部署）

**Files:**
- Modify: `web/api/evolve.py`（文件末尾追加两个端点）
- Create: `tests/test_evolve_factor_deploy_api.py`

**Interfaces:**
- Produces:
  - `GET /api/evolve/factor-strategies` → `{"strategies": [{name, symbol, timeframe, version, weights, oos_report, selected_factors, fitness, deployed_at, backtest}], "count": int}`
  - `POST /api/evolve/factor/deploy?symbol=BTC%2FUSDT` → `{"ok": True, "name", "version", "spec"}`；权重文件缺失 404，非法 symbol 400，空权重 400
- Consumes: `EvolveEngine._deploy_factor_strategy`、`_combo_strategy_name`、`self.zoo.models_dir`、`strategies.dynamic_names` / `strategies.get_dynamic`、`_SYMBOL_RE`（web/api/evolve.py 已有）、`evolve._factor_miner_status`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_evolve_factor_deploy_api.py`：

```python
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
    from web.api.evolve import router
    app = FastAPI()
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    app.state.engine = _EngineHolder(eng)
    app.include_router(router)
    return TestClient(app)


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evolve_factor_deploy_api.py -v`
Expected: FAIL（404/405：`POST /api/evolve/factor/deploy` 不存在）

- [ ] **Step 3: 实现**

在 `web/api/evolve.py` 文件末尾（`reset_anchor` 之后）追加：

```python
@router.get("/factor-strategies")
async def factor_strategies(request: Request):
    """列出全部已部署的进化组合因子策略（权重/OOS报告/版本/回测摘要）。

    实现：扫描 strategies 动态注册表，过滤 created_by=evolve_engine 且
    name 前缀 evolve_combo_；回测摘要优先取该策略最近一次部署（内存 status）。
    """
    from strategies import dynamic_names, get_dynamic
    evolve = _get_evolve(request)
    fm = evolve._factor_miner_status
    latest_name = fm.get("deployed_strategy") or ""
    latest_bt = fm.get("backtest_summary")
    out = []
    for name in sorted(dynamic_names()):
        if not name.startswith("evolve_combo_"):
            continue
        spec = get_dynamic(name) or {}
        if spec.get("created_by") != "evolve_engine":
            continue
        params = spec.get("params") or {}
        combo_spec = params.get("combo_spec", "")
        weights = {}
        if combo_spec:
            try:
                weights = json.loads(combo_spec)
            except Exception:  # noqa: BLE001
                weights = {}
        em = spec.get("evolve_meta") or {}
        out.append({
            "name": name,
            "symbol": spec.get("base_symbol", ""),
            "timeframe": spec.get("base_timeframe", ""),
            "version": spec.get("version", ""),
            "weights": weights,
            "oos_report": em.get("oos_report"),
            "selected_factors": em.get("selected_factors", []),
            "fitness": em.get("fitness"),
            "deployed_at": em.get("deployed_at"),
            "backtest": latest_bt if name == latest_name else None,
        })
    return {"strategies": out, "count": len(out)}


@router.post("/factor/deploy")
async def deploy_factor(symbol: str = Query(..., description="交易对，如 BTC/USDT"),
                        request: Request = None):
    """手动重新部署进化组合因子策略（读最新 _cascade_weights_<symbol>.json）。

    symbol 走 query 参数（路径段含 / 无法路由）；与训练写路径互斥持 _train_lock。
    """
    if not _SYMBOL_RE.fullmatch(str(symbol or "")):
        raise HTTPException(status_code=400, detail=f"非法交易对: {symbol!r}")
    evolve = _get_evolve(request)
    wpath = evolve.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
    if not wpath.exists():
        raise HTTPException(status_code=404,
                            detail=f"{symbol} 暂无进化因子权重（先训练 factor_miner）")
    try:
        weights = json.loads(wpath.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"权重文件解析失败: {e}")
    if not isinstance(weights, dict) or not weights:
        raise HTTPException(status_code=400, detail="权重文件为空")
    meta = {
        "fitness": float(evolve._factor_miner_status.get("fitness") or 0.0),
        "selected_factors": list(weights.keys()),
        "round_no": int(evolve._factor_miner_status.get("episode", 0)),
    }
    async with evolve._train_lock:
        result = await evolve._deploy_factor_strategy(symbol, weights, None, meta)
    return {"ok": True, **result}
```

（`Query` 需从 fastapi 导入——`web/api/evolve.py` L6 已有 `from fastapi import APIRouter, HTTPException, Query, Request`，无需改动。）

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evolve_factor_deploy_api.py tests/test_evolve_factor_deploy.py -v`
Expected: PASS（全部通过）

- [ ] **Step 5: 提交**

```bash
git add web/api/evolve.py tests/test_evolve_factor_deploy_api.py
git commit -m "feat: 进化因子部署 API（策略列表 + 手动重新部署，symbol 走 query 参数）"
```

---

### Task 5: 前端进化面板部署状态区

**Files:**
- Modify: `web/static/index.html`

**Interfaces:**
- Consumes: `status()["factor_miner"]` 新键（`deployed_strategy`/`deployed_version`/`last_deploy_ts`/`deploy_error`/`backtest_summary`）、`GET /api/evolve/factor-strategies`、`POST /api/evolve/factor/deploy?symbol=...`、既有 `api()` 助手、`switchView('backtest')` + `nextTick(()=>{ runBacktest(); })` 跳转模式（参考既有 `runFactorBacktest`）、`fmtTs()`、`toast()`、`bt` reactive（`bt.strategy_name`）

- [ ] **Step 1: 加 JS 函数与数据**

在 `web/static/index.html` 的进化引擎 JS 区（`fetchEvolveStatus` 定义附近）追加：

```js
    // ---- 进化组合因子部署（Task 5）----
    const evolveComboStrategies = ref([]);
    async function loadEvolveFactorStrategies() {
      try { const d = await api('/evolve/factor-strategies'); evolveComboStrategies.value = d.strategies||[]; } catch(e){}
    }
    async function deployEvolveFactor(symbol) {
      if (!symbol) { toast('暂无训练标的，无法部署','warn'); return; }
      try {
        const d = await api('/evolve/factor/deploy?symbol='+encodeURIComponent(symbol), {method:'POST', body:'{}'});
        toast('已部署进化组合因子策略：'+d.name+' '+d.version,'ok');
        fetchEvolveStatus(); loadEvolveFactorStrategies();
      } catch(e){ toast(String(e.message||e),'error'); }
    }
    function goEvolveComboBacktest(name) {
      bt.strategy_name = name;
      bt.strategy_params = {};
      switchView('backtest');
      nextTick(()=>{ runBacktest(); });
    }
```

在页面初始化/切到进化页时加载列表：找到 `fetchEvolveStatus();` 首次出现处（evolve 视图 setup 块），在其后追加 `loadEvolveFactorStrategies();`（同一处也 `fetchEvolveStatus()` 与 `fetchEvolveConfig()` 并列）。

- [ ] **Step 2: 加卡片部署状态区**

在进化面板模型卡片模板（`v-for="m in EVOLVE_MODELS"` 的 `mac-card` 内）的「策略 DRL 独有」区块（`v-if="evolveModel(m.key).candidate_fitness!=null || ..."`）**之后**、卡片闭合 `</div>` 之前，插入（仅 factor_miner 卡片渲染）：

```html
            <!-- 因子挖掘独有：进化组合因子部署状态（Task 5） -->
            <div v-if="m.key==='factor_miner'" class="mt-1 grid gap-1 text-[12px] text-slate-400">
              <div v-if="evolveModel(m.key).deployed_strategy" class="flex items-center gap-2 flex-wrap">
                <span class="pill" style="background:rgba(48,209,88,.15);color:#30d158">已部署</span>
                <span class="font-mono">{{evolveModel(m.key).deployed_strategy}} {{evolveModel(m.key).deployed_version}}</span>
                <span v-if="evolveModel(m.key).last_deploy_ts">{{fmtTs(evolveModel(m.key).last_deploy_ts*1000)}}</span>
              </div>
              <div v-else-if="evolveModel(m.key).deploy_error" class="evolve-model-warning">
                <svg width="12" height="12" style="flex:none;margin-top:2px"><use href="#i-warn"/></svg>
                <span>{{evolveModel(m.key).deploy_error}}</span>
              </div>
              <div v-else class="text-slate-500">尚未部署 · OOS 安检通过后自动部署</div>
              <div v-if="evolveModel(m.key).backtest_summary" class="flex items-center gap-x-3 gap-y-0.5 flex-wrap">
                <span>回测收益 <span class="font-mono" :class="(evolveModel(m.key).backtest_summary.total_ret||0)>=0?'pos':'neg'">{{((evolveModel(m.key).backtest_summary.total_ret||0)*100).toFixed(2)}}%</span></span>
                <span>回撤 <span class="font-mono">{{((evolveModel(m.key).backtest_summary.max_drawdown||0)*100).toFixed(2)}}%</span></span>
                <span>夏普 <span class="font-mono">{{(evolveModel(m.key).backtest_summary.sharpe||0).toFixed(2)}}</span></span>
                <span>交易 {{evolveModel(m.key).backtest_summary.trades||0}} 笔</span>
              </div>
              <div class="flex items-center gap-2 mt-1">
                <button class="mac-btn text-[12px]" style="padding:3px 10px"
                        @click="deployEvolveFactor(evolveModel(m.key).symbol)"
                        :disabled="!evolveModel(m.key).symbol">部署为因子策略</button>
                <button v-if="evolveModel(m.key).deployed_strategy" class="mac-btn text-[12px]" style="padding:3px 10px"
                        @click="goEvolveComboBacktest(evolveModel(m.key).deployed_strategy)">去回测</button>
              </div>
            </div>
```

先确认图标 `#i-warn` 在页面 sprite 中存在；若不存在，把 `<svg ...><use href="#i-warn"/></svg>` 换成文本 `⚠`，避免破图。

- [ ] **Step 3: 手动验证**

启动应用（`python run.py` 或既有启动方式），打开进化面板：

1. factor_miner 卡片出现「尚未部署 · OOS 安检通过后自动部署」；训练通过一轮后变为「已部署 evolve_combo_BTC_USDT vN + 时间戳」，随后出现回测摘要（收益/回撤/夏普/交易数）；
2. 点「去回测」跳转回测页且 strategy 下拉选中该策略并自动开跑；
3. 点「部署为因子策略」toast 提示部署成功，列表刷新；
4. 手动挖矿（/api/drl/mine-factor）路径的「应用 RL 组合」按钮仍可用，互不覆盖。

- [ ] **Step 4: 提交**

```bash
git add web/static/index.html
git commit -m "feat: 进化面板因子部署状态区（部署状态/回测摘要/一键部署/去回测）"
```

---

## Self-Review（实施前自查）

- **Spec 覆盖**：部署方法（§4.1→Task1）、自动钩子（§4.2→Task3）、回测摘要（§4.3→Task2）、API（§4.4→Task4）、UI（§4.5→Task5）、错误处理（§5→各任务异常自吞+deploy_error）、测试（§6→各任务含失败-通过测试）全部有对应任务。
- **占位符**：无 TBD/TODO；所有代码块为可直接落地的完整实现。
- **类型一致性**：`_deploy_factor_strategy` 返回 `{name, version, spec}`（Task1 定义，Task3/4 消费）；状态键 `deployed_strategy/deployed_version/last_deploy_ts/deploy_error/backtest_summary` 在 Task3 定义并在 Task5 消费；回测摘要字段 `total_ret/max_drawdown/sharpe/trades/benchmark_ret` 在 Task2 写入、Task5 展示；API 路径与前端调用一致（`/evolve/factor-strategies`、`/evolve/factor/deploy?symbol=`）。
- **已知环境差异**：Task3 集成测试的 symbol 依赖 `_next_symbol()`（默认首轮 BTC/USDT）；若测试环境 settings 改了默认 symbols，断言改从 `eng._factor_miner_status["symbol"]` 动态取。
