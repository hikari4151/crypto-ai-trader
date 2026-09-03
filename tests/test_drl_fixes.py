"""模块 B（DRL）修复的 TDD 测试：T1 熵符号 / T2 reset 塑形 / T3 z_mode 前视 /
T4 评估端点维度重建 / T5 val_skipped / T6 跨 loop DB / T7 worker finally close_loop。

覆盖 REFACTOR_PLAN P0-4 / P0-5 / P1-8 / P1-9 / P1-10 / P1-1(drl) / P1-4 配套，
按 docs/plans/module_B.md 实施。
"""
import asyncio
import json
import threading
import time
import types

import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import generate_demo


# ============ T1. P0-4 PPO 熵正则梯度符号 ============

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def test_entropy_sign_maximizes_entropy():
    """单步零优势批量：actor 更新只剩熵正则项。

    网络是梯度下降（apply_grad 做 W -= lr*g），修复后 d_entropy=+p*(H+log p)
    等价于对熵做梯度上升 → 训练后策略熵上升（探索被鼓励）。
    修复前为负号 → 熵下降（压缩单峰）→ 本用例失败。
    """
    from drl.agent import ACAgent
    agent = ACAgent(state_dim=6, n_actions=3, hidden=(16, 16), seed=42,
                    entropy_coef=2.0, lr_actor=1e-3, lr_critic=0.01)
    # 注：lr 必须小到一阶项主导——Adam 大步长（≈lr）在近均匀分布上会越过熵峰顶
    # 产生过冲（二阶项占优），小步长下 ΔH≈η·‖dH/dz‖² 严格为正
    state = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], dtype=float)
    proba0 = agent.actor.predict_proba(state.reshape(1, -1))[0]
    h0 = _entropy(proba0)
    act = int(np.argmax(proba0))
    logp0 = float(np.log(proba0[act] + 1e-12))
    # 奖励全 0 → 优势恒 0 → 策略梯度项归零，纯熵正则驱动
    agent.train_batch(np.asarray([state]), np.asarray([act]),
                      np.asarray([0.0]), np.asarray([logp0]),
                      ppo_epochs=1, mini_batch_size=1)
    proba1 = agent.actor.predict_proba(state.reshape(1, -1))[0]
    h1 = _entropy(proba1)
    assert h1 > h0 + 1e-6, f"熵应上升（探索被鼓励），实际 {h0:.6f} -> {h1:.6f}"


# ============ T2. P0-5 env.reset 重置奖励塑形状态 ============

def _shape_df() -> pd.DataFrame:
    """先涨后跌的合成行情（用于确定性塑形验证）。"""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
              100.0, 99.0, 98.0, 97.0, 96.0]
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h")
    return pd.DataFrame({"close": closes, "volume": [1000.0] * len(closes)}, index=idx)


def test_reset_clears_shaping_state():
    """reset 后 _peak_equity/_last_dd/_losing_streak 必须回零（修复前遗留跨 episode）。"""
    from drl.env import TradingEnv
    env = TradingEnv(_shape_df(), start_cash=10000.0, vol_penalty=0.0,
                     reward_dd_penalty=0.02, reward_losing_penalty=0.02, warmup=1)
    env.reset()
    env.step(1)
    # 模拟上一段 episode 遗留的塑形状态
    env._peak_equity = 15000.0
    env._last_dd = 0.05
    env._losing_streak = 3
    env.reset()
    assert env._peak_equity == env.start_cash
    assert env._last_dd == 0.0
    assert env._losing_streak == 0


def test_reset_first_step_reward_isolated():
    """两个连续 episode 的首步奖励互不影响（修复前带旧峰值 → 首步多罚回撤）。"""
    from drl.env import TradingEnv
    kw = dict(start_cash=10000.0, vol_penalty=0.0, warmup=1,
              reward_dd_penalty=0.02, reward_losing_penalty=0.02)
    env = TradingEnv(_shape_df(), **kw)
    # episode A：上涨段持仓，把峰值抬到 start_cash 之上
    env.reset()
    _s, r_a1, _d, _i = env.step(1)
    env.step(1)
    env.step(1)
    assert env._peak_equity > env.start_cash
    # episode B：reset 后首步应与全新环境（同样 reset 后）首步完全一致
    env.reset()
    _s, r_b1, _d, _i = env.step(1)
    fresh = TradingEnv(_shape_df(), **kw)
    fresh.reset()  # 全新环境必须 reset 才对齐到 warmup 起点（__init__ 不设 _t）
    _s, r_f1, _d, _i = fresh.step(1)
    assert r_b1 == r_f1, f"reset 后首步奖励 {r_b1} != 全新环境 {r_f1}（塑形状态未重置）"


# ============ T3. P1-8 composite_factor z_mode（无前视） ============

def _drift_mat() -> pd.DataFrame:
    """含明显分布漂移的因子矩阵（前半均值 0，后半均值 3）。"""
    rng = np.random.default_rng(7)
    n = 100
    f1 = np.concatenate([rng.normal(0.0, 0.5, n // 2), rng.normal(3.0, 0.5, n // 2)])
    f2 = np.concatenate([rng.normal(1.0, 0.3, n // 2), rng.normal(-1.0, 0.3, n // 2)])
    return pd.DataFrame({"f1": f1, "f2": f2})


def test_composite_full_default_unchanged():
    """z_mode 默认 full：输出与手工全样本标准化逐位一致（回归锚点）。"""
    from factors.mining import composite_factor
    mat = _drift_mat()
    weights = {"f1": 0.7, "f2": -0.3}
    out = composite_factor(mat, weights, method="ic")
    # 手工复算：每列全样本 z，按权重归一化线性叠加
    z1 = (mat["f1"] - mat["f1"].mean()) / (mat["f1"].std() + 1e-12)
    z2 = (mat["f2"] - mat["f2"].mean()) / (mat["f2"].std() + 1e-12)
    total = abs(weights["f1"]) + abs(weights["f2"])
    w1, w2 = weights["f1"] / total, weights["f2"] / total
    expected = z1 * w1 + z2 * w2
    assert out.equals(expected)
    # 显式传 full 与默认一致
    assert out.equals(composite_factor(mat, weights, method="ic", z_mode="full"))


def test_composite_expanding_no_lookahead():
    """expanding 模式：只用截至当期统计量，早段与 full 不同、无 NaN。"""
    from factors.mining import composite_factor
    mat = _drift_mat()
    weights = {"f1": 0.7, "f2": -0.3}
    full = composite_factor(mat, weights, method="ic")
    exp = composite_factor(mat, weights, method="ic", z_mode="expanding")
    assert not exp.isna().any(), "expanding 输出不得含 NaN（fillna(0) 缺失）"
    assert exp.iloc[0] == 0.0, "早期样本不足处应按 0（均值）填充"
    # 漂移数据下，早段 expanding 必须与 full 明显不同（full 携带未来分布信息）
    assert (exp.iloc[:15].to_numpy() != full.iloc[:15].to_numpy()).any()


def test_factor_env_passes_expanding_z_mode(monkeypatch):
    """FactorMiningEnv._composite_fitness 合成路径必须传 z_mode='expanding'（无前视）。

    P2-12 同步：fitness 奖励路径改为传预计算 z_map（__init__ 一次性算好的
    expanding z），仍必须带 z_mode='expanding'——防全样本 z 前视的保证不变。
    """
    import drl.factor_env as fe_mod
    from drl.factor_env import FactorMiningEnv
    from factors.engine import compute_factor_matrix
    called: dict = {}
    real = fe_mod.composite_factor

    def spy(mat, weights, method="ic", z_mode="full", z_map=None):
        called["z_mode"] = z_mode
        called["z_map"] = z_map
        return real(mat, weights, method=method, z_mode=z_mode)

    monkeypatch.setattr(fe_mod, "composite_factor", spy)
    df = generate_demo(n=400)
    mat = compute_factor_matrix(df)
    env = FactorMiningEnv(mat, df["close"], max_steps=3, seed=42)
    env.reset()
    env.step(0)  # 选中一个因子 → _composite_fitness 触发合成
    assert called.get("z_mode") == "expanding", "RL 奖励路径不得使用全样本 z（前视）"
    assert called.get("z_map") is not None, "P2-12：奖励路径应复用预计算的 expanding z（z_map）"


def test_factor_miner_passes_expanding_z_mode(monkeypatch):
    """train_factor_miner 最终合成/OOS 安检同口径 expanding。"""
    import drl.factor_miner as fm_mod
    from drl.factor_miner import train_factor_miner
    called: dict = {}
    real = fm_mod.composite_factor

    def spy(mat, weights, method="ic", z_mode="full"):
        called["z_mode"] = z_mode
        return real(mat, weights, method=method, z_mode=z_mode)

    monkeypatch.setattr(fm_mod, "composite_factor", spy)
    df = generate_demo(n=400)
    cfg = {"episodes": 1, "n_episodes": 1, "ppo_epochs": 1,
           "mini_batch_size": 64, "max_steps": 2, "seed": 42}
    res = train_factor_miner(df, cfg=cfg)
    assert called.get("z_mode") == "expanding"
    assert res["selected_factors"], "应选出至少 1 个因子"


def test_factor_env_fitness_cache_consistent():
    """P2-12：fitness 缓存（键含 t 与选择集合）命中值与无缓存重算值一致，
    z_map 预计算口径与 composite_factor 现场计算逐位一致（无脏缓存）。"""
    import pytest
    from drl.factor_env import FactorMiningEnv
    from factors.analysis import factor_quality_gate
    from factors.engine import compute_factor_matrix
    from factors.mining import composite_factor

    df = generate_demo(n=400)
    mat = compute_factor_matrix(df)
    env = FactorMiningEnv(mat, df["close"], max_steps=3, seed=7)
    env.reset()
    env.step(0)
    env.step(1)
    t, sel = env._t, list(env._selected)
    got = env._composite_fitness()
    # 手工重算（不走缓存、不走 z_map）：必须与缓存值一致
    weights = {}
    for i in sel:
        ic = env._ics.iloc[t, i]
        ic = 0.0 if ic != ic else ic
        weights[env.cols[i]] = ic if abs(ic) > 1e-6 else 0.01
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")
    gate = factor_quality_gate(combo.iloc[env.n_train:env.n_val],
                               env.close.iloc[env.n_train:env.n_val], h=env.h)
    assert got == pytest.approx(gate["fitness"], abs=1e-9)
    # 清缓存重算结果不变（缓存无污染）
    env._fitness_cache.clear()
    assert env._composite_fitness() == got
    # 缓存键含 t：t 改变后用新 t 的 IC 权重独立计算，手工核对正确性（脏缓存防护）
    env._t = env._t + 1
    got2 = env._composite_fitness()
    assert (env._t, tuple(sorted(sel))) in env._fitness_cache
    weights2 = {}
    for i in sel:
        ic = env._ics.iloc[env._t, i]
        ic = 0.0 if ic != ic else ic
        weights2[env.cols[i]] = ic if abs(ic) > 1e-6 else 0.01
    combo2 = composite_factor(mat, weights2, method="ic", z_mode="expanding")
    gate2 = factor_quality_gate(combo2.iloc[env.n_train:env.n_val],
                                env.close.iloc[env.n_train:env.n_val], h=env.h)
    assert got2 == pytest.approx(gate2["fitness"], abs=1e-9)


# ============ T4. P1-9 评估维度重建 ============

def test_evaluate_agent_default_path_unchanged():
    """state_window=1 无因子模型：cfg=None 与显式默认 cfg 结果一致（回归锚点）。"""
    from drl.agent import ACAgent, evaluate_agent
    from drl.env import TradingEnv
    df = generate_demo(n=500)
    env0 = TradingEnv(df)
    agent = ACAgent(env0.state_dim, 5, seed=3)
    r1 = evaluate_agent(agent, df)
    r2 = evaluate_agent(agent, df, {"start_cash": 10000.0, "fee_rate": 0.001})
    assert r1 == r2
    assert set(r1) == {"total_ret", "final_equity", "final_position_ratio", "steps"}


def test_evaluate_agent_state_window_rebuild():
    """state_window>1 模型评估不再维度失配（修复前必炸）。"""
    from drl.agent import ACAgent, evaluate_agent
    from drl.env import TradingEnv
    df = generate_demo(n=500)
    env3 = TradingEnv(df, state_window=3)
    agent = ACAgent(env3.state_dim, 5, state_window=3, seed=3)
    # 显式传 state_window 与缺省（取 agent.state_window）两条路径都必须工作
    r1 = evaluate_agent(agent, df, {"state_window": 3})
    r2 = evaluate_agent(agent, df, None)
    assert "total_ret" in r1 and r1 == r2


def test_evaluate_agent_factor_column_rebuild():
    """含因子列模型：按 factor_expression/mu/sd 重建 extra 状态列。"""
    from drl.agent import ACAgent, evaluate_agent
    from drl.env import TradingEnv
    df = generate_demo(n=500)
    env = TradingEnv(df, extra_factors=np.zeros((len(df), 1)))
    agent = ACAgent(env.state_dim, 5, seed=3)
    cfg = {"factor_expression": "close/ma(close,20)-1",
           "factor_mu": 0.0, "factor_sd": 1.0, "state_window": 1,
           "min_trade_zone": 0.05}
    r = evaluate_agent(agent, df, cfg)
    assert "total_ret" in r


# ============ T5. P1-10 退化分支显式跳过 val ============

def test_train_drl_val_skipped_small_data():
    """数据不足（<切分下限）且带因子列：不再维度失配崩溃，显式 val_skipped。"""
    from drl.agent import train_drl
    df = generate_demo(n=120)
    cfg = {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1, "mini_batch_size": 64,
           "seed": 42, "val_eval_interval": 1,
           "factor_expression": "close/ma(close,20)-1"}
    res = train_drl(df, cfg)
    assert res.get("val_skipped") is True
    assert all(row.get("val_skipped") is True for row in res["history"])
    assert all("val_ret" not in row for row in res["history"])
    assert "agent" in res


def test_train_drl_normal_path_val_runs():
    """正常数据路径：val 评估照常进行（回归红线：不破坏 val/早停逻辑）。"""
    from drl.agent import train_drl
    df = generate_demo(n=1000)
    cfg = {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1, "mini_batch_size": 128,
           "seed": 42, "val_eval_interval": 1}
    res = train_drl(df, cfg)
    assert "val_skipped" not in res, "正常路径不得带 val_skipped 标记"
    assert any(row.get("val_ret") is not None for row in res["history"])
    assert "oos_report" in res and res["oos_report"].get("enabled") is True


# ============ T6. P1-1 跨 loop DB 持久化 ============

def test_persist_strategy_fallback_no_loop():
    """engine._loop 为 None（CLI/单测环境）：降级在当前循环直接执行，不崩溃。"""
    import web.api.drl as drlmod
    from core.database import AiStrategy, Database
    from sqlalchemy import select
    db_path = tmp_db_path("fallback")

    async def _go():
        db = Database(f"sqlite+aiosqlite:///{db_path}")
        await db.init()
        fake_engine = types.SimpleNamespace(_loop=None)
        await drlmod._persist_strategy(db, "rl_fallback",
                                       {"name": "rl_fallback", "title": "fb"}, fake_engine)
        async with db.session() as s:
            row = (await s.execute(select(AiStrategy).where(AiStrategy.name == "rl_fallback"))).scalar_one_or_none()
            return json.loads(row.spec_json)["title"] if row else None

    title = asyncio.run(_go())
    assert title == "fb"


def test_persist_strategy_schedules_to_engine_loop():
    """worker 线程 → run_coroutine_threadsafe 调度回主循环执行（连接池绑定主循环）。

    拓扑：主线程跑主循环 run_forever；测试线程（模拟 worker）调用 _persist_strategy
    调度回主循环写库；再回主循环读回验证。
    """
    import web.api.drl as drlmod
    from core.database import AiStrategy, Database
    from sqlalchemy import select
    db_path = tmp_db_path("sched")
    main = asyncio.new_event_loop()
    state: dict = {"db": None, "error": None}

    def _main_thread():
        asyncio.set_event_loop(main)
        async def _setup():
            db = Database(f"sqlite+aiosqlite:///{db_path}")
            await db.init()
            async with db.session() as s:
                s.add(AiStrategy(name="rl_sched_test", spec_json="{}"))
                await s.commit()
            state["db"] = db
        try:
            main.run_until_complete(_setup())
        except Exception as e:  # noqa: BLE001
            state["error"] = e
            return
        main.run_forever()

    t = threading.Thread(target=_main_thread, daemon=True)
    t.start()
    try:
        for _ in range(1000):
            if state["db"] is not None or state["error"]:
                break
            time.sleep(0.01)
        assert state["db"] is not None, state["error"]
        fake_engine = types.SimpleNamespace(_loop=main)
        spec = {"name": "rl_sched_test", "title": "sched", "params": {"x": 1}}
        # 测试线程即"worker 线程"（无运行中循环），asyncio.run 模拟 worker 专属循环
        asyncio.run(drlmod._persist_strategy(state["db"], "rl_sched_test", spec, fake_engine))

        async def _check():
            async with state["db"].session() as s:
                row = (await s.execute(select(AiStrategy).where(AiStrategy.name == "rl_sched_test"))).scalar_one_or_none()
                return json.loads(row.spec_json)["title"] if row else None

        fut = asyncio.run_coroutine_threadsafe(_check(), main)
        assert fut.result(timeout=10) == "sched"
    finally:
        main.call_soon_threadsafe(main.stop)
        t.join(timeout=10)


def tmp_db_path(kind: str) -> str:
    import tempfile
    from pathlib import Path
    d = Path(tempfile.gettempdir()) / "drl_persist_tests"
    d.mkdir(exist_ok=True)
    p = d / f"{kind}.db"
    if p.exists():
        p.unlink()  # 清掉上次运行的残留（防 UNIQUE 约束冲突）
    return str(p)


# ============ 端点级：T4 友好错误 / T7 worker finally close_loop ============

@pytest.fixture
def api_app():
    """仅挂载 drl 路由的最小 FastAPI 应用（evaluate 端点无 db/engine 依赖）。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.drl import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def model_cleanup():
    """清理测试期间写入 data/models 的模型文件。"""
    from web.api.drl import MODEL_DIR
    before = set(MODEL_DIR.glob("*.json")) if MODEL_DIR.exists() else set()
    yield
    for f in set(MODEL_DIR.glob("*.json")) - before:
        try:
            f.unlink()
        except OSError:
            pass


def _write_test_model(name: str, data: dict) -> None:
    from web.api.drl import MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_evaluate_endpoint_old_model_200(api_app, model_cleanup):
    """旧模型（缺 state_window/factor 键）：evaluate 不 500，走缺省兼容路径。"""
    from drl.agent import ACAgent
    agent = ACAgent(15, 5, seed=1)
    data = agent.to_dict()
    data.pop("state_window", None)  # 模拟旧格式模型
    _write_test_model("old_compat_test", data)
    r = api_app.post("/api/drl/models/old_compat_test/evaluate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    for k in ("total_ret", "final_equity", "final_position_ratio", "steps"):
        assert k in body


def test_evaluate_oos_endpoint_old_model_200(api_app, model_cleanup):
    """旧模型：evaluate_oos 同样缺省兼容（state_window=1、无因子列）。"""
    from drl.agent import ACAgent
    agent = ACAgent(15, 5, seed=1)
    data = agent.to_dict()
    data.pop("state_window", None)
    _write_test_model("old_compat_test", data)
    r = api_app.post("/api/drl/models/old_compat_test/evaluate_oos", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "total_ret" in body and "train_meta" in body


def test_evaluate_endpoint_state_window_and_factor_model_200(api_app, model_cleanup):
    """state_window=3 + 因子列模型：evaluate 按元数据重建 env，200 且结构完整。"""
    from drl.agent import ACAgent
    df = generate_demo(n=600)
    from drl.env import TradingEnv
    env3 = TradingEnv(df, state_window=3, extra_factors=np.zeros((len(df), 1)))
    agent = ACAgent(env3.state_dim, 5, state_window=3, seed=5)
    data = agent.to_dict()
    data["factor_expression"] = "close/ma(close,20)-1"
    data["factor_mu"] = 0.0
    data["factor_sd"] = 1.0
    data["min_trade_zone"] = 0.05
    _write_test_model("sw3_factor_test", data)
    r = api_app.post("/api/drl/models/sw3_factor_test/evaluate")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


def test_evaluate_endpoint_corrupt_model_400(api_app, model_cleanup):
    """损坏模型文件：返回 400 友好错误而非裸 500。"""
    from web.api.drl import MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "corrupt_test.json").write_text("{not-json", encoding="utf-8")
    r = api_app.post("/api/drl/models/corrupt_test/evaluate")
    assert r.status_code == 400, r.text
    assert "评估失败" in r.json()["detail"]


def test_train_endpoint_worker_finally_close_loop(api_app, model_cleanup):
    """端到端：/train 后台 worker 完成训练后 finally 调用 close_loop 关闭专属 client。"""
    import web.api.drl as drlmod
    from web.deps import get_db, get_engine

    closed_loops: list = []

    class FakeAiClient:
        async def close_loop(self, loop):
            closed_loops.append(loop)

    class FakeEngine:
        _loop = None

        def __init__(self):
            self.ai_client = FakeAiClient()

    class FakeResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt):
            return FakeResult()

        async def commit(self):
            pass

    class FakeDB:
        def session(self):
            return FakeSession()

    api_app.app.dependency_overrides[get_db] = lambda: FakeDB()
    api_app.app.dependency_overrides[get_engine] = lambda: FakeEngine()

    r = api_app.post("/api/drl/train", json={"episodes": 1, "n_episodes": 1,
                                             "ppo_epochs": 1, "limit": 500})
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    done = False
    for _ in range(400):
        p = api_app.get(f"/api/drl/train/{task_id}").json()
        if p.get("done"):
            done = True
            break
        time.sleep(0.1)
    assert done, "训练任务未在超时内完成"
    assert not p.get("error"), p
    assert closed_loops, "worker finally 未调用 engine.ai_client.close_loop(loop)"
    # 训练结果已标记完成（mark_done 时机不变）
    assert p.get("model_name")
