"""P0 并行采样上量（经验规模）专项测试。

背景（2026-10）：并行收集 worker 上限此前硬编码 min(n_episodes,4)，是
经验规模的吞吐瓶颈——n_episodes 提到 16/32 时仍只有 4 条同时在跑。
本次改动：
1) `_resolve_collect_workers` 按配置 + CPU 核数解析并行线程数（0=自动）；
2) train_drl 的 ThreadPoolExecutor 用它，不再写死 4；
3) 默认 n_episodes 提升（持续进化 8→16、手动训练 API 4→8）；
4) result/API/UI 透出实际线程数供审计。

覆盖：
- worker 解析纯函数：自动/显式/钳位到 n_episodes
- 多线程训练可复现（同 seed 两次 history 逐值一致，含 >4 线程）
- n_episodes 大值时批量拼接/PPO 更新正常
- result 元数据含 collect_workers
- evolve train_cfg 透传 collect_workers 与 n_episodes=16 默认
"""
import pytest


def _demo_df(timeframe="1h", n=500, seed=7):
    from backtest.data_loader import generate_demo
    return generate_demo(timeframe=timeframe, n=n, seed=seed)


def _base_cfg(**over):
    cfg = {
        "episodes": 3, "hidden": [16, 16], "n_episodes": 8, "ppo_epochs": 2,
        "mini_batch_size": 64, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
        "val_eval_interval": 1,
    }
    cfg.update(over)
    return cfg


# ============ 1. worker 解析纯函数 ============

def test_resolve_workers_auto_uses_cpu():
    """auto（0）：至少 4 线程、不超 n_episodes。"""
    from drl.agent import _resolve_collect_workers
    assert _resolve_collect_workers(16, 0) >= 4
    assert _resolve_collect_workers(16, 0) <= 16
    assert _resolve_collect_workers(8, 0) >= 4
    assert _resolve_collect_workers(8, 0) <= 8


def test_resolve_workers_explicit():
    """显式指定：直接用，并夹到 [1, n_episodes]。"""
    from drl.agent import _resolve_collect_workers
    assert _resolve_collect_workers(16, 2) == 2
    assert _resolve_collect_workers(16, 99) == 16      # 不超 n_episodes
    assert _resolve_collect_workers(8, 1) == 1
    assert _resolve_collect_workers(3, 0) >= 1         # 小 n_episodes 不造空线程
    assert _resolve_collect_workers(3, 0) <= 3


def test_resolve_workers_negative_treated_as_auto():
    """负值/非正视为 auto。"""
    from drl.agent import _resolve_collect_workers
    assert _resolve_collect_workers(16, -1) == _resolve_collect_workers(16, 0)


# ============ 2. 多线程训练正确性 ============

def test_train_drl_scaled_workers_deterministic():
    """n_episodes=8（>4 旧上限）自动多线程：同 seed 两次训练 history 逐值一致。"""
    from drl.agent import train_drl
    df = _demo_df(n=600)
    cfg = _base_cfg(episodes=2)
    r1 = train_drl(df, dict(cfg))
    r2 = train_drl(df, dict(cfg))
    h1 = [(row.get("episode"), row.get("policy_loss"), row.get("val_ret"),
           row.get("total_ret")) for row in r1["history"]]
    h2 = [(row.get("episode"), row.get("policy_loss"), row.get("val_ret"),
           row.get("total_ret")) for row in r2["history"]]
    assert h1 == h2, "多线程并行收集应保持可复现（per-worker rng 独立）"


def test_train_drl_explicit_workers_matches_result():
    """显式 collect_workers=2：result 元数据记录实际线程数=2。"""
    from drl.agent import train_drl
    df = _demo_df(n=600)
    cfg = _base_cfg(episodes=2, collect_workers=2)
    res = train_drl(df, cfg)
    assert res.get("collect_workers") == 2


def test_train_drl_auto_workers_reported():
    """auto：result 记录解析后的线程数（>4 时并行上量生效）。"""
    from drl.agent import train_drl, _resolve_collect_workers
    df = _demo_df(n=600)
    cfg = _base_cfg(episodes=2, n_episodes=8)
    res = train_drl(df, cfg)
    expected = _resolve_collect_workers(8, 0)
    assert res.get("collect_workers") == expected


def test_train_drl_large_batch_concat_and_update():
    """n_episodes=12 大批量：拼接 + PPO 更新正常，history 非空。"""
    from drl.agent import train_drl
    df = _demo_df(n=600)
    cfg = _base_cfg(episodes=2, n_episodes=12, mini_batch_size=64)
    res = train_drl(df, cfg)
    assert res["history"], "大批量训练应产生历史"
    assert res["agent"] is not None
    assert all("policy_loss" in h for h in res["history"])


def test_train_drl_scaled_workers_multi_symbol():
    """多标的 + 大并行：混合训练可跑通（每 episode 独立标的×时间窗）。"""
    from drl.agent import train_drl
    main_df = _demo_df(n=600, seed=1)
    eth_df = _demo_df(n=600, seed=2)
    sol_df = _demo_df(n=600, seed=3)
    cfg = _base_cfg(episodes=2, n_episodes=8, extra_mix_ratio=0.6)
    res = train_drl(main_df, cfg,
                    extra_dfs={"ETH/USDT": eth_df, "SOL/USDT": sol_df})
    assert res.get("multi_symbol") is True
    assert sorted(res.get("multi_extra_names", [])) == ["ETH/USDT", "SOL/USDT"]
    assert res["history"]


# ============ 3. evolve train_cfg 默认与透传 ============

def test_evolve_train_cfg_scaled_default(monkeypatch):
    """持续进化默认 n_episodes=16、collect_workers 透传（0=自动）。"""
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    monkeypatch.setattr(settings, "evolve_strategy_drl_n_episodes", 16,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_collect_workers", 0,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["n_episodes"] == 16
    assert cfg["collect_workers"] == 0


def test_evolve_train_cfg_collect_workers_override(monkeypatch):
    """settings 显式指定线程数生效。"""
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    monkeypatch.setattr(settings, "evolve_strategy_drl_collect_workers", 2,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["collect_workers"] == 2


# ============ 4. API 层校验 ============

@pytest.fixture
def api_app():
    """仅挂载 drl 路由的最小 FastAPI 应用，桩掉 db/engine 依赖。

    校验拒绝路径只需依赖可解析（handler 内第一句 _validate_train 抛 400），
    不会真正启动训练 worker。db/engine 桩与 test_drl_fixes 的端到端用例一致。
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.drl import router
    from web.deps import get_db, get_engine

    class _StubDB:
        pass

    class _StubEngine:
        pass

    app = FastAPI()
    app.include_router(router)
    app.state.db = _StubDB()
    app.state.engine = _StubEngine()
    app.dependency_overrides[get_db] = lambda: _StubDB()
    app.dependency_overrides[get_engine] = lambda: _StubEngine()
    return TestClient(app)


def test_api_collect_workers_validation(api_app):
    """超范围/负值 collect_workers 被 400 拒绝（字段已在 schema 中生效）。"""
    r_bad = api_app.post("/api/drl/train",
                         json={"episodes": 1, "collect_workers": 999})
    assert r_bad.status_code == 400, "collect_workers 超范围应 400"
    r_neg = api_app.post("/api/drl/train",
                         json={"episodes": 1, "collect_workers": -1})
    assert r_neg.status_code == 400, "负 collect_workers 应 400"
    r_ep = api_app.post("/api/drl/train",
                        json={"episodes": 1, "n_episodes": 999})
    assert r_ep.status_code == 400, "n_episodes 超上限应 400"
