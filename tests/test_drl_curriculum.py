"""P1 课程学习（能力层次）专项测试。

背景（2026-10）：训练早期只从低难度（低波动/趋势清晰）窗口采样，
随训练进度逐步放开高波动段——先学"什么时候该持仓"，再学"什么时候该降仓"。
难度 = 窗口内已实现波动率（vol_5 特征）分位。

覆盖：
- 难度分位计算：单调、[0,1]、窗口起点一一对应
- 课程采样：frac=0 只采最低难度段、frac=1 全放开、中间进度线性放开
- 课程关闭（默认）：采样行为与旧路径完全一致（全范围随机）
- train_drl 单标的课程模式：可跑通、result 元数据、确定性
- train_drl 多标 + 课程：正交叠加可跑通
- evolve train_cfg 透传
- API 校验
"""
import numpy as np
import pytest


def _demo_df(timeframe="1h", n=900, seed=7):
    from backtest.data_loader import generate_demo
    return generate_demo(timeframe=timeframe, n=n, seed=seed)


def _base_cfg(**over):
    cfg = {
        "episodes": 3, "hidden": [16, 16], "n_episodes": 4, "ppo_epochs": 2,
        "mini_batch_size": 64, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
        "val_eval_interval": 1,
    }
    cfg.update(over)
    return cfg


# ============ 1. 难度分位计算 ============

def test_difficulty_ranks_bounded_and_monotonic():
    """难度分位在 [0,1] 且与窗口起点一一对应（长度 = n - win + 1）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=600)
    feat = _precompute_features(df)
    pool = MultiAssetPool(df, feat, window_len=200, log=None)
    ranks = pool._assets[0].difficulty_ranks(200)
    assert ranks.shape[0] == 600 - 200 + 1
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0
    # 分位应覆盖 0（最易）与接近 1（最难）
    assert ranks.min() == 0.0
    # 单起点（win==n）兜底：返回 [0]
    r1 = pool._assets[0].difficulty_ranks(600)
    assert r1.shape == (1,) and r1[0] == 0.0


# ============ 2. 课程采样 ============

def test_curriculum_frac0_only_easy_windows():
    """frac=0：只从最低 curriculum_start 分位的窗口采样。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=900)
    feat = _precompute_features(df)
    pool = MultiAssetPool(df, feat, window_len=300, curriculum=True,
                          curriculum_start=0.3, log=None)
    ranks = pool._assets[0].difficulty_ranks(300)
    rng = np.random.default_rng(1)
    used = [pool.sample(rng, frac=0.0)["start"] for _ in range(50)]
    assert max(ranks[s] for s in used) <= 0.3 + 1e-6, \
        "frac=0 应只采最低 30% 难度窗口"


def test_curriculum_frac1_unrestricted():
    """frac=1：放开全部难度（能采到高难度窗口）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=900)
    feat = _precompute_features(df)
    pool = MultiAssetPool(df, feat, window_len=300, curriculum=True,
                          curriculum_start=0.3, log=None)
    rng = np.random.default_rng(2)
    starts = [pool.sample(rng, frac=1.0)["start"] for _ in range(50)]
    assert max(starts) > 300, "frac=1 应能采到高难度（远端点）窗口"


def test_curriculum_allowed_difficulty_linear():
    """难度放开曲线线性：allowed = start + (1-start)*frac。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=500)
    feat = _precompute_features(df)
    pool = MultiAssetPool(df, feat, window_len=200, curriculum=True,
                          curriculum_start=0.2, log=None)
    assert pool._allowed_difficulty(0.0) == pytest.approx(0.2)
    assert pool._allowed_difficulty(0.5) == pytest.approx(0.6)
    assert pool._allowed_difficulty(1.0) == pytest.approx(1.0)


def test_curriculum_starts_higher_than_resample_floor():
    """frac=0.5 允许的难度应高于 frac=0（进度推进 → 放开更多窗口）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=900)
    feat = _precompute_features(df)
    pool = MultiAssetPool(df, feat, window_len=300, curriculum=True,
                          curriculum_start=0.3, log=None)
    ranks = pool._assets[0].difficulty_ranks(300)
    rng = np.random.default_rng(5)
    used_mid = [pool.sample(rng, frac=0.5)["start"] for _ in range(50)]
    allowed_mid = pool._allowed_difficulty(0.5)
    assert max(ranks[s] for s in used_mid) <= allowed_mid + 1e-6
    assert allowed_mid > 0.3, "进度 0.5 时允许难度应高于起点"


def test_curriculum_off_matches_old_range():
    """课程关闭（默认）：采样范围与旧路径一致（全范围随机）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    df = _demo_df(n=900)
    feat = _precompute_features(df)
    pool_off = MultiAssetPool(df, feat, window_len=300, curriculum=False, log=None)
    rng = np.random.default_rng(1)
    starts = [pool_off.sample(rng)["start"] for _ in range(50)]
    assert min(starts) >= 0 and max(starts) <= 900 - 300
    # 关闭时难度不限制——应能自然覆盖高难度端点
    rng2 = np.random.default_rng(2)
    starts2 = [pool_off.sample(rng2)["start"] for _ in range(100)]
    assert max(starts2) > 300


# ============ 3. train_drl 课程模式 ============

def test_train_drl_curriculum_single_runs():
    """单标的 + 课程：训练跑通，result 元数据标记课程启用。"""
    from drl.agent import train_drl
    df = _demo_df(n=900)
    cfg = _base_cfg(episodes=2, curriculum=True, curriculum_start=0.3)
    res = train_drl(df, cfg)
    assert res.get("curriculum") is True
    assert res.get("curriculum_start") == pytest.approx(0.3)
    assert res["history"], "课程模式训练应产生历史"
    assert res["agent"] is not None


def test_train_drl_curriculum_off_metadata():
    """课程默认关闭：result 元数据 False，行为与旧路径一致。"""
    from drl.agent import train_drl
    df = _demo_df(n=600)
    res = train_drl(df, _base_cfg(episodes=2))
    assert res.get("curriculum") is False


def test_train_drl_curriculum_deterministic():
    """课程模式可复现：同 seed 两次训练 history 逐值一致。"""
    from drl.agent import train_drl
    df = _demo_df(n=900)
    cfg = _base_cfg(episodes=2, curriculum=True, curriculum_start=0.3)
    r1 = train_drl(df, dict(cfg))
    r2 = train_drl(df, dict(cfg))
    h1 = [(row.get("policy_loss"), row.get("val_ret")) for row in r1["history"]]
    h2 = [(row.get("policy_loss"), row.get("val_ret")) for row in r2["history"]]
    assert h1 == h2


def test_train_drl_curriculum_multi_symbol():
    """课程 × 多标的：正交叠加可跑通（每个 episode 标的×难度双过滤）。"""
    from drl.agent import train_drl
    main_df = _demo_df(n=900, seed=1)
    eth_df = _demo_df(n=900, seed=2)
    cfg = _base_cfg(episodes=2, curriculum=True, curriculum_start=0.3,
                    extra_mix_ratio=0.6, n_episodes=4)
    res = train_drl(main_df, cfg, extra_dfs={"ETH/USDT": eth_df})
    assert res.get("curriculum") is True
    assert res.get("multi_symbol") is True
    assert res.get("multi_extra_names") == ["ETH/USDT"]
    assert res["history"]


# ============ 4. evolve train_cfg 透传 ============

def test_evolve_train_cfg_curriculum_default_off(monkeypatch):
    """持续进化默认课程关闭（向后兼容），开启时透传 start。"""
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    monkeypatch.setattr(settings, "evolve_strategy_drl_curriculum", False,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_curriculum_start", 0.3,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["curriculum"] is False
    assert cfg["curriculum_start"] == pytest.approx(0.3)


def test_evolve_train_cfg_curriculum_override(monkeypatch):
    """settings 开启课程生效。"""
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    monkeypatch.setattr(settings, "evolve_strategy_drl_curriculum", True,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_curriculum_start", 0.1,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["curriculum"] is True
    assert cfg["curriculum_start"] == pytest.approx(0.1)


# ============ 5. API 校验 ============

@pytest.fixture
def api_app():
    """仅挂载 drl 路由的最小 FastAPI 应用，桩掉 db/engine 依赖。"""
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


def test_api_curriculum_validation(api_app):
    """curriculum_start 超范围被 400 拒绝。"""
    r_bad = api_app.post("/api/drl/train",
                         json={"episodes": 1, "curriculum_start": 1.5})
    assert r_bad.status_code == 400, "curriculum_start>1 应 400"
    r_neg = api_app.post("/api/drl/train",
                         json={"episodes": 1, "curriculum_start": -0.1})
    assert r_neg.status_code == 400, "负 curriculum_start 应 400"
