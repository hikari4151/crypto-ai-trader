"""P0 多标的混合训练（经验广度）专项测试。

覆盖：
1) 单标的向后兼容：不传 extra_dfs 时 train_drl 行为与旧路径一致（
   multi_symbol=False、history 结构不变、结果可复现）；
2) MultiAssetPool 采样正确性：随机标的×随机时间窗切片，切片特征维度与
   主标的一致、切片长度与窗口一致、额外标的因子列可复算；
3) train_drl 多标的模式：extra_dfs 传入后训练可跑通，返回 multi_symbol/
   multi_extra_names 元数据，且主标的 val/OOS 判据仍以主标的为准；
4) 维度守卫：主标的有因子列而额外标的无因子列时自动降级单标的；
5) evolve 引擎集成：多标模式下 extra_dfs_pool 喂入 train_drl、跨标 OOS 跳过。
"""
import asyncio

import numpy as np
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


def _demo_df(timeframe="5m", n=700, seed=7):
    from backtest.data_loader import generate_demo
    return generate_demo(timeframe=timeframe, n=n, seed=seed)


def _base_cfg(**over):
    cfg = {
        "episodes": 3, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
        "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
        "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
        "state_window": 2, "oos_min_bars": 80, "time_budget": 0,
        "val_eval_interval": 1,
    }
    cfg.update(over)
    return cfg


# ============ 1. 单标的向后兼容 ============

def test_train_drl_no_extra_is_single_symbol():
    """不传 extra_dfs：multi_symbol=False，历史结构不变，无额外标的元数据。"""
    from drl.agent import train_drl
    df = _demo_df(n=700)
    res = train_drl(df, _base_cfg())
    assert res.get("multi_symbol") is False
    assert res.get("multi_extra_names") == []
    assert res.get("multi_mix_ratio") == 0.0
    assert "history" in res and "agent" in res
    assert "best_ret" in res


def test_train_drl_no_extra_deterministic():
    """单标的路径保持可复现（同 seed 两次训练 history 逐值一致）。"""
    from drl.agent import train_drl
    df = _demo_df(n=500)
    cfg = _base_cfg(episodes=2)
    r1 = train_drl(df, dict(cfg))
    r2 = train_drl(df, dict(cfg))
    h1 = [(row.get("policy_loss"), row.get("val_ret")) for row in r1["history"]]
    h2 = [(row.get("policy_loss"), row.get("val_ret")) for row in r2["history"]]
    assert h1 == h2


# ============ 2. MultiAssetPool 采样正确性 ============

def test_pool_samples_consistent_dims():
    """额外标的切片特征维度与主标的一致（共享网络的前提）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    main_df = _demo_df(n=700, seed=1)
    extra_df = _demo_df(n=800, seed=2)
    main_feat = _precompute_features(main_df)
    pool = MultiAssetPool(main_df, main_feat,
                          extra_dfs={"ETH/USDT": extra_df},
                          state_window=2)
    assert pool.n_assets == 2
    assert pool.extra_names == ["ETH/USDT"]
    rng = np.random.default_rng(42)
    for _ in range(20):
        s = pool.sample(rng)
        assert s["features_slice"].shape[1] == main_feat.shape[1]
        assert s["df_slice"].shape[0] == s["features_slice"].shape[0]
        assert s["window_len"] == len(s["df_slice"])


def test_pool_main_only_samples_full_train_segment():
    """单标的池：sample() 恒返回主标的完整训练段（start=0，window=主标的长度）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    main_df = _demo_df(n=700)
    main_feat = _precompute_features(main_df)
    pool = MultiAssetPool(main_df, main_feat, state_window=2)
    rng = np.random.default_rng(1)
    for _ in range(10):
        s = pool.sample(rng)
        assert s["name"] == "main"
        assert s["start"] == 0
        assert s["window_len"] == len(main_df)
        assert s["df_slice"].shape[0] == len(main_df)


def test_pool_mix_ratio_zero_only_main():
    """extra_mix_ratio=0：即使有额外标的也只采样主标的。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    main_df = _demo_df(n=600, seed=1)
    extra_df = _demo_df(n=600, seed=2)
    main_feat = _precompute_features(main_df)
    pool = MultiAssetPool(main_df, main_feat,
                          extra_dfs={"ETH/USDT": extra_df},
                          state_window=1, extra_mix_ratio=0.0)
    rng = np.random.default_rng(1)
    names = {pool.sample(rng)["name"] for _ in range(20)}
    assert names == {"main"}


def test_pool_skips_too_short_extra():
    """过短的额外标的被排除（不满足最小K线数），池退回单标的。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    main_df = _demo_df(n=600)
    main_feat = _precompute_features(main_df)
    short_df = _demo_df(n=100)  # < 300 根
    pool = MultiAssetPool(main_df, main_feat,
                          extra_dfs={"SHORT/USDT": short_df},
                          state_window=1)
    assert pool.n_assets == 1
    assert pool.extra_names == []


def test_pool_extra_factor_column_built():
    """factor_composite 配方存在时，额外标的复算出因子列（维度一致）。"""
    from drl.multi_asset import MultiAssetPool, _build_composite_factor
    from drl.env import _precompute_features

    main_df = _demo_df(n=600, seed=1)
    extra_df = _demo_df(n=600, seed=2)
    main_feat = _precompute_features(main_df)
    # 真实因子配方（factors.library 的复合因子）：按 weights 在任意标的 df 上
    # 复算 expanding-z 合成因子（与 factor_miner/部署端同口径）。
    real_weights = {"mom_5": 1.0, "vol_10": 0.5}
    factors_main = _build_composite_factor(main_df, real_weights)
    if factors_main is None:
        pytest.skip("测试环境因子引擎不可用（compute_factor_matrix 失败），跳过")
    pool = MultiAssetPool(
        main_df, main_feat, main_factors=factors_main,
        extra_dfs={"ETH/USDT": extra_df},
        state_window=1,
        factor_builder=lambda df: _build_composite_factor(df, real_weights))
    assert pool.n_assets == 2, "主标的有因子列、额外标的可复算 → 两者都保留"
    rng = np.random.default_rng(3)
    for _ in range(5):
        s = pool.sample(rng)
        assert s["factors_slice"] is not None
        assert s["factors_slice"].shape[0] == len(s["df_slice"])


def test_pool_degrades_when_main_has_factor_extra_has_none():
    """主标的有因子列、额外标的无法复算 → 额外标的被排除（维度不一致守卫）。"""
    from drl.multi_asset import MultiAssetPool
    from drl.env import _precompute_features

    main_df = _demo_df(n=600, seed=1)
    extra_df = _demo_df(n=600, seed=2)
    main_feat = _precompute_features(main_df)
    # 主标的给一个假因子列（长度==主标的），额外标的无因子构建器
    fake_factors = np.zeros(len(main_df))
    pool = MultiAssetPool(main_df, main_feat, main_factors=fake_factors,
                          extra_dfs={"ETH/USDT": extra_df}, state_window=1)
    assert pool.n_assets == 1
    assert pool.extra_names == []


# ============ 3. train_drl 多标的模式 ============

def test_train_drl_multi_symbol_runs():
    """传入 extra_dfs：训练跑通，返回 multi_symbol=True + 额外标的名单。"""
    from drl.agent import train_drl
    main_df = _demo_df(n=700, seed=1)
    eth_df = _demo_df(n=700, seed=2)
    sol_df = _demo_df(n=700, seed=3)
    res = train_drl(main_df, _base_cfg(extra_mix_ratio=0.5),
                    extra_dfs={"ETH/USDT": eth_df, "SOL/USDT": sol_df})
    assert res.get("multi_symbol") is True
    assert sorted(res.get("multi_extra_names", [])) == ["ETH/USDT", "SOL/USDT"]
    assert res.get("multi_mix_ratio") == 0.5
    assert res["history"], "多标的训练应产生历史"
    assert res["agent"].state_dim == res["state_dim"]


def test_train_drl_multi_symbol_all_extra_ratio():
    """extra_mix_ratio=1.0：全部轨迹来自额外标的（主标的仅承担 val/OOS）。"""
    from drl.agent import train_drl
    main_df = _demo_df(n=700, seed=1)
    eth_df = _demo_df(n=700, seed=2)
    res = train_drl(main_df, _base_cfg(extra_mix_ratio=1.0),
                    extra_dfs={"ETH/USDT": eth_df})
    assert res.get("multi_symbol") is True
    assert res.get("multi_extra_names") == ["ETH/USDT"]


def test_train_drl_multi_symbol_degrades_to_single():
    """主标的有因子列（预计算数组）、额外标的无法复算 → 自动降级单标的。"""
    from drl.agent import train_drl
    main_df = _demo_df(n=700, seed=1)
    eth_df = _demo_df(n=700, seed=2)
    # factor_values 预计算数组（无配方无表达式）→ 额外标的无法重建因子列
    cfg = _base_cfg()
    cfg["factor_values"] = np.linspace(0.0, 1.0, len(main_df))
    res = train_drl(main_df, cfg, extra_dfs={"ETH/USDT": eth_df})
    assert res.get("multi_symbol") is False
    assert res.get("multi_extra_names") == []
    assert res["agent"] is not None


# ============ 5. evolve 引擎集成 ============

def test_api_load_extra_dfs_skips_bad_symbols(monkeypatch):
    """web/api 的 _load_extra_dfs：非法符号/加载失败/数据不足的标的被跳过。"""
    from web.api.drl import TrainIn, _load_extra_dfs
    import backtest.data_loader as bdl

    async def _fake_load(exchange, symbol, timeframe, limit):
        if symbol == "ETH/USDT":
            return _demo_df(n=500, seed=2)
        if symbol == "SHORT/USDT":
            return _demo_df(n=100, seed=3)  # 不足 300 根
        raise RuntimeError("exchange down")  # SOL/USDT 加载失败
    monkeypatch.setattr(bdl, "load_klines_cached", _fake_load)

    async def _run():
        body = TrainIn(data_source="exchange", symbol="BTC/USDT",
                       timeframe="5m", limit=500,
                       extra_symbols=["ETH/USDT", "SOL/USDT", "SHORT/USDT", "bad-sym"])
        return await _load_extra_dfs(body)
    out = asyncio.run(_run())
    assert list(out) == ["ETH/USDT"], "只应保留成功加载且数据充足的标的"


def test_api_load_extra_dfs_empty_and_csv(monkeypatch):
    """无 extra_symbols 或 CSV 源：返回空（CSV 不支持多标的）。"""
    from web.api.drl import TrainIn, _load_extra_dfs

    async def _run():
        body = TrainIn(data_source="demo", symbol="BTC/USDT", timeframe="5m")
        return await _load_extra_dfs(body)
    out = asyncio.run(_run())
    assert out == {}

    async def _run_csv():
        body = TrainIn(data_source="csv", symbol="BTC/USDT", timeframe="5m",
                       csv_path="x.csv", extra_symbols=["ETH/USDT"])
        return await _load_extra_dfs(body)
    out = asyncio.run(_run_csv())
    assert out == {}


def _make_engine(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_population", False,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_symbols",
                        ["BTC/USDT", "ETH/USDT"], raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    return eng


@pytest.mark.asyncio
async def test_engine_builds_extra_dfs_pool(tmp_path, monkeypatch):
    """多标模式开启时，引擎把池内其他标的数据喂给 train_drl 作为额外经验。"""
    import drl.evolve_engine as ev
    eng = _make_engine(tmp_path, monkeypatch)
    btc_df = _demo_df(n=500, seed=1)
    eth_df = _demo_df(n=500, seed=2)
    captured = {}

    async def _fake_fetch(symbol: str = ""):
        return btc_df if symbol == "BTC/USDT" else eth_df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    monkeypatch.setattr(eng, "_data_incremented", lambda *a, **k: True)
    fake_agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        captured["extra_dfs"] = extra_dfs
        return {
            "agent": fake_agent,
            "history": [{"best_ret": 0.1}],
            "best_ret": 0.1,
            "best_val_ret": 0.05,
            "oos_report": {"enabled": False},
            "multi_symbol": bool(extra_dfs),
            "multi_extra_names": list(extra_dfs or {}),
            "multi_mix_ratio": 0.5,
        }
    monkeypatch.setattr(ev, "train_drl", _fake_train)
    await eng._train_strategy_drl_once(force=True)
    assert captured.get("extra_dfs") is not None, "多标模式应把额外标的喂给 train_drl"
    assert list(captured["extra_dfs"]) == ["ETH/USDT"]


@pytest.mark.asyncio
async def test_engine_multi_symbol_skips_cross_oos(tmp_path, monkeypatch):
    """多标训练产出模型后，跨标 OOS 跳过（训练已见全部池内标的，不再构成样本外）。"""
    import drl.evolve_engine as ev
    eng = _make_engine(tmp_path, monkeypatch)
    btc_df = _demo_df(n=500, seed=1)
    eth_df = _demo_df(n=500, seed=2)

    async def _fake_fetch(symbol: str = ""):
        return btc_df if symbol == "BTC/USDT" else eth_df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    monkeypatch.setattr(eng, "_data_incremented", lambda *a, **k: True)
    fake_agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)
    monkeypatch.setattr(eng, "_cross_symbol_oos",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                            "多标模式下跨标 OOS 不应被调用")))

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        return {
            "agent": fake_agent,
            "history": [{"best_ret": 0.1}],
            "best_ret": 0.1,
            "best_val_ret": 0.05,
            "oos_report": {"enabled": True, "oos_ret": 0.03},
            "multi_symbol": True,
            "multi_extra_names": list(extra_dfs or {}),
            "multi_mix_ratio": 0.5,
        }
    monkeypatch.setattr(ev, "train_drl", _fake_train)
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_status.get("cross_oos") is None


@pytest.mark.asyncio
async def test_engine_single_symbol_still_calls_cross_oos(tmp_path, monkeypatch):
    """单标模式（多标开关关闭）：跨标 OOS 照常执行（行为不变）。"""
    import drl.evolve_engine as ev
    from config.settings import settings
    eng = _make_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_multi_symbol", False,
                        raising=False)
    btc_df = _demo_df(n=500, seed=1)
    called_cross = {"n": 0}

    async def _fake_fetch(symbol: str = ""):
        return btc_df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    monkeypatch.setattr(eng, "_data_incremented", lambda *a, **k: True)
    fake_agent = ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_agent)

    async def _fake_cross(agent, train_symbol, state_window):
        called_cross["n"] += 1
        return {"symbol": "ETH/USDT", "ret": 0.02, "position_ratio": 0.5}
    monkeypatch.setattr(eng, "_cross_symbol_oos", _fake_cross)

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        return {
            "agent": fake_agent,
            "history": [{"best_ret": 0.1}],
            "best_ret": 0.1,
            "best_val_ret": 0.05,
            "oos_report": {"enabled": True, "oos_ret": 0.03},
            "multi_symbol": False,
            "multi_extra_names": [],
            "multi_mix_ratio": 0.0,
        }
    monkeypatch.setattr(ev, "train_drl", _fake_train)
    await eng._train_strategy_drl_once(force=True)
    assert called_cross["n"] == 1, "单标模式跨标 OOS 应照常执行"
