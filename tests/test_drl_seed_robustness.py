"""多 seed 中位数选模 + 固定种子的回归锁（2026-10）。

背景（docs/DRL做空支持-评估与落地.md §4、docs/DRL成本标定-决策频率与滑点复核.md §8）：
生产配置下 3 个 seed 的验证段超额 13.1%/6.4%/16.0%、OOS 超额 +2.4%/−1.5%/−5.3%——
**单次训练成败接近抛硬币**。修复：
1. 固定默认种子（可复现、跨重启可比；旧值是 None=每轮随机）；
2. seeds>1 时本轮训 N 个 seed，按**验证段超额中位数**选模（OOS 不参与选模，
   它是选完之后的独立安检，用它选模等于把样本外泄漏进选择）。
"""
import pytest

from drl.evolve_engine import _run_multi_seed, _seed_meta, _seed_score


def _mk_result(val_excess=None, val_ret=0.0, oos_ret=0.0, oos_excess=0.0):
    return {"history": [{"ep": 1}], "best_val_excess": val_excess,
            "best_val_ret": val_ret, "best_ret": val_ret,
            "oos_report": {"oos_ret": oos_ret, "oos_excess": oos_excess},
            "elapsed_sec": 1.0}


# ============ 选模评分 ============

def test_seed_score_prefers_val_excess_and_handles_none():
    assert _seed_score(None) == -1e18
    assert _seed_score({"best_val_excess": 0.12}) == 0.12
    assert _seed_score({"best_val_excess": None, "best_val_ret": 0.05}) == 0.05
    assert _seed_score({"best_val_excess": "bad", "best_val_ret": 0.05}) == 0.05
    assert _seed_score({}) == -1e18


# ============ 中位数选模 ============

def test_single_seed_returns_it_verbatim():
    cfg = {}
    calls = []

    def fake_train(c):
        calls.append(c["seed"])
        return _mk_result(val_excess=0.1)

    chosen, meta = _run_multi_seed(fake_train, cfg, 1, 42)
    assert calls == [42]
    assert chosen["seed_chosen"] == 42
    assert len(chosen["seed_runs"]) == 1 and chosen["seed_runs"][0]["seed"] == 42
    assert meta["n_seeds"] == 1


def test_median_selection_picks_middle_by_val_excess():
    cfg = {}
    # 三个 seed 的超额：0.30 / 0.10 / 0.20 → 中位 = 0.20（seed=44）
    by_seed = {42: 0.30, 43: 0.10, 44: 0.20}

    def fake_train(c):
        return _mk_result(val_excess=by_seed[c["seed"]])

    chosen, meta = _run_multi_seed(fake_train, cfg, 3, 42)
    assert chosen["seed_chosen"] == 44
    assert chosen["best_val_excess"] == 0.20
    assert [r["seed"] for r in chosen["seed_runs"]] == [42, 43, 44]
    assert [r["val_excess"] for r in chosen["seed_runs"]] == [0.30, 0.10, 0.20]


def test_even_seed_count_takes_lower_middle():
    cfg = {}
    # 4 个 seed：超额 -0.1 / 0.5 / 0.2 / 0.1 → 排序 [-0.1, 0.1, 0.2, 0.5]，
    # 偶数取**下半**中位数 (len-1)//2=1 → 第 2 小 = 0.1（seed=45，保守侧）。
    # （2026-10 审计修复：旧实现 len//2=2 取 0.2 是"上半中位数"，n=2 时会退化为
    #  最大值挑选——与文档"下半保守"自相矛盾。）
    values = {42: -0.1, 43: 0.5, 44: 0.2, 45: 0.1}

    def fake_train(c):
        return _mk_result(val_excess=values[c["seed"]])

    chosen, _ = _run_multi_seed(fake_train, cfg, 4, 42)
    assert chosen["best_val_excess"] == 0.1
    assert chosen["seed_chosen"] == 45


def test_failed_seed_is_skipped_not_fatal():
    cfg = {}

    def fake_train(c):
        if c["seed"] == 42:
            raise RuntimeError("boom")          # 单 seed 失败 → 跳过
        return _mk_result(val_excess=0.9)

    chosen, meta = _run_multi_seed(fake_train, cfg, 2, 42)
    assert meta["n_seeds"] == 1
    assert chosen["seed_chosen"] == 43


def test_all_seeds_failed_raises():
    cfg = {}

    def fake_train(c):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _run_multi_seed(fake_train, cfg, 3, 1)


def test_seed_meta_carries_oos_for_transparency_only():
    m = _seed_meta(_mk_result(val_excess=0.05, oos_ret=0.01, oos_excess=0.02), 7)
    assert m["seed"] == 7
    assert m["oos_excess"] == 0.02          # 展示用；选模只用 val_excess


# ============ 固定种子 ============

def test_train_cfg_uses_fixed_seed(monkeypatch):
    from config.settings import settings
    from drl.evolve_engine import _strategy_drl_train_cfg
    monkeypatch.setattr(settings, "evolve_strategy_drl_seed", 7, raising=False)
    assert _strategy_drl_train_cfg(8)["seed"] == 7
    monkeypatch.setattr(settings, "evolve_strategy_drl_seed", None, raising=False)
    assert _strategy_drl_train_cfg(8)["seed"] == 42    # None → 兜底 42


# ============ 多 seed 端到端（真实 train_drl，小数据） ============

def test_multi_seed_end_to_end_selects_a_valid_model():
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl

    df = generate_demo(timeframe="1h", n=900, seed=3)

    def fake_train(cfg_):
        cfg_ = dict(cfg_)
        cfg_.update(episodes=2, n_episodes=1, ppo_epochs=1, hidden=[8, 8],
                    val_eval_interval=1, oos_min_bars=80, time_budget=0,
                    train_ratio=0.6, val_ratio=0.2)
        return train_drl(df, cfg_)

    chosen, meta = _run_multi_seed(fake_train, {}, 3, 1)
    assert chosen["seed_chosen"] in (1, 2, 3)
    assert len(chosen["seed_runs"]) == 3
    # 选中模型的验证段超额必须存在（OOS 是独立安检，不进 seed_runs 判据之外）
    assert chosen["best_val_excess"] is not None
