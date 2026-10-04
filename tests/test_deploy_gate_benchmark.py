"""验收口径修复的回归锁（2026-10）：基准相对 + 活跃度守卫。

背景（docs/DRL成本标定-决策频率与滑点复核.md §4）：
  动作空间 [0..1] 只能做多/空仓 + OOS 落在熊市 + 旧门要求 `oos_ret > 0`
  ⇒ **结构上不可能有任何多头策略过门**（唯一不亏的做法是空仓=0，而 0 被 >0 拒掉）。
  产线实测 452/501 轮被拒、fitness 多为 0.0，与这条死结一致。
修复：`超额 = oos_ret − 同段买入持有` 为正才过门；同时加活跃度守卫，
防止"从不交易"在下跌窗口里白拿超额被当成择时能力。
"""
import pytest

from drl.evolve_engine import _oos_inactive_reason, _strategy_deploy_gate


def _oos(**kw) -> dict:
    base = {"enabled": True, "oos_ret": 0.0, "oos_bh_ret": 0.0,
            "oos_turnover": 5.0, "regime_shift": False}
    base.update(kw)
    return base


def _result(**kw) -> dict:
    return {"oos_report": _oos(**kw), "deployment_blocked": False}


# ============ 基准相对验收 ============

def test_beats_benchmark_passes():
    """熊市里保住资本（0% vs 基准 −27%）应过门——旧口径下这必然被拒。"""
    ok, reason = _strategy_deploy_gate(_result(oos_ret=0.0, oos_bh_ret=-0.27))
    assert ok, reason


def test_losing_to_benchmark_rejected():
    """跑输买入持有 → 拒（牛市里"只会满仓躺赢"就是这种：超额 = 0）。"""
    ok, reason = _strategy_deploy_gate(_result(oos_ret=0.20, oos_bh_ret=0.20))
    assert not ok and "未跑赢买入持有" in reason
    ok2, reason2 = _strategy_deploy_gate(_result(oos_ret=-0.30, oos_bh_ret=-0.27))
    assert not ok2 and "未跑赢买入持有" in reason2
    ok3, _ = _strategy_deploy_gate(_result(oos_ret=0.05, oos_bh_ret=0.20))
    assert not ok3


def test_absolute_gate_still_used_without_benchmark_field():
    """老报告没有 oos_bh_ret → 退回旧的绝对判据（向后兼容）。"""
    oos = {"enabled": True, "oos_ret": 0.05, "oos_turnover": 5.0}
    ok, _ = _strategy_deploy_gate({"oos_report": oos, "deployment_blocked": False})
    assert ok
    oos2 = {"enabled": True, "oos_ret": -0.05, "oos_turnover": 5.0}
    ok2, reason2 = _strategy_deploy_gate({"oos_report": oos2, "deployment_blocked": False})
    assert not ok2 and "非正" in reason2


def test_benchmark_can_be_disabled_by_setting(monkeypatch):
    """开关关闭 → 回到旧的绝对收益判据（可回退）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_requires_benchmark", False,
                        raising=False)
    ok, reason = _strategy_deploy_gate(_result(oos_ret=-0.10, oos_bh_ret=-0.27))
    assert not ok and "非正" in reason


def test_hard_gate_and_missing_oos_still_reject():
    ok, reason = _strategy_deploy_gate({"oos_report": _oos(), "deployment_blocked": True})
    assert not ok and "硬门" in reason
    ok2, reason2 = _strategy_deploy_gate({"oos_report": {"enabled": False}})
    assert not ok2 and "样本不足" in reason2
    ok3, reason3 = _strategy_deploy_gate(_result(oos_ret=float("nan"), oos_bh_ret=-0.3))
    assert not ok3 and "无效" in reason3 or not ok3


# ============ 活跃度守卫 ============

def test_inactive_model_does_not_pass_on_bear_benchmark():
    """从不交易 → 熊市里超额为正，但活跃度守卫必须拦下（那是没参与，不是会择时）。"""
    ok, reason = _strategy_deploy_gate(_result(oos_ret=0.0, oos_bh_ret=-0.27,
                                               oos_turnover=0.0))
    assert not ok and "换手仅" in reason


def test_inactive_reason_falls_back_to_trades_then_position():
    assert "从未发出交易指令" in _oos_inactive_reason(
        {"oos_trades": 0, "oos_position_ratio": 0.0}, 0.5)
    assert "几乎无信号" in _oos_inactive_reason(
        {"oos_position_ratio": 0.0}, 0.5)
    assert _oos_inactive_reason({"oos_turnover": 1.0}, 0.5) is None
    # 报告什么活跃度字段都没有 → 不拦（老报告兼容，交由收益判据把关）
    assert _oos_inactive_reason({}, 0.5) is None


def test_active_model_passes_activity_guard():
    assert _strategy_deploy_gate(_result(oos_ret=0.02, oos_bh_ret=-0.05,
                                         oos_turnover=3.2))[0]


def test_min_turnover_setting_respected(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_oos_turnover", 10.0,
                        raising=False)
    ok, reason = _strategy_deploy_gate(_result(oos_ret=0.02, oos_bh_ret=-0.05,
                                               oos_turnover=3.2))
    assert not ok and "换手仅" in reason


# ============ OOS 报告必须带基准/活跃度字段（gate 的输入） ============

def test_train_drl_oos_report_has_benchmark_and_activity_fields():
    import numpy as np
    import pandas as pd

    from drl.agent import train_drl

    rng = np.random.default_rng(5)
    n = 420 + 200
    close = 100.0 * np.cumprod(1.0 + 0.0004 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": np.full(n, 100.0)}, index=idx)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "oos_min_bars": 80})
    oos = res.get("oos_report") or {}
    assert oos.get("enabled"), oos
    for key in ("oos_bh_ret", "oos_excess", "oos_excess_segments",
                "oos_excess_positive_segments", "oos_turnover", "oos_active_segments"):
        assert key in oos, f"OOS 报告缺 {key}"
    # 超额 == 收益 − 基准（同一份报告内自洽）
    assert oos["oos_excess"] == pytest.approx(oos["oos_ret"] - oos["oos_bh_ret"], abs=1e-6)
    # 基准 == 同段买入持有（切分口径与 train_drl 默认一致：train 0.7 / val 0.15）
    n_train = max(300, int(n * 0.7))
    n_val = max(100, int(n * 0.15))
    oos_df = df.iloc[n_train + n_val:]
    bh = float(oos_df["close"].iloc[-1] / oos_df["close"].iloc[0] - 1.0)
    assert oos["oos_bh_ret"] == pytest.approx(bh, abs=1e-5)
    # fitness 口径字段存在（跨轮可比的超额）
    assert "best_val_excess" in res


# ============ 过拟合衰减也必须走超额口径 ============

def _declining_df(n: int = 700, drift: float = -0.001, vol: float = 0.01, seed: int = 11):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0.0, vol, n)
    close = 100.0 * np.cumprod(1.0 + np.maximum(rets, -0.5))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": np.full(n, 100.0)}, index=idx)


def test_overfit_decay_uses_excess_basis():
    """下跌窗口下，衰减判据必须走**超额**口径。

    旧口径：`_oos_decay(正训练段, 负OOS) = 1.0` ⇒ 任何只能做多的模型在熊市 OOS
    里都被判"疑似过拟合"，deployment_blocked=True，**新的基准相对部署门根本没机会
    被问到**（端到端实测：超额 +2.4% 仍被硬门以"衰减 100%"拦下）。
    新口径：参照 = 验证段超额，OOS 侧 = 同段超额。
    """
    from drl.agent import _oos_decay, train_drl

    df = _declining_df()
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "excess", "decision_interval": 4,
                         "oos_min_bars": 80})
    oos = res["oos_report"]
    assert oos["enabled"]
    assert oos["oos_bh_ret"] < 0, "构造的是下跌行情，基准应为负"
    assert oos["decay_basis"] == "excess"
    assert oos["decay_ref"] == pytest.approx(res["best_val_excess"], abs=1e-6)
    assert oos["decay"] == pytest.approx(
        _oos_decay(oos["decay_ref"], oos["oos_excess"]), abs=1e-4)


def test_overfit_decay_falls_back_to_absolute_without_benchmark():
    """无基准字段的老路径：仍按绝对收益判衰减（向后兼容）。"""
    import numpy as np
    import pandas as pd

    from drl.agent import _oos_decay, _bh_return, train_drl

    rng = np.random.default_rng(7)
    n = 700
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": np.full(n, 100.0)}, index=idx)
    res = train_drl(df, {"episodes": 2, "n_episodes": 1, "ppo_epochs": 1,
                         "hidden": [8, 8], "seed": 1, "val_eval_interval": 1,
                         "reward_mode": "absolute", "decision_interval": 4,
                         "oos_min_bars": 80})
    oos = res["oos_report"]
    # 两条口径并存时以超额为准（本用例只验证字段自洽：decay 一定由 decay_ref 算出）
    assert oos["decay_basis"] in ("excess", "absolute")
    _ref = oos["decay_ref"] if oos["decay_basis"] == "excess" else oos["train_ret"]
    _obs = oos["oos_excess"] if oos["decay_basis"] == "excess" else oos["oos_ret"]
    assert oos["decay"] == pytest.approx(_oos_decay(_ref, _obs), abs=1e-4)
    # 基准字段与买入持有自洽（切分口径 train 0.7 / val 0.15）
    n_train, n_val = max(300, int(n * 0.7)), max(100, int(n * 0.15))
    assert oos["oos_bh_ret"] == pytest.approx(_bh_return(df.iloc[n_train + n_val:]),
                                              abs=1e-5)


# ============ 分段稳健性：≥2/3 子段超额为正 ============
# 背景（docs/DRL成本标定-决策频率与滑点复核.md §8）：中位数超额会被"一段灾难行情
# 掩盖"——例如 [-0.45, +0.02, +0.03] 的中位超额 +0.02 > 0，旧门可过；但 3 段里 2 段
# 亏损，"择时能力"并不存在，只是运气在两个小段上微赚。要求正超额子段 ≥ ceil(n×2/3)。

def test_segment_majority_rejects_when_median_marginally_positive(monkeypatch):
    """中位超额为正但仅 1/3 子段为正 → 拒（这正是分段门的价值所在）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        2.0 / 3.0, raising=False)
    res = _result(oos_ret=0.02, oos_bh_ret=0.0,
                  oos_segments=3, oos_excess_segments=[-0.45, 0.02, 0.03],
                  oos_excess_positive_segments=1)
    ok, reason = _strategy_deploy_gate(res)
    assert not ok, "仅 1/3 段为正必须被拒"
    assert "分段稳健性不足" in reason and "1 段" in reason


def test_segment_majority_passes_two_of_three(monkeypatch):
    """2/3 段超额为正 → 过（上一轮 e2e：+2.4/+12.1/−41.5 → 2/3 合格）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        2.0 / 3.0, raising=False)
    res = _result(oos_ret=0.02, oos_bh_ret=-0.17,
                  oos_segments=3, oos_excess_segments=[0.12, 0.02, -0.41],
                  oos_excess_positive_segments=2)
    ok, reason = _strategy_deploy_gate(res)
    assert ok, reason


def test_segment_majority_all_segments_positive_passes(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        2.0 / 3.0, raising=False)
    res = _result(oos_ret=0.30, oos_bh_ret=0.05,
                  oos_segments=3, oos_excess_segments=[0.2, 0.1, 0.3],
                  oos_excess_positive_segments=3)
    ok, reason = _strategy_deploy_gate(res)
    assert ok, reason


def test_segment_majority_disabled_keeps_median_only(monkeypatch):
    """fraction<=0 = 关闭：回到只查中位超额（旧行为，向后兼容）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        0.0, raising=False)
    res = _result(oos_ret=0.02, oos_bh_ret=0.0,
                  oos_segments=3, oos_excess_segments=[-0.45, 0.02, 0.03],
                  oos_excess_positive_segments=1)
    ok, reason = _strategy_deploy_gate(res)
    assert ok, reason


def test_segment_majority_skipped_for_old_report(monkeypatch):
    """老报告没有逐段字段 → 跳过（向前兼容，不因缺字段误伤）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        2.0 / 3.0, raising=False)
    ok, reason = _strategy_deploy_gate(_result(oos_ret=0.02, oos_bh_ret=-0.17))
    assert ok, reason


def test_segment_majority_derives_positive_count_from_list(monkeypatch):
    """缺 oos_excess_positive_segments 时从逐段列表现算（仅 1 段为正 → 拒）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        2.0 / 3.0, raising=False)
    res = _result(oos_ret=0.02, oos_bh_ret=0.0,
                  oos_segments=3, oos_excess_segments=[-0.45, -0.02, 0.03])
    ok, reason = _strategy_deploy_gate(res)
    assert not ok and "分段稳健性不足" in reason


def test_segment_majority_fraction_100pct_requires_all(monkeypatch):
    """fraction=1.0 → 所有段都必须为正。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        1.0, raising=False)
    res = _result(oos_ret=0.02, oos_bh_ret=0.0,
                  oos_segments=3, oos_excess_segments=[0.2, 0.1, -0.3],
                  oos_excess_positive_segments=2)
    ok, reason = _strategy_deploy_gate(res)
    assert not ok and "分段稳健性不足" in reason


def test_segment_majority_production_value_rounds_to_two_of_three(monkeypatch):
    """生产配置 0.6666666（config.yaml）必须是"3 段要 2 段"而非"要 3 段"。

    回归锁（2026-10 实测）：config.yaml 曾写 0.666667 → ceil(3×0.666667)=3，
    把"≥2/3"悄悄变成"必须3/3"，多标的 E 配置（OOS 超额 +6.4%、2 段为负但 1 段为正
    之外的第 2 段微负）被误拒且原因写"需 ≥3 段"。同理 0.6666666667（>2/3 的 fp 值）
    也会 3×→ceil=3——必须用严格 <2/3 的表示，或指望门内 epsilon 兜底。
    """
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_deploy_min_positive_fraction",
                        0.6666666, raising=False)
    # 2/3 段为正 → 过（0.6666666×3=1.9999998 → ceil=2）
    res_ok = _result(oos_ret=0.02, oos_bh_ret=0.0,
                     oos_segments=3, oos_excess_segments=[0.2, -0.1, 0.3],
                     oos_excess_positive_segments=2)
    ok, reason = _strategy_deploy_gate(res_ok)
    assert ok, f"0.6666666 下 2/3 段为正应过门，实际 {reason!r}"
    # 仅 1/3 段为正 → 拒，且原因说"需 ≥2 段"（不是 ≥3）
    res_rej = _result(oos_ret=0.02, oos_bh_ret=0.0,
                      oos_segments=3, oos_excess_segments=[0.2, -0.1, -0.3],
                      oos_excess_positive_segments=1)
    ok2, reason2 = _strategy_deploy_gate(res_rej)
    assert not ok2 and "需 ≥2 段" in reason2, f"实际 {reason2!r}"
