"""TDD：GAP-01 验收判定（诊断报告建议第6条(b)——接进进化引擎）。

口径（docs/LIVE_READINESS.md GAP-01 + E7 单标的豁免）：
- 主判据 = 同品种多折 walk-forward：≥75% 折为正、中位数>0、每折成交≥min_trades；
- 成本鲁棒 = 成本翻倍（fee/slippage × gap01_cost_multiplier）后仍满足主判据；
- PASS：基准与成本翻倍均 PASS；INCONCLUSIVE：任一档样本不足；其余 FAIL。
- 跨品种按单标的部署策略豁免（E7），本判定不含跨品种门禁。
"""
import pytest

import pandas as pd

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from drl.gap01 import summarize_gap01
from tests.test_drl_optimizations import _MetaDB


def _folds_summary(verdict, n=4, pos=3, med=0.01, trades_ok=True):
    return {
        "verdict": verdict, "n_folds": n, "positive_folds": pos,
        "win_rate": pos / n, "median_ret": med, "mean_ret": med,
        "reason": "test",
    }


def test_gap01_pass_requires_both_cost_tiers_pass():
    base = _folds_summary("PASS")
    assert summarize_gap01(base, _folds_summary("PASS"))["verdict"] == "PASS"


def test_gap01_fail_when_cost_doubled_fails():
    """基准过、成本翻倍不过 → FAIL（成本鲁棒性不满足）。"""
    base = _folds_summary("PASS")
    out = summarize_gap01(base, _folds_summary("FAIL", med=-0.02))
    assert out["verdict"] == "FAIL"
    assert "成本" in out["reason"]


def test_gap01_fail_when_base_fails():
    out = summarize_gap01(_folds_summary("FAIL"), _folds_summary("PASS"))
    assert out["verdict"] == "FAIL"


def test_gap01_inconclusive_when_samples_insufficient():
    """任一档样本不足（INCONCLUSIVE）→ 整体 INCONCLUSIVE，不误判 PASS/FAIL。"""
    base = _folds_summary("INCONCLUSIVE")
    out = summarize_gap01(base, _folds_summary("PASS"))
    assert out["verdict"] == "INCONCLUSIVE"


# ---------- 引擎观测方法 ----------

@pytest.fixture
def engine(tmp_path):
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


@pytest.mark.asyncio
async def test_observe_gap01_skips_without_model(tmp_path, engine):
    """无 strategy_drl 模型时静默跳过，不写 gap01_obs.json。"""
    from backtest.data_loader import generate_demo
    df = generate_demo(n=1200, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    out = await engine._observe_gap01(df=df)
    assert out is None
    assert not (tmp_path / "models" / "gap01_obs.json").exists()


@pytest.mark.asyncio
async def test_observe_gap01_respects_disable(tmp_path, engine, monkeypatch):
    """开关关闭时静默跳过。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_gap01_obs", False, raising=False)
    from backtest.data_loader import generate_demo
    df = generate_demo(n=1200, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    out = await engine._observe_gap01(df=df)
    assert out is None
    assert not (tmp_path / "models" / "gap01_obs.json").exists()


# ---------- GAP-01 FAIL → from-scratch 重学闭环 ----------


def _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, verdict):
    from config.settings import settings
    from backtest.data_loader import generate_demo
    import drl.evolve_engine as ev
    # 软信号测试只测 GAP-01 闭环：关掉冠军/挑战者交替，每轮都是冠军轮
    monkeypatch.setattr(settings, "evolve_strategy_drl_population", False,
                        raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)
    fetch_calls = 0

    async def _fake_fetch(symbol: str = ""):
        nonlocal fetch_calls
        fetch_calls += 1
        out = df.copy()
        # 增量门按尾时间戳判重：每轮让尾戳前进，保证三轮都真实训练
        out.index = df.index + pd.Timedelta(minutes=5 * fetch_calls)
        return out
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    # 本测试只验证 GAP-01 软信号：增量门恒过，避免尾戳/缓存干扰三轮都训练
    monkeypatch.setattr(eng, "_data_incremented",
                        lambda *a, **k: True)

    async def _fake_gap01(df=None):
        return {"verdict": verdict}
    monkeypatch.setattr(eng, "_observe_gap01", _fake_gap01)
    fake_best = _agent()
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_best)
    # 稳定部署版本（多窗口 PASS 判定按版本累计；版本变化测试另行覆盖）
    monkeypatch.setattr(eng.zoo, "best_info",
                        lambda name: {"version": 1, "fitness": 0.0})
    captured = []

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        captured.append(cfg_.get("base_agent"))
        return {"history": [{"best_ret": 0.01}], "agent": _agent(),
                "oos_report": {"enabled": False}}
    monkeypatch.setattr(ev, "train_drl", _fake_train)
    return eng, captured


def _agent():
    from drl.agent import ACAgent
    return ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)


@pytest.mark.asyncio
async def test_gap01_fail_triggers_from_scratch_after_threshold(tmp_path, monkeypatch):
    """连续 GAP-01 FAIL 达阈值 → 冠军轮 from-scratch（base_agent=None）。"""
    from config.settings import settings
    eng, captured = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "FAIL")
    monkeypatch.setattr(settings, "evolve_gap01_fail_reset_after", 3, raising=False)
    eng._strategy_drl_status["_pop_round"] = 0  # 冠军轮
    for _ in range(3):
        await eng._train_strategy_drl_once(force=False)
    # 第3轮达阈值（streak=3）触发并清零；前两轮续训、第三轮 from-scratch
    assert eng._gap01_streak.get("strategy_drl") == 0, "触发后应清零给续训窗口"
    assert eng._strategy_drl_status.get("gap01_streak") == 0
    assert captured[0] is not None and captured[1] is not None
    assert captured[2] is None, "连续3轮 GAP-01 FAIL 后应 from-scratch 重训"


@pytest.mark.asyncio
async def test_gap01_pass_clears_streak(tmp_path, monkeypatch):
    """GAP-01 PASS 清零连续 FAIL 计数。"""
    from config.settings import settings
    eng, captured = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(settings, "evolve_gap01_fail_reset_after", 3, raising=False)
    eng._gap01_streak["strategy_drl"] = 2
    eng._strategy_drl_status["_pop_round"] = 0
    await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_streak.get("strategy_drl") == 0
    assert eng._strategy_drl_status.get("gap01_verdict") == "PASS"


@pytest.mark.asyncio
async def test_gap01_reset_disabled_keeps_continuation(tmp_path, monkeypatch):
    """evolve_gap01_fail_reset_after=0（关闭）：streak 照记但不触发 from-scratch。"""
    from config.settings import settings
    eng, captured = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "FAIL")
    monkeypatch.setattr(settings, "evolve_gap01_fail_reset_after", 0, raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(3):
        await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_streak.get("strategy_drl") == 3
    assert all(a is not None for a in captured), "关闭时仍应续训"


@pytest.mark.asyncio
async def test_gap01_inconclusive_also_accumulates_streak(tmp_path, monkeypatch):
    """INCONCLUSIVE（样本不足/不交易）同样累计 streak。

    运行态实录（2026-09-20）：from-scratch 重训后的模型不交易 → GAP-01 判
    INCONCLUSIVE；若不计入 streak，软信号会被"不交易"卡死（gap01_streak 停在
    2 永远到不了阈值）。拿不出样本外证据=不能上线，连续如此就该触发重学。
    """
    from config.settings import settings
    eng, captured = _make_strategy_engine_with_fake_obs(
        tmp_path, monkeypatch, "INCONCLUSIVE")
    monkeypatch.setattr(settings, "evolve_gap01_fail_reset_after", 2, raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(2):
        await eng._train_strategy_drl_once(force=False)
    # 第2轮达阈值（streak=2）触发并清零
    assert eng._gap01_streak.get("strategy_drl") == 0, "触发后应清零给续训窗口"
    assert captured[1] is None, "INCONCLUSIVE 连续达阈值也应触发 from-scratch"


@pytest.mark.asyncio
async def test_gap01_trigger_clears_streak_for_retry_window(tmp_path, monkeypatch):
    """触发 from-scratch 后 streak 清零：给新起点续训窗口，避免每轮都重置。

    运行态实录（2026-09-20）：11:00:43 触发后 11:00:47（下一轮）又触发
    （连续4轮）——触发后不清零会让模型没有续训机会，每轮都在 from-scratch 空转。
    """
    from config.settings import settings
    eng, captured = _make_strategy_engine_with_fake_obs(
        tmp_path, monkeypatch, "FAIL")
    monkeypatch.setattr(settings, "evolve_gap01_fail_reset_after", 2, raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(3):
        await eng._train_strategy_drl_once(force=False)
    # 第2轮达阈值（streak=2）触发并清零；第3轮重新累计 streak=1，不再触发
    assert eng._gap01_streak.get("strategy_drl") == 1
    assert captured[1] is None, "第2轮达阈值应触发 from-scratch"
    assert captured[2] is not None, "触发后应给续训窗口（第3轮不再重置）"


# ---------- 多窗口连续 PASS 判定（§16.5 遗留1，2026-09-21） ----------


@pytest.mark.asyncio
async def test_gap01_pass_streak_accumulates_and_stable(tmp_path, monkeypatch):
    """同一部署模型连续 N 次 PASS → gap01_stable=True（实盘候选判定）。"""
    from config.settings import settings
    eng, _ = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(settings, "evolve_gap01_stable_pass_required", 3,
                        raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(3):
        await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_pass_streak == 3
    assert eng._strategy_drl_status.get("gap01_pass_streak") == 3
    assert eng._strategy_drl_status.get("gap01_stable") is True


@pytest.mark.asyncio
async def test_gap01_pass_streak_resets_on_fail(tmp_path, monkeypatch):
    """连续 PASS 被 FAIL 打断 → pass_streak 清零、stable=False。"""
    from config.settings import settings
    eng, _ = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(settings, "evolve_gap01_stable_pass_required", 3,
                        raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(2):
        await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_pass_streak == 2
    # 第三轮观测变 FAIL
    async def _fake_gap01_fail(df=None):
        return {"verdict": "FAIL"}
    monkeypatch.setattr(eng, "_observe_gap01", _fake_gap01_fail)
    await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_pass_streak == 0
    assert eng._strategy_drl_status.get("gap01_stable") is False


@pytest.mark.asyncio
async def test_gap01_pass_streak_resets_on_version_change(tmp_path, monkeypatch):
    """部署版本变化（新模型）→ pass_streak 从 0 重新累计。"""
    from config.settings import settings
    eng, _ = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(settings, "evolve_gap01_stable_pass_required", 3,
                        raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_pass_streak == 1
    # 部署版本从 1 变到 2
    monkeypatch.setattr(eng.zoo, "best_info",
                        lambda name: {"version": 2, "fitness": 0.0})
    await eng._train_strategy_drl_once(force=False)
    assert eng._gap01_pass_streak == 1, "版本变化后应从 0 重新累计（本次PASS后=1）"


@pytest.mark.asyncio
async def test_gap01_stable_disabled_when_required_zero(tmp_path, monkeypatch):
    """evolve_gap01_stable_pass_required=0：不启用稳定判定（stable=False）。"""
    from config.settings import settings
    eng, _ = _make_strategy_engine_with_fake_obs(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(settings, "evolve_gap01_stable_pass_required", 0,
                        raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    for _ in range(3):
        await eng._train_strategy_drl_once(force=False)
    assert eng._strategy_drl_status.get("gap01_stable") is False
