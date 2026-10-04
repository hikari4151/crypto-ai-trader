"""TDD：strategy_drl 连续拒绝 from-scratch 重置（对齐 factor_miner）。

背景（诊断报告 §7 建议第 2 条）：factor_miner 已有
evolve_factor_miner_reset_after_rejects（连续 OOS 被拒达阈值→下一轮强制从零重训，
打破"永久连拒"）；strategy_drl 缺失同款机制，锚点锁死时只能无限续训。本改动补上，
并保证：
- 策略 DRL 只在冠军轮触发（挑战者轮有独立训练基线，不清除）；
- 触发后 base_agent=None（from-scratch），未触发保持续训起点。

注：meta_controller 的同款用例已随元策略训练功能移除（2026-09-28）。
"""
import pytest

from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine, _should_from_scratch_reset
from tests.test_drl_optimizations import _MetaDB


def _agent():
    return ACAgent(state_dim=4, n_actions=3, hidden=(8, 8), seed=7)


def test_should_from_scratch_reset_pure():
    """阈值语义：0=关闭；未达阈值不触发；达到/超过触发。"""
    assert _should_from_scratch_reset(50, 0) is False
    assert _should_from_scratch_reset(4, 5) is False
    assert _should_from_scratch_reset(5, 5) is True
    assert _should_from_scratch_reset(10, 5) is True


def _fake_result(oos_enabled=False):
    return {
        "agent": _agent(),
        "history": [{"best_ret": 0.01}],
        "best_ret": 0.01,
        "oos_report": {"enabled": oos_enabled, "oos_ret": 0.01} if oos_enabled else {"enabled": False},
        "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
        "factor_composite": None, "pine_code": "", "pine_note": "",
    }


def _make_strategy_engine(tmp_path, monkeypatch):
    from backtest.data_loader import generate_demo
    from config.settings import settings
    # 本组测试只验证轮次/重置/连拒逻辑：把多 seed 钉到 1，避免
    # evolve_strategy_drl_seeds=3 让桩训练被调用 3 次、计数断言失真
    monkeypatch.setattr(settings, "evolve_strategy_drl_seeds", 1, raising=False)
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="5m", n=500, seed=42)

    async def _fake_fetch(symbol: str = "", bars: int = 0):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    fake_best = _agent()
    monkeypatch.setattr(eng.zoo, "load_agent",
                        lambda name, version=None: fake_best)
    return eng, fake_best


@pytest.mark.asyncio
async def test_strategy_champion_round_resets_to_scratch_on_reject_streak(tmp_path, monkeypatch):
    """冠军轮连续拒绝达阈值 → from-scratch（base_agent=None）。"""
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 10
    eng._strategy_drl_status["_pop_round"] = 0  # 冠军轮
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is None, "冠军轮达阈值应 from-scratch 重训"


@pytest.mark.asyncio
async def test_strategy_champion_round_keeps_best_below_threshold(tmp_path, monkeypatch):
    """未达阈值 → 继续从 best 续训（base_agent 非 None）。"""
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 2
    eng._strategy_drl_status["_pop_round"] = 0
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is fake_best, "未达阈值应续训 best"


@pytest.mark.asyncio
async def test_reject_trigger_clears_streak(tmp_path, monkeypatch):
    """触发 from-scratch 后必须清零连拒计数（否则每轮都从零重训）。

    回归防护（2026-09-28）：reject 分支原先把 best_agent 置 None 却**不清零**
    连拒计数（只有 gap01 分支清零），而 _reject_streak 仅在 status=="ok" 或
    手动重置锚点时归零。strategy_drl 的 from-scratch 过门率为 0（REPORT §22：
    生产窗口复刻 16 次全不过门）→ 计数恒 >= 阈值 → **每一轮都是 base_agent=None**，
    累积学习被反复丢弃，表现为灾难性遗忘（现场实测 reject_streak=19 > 阈值 5）。

    期望：触发时清零到 0，本轮被拒再 +1 = 1（修复前为 10+1=11）。
    """
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, _ = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 10
    eng._strategy_drl_status["_pop_round"] = 0  # 冠军轮
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None, extra_dfs=None: _fake_result())

    await eng._train_strategy_drl_once(force=False)
    assert eng._reject_streak["strategy_drl"] == 1, (
        "触发后应清零再被本轮 +1；若为 11 说明清零缺失，连拒再也不会回落")


@pytest.mark.asyncio
async def test_round_after_reset_resumes_from_best(tmp_path, monkeypatch):
    """触发轮之后的一轮应恢复续训（给出 N 轮续训窗口，而非继续从零）。

    这是"清零"的实际收益：from-scratch 只在达阈值时发生一次，其后回到
    "best 模型 + 8ep 增量"的累积学习路径。
    """
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._strategy_drl_status["_pop_round"] = 0
    seen = []

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        seen.append(cfg_.get("base_agent"))
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    # 第 1 轮：连拒达阈值 → 从零。force=True 跳过"新增根数"增量门槛
    #（mock fetch 返回同一 df，尾戳不变，force=False 时第二轮会被门槛跳过）
    eng._reject_streak["strategy_drl"] = 7
    await eng._train_strategy_drl_once(force=True)
    # 第 2 轮：清零后计数为 1（< 阈值）→ 必须续训
    await eng._train_strategy_drl_once(force=True)

    assert len(seen) == 2, f"两轮都应进入训练，实际 {len(seen)} 轮"
    assert seen[0] is None, "第 1 轮达阈值应从零重训"
    assert seen[1] is fake_best, (
        "第 2 轮应恢复续训；若仍为 None 说明计数未清零，陷入每轮从零")


@pytest.mark.asyncio
async def test_replay_buffer_rebuilt_on_symbol_switch(tmp_path, monkeypatch):
    """单标模式（multi_symbol 关闭）：符号轮换必须重建回放缓冲区。

    同一标的的连续轮之间复用（累积历史，对抗遗忘）；一旦轮换标的，
    旧标的轨迹会混入本轮策略更新——按符号重建更保守。
    注意：多标模式（默认开启）不重建——训练本体已是跨标混合采样，
    按主标的重建会把缓冲每轮清空（见 test_replay_buffer_shared_across_symbols_multi）。
    """
    from config.settings import settings
    from drl.agent import ReplayBuffer
    import drl.evolve_engine as ev
    monkeypatch.setattr(settings, "evolve_strategy_drl_multi_symbol", False,
                        raising=False)
    eng, _ = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None, extra_dfs=None: _fake_result())

    syms = iter(["ETH/USDT", "ETH/USDT", "SOL/USDT"])
    monkeypatch.setattr(eng, "_next_symbol", lambda: next(syms))

    await eng._train_strategy_drl_once(force=True)
    buf_eth = eng._strategy_drl_replay
    assert isinstance(buf_eth, ReplayBuffer)
    assert eng._strategy_drl_replay_symbol == "ETH/USDT"
    # 模拟上一轮留下的轨迹
    import numpy as np
    buf_eth.push(np.zeros((5, 4)), np.zeros(5, dtype=int), np.zeros(5), np.zeros(5))

    # 同标的下一轮：复用同一实例，历史保留
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_replay is buf_eth, "同标的应复用同一缓冲区"
    assert len(eng._strategy_drl_replay) == 1, "同标的应保留历史轨迹"

    # 轮换标的：重建（历史清空）
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_replay_symbol == "SOL/USDT"
    assert eng._strategy_drl_replay is not buf_eth, "轮换标的应重建缓冲区"
    assert len(eng._strategy_drl_replay) == 0, "新标的缓冲区不应含旧标的轨迹"


@pytest.mark.asyncio
async def test_replay_buffer_shared_across_symbols_multi(tmp_path, monkeypatch):
    """多标模式（默认）：主标轮换不重建回放缓冲区（跨标经验持续累积）。

    P0 多标的混合训练：训练本体是跨标混合采样，缓冲保留全部标的的轨迹
    正是"混合经验"的一部分；若按主标的重建，主标每轮轮换 → 缓冲区每轮
    清空，跨轮经验（对抗遗忘的关键）被丢掉，与多标混合的目的相悖。
    """
    from config.settings import settings
    from drl.agent import ReplayBuffer
    import drl.evolve_engine as ev
    monkeypatch.setattr(settings, "evolve_strategy_drl_multi_symbol", True,
                        raising=False)
    eng, _ = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None, extra_dfs=None: _fake_result())

    syms = iter(["ETH/USDT", "SOL/USDT", "BTC/USDT"])
    monkeypatch.setattr(eng, "_next_symbol", lambda: next(syms))

    await eng._train_strategy_drl_once(force=True)
    buf_1 = eng._strategy_drl_replay
    assert isinstance(buf_1, ReplayBuffer)
    import numpy as np
    buf_1.push(np.zeros((5, 4)), np.zeros(5, dtype=int), np.zeros(5), np.zeros(5))

    # 主标轮换（SOL）：多标模式不重建 → 同一实例、历史保留
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_replay is buf_1, "多标模式主标轮换不应重建缓冲区"
    assert len(eng._strategy_drl_replay) == 1, "跨标轨迹应保留（混合经验）"

    # 再轮换（BTC）：仍复用
    await eng._train_strategy_drl_once(force=True)
    assert eng._strategy_drl_replay is buf_1, "多标模式跨标轮换应持续复用"


@pytest.mark.asyncio
async def test_challenger_round_reset_does_not_clear_streak(tmp_path, monkeypatch):
    """挑战者轮不参与该重置：既不清基线，也不清零计数（保持冠军轮阈值语义）。

    2026-10 审计修复（MED）：挑战者轮**不递增** _reject_streak——challenger_kept/
    degraded 是"晋级失败被冠军压住"的健康交替，计入 streak 会让选模合格的冠军约每
    5 个挑战者轮被强制 from-scratch（丢经验）。旧断言 ==11（递增）锁的正是这个 bug。
    """
    from config.settings import settings
    import drl.evolve_engine as ev
    eng, fake_best = _make_strategy_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "evolve_strategy_drl_reset_after_rejects", 5,
                        raising=False)
    eng._reject_streak["strategy_drl"] = 10
    eng._strategy_drl_status["_pop_round"] = 1  # 挑战者轮
    captured = {}

    def _fake_train(df_, cfg_, on_progress=None, extra_dfs=None):
        captured["base_agent"] = cfg_.get("base_agent")
        return _fake_result()
    monkeypatch.setattr(ev, "train_drl", _fake_train)

    await eng._train_strategy_drl_once(force=False)
    assert captured.get("base_agent") is fake_best, "挑战者轮不应被 from-scratch 重置"
    assert eng._reject_streak["strategy_drl"] == 10, "挑战者轮不递增（健康交替不累积）"


