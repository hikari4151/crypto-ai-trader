"""TDD：AI 数学期望信号门（2026-09-22，§19 落地）。

背景：AI 行情分析 `_math.expected_value_pct` 是可信量化护栏——实测 ETH/5m
期望 -0.16%（负），即使 LLM 方向看空也不应入场。落地：buy 信号在风控通过后
再经"数学期望门"——期望 < signal_math_ev_min（默认 0.0）则拦截新开多；
sell/止损/减仓不受影响（风控必须放行），关闭开关则全部放行。
"""
import pytest

from ai.math_gate import math_ev_gate_decision


def test_buy_negative_ev_denied():
    allow, reason = math_ev_gate_decision("buy", -0.0016, enabled=True, min_ev=0.0)
    assert allow is False
    assert "期望" in reason


def test_buy_positive_ev_allowed():
    allow, _ = math_ev_gate_decision("buy", 0.01, enabled=True, min_ev=0.0)
    assert allow is True


def test_buy_none_ev_allowed():
    """无数学结论（数据不足）不拦截——保守放行，不让数据缺失阻断交易。"""
    allow, _ = math_ev_gate_decision("buy", None, enabled=True, min_ev=0.0)
    assert allow is True


def test_sell_never_denied():
    """卖出/止损/减仓必须放行（负期望的做空也属风控自由，不做交易方向审查）。"""
    allow, _ = math_ev_gate_decision("sell", -0.5, enabled=True, min_ev=0.0)
    assert allow is True


def test_disabled_allows():
    allow, _ = math_ev_gate_decision("buy", -0.5, enabled=False, min_ev=0.0)
    assert allow is True


def test_settings_defaults():
    from config.settings import settings
    assert settings.signal_math_ev_gate is True
    assert settings.signal_math_ev_min == 0.0


@pytest.mark.asyncio
async def test_execute_signal_blocks_buy_when_ev_negative(monkeypatch):
    """引擎 _execute_signal：风控通过但数学期望为负时 buy 信号被拦截（RISK_BLOCKED，
    不发布 SIGNAL）。"""
    from engine.trading_engine import TradingEngine
    from core.bus import EventBus
    from strategies.base import Signal

    class FakeDB:
        def __init__(self):
            self.kv = {}

        async def kv_get(self, key, default=None):
            return self.kv.get(key, default)

        async def kv_set(self, key, value, is_secret=False):
            self.kv[key] = value

    eng = TradingEngine(FakeDB(), EventBus())
    events = []

    async def _publish(e):
        events.append(e)
    monkeypatch.setattr(eng.bus, "publish", _publish)

    async def _risk_ok(*a, **k):
        return True, ""
    monkeypatch.setattr(eng.risk, "check", _risk_ok)

    async def _snap():
        # 非空快照（engine 的 `if _snap else {}` 对空 dict 视为无快照）
        return {"symbol": "ETH/USDT", "closes": [2700.0, 2701.0, 2690.0],
                "candles": [[1, 2, 3, 4, 5]] * 3, "indicators": {}}
    monkeypatch.setattr(eng, "_current_snapshot", _snap)

    monkeypatch.setattr("ai.math_analysis.math_analysis",
                        lambda snap: {"expected_value_pct": -0.0016})
    import ai.math_analysis as _ma
    assert _ma.math_analysis({}).get("expected_value_pct") == -0.0016, \
        "monkeypatch 未生效"
    monkeypatch.setattr(eng.order_manager, "place",
                        lambda *a, **k: None)

    sig = Signal("ETH/USDT", "buy")
    sig.limit_price = 2700.0
    sig.reason = "t"
    await eng._execute_signal(sig, 2700.0, 1.0)
    kinds = [getattr(e, "type", None) for e in events]
    from core.events import EventType
    assert EventType.RISK_BLOCKED in kinds, f"应发布 RISK_BLOCKED，实际 {kinds}"
    assert EventType.SIGNAL not in kinds, "数学期望为负的 buy 不应进入下单"


# ---------- P1-① 信号门分析缓存（2026-09-22） ----------


def test_math_ev_cache_reuses_within_same_candle():
    """同一根 K 线（相同时间戳）：缓存命中，compute 只调一次。"""
    from ai.math_gate import math_ev_cached
    calls = {"n": 0}

    def compute(snap):
        calls["n"] += 1
        return {"expected_value_pct": -0.0016}

    cache = {}
    snap = {"candles": [[1, 2, 3, 4, 5, 1000]]}
    ev1, hit1 = math_ev_cached(snap, compute, cache)
    ev2, hit2 = math_ev_cached(snap, compute, cache)
    assert (ev1, hit1) == (-0.0016, False)
    assert (ev2, hit2) == (-0.0016, True)
    assert calls["n"] == 1


def test_math_ev_cache_invalidates_on_new_candle():
    """新 K 线（时间戳变化）→ 重新计算，缓存更新。"""
    from ai.math_gate import math_ev_cached
    calls = {"n": 0}

    def compute(snap):
        calls["n"] += 1
        return {"expected_value_pct": -0.0016 * calls["n"]}

    cache = {}
    math_ev_cached({"candles": [[1, 2, 3, 4, 5, 1000]]}, compute, cache)
    ev2, hit2 = math_ev_cached({"candles": [[1, 2, 3, 4, 5, 1001]]}, compute, cache)
    assert hit2 is False
    assert calls["n"] == 2


def test_math_ev_cache_no_candles_does_not_cache():
    """无 candles（无时间戳）→ 不缓存，每次都计算不入缓存。"""
    from ai.math_gate import math_ev_cached
    calls = {"n": 0}

    def compute(snap):
        calls["n"] += 1
        return {"expected_value_pct": 0.01}

    cache = {}
    ev1, hit1 = math_ev_cached({}, compute, cache)
    ev2, hit2 = math_ev_cached({}, compute, cache)
    assert (ev1, hit1) == (0.01, False)
    assert (ev2, hit2) == (0.01, False)
    assert calls["n"] == 2
    assert cache == {}