"""风控防线回归测试：NaN/Infinity 拒绝 + 平仓豁免冷却 + 止损放行 + 每日盈亏跨日。

覆盖 2026-08-13 修复：JSON NaN/Infinity 绕过钳制（风控被拆解）、
冷却规则拦截平仓/止损（亏损不可控）、单笔亏损检查拦截止损单。
覆盖 2026-08-28 修复：每日盈亏只在有成交时归档 + 引擎自持副本 → 每日亏损
熔断触发后跨日不解封（次日仍被昨日亏损拦住，再也开不了仓）。
"""
import asyncio
import math
import time as _time

import pytest

import engine.risk as risk_mod
from engine.risk import DEFAULT_RULES, RiskManager
from strategies.base import Signal


class FakeClock:
    """替换 engine.risk 的 time 引用：strftime 可控，time 走真实时钟。"""

    def __init__(self, today: str) -> None:
        self.today = today

    def strftime(self, fmt: str) -> str:
        return self.today if fmt == "%Y-%m-%d" else _time.strftime(fmt)

    def time(self) -> float:
        return _time.time()


class FakeDB:
    """内存 KV，模拟 Database 的 kv 接口（无需真 SQLite）。"""

    def __init__(self):
        self.kv = {}

    async def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    async def kv_set(self, key, value, is_secret=False):
        self.kv[key] = value

    async def kv_json_get(self, key, default=None):
        import json
        raw = self.kv.get(key)
        if raw is None:
            return default
        return json.loads(raw)

    async def kv_json_set(self, key, value):
        import json
        self.kv[key] = json.dumps(value, ensure_ascii=False)


def make_rm() -> RiskManager:
    return RiskManager(FakeDB())


def test_nan_rules_rejected():
    """NaN 规则值必须被拒绝并保持默认（曾绕过钳制把每日亏损上限推到 100 万）。"""
    rm = make_rm()

    async def run():
        rules = await rm.update_rules({
            "max_daily_loss_usd": float("nan"),
            "max_loss_per_trade_usd": float("inf"),
            "flash_crash_5min_drop_pct": float("nan"),
            "max_vol_ratio": float("inf"),
        })
        return rules

    rules = asyncio.run(run())
    assert rules["max_daily_loss_usd"] == DEFAULT_RULES["max_daily_loss_usd"]
    assert rules["max_loss_per_trade_usd"] == DEFAULT_RULES["max_loss_per_trade_usd"]
    assert rules["flash_crash_5min_drop_pct"] == DEFAULT_RULES["flash_crash_5min_drop_pct"]
    assert rules["max_vol_ratio"] == DEFAULT_RULES["max_vol_ratio"]


def test_finite_rules_accepted():
    """正常有限值仍可更新（钳制到边界内）。"""
    rm = make_rm()

    async def run():
        return await rm.update_rules({"max_daily_loss_usd": 123.0, "max_position_pct": 5.0})

    rules = asyncio.run(run())
    assert rules["max_daily_loss_usd"] == 123.0
    assert rules["max_position_pct"] == 1.0  # 超上界被钳制


def test_cooldown_blocks_open_but_allows_close():
    """连续亏损冷却：拦截新开仓，放行平仓（止损必须可执行）。"""
    rm = make_rm()

    async def run():
        await rm.update_rules({"max_consecutive_losses": 1, "cooldown_minutes": 1})
        for _ in range(2):
            await rm.record_trade(-10.0, "test")
        ok_open, why_open = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        ok_close, why_close = await rm.check(
            Signal("BTC/USDT", "sell", qty=1.0), 100.0, 0.0, 1000.0, 0.0, [], 100.0)
        return ok_open, why_open, ok_close, why_close

    ok_open, why_open, ok_close, _ = asyncio.run(run())
    assert not ok_open
    assert "冷却" in why_open
    assert ok_close  # 平仓豁免冷却


def test_stop_loss_not_blocked_by_per_trade_limit():
    """单笔亏损上限不拦截止损单（拦截止损=拒绝唯一能封顶亏损的订单）。"""
    rm = make_rm()

    async def run():
        await rm.update_rules({"max_loss_per_trade_usd": 10.0})
        return await rm.check(
            Signal("BTC/USDT", "sell", qty=1.0, reason="止损 -5.00%"),
            90.0, 0.0, 1000.0, 0.0, [], 100.0)

    ok, _ = asyncio.run(run())
    assert ok  # 预估亏损 10 美元恰在上限，放行（即使超限也放行+告警）


def test_daily_loss_blocks_open_but_allows_close():
    """每日亏损上限：拦截新开仓，放行平仓。"""
    rm = make_rm()

    async def run():
        await rm.update_rules({"max_daily_loss_usd": 200.0})
        ok_open, _ = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, -250.0, [], None)
        ok_close, _ = await rm.check(
            Signal("BTC/USDT", "sell", qty=1.0), 100.0, 0.0, 1000.0, -250.0, [], 100.0)
        return ok_open, ok_close

    ok_open, ok_close = asyncio.run(run())
    assert not ok_open
    assert ok_close


def test_daily_loss_breaker_clears_on_new_day(monkeypatch):
    """每日亏损熔断跨日自动解除：get_daily_pnl 必须自己归档归零。

    曾两处失效叠加：归档只发生在 add_daily_pnl（有成交时），且引擎把值复制
    成 self._daily_pnl 只增不减——隔天后 check() 仍拿昨日亏损，熔断到重启前
    再也不放行新单。
    """
    clock = FakeClock("2026-08-28")
    monkeypatch.setattr(risk_mod, "time", clock)
    db = FakeDB()
    rm = RiskManager(db)

    async def run():
        await rm.update_rules({"max_daily_loss_usd": 200.0})
        await rm.record_trade(-250.0, "止损")
        assert await rm.get_daily_pnl() == pytest.approx(-250.0)

        blocked, why = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0,
            await rm.get_daily_pnl(), [], None)
        assert not blocked and "每日最大亏损" in why

        # 日历跨日且当天尚无成交：仍应归零（归档只在 add_daily_pnl 做时此处为 -250）
        clock.today = "2026-08-29"
        assert await rm.get_daily_pnl() == 0.0
        # 昨日累计归档，不丢历史
        assert db.kv["daily_pnl_2026-08-28"] == "2026-08-28|-250.0000"

        allowed, why2 = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0,
            await rm.get_daily_pnl(), [], None)
        return blocked, allowed, why2

    blocked, allowed, why2 = asyncio.run(run())
    assert not blocked
    assert allowed, f"跨日后应放行新开仓，实际: {why2}"


def test_daily_pnl_accumulates_after_rollover(monkeypatch):
    """跨日归零后新交易继续按今日累计（归档不吞掉同一笔 pnl）。"""
    clock = FakeClock("2026-08-28")
    monkeypatch.setattr(risk_mod, "time", clock)
    db = FakeDB()
    rm = RiskManager(db)

    async def run():
        await rm.record_trade(-30.0, "止损")
        clock.today = "2026-08-29"
        await rm.record_trade(-40.0, "止损")
        return await rm.get_daily_pnl()

    assert asyncio.run(run()) == pytest.approx(-40.0)
    assert db.kv["daily_pnl_2026-08-29"] == "2026-08-29|-40.0000"
