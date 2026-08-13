"""风控防线回归测试：NaN/Infinity 拒绝 + 平仓豁免冷却 + 止损放行。

覆盖 2026-08-13 修复：JSON NaN/Infinity 绕过钳制（风控被拆解）、
冷却规则拦截平仓/止损（亏损不可控）、单笔亏损检查拦截止损单。
"""
import asyncio
import math

import pytest

from engine.risk import DEFAULT_RULES, RiskManager
from strategies.base import Signal


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
