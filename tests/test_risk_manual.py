"""T3 熔断人工确认恢复测试：cooldown_status 新增字段 + clear_cooldown 端点。

覆盖：
- cooldown_status 返回 manual_recovery / awaiting_clear 键
- manual_recovery=0 时 await_clear 恒 False（回归）
- manual_recovery=1 且冷却到期 → awaiting_clear=True，check 仍拦截
- clear_cooldown 后 check 放行
- 默认配置（risk_manual_recovery=0）冷却到期自动恢复（现有行为不变）
"""
import asyncio
import time

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


def test_cooldown_status_has_new_keys():
    """cooldown_status 返回 manual_recovery / awaiting_clear 键。"""
    rm = make_rm()
    status = rm.cooldown_status()
    assert "manual_recovery" in status
    assert "awaiting_clear" in status
    # 默认值
    assert status["manual_recovery"] is False
    assert status["awaiting_clear"] is False
    # 现有键不变
    assert "active" in status
    assert "remaining_sec" in status
    assert "consecutive_losses" in status


def test_manual_recovery_off_by_default():
    """默认 risk_manual_recovery=0 时，awaiting_clear 恒 False。"""
    rm = make_rm()
    status = rm.cooldown_status()
    assert status["awaiting_clear"] is False


def test_clear_cooldown_exists():
    """RiskManager 有 clear_cooldown 方法。"""
    rm = make_rm()
    assert hasattr(rm, "clear_cooldown")
    assert callable(rm.clear_cooldown)


def test_clear_cooldown_resets():
    """clear_cooldown 后 cooldown_until=0 且 manual_recovery=False。"""
    rm = make_rm()
    rm._cooldown_until = time.time() + 100  # 模拟冷却中
    rm._manual_recovery = True
    rm.clear_cooldown()
    assert rm._cooldown_until == 0.0
    assert rm._manual_recovery is False


async def test_manual_recovery_blocks_after_cooldown_expires():
    """manual_recovery=1 且冷却到期后 awaiting_clear=True，check 仍拦截新开仓。"""
    rm = make_rm()

    async def run():
        # 设置 manual_recovery 规则
        await rm.update_rules({
            "max_consecutive_losses": 1,
            "cooldown_minutes": 1,
            "risk_manual_recovery": 1,
        })
        # 触发冷却：连亏达标后需 check() 才会真正进入冷却
        for _ in range(2):
            await rm.record_trade(-10.0, "test")
        ok_first, why_first = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        assert not ok_first and "触发冷却" in why_first
        # 冷却已触发，_cooldown_until > now，manual_recovery 标志应置位
        status = rm.cooldown_status()
        assert status["active"] is True
        assert status["manual_recovery"] is True
        # 模拟冷却到期：把 _cooldown_until 设为过去
        rm._cooldown_until = time.time() - 1
        # 冷却到期后但 manual_recovery=True → awaiting_clear=True
        status2 = rm.cooldown_status()
        assert status2["awaiting_clear"] is True, "冷却到期后 awaiting_clear 应为 True"
        assert status2["active"] is True, "awaiting_clear 时 active 应为 True"

        # 新开仓应被拦截
        ok_open, why_open = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        assert not ok_open, "awaiting_clear 时应拦截新开仓"
        assert "人工确认" in why_open or "冷却" in why_open

        # 平仓应放行（平仓豁免）
        ok_close, _ = await rm.check(
            Signal("BTC/USDT", "sell", qty=1.0), 100.0, 0.0, 1000.0, 0.0, [], 100.0)
        assert ok_close, "awaiting_clear 时平仓仍应放行"

        # 调用 clear_cooldown
        rm.clear_cooldown()
        status3 = rm.cooldown_status()
        assert status3["awaiting_clear"] is False
        assert status3["active"] is False

        # 再次新开仓应放行
        ok_open2, _ = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        assert ok_open2, "clear_cooldown 后应放行新开仓"

    await run()


async def test_default_manual_recovery_auto_recovers():
    """默认配置（risk_manual_recovery=0）awaiting_clear 恒 False，冷卻到期不需人工确认。"""
    rm = make_rm()

    async def run():
        await rm.update_rules({
            "max_consecutive_losses": 1,
            "cooldown_minutes": 1,
        })
        # 触发冷却：连亏达标后需 check() 才会真正进入冷却
        for _ in range(2):
            await rm.record_trade(-10.0, "test")
        ok_first, why_first = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        assert not ok_first and "触发冷却" in why_first
        assert rm._cooldown_until > time.time()
        # 一场盈利交易打破连亏（_consecutive_losses 归零——现有自动恢复的必经路径）
        await rm.record_trade(20.0, "test")
        # 模拟冷却到期
        rm._cooldown_until = time.time() - 1
        status = rm.cooldown_status()
        assert status["awaiting_clear"] is False, "manual_recovery=0 时 awaiting_clear 应为 False"
        assert status["active"] is False, "冷却到期且 manual_recovery=0 时 active 应为 False"

        # 新开仓应放行（自动恢复，无人工确认等待）
        ok_open, _ = await rm.check(
            Signal("BTC/USDT", "buy", qty=0.1), 100.0, 10000.0, 0.0, 0.0, [], None)
        assert ok_open, "manual_recovery=0 冷却到期且连亏打破后应放行新开仓"

    await run()


async def test_risk_manual_recovery_in_rules():
    """risk_manual_recovery 出现在 DEFAULT_RULES 和 update_rules 白名单中。"""
    assert "risk_manual_recovery" in DEFAULT_RULES
    assert DEFAULT_RULES["risk_manual_recovery"] == 0

    rm = make_rm()

    async def run():
        rules = await rm.get_rules()
        assert "risk_manual_recovery" in rules
        assert rules["risk_manual_recovery"] == 0

        # 更新为 1
        updated = await rm.update_rules({"risk_manual_recovery": 1})
        assert updated["risk_manual_recovery"] == 1

        # 再更新为 0（回归）
        updated2 = await rm.update_rules({"risk_manual_recovery": 0})
        assert updated2["risk_manual_recovery"] == 0

    await run()