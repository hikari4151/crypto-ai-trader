"""24h 跌幅检测回归测试：解包错误 + 最小跨度约束 + 平仓豁免。

覆盖 2026-08-14 修复（P0-1）：
- risk.py:330 曾对 float 元素做 prices_24h[-1][0] 下标 → TypeError，
  被 bus 吞掉后"引擎看似运行实则已死"（每次风控检查必炸）
- 修复后：跨度取自 _price_history 元组首尾 ts（保留 48h 口径）；
  check() 顶层 try/except fail-safe（异常=拦截下单，绝不放行新单）
"""
import asyncio
import time

from engine.risk import RiskManager
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


def _seed_history(rm: RiskManager, n: int, span_sec: float, start_price: float, end_price: float):
    """构造均匀下跌的价格历史：n 个样本，时间跨度 span_sec，价格从 start 到 end。"""
    rm._price_history = []
    now = time.time()
    for i in range(n):
        t = now - span_sec + span_sec * i / max(n - 1, 1)
        p = start_price + (end_price - start_price) * i / max(n - 1, 1)
        rm._price_history.append((t, p))


def test_empty_history_allowed():
    """(a) 空价格历史：check() 不抛异常且放行（无 24h 数据可判）。"""
    rm = make_rm()

    async def run():
        return await rm.check(
            Signal("BTC/USDT", "buy", qty=0.2), 100.0, 10000.0, 0.0, 0.0, [], None)

    ok, why = asyncio.run(run())
    assert ok, f"空历史应放行，实际: {why}"


def test_short_span_no_cooldown():
    """(b) 样本 ≥10 但跨度 <2h：跌幅 30% 也不触发 24h 冷却（防刚启动误判）。

    修复前此处抛 TypeError（对 float 下标）——测试即失败；修复后应放行。
    """
    rm = make_rm()
    # 54 分钟跨度、均匀下跌 30%（5 分钟窗口跌幅约 2.5%，不会误触闪崩检测）
    _seed_history(rm, 10, 54 * 60, 100.0, 70.0)

    async def run():
        return await rm.check(
            Signal("BTC/USDT", "buy", qty=0.2), 70.0, 10000.0, 0.0, 0.0, [], None)

    ok, why = asyncio.run(run())
    assert ok, f"跨度不足 2h 不应触发 24h 冷却，实际: {why}"
    assert rm._flash_cooldown_until <= time.time(), "不应进入闪崩冷却"


def test_long_span_drop_triggers_cooldown():
    """(c) 样本跨度 ≥2h 且跌幅 ≥25%：触发冷却并返回原因。"""
    rm = make_rm()
    _seed_history(rm, 10, 3 * 3600, 100.0, 70.0)

    async def run():
        return await rm.check(
            Signal("BTC/USDT", "buy", qty=0.2), 70.0, 10000.0, 0.0, 0.0, [], None)

    ok, why = asyncio.run(run())
    assert not ok, "3 小时跌 30% 应触发 24h 冷却"
    assert "24小时跌幅" in why, f"原因应含 24小时跌幅，实际: {why}"
    assert rm._flash_cooldown_until > time.time(), "应进入闪崩冷却"


def test_closing_exempts_24h_rule():
    """(d) 平仓信号（closing=True）豁免 24h 规则（止损必须可执行）。"""
    rm = make_rm()
    _seed_history(rm, 10, 3 * 3600, 100.0, 70.0)

    async def run():
        return await rm.check(
            Signal("BTC/USDT", "sell", qty=1.0), 70.0, 0.0, 1000.0, 0.0, [], 100.0)

    ok, why = asyncio.run(run())
    assert ok, f"平仓应豁免 24h 冷却，实际: {why}"
