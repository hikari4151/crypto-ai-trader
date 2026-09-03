"""波动率目标仓位测试：高波动自动缩仓（只缩不加）。"""
import asyncio
import math
import random

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


def _feed_prices(rm: RiskManager, n: int, dt: float, annual_vol: float, seed: int = 7) -> float:
    """向价格历史注入 n 个样本（间隔 dt 秒），构造已知年化波动的几何随机游走。

    返回期末价格。
    """
    rng = random.Random(seed)
    period_sd = annual_vol / math.sqrt(31536000.0 / dt)
    now = 1_800_000_000.0
    price = 100.0
    for i in range(n):
        t = now - (n - i) * dt
        price *= math.exp(rng.gauss(0.0, period_sd))
        rm._price_history.append((t, price))
    return price


class TestRealizedVol:
    def test_known_vol_recovered(self):
        """构造 100% 年化波动 → 估计值应在 ±30% 内"""
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        rv = rm._realized_vol_annualized(24.0)
        assert rv is not None
        assert 0.7 < rv < 1.3, f"rv={rv}"

    def test_insufficient_samples_none(self):
        rm = make_rm()
        _feed_prices(rm, n=10, dt=5.0, annual_vol=1.0)
        assert rm._realized_vol_annualized(24.0) is None

    def test_short_span_none(self):
        """样本够多但时间跨度不足 15 分钟 → None"""
        rm = make_rm()
        _feed_prices(rm, n=50, dt=2.0, annual_vol=1.0)
        assert rm._realized_vol_annualized(24.0) is None


class TestVolTargetScaling:
    def test_high_vol_scales_down(self):
        """已实现波动 100% > 目标 40% → size_pct 缩到 ~0.4x"""
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        rules = {**DEFAULT_RULES, "vol_target_enabled": 1,
                 "vol_target_annual_pct": 0.40, "vol_target_lookback_hours": 24.0}
        sig = Signal("BTC/USDT", "buy", size_pct=0.5)
        rm._apply_vol_target(sig, rules)
        assert 0.15 < sig.size_pct < 0.25, f"scaled={sig.size_pct}"

    def test_low_vol_no_scale_up(self):
        """已实现波动 20% < 目标 40% → 不放大（只缩不加）"""
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=0.2)
        rules = {**DEFAULT_RULES, "vol_target_enabled": 1,
                 "vol_target_annual_pct": 0.40, "vol_target_lookback_hours": 24.0}
        sig = Signal("BTC/USDT", "buy", size_pct=0.5)
        rm._apply_vol_target(sig, rules)
        assert sig.size_pct == 0.5

    def test_disabled_noop(self):
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        rules = {**DEFAULT_RULES, "vol_target_enabled": 0}
        sig = Signal("BTC/USDT", "buy", size_pct=0.5)
        rm._apply_vol_target(sig, rules)
        assert sig.size_pct == 0.5

    def test_explicit_qty_not_touched(self):
        """qty 明确的单不干预（只作用于比例下单）"""
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        rules = {**DEFAULT_RULES, "vol_target_enabled": 1,
                 "vol_target_annual_pct": 0.40, "vol_target_lookback_hours": 24.0}
        sig = Signal("BTC/USDT", "buy", qty=0.01)
        rm._apply_vol_target(sig, rules)
        assert sig.qty == 0.01

    def test_sell_not_touched(self):
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        rules = {**DEFAULT_RULES, "vol_target_enabled": 1,
                 "vol_target_annual_pct": 0.40, "vol_target_lookback_hours": 24.0}
        sig = Signal("BTC/USDT", "sell", size_pct=1.0)
        rm._apply_vol_target(sig, rules)
        assert sig.size_pct == 1.0


class TestVolTargetIntegration:
    def test_check_scales_buy_signal(self):
        """端到端：check() 放行的买单 size_pct 被缩放（update_rules 开启后）"""
        rm = make_rm()
        _feed_prices(rm, n=2000, dt=5.0, annual_vol=1.0)
        price = 100.0

        async def run():
            await rm.update_rules({"vol_target_enabled": 1, "vol_target_annual_pct": 0.40})
            sig = Signal("BTC/USDT", "buy", size_pct=0.5)
            allowed, reason = await rm.check(sig, price, cash=10000.0, position_value=0.0,
                                             daily_pnl=0.0, trade_times=[], entry_price=None)
            return allowed, reason, sig

        allowed, reason, sig = asyncio.run(run())
        assert allowed, reason
        assert sig.size_pct < 0.5, f"size not scaled: {sig.size_pct}"

    def test_update_rules_accepts_and_bounds(self):
        rm = make_rm()

        async def run():
            return await rm.update_rules({"vol_target_enabled": 1,
                                          "vol_target_annual_pct": 99.0,
                                          "vol_target_lookback_hours": 0.1})

        rules = asyncio.run(run())
        assert rules["vol_target_enabled"] == 1.0
        assert rules["vol_target_annual_pct"] == 5.0  # 钳制到上界
        assert rules["vol_target_lookback_hours"] == 1.0

    def test_nan_vol_target_rejected(self):
        rm = make_rm()

        async def run():
            return await rm.update_rules({"vol_target_annual_pct": float("nan")})

        rules = asyncio.run(run())
        assert rules["vol_target_annual_pct"] == DEFAULT_RULES["vol_target_annual_pct"]
