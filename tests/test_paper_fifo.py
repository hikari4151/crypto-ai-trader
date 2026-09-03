"""M1 回测口径对齐：FIFO 逐笔记账单元测试。

覆盖：
- PaperAccount.apply_fill FIFO 摊销（与回测引擎 lots.pop(0) 逻辑一致）
- avg_price / positions_value 辅助方法
- OrderManager._apply_fifo 实盘 FIFO 队列
- TradingEngine._record_fill 成本计算使用 fill["cost_price"]
"""

import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")

import asyncio
import pytest

from exchange.paper import PaperAccount
from strategies.base import Signal


# ========== PaperAccount FIFO 单元测试 ==========


class TestPaperAccountFIFO:
    """apply_fill FIFO 摊销：买多笔后卖出一部分，成本应为最早买入价。"""

    def test_fifo_cost_first_lot(self):
        """买 1@100、买 1@200、卖 1 → FIFO 成本 100（非加权平均 150）。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        # FIFO 成本应为第一笔买入价 100
        assert result["cost_price"] == 100.0, f"FIFO cost_price 应为 100，实际 {result['cost_price']}"
        # 剩余 1 个 lot，价格 200
        assert acc.positions["BTC/USDT"]["qty"] == 1.0
        assert len(acc.positions["BTC/USDT"]["lots"]) == 1
        assert acc.positions["BTC/USDT"]["lots"][0] == (1.0, 200.0)

    def test_fifo_partial_lot_consume(self):
        """买 2@100、卖 1 → FIFO 成本 100，剩余 lot 缩减为 (1@100)。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 2.0, 100.0)
        result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        assert result["cost_price"] == 100.0
        assert acc.positions["BTC/USDT"]["qty"] == 1.0
        assert acc.positions["BTC/USDT"]["lots"][0] == (1.0, 100.0)

    def test_fifo_full_close(self):
        """买 1@100、买 1@200、卖 2 → 全额平仓，FIFO 成本 = (100+200)/2 = 150。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        result = acc.apply_fill("BTC/USDT", "sell", 2.0, 250.0)
        assert result["cost_price"] == 150.0
        assert acc.positions["BTC/USDT"]["qty"] == 0.0
        # 全额平仓后 lots 应清空
        assert len(acc.positions["BTC/USDT"]["lots"]) == 0

    def test_fifo_crossing_lot_boundary(self):
        """买 1@100、买 1@200、卖 1.5 → 跨 2 个 lot，FIFO 成本 = (100 + 0.5*200)/1.5 ≈ 133.33。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        result = acc.apply_fill("BTC/USDT", "sell", 1.5, 150.0)
        expected_cost = (1.0 * 100 + 0.5 * 200) / 1.5
        assert abs(result["cost_price"] - expected_cost) < 1e-12
        # 剩余 lot：(0.5@200)
        assert acc.positions["BTC/USDT"]["qty"] == 0.5
        assert abs(acc.positions["BTC/USDT"]["lots"][0][0] - 0.5) < 1e-12
        assert acc.positions["BTC/USDT"]["lots"][0][1] == 200.0

    def test_fifo_cash_consistency(self):
        """现金变化应与成交价*数量-手续费一致（FIFO 不影响现金）。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        cash_before = acc.cash
        sell_price = 150.0
        acc.apply_fill("BTC/USDT", "sell", 1.0, sell_price)
        # 现金增加 = 卖价 * 1（fee 0）
        assert abs(acc.cash - (cash_before + sell_price)) < 1e-9

    def test_fifo_with_fee(self):
        """带手续费：FIFO 成本价不受手续费影响，现金变化含手续费。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.001)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        # 成本价仍为 100
        assert result["cost_price"] == 100.0

    def test_fifo_sell_insufficient_position(self):
        """卖出超过持仓应抛出 ValueError。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        with pytest.raises(ValueError, match="纸面账户持仓不足"):
            acc.apply_fill("BTC/USDT", "sell", 2.0, 150.0)

    def test_fifo_no_position_no_sell(self):
        """无持仓时卖出应抛出 ValueError。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        with pytest.raises(ValueError, match="纸面账户持仓不足"):
            acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)

    def test_fifo_buy_over_cash(self):
        """买入超出余额应抛出 ValueError。"""
        acc = PaperAccount(start_cash=100.0, fee_rate=0.0)
        with pytest.raises(ValueError, match="纸面账户余额不足"):
            acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)


class TestPaperAccountAvgPrice:
    """avg_price 辅助方法测试。"""

    def test_avg_price_single_lot(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        assert acc.avg_price("BTC/USDT") == 100.0

    def test_avg_price_multi_lot(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        # 加权平均：150
        assert acc.avg_price("BTC/USDT") == 150.0

    def test_avg_price_after_partial_sell(self):
        """部分卖出后 avg_price 反映剩余 lots 的加权成本。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)  # 卖掉 1@100
        # 剩余：1@200
        assert acc.avg_price("BTC/USDT") == 200.0

    def test_avg_price_no_position(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        assert acc.avg_price("BTC/USDT") == 0.0

    def test_avg_price_closed_position(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        # 平仓后 avg_price = 0
        assert acc.avg_price("BTC/USDT") == 0.0


class TestPaperAccountPositionsValue:
    """positions_value 使用最新价兜底 avg_price。"""

    def test_positions_value_with_last_price(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 2.0, 100.0)
        val = acc.positions_value({"BTC/USDT": 200.0})
        assert val == 400.0

    def test_positions_value_fallback_to_avg_price(self):
        """无最新价时使用 FIFO 加权成本（avg_price）。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        val = acc.positions_value({})
        # 无 last_price，用 avg_price = 150
        assert val == 2.0 * 150.0

    def test_positions_value_empty(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        assert acc.positions_value({}) == 0.0


# ========== OrderManager._apply_fifo 实盘 FIFO 队列测试 ==========


class TestLiveFIFO:
    """OrderManager._apply_fifo 实盘 FIFO 成本队列。"""

    def _make_om(self):
        """创建最小 OrderManager 实例（仅测试 _apply_fifo，不依赖 DB/总线）。"""
        from core.bus import EventBus
        from core.database import Database
        bus = EventBus()
        db = Database("sqlite+aiosqlite:///:memory:")
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        return om

    def test_buy_appends_lot(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        assert om._live_lots["BTC/USDT"] == [(1.0, 100.0)]

    def test_sell_fifo_cost(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        om._apply_fifo("BTC/USDT", "buy", 1.0, 200.0)
        cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        # FIFO 成本 100
        assert cost == 100.0
        # 剩余 1 lot (1@200)
        assert om._live_lots["BTC/USDT"] == [(1.0, 200.0)]

    def test_sell_full_close(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert cost == 100.0
        # m3: 清仓后 key 应被清理
        assert "BTC/USDT" not in om._live_lots

    def test_sell_partial_lot(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 2.0, 100.0)
        cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert cost == 100.0
        assert om._live_lots["BTC/USDT"] == [(1.0, 100.0)]

    def test_sell_crossing_lot_boundary(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        om._apply_fifo("BTC/USDT", "buy", 1.0, 200.0)
        cost = om._apply_fifo("BTC/USDT", "sell", 1.5, 150.0)
        expected = (1.0 * 100 + 0.5 * 200) / 1.5
        assert abs(cost - expected) < 1e-12
        assert len(om._live_lots["BTC/USDT"]) == 1
        assert abs(om._live_lots["BTC/USDT"][0][0] - 0.5) < 1e-12
        assert om._live_lots["BTC/USDT"][0][1] == 200.0

    def test_sell_beyond_tracked_lots_returns_none(self):
        """卖出量超过 FIFO 队列覆盖时返回 None（走 entry_price 兜底）。"""
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        cost = om._apply_fifo("BTC/USDT", "sell", 2.0, 150.0)
        # 队列只覆盖 1，超出部分无成本参考 → None
        assert cost is None
        # m3: 队列已全量消费（含超卖部分）→ key 清理
        assert "BTC/USDT" not in om._live_lots

    def test_sell_with_empty_lots_returns_none(self):
        """无队列记录时卖出一律返回 None。"""
        om = self._make_om()
        cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert cost is None

    def test_buy_returns_none(self):
        om = self._make_om()
        cost = om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        assert cost is None


# ========== TradingEngine._record_fill FIFO 成本对齐测试 ==========


class FakeDB:
    """内存 KV，模拟 Database 的 kv 接口（无需真 SQLite）。"""
    def __init__(self):
        self.kv = {}
        self.secrets = {}

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

    async def kv_get_secret(self, key):
        return self.secrets.get(key)

    async def kv_set_secret(self, key, value):
        self.secrets[key] = value

    def session(self):
        class FakeSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass
            async def execute(self, *a, **k):
                return EmptyResult()
            def add(self, *a, **k):
                pass
            async def commit(self, *a, **k):
                pass
            async def refresh(self, *a, **k):
                pass
        return FakeSession()

    async def update_trade_pnl(self, *a, **k):
        pass


class EmptyResult:
    def scalars(self):
        return self
    def all(self):
        return []


class CollectingBus:
    """事件总线，收集所有发布事件供断言。"""
    def __init__(self):
        from core.bus import EventBus
        self._bus = EventBus()
        self.events = []

    async def publish(self, event) -> bool:
        self.events.append(event)
        return await self._bus.publish(event)

    def subscribe(self, event_type, handler):
        return self._bus.subscribe(event_type, handler)

    def health(self):
        return {"degraded": False, "handlers": 0}


@pytest.mark.asyncio
async def test_record_fill_paper_sell_uses_fifo_cost():
    """纸面模式卖出时，_record_fill 使用 fill['cost_price'] 计算 PnL（FIFO 成本）。"""
    db = FakeDB()
    bus = CollectingBus()
    from engine.trading_engine import TradingEngine
    eng = TradingEngine(db, bus)
    eng.paper = True
    eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)

    # 建仓：买 1@100
    eng.paper_account.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
    # 再买 1@200（FIFO 队列：[(1,100), (1,200)]）
    eng.paper_account.apply_fill("BTC/USDT", "buy", 1.0, 200.0)

    # 卖 1@150，cost_price 应为 100（FIFO）
    fill = {"price": 150.0, "qty": 1.0, "fee": 0.0, "trade_id": 1, "cost_price": 100.0}
    await eng._record_fill("BTC/USDT", "sell", fill, "dual_ma", entry_price=200.0)

    # PnL = (150 - 100) * 1 - 0 = 50（每日盈亏唯一来源在风控）
    daily = await eng.risk.get_daily_pnl()
    assert abs(daily - 50.0) < 1e-9, f"PnL 应为 50，实际 {daily}"
    assert not hasattr(eng, "_daily_pnl"), "引擎不得自持每日盈亏副本（双源会让熔断跨日不解封）"


@pytest.mark.asyncio
async def test_record_fill_falls_back_to_entry_price():
    """fill 无 cost_price 时使用 entry_price 兜底。"""
    db = FakeDB()
    bus = CollectingBus()
    from engine.trading_engine import TradingEngine
    eng = TradingEngine(db, bus)
    eng.paper = True
    eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)

    eng.paper_account.apply_fill("BTC/USDT", "buy", 1.0, 100.0)

    # 卖 1@150，无 cost_price → 用 entry_price = 100
    fill = {"price": 150.0, "qty": 1.0, "fee": 0.0, "trade_id": 1}  # cost_price 缺失
    await eng._record_fill("BTC/USDT", "sell", fill, "dual_ma", entry_price=100.0)

    assert abs(await eng.risk.get_daily_pnl() - 50.0) < 1e-9


@pytest.mark.asyncio
async def test_record_fill_entry_price_none():
    """cost_price 和 entry_price 都 None 时 PnL=0。"""
    db = FakeDB()
    bus = CollectingBus()
    from engine.trading_engine import TradingEngine
    eng = TradingEngine(db, bus)
    eng.paper = True
    eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)

    fill = {"price": 150.0, "qty": 1.0, "fee": 0.0, "trade_id": 1}
    await eng._record_fill("BTC/USDT", "sell", fill, "dual_ma", entry_price=None)

    assert await eng.risk.get_daily_pnl() == 0.0


@pytest.mark.asyncio
async def test_record_fill_buy_does_not_affect_pnl():
    """买入不触发 PnL 计算。"""
    db = FakeDB()
    bus = CollectingBus()
    from engine.trading_engine import TradingEngine
    eng = TradingEngine(db, bus)
    eng.paper = True
    eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)

    daily_before = await eng.risk.get_daily_pnl()
    fill = {"price": 100.0, "qty": 1.0, "fee": 0.0, "trade_id": 1}
    await eng._record_fill("BTC/USDT", "buy", fill, "dual_ma", entry_price=None)

    # 买入不应改变 PnL
    assert await eng.risk.get_daily_pnl() == daily_before


# ========== m5: 补充测试 ==========


class TestToleranceOversell:
    """超卖容差：qty 刚好等于持仓量（含浮点容差）。"""

    def test_sell_exact_qty_with_tolerance(self):
        """买 1.0，卖 1.0（精确等于持仓）。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        assert result["cost_price"] == 100.0
        assert acc.positions["BTC/USDT"]["qty"] == 0.0

    def test_sell_slightly_above_position_raises(self):
        """卖 1.0+1e-9（超出容差）应抛出 ValueError。"""
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        with pytest.raises(ValueError, match="纸面账户持仓不足"):
            acc.apply_fill("BTC/USDT", "sell", 1.0 + 1e-9, 150.0)


class TestMultiSymbolIsolation:
    """多品种 FIFO 隔离：各自独立摊销。"""

    def test_multi_symbol_fifo(self):
        acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        # BTC: 买 1@100, 买 1@200
        # ETH: 买 2@50
        acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
        acc.apply_fill("ETH/USDT", "buy", 2.0, 50.0)
        acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
        # 卖 1 BTC → FIFO 成本 100
        result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)
        assert result["cost_price"] == 100.0
        assert acc.positions["BTC/USDT"]["qty"] == 1.0
        assert acc.positions["ETH/USDT"]["qty"] == 2.0
        # ETH 不受影响
        assert acc.positions["ETH/USDT"]["lots"] == [(2.0, 50.0)]


class TestFIFOOrderManager:
    """OrderManager._apply_fifo 实盘 FIFO 队列覆盖。"""

    def _make_om(self):
        from core.bus import EventBus
        from core.database import Database
        bus = EventBus()
        db = Database("sqlite+aiosqlite:///:memory:")
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        return om

    def test_empty_queue_returns_none(self):
        om = self._make_om()
        cost = om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert cost is None
        # m3: 空 key 应被清理
        assert "BTC/USDT" not in om._live_lots

    def test_full_consume_cleans_key(self):
        """sell 消费全部 lot 后 symbol key 应被清理。"""
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert "BTC/USDT" not in om._live_lots

    def test_partial_consume_keeps_key(self):
        om = self._make_om()
        om._apply_fifo("BTC/USDT", "buy", 2.0, 100.0)
        om._apply_fifo("BTC/USDT", "sell", 1.0, 150.0)
        assert "BTC/USDT" in om._live_lots
        assert om._live_lots["BTC/USDT"] == [(1.0, 100.0)]


class TestLiveFIFODBFailure:
    """C1: _record_trade 失败时 _live_lots 不应被突变。"""

    @pytest.mark.asyncio
    async def test_place_live_db_failure_no_mutate(self):
        """_place_live 中 DB 失败 → _live_lots 不变。"""
        from core.bus import EventBus
        from exchange.manager import ExchangeManager
        from unittest.mock import AsyncMock, MagicMock

        class FailingDB:
            def session(self):
                raise RuntimeError("db down")
            async def kv_get(self, *a, **k):
                return None
            async def kv_set(self, *a, **k):
                pass
            async def kv_json_get(self, *a, **k):
                return None
            async def kv_json_set(self, *a, **k):
                pass
            async def kv_get_secret(self, *a, **k):
                return None
            async def update_trade_pnl(self, *a, **k):
                pass

        db = FailingDB()
        bus = EventBus()
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        # 模拟 exchange
        om.exchange = MagicMock()
        om.exchange.create_order = AsyncMock(return_value={
            "status": "closed", "filled": "1.0", "amount": "1.0",
            "average": "100.0", "price": "100.0", "fee": {"cost": "0.0"},
        })
        om.exchange.fetch_balance = AsyncMock(return_value={
            "BTC": {"free": 1.0}, "USDT": {"free": 10000.0}
        })
        # 预设 FIFO 队列
        om._live_lots["BTC/USDT"] = [(1.0, 100.0)]
        sig = Signal("BTC/USDT", "sell", qty=1.0, strategy="dual_ma")
        result = await om.place(sig, 100.0, "dual_ma")
        # 应返回 None（DB 失败，try 捕获）
        assert result is None
        # FIFO 队列应未被突变
        assert om._live_lots["BTC/USDT"] == [(1.0, 100.0)]

    @pytest.mark.asyncio
    async def test_backfill_db_failure_no_mutate(self):
        """_backfill_fill 中 DB 失败 → _live_lots 不变。"""
        from core.bus import EventBus
        from unittest.mock import MagicMock, AsyncMock

        class FailingDB:
            def session(self):
                raise RuntimeError("db down")
            async def kv_get(self, *a, **k):
                return None
            async def kv_set(self, *a, **k):
                pass
            async def update_trade_pnl(self, *a, **k):
                pass

        db = FailingDB()
        bus = EventBus()
        bus.subscribe = MagicMock()
        bus.publish = AsyncMock()
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        # 预设 FIFO 队列和注册表
        om._live_lots["BTC/USDT"] = [(1.0, 100.0)]
        om._open_orders["oid1"] = {
            "symbol": "BTC/USDT", "side": "sell", "qty": 1.0,
            "strategy_name": "dual_ma", "placed_ts": 0.0,
            "limit_price": 100.0, "recorded_filled": 0.0,
        }
        order = {"filled": "1.0", "average": "100.0", "price": "100.0",
                 "fee": {"cost": "0.0"}, "status": "closed"}
        with pytest.raises(RuntimeError):
            await om._backfill_fill("oid1", om._open_orders["oid1"], order)
        # FIFO 队列应未被突变
        assert om._live_lots["BTC/USDT"] == [(1.0, 100.0)]


class TestM4CostPriceZeroFallback:
    """m4: cost_price=0.0 应回退到 entry_price。"""

    @pytest.mark.asyncio
    async def test_cost_price_zero_falls_to_entry(self):
        db = FakeDB()
        bus = CollectingBus()
        from engine.trading_engine import TradingEngine
        eng = TradingEngine(db, bus)
        eng.paper = True
        eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)
        eng.paper_account.apply_fill("BTC/USDT", "buy", 1.0, 100.0)

        # cost_price=0.0 + entry_price=100 → 应使用 100
        fill = {"price": 150.0, "qty": 1.0, "fee": 0.0, "trade_id": 1, "cost_price": 0.0}
        await eng._record_fill("BTC/USDT", "sell", fill, "dual_ma", entry_price=100.0)
        assert abs(await eng.risk.get_daily_pnl() - 50.0) < 1e-9

    @pytest.mark.asyncio
    async def test_cost_price_zero_and_entry_none_returns_zero(self):
        db = FakeDB()
        bus = CollectingBus()
        from engine.trading_engine import TradingEngine
        eng = TradingEngine(db, bus)
        eng.paper = True
        eng.paper_account = PaperAccount(start_cash=10000.0, fee_rate=0.0)

        # 两者都无效 → pnl=0
        fill = {"price": 150.0, "qty": 1.0, "fee": 0.0, "trade_id": 1, "cost_price": 0.0}
        await eng._record_fill("BTC/USDT", "sell", fill, "dual_ma", entry_price=None)
        assert await eng.risk.get_daily_pnl() == 0.0


class TestM2DeltaBackfill:
    """M2: 部分成交登记对账 + delta 回灌。"""

    @pytest.mark.asyncio
    async def test_register_partial_fill(self):
        """_place_live 部分成交且仍挂单时登记对账并记录 recorded_filled。"""
        from core.bus import EventBus
        from unittest.mock import MagicMock, AsyncMock

        db = FakeDB()
        bus = EventBus()
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        om.exchange = MagicMock()
        # 模拟 0.5 部分成交，订单仍 open
        om.exchange.create_order = AsyncMock(return_value={
            "id": "oid1", "status": "open", "filled": "0.5", "amount": "0.5",
            "average": "100.0", "price": "100.0", "fee": {"cost": "0.0"},
        })
        # 轮询 fetch_order 仍返回 open 状态（保持部分成交）
        om.exchange.fetch_order = AsyncMock(return_value={
            "id": "oid1", "status": "open", "filled": "0.5", "amount": "0.5",
            "average": "100.0", "price": "100.0", "fee": {"cost": "0.0"},
        })
        om.exchange.fetch_balance = AsyncMock(return_value={
            "BTC": {"free": 1.0}, "USDT": {"free": 10000.0}
        })
        sig = Signal("BTC/USDT", "sell", qty=1.0, strategy="dual_ma")
        # 执行 _place_live
        result = await om.place(sig, 100.0, "dual_ma")
        # 部分成交应返回成交数据
        assert result is not None
        assert result["qty"] == 0.5
        # 应登记对账
        assert "oid1" in om._open_orders
        assert om._open_orders["oid1"]["recorded_filled"] == 0.5

    @pytest.mark.asyncio
    async def test_backfill_delta(self):
        """_backfill_fill 应只回灌未记录的成交 delta。"""
        from core.bus import EventBus
        from unittest.mock import MagicMock, AsyncMock

        db = FakeDB()
        bus = EventBus()
        bus.subscribe = MagicMock()
        bus.publish = AsyncMock()
        from engine.order_manager import OrderManager
        om = OrderManager(db, bus, paper=False, paper_account=None)
        # 预设 FIFO 队列：1 BTC @ 100
        om._apply_fifo("BTC/USDT", "buy", 1.0, 100.0)
        # 注册表：已记录 0.5，总成交 1.0 → delta = 0.5
        rec = {
            "symbol": "BTC/USDT", "side": "sell", "qty": 1.0,
            "strategy_name": "dual_ma", "placed_ts": 0.0,
            "limit_price": 100.0, "recorded_filled": 0.5,
        }
        order = {"filled": "1.0", "average": "100.0", "price": "100.0",
                 "fee": {"cost": "0.0"}, "status": "closed"}
        # 执行 backfill（应只记录 0.5 delta）
        await om._backfill_fill("oid1", rec, order)
        # 事件 qty 应为 0.5（delta）
        published = [e for e in bus.publish.call_args_list if e[0][0].type.name == "ORDER_FILL"]
        assert len(published) >= 1
        fill_qty = float(published[0][0][0].payload.get("qty", 0))
        assert fill_qty == 0.5, f"delta 应回灌 qty=0.5，实际 {fill_qty}"
        # FIFO 队列应消费 0.5 BTC（剩余 0.5）
        assert len(om._live_lots["BTC/USDT"]) == 1
        assert abs(om._live_lots["BTC/USDT"][0][0] - 0.5) < 1e-12


def test_manual_verification_fifo_vs_avg():
    """手动验证：买 1@100、买 1@200、卖 1——FIFO 成本 100（非加权平均 150）。

    此测试打印中间结果供人工核对。
    """
    acc = PaperAccount(start_cash=10000.0, fee_rate=0.0)
    acc.apply_fill("BTC/USDT", "buy", 1.0, 100.0)
    acc.apply_fill("BTC/USDT", "buy", 1.0, 200.0)
    result = acc.apply_fill("BTC/USDT", "sell", 1.0, 150.0)

    fifo_cost = result["cost_price"]
    # 加权平均成本（旧 avg_price 逻辑）
    qty_remaining = acc.positions["BTC/USDT"]["qty"]
    cost_remaining = acc.avg_price("BTC/USDT")

    print(f"\n=== FIFO 验证 ===")
    print(f"FIFO 卖出成本价: {fifo_cost}")  # 应为 100
    print(f"加权平均成本: {(1.0 * 100 + 1.0 * 200) / 2}")  # 150
    print(f"剩余持仓: {qty_remaining} @ {cost_remaining}")  # 1@200
    print(f"现金: {acc.cash}")  # 10000 - 100 - 200 + 150 = 9850

    assert fifo_cost == 100.0, f"FIFO 成本应为 100，实际 {fifo_cost}"
    assert qty_remaining == 1.0
    assert cost_remaining == 200.0
    assert abs(acc.cash - 9850.0) < 1e-9