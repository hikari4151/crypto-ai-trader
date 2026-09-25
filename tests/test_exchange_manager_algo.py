"""ExchangeManager.fetch_algo_open_orders 回归：OKX 条件单可见性 + 其他交易所安全降级。

实盘背景（2026-09-18）：OKX 兜底止损（条件单）挂单成功后，ccxt 统一
`fetch_open_orders` 返回 0 条；必须走 `/api/v5/trade/orders-algo-pending`。
本测试固定两点：
1. 非 OKX / 无 private 客户端 → 返回 []（绝不影响既有路径）；
2. OKX → 正确解析 algo 数据为统一结构（含 clientOrderId，供 STO P_TAG 收养）。
"""
import pytest

from exchange.manager import ExchangeManager


@pytest.mark.asyncio
async def test_algo_orders_empty_for_non_okx():
    mgr = ExchangeManager("binance")
    assert await mgr.fetch_algo_open_orders() == []


@pytest.mark.asyncio
async def test_algo_orders_parsed_for_okx(monkeypatch):
    mgr = ExchangeManager("okx")

    class FakePrivate:
        async def private_get_trade_orders_algo_pending(self, params):
            if params.get("ordType") != "trigger":
                return {"data": []}
            return {"data": [{
                "algoId": "algo1", "algoClOrdId": "ps1789732313", "instId": "BTC-USDT",
                "side": "sell", "sz": "0.00001", "slTriggerPx": "77271.78",
                "state": "live", "ordType": "trigger"}]}

    mgr._private = FakePrivate()
    out = await mgr.fetch_algo_open_orders("BTC/USDT")
    assert len(out) == 1
    o = out[0]
    assert o["id"] == "algo1"
    assert o["clientOrderId"] == "ps1789732313"
    assert o["symbol"] == "BTC/USDT"
    assert o["amount"] == pytest.approx(0.00001)
    assert float(o["triggerPrice"]) == pytest.approx(77271.78)

    # symbol 过滤生效
    assert await mgr.fetch_algo_open_orders("ETH/USDT") == []
