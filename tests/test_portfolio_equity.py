"""P0-2 回归测试：实盘资产页 USDT 单计（不双计则 equity 虚高一份 USDT 余额）。

对齐口径（engine/portfolio.py:56-59）：USDT 计价币只计 cash，不进 positions、
不加 positions_value。本测试直接调用 web/api/portfolio.py 的 _do_fetch 逻辑，
用假 mgr 模拟交易所余额与行情，无需真网络。
"""
import asyncio

import pytest

from web.api.portfolio import _do_fetch


class FakeMgr:
    """模拟 ExchangeManager：仅提供 _do_fetch 用到的三个异步方法。"""

    def __init__(self, balance: dict, prices: dict):
        self._balance = balance
        self._prices = dict(prices)
        self.started = False

    async def start(self, *args, **kwargs):
        self.started = True

    async def fetch_balance(self):
        return self._balance

    async def fetch_ticker(self, symbol):
        last = self._prices.get(symbol)
        if last is None:
            raise RuntimeError(f"no price for {symbol}")
        return {"last": last}


def _run(balance: dict, prices: dict) -> dict:
    """在独立事件循环中执行 _do_fetch（与真实 worker 线程使用方式一致）。"""
    mgr = FakeMgr(balance, prices)
    return asyncio.run(_do_fetch(mgr, "k", "s", ""))


def test_usdt_not_double_counted():
    """USDT 只进 cash：不进 positions、不加 positions_value（曾双计虚高 equity）。"""
    result = _run({"total": {"USDT": 100, "BTC": 0.01, "ETH": 0}},
                  {"BTC/USDT": 50000.0})
    # USDT 不在持仓列表
    assert "USDT" not in [p["symbol"] for p in result["positions"]]
    assert [p["symbol"] for p in result["positions"]] == ["BTC/USDT"]
    # 现金只计一次 100
    assert result["cash"] == 100
    # 持仓市值只有 BTC（USDT 不重复计入）
    assert result["positions_value"] == pytest.approx(0.01 * 50000.0)
    # equity = cash + 真实持仓市值（无 USDT 重复项）
    assert result["equity"] == pytest.approx(100 + 0.01 * 50000.0)
    assert result["source"] == "live_exchange"


def test_zero_balance_assets_skipped():
    """0 余额资产不产生持仓条目，也不影响其它资产计价。"""
    result = _run({"total": {"USDT": 100, "BTC": 0, "ETH": 0.5}},
                  {"ETH/USDT": 2000.0})
    assert [p["symbol"] for p in result["positions"]] == ["ETH/USDT"]
    assert result["cash"] == 100
    assert result["positions_value"] == pytest.approx(0.5 * 2000.0)
    assert result["equity"] == pytest.approx(100 + 0.5 * 2000.0)


def test_ticker_failure_skips_asset():
    """行情获取失败的资产被跳过（_safe_price 返回 None），不影响 cash/equity。"""
    mgr = FakeMgr({"total": {"USDT": 100, "BTC": 0.01}}, {})
    result = asyncio.run(_do_fetch(mgr, "k", "s", ""))
    assert result["cash"] == 100
    assert result["positions"] == []
    assert result["positions_value"] == 0.0
    assert result["equity"] == 100.0


def test_empty_balance():
    """空余额：全 0，返回结构键不变。"""
    result = _run({"total": {}}, {})
    assert result == {"equity": 0.0, "cash": 0.0, "positions_value": 0.0,
                      "positions": [], "source": "live_exchange"}
