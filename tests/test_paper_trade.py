"""验证纸面交易核心闭环：size_pct 比例信号应正确换算 qty 成交。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import asyncio

from core.database import Database
from core.bus import EventBus
from engine.order_manager import OrderManager
from exchange.paper import PaperAccount
from strategies.base import Signal


async def main():
    # 简化：不连数据库，直接测试 qty 换算与纸面撮合
    acc = PaperAccount(start_cash=10000.0, fee_rate=0.001)
    bus = EventBus()
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()  # 建表，_record_trade 需要 trades 表
    om = OrderManager(db, bus, paper=True, paper_account=acc)

    # 比例买入信号：cash=10000, size_pct=0.5, price=50000
    sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="dual_ma")
    qty = om._resolve_qty(sig, 50000.0)
    print(f"比例买入 qty={qty:.6f} (期望≈0.0999，即 5000/50000 且留手续费)")
    assert 0.09 < qty < 0.1005, "qty 换算错误"
    fill = await om.place(sig, 50000.0, "dual_ma")
    print(f"成交: {fill}")
    assert fill["qty"] > 0, "买入 qty 必须 > 0"
    assert acc.positions.get("BTC/USDT", {}).get("qty", 0) > 0, "持仓未建立"
    cash_after = acc.cash
    print(f"建仓后现金={cash_after:.2f} 持仓={acc.positions['BTC/USDT']['qty']:.6f}")
    assert cash_after < 10000.0, "现金应减少"

    # 比例卖出：清仓
    sig2 = Signal("BTC/USDT", "sell", size_pct=1.0, strategy="dual_ma")
    qty2 = om._resolve_qty(sig2, 51000.0)
    print(f"卖出 qty={qty2:.6f} (期望≈全部持仓 {acc.positions['BTC/USDT']['qty']:.6f})")
    assert abs(qty2 - acc.positions["BTC/USDT"]["qty"]) < 1e-9, "卖出 qty 应等于全部持仓"
    await om.place(sig2, 51000.0, "dual_ma")
    print(f"平仓后持仓={acc.positions['BTC/USDT']['qty']:.6f} 现金={acc.cash:.2f}")
    assert acc.positions["BTC/USDT"]["qty"] == 0, "持仓应清零"
    print("\nPAPER TRADE CLOSED LOOP OK")


if __name__ == "__main__":
    asyncio.run(main())
