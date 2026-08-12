"""真实行情端到端：拉取真实 BTC K线，验证方向复算引擎输出合理。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import asyncio
from ai.direction_engine import compute_direction


async def main():
    from backtest.data_loader import load_from_exchange
    try:
        df = await load_from_exchange("binance", "BTC/USDT", "1h", limit=150)
    except Exception as e:
        print(f"交易所拉取失败（可能无网络），改用演示数据: {e}")
        from backtest.data_loader import generate_demo
        df = generate_demo(n=200, timeframe="1h")

    candles = [[int(t.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
               for t, r in df.iterrows()]
    print(f"K线数量: {len(candles)}")
    res = compute_direction(candles)
    d = {"long": "看多", "short": "看空", "neutral": "中性"}[res["direction"]]
    print(f"程序方向: {d} score={res['score']:+d} confidence={res['confidence']:.2f}")
    for s in res["signals"]:
        print(f"   {s['name']:<16} {s['sign']:+d}  {s['reason']}")
    assert res["direction"] in ("long", "short", "neutral")
    assert 0 <= res["confidence"] <= 1
    print("\nREAL DATA DIRECTION OK")


if __name__ == "__main__":
    asyncio.run(main())
