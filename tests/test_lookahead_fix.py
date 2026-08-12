"""验证前视偏差修复：回测仍正常运行，输出指标完整。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
from backtest.data_loader import generate_demo
from backtest.fast_engine import run_backtest_fast
from backtest.engine import BacktestConfig

df = generate_demo(n=2000, timeframe="1h")
print(f"数据量: {len(df)}")

for strategy in ["dual_ma", "price_action"]:
    cfg = BacktestConfig(symbol="BTC/USDT", timeframe="1h",
                         strategy_name=strategy, start_cash=10000.0)
    res = run_backtest_fast(df, cfg)
    m = res["metrics"]
    print(f"\n策略 {strategy}:")
    print(f"  总收益={m.get('total_return'):.4f} 夏普={m.get('sharpe'):.2f} "
          f"回撤={m.get('max_drawdown'):.4f} 胜率={m.get('win_rate'):.2f} 交易数={len(res['trades'])}")
    print(f"  期末权益={res['equity_curve'][-1]:.2f}")
    assert res["equity_curve"], "权益曲线为空"
    print("  OK")
print("\nLOOKAHEAD FIX VERIFY OK")
