"""启动入口：
  python run.py                    # 启动 Web 服务（推荐）
  python run.py --host 0.0.0.0 --port 8000
  python run.py --backtest         # 命令行快速回测（演示数据）
  python run.py --backfill --exchange binance --symbol BTC/USDT --timeframe 1h --limit 1000
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.engine import BacktestConfig, run_backtest

    async def _run():
        if args.source == "demo":
            df = generate_demo(timeframe=args.timeframe)
        elif args.source == "csv":
            df = load_csv(args.csv)
        else:
            df = await load_from_exchange(args.exchange, args.symbol, args.timeframe, limit=args.limit)
        cfg = BacktestConfig(symbol=args.symbol, timeframe=args.timeframe,
                             strategy_name=args.strategy, start_cash=args.cash)
        result = run_backtest(df, cfg)
        print("\n===== 回测报告 =====")
        for k, v in result["metrics"].items():
            print(f"{k}: {v}")
        print(f"交易数: {len(result['trades'])}")
        out = Path("data") / f"backtest_{args.strategy}.json"
        out.write_text(__import__("json").dumps(result, ensure_ascii=False, default=str))
        print(f"完整结果已保存: {out}")

    asyncio.run(_run())


def cmd_backfill(args: argparse.Namespace) -> None:
    from backtest.data_loader import load_from_exchange

    async def _run():
        df = await load_from_exchange(args.exchange, args.symbol, args.timeframe, limit=args.limit)
        out = Path("data") / f"{args.exchange}_{args.symbol.replace('/', '')}_{args.timeframe}.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out)
        print(f"已保存 {len(df)} 根K线 -> {out}")

    asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto AI Trader")
    sub = parser.add_subparsers(dest="cmd")

    web = sub.add_parser("web", help="启动 Web 服务")
    web.add_argument("--host", default=None)
    web.add_argument("--port", type=int, default=None)

    bt = sub.add_parser("backtest", help="命令行回测")
    bt.add_argument("--source", default="demo", choices=["demo", "csv", "exchange"])
    bt.add_argument("--csv", default="")
    bt.add_argument("--exchange", default="binance")
    bt.add_argument("--symbol", default="BTC/USDT")
    bt.add_argument("--timeframe", default="1h")
    bt.add_argument("--strategy", default="dual_ma")
    bt.add_argument("--cash", type=float, default=10000.0)
    bt.add_argument("--limit", type=int, default=1000)

    bf = sub.add_parser("backfill", help="拉取历史K线到 CSV")
    bf.add_argument("--exchange", default="binance")
    bf.add_argument("--symbol", default="BTC/USDT")
    bf.add_argument("--timeframe", default="1h")
    bf.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()
    if args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "backfill":
        cmd_backfill(args)
    else:
        from core.logging_config import setup_logging
        setup_logging()
        from config.settings import settings
        import uvicorn
        host = args.host or settings.web_host
        port = args.port or settings.web_port
        uvicorn.run("web.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()