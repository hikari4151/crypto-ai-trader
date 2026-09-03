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


def cmd_cost_scan(args: argparse.Namespace) -> None:
    """成本压力测试（P1-4）：同一策略在费用/滑点放大倍数下的敏感性扫描。"""
    from backtest.cost_scan import CostScanConfig, run_cost_scan
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange

    async def _run():
        if args.source == "demo":
            df = generate_demo(timeframe=args.timeframe)
        elif args.source == "csv":
            df = load_csv(args.csv)
        else:
            df = await load_from_exchange(args.exchange, args.symbol, args.timeframe, limit=args.limit)
        cfg = CostScanConfig(symbol=args.symbol, timeframe=args.timeframe,
                             strategy_name=args.strategy, strategy_params={},
                             start_cash=args.cash, base_fee_rate=args.fee_rate,
                             base_slippage=args.slippage)
        report = run_cost_scan(df, cfg)
        print("\n===== 成本压力测试 =====")
        print(f"基准 fee={args.fee_rate} slippage={args.slippage}")
        print(f"敏感性: {report['sensitivity']}")
        print(f"{'fee×':>6} {'slip×':>6} {'总收益':>10} {'夏普':>8} {'回撤':>8} {'交易数':>6} {'成本':>10}")
        for r in report["scan"]:
            print(f"{r['fee_mult']:>6} {r['slip_mult']:>6} "
                  f"{'-' if r['total_return'] is None else round(r['total_return'], 4):>10} "
                  f"{'-' if r['sharpe'] is None else round(r['sharpe'], 2):>8} "
                  f"{'-' if r['max_drawdown'] is None else round(r['max_drawdown'], 4):>8} "
                  f"{r['total_trades'] if r['total_trades'] is not None else '-':>6} "
                  f"{'-' if r['total_costs'] is None else round(r['total_costs'], 2):>10}")

    asyncio.run(_run())


def cmd_portfolio(args: argparse.Namespace) -> None:
    """多标的组合层回测（P2-11）：研究层组合视角（等权/波动率倒数权重）。"""
    import json as _json
    from backtest.data_loader import generate_demo, load_csv, load_from_exchange
    from backtest.portfolio import PortfolioConfig, run_portfolio_backtest

    async def _load(symbol: str):
        if args.source == "demo":
            return generate_demo(timeframe=args.timeframe)
        elif args.source == "csv":
            return load_csv(args.csv)
        else:
            return await load_from_exchange(args.exchange, symbol, args.timeframe, limit=args.limit)

    async def _run():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if len(symbols) < 2:
            print("组合回测需要至少 2 个标的（--symbols 'BTC/USDT,ETH/USDT'）")
            return
        data = {}
        for s in symbols:
            try:
                data[s] = await _load(s)
                print(f"已加载 {s}: {len(data[s])} 根K线")
            except Exception as e:  # noqa: BLE001
                print(f"加载 {s} 失败: {e}")
        if len(data) < 2:
            print("可用标的数据不足（至少 2 个），组合回测终止")
            return
        cfg = PortfolioConfig(symbols=symbols, strategy_name=args.strategy,
                              strategy_params={}, weights=args.weights,
                              start_cash=args.cash, timeframe=args.timeframe,
                              fee_rate=args.fee_rate, slippage=args.slippage)
        report = run_portfolio_backtest(data, cfg)
        print("\n===== 组合回测报告 =====")
        print(f"权重分配: {report['weights']}")
        print("\n各标的:")
        for ps in report["per_symbol"]:
            m = ps["metrics"]
            print(f"  {ps['symbol']} ({ps['strategy']}, w={ps['weight']}): "
                  f"ret={m['total_return']:.4f} sharpe={m['sharpe']:.2f} dd={m['max_drawdown']:.4f}")
        print("\n组合:")
        pm = report["portfolio_metrics"]
        print(f"  组合收益={pm['total_return']:.4f} 年化={pm['annual_return']:.4f} "
              f"夏普={pm['sharpe']:.2f} 回撤={pm['max_drawdown']:.4f}")
        print(f"  基准买入持有={report['benchmark']['buy_hold_ret']:.4f}")
        print("\n标的收益相关性:")
        corr = report["correlation"]
        print("        " + " ".join(f"{s.split('/')[0]:>9}" for s in symbols))
        for s1 in symbols:
            row = " ".join(f"{corr[s1].get(s2, 0):>9.3f}" for s2 in symbols)
            print(f"{s1.split('/')[0]:>7}  {row}")
        out = Path("data") / "portfolio_result.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(_json.dumps(report, ensure_ascii=False, default=str))
        print(f"\n完整结果已保存: {out}")

    asyncio.run(_run())


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
                             strategy_name=args.strategy, start_cash=args.cash,
                             fee_rate=args.fee_rate, slippage=args.slippage,
                             participation_rate=args.participation_rate,
                             funding_rate=args.funding_rate)
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
    bt.add_argument("--fee-rate", type=float, default=0.001, dest="fee_rate")
    bt.add_argument("--slippage", type=float, default=0.0005)
    bt.add_argument("--participation-rate", type=float, default=0.0, dest="participation_rate")
    bt.add_argument("--funding-rate", type=float, default=0.0, dest="funding_rate")

    cs = sub.add_parser("cost-scan", help="成本压力测试（P1-4）")
    cs.add_argument("--source", default="demo", choices=["demo", "csv", "exchange"])
    cs.add_argument("--csv", default="")
    cs.add_argument("--exchange", default="binance")
    cs.add_argument("--symbol", default="BTC/USDT")
    cs.add_argument("--timeframe", default="1h")
    cs.add_argument("--strategy", default="dual_ma")
    cs.add_argument("--cash", type=float, default=10000.0)
    cs.add_argument("--limit", type=int, default=1000)
    cs.add_argument("--fee-rate", type=float, default=0.001, dest="fee_rate")
    cs.add_argument("--slippage", type=float, default=0.0005)

    pf = sub.add_parser("portfolio", help="多标的组合回测（P2-11）")
    pf.add_argument("--source", default="demo", choices=["demo", "csv", "exchange"])
    pf.add_argument("--csv", default="")
    pf.add_argument("--exchange", default="binance")
    pf.add_argument("--symbols", default="BTC/USDT,ETH/USDT", help="逗号分隔的标的列表")
    pf.add_argument("--timeframe", default="1h")
    pf.add_argument("--strategy", default="dual_ma")
    pf.add_argument("--weights", default="equal", choices=["equal", "vol_inv"], help="组合权重方案")
    pf.add_argument("--cash", type=float, default=20000.0)
    pf.add_argument("--limit", type=int, default=1000)
    pf.add_argument("--fee-rate", type=float, default=0.001, dest="fee_rate")
    pf.add_argument("--slippage", type=float, default=0.0005)

    bf = sub.add_parser("backfill", help="拉取历史K线到 CSV")
    bf.add_argument("--exchange", default="binance")
    bf.add_argument("--symbol", default="BTC/USDT")
    bf.add_argument("--timeframe", default="1h")
    bf.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()
    if args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "cost-scan":
        cmd_cost_scan(args)
    elif args.cmd == "portfolio":
        cmd_portfolio(args)
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