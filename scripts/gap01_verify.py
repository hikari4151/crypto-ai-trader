"""grid GAP-01 完整验收（诊断报告建议第6条(a)）。

对指定标的池跑 detect_overfit（walk-forward/CSCV-PBO/DSR）+ 成本扫描
（run_cost_scan），再用 backtest.gap01_assess.assess_gap01 按 GAP-01 四条
给出 PASS/INCONCLUSIVE/FAIL，输出 JSON 报告。

用法：
  .venv/Scripts/python.exe scripts/gap01_verify.py \
      --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h \
      --bars 15000 --folds 4 --cscv-splits 4 \
      --out data/gap01_grid_report.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def grid_default_params() -> dict:
    from strategies import list_strategies
    for s in list_strategies():
        if s.get("builtin") and s.get("name") == "grid":
            return s.get("default_params") or {}
    return {}


async def _load(symbol: str, timeframe: str, bars: int):
    from config.settings import settings
    from backtest.data_loader import load_klines_cached
    df = await load_klines_cached(settings.default_exchange, symbol, timeframe,
                                  limit=bars, max_candles=bars)
    if df is None or len(df) == 0:
        raise SystemExit(f"K线获取失败: {symbol} {timeframe}")
    return df.iloc[-bars:]


def _run_detect(df, symbol: str, timeframe: str, strategy: str, params: dict,
                folds: int, cscv_splits: int) -> dict:
    from backtest.overfit import OverfitConfig, detect_overfit
    cfg = OverfitConfig(symbol=symbol, timeframe=timeframe,
                        n_folds=folds, fold_ratio=0.2, cscv_splits=cscv_splits)
    rep = detect_overfit(df, strategy, params, cfg)
    avg_trades = (sum(f.oos_trades for f in rep.folds) / len(rep.folds)
                  if rep.folds else 0)
    return {
        "symbol": symbol,
        "verdict": rep.verdict,
        "score": round(rep.score, 2),
        "oos_ret": round(rep.oos_ret, 6),
        "oos_sharpe": round(rep.oos_sharpe, 4),
        "oos_trades": round(avg_trades, 1),
        "stability": round(rep.stability, 4),
        "dsr": round(rep.dsr, 4) if rep.dsr is not None else None,
        "pbo": round(rep.pbo, 4),
        "decay": round(rep.decay, 4),
        "n_folds": rep.n_folds,
    }


def _cost_scan(df, symbol: str, timeframe: str, strategy: str, params: dict) -> dict:
    from backtest.cost_scan import CostScanConfig, run_cost_scan
    return run_cost_scan(df, CostScanConfig(
        symbol=symbol, timeframe=timeframe,
        strategy_name=strategy, strategy_params=params))


def main() -> dict:
    parser = argparse.ArgumentParser(description="GAP-01 完整验收（默认 grid）")
    parser.add_argument("--strategy", default="grid")
    parser.add_argument("--params-json", default="",
                        help='覆盖策略参数，如 \'{"grid_pct":0.005,"qty_per_grid":0.002,"max_positions":10}\'')
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--bars", type=int, default=15000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--cscv-splits", type=int, default=4)
    parser.add_argument("--cost-symbol", default="BTC/USDT")
    parser.add_argument("--out", default="data/gap01_grid_report.json")
    args = parser.parse_args()

    strategy = args.strategy.strip() or "grid"
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.params_json:
        params = json.loads(args.params_json)
    else:
        params = grid_default_params()
    reports = []
    for sym in symbols:
        df = asyncio.run(_load(sym, args.timeframe, args.bars))
        r = _run_detect(df, sym, args.timeframe, strategy, params,
                        args.folds, args.cscv_splits)
        reports.append(r)
        print(f"[{sym}] verdict={r['verdict']} OOS={r['oos_ret']:+.4f} "
              f"trades={r['oos_trades']} DSR={r['dsr']} PBO={r['pbo']}")

    cost = None
    if args.cost_symbol in symbols:
        df = asyncio.run(_load(args.cost_symbol, args.timeframe, args.bars))
        cost = _cost_scan(df, args.cost_symbol, args.timeframe, strategy, params)
        sens = cost.get("sensitivity") or {}
        print(f"[cost-scan {args.cost_symbol}] base_ret={sens.get('base_ret')} "
              f"worst_ret(5x)={sens.get('worst_ret')}")

    from backtest.gap01_assess import assess_gap01
    out = assess_gap01(reports, cost)
    report = {
        "strategy": strategy,
        "strategy_params": params,
        "timeframe": args.timeframe,
        "bars": args.bars,
        "folds": args.folds,
        "cscv_splits": args.cscv_splits,
        "cost_symbol": args.cost_symbol,
        "symbols": reports,
        "cost_scan": cost,
        "verdict": out["verdict"],
        "reason": out["reason"],
        "checks": out["checks"],
        "generated_at": time.time(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n=== GAP-01 验收: {strategy} ===")
    for c in out["checks"]:
        mark = {"True": "PASS", "False": "FAIL", "None": "--"}[str(c["ok"])]
        print(f"  [{mark}] {c['key']}: {c['note']}")
    print(f"\n结论: {out['verdict']}  （{out['reason']}）")
    print(f"报告已写入: {out_path}")
    return report


if __name__ == "__main__":
    main()