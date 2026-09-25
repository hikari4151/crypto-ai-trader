"""滑点敏感性验证（2026-09-21，v44 实盘关卡）：同一模型 4 折 × 15000 根
× fee=0.001 × slippage 0.0005 vs 0.0010（2×滑点）——量化滑点放大对收益的侵蚀。

判定：2×滑点档仍 3/4 折正且中位数>0 → 滑点鲁棒（PASS）；否则敏感（FAIL）。
数据：ETH/USDT 5m 真实 15000 根（与 gap01/walkforward 同口径）。

用法：
  .venv/Scripts/python.exe scripts/slippage_sensitivity_drl.py --out data/slippage_sensitivity_v44.json
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


def _run_tier(df, agent, sw, warmup, folds, fee, slip, vol_penalty,
              factor_composite, factor_mu, factor_sd):
    from drl.env import TradingEnv, run_episode
    from drl.walkforward import (build_factor_column, expanding_fold_bounds,
                                 metrics_from_traj)
    out = []
    for i, (s, e) in enumerate(expanding_fold_bounds(len(df), warmup, folds), 1):
        seg = df.iloc[s:e]
        extra = build_factor_column(seg, (factor_composite or {}).get("weights") or {},
                                    mu=factor_mu, sd=factor_sd)
        env = TradingEnv(seg, start_cash=10000.0, fee_rate=fee,
                         vol_penalty=vol_penalty, slippage=slip,
                         min_trade_zone=0.05, extra_factors=extra,
                         state_window=sw)
        if env.state_dim != agent.state_dim:
            raise RuntimeError(f"state_dim 不匹配: env={env.state_dim} agent={agent.state_dim}")
        traj = run_episode(env, lambda st: agent.greedy_action(st))
        m = metrics_from_traj(traj, "5m")
        n = len(seg)
        m.update({"fold": i, "start": str(seg.index[0]), "end": str(seg.index[-1]),
                  "trades_per_1000": round(m["oos_trades"] / n * 1000.0, 2)})
        out.append(m)
    return out


async def main() -> dict:
    parser = argparse.ArgumentParser(description="v44 滑点敏感性验证")
    parser.add_argument("--bars", type=int, default=15000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slip-base", type=float, default=0.0005)
    parser.add_argument("--slip-stress", type=float, default=0.0010)
    parser.add_argument("--version", type=int, default=None,
                        help="指定模型版本（默认=当前部署 best）")
    parser.add_argument("--out", default="data/slippage_sensitivity_v44.json")
    args = parser.parse_args()

    from config.settings import settings
    from backtest.data_loader import load_klines_cached
    from drl.model_zoo import ModelZoo
    from drl.walkforward import summarize_folds

    zoo = ModelZoo(settings.data_dir / "models")
    agent = zoo.load_agent("strategy_drl", args.version)
    if agent is None:
        raise SystemExit(f"无模型 strategy_drl v{args.version or 'best'}")
    if args.version is not None:
        _payload = zoo._read_version_payload("strategy_drl", args.version)
        _meta = dict(_payload.get("_meta") or {})
        fc = _meta.get("factor_composite") or {}
        mu = float(_meta.get("factor_mu", 0.0) or 0.0)
        sd = float(_meta.get("factor_sd", 1.0) or 1.0)
        _model_version = args.version
    else:
        flat = json.loads(zoo._flat_path("strategy_drl").read_text(encoding="utf-8"))
        fc = flat.get("factor_composite") or {}
        mu = float(flat.get("factor_mu", 0.0) or 0.0)
        sd = float(flat.get("factor_sd", 1.0) or 1.0)
        _model_version = zoo.best_version("strategy_drl")

    df = await load_klines_cached(
        getattr(settings, "default_exchange", "binance"), "ETH/USDT", "5m",
        limit=args.bars, max_candles=args.bars)
    if df is None or len(df) == 0:
        raise SystemExit("K线获取失败")
    df = df.iloc[-args.bars:]
    sw = max(1, int(getattr(settings, "evolve_strategy_drl_state_window", 4)))
    warmup = max(300, sw * 50)
    vol = float(getattr(settings, "evolve_vol_penalty", 40.0))

    print(f"模型: v{_model_version} | bars={len(df)} folds={args.folds} "
          f"fee={args.fee} slip={args.slip_base} vs {args.slip_stress}")
    t1 = await asyncio.to_thread(_run_tier, df, agent, sw, warmup, args.folds,
                                 args.fee, args.slip_base, vol, fc, mu, sd)
    t2 = await asyncio.to_thread(_run_tier, df, agent, sw, warmup, args.folds,
                                 args.fee, args.slip_stress, vol, fc, mu, sd)
    s1 = summarize_folds(t1, min_trades=8)
    s2 = summarize_folds(t2, min_trades=8)

    rows = []
    for a, b in zip(t1, t2):
        rows.append({"fold": a["fold"], "ret_base": round(a["oos_ret"], 6),
                     "ret_stress": round(b["oos_ret"], 6),
                     "drag_pct": round((a["oos_ret"] - b["oos_ret"]) * 100.0, 4),
                     "trades": a["oos_trades"]})
        print(f"  折{rows[-1]['fold']}: base={rows[-1]['ret_base']:+.4f} "
              f"stress(2x滑点)={rows[-1]['ret_stress']:+.4f} drag={rows[-1]['drag_pct']}% "
              f"trades={rows[-1]['trades']}")
    print(f"base verdict={s1['verdict']} ({s1['reason'][:40]})")
    print(f"2x滑点 verdict={s2['verdict']} ({s2['reason'][:40]})")
    robust = s2["verdict"] == "PASS"
    print(f"=> 滑点鲁棒性: {'PASS（2x滑点仍正期望）' if robust else 'FAIL（2x滑点侵蚀明显）'}")
    report = {
        "model_version": _model_version,
        "bars": len(df), "fold_count": args.folds, "fee": args.fee,
        "slip_base": args.slip_base, "slip_stress": args.slip_stress,
        "folds": rows,
        "base_verdict": s1, "stress_verdict": s2,
        "slippage_robust": robust,
        "generated_at": time.time(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("saved:", out_path)
    return report


if __name__ == "__main__":
    asyncio.run(main())
