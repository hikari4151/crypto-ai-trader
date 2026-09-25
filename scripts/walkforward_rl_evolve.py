"""ETH/5m 同品种 walk-forward 多折评估（固定 DRL 模型 × 时间窗口）。

背景（用户选择"2"→"1"，2026-09-19）：
  策略只做 ETH/5m，不引入跨品种门（避免误杀单品种专用模型）；
  改为验证"换一段历史也赚"——把 ETH/5m 真实历史切成多折，
  用已部署的 strategy_drl 最佳模型在每折（互不重叠的时间段）上
  greedy 评估（TradingEnv + run_episode，与进化引擎 _cross_symbol_oos 同构）。

纯函数（分折/汇总/复算）在 `drl/walkforward.py`，本脚本复用同一实现；
进化引擎的常规观测也走同一模块（`_observe_strategy_drl_walkforward`）。

判定口径：
  - 每折收益/成交数/最大回撤/年化夏普/末仓位
  - PASS：≥75% 折为正 且 中位数收益>0 且 每折成交数≥min_trades
  - INCONCLUSIVE：存在成交样本不足的折（证据不足，不误判）
  - FAIL：方向一致性不足或中位数≤0

用法：
  .venv/Scripts/python.exe scripts/walkforward_rl_evolve.py \
      --symbol ETH/USDT --timeframe 5m --bars 15000 --folds 4 --min-trades 8 \
      --out data/walkforward_rl_evolve.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drl.walkforward import (  # noqa: E402,F401  （re-export，向后兼容直接 import 本模块）
    _annual_bars,
    build_factor_column,
    expanding_fold_bounds,
    metrics_from_traj,
    summarize_folds,
)

log = logging.getLogger("walkforward_rl_evolve")


def main() -> dict:
    parser = argparse.ArgumentParser(description="ETH/5m 同品种 walk-forward 多折评估")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--bars", type=int, default=15000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--min-trades", type=int, default=8)
    parser.add_argument("--start-cash", type=float, default=10000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--out", default="data/walkforward_rl_evolve.json")
    args = parser.parse_args()

    from config.settings import settings
    from backtest.data_loader import load_klines_cached

    import asyncio
    df = asyncio.run(load_klines_cached(
        settings.default_exchange, args.symbol, args.timeframe,
        limit=args.bars, max_candles=args.bars))
    if df is None or len(df) == 0:
        raise SystemExit("真实K线获取失败（load_klines_cached 返回空）")
    if len(df) > args.bars:
        df = df.iloc[-args.bars:]

    from drl.model_zoo import ModelZoo
    zoo = ModelZoo(settings.data_dir / "models")
    agent = zoo.load_agent("strategy_drl")
    if agent is None:
        raise SystemExit("未找到 strategy_drl 最佳模型（先跑持续进化/手动 DRL 训练）")

    state_window = int(getattr(settings, "evolve_strategy_drl_state_window", 4))
    # 模型是否带级联因子列：读 flat 模型元数据（rl_adaptive 同源），
    # 评估环境必须按训练口径注入同一因子列，否则 state_dim 不匹配。
    factor_composite: Optional[dict] = None
    factor_mu, factor_sd = 0.0, 1.0
    flat_path = settings.data_dir / "models" / "strategy_drl.json"
    if flat_path.exists():
        try:
            md = json.loads(flat_path.read_text(encoding="utf-8"))
            fc = md.get("factor_composite")
            if isinstance(fc, dict) and fc.get("weights"):
                factor_composite = fc
            factor_mu = float(md.get("factor_mu", 0.0) or 0.0)
            factor_sd = float(md.get("factor_sd", 1.0) or 1.0)
        except Exception as e:  # noqa: BLE001
            log.warning("读取 flat 模型元数据失败（按无因子列评估）: %s", e)
    # 基础状态维度（无因子列）：模型 state_dim 大于它时说明带级联因子列
    from drl.env import TradingEnv
    probe = TradingEnv(df.iloc[:min(200, len(df))], start_cash=args.start_cash,
                       fee_rate=args.fee_rate,
                       vol_penalty=float(getattr(settings, "evolve_vol_penalty", 40.0)),
                       slippage=args.slippage, min_trade_zone=0.05,
                       extra_factors=None, state_window=state_window)
    base_dim = probe.state_dim
    factor_mode = "model_metadata" if factor_composite is not None else (
        "deployed_weights_fold_local_approx" if agent.state_dim > base_dim else "none")
    if agent.state_dim > base_dim and factor_composite is None:
        log.warning("模型 state_dim=%d > 基础 %d：flat 无因子元数据，"
                    "将用已部署权重 + 折内局部标准化近似重建因子列",
                    agent.state_dim, base_dim)

    folds_out = []
    for i, (s, e) in enumerate(expanding_fold_bounds(len(df), args.warmup, args.folds), 1):
        seg = df.iloc[s:e]
        if len(seg) < state_window + 10:
            log.warning("跳过第 %d 折（段太短 %d 根）", i, len(seg))
            continue
        from drl.env import run_episode
        extra = None
        if factor_composite is not None:
            extra = build_factor_column(seg, factor_composite.get("weights") or {},
                                        mu=factor_mu, sd=factor_sd)
        elif agent.state_dim > base_dim:
            weights_path = (settings.data_dir / "models"
                            / f"_cascade_weights_{args.symbol.replace('/', '_')}.json")
            if weights_path.exists():
                w = json.loads(weights_path.read_text(encoding="utf-8"))
                raw = build_factor_column(seg, w, mu=0.0, sd=1.0)  # expanding z 原始合成
                if raw is not None:
                    n_train = max(60, int(len(seg) * 0.6))
                    mu = float(raw[:n_train].mean())
                    sd = float(raw[:n_train].std()) + 1e-8
                    extra = (raw - mu) / sd
        env = TradingEnv(seg, start_cash=args.start_cash, fee_rate=args.fee_rate,
                         vol_penalty=float(getattr(settings, "evolve_vol_penalty", 40.0)),
                         slippage=args.slippage, min_trade_zone=0.05,
                         extra_factors=extra, state_window=state_window)
        if env.state_dim != agent.state_dim:
            raise SystemExit(
                f"模型 state_dim={agent.state_dim} 与评估环境 {env.state_dim} 不一致"
                f"（无法重建模型状态；请先让下一次接受轮持久化因子元数据）")
        traj = run_episode(env, lambda st: agent.greedy_action(st))
        m = metrics_from_traj(traj, args.timeframe)
        m.update({"fold": i, "start": str(seg.index[0]), "end": str(seg.index[-1]),
                  "bars": len(seg), "symbol": args.symbol, "timeframe": args.timeframe})
        folds_out.append(m)
        log.info("第 %d 折 [%s ~ %s] ret=%+.4f trades=%d dd=%.4f",
                 i, m["start"], m["end"], m["oos_ret"], m["oos_trades"], m["max_drawdown"])

    summary = summarize_folds(folds_out, min_trades=args.min_trades)
    report = {
        "symbol": args.symbol, "timeframe": args.timeframe,
        "model": "strategy_drl(best)", "state_window": state_window,
        "bars": len(df), "warmup": args.warmup, "min_trades": args.min_trades,
        "factor_mode": factor_mode,
        "folds": folds_out, **summary, "generated_at": time.time(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== walk-forward 多折评估: {args.symbol} {args.timeframe} ===")
    print(f"模型: strategy_drl(best)  state_window={state_window}  K线={len(df)}")
    print(f"{'折':>3} {'区间':>28} {'收益':>9} {'成交':>5} {'最大回撤':>9} {'年化夏普':>9}")
    for f in folds_out:
        print(f"{f['fold']:>3} {f['start']}~{f['end'][:10]:>16} {f['oos_ret']:>+9.4f} "
              f"{f['oos_trades']:>5} {f['max_drawdown']:>9.4f} {f['sharpe_annual']:>9.3f}")
    print(f"\n结论: {summary['verdict']}  （{summary['reason']}）")
    print(f"报告已写入: {out_path}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
