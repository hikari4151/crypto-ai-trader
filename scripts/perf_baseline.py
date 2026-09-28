"""性能基线门禁（2026-09-28，性能/延迟探索落地）：关键路径耗时不得灾难性回归。

设计：本项目多数热路径已向量化优化（fast_engine / support_resistance_series /
IncrIndicators），单点延迟都很低（实测 3000 根回测 ~64ms、compute_latest ~32ms、
K 线处理 p50 <0.01ms）。但此前**没有任何性能门禁**——某次改动若把回测拖成 O(n²)
（历史上 fast_engine 之前的 O(n²) 指标重算、combo 因子每根全量重算都是教训），
测试全绿也不会发现。本脚本为关键路径建立"防灾难性回归"基线：

- 阈值 = 实测中位数 × 余量（默认 4×，`--tolerance` 可调）：CI 机器抖动/并行负载
  会让耗时波动，绝对毫秒断言必然 flaky；4× 只拦"数量级回归"（O(n²) 化/重算热点）。
- 基线文件 .optim/perf_baseline.json 由 `--update` 重新校准（实测取中位数）。
- 纳入 verify_all 门禁链 + ci.yml（fast 集跑一次，与 coverage_gate 同级防线）。

用法：
  python scripts/perf_baseline.py                # 0=通过 1=超阈值（打印各路径耗时）
  python scripts/perf_baseline.py --report       # 只打印不执行门禁
  python scripts/perf_baseline.py --update       # 重新校准基线（谨慎：改阈值前先确认
                                                 # 是"有意优化后更快"，而非掩盖回归）
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_FILE = ROOT / ".optim" / "perf_baseline.json"
DEFAULT_TOLERANCE = 4.0
# 阈值下限（ms）：亚毫秒路径（如 K 线引擎 p50<0.01ms）基线≈0，`基线×余量` 仍是 0，
# 任何实测波动都 > 0 → 门禁必然红。下限 = 线性增长可被接受的绝对预算，只拦数量级回归。
FLOOR_MS = 25.0


def _median_ms(fn, repeat: int = 3) -> float:
    """测单条路径中位耗时（ms），重复取中位数抗尖峰。"""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def measure() -> dict[str, float]:
    """测量全部关键路径耗时（ms）。每项独立 import，避免互相污染缓存。"""
    out: dict[str, float] = {}

    # 1) 回测引擎（双引擎基线同款 demo 数据）
    from backtest.data_loader import generate_demo
    from backtest.engine import BacktestConfig, run_backtest
    from backtest.fast_engine import run_backtest_fast

    df = generate_demo(n=3000)
    cfg = BacktestConfig(start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
    out["backtest_engine_3000"] = _median_ms(lambda: run_backtest(df, cfg))
    out["backtest_fast_3000"] = _median_ms(lambda: run_backtest_fast(df, cfg))

    # 2) 指标 compute_latest（AI 解读 / 实时快照共用）
    from indicators.technical import compute_latest
    ohlcv = [[1700000000000 + i * 3600000, 100.0 + i * 0.1, 101.0 + i * 0.1,
              99.0 + i * 0.1, 100.0 + i * 0.1, 1000.0] for i in range(3000)]
    out["compute_latest_3000"] = _median_ms(lambda: compute_latest(ohlcv))

    # 2b) evolve 批量候选评估（combo 策略 × 候选数——历史上"每根全量重算"热点
    #     被向量化优化后不再是灾难，但批量乘上候选数后仍需防回归）
    def bench_batch():
        from strategies.factor_signal import FactorSignalStrategy
        df2 = generate_demo(n=5000)
        t0 = time.perf_counter()
        for i in range(10):
            st = FactorSignalStrategy()
            st.update_params({"factor": "combo", "mode": "trend",
                              "buy_threshold": 0.4 + i * 0.01,
                              "combo_spec": '{"vol_ratio": 0.8, "mom_10": 0.8, "rsi_14": %.1f}'
                                            % (0.2 + i * 0.03),
                              "size_pct": 0.3})
            cfg2 = BacktestConfig(start_cash=10000.0, fee_rate=0.001, slippage=0.0005,
                                  strategy_name="factor_signal",
                                  strategy_params={"spec": st})
            run_backtest(df2, cfg2)
        return (time.perf_counter() - t0) * 1000.0

    out["evolve_batch_10x5000"] = bench_batch()

    # 3) 实时 K 线处理（引擎 _on_candle 热路径，100 根）
    def bench_klines():
        import asyncio
        from core.bus import EventBus
        from core.database import Database
        from core.events import Event, EventType
        from engine.trading_engine import TradingEngine
        from strategies.base import Strategy

        class _Quiet(Strategy):
            name = "perf_baseline"
            default_params = {"size_pct": 0.5}

            def on_candle(self, ctx):
                return None  # 不出信号，只测处理开销

            def on_fill(self, symbol, side, price):
                pass

        async def run():
            db = Database("sqlite+aiosqlite:///:memory:")
            bus = EventBus()
            eng = TradingEngine(db, bus)
            eng.running = True
            eng.paper = True
            eng.symbol = "BTC/USDT"
            eng.timeframe = "1h"
            eng.strategy = _Quiet()
            eng.trade_on_open = False

            async def _snap(*a, **k):
                return {"symbol": "BTC/USDT", "timeframe": "1h",
                        "candles": [[1700000000000, 50000.0, 50000.0, 50000.0,
                                     50000.0, 100.0]],
                        "closes": [50000.0],
                        "indicators": {"close": 50000.0, "vol_ratio": 1.0}}
            eng._current_snapshot = _snap
            ts = []
            for i in range(100):
                snap = {"symbol": "BTC/USDT", "timeframe": "1h",
                        "candles": [[1700000000000 + i * 3600000, 50000.0 + i,
                                     50000.0 + i, 50000.0 + i, 50000.0 + i, 100.0]],
                        "closes": [50000.0 + i],
                        "indicators": {"close": 50000.0 + i, "vol_ratio": 1.0}}
                t0 = time.perf_counter()
                await eng._on_candle(Event(EventType.MARKET_CANDLE,
                                           {"symbol": "BTC/USDT", "snapshot": snap},
                                           source="test"))
                ts.append((time.perf_counter() - t0) * 1000.0)
            await db.close()
            return statistics.median(ts)

        return asyncio.run(run())

    out["kline_engine_100"] = bench_klines()
    return out


def load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        return {}
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="性能基线门禁（防灾难性回归）")
    ap.add_argument("--report", action="store_true", help="只打印测量结果，不执行门禁")
    ap.add_argument("--update", action="store_true",
                    help="重新校准基线文件（谨慎：确认是真实优化而非掩盖回归）")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="阈值余量倍数（默认 4×，防 CI 抖动误报；只拦数量级回归）")
    args = ap.parse_args(argv)

    print("[perf] 测量关键路径耗时…", flush=True)
    measured = measure()
    baseline = load_baseline()

    if args.update:
        data = {"paths": measured, "tolerance": args.tolerance,
                "note": "2026-09-28 实测校准（median × tolerance 作为阈值，防灾难性回归）"}
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[perf] 已校准基线 → {BASELINE_FILE}")
        return 0

    if not baseline:
        print("[perf][FAIL] 无基线文件，先运行: python scripts/perf_baseline.py --update")
        return 1

    tol = float(baseline.get("tolerance", args.tolerance))
    ref = baseline.get("paths") or {}
    issues = []
    for name, val in sorted(measured.items()):
        base = ref.get(name)
        if base is None:
            issues.append(f"{name}: {val:.1f}ms 无基线参照（--update 校准后纳入）")
            continue
        limit = max(float(base) * tol, FLOOR_MS)
        flag = "OK" if val <= limit else "FAIL"
        print(f"[perf] {name:26s} {val:8.1f}ms  (基线 {base:.1f}ms × {tol:.1f} = {limit:.1f}ms)  {flag}")
        if val > limit:
            issues.append(f"{name}: {val:.1f}ms > {limit:.1f}ms（基线 {base:.1f}ms，"
                          f"可能 O(n²) 化/重算热点回归）")
    if args.report:
        return 0 if not issues else 1
    if issues:
        print("[perf][FAIL] " + "; ".join(issues))
        return 1
    print("[perf] ALL PASS（关键路径耗时均在基线余量内）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
