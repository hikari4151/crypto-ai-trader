"""DRL 成本标定：滑点复核 + 决策频率/周期标定（2026-10）。

背景（docs/DRL训练根本性逻辑错误-诊断报告.md §8.6）：修完奖励/探索/经验量后，
默认口径（fee=0.001 taker + slippage=0.0005）下 OOS 仍为负，而**仅去掉 0.05% 滑点**
OOS 就转正（+1.61%）——瓶颈从"学习能力"转移到"成本口径"。本工具回答两个问题：

1) 滑点复核（--mode slippage，纯数据、秒级）：
   `paper_slippage=0.0005` 从未被实测验证过（LIVE_READINESS GAP-04 待办）。
   本工具用本地K线**数据驱动**地估一个上界：
     - 冲击成本：平方根律 impact ≈ c·σ_bar·sqrt(Q/V)（Q=本单名义，V=该根报价成交额）
     - 买卖价差：Roll(1984) 序列协方差估计 s = 2·sqrt(-cov(r_t, r_{t-1}))（下界）
   并输出"成本/波动率"比值——同一个 k 在 5m 与 1h 上的**相对成本**完全不同，
   这才是决策频率该看的口径。

2) 决策频率/周期标定（--mode frequency，需训练）：
   对每个 (周期, 决策间隔 k) 训练一个策略，把 OOS 收益拆成
     gross（零成本，= 策略自身 alpha） 与 net（生产成本）
   并算**盈亏平衡成本** breakeven = gross / 换手，与实际成本比较：
     margin = breakeven / 实际成本 —— >1 才是"成本之内还有 alpha"。

用法：
  .venv/Scripts/python.exe scripts/drl_cost_calibration.py --mode slippage
  .venv/Scripts/python.exe scripts/drl_cost_calibration.py --mode frequency --quick
  .venv/Scripts/python.exe scripts/drl_cost_calibration.py --mode all --out .optim/cost_calib.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 每单位换手（|Δ仓位|）被收的成本：手续费 + 滑点（与 TradingEnv/回测撮合同口径）
DEFAULT_FEE = 0.001
DEFAULT_SLIP = 0.0005


# ============ 1. 滑点复核（数据驱动） ============

def roll_spread_bp(close: np.ndarray) -> float:
    """Roll(1984) 有效价差估计（bp）：s = 2·sqrt(-cov(r_t, r_{t-1}))。

    买卖价差会让价格在买价/卖价间跳动，产生负的一阶自协方差。
    这是**下界**估计（低频K线把跳动平均掉了一部分），仅用于量级判断。
    """
    if len(close) < 3:
        return 0.0
    r = np.diff(np.log(close))
    if len(r) < 2:
        return 0.0
    cov = float(np.cov(r[:-1], r[1:])[0, 1])
    if cov >= 0:
        return 0.0
    return float(2.0 * np.sqrt(-cov) * 10000.0)


def audit_one(df: pd.DataFrame, start_cash: float, pos_deltas: tuple,
              coeffs: tuple) -> dict:
    """单个 (标的, 周期) 的滑点上界复核。"""
    close = df["close"].to_numpy(float)
    volume = df["volume"].to_numpy(float) if "volume" in df else np.zeros(len(df))
    quote_vol = volume * close            # 库内 volume 是 base 量 → ×收盘价得报价成交额
    ret_bp = np.diff(close) / close[:-1] * 10000.0
    sigma_bp = float(np.std(ret_bp)) if len(ret_bp) else 0.0
    roll_bp = roll_spread_bp(close)
    out = {"bars": int(len(df)), "sigma_bar_bp": round(sigma_bp, 2),
           "roll_half_spread_bp": round(roll_bp / 2.0, 3),
           "quote_vol_median": float(np.median(quote_vol)) if len(quote_vol) else 0.0,
           "rows": []}
    for pd_ in pos_deltas:
        q = start_cash * pd_                       # 本单名义（USDT）
        part = q / np.maximum(quote_vol, 1e-9)
        for c in coeffs:
            # 冲击成本按平方根律；用该根实际 participation 的 P95（保守档）
            impact_bp = c * sigma_bp * np.sqrt(part)
            total_bp = roll_bp / 2.0 + impact_bp
            out["rows"].append({
                "dpos": pd_, "notional": q, "coef": c,
                "participation_median": round(float(np.median(part)), 8),
                "participation_p95": round(float(np.percentile(part, 95)), 8),
                "impact_bp_median": round(float(np.median(impact_bp)), 4),
                "impact_bp_p95": round(float(np.percentile(impact_bp, 95)), 4),
                "slip_bp_p95": round(float(np.percentile(total_bp, 95)), 4),
            })
    return out


def run_slippage_audit(args) -> dict:
    from backtest import kline_store
    report = {"generated_at": time.time(), "start_cash": args.start_cash,
              "pos_deltas": list(args.pos_deltas), "coeffs": list(args.coeffs),
              "series": {}}
    print("\n=== 滑点复核（Q=本金×|Δ仓位|，成本 = 半价差(Roll) + 平方根律冲击）===")
    print(f"生产假设：fee={DEFAULT_FEE} + slippage={DEFAULT_SLIP} "
          f"→ 每单位换手 {int((DEFAULT_FEE + DEFAULT_SLIP) * 10000)}bp\n")
    for sym in args.symbols:
        for tf in args.timeframes:
            df = kline_store.load_df(args.exchange, sym, tf, args.bars)
            if df is None or len(df) < 200:
                print(f"  {sym} {tf}: 本地数据不足（{0 if df is None else len(df)} 根），跳过")
                continue
            res = audit_one(df, args.start_cash, tuple(args.pos_deltas),
                            tuple(args.coeffs))
            report["series"][f"{sym}:{tf}"] = res
            cost_bp = (DEFAULT_FEE + DEFAULT_SLIP) * 10000.0
            print(f"  {sym} {tf}: {res['bars']} 根 | 单根σ={res['sigma_bar_bp']}bp | "
                  f"Roll半价差={res['roll_half_spread_bp']}bp | "
                  f"成本/σ={cost_bp / max(res['sigma_bar_bp'], 1e-9):.2f}")
            for r in res["rows"]:
                if r["dpos"] != max(args.pos_deltas):
                    continue
                print(f"      满仓换手(名义{r['notional']:.0f}U) c={r['coef']}: "
                      f"参与率中位={r['participation_median']:.2e} "
                      f"冲击P95={r['impact_bp_p95']:.3f}bp "
                      f"→ 滑点P95合计={r['slip_bp_p95']:.3f}bp"
                      f"{'  ← 远低于 5bp 假设' if r['slip_bp_p95'] < DEFAULT_SLIP * 10000 else ''}")
    return report


# ============ 2. 决策频率/周期标定（训练） ============

def _split_three_way(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    n_train = max(300, int(n * train_ratio))
    n_val = max(100, int(n * val_ratio))
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def _eval_policy(agent, seg: pd.DataFrame, cfg: dict, fee: float, slip: float):
    """用给定成本口径跑贪心评估，返回 (收益, 被收费换手, 平均仓位)。

    换手取环境自己记的 `_turnover_units`（= 实际成交名义/权益，成本基数），
    而不是外部对"相邻仓位差"求和——后者会把价格漂移造成的被动仓位变化算进去，
    实测能把真实换手高估约 3 倍，breakeven/margin 全部失真。
    """
    from drl.env import TradingEnv, run_episode, policy_env_kwargs
    env = TradingEnv(seg, start_cash=float(cfg.get("start_cash", 10000.0)),
                     fee_rate=fee, slippage=slip, vol_penalty=0.0, cost_scale=1.0,
                     min_trade_zone=float(cfg.get("min_trade_zone", 0.05)),
                     state_window=int(cfg.get("state_window", 1)),
                     **policy_env_kwargs(cfg))
    traj = run_episode(env, lambda s: agent.greedy_action(s))
    pos = np.asarray([i["position_ratio"] for i in traj["infos"]], dtype=float)
    avg_pos = float(np.mean(pos)) if len(pos) else 0.0
    return float(traj["total_ret"]), float(traj.get("turnover", 0.0)), avg_pos


def _bh_ret(seg: pd.DataFrame) -> float:
    """同段买入持有收益（基准：DDL 学不到择时时它最多也就是这个数）。"""
    c = seg["close"].to_numpy(float)
    return float(c[-1] / c[0] - 1.0) if len(c) > 1 else 0.0


def run_frequency_scan(args) -> list:
    from backtest import kline_store
    from drl.agent import train_drl
    from drl.evolve_engine import _strategy_drl_train_cfg

    cells = [(tf, k) for tf in args.timeframes for k in args.intervals]
    rows = []
    print(f"\n=== 决策频率/周期标定（episodes={args.episodes}, "
          f"fee={args.fee}, slip={args.slip}）===")
    print("口径：净=生产成本下收益；毛=同策略零成本收益；基准=同段买入持有；"
          "择时=gross-基准(>0 才是真会择时)；换手/决策=环境实收成本基数/决策数")
    print(f"{'周期':>5}{'k':>4}{'验证段':>10}{'OOS净':>10}{'OOS毛':>10}{'基准':>10}"
          f"{'择时':>10}{'均仓':>7}{'换手':>8}{'换手/决':>9}{'要求年化夏普':>13}")
    for tf, k in cells:
        df = kline_store.load_df(args.exchange, args.symbol, tf, args.bars)
        if df is None or len(df) < 2000:
            print(f"{tf:>5}{k:>4}  本地数据不足，跳过")
            continue
        cfg = _strategy_drl_train_cfg(args.episodes)
        cfg.update(reward_mode="excess", decision_interval=k, seed=args.seed,
                   n_episodes=args.n_episodes, slippage=args.slip)
        cfg["episode_window"] = args.episode_window
        cfg["fee_rate"] = args.fee
        t0 = time.time()
        res = train_drl(df, cfg)
        _, _, oos_df = _split_three_way(df, float(cfg.get("train_ratio", 0.6)),
                                        float(cfg.get("val_ratio", 0.2)))
        agent = res["agent"]
        # OOS 切 3 段逐段评估（单段行情运气太强，与生产 oos_segments=3 同构）
        n_seg = 3
        seg_len = len(oos_df) // n_seg
        nets, grosses, bhs, poss, turns, bes = [], [], [], [], [], []
        for j in range(n_seg):
            s = j * seg_len
            e = len(oos_df) if j == n_seg - 1 else (j + 1) * seg_len
            seg = oos_df.iloc[s:e]
            if len(seg) < 50:
                continue
            _net, turn, pos = _eval_policy(agent, seg, cfg, args.fee, args.slip)
            _gross, _turn_g, _ = _eval_policy(agent, seg, cfg, 0.0, 0.0)
            nets.append(_net)
            grosses.append(_gross)
            bhs.append(_bh_ret(seg))
            poss.append(pos)
            turns.append(turn)
            # 逐段算盈亏平衡成本再取中位数（"中位数之差"与"差值中位数"不等价，
            # 混用会让 breakeven/margin 与成本侵蚀口径对不上）
            bes.append((_gross / _turn_g) if _turn_g > 1e-9 else 0.0)
        if not nets:
            continue
        net = float(np.median(nets))
        gross = float(np.median(grosses))
        bh = float(np.median(bhs))
        turnover = float(np.sum(turns))
        avg_pos = float(np.mean(poss))
        breakeven = float(np.median(bes))
        decisions = max(1, int(len(oos_df) / max(1, k)))
        tpd = turnover / decisions
        cost_per_unit = args.fee + args.slip
        row_hurdle = _hurdle_row(oos_df, tf, k, tpd, args.fee + args.slip)
        close = oos_df["close"].to_numpy(float)
        sigma_bp = float(np.std(np.diff(close) / close[:-1]) * 10000.0) if len(close) > 2 else 0.0
        row = {"timeframe": tf, "decision_interval": k, "bars": int(len(df)),
               "val_ret": res.get("best_val_ret"),
               "oos_net": round(net, 6), "oos_gross": round(gross, 6),
               "bh_ret": round(bh, 6), "timing_value": round(gross - bh, 6),
               "avg_position": round(avg_pos, 4), "turnover": round(turnover, 3),
               "turnover_per_decision": round(tpd, 4),
               "breakeven_bp": round(breakeven * 10000.0, 3),
               "actual_cost_bp": round(cost_per_unit * 10000.0, 3),
               "margin": round(breakeven / cost_per_unit, 3) if cost_per_unit > 0 else 0.0,
               "cost_over_sigma": round(cost_per_unit * 10000.0 / max(sigma_bp, 1e-9), 3),
               "required_annual_sharpe": row_hurdle["required_annual_sharpe"],
               "seg_net": [round(x, 5) for x in nets],
               "seg_gross": [round(x, 5) for x in grosses],
               "seg_bh": [round(x, 5) for x in bhs],
               "elapsed": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"{tf:>5}{k:>4}{row['val_ret']!s:>10}{row['oos_net']:>10.4f}"
              f"{row['oos_gross']:>10.4f}{row['bh_ret']:>10.4f}"
              f"{row['timing_value']:>10.4f}{row['avg_position']:>7.2f}"
              f"{row['turnover']:>8.1f}{tpd:>8.3f}"
              f"{row['required_annual_sharpe']:>10.2f}")
    return rows


def _summarize(rows: list) -> None:
    if not rows:
        return
    print("\n=== 结论 ===")
    for r in rows:
        verdict = []
        if r["timing_value"] > 0:
            verdict.append(f"有择时价值(+{r['timing_value']:.4f})")
        else:
            verdict.append(f"无择时价值({r['timing_value']:+.4f})")
        if r["margin"] > 1.0:
            verdict.append(f"成本可覆盖(margin {r['margin']:.2f})")
        else:
            verdict.append(f"成本不可覆盖(margin {r['margin']:.2f})")
        print(f"  {r['timeframe']} k={r['decision_interval']}: " + "；".join(verdict))
    best = max(rows, key=lambda r: r["oos_net"])
    print(f"净收益最优: {best['timeframe']} k={best['decision_interval']} "
          f"OOS净={best['oos_net']:+.4f}（毛={best['oos_gross']:+.4f} "
          f"基准={best['bh_ret']:+.4f}）")


# ============ 3. 成本壁垒表（决策频率/周期的"最小能力要求"） ============

def _bars_per_year(timeframe: str) -> float:
    """每年多少根K线（用于把"每决策成本"折成年化夏普要求）。"""
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24,
               "2h": 12, "4h": 6, "6h": 4, "12h": 2, "1d": 1}.get(timeframe, 24)
    return per_day * 365.0


def _hurdle_row(df: pd.DataFrame, tf: str, k: int,
                turnover_per_decision: float, cost_per_unit: float) -> dict:
    """单个 (周期, k) 的成本壁垒（单一定义处：壁垒表与频率扫描共用）。

    要求年化夏普 = (每决策成本 / 持有期σ) × sqrt(每年决策数)。
    """
    close = df["close"].to_numpy(float)
    logret = np.diff(np.log(close))
    if len(logret) <= k:
        return {"sigma_k_bp": 0.0, "cost_per_decision_bp": 0.0,
                "required_decision_sharpe": 0.0, "required_annual_sharpe": 0.0,
                "band": "数据不足"}
    kr = np.convolve(logret, np.ones(k), mode="valid")   # 重叠 k 根持有期收益
    sigma_k_bp = float(np.std(kr) * 10000.0)
    n_per_year = _bars_per_year(tf) / k
    cost_dec = turnover_per_decision * cost_per_unit * 10000.0
    s_dec = cost_dec / max(sigma_k_bp, 1e-9)
    s_ann = s_dec * np.sqrt(n_per_year)
    if s_ann < 1.0:
        band = "可争取"
    elif s_ann < 3.0:
        band = "极难"
    elif s_ann < 10.0:
        band = "近乎不可能"
    else:
        band = "不可能"
    return {"sigma_k_bp": round(sigma_k_bp, 1),
            "cost_per_decision_bp": round(cost_dec, 3),
            "required_decision_sharpe": round(s_dec, 4),
            "required_annual_sharpe": round(s_ann, 2),
            "band": band}


def run_hurdle_table(args) -> list:
    """成本壁垒：在给定成本与换手纪律下，每个 (周期, k) 要求策略有多强。

    口径（全部模型无关、纯算术，不含任何 DRL 假设）：
      σ_k        = k 根持有期的收益标准差（bp）
      每决策成本 = 每次决策的换手量 × (fee+slip)
      要求每决策超额 = 每决策成本
      要求年化夏普 = (要求每决策超额 / σ_k) × sqrt(每年决策数)

    为什么用"年化夏普要求"而不是"完美前视上界"：完美前视会在 28000 根上
    复利成几百倍，得到"任何成本都能覆盖"的废结论；而年化夏普是交易员
    真正用来判断"这点成本我能不能挣回来"的口径（优秀策略 1~3，顶级 5+）。
    """
    from backtest import kline_store
    cost_per_unit = args.fee + args.slip
    rows = []
    print(f"\n=== 成本壁垒表（fee={args.fee} slip={args.slip} → 每单位换手 "
          f"{cost_per_unit * 10000:.0f}bp；换手纪律 {args.turnover}/决策）===")
    print(f"{'周期':>5}{'k':>4}{'持有':>8}{'σ_k bp':>9}{'每决策成本bp':>13}"
          f"{'要求决策夏普':>13}{'要求年化夏普':>13}{'可达性':>10}")
    for tf in args.timeframes:
        df = kline_store.load_df(args.exchange, args.symbol, tf, args.bars)
        if df is None or len(df) < 2000:
            continue
        for k in args.intervals:
            h = _hurdle_row(df, tf, k, args.turnover, cost_per_unit)
            if h["band"] == "数据不足":
                continue
            rows.append({"timeframe": tf, "decision_interval": k,
                         "hold": f"{k}{tf}", **h,
                         "n_decisions_per_year": round(_bars_per_year(tf) / k, 1)})
            print(f"{tf:>5}{k:>4}{f'{k}{tf}':>8}{h['sigma_k_bp']:>9.1f}"
                  f"{h['cost_per_decision_bp']:>13.2f}"
                  f"{h['required_decision_sharpe']:>13.3f}"
                  f"{h['required_annual_sharpe']:>13.2f}{h['band']:>10}")
    ok = [r for r in rows if r["band"] == "可争取"]
    print("\n成本算术上「可争取」的格子: "
          + (", ".join(f"{r['timeframe']} k={r['decision_interval']}"
                       f"(年化夏普要求 {r['required_annual_sharpe']})" for r in ok)
             if ok else "无 —— 这些 (周期,k) 在成本算术上已被排除"))
    print("注：要求年化夏普 = 每决策成本/持有期σ × sqrt(每年决策数)；"
          "换手纪律越松（每次决策改仓越多）要求越高。")
    return rows


async def main() -> dict:
    ap = argparse.ArgumentParser(description="DRL 成本标定（滑点复核 + 决策频率/周期）")
    ap.add_argument("--mode", choices=["slippage", "frequency", "hurdle", "all"],
                    default="all")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbols", nargs="*", default=["ETH/USDT", "BTC/USDT"])
    ap.add_argument("--symbol", default="ETH/USDT", help="频率扫描用的标的")
    ap.add_argument("--timeframes", nargs="*", default=["5m", "1h"])
    ap.add_argument("--intervals", nargs="*", type=int, default=[2, 4, 8])
    ap.add_argument("--bars", type=int, default=30000)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--n-episodes", type=int, default=16)
    ap.add_argument("--episode-window", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE)
    ap.add_argument("--slip", type=float, default=DEFAULT_SLIP)
    ap.add_argument("--start-cash", type=float, default=10000.0)
    ap.add_argument("--pos-deltas", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--coeffs", nargs="*", type=float, default=[0.5, 1.0, 1.5])
    ap.add_argument("--turnover", type=float, default=0.15,
                    help="每次决策的平均换手量（|Δ仓位|；默认 0.15=实测 1h k=4 水平）")
    ap.add_argument("--quick", action="store_true", help="快速档：episodes=8")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.quick:
        args.episodes = 8

    report: dict = {"args": vars(args)}
    if args.mode in ("slippage", "all"):
        report["slippage"] = run_slippage_audit(args)
    if args.mode in ("hurdle", "all"):
        report["hurdle"] = run_hurdle_table(args)
    if args.mode in ("frequency", "all"):
        rows = run_frequency_scan(args)
        report["frequency"] = rows
        _summarize(rows)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nsaved:", p)
    return report


if __name__ == "__main__":
    asyncio.run(main())
