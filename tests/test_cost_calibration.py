"""成本标定工具的回归测试（scripts/drl_cost_calibration.py）。

锁死两件事：
1) Roll(1984) 价差估计在"纯买卖价差跳动"上能还原出价差量级、在"无价差趋势"上为 0；
2) 频率/周期标定的核心口径——gross（零成本）与 net（含成本）的差 == 成本×换手，
   即"盈亏平衡成本 = gross / 换手"这一判据本身自洽。
"""
import numpy as np
import pandas as pd
import pytest

from scripts.drl_cost_calibration import roll_spread_bp


def _bounce_series(n: int = 20000, spread_bp: float = 10.0, seed: int = 0) -> np.ndarray:
    """纯价差跳动：中间价恒定，成交价随机落在买价/卖价（iid）。

    Roll 估计量假设成交方向 iid —— 用确定性交替构造会退化成 2×价差
    （连续两笔必然换边），故这里用随机方向，估计量才收敛到真实价差。
    """
    rng = np.random.default_rng(seed)
    q = rng.choice([-1.0, 1.0], size=n)
    half = spread_bp / 20000.0
    return 100.0 * (1.0 + q * half)


def test_roll_spread_recovers_bid_ask_bounce():
    est = roll_spread_bp(_bounce_series(spread_bp=10.0))
    assert 5.0 < est < 20.0, f"应还原到 10bp 量级，实际 {est}"


def test_roll_spread_zero_for_trending_series():
    s = 100.0 * np.cumprod(np.full(500, 1.001))  # 单调上涨，无负自协方差
    assert roll_spread_bp(s) < 1e-6


def test_roll_spread_handles_short_series():
    assert roll_spread_bp(np.asarray([1.0, 2.0])) == 0.0
    assert roll_spread_bp(np.asarray([])) == 0.0


def test_breakeven_cost_identity():
    """盈亏平衡成本 = gross / 换手：用 env 直接验证成本线性可加。

    同一策略、同一段行情，成本 c 下的净收益应等于毛收益 - c×换手（一阶口径），
    这是频率标定"margin = breakeven/实际成本"判据的基础。
    """
    from drl.env import TradingEnv, run_episode

    n = 800
    rng = np.random.default_rng(3)
    close = 100.0 * np.cumprod(1.0 + 0.0005 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": np.full(n, 100.0)}, index=idx)
    acts = rng.integers(0, 5, size=n).tolist()

    def _run(fee: float, slip: float) -> tuple[float, float]:
        env = TradingEnv(df, fee_rate=fee, slippage=slip, min_trade_zone=0.01)
        box = {"i": 0}

        def _policy(_s):
            i = box["i"]
            box["i"] = i + 1
            return int(acts[i % len(acts)])

        traj = run_episode(env, _policy)
        pos = np.asarray([i["position_ratio"] for i in traj["infos"]], dtype=float)
        turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
        return float(traj["total_ret"]), turnover

    gross, turnover = _run(0.0, 0.0)
    net, turnover2 = _run(0.002, 0.001)
    # 换手是环境实收成本基数：同一动作序列下与成本无关
    assert turnover == pytest.approx(turnover2, rel=0.05)
    assert turnover > 1.0
    # 成本是"每单位换手 × (fee+slip)"，净收益必须显著低于毛收益
    assert net < gross
    drag = gross - net
    approx = 0.003 * turnover
    assert drag == pytest.approx(approx, rel=0.35), \
        f"成本侵蚀应≈成本率×换手：drag={drag:.4f} 期望≈{approx:.4f}"


# ============ 成本记账（turnover / cost_paid） ============

def _mk_df(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": np.full(len(close), 100.0)}, index=idx)


def test_no_trade_policy_has_zero_turnover_and_cost():
    from drl.env import TradingEnv, run_episode
    df = _mk_df(100.0 * np.cumprod(1.0 + np.full(300, 0.001)))
    env = TradingEnv(df, fee_rate=0.001, slippage=0.0005)
    traj = run_episode(env, lambda s: 0)          # 始终空仓 → 从不成交
    assert traj["turnover"] == 0.0
    assert traj["cost_paid"] == 0.0


def test_single_entry_turnover_and_cost_identity():
    """一次满仓买入：换手≈1（成本基数=成交名义/权益），且 cost≈换手×费率×权益。"""
    from drl.env import TradingEnv, run_episode
    df = _mk_df(np.full(300, 100.0))              # 恒价：只有一次建仓，无后续成交
    env = TradingEnv(df, fee_rate=0.001, slippage=0.0005, min_trade_zone=0.05)
    traj = run_episode(env, lambda s: 4)
    assert traj["turnover"] == pytest.approx(1.0, abs=0.02)
    assert traj["cost_paid"] == pytest.approx(10000.0 * 0.0015, rel=0.05)


def test_turnover_ignores_passive_position_drift():
    """锁死"外部按相邻仓位差求和"的口径错误：价格漂移造成的被动仓位变化不是换手。

    单调上涨且始终满仓时，仓位比例因价格漂移而波动（旧口径会累计出可观"换手"），
    但实际只在建仓时成交一次 → 环境记账必须保持 ≈1。
    """
    from drl.env import TradingEnv, run_episode
    rng = np.random.default_rng(1)
    rets = 0.002 + rng.normal(0, 0.01, 500)
    df = _mk_df(100.0 * np.cumprod(1.0 + rets))
    env = TradingEnv(df, fee_rate=0.001, slippage=0.0005, min_trade_zone=0.05)
    traj = run_episode(env, lambda s: 4)
    pos = np.asarray([i["position_ratio"] for i in traj["infos"]], dtype=float)
    drift_metric = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    assert traj["turnover"] < 2.0, f"实收换手应≈1（仅建仓），实际 {traj['turnover']}"
    assert drift_metric > traj["turnover"], \
        "相邻仓位差口径应显著大于实收换手（被动漂移）"


def test_cost_paid_zero_when_costless():
    from drl.env import TradingEnv, run_episode
    rng = np.random.default_rng(2)
    df = _mk_df(100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 300)))
    env = TradingEnv(df, fee_rate=0.0, slippage=0.0, min_trade_zone=0.05)
    traj = run_episode(env, lambda s: 4)
    assert traj["cost_paid"] == 0.0
    assert traj["turnover"] > 0.9       # 有成交但零成本
