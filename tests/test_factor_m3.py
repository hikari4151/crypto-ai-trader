# -*- coding: utf-8 -*-
"""M3 因子统计严谨性 — 回归测试套件。

T1: _newey_west_std 函数正确性
  - 白噪声 ≈ np.std(ddof=1)（差异 < 20%）
  - 正自相关序列（cumsum）> np.std(ddof=1）
  - 短序列 fallback（n<10 → np.std(ddof=1)）

T2: factor_ic 集成 NW 修正
  - ic_std_nw 键存在且计算正确
  - icir 基于 ic_std_nw 计算（与 _newey_west_std 一致）
  - 在 IC 自相关显著的因子上，ic_std_nw > ic_std

T3: min_abs_ic=0.03 收紧
  - DEFAULT_GATES["min_abs_ic"] == 0.03
  - factor_quality_gate 在 0.03 下有效因子数 ≤ 0.01 下
"""
import datetime
import math

import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import generate_demo
from factors.analysis import (DEFAULT_GATES, _newey_west_std, _rolling_ic,
                              factor_ic, factor_quality_gate)
from factors.engine import compute_factor_matrix
from factors.mining import (dynamic_composite, ic_weighted_composite)


# ============================================================
# T1: _newey_west_std 函数单元测试
# ============================================================

def test_nw_white_noise_close_to_raw_std():
    """白噪声序列：NW 标准差 ≈ np.std(ddof=1)，差异 < 20%"""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 500)
    nw = _newey_west_std(x)
    raw = np.std(x, ddof=1)
    rel_diff = abs(nw - raw) / raw
    assert rel_diff < 0.20, f"nw={nw:.4f} raw={raw:.4f} diff={rel_diff:.2%}"


def test_nw_autocorrelated_larger_than_raw_std():
    """平稳高自相关序列（AR(1), phi=0.9）：NW 标准差 > np.std(ddof=1)"""
    rng = np.random.default_rng(7)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    nw = _newey_west_std(x)
    raw = np.std(x, ddof=1)
    assert nw > raw, f"nw={nw:.4f} <= raw={raw:.4f}"


def test_nw_short_series_fallback():
    """短序列（n<10）：直接返回 np.std(ddof=1)"""
    for n in range(2, 10):
        x = np.arange(1, n + 1, dtype=float)
        assert _newey_west_std(x) == np.std(x, ddof=1)


def test_nw_constant_series():
    """常数列：返回 0.0"""
    x = np.ones(50)
    assert _newey_west_std(x) == 0.0


def test_nw_negative_autocorrelation():
    """负自相关 AR(1)（phi=-0.5）：NW 标准差 < np.std(ddof=1)"""
    rng = np.random.default_rng(42)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = -0.5 * x[i - 1] + rng.normal(0, 1)
    nw = _newey_west_std(x)
    raw = np.std(x, ddof=1)
    assert nw < raw, f"nw={nw:.4f} >= raw={raw:.4f}（负自相关下 NW 应 < raw）"


# ============================================================
# T2: factor_ic 集成 NW 修正
# ============================================================

def test_factor_ic_ic_std_nw_key_exists():
    """factor_ic 返回字典包含 ic_std_nw 和 ic_std 键"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    r = factor_ic(mat[mat.columns[0]], df["close"], h=1, method="rank")
    assert "ic_std_nw" in r, "缺少 ic_std_nw 键"
    assert "ic_std" in r, "缺少 ic_std 键（保留字段）"


def test_factor_ic_ic_std_nw_correctly_computed():
    """ic_std_nw 值 = round(_newey_west_std(ics), 6)"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    for c in mat.columns:
        r = factor_ic(mat[c], df["close"], h=1, method="rank")
        if r["n"] > 5:
            # 重现 IC 序列
            ics = _rolling_ic(mat[c], df["close"], h=1, method="rank", window=30).dropna()
            expected = round(_newey_west_std(ics.to_numpy(dtype=float)), 6)
            assert r["ic_std_nw"] == expected, (
                f"factor {c}: ic_std_nw={r['ic_std_nw']} != expected={expected}"
            )


def test_factor_ic_icir_uses_newey_west_std():
    """icir = round(mean / (ic_std_nw + 1e-12), 6)"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    for c in mat.columns:
        r = factor_ic(mat[c], df["close"], h=1, method="rank")
        if r["n"] > 5:
            ics = _rolling_ic(mat[c], df["close"], h=1, method="rank", window=30).dropna()
            expected = float(ics.mean() / (_newey_west_std(ics.to_numpy(dtype=float)) + 1e-12))
            expected_rounded = round(expected, 6)
            assert r["icir"] == expected_rounded, (
                f"factor {c}: icir={r['icir']} != expected={expected_rounded}"
            )


def _regime_trend_data(n: int = 2000, seed: int = 11, vol: float = 0.01,
                       regime_len: int = 400, drift: float = 0.0006) -> pd.DataFrame:
    """生成趋势切换型合成OHLCV（促成因子IC序列的正自相关）。"""
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    t = 0
    while t < n:
        s = rng.choice([-1.0, 1.0])
        rets[t: t + regime_len] = s * drift + rng.normal(0, vol, regime_len)
        t += regime_len
    rets = rets[:n]
    close = 100.0 * np.exp(np.cumsum(rets))
    step_s = 3600
    start_ts = (
        int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        // step_s * step_s - n * step_s
    )
    idx = pd.date_range(
        start=pd.Timestamp(start_ts, unit="s", tz="UTC"), periods=n, freq="h"
    )
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 100.0}, index=idx
    )


def _ac1(x: np.ndarray) -> float:
    """lag-1 自相关系数（Pearson 型）。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum(x[1:] * x[:-1]) / denom)


def test_factor_ic_nw_std_gt_raw_std_when_ic_autocorrelated():
    """IC 序列存在正自相关时，ic_std_nw > ic_std。

    使用趋势切换型合成数据 + trend_regime 因子（其 IC 在趋势切换下具正自相关）。
    """
    df = _regime_trend_data(n=2000, seed=11)
    mat = compute_factor_matrix(df)
    close = df["close"]
    # 找 IC 自相关 > 0.2 的因子
    found = False
    for c in mat.columns:
        ics = _rolling_ic(mat[c], close, h=1, method="rank", window=30).dropna()
        if len(ics) < 20:
            continue
        a1 = _ac1(ics.to_numpy())
        if a1 <= 0.2:
            continue
        r = factor_ic(mat[c], close, h=1, method="rank")
        if r["n"] > 5:
            assert r["ic_std_nw"] >= r["ic_std"] - 1e-9, (
                f"factor {c}: AC1={a1:.3f}, ic_std_nw={r['ic_std_nw']:.4f} "
                f"< ic_std={r['ic_std']:.4f}"
            )
            found = True
    assert found, "未找到 IC 自相关 > 0.2 的因子，无法验证 NW 修正行为"


# ============================================================
# T3: min_abs_ic 收紧
# ============================================================

def test_default_gates_min_abs_ic_updated():
    """DEFAULT_GATES["min_abs_ic"] 已收紧到 0.03"""
    assert DEFAULT_GATES["min_abs_ic"] == 0.03, (
        f"当前值={DEFAULT_GATES['min_abs_ic']}，期望 0.03"
    )


def test_min_abs_ic_03_filters_more_than_01():
    """min_abs_ic=0.03 下有效因子数 ≤ 0.01 下（单调性保证）。"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    close = df["close"]
    old_gates = {"min_abs_ic": 0.01}
    new_gates = {"min_abs_ic": 0.03}
    old_valid = sum(
        factor_quality_gate(mat[c], close, h=1, gates=old_gates)["valid"]
        for c in mat.columns
    )
    new_valid = sum(
        factor_quality_gate(mat[c], close, h=1, gates=new_gates)["valid"]
        for c in mat.columns
    )
    assert new_valid <= old_valid, (
        f"new_valid={new_valid} > old_valid={old_valid}（违反单调性）"
    )


# ============================================================
# M2: mining.py 默认 min_abs_ic=0.03 参数生效
# ============================================================

def test_ic_weighted_composite_default_min_abs_ic():
    """ic_weighted_composite 默认 min_abs_ic=0.03：只选 |rank_ic|>=0.03 的因子"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    close = df["close"]
    res = ic_weighted_composite(mat, close, h=1)
    sel = res["selected"]
    assert sel, "应至少选出一个 |rank_ic|>=0.03 的因子"
    for s in sel:
        assert abs(s["rank_ic"]) >= 0.03, (
            f"因子 {s['key']} rank_ic={s['rank_ic']:.4f} < 0.03"
        )
    # 单调性：0.03 的入选集合是 0.01 的子集
    res_lo = ic_weighted_composite(mat, close, h=1, min_abs_ic=0.01)
    keys_hi = {s["key"] for s in sel}
    keys_lo = {s["key"] for s in res_lo["selected"]}
    assert keys_hi <= keys_lo, "0.03 的入选集合应是 0.01 的子集"


def test_dynamic_composite_default_min_abs_ic():
    """dynamic_composite 默认 min_abs_ic=0.03：非零权重处 |IC|>=0.03"""
    df = generate_demo(n=2000, seed=5)
    mat = compute_factor_matrix(df)
    close = df["close"]
    res = dynamic_composite(mat, close, h=1)
    wh = res["weight_history"]
    # 重建 ic_avail（与 dynamic_composite 内部逻辑一致）
    ics = {c: _rolling_ic(mat[c], close, h=1, method="rank", window=120)
           for c in mat.columns}
    ic_matrix = pd.DataFrame(ics, index=mat.index)
    ic_avail = ic_matrix.shift(1).ffill()
    cells = 0
    for c in mat.columns:
        for w, v in zip(wh[c].to_numpy(), ic_avail[c].to_numpy()):
            if abs(w) > 0:
                cells += 1
                assert v == v and abs(v) >= 0.03 - 1e-9, (
                    f"因子 {c}: 非零权重 |IC|={abs(v):.4f} < 0.03"
                )
    assert cells > 0, "dynamic_composite 应有因子入选"


# ============================================================
# m5: ic_std_nw 边界回归测试
# ============================================================

def test_factor_ic_std_nw_empty_ic_series():
    """IC 序列为空（K线<30）时 ic_std_nw=0.0，不报错不产生 NaN"""
    rng = np.random.default_rng(1)
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0, index=idx)
    factor = pd.Series(np.sin(np.arange(n)), index=idx)
    r = factor_ic(factor, close, h=1, method="rank")
    assert r["n"] == 0
    assert r["ic_std_nw"] == 0.0
    assert r["icir"] == 0.0
    assert math.isfinite(r["ic_std_nw"])


def test_factor_ic_std_nw_single_ic():
    """IC 序列仅 1 个样本时 ic_std_nw 兜底为 0.0（无 NaN），icir=0.0"""
    rng = np.random.default_rng(2)
    n = 40  # 30 ≤ n < 60 → 恰 1 个非重叠 IC 窗口
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0, index=idx)
    factor = pd.Series(np.sin(np.arange(n)), index=idx)
    r = factor_ic(factor, close, h=1, method="rank")
    assert r["n"] == 1
    assert math.isfinite(r["ic_std_nw"])
    assert r["ic_std_nw"] == 0.0
    assert r["icir"] == 0.0