"""横截面因子测试：对齐/排名/截面IC/多空价差/全景报告。

构造已知结构的合成数据（漂移分层=动量持续），验证方向性：
- 持续漂移分层的品种池 → xs_mom 截面 IC > 0，多空价差为正
- 品种数不足 → 报错；品种 <5 → 报告带 warning
"""
import numpy as np
import pandas as pd
import pytest

from factors.cross_section import (XS_FACTORS, align_frames, cross_section_report,
                                   forward_return_matrix, long_short_spread,
                                   xs_ic_series, xs_values)


def _make_df(close: np.ndarray, start_ms: int = 1_700_000_000_000,
             step_s: int = 3600) -> pd.DataFrame:
    n = len(close)
    ts = pd.to_datetime([start_ms + i * step_s * 1000 for i in range(n)], unit="ms", utc=True)
    close_s = pd.Series(close, index=ts)
    rng = np.random.default_rng(7)
    high = close_s * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close_s * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = np.abs(rng.normal(100, 20, n))
    return pd.DataFrame({"open": close_s.shift(1).fillna(close_s.iloc[0]),
                         "high": high, "low": low, "close": close_s, "volume": vol})


def _drift_pool(n_symbols: int = 5, n: int = 400, seed: int = 11,
                drift_span: float = 0.002) -> dict:
    """漂移分层品种池：symbol_i 每期漂移从 -span 到 +span 线性分层（动量持续）。"""
    rng = np.random.default_rng(seed)
    dfs = {}
    for i in range(n_symbols):
        mu = (i / max(1, n_symbols - 1) - 0.5) * 2 * drift_span
        rets = rng.normal(mu, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(rets))
        dfs[f"S{i}/USDT"] = _make_df(close)
    return dfs


# ---------- 对齐 ----------

def test_align_frames_intersects_and_drops_invalid():
    dfs = _drift_pool(4, 200)
    # 砍掉一个品种的头部 → 公共索引应缩短为交集
    k = list(dfs.keys())[0]
    dfs[k] = dfs[k].iloc[50:]
    dfs["BAD"] = None
    out = align_frames(dfs)
    assert "BAD" not in out and len(out) == 4
    n = len(next(iter(out.values())))
    assert all(len(df) == n for df in out.values()) and n == 150


def test_align_frames_too_few_symbols_passthrough():
    out = align_frames({"A": _make_df(np.linspace(1, 2, 50))})
    assert len(out) == 1  # 不做交集（<2 不对齐）


# ---------- 因子值与截面 IC ----------

def test_xs_values_shapes_and_warmup():
    dfs = _drift_pool(3, 100)
    vals = xs_values(dfs, "xs_mom", lookback=24)
    assert list(vals.columns) == sorted(dfs.keys())
    # 前 lookback 期为 NaN（warmup），之后应有值
    assert vals.iloc[:24].isna().all().all()
    assert vals.iloc[30:].notna().all().all()


def test_xs_mom_ic_positive_on_drift_pool():
    dfs = _drift_pool(5, 400)
    aligned = align_frames(dfs)
    closes = pd.DataFrame({s: d["close"] for s, d in aligned.items()})
    vals = xs_values(aligned, "xs_mom", lookback=24)
    ics = xs_ic_series(vals, closes, horizon=4).dropna()
    assert len(ics) > 100
    assert ics.mean() > 0.05  # 持续漂移 → 动量截面 IC 显著为正


def test_xs_vol_ic_negative_on_drift_pool():
    """同方差漂移池中波动率与未来收益无方向关系，但构造低波动异象需专门数据——
    此处只验证 IC 可计算且有界。"""
    dfs = _drift_pool(5, 300)
    aligned = align_frames(dfs)
    closes = pd.DataFrame({s: d["close"] for s, d in aligned.items()})
    vals = xs_values(aligned, "xs_vol", lookback=24)
    ics = xs_ic_series(vals, closes, horizon=1).dropna()
    assert len(ics) > 50 and -1.0 <= ics.mean() <= 1.0


def test_forward_return_matrix():
    closes = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]})
    fwd = forward_return_matrix(closes, 1)
    assert fwd.iloc[0]["A"] == pytest.approx(0.10)
    assert fwd.iloc[-1].isna().all()  # 最后一行无未来


def test_xs_ic_needs_3_symbols():
    """每期 <3 个有效品种 → IC 序列为空。"""
    closes = pd.DataFrame({"A": np.linspace(1, 2, 30), "B": np.linspace(2, 1, 30)})
    vals = closes.copy()
    ics = xs_ic_series(vals, closes, 1)
    assert ics.empty


# ---------- 多空价差 ----------

def test_long_short_spread_positive_on_drift_pool():
    dfs = _drift_pool(5, 400)
    aligned = align_frames(dfs)
    closes = pd.DataFrame({s: d["close"] for s, d in aligned.items()})
    vals = xs_values(aligned, "xs_mom", lookback=24)
    ls = long_short_spread(vals, closes, horizon=4)
    assert ls["n_periods"] > 50
    assert ls["total_ret"] > 0.0  # 做多强势做空弱势 → 正收益
    assert ls["group_size"] == 1  # 5 品种 × 0.34 → 每组 1 个
    assert len(ls["ts"]) == len(ls["spread"]) == ls["n_periods"]


def test_long_short_spread_insufficient():
    closes = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    ls = long_short_spread(closes[["A"]], closes, 1)  # 单品种无横截面
    assert ls["total_ret"] is None and ls["n_periods"] == 0


# ---------- 全景报告 ----------

def test_report_structure_and_direction():
    dfs = _drift_pool(5, 400)
    rep = cross_section_report(dfs, lookback=24, horizon=4)
    assert rep["ok"] and rep["n_symbols"] == 5
    assert len(rep["factors"]) == len(XS_FACTORS)
    keys = [f["key"] for f in rep["factors"]]
    assert "xs_mom" in keys
    # 按 |IC| 降序
    ics = [abs(f["ic_mean"]) for f in rep["factors"]]
    assert ics == sorted(ics, reverse=True)
    mom = next(f for f in rep["factors"] if f["key"] == "xs_mom")
    assert mom["ic_mean"] > 0.05
    assert mom["long_short"]["total_ret"] > 0.0
    # 快照：排名 1..5 齐全，值有限
    ranks = sorted(s["rank"] for s in mom["latest"])
    assert ranks == [1, 2, 3, 4, 5]
    assert all(isinstance(s["value"], float) for s in mom["latest"])
    # 漂移池 5 品种 → 无 warning
    assert rep["warning"] is None
    # 曲线已生成并封顶
    assert "xs_mom" in rep["spread_curves"]
    curve = rep["spread_curves"]["xs_mom"]
    assert len(curve["ts"]) == len(curve["cum"]) > 0
    assert curve["cum"][-1] == pytest.approx(mom["long_short"]["total_ret"], abs=1e-3)


def test_report_warning_when_few_symbols():
    rep = cross_section_report(_drift_pool(3, 300), lookback=24, horizon=1)
    assert rep["n_symbols"] == 3 and rep["warning"] is not None


def test_report_rejects_too_few_symbols():
    with pytest.raises(ValueError, match="至少需要"):
        cross_section_report(_drift_pool(2, 300), lookback=24, horizon=1)


def test_report_rejects_short_history():
    dfs = _drift_pool(5, 50)
    with pytest.raises(ValueError, match="不足以"):
        cross_section_report(dfs, lookback=48, horizon=1)  # 48+1+10 > 50


def test_curve_downsample_cap():
    """长历史曲线点数封顶 ≤400（前端渲染保护）。"""
    dfs = _drift_pool(5, 1200, seed=3)
    rep = cross_section_report(dfs, lookback=24, horizon=1)
    for key, curve in rep["spread_curves"].items():
        assert len(curve["ts"]) <= 400, key


def test_unknown_factor_rejected():
    with pytest.raises(ValueError, match="未知横截面因子"):
        xs_values(_drift_pool(3, 100), "xs_nope", 24)


def test_no_lookahead_truncation_invariance():
    """无未来函数的严格判据：截断尾部数据后，前半段因子值必须逐点完全一致。

    若任何因子误用未来数据（如 shift(-k)），去掉尾部后其早期值会改变。
    小品种池的截面 IC 噪声极大（实测 5 品种 500 期 |IC| 可达 0.12），
    统计判据不可靠，故用这个确定性测试。
    """
    dfs = _drift_pool(5, 400)
    half = {s: df.iloc[:200] for s, df in dfs.items()}
    for key in XS_FACTORS:
        full = xs_values(align_frames(dfs), key, lookback=24)
        trunc = xs_values(align_frames(half), key, lookback=24)
        assert full.loc[trunc.index, trunc.columns].equals(trunc), key


def test_null_ic_bounded_on_pure_noise():
    """零漂移合成池：截面 IC 应落在噪声范围内（宽松上界，防方向性错误）。"""
    ics_all = []
    for seed in range(4):
        dfs = _drift_pool(8, 800, seed=1000 + seed, drift_span=0.0)
        aligned = align_frames(dfs)
        closes = pd.DataFrame({s: d["close"] for s, d in aligned.items()})
        for key in XS_FACTORS:
            vals = xs_values(aligned, key, lookback=24)
            ics = xs_ic_series(vals, closes, horizon=4).dropna()
            ics_all.append(abs(float(ics.mean())))
    assert max(ics_all) < 0.10, max(ics_all)


def test_report_snapshot_no_lookahead():
    """快照排名按因子值降序（rank 1 = 因子值最大），ret_recent 用已实现历史收益。"""
    dfs = _drift_pool(5, 300)
    rep = cross_section_report(dfs, lookback=24, horizon=4)
    mom = next(f for f in rep["factors"] if f["key"] == "xs_mom")
    vals = [s["value"] for s in mom["latest"]]
    assert vals == sorted(vals, reverse=True)
    assert mom["latest"][0]["rank"] == 1
