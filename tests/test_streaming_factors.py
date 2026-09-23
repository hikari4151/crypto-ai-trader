"""P1-② 流式组合因子对照测试（2026-09-23）：
StreamingCompositeFactor（增量）必须与 CompositeFactorEvaluator.eval（每根全量
pandas）**逐位一致**——部署因子列口径不变式。用真实 v49 配方（6因子+mu/sd）。
"""
import json
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from factors.mining import CompositeFactorEvaluator
from drl.streaming_factors import StreamingCompositeFactor

# 2026-09-23 实证结论：因子侧流式不可安全等价——eval 对"最近 400 根窗口"重算
#（EWM/expanding 从窗口起点重新开始），流式持续累积跨窗口，>400 根后语义冲突
#（diff≈0.08）。本专项已关闭，测试保留（xfail）以文档化问题与验收标准。
pytestmark = pytest.mark.xfail(
    reason="因子侧流式不可安全等价（窗口重算 vs 持续累积，diff≈0.08），专项已实证关闭",
    strict=False)


def _load_v49_recipe():
    flat = json.loads((settings.data_dir / "models" / "strategy_drl.json").read_text(encoding="utf-8"))
    fc = flat.get("factor_composite") or {}
    return {
        "weights": fc.get("weights") or {},
        "mu": float(flat.get("factor_mu", 0.0) or 0.0),
        "sd": float(flat.get("factor_sd", 1.0) or 1.0),
    }


def _gen_data(n: int = 600, seed: int = 7):
    rng = np.random.default_rng(seed)
    close = 2700.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    high = close * (1 + rng.random(n) * 0.004)
    low = close * (1 - rng.random(n) * 0.004)
    volume = 100.0 + rng.random(n) * 60.0
    return close, high, low, volume


def _pair_stream_vs_batch(seed: int = 7, n: int = 600):
    recipe = _load_v49_recipe()
    close, high, low, volume = _gen_data(n, seed)
    closes, highs, lows, vols = [], [], [], []
    sf = StreamingCompositeFactor(recipe["weights"], recipe["mu"], recipe["sd"])
    maxdiff = 0.0
    for c, h, l, v in zip(close, high, low, volume):
        closes.append(float(c)); highs.append(float(h)); lows.append(float(l)); vols.append(float(v))
        if len(closes) > 400:
            closes = closes[-400:]; highs = highs[-400:]; lows = lows[-400:]; vols = vols[-400:]
        df = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                           "close": closes, "volume": vols})
        ref = CompositeFactorEvaluator(recipe["weights"], recipe["mu"], recipe["sd"]).eval(df)
        got = sf.update(float(c), float(h), float(l), float(v))
        if ref is not None and got is not None:
            maxdiff = max(maxdiff, abs(ref - got))
    return maxdiff


def test_streaming_factors_match_batch():
    md = _pair_stream_vs_batch()
    print("max diff:", md)
    assert md < 1e-9, f"流式因子与 eval 不一致 max={md:.2e}"


@pytest.mark.parametrize("seed", [1, 42, 2026])
def test_streaming_factors_match_across_seeds(seed):
    md = _pair_stream_vs_batch(seed=seed)
    assert md < 1e-9, f"seed={seed} max={md:.2e}"