"""P1-② 流式特征引擎对照测试（2026-09-23）：
流式（增量）输出必须与 drl.env._precompute_features（全量 pandas）**逐位一致**——
训练/部署端口径不变式，数值漂移会让模型行为漂移。
"""
import numpy as np
import pandas as pd
import pytest

from drl.env import _precompute_features
from drl.streaming_features import StreamingFeatureEngine


def _gen_data(n: int = 1000, seed: int = 7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.004, n)
    close = 2700.0 * np.exp(np.cumsum(rets))
    volume = 100.0 + rng.random(n) * 50.0
    return close, volume


def test_streaming_matches_batch_close_to_zero():
    close, volume = _gen_data()
    df = pd.DataFrame({"close": close, "volume": volume})
    batch = _precompute_features(df)          # (n, 13) 全量 pandas
    eng = StreamingFeatureEngine()
    rows = [eng.update(float(c), float(v)) for c, v in zip(close, volume)]
    stream = np.array(rows)                    # (n, 13) 增量
    assert stream.shape == batch.shape, (stream.shape, batch.shape)
    diff = np.abs(batch - stream)
    per_col = diff.max(axis=0)
    print("per-col max diff:", np.round(per_col, 10))
    assert diff.max() < 1e-8, f"流式与全量不一致 max={diff.max():.2e}"


def test_streaming_tail_rows_match():
    close, volume = _gen_data()
    df = pd.DataFrame({"close": close, "volume": volume})
    batch = _precompute_features(df)
    eng = StreamingFeatureEngine()
    for c, v in zip(close, volume):
        eng.update(float(c), float(v))
    # 取引擎最近一行的内部状态（重新播放到末尾）与该行 batch 对比
    # 直接重放并取最后一行
    eng2 = StreamingFeatureEngine()
    last = None
    for c, v in zip(close, volume):
        last = eng2.update(float(c), float(v))
    assert np.abs(last - batch[-1]).max() < 1e-8


@pytest.mark.parametrize("seed", [1, 42, 2026])
def test_streaming_matches_across_seeds(seed):
    close, volume = _gen_data(seed=seed)
    df = pd.DataFrame({"close": close, "volume": volume})
    batch = _precompute_features(df)
    eng = StreamingFeatureEngine()
    rows = [eng.update(float(c), float(v)) for c, v in zip(close, volume)]
    diff = np.abs(batch - np.array(rows)).max()
    assert diff < 1e-8, f"seed={seed} diff={diff:.2e}"
