"""因子计算引擎：从标准 OHLCV DataFrame 批量计算因子矩阵。

- compute_factor_matrix: 一次算全部因子 → DataFrame（列=因子key，行=时间）
- 因子计算为纯 pandas 向量化，复用 indicators 数据时也可传入预计算快照
- 支持 GPU 后端（与 fast_engine 的 cupy 路径对齐）：因子量小时 numpy 已足够，
  此处保留统一的 backend 参数便于扩展
"""
import logging
from typing import Iterable, Optional

import pandas as pd

from .library import factor_keys, get_factor

log = logging.getLogger(__name__)

# 默认分析需要的 OHLCV 列
_REQUIRED = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> None:
    """校验 OHLCV 结构，缺失列时给出清晰报错。"""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"因子计算需要 OHLCV 列 {_REQUIRED}，缺少: {missing}")
    if len(df) < 5:
        raise ValueError("因子计算需要至少 5 根K线")


def compute_factor_matrix(df: pd.DataFrame, keys: Optional[Iterable[str]] = None,
                          backend: str = "numpy") -> pd.DataFrame:
    """批量计算因子矩阵。

    df: 标准 OHLCV（index=时间戳）
    keys: 要计算的因子 key 列表；None=全部内置因子
    backend: 预留，当前统一 numpy 向量化
    返回: DataFrame，列=因子key，index 与 df 一致
    """
    validate_ohlcv(df)
    keys = list(keys) if keys is not None else factor_keys()
    out = {}
    for key in keys:
        f = get_factor(key)
        if f is None:
            log.warning("[factor] 跳过未知因子 %s", key)
            continue
        s = f.series(df)
        if s is not None:
            out[key] = s.astype(float)
    if not out:
        raise ValueError("没有可计算的因子")
    mat = pd.DataFrame(out, index=df.index)
    # 全 NaN 列剔除（避免后续相关性/IC 崩坏）
    valid = mat.columns[mat.notna().sum() > 5]
    return mat[list(valid)]


def factor_frame_meta(keys: Iterable[str]) -> list[dict]:
    """返回因子元信息列表（前端渲染用）。"""
    out = []
    for key in keys:
        f = get_factor(key)
        if f is not None:
            out.append(f.meta())
    return out


def safe_zscores(mat: pd.DataFrame) -> pd.DataFrame:
    """逐列 z-score 标准化（去均值/除标准差），NaN 保持。"""
    mu = mat.mean()
    sd = mat.std()
    return (mat - mu) / (sd + 1e-12)
