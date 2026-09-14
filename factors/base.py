"""因子系统基础类型：因子定义与计算约定。

每个因子是一个纯函数：`compute(df) -> pd.Series`，输入为标准 OHLCV DataFrame
（index 为时间戳，列 open/high/low/close/volume），输出为该因子在每个时点的值。

设计约束：
- 纯向量化（pandas/numpy），输入任意长度，O(n) 或 O(n log n)
- 不得使用未来数据（rolling 窗口右端闭合，shift 取前值）
- 允许返回 NaN（不足窗口期），分析时自动剔除
- 计算必须可重复（无随机）
"""
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

# 因子类别
CATEGORY_MOMENTUM = "momentum"        # 动量
CATEGORY_VOLATILITY = "volatility"    # 波动率
CATEGORY_VOLUME = "volume"            # 量价
CATEGORY_TREND = "trend"              # 趋势
CATEGORY_MEAN_REV = "mean_reversion"  # 均值回归
CATEGORY_MACRO = "macro"              # 宏观/市场状态（用可计算的市场代理）

CATEGORY_LABELS = {
    CATEGORY_MOMENTUM: "动量",
    CATEGORY_VOLATILITY: "波动率",
    CATEGORY_VOLUME: "量价",
    CATEGORY_TREND: "趋势",
    CATEGORY_MEAN_REV: "均值回归",
    CATEGORY_MACRO: "宏观/状态",
}


@dataclass
class Factor:
    """一个量化因子。"""
    key: str                                   # 唯一标识，如 "mom_20"
    name: str                                  # 中文名
    category: str                              # 类别
    description: str                           # 含义说明
    compute: Callable[[pd.DataFrame], pd.Series]  # 计算函数
    default_params: dict = field(default_factory=dict)  # 关键参数（展示用）

    def series(self, df: pd.DataFrame) -> pd.Series:
        """安全执行因子计算；异常时返回空 Series 而不是抛错。"""
        try:
            return self.compute(df)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).exception("[factor] %s 计算失败", self.key)
            return pd.Series(index=df.index, dtype=float)

    def meta(self) -> dict:
        return {
            "key": self.key, "name": self.name,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "description": self.description,
            "default_params": self.default_params,
        }
