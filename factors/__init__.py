"""量化因子系统：因子库 / 因子计算引擎 / IC分析 / AI挖掘与合成 / 模型因子。"""
from .analysis import (DEFAULT_GATES, factor_correlation, factor_exposure,
                       factor_fitness, factor_group_returns, factor_ic,
                       factor_ic_table, factor_quality_gate, factor_turnover,
                       find_redundant, forward_returns, periodic_ic_refresh)
from .base import Factor, CATEGORY_LABELS
from .cross_section import XS_FACTORS, cross_section_report, xs_values
from .engine import (compute_factor_matrix, factor_frame_meta, safe_zscores,
                     validate_ohlcv)
from .library import (factor_is_live, factor_keys, factors_by_category,
                      get_factor, get_factor_status, ignoring_factor_liveness,
                      list_factors)

__all__ = [
    "Factor", "CATEGORY_LABELS",
    "factor_keys", "get_factor", "list_factors", "factors_by_category",
    "factor_is_live", "get_factor_status", "ignoring_factor_liveness",
    "compute_factor_matrix", "factor_frame_meta", "safe_zscores", "validate_ohlcv",
    "factor_ic", "factor_ic_table", "factor_group_returns",
    "factor_correlation", "factor_exposure", "find_redundant", "forward_returns",
    "factor_turnover", "factor_fitness", "factor_quality_gate", "DEFAULT_GATES",
    "periodic_ic_refresh",
    "XS_FACTORS", "cross_section_report", "xs_values",
]
