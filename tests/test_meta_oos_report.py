"""元策略 OOS 必须完成真实评估并返回启用报告。"""

import numpy as np

from strategies.meta import MetaControllerEnv


def _df(n=240):
    import pandas as pd
    close = 100.0 + np.sin(np.arange(n) / 10.0)
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1000.0),
    })


def test_meta_environment_records_final_equity_snapshot():
    env = MetaControllerEnv(_df(), ["dual_ma"], warmup=20)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(0)
    assert env._final_equity > 0
    assert env._final_position_ratio == 0.0
