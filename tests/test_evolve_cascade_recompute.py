"""TDD（3号修复）：级联因子窗口同步——npy 陈旧/失配时回退到「已部署权重复算」。

观察项（2026-09-19 诊断）：factor_miner 连续回退 → npy 停在 v307（09-18 21:26），
strategy_drl 每次加载都因「窗口尾时间戳不一致」被拒（当日 21 次拒绝/0 次成功），
导致策略 DRL 无因子特征、OOS 不交易、32 连拒。

修复：`_load_cascade_factor_values` 在 npy 缺失/窗口失配时，用已部署的
`_cascade_weights_<symbol>.json` 在**当轮 df** 上按 `composite_factor(z_mode="expanding")`
复算组合因子——与 factor_miner 训练端同式、与 rl_adaptive 实盘复算同口径，
天然按当前窗口对齐（无错位、无前视），不再依赖陈旧 npy 的尾戳匹配。
"""
import json

import numpy as np

from backtest.data_loader import generate_demo
from core.bus import EventBus
from drl.evolve_engine import EvolveEngine


def _make_engine(tmp_path) -> EvolveEngine:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    return EvolveEngine(object(), EventBus(), models)


def test_cascade_load_falls_back_to_recompute_when_npy_stale(tmp_path):
    """npy 尾戳比当前 df 旧（滚动窗口右移/连续回退）时，必须复算而不是返回 None。"""
    symbol = "ETH/USDT"
    sym_key = symbol.replace("/", "_")
    df = generate_demo(n=300, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    weights = {"mom_7": -0.05, "bb_width": 0.06}
    models = tmp_path / "models"
    models.mkdir()
    (models / f"_cascade_weights_{sym_key}.json").write_text(
        json.dumps(weights), encoding="utf-8")
    # 陈旧 npy + companion（tail 比当前 df 晚 1 小时前的窗口）——旧逻辑在此拒绝注入
    old_tail = int(df.index[-1].timestamp()) - 3600
    np.save(str(models / f"_cascade_factor_values_{sym_key}.npy"),
            np.zeros(len(df)))
    (models / f"_cascade_factor_values_{sym_key}.json").write_text(
        json.dumps({"tail_ts": old_tail, "n": len(df)}), encoding="utf-8")

    engine = _make_engine(tmp_path)
    out = engine._load_cascade_factor_values(df, symbol=symbol)

    assert out is not None, "npy 窗口失配时应回退到已部署权重复算，而不是拒绝注入"
    assert len(out) == len(df)
    # 复算值应跟随当轮行情，而不是陈旧 npy 的全 0
    assert float(np.abs(np.asarray(out)).sum()) > 0.0


def test_cascade_load_prefers_fresh_npy_when_tail_matches(tmp_path):
    """当轮 npy 尾戳一致时仍走原快路径（保持既有行为，不重复复算）。"""
    symbol = "ETH/USDT"
    sym_key = symbol.replace("/", "_")
    df = generate_demo(n=300, start_price=3000.0, vol=0.02,
                       timeframe="5m", seed=7)
    models = tmp_path / "models"
    models.mkdir()
    # 权重文件存在但不是首选（快路径优先 npy）
    (models / f"_cascade_weights_{sym_key}.json").write_text(
        json.dumps({"mom_7": -0.05, "bb_width": 0.06}), encoding="utf-8")
    cur_tail = int(df.index[-1].timestamp())
    ones = np.ones(len(df))
    np.save(str(models / f"_cascade_factor_values_{sym_key}.npy"), ones)
    (models / f"_cascade_factor_values_{sym_key}.json").write_text(
        json.dumps({"tail_ts": cur_tail, "n": len(df)}), encoding="utf-8")

    engine = _make_engine(tmp_path)
    out = engine._load_cascade_factor_values(df, symbol=symbol)

    assert out is not None
    np.testing.assert_array_equal(out, ones)  # 快路径：直接用 npy，不复算
