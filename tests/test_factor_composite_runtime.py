"""持续进化挖掘的组合因子：实盘复算链路回归测试。

背景断裂（修复前）：factor_miner 级联组合因子只以预计算数组注入策略 DRL 训练，
部署端 rl_adaptive 只读 factor_expression（级联路径为空）→ 因子列恒 None →
带级联因子的 rl_evolve 每根K线都"跳过本根"、永不交易。挖掘的因子没有使用场景。

修复：配方（weights 映射）随模型保存，实盘在滚动缓冲上按训练同式复算
（CompositeFactorEvaluator：expanding z + 权重归一 + mu/sd 二次标准化）。
"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import generate_demo
from factors.engine import compute_factor_matrix
from factors.mining import CompositeFactorEvaluator, composite_factor


@pytest.fixture(scope="module")
def df():
    return generate_demo(timeframe="1h", n=800, seed=7)


@pytest.fixture(scope="module")
def weights():
    return {"mom_5": 1.0, "rsi_14": -0.5, "ma_dist_30": 0.8}


# ---------------- 复算器与训练端逐位一致 ----------------

def test_evaluator_matches_training_expanding(df, weights):
    """同一 df 上，复算器与 composite_factor(z_mode='expanding') 逐位一致。"""
    mat = compute_factor_matrix(df)
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")
    v = CompositeFactorEvaluator(weights).eval(df)
    assert v is not None
    assert abs(v - float(combo.iloc[-1])) < 1e-9, f"{v} vs {combo.iloc[-1]}"


def test_evaluator_second_normalization_matches_train_drl(df, weights):
    """二次标准化（train_drl 对训练段组合值拟合 mu/sd）口径一致。"""
    mat = compute_factor_matrix(df)
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")
    n_train = int(len(df) * 0.6)
    seg = combo.iloc[:n_train]
    mu, sd = float(seg.mean()), float(seg.std()) + 1e-8
    v = CompositeFactorEvaluator(weights, mu=mu, sd=sd).eval(df)
    expected = (float(combo.iloc[-1]) - mu) / sd
    assert abs(v - expected) < 1e-9, f"{v} vs {expected}"


def test_evaluator_short_buffer_still_produces_value(df, weights):
    """实盘滚动缓冲（400 根）也能给出有限值（数值漂移属预期，但不得为 None）。"""
    v = CompositeFactorEvaluator(weights).eval(df.iloc[-400:])
    assert v is not None and v == v


def test_evaluator_bad_input_safe(df):
    assert CompositeFactorEvaluator({}).eval(df) is None
    assert CompositeFactorEvaluator({"no_such_factor": 1.0}).eval(df) is None
    assert CompositeFactorEvaluator({"mom_5": 1.0}).eval(df.iloc[:3]) is None


# ---------------- rl_adaptive / meta_controller 实盘取值 ----------------

def _feed_bars(st, df):
    for _, r in df.iterrows():
        st._closes.append(float(r["close"]))
        st._opens.append(float(r["open"]))
        st._highs.append(float(r["high"]))
        st._lows.append(float(r["low"]))
        st._volumes.append(float(r["volume"]))


def test_rl_adaptive_composite_factor_value(df, weights):
    """rl_adaptive 带配方时 _factor_value 返回有限值（修复前恒 None）。"""
    from strategies.rl_adaptive import RLAdaptiveStrategy

    mat = compute_factor_matrix(df)
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")
    seg = combo.iloc[:int(len(df) * 0.6)]
    st = RLAdaptiveStrategy()
    st.reset()
    st._factor_composite = {"weights": weights}
    st._factor_mu = float(seg.mean())
    st._factor_sd = float(seg.std()) + 1e-8
    _feed_bars(st, df.iloc[-200:])
    fv = st._factor_value()
    assert fv is not None and fv == fv and not math.isnan(fv)


def test_meta_controller_composite_factor_value(df, weights):
    """meta_controller 同口径（配方优先，其次表达式）。"""
    from strategies.meta import MetaController

    mat = compute_factor_matrix(df)
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding")
    seg = combo.iloc[:int(len(df) * 0.6)]
    mc = MetaController()
    mc.reset()
    mc._factor_composite = {"weights": weights}
    mc._factor_mu = float(seg.mean())
    mc._factor_sd = float(seg.std()) + 1e-8
    _feed_bars(mc, df.iloc[-200:])
    fv = mc._factor_value()
    assert fv is not None and fv == fv


def test_rl_adaptive_expression_path_unchanged(df):
    """无配方时表达式路径保持可用（兼容旧模型）。"""
    from strategies.rl_adaptive import RLAdaptiveStrategy

    st = RLAdaptiveStrategy()
    st.reset()
    st._factor_expr_from_model = "ma(close,5)/ma(close,20)-1"
    st._factor_mu, st._factor_sd = 0.0, 1.0
    _feed_bars(st, df.iloc[-200:])
    fv = st._factor_value()
    assert fv is not None and fv == fv


# ---------------- train_drl / evolve 配方透传 ----------------

def test_train_drl_passthrough_factor_composite():
    """train_drl 级联路径把配方透传到结果（模型文件写入端消费）。"""
    from drl.agent import train_drl

    dfd = generate_demo(timeframe="1h", n=400, seed=42)
    factor_values = np.random.randn(len(dfd)).astype(float)
    recipe = {"weights": {"mom_5": 1.0, "rsi_14": -0.5}}
    cfg = {
        "episodes": 2, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64,
        "seed": 42, "val_eval_interval": 1,
        "factor_values": factor_values, "factor_composite": recipe,
    }
    result = train_drl(dfd, cfg)
    assert result["factor_composite"] == recipe, "级联配方必须随结果透传"
    assert result["factor_expression"] == "", "级联路径不进表达式（实盘走配方）"


def test_evolve_load_cascade_factor_recipe(tmp_path):
    """_load_cascade_factor_recipe：按标读权重 JSON，缺失/坏文件返回 None。"""
    from drl.evolve_engine import EvolveEngine

    eng = EvolveEngine.__new__(EvolveEngine)
    eng.zoo = SimpleNamespace(models_dir=tmp_path)
    assert eng._load_cascade_factor_recipe("ETH/USDT") is None, "无文件应 None"
    (tmp_path / "_cascade_weights_ETH_USDT.json").write_text(
        '{"mom_5": 1.0, "rsi_14": -0.5, "bad": "x"}', encoding="utf-8")
    rec = eng._load_cascade_factor_recipe("ETH/USDT")
    assert rec == {"weights": {"mom_5": 1.0, "rsi_14": -0.5}}, rec
    (tmp_path / "_cascade_weights_ETH_USDT.json").write_text("not json", encoding="utf-8")
    assert eng._load_cascade_factor_recipe("ETH/USDT") is None, "坏文件应 None"