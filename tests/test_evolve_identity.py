"""训练身份指纹与因子 OOS 基线隔离测试（Task 4）。

1. _identity_mismatch：缺身份字段（旧锚点）与跨 symbol/timeframe/config 均不可比；
2. 训练路径：身份一致的锚点仍保护，身份不一致（旧锚点缺字段）不压新模型；
3. 因子挖掘：OOS 安检不过的因子不得成为 factor best，也不得发布级联因子文件。
"""
import json

import pytest

import drl.evolve_engine as ev
from backtest.data_loader import generate_demo
from core.bus import EventBus
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine, _identity_mismatch, _training_identity
from tests.test_drl_optimizations import _MetaDB


def _agent():
    return ACAgent(state_dim=2, n_actions=2, hidden=(2, 2), seed=7)


def test_identity_mismatch_blocks_cross_symbol_comparison():
    anchor = {"model": "strategy_drl", "symbol": "BTC/USDT", "timeframe": "1h",
              "state_window": 1, "factor_signature": "", "config_fingerprint": "abc"}
    current = {**anchor, "symbol": "ETH/USDT"}
    reason = _identity_mismatch(anchor, current)
    assert reason is not None and "symbol" in reason


def test_identity_mismatch_blocks_timeframe_change():
    anchor = {"model": "strategy_drl", "symbol": "BTC/USDT", "timeframe": "1h",
              "state_window": 1, "factor_signature": "", "config_fingerprint": "abc"}
    current = {**anchor, "timeframe": "15m"}
    assert _identity_mismatch(anchor, current) is not None


def test_missing_identity_is_unknown_not_comparable():
    anchor = {"symbol": "BTC/USDT"}
    current = {"symbol": "BTC/USDT"}
    reason = _identity_mismatch(anchor, current)
    assert reason is not None and "缺少身份字段" in reason


def test_matching_identity_is_comparable():
    anchor = {"model": "strategy_drl", "symbol": "BTC/USDT", "timeframe": "1h",
              "state_window": 4, "factor_signature": "a", "config_fingerprint": "abc"}
    current = dict(anchor)
    assert _identity_mismatch(anchor, current) is None


def test_training_identity_is_deterministic_canonical():
    a = _training_identity("strategy_drl", "BTC/USDT", "1h",
                           config={"b": 2, "a": 1})
    b = _training_identity("strategy_drl", "BTC/USDT", "1h",
                           config={"a": 1, "b": 2})
    assert a["config_fingerprint"] == b["config_fingerprint"]
    assert a["symbol"] == "BTC/USDT" and a["timeframe"] == "1h"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    holder = {"result": None}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    return eng, holder


def _factor_result(oos_valid=True, fitness=0.6):
    return {
        "agent": _agent(),
        "history": [{"best_fitness": fitness}],
        "composite": None,
        "weights": {},
        "selected_factors": ["rsi(close,14)"],
        "report": {"enabled": True, "valid": oos_valid, "reason": ""},
    }


@pytest.mark.asyncio
async def test_factor_oos_rejected_does_not_become_best_and_no_cascade_files(engine, monkeypatch):
    """OOS 安检不过的因子：只存档，不写 best、不发布级联（npy/权重/元数据）。"""
    eng, holder = engine
    holder["result"] = _factor_result(oos_valid=False)

    await eng._train_factor_miner_once()

    assert eng.zoo.best_fitness("factor_miner") is None, "被拦因子不得成为 best"
    assert not eng.zoo._best_path("factor_miner").exists(), "best 文件不得生成"
    # 级联文件不得生成
    cascade_files = [p for p in eng.zoo.models_dir.glob("_cascade_*")]
    assert cascade_files == [], f"被拦因子不得发布级联文件: {cascade_files}"
    # 存档留证且不占版本名额
    arch_dir = eng.zoo._model_dir("factor_miner") / "archive"
    assert arch_dir.exists() and any(arch_dir.glob("*.json.gz"))
    assert not eng.zoo.list_versions("factor_miner")
    assert eng._factor_miner_status["oos_rejected"]


@pytest.mark.asyncio
async def test_factor_oos_passed_becomes_best(engine, monkeypatch):
    eng, holder = engine
    holder["result"] = _factor_result(oos_valid=True)

    await eng._train_factor_miner_once()

    assert eng.zoo.best_fitness("factor_miner") == 0.6
    assert eng._factor_miner_status["oos_rejected"] == ""