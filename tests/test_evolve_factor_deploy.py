"""进化因子自动部署为组合因子策略——部署方法单测（Task 1/3）。"""
import json

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from strategies import get_dynamic, get_strategy, remove_dynamic
from strategies.factor_signal import FactorSignalStrategy
from tests.test_drl_optimizations import _MetaDB


def _engine(tmp_path):
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


def _cleanup(*names):
    for n in names:
        remove_dynamic(n)


def test_name_flattens_symbol(tmp_path):
    eng = _engine(tmp_path)
    assert eng._combo_strategy_name("BTC/USDT") == "evolve_combo_BTC_USDT"
    assert eng._combo_strategy_name("SOL/USDT") == "evolve_combo_SOL_USDT"


async def test_deploy_creates_factor_signal_spec(tmp_path):
    eng = _engine(tmp_path)
    report = {"enabled": True, "valid": True, "rank_ic": 0.04, "icir": 1.2}
    out = await eng._deploy_factor_strategy(
        "BTC/USDT", {"vol_ratio": 0.5, "rsi_osc": -0.3}, report,
        {"fitness": 0.031, "selected_factors": ["vol_ratio", "rsi_osc"],
         "round_no": 7})
    try:
        assert out["name"] == "evolve_combo_BTC_USDT"
        assert out["version"] == "v1"
        spec = get_dynamic("evolve_combo_BTC_USDT")
        assert spec["executor"] == "factor_signal"
        assert spec["created_by"] == "evolve_engine"
        assert spec["params"]["factor"] == "combo"
        assert json.loads(spec["params"]["combo_spec"]) == {"vol_ratio": 0.5, "rsi_osc": -0.3}
        assert spec["params"]["mode"] == "trend"
        assert spec["evolve_meta"]["oos_report"] == report
        assert spec["evolve_meta"]["fitness"] == 0.031
        assert spec["evolve_meta"]["selected_factors"] == ["vol_ratio", "rsi_osc"]
        assert spec["base_symbol"] == "BTC/USDT"
    finally:
        _cleanup("evolve_combo_BTC_USDT")


async def test_deploy_overwrites_and_increments_version(tmp_path):
    eng = _engine(tmp_path)
    await eng._deploy_factor_strategy("ETH/USDT", {"vol_ratio": 1.0}, None, {})
    try:
        out2 = await eng._deploy_factor_strategy("ETH/USDT", {"vol_ratio": 2.0}, None, {})
        assert out2["version"] == "v2"
        spec = get_dynamic("evolve_combo_ETH_USDT")
        assert json.loads(spec["params"]["combo_spec"]) == {"vol_ratio": 2.0}
    finally:
        _cleanup("evolve_combo_ETH_USDT")


async def test_deployed_strategy_instantiates_as_factor_signal(tmp_path):
    eng = _engine(tmp_path)
    await eng._deploy_factor_strategy("BTC/USDT", {"vol_ratio": 0.5}, None, {})
    try:
        st = get_strategy("evolve_combo_BTC_USDT")
        assert isinstance(st, FactorSignalStrategy)
        assert st.params["factor"] == "combo"
        assert st.params["combo_spec"] == '{"vol_ratio": 0.5}'
    finally:
        _cleanup("evolve_combo_BTC_USDT")
