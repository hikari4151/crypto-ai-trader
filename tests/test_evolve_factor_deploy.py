"""进化因子自动部署为组合因子策略——部署方法单测（Task 1/3）。"""
import asyncio
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


def test_backtest_summary_written_to_status(tmp_path, monkeypatch):
    import backtest.engine as be
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    df = generate_demo(timeframe="1h", n=400, seed=42)
    captured = {}

    def _fake_run(df_, cfg, **kw):
        captured["cfg"] = cfg
        return {"metrics": {"total_return": 0.12, "max_drawdown": 0.05,
                            "sharpe": 1.1, "total_trades": 7},
                "benchmark": {"buy_hold_ret": 0.03}}
    monkeypatch.setattr(be, "run_backtest", _fake_run)

    async def _go():
        await eng._refresh_factor_strategy_backtest(
            "evolve_combo_BTC_USDT", "BTC/USDT", df)

    asyncio.run(_go())
    summary = eng._factor_miner_status["backtest_summary"]
    assert summary["total_ret"] == 0.12
    assert summary["max_drawdown"] == 0.05
    assert summary["sharpe"] == 1.1
    assert summary["trades"] == 7
    assert summary["benchmark_ret"] == 0.03
    assert captured["cfg"].strategy_name == "evolve_combo_BTC_USDT"
    assert eng._factor_miner_status["deploy_error"] == ""


class _StubAgent:
    def to_dict(self) -> dict:
        return {"stub": True}


def _factor_result(weights=None, valid=True):
    return {
        "agent": _StubAgent(),
        "history": [{"best_fitness": 0.02}],
        "selected_factors": list((weights or {"vol_ratio": 0.5}).keys()),
        "weights": weights or {"vol_ratio": 0.5},
        "composite": None,   # 跳过级联 npy 保存，聚焦部署断言
        "report": {"enabled": True, "valid": valid, "rank_ic": 0.04,
                   "icir": 1.2, "reason": ""},
        "meta": {},
    }


async def test_passing_factor_miner_auto_deploys(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    from strategies import get_dynamic
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    async def _stub_fetch(symbol=""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _stub_fetch)
    holder = {"result": _factor_result()}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    saved = []
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: saved.append(name))
    calls = []
    async def _stub_refresh(name, symbol, df_):
        calls.append((name, symbol))
    monkeypatch.setattr(eng, "_refresh_factor_strategy_backtest", _stub_refresh)

    try:
        await eng._train_factor_miner_once(force=True)
        # fire-and-forget 的后台回测任务要等训练协程返回、事件循环下一轮
        # 才被调度（CPython asyncio 语义），先让出一次循环再断言调用记录
        await asyncio.sleep(0)

        # 环境差异：repo config/config.yaml 的 evolve_symbols 被覆盖为 [ETH/USDT]，
        # 首轮标的并非简报默认假设的 BTC/USDT——按简报 Step4 注释改为动态取 symbol。
        _symbol = eng._factor_miner_status["symbol"]
        _strat = f"evolve_combo_{_symbol.replace('/', '_')}"

        assert saved == ["factor_miner"]
        assert eng._factor_miner_status["deployed_strategy"] == _strat
        assert eng._factor_miner_status["deployed_version"] == "v1"
        assert eng._factor_miner_status["deploy_error"] == ""
        spec = get_dynamic(_strat)
        assert spec is not None and spec["executor"] == "factor_signal"
        assert calls == [(_strat, _symbol)]
        # 状态键经 _pipeline_view 透传（前端消费入口）
        view = eng.status()["factor_miner"]
        assert view["deployed_strategy"] == _strat
    finally:
        from strategies import remove_dynamic
        _strat_n = locals().get("_strat")
        if _strat_n:
            remove_dynamic(_strat_n)


async def test_rejected_factor_miner_does_not_deploy(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    from strategies import get_dynamic
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    async def _stub_fetch(symbol=""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _stub_fetch)
    holder = {"result": _factor_result(valid=False)}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: None)

    await eng._train_factor_miner_once(force=True)

    assert eng._factor_miner_status["deployed_strategy"] == ""
    _symbol = eng._factor_miner_status["symbol"]
    assert get_dynamic(f"evolve_combo_{_symbol.replace('/', '_')}") is None
