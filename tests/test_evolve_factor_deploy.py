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
    # 归属守卫：当前部署必须等于回测对象（name+version），否则写回被丢弃
    eng._factor_miner_status["combo_strategy"] = "evolve_combo_BTC_USDT"
    eng._factor_miner_status["combo_version"] = "v1"
    captured = {}

    def _fake_run(df_, cfg, **kw):
        captured["cfg"] = cfg
        captured["df_len"] = len(df_)
        captured["bootstrap"] = kw.get("bootstrap", True)
        return {"metrics": {"total_return": 0.12, "max_drawdown": 0.05,
                            "sharpe": 1.1, "total_trades": 7},
                "benchmark": {"buy_hold_ret": 0.03}}
    monkeypatch.setattr(be, "run_backtest", _fake_run)

    async def _go():
        await eng._refresh_factor_strategy_backtest(
            "evolve_combo_BTC_USDT", "BTC/USDT", df, "v1")

    asyncio.run(_go())
    summary = eng._factor_miner_status["combo_backtest"]
    assert summary["total_ret"] == 0.12
    assert summary["max_drawdown"] == 0.05
    assert summary["sharpe"] == 1.1
    assert summary["trades"] == 7
    assert summary["benchmark_ret"] == 0.03
    assert captured["cfg"].strategy_name == "evolve_combo_BTC_USDT"
    assert eng._factor_miner_status["combo_backtest_error"] == ""
    # 优化（自测实验）：摘要展示用途关闭 bootstrap、默认不截断窗口
    assert captured["bootstrap"] is False
    assert captured["df_len"] == 400


def test_summary_window_capped_by_setting(tmp_path, monkeypatch):
    """优化（自测实验）：evolve_factor_summary_bars>0 时摘要只回测最近 N 根
    （实测摘要耗时随窗口线性增长：1000 根 1.9s / 5000 根 10s）。"""
    import backtest.engine as be
    from backtest.data_loader import generate_demo
    from config.settings import settings
    eng = _engine(tmp_path)
    df = generate_demo(timeframe="1h", n=400, seed=42)
    eng._factor_miner_status["combo_strategy"] = "evolve_combo_BTC_USDT"
    eng._factor_miner_status["combo_version"] = "v1"
    captured = {}
    monkeypatch.setattr(settings, "evolve_factor_summary_bars", 100)

    def _fake_run(df_, cfg, **kw):
        captured["df_len"] = len(df_)
        return {"metrics": {"total_return": 0.1, "max_drawdown": 0.02,
                            "sharpe": 0.9, "total_trades": 3},
                "benchmark": {"buy_hold_ret": 0.01}}
    monkeypatch.setattr(be, "run_backtest", _fake_run)

    asyncio.run(eng._refresh_factor_strategy_backtest(
        "evolve_combo_BTC_USDT", "BTC/USDT", df, "v1"))

    assert captured["df_len"] == 100, "窗口应被截断为配置的最近 N 根"
    assert eng._factor_miner_status["combo_backtest"]["total_ret"] == 0.1


def test_backtest_failure_sets_backtest_error(tmp_path, monkeypatch):
    import backtest.engine as be
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    df = generate_demo(timeframe="1h", n=400, seed=42)
    eng._factor_miner_status["combo_strategy"] = "evolve_combo_BTC_USDT"
    eng._factor_miner_status["combo_version"] = "v1"
    # 预置旧摘要：失败只写错误键，不得覆盖旧摘要
    eng._factor_miner_status["combo_backtest"] = {"total_ret": 0.5, "at": 1.0}

    def _boom_run(df_, cfg, **kw):
        raise RuntimeError("backtest boom")
    monkeypatch.setattr(be, "run_backtest", _boom_run)

    async def _go():
        await eng._refresh_factor_strategy_backtest(
            "evolve_combo_BTC_USDT", "BTC/USDT", df, "v1")

    asyncio.run(_go())
    assert eng._factor_miner_status["combo_backtest_error"] != ""
    assert "回测摘要失败" in eng._factor_miner_status["combo_backtest_error"]
    assert eng._factor_miner_status["combo_backtest"] == {"total_ret": 0.5, "at": 1.0}


def test_stale_backtest_dropped_by_ownership_guard(tmp_path, monkeypatch):
    import backtest.engine as be
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    df = generate_demo(timeframe="1h", n=400, seed=42)
    # 当前部署已是 v2：v1 的旧轮慢任务写回必须被归属守卫丢弃
    eng._factor_miner_status["combo_strategy"] = "evolve_combo_ETH_USDT"
    eng._factor_miner_status["combo_version"] = "v2"

    def _fake_run(df_, cfg, **kw):
        return {"metrics": {"total_return": 0.99},
                "benchmark": {"buy_hold_ret": 0.01}}
    monkeypatch.setattr(be, "run_backtest", _fake_run)

    async def _go():
        await eng._refresh_factor_strategy_backtest(
            "evolve_combo_ETH_USDT", "ETH/USDT", df, "v1")

    asyncio.run(_go())
    assert eng._factor_miner_status.get("combo_backtest") is None
    assert eng._factor_miner_status.get("combo_backtest_error") == ""


class _StubAgent:
    def to_dict(self) -> dict:
        return {"stub": True}


def _factor_result(weights=None, valid=True):
    _w = weights if weights is not None else {"vol_ratio": 0.5}
    return {
        "agent": _StubAgent(),
        "history": [{"best_fitness": 0.02}],
        "selected_factors": list(_w.keys()),
        "weights": _w,
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
    async def _stub_refresh(name, symbol, df_, version):
        calls.append((name, symbol, version))
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
        assert eng._factor_miner_status["combo_strategy"] == _strat
        assert eng._factor_miner_status["combo_version"] == "v1"
        assert eng._factor_miner_status["combo_deploy_error"] == ""
        spec = get_dynamic(_strat)
        assert spec is not None and spec["executor"] == "factor_signal"
        assert calls == [(_strat, _symbol, "v1")]
        # 状态键经 _pipeline_view 透传（前端消费入口）
        view = eng.status()["factor_miner"]
        assert view["combo_strategy"] == _strat
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

    assert eng._factor_miner_status["combo_strategy"] == ""
    _symbol = eng._factor_miner_status["symbol"]
    assert get_dynamic(f"evolve_combo_{_symbol.replace('/', '_')}") is None


async def test_empty_weights_skips_deploy_with_error(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    async def _stub_fetch(symbol=""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _stub_fetch)
    holder = {"result": _factor_result(weights={})}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: None)

    await eng._train_factor_miner_once(force=True)
    # 无回测任务被创建；保持一致让出一次事件循环（无害）
    await asyncio.sleep(0)

    assert eng._factor_miner_status["combo_deploy_error"] != ""
    assert "权重缺失" in eng._factor_miner_status["combo_deploy_error"]
    assert eng._factor_miner_status["combo_strategy"] == ""


async def test_deploy_exception_does_not_break_training(tmp_path, monkeypatch):
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    async def _stub_fetch(symbol=""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _stub_fetch)
    holder = {"result": _factor_result()}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: None)

    async def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(eng, "_deploy_factor_strategy", _boom)

    await eng._train_factor_miner_once(force=True)
    # 无回测任务被创建（异常发生在 create_task 之前）；保持一致让出一次事件循环
    await asyncio.sleep(0)

    assert eng._factor_miner_status["combo_deploy_error"] != ""
    assert "自动部署失败" in eng._factor_miner_status["combo_deploy_error"]
    # 核心安全约束：部署异常不阻断训练——训练主体已完成，fitness 已更新
    assert eng._factor_miner_status["combo_strategy"] == ""
    assert eng._factor_miner_status["fitness"] == 0.02


async def test_restore_combo_deploy_status_after_restart(tmp_path):
    """自实验 bug-fix：重启后 combo_* 状态键应从已注册的最新部署恢复，
    进化面板不误显示「尚未部署」（此前仅训练轮写这些键，重启即丢）。"""
    eng1 = _engine(tmp_path)
    await eng1._deploy_factor_strategy(
        "BTC/USDT", {"vol_ratio": 0.5},
        {"enabled": True, "valid": True, "rank_ic": 0.04},
        {"fitness": 0.02, "selected_factors": ["vol_ratio"], "round_no": 3})
    try:
        # 模拟重启：新引擎实例（combo_* 内存状态全空），注册表（持久化恢复）仍在
        eng2 = _engine(tmp_path)
        await eng2._restore_combo_deploy_status()
        assert eng2._factor_miner_status["combo_strategy"] == "evolve_combo_BTC_USDT"
        assert eng2._factor_miner_status["combo_version"] == "v1"
        assert eng2._factor_miner_status["combo_deployed_at"] > 0
        # 幂等：已填充时不覆盖
        await eng2._restore_combo_deploy_status()
        assert eng2._factor_miner_status["combo_strategy"] == "evolve_combo_BTC_USDT"
        # 无部署时不报错、保持空
        eng3 = _engine(tmp_path)
        remove_dynamic("evolve_combo_BTC_USDT")
        await eng3._restore_combo_deploy_status()
        assert eng3._factor_miner_status["combo_strategy"] == ""
    finally:
        _cleanup("evolve_combo_BTC_USDT")


async def test_cascade_report_companion_written_on_accepted_round(tmp_path, monkeypatch):
    """自实验 bug-fix：安检通过轮在落权重文件的同时写 _cascade_report_<symbol>.json
    （手动重部署读它还原 OOS 报告/fitness/选中因子的真实 provenance）。"""
    import json as _json
    import drl.evolve_engine as ev
    from backtest.data_loader import generate_demo
    eng = _engine(tmp_path)
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)
    async def _stub_fetch(symbol=""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _stub_fetch)
    holder = {"result": _factor_result()}
    monkeypatch.setattr(ev, "train_factor_miner",
                        lambda df_, mat=None, cfg=None, on_progress=None: holder["result"])
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None, is_best=True: None)
    async def _stub_refresh(name, symbol, df_, version):
        return None
    monkeypatch.setattr(eng, "_refresh_factor_strategy_backtest", _stub_refresh)

    await eng._train_factor_miner_once(force=True)
    await asyncio.sleep(0)

    _symbol = eng._factor_miner_status["symbol"]
    rep_path = eng.zoo.models_dir / f"_cascade_report_{_symbol.replace('/', '_')}.json"
    assert rep_path.exists(), "安检通过轮应写出伴随报告元数据文件"
    rep = _json.loads(rep_path.read_text(encoding="utf-8"))
    assert rep["fitness"] == 0.02
    assert rep["oos_report"]["valid"] is True
    assert rep["selected_factors"] == ["vol_ratio"]
    assert rep["round_no"] == 1
    _cleanup(f"evolve_combo_{_symbol.replace('/', '_')}")
