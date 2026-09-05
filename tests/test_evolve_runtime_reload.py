"""持续进化部署后运行时热加载测试（Task 3）。

部署成功 != 文件成功：当前策略实例必须切到新模型，否则实盘继续用旧内存权重。
这里锁三件事：
1. RLAdaptiveStrategy.reload_model 清掉旧 _agent 后重新加载当前 model_path；
2. MetaController.reload_model 清掉 _meta_agent/_meta_state_dim 后重载；
3. TradingEngine.reload_active_strategy_model 持策略锁重载目标策略实例，
   与部署目标无关的当前策略只返回 skipped，不替换用户策略。
"""
import pytest

from core.bus import EventBus
from engine.trading_engine import TradingEngine
from strategies.meta import MetaController
from strategies.rl_adaptive import RLAdaptiveStrategy
from tests.test_drl_optimizations import _MetaDB


def _set_and_true(obj, attr, value):
    setattr(obj, attr, value)
    return True


def test_rl_adaptive_reload_model_swaps_agent(monkeypatch):
    strategy = RLAdaptiveStrategy()
    strategy.params["model_path"] = "data/models/strategy_drl.json"
    old = object()
    new = object()
    strategy._agent = old
    monkeypatch.setattr(strategy, "_load_agent",
                        lambda: _set_and_true(strategy, "_agent", new))

    assert strategy.reload_model() is True
    assert strategy._agent is new, "reload_model 必须替换运行实例"


def test_rl_adaptive_reload_model_failure_clears_agent(monkeypatch):
    strategy = RLAdaptiveStrategy()
    strategy.params["model_path"] = "data/models/strategy_drl.json"
    strategy._agent = object()

    def _failing_load():
        strategy._agent = None
        return False

    monkeypatch.setattr(strategy, "_load_agent", _failing_load)

    assert strategy.reload_model() is False
    assert strategy._agent is None, "重载失败不得保留旧权重继续跑"


def test_meta_controller_reload_model_reloads_agent(monkeypatch):
    strategy = MetaController()
    strategy.params["model_path"] = "data/models/meta_controller.json"
    old = object()
    new = object()
    strategy._meta_agent = old
    strategy._meta_state_dim = 15
    monkeypatch.setattr(strategy, "_load_meta_agent",
                        lambda: _set_and_true(strategy, "_meta_agent", new))

    assert strategy.reload_model() is True
    assert strategy._meta_agent is new
    assert strategy._meta_state_dim == 0, "重载前必须清空旧状态维度"


@pytest.mark.asyncio
async def test_trading_engine_reload_refreshes_active_rl_instance(monkeypatch):
    engine = TradingEngine(_MetaDB(), EventBus())
    strategy = RLAdaptiveStrategy()
    strategy.params["model_path"] = "data/models/strategy_drl.json"
    old = object()
    new = object()
    strategy._agent = old
    monkeypatch.setattr(strategy, "_load_agent",
                        lambda: _set_and_true(strategy, "_agent", new))
    engine.strategy = strategy

    result = await engine.reload_active_strategy_model("strategy_drl", 4, {})

    assert result["status"] == "reloaded"
    assert strategy._agent is new


@pytest.mark.asyncio
async def test_trading_engine_reload_skips_unrelated_strategy():
    from strategies.base import Strategy, Signal

    class _Stub(Strategy):
        name = "dual_ma"

        def on_candle(self, ctx):
            return None

    engine = TradingEngine(_MetaDB(), EventBus())
    engine.strategy = _Stub()

    result = await engine.reload_active_strategy_model("strategy_drl", 4, {})

    assert result["status"] == "skipped"
    assert engine.strategy.name == "dual_ma", "不得替换用户当前策略"


@pytest.mark.asyncio
async def test_trading_engine_reload_reports_failure(monkeypatch):
    engine = TradingEngine(_MetaDB(), EventBus())
    strategy = RLAdaptiveStrategy()
    strategy.params["model_path"] = "data/models/strategy_drl.json"
    strategy._agent = object()

    def _failing_load():
        strategy._agent = None
        return False

    monkeypatch.setattr(strategy, "_load_agent", _failing_load)
    engine.strategy = strategy

    result = await engine.reload_active_strategy_model("strategy_drl", 4, {})

    assert result["status"] == "failed"
    assert strategy._agent is None
