"""AI 参数候选必须先验证，再进入策略热更新。"""
import asyncio

from engine.trading_engine import TradingEngine


class _Strategy:
    name = "dual_ma"
    params = {"fast_period": 10, "slow_period": 30}

    def update_params(self, params):
        self.params.update(params)
        return dict(self.params)


class _EngineStub:
    def __init__(self, validation):
        self.strategy = _Strategy()
        self._validation = validation
        self._strategy_lock = asyncio.Lock()
        self.applied = []

    async def _validate_param_update(self, name, proposed, current):
        return self._validation

    async def apply_strategy_params(self, name, params):
        self.applied.append((name, dict(params)))
        return self.strategy.update_params(params)

    async def validate_and_apply_ai_params(self, name, proposed, current=None):
        return await TradingEngine.validate_and_apply_ai_params(
            self, name, proposed, current)


def test_rejected_candidate_never_applies():
    async def run():
        eng = _EngineStub((False, {"reason": "OOS rejected"}))
        ok, info = await eng.validate_and_apply_ai_params(
            "dual_ma", {"fast_period": 12})
        assert ok is False
        assert info["applied"] is False
        assert eng.applied == []
        assert eng.strategy.params == {"fast_period": 10, "slow_period": 30}

    asyncio.run(run())


def test_accepted_candidate_applies_once():
    async def run():
        eng = _EngineStub((True, {"reason": "passed"}))
        ok, info = await eng.validate_and_apply_ai_params(
            "dual_ma", {"fast_period": 12})
        assert ok is True
        assert info["applied"] is True
        assert eng.applied == [("dual_ma", {"fast_period": 12})]
        assert eng.strategy.params["fast_period"] == 12

    asyncio.run(run())


def test_candidate_is_rejected_if_strategy_changed_during_validation():
    async def run():
        eng = _EngineStub((True, {"reason": "passed"}))
        original_validate = eng._validate_param_update

        async def validate_then_change(name, proposed, current):
            result = await original_validate(name, proposed, current)
            eng.strategy.params["fast_period"] = 11
            return result

        eng._validate_param_update = validate_then_change
        ok, info = await eng.validate_and_apply_ai_params(
            "dual_ma", {"fast_period": 12})
        assert ok is False
        assert info["applied"] is False
        assert "已变化" in info["reason"]
        assert eng.applied == []
        assert eng.strategy.params["fast_period"] == 11

    asyncio.run(run())
