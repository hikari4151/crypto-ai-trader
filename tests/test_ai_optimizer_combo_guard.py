"""观察项解决测试：AI 参数优化器不得改写进化引擎维护的组合因子字段。

组合因子策略（factor=combo）的因子选择/权重/模式/阈值由持续进化引擎自动部署
维护——AI 优化只允许调风控旋钮（止损/止盈/仓位），不得动 combo_spec/factor/
mode/阈值，否则会静默丢弃进化引擎的权重产出（自实验观察项）。
"""
import pytest

from ai import optimizer as opt_mod
from ai.optimizer import ParamOptimizer
from strategies.factor_signal import FactorSignalStrategy


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def add(self, *a, **k):
        pass

    async def commit(self):
        pass


class _FakeDb:
    def session(self):
        return _FakeSession()


async def _fake_run_on_main(db, coro):
    await coro()


@pytest.fixture
def no_db_persist(monkeypatch):
    """桩掉 OptimizationLog 落库（测试环境无运行中的主事件循环调度）。"""
    monkeypatch.setattr(opt_mod, "_run_on_main", _fake_run_on_main)


class _FakeClient:
    """返回含越权字段的 AI 建议参数（模拟 LLM 建议 combo_spec/factor/mode 改写）。"""

    async def chat_json_validated(self, *a, **k):
        return {
            "params": {
                "factor": "custom",
                "mode": "reversal",
                "combo_spec": '{"bad_factor": 1.0}',
                "buy_threshold": -1.0,
                "sell_threshold": 1.0,
                "expression": "close * 2",
                # 风控旋钮：允许被优化
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "size_pct": 0.3,
            },
            "reason": "调风控",
            "focus": "test",
        }


@pytest.mark.asyncio
async def test_ai_optimizer_guards_combo_fields(no_db_persist):
    st = FactorSignalStrategy()
    st.update_params({"factor": "combo",
                      "combo_spec": '{"vol_ratio": 0.5}'})
    opt = ParamOptimizer(_FakeClient(), _FakeDb())

    out = await opt.optimize_price_action(st, performance={}, snap={}, apply=True)

    assert out is not None
    assert st.params["factor"] == "combo", "AI 不得改写因子选择"
    assert st.params["combo_spec"] == '{"vol_ratio": 0.5}', "AI 不得改写组合权重"
    assert st.params["mode"] == "trend", "AI 不得改写模式"
    assert st.params["buy_threshold"] == 0.0, "AI 不得改写阈值"
    assert st.params["sell_threshold"] == 0.0
    assert st.params["expression"] == "", "AI 不得改写表达式"
    # 风控旋钮仍可被 AI 优化
    assert st.params["stop_loss_pct"] == 0.02
    assert st.params["take_profit_pct"] == 0.04
    assert st.params["size_pct"] == 0.3


@pytest.mark.asyncio
async def test_ai_optimizer_does_not_filter_non_combo(no_db_persist):
    """非 combo 的 factor_signal（如 macd_hist 趋势）不受此保护限制——AI 可自由调。"""
    st = FactorSignalStrategy()
    st.update_params({"factor": "macd_hist", "mode": "trend"})
    opt = ParamOptimizer(_FakeClient(), _FakeDb())

    out = await opt.optimize_price_action(st, performance={}, snap={}, apply=True)

    assert out is not None
    # macd_hist 策略无 factor=combo 语义，AI 建议（含 mode 等）照常生效
    assert st.params["mode"] == "reversal"
    assert st.params["stop_loss_pct"] == 0.02