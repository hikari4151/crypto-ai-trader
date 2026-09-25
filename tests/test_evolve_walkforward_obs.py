"""TDD：walk-forward 多折观测接入进化引擎（用户选"1"）。

行为契约（不改接受逻辑）：
- `_observe_strategy_drl_walkforward(df=...)`：每轮真实训练后，用部署中的
  strategy_drl 最佳模型在最近 N 根K线上做同品种多折 greedy 评估，
  结果写 `models/walkforward_obs.json` + 日志摘要；
- 无模型 / 开关关闭 / 数据不足 → 静默跳过（返回 None，不写文件、不抛异常）。
"""
import pytest

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


@pytest.fixture
def engine(tmp_path):
    return EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")


def _demo_df(n=1200):
    from backtest.data_loader import generate_demo
    return generate_demo(n=n, start_price=3000.0, vol=0.02,
                         timeframe="5m", seed=7)


@pytest.mark.asyncio
async def test_observe_skips_without_model(tmp_path, engine):
    """无 strategy_drl 模型时静默跳过，不写观测文件。"""
    out = await engine._observe_strategy_drl_walkforward(df=_demo_df())
    assert out is None
    assert not (tmp_path / "models" / "walkforward_obs.json").exists()


@pytest.mark.asyncio
async def test_observe_respects_disable(tmp_path, engine, monkeypatch):
    """开关关闭时静默跳过（即使有模型也不跑）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_walkforward_obs", False)
    out = await engine._observe_strategy_drl_walkforward(df=_demo_df())
    assert out is None
    assert not (tmp_path / "models" / "walkforward_obs.json").exists()
