"""TDD：进化产物「仅软件内运行（无 Pine）」显式标记（诊断报告建议第5条(b)）。

背景：带级联因子列的 rl_evolve 模型当前无法导出 Pine（预计算因子列无法逐根复算），
pine_note 已说明原因，但策略列表缺一个显式徽标，用户仍会误以为可以上图。
修复：_register_rl_evolve 在 pine_note 非空时给 spec 打 internal_only=True，
前端据此显示"仅软件内运行"徽标；有真实 Pine 代码的模型不标记。
"""
import pytest

from core.bus import EventBus
from drl.evolve_engine import EvolveEngine
from tests.test_drl_optimizations import _MetaDB


@pytest.mark.asyncio
async def test_register_rl_evolve_marks_internal_only_when_no_pine(tmp_path, monkeypatch):
    """无法导出 Pine（级联因子模型）→ spec.internal_only=True。"""
    import drl.evolve_engine as ev
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    monkeypatch.setattr(
        eng, "_export_rl_evolve_pine",
        lambda: ("", "模型无法导出 Pine：它依赖 factor_miner 组合因子列…"))
    captured = {}
    monkeypatch.setattr(ev, "register_dynamic",
                        lambda name, spec: captured.update(spec=spec))

    async def _noop(name, spec):
        pass
    monkeypatch.setattr(eng, "_upsert_ai_strategy", _noop)

    await eng._register_rl_evolve()
    assert captured["spec"].get("internal_only") is True


@pytest.mark.asyncio
async def test_register_rl_evolve_not_internal_only_when_pine_ok(tmp_path, monkeypatch):
    """有真实 Pine 代码的模型不标 internal_only。"""
    import drl.evolve_engine as ev
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    monkeypatch.setattr(eng, "_export_rl_evolve_pine",
                        lambda: ("//@version=5\nstrategy('x')", ""))
    captured = {}
    monkeypatch.setattr(ev, "register_dynamic",
                        lambda name, spec: captured.update(spec=spec))

    async def _noop(name, spec):
        pass
    monkeypatch.setattr(eng, "_upsert_ai_strategy", _noop)

    await eng._register_rl_evolve()
    assert captured["spec"].get("internal_only") is not True