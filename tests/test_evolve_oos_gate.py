"""P1-3 进化引擎注册前必须过 OOS 硬门，且 fitness 与研究端点同口径。

进化引擎是无人值守的：它把新模型设为最佳（覆盖实盘读的 strategy_drl.json）
并注册 rl_evolve 策略。旧逻辑只看数据源不是 demo 就放行，完全没读 train_drl
已经算好的 deployment_blocked / oos_report，等于让未通过样本外检验的模型自动上线。
"""
import json

import pytest

import drl.evolve_engine as ev
from backtest.data_loader import generate_demo
from drl.evolve_engine import EvolveEngine, _oos_gate_passed, _strategy_deploy_gate
from tests.test_drl_optimizations import _MetaDB


# ---------------- 硬门判据本身 ----------------

def test_gate_blocks_when_train_drl_marks_deployment_blocked():
    ok, reason = _strategy_deploy_gate({
        "deployment_blocked": True,
        "oos_report": {"enabled": True, "oos_ret": -0.2, "reason": "OOS 衰减 80%，疑似过拟合"},
    })
    assert ok is False and "过拟合" in reason


def test_gate_blocks_without_any_oos_evidence():
    ok, reason = _strategy_deploy_gate({"oos_report": {"enabled": False}})
    assert ok is False and "样本" in reason


def test_gate_blocks_when_oos_loses_money_even_if_train_profitable():
    """训练段亏损时 train_drl 不判过拟合（decay=0），但亏钱的模型同样不能自动上线。"""
    ok, reason = _strategy_deploy_gate({
        "oos_report": {"enabled": True, "oos_ret": -0.03, "train_ret": -0.05, "decay": 0.0},
    })
    assert ok is False and "非正" in reason


def test_gate_passes_with_profitable_oos():
    ok, reason = _strategy_deploy_gate({
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": 0.04, "train_ret": 0.06, "decay": 0.33},
    })
    assert ok is True and reason == ""


def test_gate_rejects_when_oos_trades_zero():
    """P4-E2：报告带 oos_trades=0 → 判"样本外未交易"（优先于段末持仓比例）。"""
    ok, reason = _strategy_deploy_gate({
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": -0.001, "oos_trades": 0,
                       "oos_position_ratio": 0.0},
    })
    assert ok is False and "未交易" in reason


def test_gate_traded_but_lost_reports_nonpositive():
    """P4-E2：OOS 有交易但段末平仓（position_ratio=0）+ 收益为负 → 判"真亏损"而非"未交易"。

    回归：元策略 OOS 策略"交易后段末平仓"会被段末持仓比例误判为从未交易；
    有了 oos_trades 计数后必须给出诚实的"非正"结论。
    """
    ok, reason = _strategy_deploy_gate({
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": -0.02, "oos_trades": 7,
                       "oos_position_ratio": 0.0},
    })
    assert ok is False and "未交易" not in reason and "非正" in reason


def test_gate_falls_back_to_position_ratio_without_trades():
    """P4-E2：老报告无 oos_trades → 回退段末持仓比例（旧行为保持不变）。"""
    ok, reason = _strategy_deploy_gate({
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": -0.001, "oos_position_ratio": 0.01},
    })
    assert ok is False and "未交易" in reason
    ok2, reason2 = _strategy_deploy_gate({
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": -0.001, "oos_position_ratio": 0.5},
    })
    assert ok2 is False and "非正" in reason2


@pytest.mark.parametrize("report,expected", [
    ({"enabled": True, "valid": True}, True),
    ({"enabled": False}, False),
    ({"enabled": True, "valid": False, "reason": "ICIR 不足"}, False),
    (None, False),
])
def test_factor_cascade_gate_matches_api_criteria(report, expected):
    """因子级联发布的判据与 /api/drl/mine-factor 注册前一致（enabled 且 valid）。"""
    assert _oos_gate_passed(report)[0] is expected


# ---------------- 进化循环真的执行了这道门 ----------------

class _StubAgent:
    def to_dict(self) -> dict:
        return {"hidden": [4, 4], "w": [[0.0]], "stub": True}


def _result(best_ret=0.30, oos_ret=0.05, blocked=False):
    return {
        "agent": _StubAgent(),
        # 故意只给 total_ret：旧代码会取到 -3.0，新代码必须取 best_ret
        "history": [{"total_ret": -3.0}],
        "best_ret": best_ret,
        "deployment_blocked": blocked,
        "oos_report": {"enabled": True, "oos_ret": oos_ret, "train_ret": best_ret,
                       "hard_rejected": blocked,
                       "reason": "OOS 衰减过半，疑似过拟合，已拦截自动部署" if blocked else ""},
        "pine_code": "", "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from core.bus import EventBus
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    holder = {"result": None}   # 每个用例塞入本轮 train_drl 的返回
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None: holder["result"])
    registered: list = []
    monkeypatch.setattr(ev, "register_dynamic", lambda name, spec: registered.append(name))
    # P4-E1：桩掉跨标 OOS 与策略注册——跨标评估要跑真实 greedy rollout
    # （测试替身 _StubAgent 无 greedy_action，此前靠 AttributeError 被吞才放行，
    # 等于"测试没测到放行路径的跨标门"）；_register_rl_evolve 读后端模型文件，
    # 与 gate 本身无关，桩掉让断言只针对部署门行为。
    async def _no_cross_oos(*a, **k):
        return None
    async def _no_register_rl(*a, **k):
        # 模拟真实 _register_rl_evolve 的注册副作用（真实实现读后端模型
        # 文件导出 Pine 并 register_dynamic("rl_evolve")；此处只保留
        # 对测试断言有意义的注册行为，避免依赖真实文件系统）
        registered.append("rl_evolve")
        return None
    monkeypatch.setattr(eng, "_cross_symbol_oos", _no_cross_oos)
    monkeypatch.setattr(eng, "_register_rl_evolve", _no_register_rl)
    return eng, registered, holder


async def test_rejected_model_does_not_take_over_live_files(engine):
    """OOS 拦截：实盘模型文件保持原样、不注册策略；被拦模型归档到 archive/ 可复查，
    但不占版本名额（P4-C2：废模型不得把真改进版挤出版本历史）。"""
    eng, registered, holder = engine
    holder["result"] = _result(blocked=True, oos_ret=-0.2)
    flat = eng.zoo._flat_path("strategy_drl")
    flat.write_text(json.dumps({"marker": "old-best"}), encoding="utf-8")

    await eng._train_strategy_drl_once()

    assert registered == [], "被 OOS 硬门拦截的模型不得注册为实盘策略"
    assert json.loads(flat.read_text(encoding="utf-8")) == {"marker": "old-best"}, \
        "被拦截的模型不得覆盖实盘读取的 strategy_drl.json"
    assert not eng.zoo._best_path("strategy_drl").exists()
    assert "过拟合" in eng._strategy_drl_status["oos_rejected"]
    # P4-C2：被拦模型归档到 archive/ 子目录（保留研究痕迹），但不占版本名额
    archive_dir = eng.zoo._model_dir("strategy_drl") / "archive"
    assert archive_dir.exists() and any(archive_dir.glob("*.json.gz")), \
        "被拦截模型应归档到 archive/ 以便复盘"
    assert not eng.zoo.list_versions("strategy_drl"), \
        "被拦截模型不得占用版本名额（避免挤掉真正的改进版）"


async def test_passing_model_registers_and_uses_summary_fitness(engine):
    """放行：注册策略、覆盖最佳模型，fitness 取 train_drl 汇总的 best_ret。"""
    eng, registered, holder = engine
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)

    await eng._train_strategy_drl_once()

    assert registered == ["rl_evolve"]
    assert eng._strategy_drl_status["fitness"] == 0.30
    assert eng._strategy_drl_status["oos_rejected"] == ""
    assert eng.zoo.best_fitness("strategy_drl") == 0.30
    assert eng.zoo._best_path("strategy_drl").exists()
