"""L2：信号密度**归一化**门槛 + 草稿（未证明）状态 + 迭代门诊断量。

背景：用户抱怨"AI 设计/迭代经常产出严重过拟合或几乎不交易的策略"。诊断出的结构性
成因之一是门槛把"交易少"制度化了：

- 密度门判据是**绝对**笔数，与窗口长度脱钩 → 主窗口 1000 根要 5 笔（0.5%），
  而备选长窗口 3000 根**仍只要 5 笔**（=1.67 笔/1000 根）。窗口越长越容易过，
  "没信号的策略"于是能靠更长的窗口蒙混过关。
- "证据不足 / 过拟合未过"的产物只是打个标记，仍然被当成正常策略：
  能作为迭代链的父代继续繁殖（低交易 → 更少交易；过拟合 → 换批参数继续过拟合），
  也会出现在可应用的策略列表里，与过了门的策略长得一样。

本文件锁住三条修复：
1. 门槛改成归一化频率（笔/1000 根，≥15）；
2. "未证明"产物进入独立 status="draft"（不当迭代父代、AI 不接管调参）；
3. 迭代门把"每笔收益 t 统计量"当**诊断量**而非第 6 道硬门槛
   （实测加硬门槛既无鉴别力增益、又会让门不可达，见 probe_l2_density.py）。
"""
import math

import pandas as pd
import pytest

import ai.iteration as iteration_mod
import strategies
from ai import strategy_designer as sd
from ai.iteration import _ITER_MIN_TRADES, _pnl_stats, evaluate_iteration_gate
from ai.strategy_designer import _DENSITY_MIN_TRADES_PER_1000, _judge_density, _min_trades_for


@pytest.fixture
def _dynamic_snapshot():
    """用例前后还原 _DYNAMIC：本文件要真注册策略，别污染其它测试。"""
    snap = dict(strategies._DYNAMIC)
    yield
    strategies._DYNAMIC.clear()
    strategies._DYNAMIC.update(snap)


# ---------------- 一、归一化密度门槛 ----------------

def test_min_trades_for_is_normalized():
    """同一频率在不同窗口长度上折算出不同的绝对门槛。

    1000 根 → 15 笔；3000 根 → 45 笔。修复前两处都是固定 5 笔。
    """
    assert _min_trades_for(1000) == 15
    assert _min_trades_for(3000) == 45
    assert _min_trades_for(500) == 8          # 向上取整
    assert _min_trades_for(0) == 1            # 退化窗口不会算出 0 笔门槛
    # 兼容旧引用：模块级常量 = 主窗口折算值
    assert sd._DENSITY_MIN_TRADES == _min_trades_for(sd._DENSITY_CANDLES) == 15


def test_judge_density_rejects_diluted_long_window():
    """L2 核心回归：长窗口不再稀释门槛。

    3000 根 37 笔 = 12.33 笔/1000 根，低于 15 → 判不过。
    修复前用绝对门槛（5 笔），这个数会**通过** —— 正是"交易少"被放行的路径。
    真实数据上 grid 默认参数就是这个量级（探针 E1：3000 根 37 笔）。
    """
    j = _judge_density(37, 3000)
    assert j["passed"] is False
    assert j["min_trades"] == 45
    assert j["trades_per_1000"] == 12.33
    assert j["min_trades_per_1000"] == _DENSITY_MIN_TRADES_PER_1000 == 15


def test_judge_density_accepts_healthy_rate_on_both_windows():
    """健康频率（≈18 笔/1000 根，dual_ma 默认参数实测）在长短窗口上判定一致。"""
    assert _judge_density(18, 1000)["passed"] is True
    assert _judge_density(53, 3000)["passed"] is True     # 17.67 笔/1000 根
    # 边界：恰好达标
    assert _judge_density(45, 3000)["passed"] is True
    assert _judge_density(44, 3000)["passed"] is False


# ---------------- 二、草稿（未证明）状态 ----------------

_DRAFT_CASES = [
    {"blocked_by_overfit": True},
    {"overfit_inconclusive": True},
]


@pytest.mark.parametrize("flags", _DRAFT_CASES)
def test_draft_status_derived_from_detection_flags(flags, _dynamic_snapshot):
    """"未证明"标记 → status=draft，且不依赖调用方记得显式设 status。"""
    name = "l2_draft_case"
    strategies.register_dynamic(name, {**flags, "params": {}, "executor": "dual_ma"})

    spec = strategies.get_dynamic(name)
    assert spec["status"] == "draft"
    assert strategies.is_draft(name) is True
    row = [s for s in strategies.list_strategies() if s["name"] == name][0]
    assert row["draft"] is True and row["status"] == "draft"
    # 原因必须能推出来（前端 tooltip / 二次确认直接用这个字段）
    assert row["draft_reason"]


def test_clean_strategy_is_active(_dynamic_snapshot):
    """过了门的策略不受影响：加草稿机制不能顺手把正常策略也标成草稿。"""
    name = "l2_active_case"
    strategies.register_dynamic(name, {"params": {}, "executor": "dual_ma"})

    assert strategies.get_dynamic(name)["status"] == "active"
    assert strategies.is_draft(name) is False
    row = [s for s in strategies.list_strategies() if s["name"] == name][0]
    assert row["draft"] is False and row["draft_reason"] == ""


def test_builtin_strategy_is_never_draft():
    """内置策略不在 _DYNAMIC 里 → 永远不是草稿（守卫不能误伤内置策略）。"""
    assert strategies.is_draft("dual_ma") is False
    assert strategies.is_draft("__does_not_exist__") is False


def test_explicit_status_wins_over_flags(_dynamic_snapshot):
    """显式 status 优先：留给"人工确认后提升为可用"这类操作。"""
    name = "l2_promoted_case"
    strategies.register_dynamic(name, {"blocked_by_overfit": True, "status": "active",
                                       "params": {}, "executor": "dual_ma"})
    assert strategies.is_draft(name) is False


@pytest.mark.parametrize("spec", [
    {"blocked_by_overfit": True},
    {"overfit_inconclusive": True},
    {},
    {"blocked_by_overfit": False, "overfit_inconclusive": False},
])
def test_draft_rule_is_consistent_across_layers(spec):
    """草稿规则在两层各有一份实现（strategies 是最底层、不能反向 import ai 层）。

    这条用例的作用是**锁住两份实现不漂移**：改一处必须同步改另一处。
    """
    assert sd.draft_reason_for(spec) == strategies._draft_reason_from_flags(spec)


def test_mark_status_writes_status_and_reason():
    """design/iterate 返回给前端的 spec 也带 status/draft_reason。"""
    draft = sd.mark_status({"overfit_inconclusive": True})
    assert draft["status"] == "draft" and draft["draft_reason"]
    ok = sd.mark_status({})
    assert ok["status"] == "active" and "draft_reason" not in ok


# ---------------- 三、迭代门：诊断量而非硬门槛 ----------------

def test_pnl_stats_flags_a_single_lucky_trade():
    """样本量感知诊断：3 笔里 1 笔大赚 → t 很小；50 笔稳定小赚 → t 很大。

    这正是"5 笔里 1 笔运气单凑出 +1pp"的量化识别方式。
    """
    lucky = _pnl_stats([100.0, -5.0, -5.0])
    steady = _pnl_stats([2.0] * 49 + [2.5])
    assert lucky["t_stat"] < 1.0 < steady["t_stat"]
    assert lucky["avg_pnl"] == pytest.approx(30.0)
    # 退化输入不发散
    assert _pnl_stats([]) == {"t_stat": None, "avg_pnl": None, "pnl_std": None}
    assert _pnl_stats([5.0])["t_stat"] is None            # 单笔算不出标准差
    assert _pnl_stats([3.0, 3.0])["t_stat"] is None       # 零方差


def _df(n=600):
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    price = [100 + 8 * math.sin(i / 11) for i in range(n)]
    return pd.DataFrame({"open": price, "high": [p + 1 for p in price],
                         "low": [p - 1 for p in price], "close": price,
                         "volume": [10.0] * n}, index=ts)


def test_gate_exposes_t_stat_as_diagnostic_not_gate():
    """t 统计量只出现在 diagnostics 里，**不进** checks / improved 判定。

    原计划把它当第 6 道硬门槛，但真实数据实测（probe_l2_density.py E2）：
    留出段每笔 t 全为负、24/24 候选被挡，门会变成一堵墙；
    且它与未见段的秩相关并不高于裸收益（+0.305 vs +0.337 等）。
    所以这里锁住"它有值、但不参与判定"。
    """
    df = _df()
    cmp_ = evaluate_iteration_gate(
        df, symbol="BTC/USDT", timeframe="1h",
        base=("dual_ma", {"fast_period": 10, "slow_period": 30}),
        parent=("dual_ma", {"fast_period": 10, "slow_period": 30}),
        new=("dual_ma", {"fast_period": 6, "slow_period": 18}))
    assert cmp_ is not None
    # 诊断量在，且是每笔口径
    diag = cmp_["diagnostics"]
    assert set(diag) == {"oos_new_t_stat", "oos_base_t_stat", "oos_par_t_stat",
                         "oos_new_avg_pnl"}
    # 判定项里没有 t / edge —— 它是诊断，不是门
    assert "edge" not in cmp_["checks"] and "t_stat" not in cmp_["checks"]
    # L2：留出段成交门槛 5 → 10（防"5 笔里 1 笔运气单"）
    assert cmp_["criteria"]["min_trades"] == _ITER_MIN_TRADES == 10


# ---------------- 四、迭代注册时草稿标记不能丢 ----------------

class _Sess:
    def __init__(self, added):
        self._added = added

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *_a, **_k):
        class _R:
            def scalar_one_or_none(self):
                return None
        return _R()

    def add(self, obj):
        self._added.append(obj)

    async def commit(self):
        pass


class _Db:
    """只提供 iterate() 实际用到的最小面：kv（版本号）+ session（落库）。"""

    def __init__(self, kv=None):
        self.kv = dict(kv or {})
        self.added = []

    async def kv_get(self, k, default=None):
        return self.kv.get(k, default)

    async def kv_set(self, k, v):
        self.kv[k] = str(v)

    def session(self):
        return _Sess(self.added)


class _Client:
    async def chat_json_validated(self, messages, feature=None, ctx=None):
        return {"name": "dual_ma", "title": "t", "description": "d", "logic": "l",
                "params": {"fast_period": 8, "slow_period": 21},
                "critique": "c", "improvements": [], "summary": "s", "risk_tips": []}


async def test_iterate_registers_inconclusive_as_draft(monkeypatch, _dynamic_snapshot):
    """回归：迭代产物"证据不足"时，注册进 _DYNAMIC 的 spec 必须带草稿标记。

    修复前 `iterate()` 只把 blocked_by_overfit 带进 iter_spec，
    overfit_inconclusive 只写在返回给前端的 spec 上 → 落库/_DYNAMIC 里没有该字段，
    重启或换页后看不出"证据不足"，而它照样能被选作迭代父代继续繁殖。
    """
    monkeypatch.setattr(iteration_mod, "iteration_messages", lambda *a, **k: [])

    async def _fake_guard(name, params, snap, executor="dual_ma", guard_candles=None,
                          **_kw):
        return {"verdict": "无法判定", "score": None, "avg_oos_trades": 0.5,
                "data_source": "真实K线", "flags": []}

    monkeypatch.setattr(iteration_mod, "_guard_ai_strategy", _fake_guard)

    it = iteration_mod.StrategyIteration(
        _Client(), _Db({"ai_iter_count_dual_ma": "0"}))
    spec = await it.iterate(strategies.get_strategy("dual_ma"), {}, {"symbol": "BTC/USDT",
                                                                    "timeframe": "1h"})

    assert spec["registered"] is True
    assert spec["overfit_inconclusive"] is True
    assert spec["status"] == "draft" and spec["draft_reason"]
    # 注册表里同样带着标记，且被判为草稿（不能当迭代父代 / AI 不接管调参）
    registered = strategies.get_dynamic(spec["name"])
    assert registered["overfit_inconclusive"] is True
    assert registered["status"] == "draft"
    assert strategies.is_draft(spec["name"]) is True
