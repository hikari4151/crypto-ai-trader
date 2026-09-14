"""本批优化项（AI设计 / 深度反思迭代）的回归测试。

覆盖：
- 一.3 迭代版本号避让已占用名（_DYNAMIC/内置名冲突不再静默覆盖）
- 二.1/二.2 验证门多指标门槛 + 回炉反馈文案（含 0 成交与绩效未提升两种）
- 一.6 few-shot 按固定执行器 schema 动态生成
- 二.8 校验重试回灌时 Pine 大字段截断
- 一.5 Pine ↔ params 一致性预检（mode 单独认声明）
- 一.4 S/R 缺失时提示不可用而非输出 None
- 二.9 设计/迭代提示词结构评测（canned produce 走通指标闭环）
"""
import asyncio
import json

import pytest

import strategies
from ai import strategy_designer as sd
from ai.client import _compact_output, _fewshot_for
from ai.evals import GOLDEN_SET, evaluate_design_prompt, evaluate_iterate_prompt
from ai.iteration import StrategyIteration, _validation_feedback
from ai.prompts import _sr_pa_block, iteration_messages
from strategies.dual_ma import DualMAStrategy
from strategies.pine_utils import check_pine_consistency


@pytest.fixture(autouse=True)
def _isolate_registry():
    snap = dict(strategies._DYNAMIC)
    yield
    strategies._DYNAMIC.clear()
    strategies._DYNAMIC.update(snap)


# ---------------- 二.1/二.2 验证门反馈 ----------------

def _cmp(old_ret, new_ret, old_dd=0.05, new_dd=0.04, old_sh=1.0, new_sh=1.1,
         old_trades=30, new_trades=40):
    return {
        "old": {"total_return": old_ret, "sharpe": old_sh, "max_drawdown": old_dd,
                "win_rate": 0.5, "total_trades": old_trades},
        "new": {"total_return": new_ret, "sharpe": new_sh, "max_drawdown": new_dd,
                "win_rate": 0.55, "total_trades": new_trades},
    }


def test_validation_feedback_no_trade_mentions_knobs():
    fb = _validation_feedback("grid", {"grid_pct": 0.05}, _cmp(0.0, 0.0, old_trades=0, new_trades=0),
                              no_trade=True)
    assert "无成交" in fb and "grid_pct" in fb, fb[:120]


def test_validation_feedback_improve_mentions_metrics_and_params():
    fb = _validation_feedback("dual_ma", {"fast_period": 12, "slow_period": 40},
                              _cmp(-0.02, -0.03), no_trade=False)
    assert "夏普" in fb and "回撤" in fb and "上一版参数" in fb and "fast_period" in fb, fb[:160]


# ---------------- 一.3 迭代版本号避让 ----------------

class _KvDb:
    def __init__(self, kv=None):
        self.kv = dict(kv or {})

    async def kv_get(self, k, default=None):
        return self.kv.get(k, default)

    async def kv_set(self, k, v):
        self.kv[k] = str(v)

    def session(self):  # pragma: no cover - 本测试不触库
        raise NotImplementedError


def test_next_version_skips_existing_collision():
    strategies._DYNAMIC["dual_ma_v1"] = {"executor": "dual_ma"}
    it = StrategyIteration(None, _KvDb({"ai_iter_count_dual_ma": "0"}))
    assert asyncio.run(it._next_version("dual_ma")) == 2, "v1 被占应跳到 v2"
    assert asyncio.run(it._next_version("dual_ma")) == 3, "第二次应继续递增"


def test_next_version_falls_back_to_one_on_db_error():
    class _BadDb:
        async def kv_get(self, *_a, **_k):
            raise RuntimeError("db down")

        async def kv_set(self, *_a, **_k):
            raise RuntimeError("db down")

    it = StrategyIteration(None, _BadDb())
    assert asyncio.run(it._next_version("dual_ma")) == 1


# ---------------- 一.6 few-shot 动态 ----------------

def test_fewshot_dynamic_uses_pinned_executor_schema():
    fs = _fewshot_for("strategy_design", {
        "allowed_executors": ("dual_ma",),
        "executor_schemas": {"dual_ma": DualMAStrategy.param_schema}})
    assert fs is not None
    d = json.loads(fs)
    assert set(d["params"]) <= set(DualMAStrategy.param_schema), "示例参数键必须属于所选执行器"
    assert _fewshot_for("market_analysis", None) is not None, "非设计功能静态示例保持"


# ---------------- 二.8 回灌截断 ----------------

def test_compact_output_truncates_pine():
    c = json.loads(_compact_output({"params": {"a": 1}, "pine_code": "// x" * 500}))
    assert len(c["pine_code"]) < 2000 and "已截断" in c["pine_code"]


# ---------------- 一.5 Pine 一致性预检 ----------------

def test_pine_consistency_price_action_complete_passes():
    code = (
        '//@version=5\n'
        'strategy("t", shorttitle="t", overlay=true, margin_long=100, margin_short=100,\n'
        '         default_qty_type=strategy.percent_of_equity, default_qty_value=100)\n'
        'mode = input.string("breakout", "模式")\n'
        'breakoutPct = input.float(0.005, "突破幅度")\n')
    ck = check_pine_consistency(code, "price_action", {"mode": "breakout", "breakout_pct": 0.01})
    assert ck["ok"], f"声明齐全应通过: {ck['missing']}"


def test_pine_consistency_reports_missing_decl():
    ck = check_pine_consistency('//@version=5\nstrategy("t", shorttitle="t")\n',
                                "price_action", {"mode": "breakout", "breakout_pct": 0.01})
    assert not ck["ok"]
    assert any("params.mode" in m for m in ck["missing"])
    assert any("params.breakout_pct" in m for m in ck["missing"]), "缺 input 声明应被报出"


def test_pine_consistency_reports_missing_version_and_strategy():
    ck = check_pine_consistency("x = 1\n", "dual_ma", {})
    assert not ck["ok"]
    miss = " ".join(ck["missing"])
    assert "//@version=5" in miss and "strategy()" in miss


def test_pine_consistency_empty_code():
    assert not check_pine_consistency("", "grid", {})["ok"]


# ---------------- 一.4 S/R 缺失提示 ----------------

def test_sr_pa_block_missing_hint_instead_of_none():
    block = _sr_pa_block({"indicators": {}})
    assert "数据当前不可用" in block and "None" not in block
    assert "支撑位" in _sr_pa_block({"indicators": {"sr": {"support": 1}, "pa": {}}})


# ---------------- 二.3 每日盈亏进入迭代 prompt ----------------

def test_iteration_prompt_renders_daily_pnl_and_gate_feedback():
    st = DualMAStrategy()
    perf = {"total_trades": 10, "win_rate": 0.5, "total_pnl": 1.2,
            "daily_pnl": [{"date": "2026-01-01", "pnl": 0.3}]}
    snap = {"symbol": "BTC/USDT", "timeframe": "1h",
            "candles": [[1700000000000, 100, 101, 99, 100.5, 1000]], "indicators": {}}
    msgs = iteration_messages(st, perf, snap, [], [], goal="g",
                              overfit_feedback="过拟合反馈", validation_feedback="验证门反馈")
    u = msgs[-1]["content"]
    assert "每日盈亏" in u and "过拟合反馈" in u and "验证门反馈" in u


# ---------------- 二.9 设计/迭代提示词结构评测 ----------------

def test_evaluate_design_prompt_structure():
    def produce(snap):
        return {"name": "t1", "title": "t", "description": "d", "logic": "l",
                "executor": "price_action",
                "params": {"mode": "breakout", "breakout_pct": 0.005},
                "risk_tips": [], "pine_code": ""}

    m = evaluate_design_prompt(produce)
    assert m["cases"] == len(GOLDEN_SET)
    assert m["validator_pass_rate"] == 1.0, m
    assert m["param_schema_coverage"] > 0
    assert m["pine_provision_rate"] == 0.0


def test_evaluate_iterate_prompt_structure():
    def produce(snap):
        return {"name": "dual_ma_v9", "title": "t", "description": "d", "logic": "l",
                "params": {"fast_period": 12, "slow_period": 40},
                "critique": "c", "improvements": [], "summary": "s", "risk_tips": []}

    m = evaluate_iterate_prompt(produce)
    assert m["cases"] == len(GOLDEN_SET)
    assert m["validator_pass_rate"] == 1.0, m
    assert m["param_schema_coverage"] > 0


# ---------------- 一.2 密度门备选长窗口 ----------------

def test_density_alt_window_rescues_on_longer_history(monkeypatch):
    """主窗口 0 成交、备选长窗口以**同一归一化频率**达标 → 设计放行
    （网格类区间策略不再被误杀）。"""

    def _fake_fast(df, cfg, **kw):
        n = len(df)
        # 3000 根 48 笔 = 16 笔/1000 根（≥15 门槛），才算"长窗口确实有信号"
        return {"metrics": {"total_trades": 48 if n >= 3000 else 0}}

    monkeypatch.setattr("backtest.fast_engine.run_backtest_fast", _fake_fast)
    candles = [[1700000000000 + i * 3600_000, 100.0, 101.0, 99.0, 100.5, 1000.0]
               for i in range(3500)]
    res = sd._measure_signal_density("grid", {"grid_pct": 0.5}, candles, "BTC/USDT", "1h")
    assert res["passed"] is True, res
    assert res["trades"] == 48
    assert "备选长窗口" in res["reason"], res["reason"]


def test_density_alt_window_does_not_rescue_diluted_rate(monkeypatch):
    """L2 回归：备选长窗口**不再稀释门槛**（这是"交易少"被制度化的主因）。

    修复前判据是绝对 5 笔 → "3000 根 8 笔"（=2.67 笔/1000 根）照样算达标，
    比主窗口 1000 根 5 笔的门槛还松 —— 窗口越长越容易过。
    现在备选窗口用同一个笔/1000 根门槛（3000 根需 ≥45 笔），8 笔必须判不过，
    且实测频率留在 alt_* 字段里供反馈引用。
    """

    def _fake_fast(df, cfg, **kw):
        n = len(df)
        return {"metrics": {"total_trades": 8 if n >= 3000 else 0}}

    monkeypatch.setattr("backtest.fast_engine.run_backtest_fast", _fake_fast)
    candles = [[1700000000000 + i * 3600_000, 100.0, 101.0, 99.0, 100.5, 1000.0]
               for i in range(3500)]
    res = sd._measure_signal_density("grid", {"grid_pct": 0.5}, candles, "BTC/USDT", "1h")
    assert res["passed"] is False, res
    assert res["alt_candles"] == 3000 and res["alt_trades"] == 8
    assert res["alt_min_trades"] == 45      # 3000 根 × 15/1000
    assert res["alt_trades_per_1000"] == round(8 / 3000 * 1000, 2)