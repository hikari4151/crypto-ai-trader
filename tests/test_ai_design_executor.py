"""AI 策略设计的注册前关卡：执行器路由 + 信号密度门。

背景一：此前所有 AI 设计都被压进 price_action 一个执行器 —— 用户要"趋势跟踪"，
AI 也只能填关键位突破参数，多层 AND 过滤后近 1000 根只出个位数信号。
背景二：这类"跑起来不出信号"的设计过去会直接注册并进策略库，被后续优化当成可用策略。
"""
import pytest

import strategies
from ai import strategy_designer as sd
from ai.validator import ValidationError, validate_ai_output

SNAP = {"symbol": "BTC/USDT", "timeframe": "1h", "price": 100.0, "change_24h": 0.1}


class _ScalarResult:
    def scalar_one_or_none(self):
        return None


class _FakeSession:
    def __init__(self, added):
        self._added = added

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *_a):
        return _ScalarResult()

    def add(self, obj):
        self._added.append(obj)

    async def commit(self):
        pass


class _FakeDb:
    def __init__(self):
        self.added = []

    def session(self):
        return _FakeSession(self.added)


class _FakeClient:
    """照真实链路走一遍校验器：ctx 不对，输出就应当在返回前被拒。"""

    def __init__(self, result):
        self.result = dict(result)
        self.messages = None
        self.ctx = None
        self.calls = []

    async def chat_json_validated(self, messages, feature="", ctx=None):
        self.messages = messages
        self.ctx = ctx
        self.calls.append(messages)
        validate_ai_output(feature, self.result, ctx or {})
        return dict(self.result)


def _designer(result, monkeypatch):
    """构造 designer 并把过拟合守卫换成记录调用的替身（守卫本身要跑数百次回测）。"""
    calls = []

    async def _fake_guard(name, params, snap, executor="price_action", guard_candles=None,
                          **_kw):
        calls.append({"name": name, "executor": executor, "params": dict(params)})
        return None

    monkeypatch.setattr(sd, "_guard_ai_strategy", _fake_guard)
    client = _FakeClient(result)
    return sd.StrategyDesigner(client, _FakeDb(), None), client, calls


@pytest.fixture(autouse=True)
def _isolate_registry():
    snap = dict(strategies._DYNAMIC)
    yield
    strategies._DYNAMIC.clear()
    strategies._DYNAMIC.update(snap)


# ---------------- 1. 类型 → 执行器映射 ----------------

@pytest.mark.parametrize("strategy_type,expected", [
    ("trend", "dual_ma"),
    ("breakout", "price_action"),
    ("mean_reversion", "factor_signal"),
    ("grid", "grid"),
])
def test_strategy_type_pins_executor(strategy_type, expected):
    # AI 即使坚持别的执行器，用户指定的类型也必须赢
    assert sd.resolve_design_executor(strategy_type, "price_action") == expected


@pytest.mark.parametrize("ai_choice,expected", [
    ("factor_signal", "factor_signal"),
    ("grid", "grid"),
    ("rl_adaptive", "price_action"),   # 需要训练权重，不在目录内
    ("", "price_action"),
    ("made_up", "price_action"),
])
def test_ai_choice_only_accepted_within_catalog(ai_choice, expected):
    assert sd.resolve_design_executor("custom", ai_choice) == expected


def test_catalog_excludes_executors_needing_trained_models():
    catalog = sd.design_executor_catalog()
    assert set(catalog) == set(sd.DESIGN_EXECUTORS)
    assert "rl_adaptive" not in catalog and "meta_controller" not in catalog
    for name, info in catalog.items():
        assert info["param_schema"], f"{name} 的 schema 不应为空"
        # 目录里的每个执行器都必须能被 strategies 真正实例化
        assert isinstance(strategies.get_strategy(name), object)


# ---------------- 2. 设计主链路：执行器落到 spec ----------------

async def test_trend_design_runs_on_dual_ma_not_price_action(monkeypatch):
    designer, client, guard_calls = _designer({
        "name": "ma_trend", "title": "均线趋势", "description": "d", "logic": "l",
        "executor": "price_action",       # AI 想赖在关键位模板上
        "params": {"fast_period": 7, "slow_period": 25, "size_pct": 0.4,
                   "stop_loss_pct": 0.02, "take_profit_pct": 0.05},
        "risk_tips": [], "pine_code": "",
    }, monkeypatch)

    spec = await designer.design(SNAP, [], [], strategy_type="trend")

    assert spec["executor"] == "dual_ma"
    assert set(spec["params"]) == set(strategies._REGISTRY["dual_ma"].param_schema)
    assert spec["params"]["fast_period"] == 7
    # 守卫也跑在所选执行器上：跑错执行器的守卫等于没守卫
    assert guard_calls[0]["executor"] == "dual_ma"
    # 注册后可被取回，且不是 price_action 的空壳
    assert strategies.get_dynamic("ma_trend")["executor"] == "dual_ma"
    assert isinstance(strategies.get_strategy("ma_trend"), strategies._REGISTRY["dual_ma"])


async def test_free_choice_design_respects_ai_executor(monkeypatch):
    designer, _, guard_calls = _designer({
        "name": "rsi_fade", "title": "超卖回归", "description": "d", "logic": "l",
        "executor": "factor_signal",
        "params": {"factor": "rsi_osc", "mode": "reversal", "buy_threshold": 25.0,
                   "sell_threshold": 55.0, "stop_loss_pct": 0.03,
                   "take_profit_pct": 0.06, "size_pct": 0.5},
        "risk_tips": [], "pine_code": "",
    }, monkeypatch)

    spec = await designer.design(SNAP, [], [], strategy_type="")

    assert spec["executor"] == "factor_signal"
    assert spec["params"]["factor"] == "rsi_osc"
    assert guard_calls[0]["executor"] == "factor_signal"


async def test_pinned_type_prompt_offers_only_that_executor(monkeypatch):
    designer, client, _ = _designer({
        "name": "grid_box", "title": "网格", "description": "d", "logic": "l",
        "executor": "grid",
        "params": {"grid_pct": 0.015, "qty_per_grid": 0.001, "max_positions": 10},
        "risk_tips": [], "pine_code": "",
    }, monkeypatch)

    await designer.design(SNAP, [], [], strategy_type="grid")

    assert client.ctx["allowed_executors"] == ("grid",)
    assert set(client.ctx["executor_schemas"]) == {"grid"}
    sys_msg = client.messages[0]["content"]
    # 固定时不给 AI 看别的执行器的参数键，否则它会串用
    assert "双均线交叉" not in sys_msg
    assert "grid_pct" in sys_msg


async def test_free_choice_prompt_lists_every_catalog_executor(monkeypatch):
    designer, client, _ = _designer({
        "name": "any_one", "title": "t", "description": "d", "logic": "l",
        "executor": "dual_ma",
        "params": {"fast_period": 5, "slow_period": 20, "size_pct": 0.5,
                   "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        "risk_tips": [], "pine_code": "",
    }, monkeypatch)

    await designer.design(SNAP, [], [], strategy_type="custom")

    assert client.ctx["allowed_executors"] == sd.DESIGN_EXECUTORS
    sys_msg = client.messages[0]["content"]
    for name in sd.DESIGN_EXECUTORS:
        assert name in sys_msg


# ---------------- 3. 参数校验跟着执行器走 ----------------

def test_params_are_validated_against_chosen_executor_schema():
    """trend 固定成 dual_ma 后，AI 再交关键位参数就是不合法的 —— 必须拒。"""
    catalog = sd.design_executor_catalog()
    ctx = {
        "param_schema": catalog["dual_ma"]["param_schema"],
        "allowed_executors": ("dual_ma",),
        "executor_schemas": {"dual_ma": catalog["dual_ma"]["param_schema"]},
    }
    bad = {"name": "x", "title": "t", "description": "d", "logic": "l",
           "executor": "dual_ma",
           "params": {"mode": "breakout", "breakout_pct": 0.002}}
    with pytest.raises(ValidationError) as e:
        validate_ai_output("strategy_design", bad, ctx)
    assert any(err["field"].startswith("params.") for err in e.value.errors)


def test_unknown_executor_is_rejected_before_registration():
    catalog = sd.design_executor_catalog()
    ctx = {
        "param_schema": catalog["price_action"]["param_schema"],
        "allowed_executors": sd.DESIGN_EXECUTORS,
        "executor_schemas": {k: v["param_schema"] for k, v in catalog.items()},
    }
    with pytest.raises(ValidationError) as e:
        validate_ai_output("strategy_design", {
            "name": "x", "title": "t", "description": "d", "logic": "l",
            "executor": "my_own_executor", "params": {}}, ctx)
    assert e.value.errors[0]["field"] == "executor"


def test_design_prompt_still_requires_trade_markers():
    from ai.prompts import design_strategy_messages
    catalog = sd.design_executor_catalog()
    sys_msg = design_strategy_messages(
        SNAP, [], executor_catalog={"dual_ma": catalog["dual_ma"]})[0]["content"]
    assert "strategy.position_size" in sys_msg
    # 无等价模板的执行器：宁可留空给原因，也不许套用关键位模板凑假代码
    assert "把 pine_code 留空" in sys_msg


# ---------------- 4. Pine 参数回读不得污染其他执行器 ----------------

async def test_pine_input_readback_only_applies_to_price_action(monkeypatch):
    """parse_pine_params 解析的是 price_action 的变量名表。
    若对 dual_ma 策略也套用，会把 breakout_pct 之类塞进一份双均线参数里。"""
    pine = "\n".join([
        "//@version=5",
        'strategy("ma_trend", overlay=true)',
        'fastLen    = input.int(5, "快线")',
        'slowLen    = input.int(20, "慢线")',
        'breakoutPct = input.float(0.001, "突破幅度")',
    ])
    designer, _, guard_calls = _designer({
        "name": "ma_pine", "title": "t", "description": "d", "logic": "l",
        "executor": "dual_ma",
        "params": {"fast_period": 7, "slow_period": 25, "size_pct": 0.5,
                   "stop_loss_pct": 0.02, "take_profit_pct": 0.05},
        "risk_tips": [], "pine_code": pine,
    }, monkeypatch)

    spec = await designer.design(SNAP, [], [], strategy_type="trend")

    assert "breakout_pct" not in spec["params"]
    assert spec["params"]["fast_period"] == 7
    assert guard_calls[0]["params"] == spec["params"]


async def test_price_action_still_reads_params_back_from_pine(monkeypatch):
    pine = "\n".join([
        "//@version=5",
        'strategy("sr", overlay=true)',
        'mode        = input.string("breakout", "入场模式")',
        'breakoutPct = input.float(0.0025, "突破幅度")',
        'volConfirm  = input.float(1.8, "放量倍数")',
        'rsiOB       = input.float(78, "RSI超买")',
        'slPct       = input.float(0.012, "止损")',
        'tpPct       = input.float(0.03, "止盈")',
        'useSRStop   = input.bool(false, "止损参考关键位")',
        'sizePct     = input.float(30.0, "仓位")',
    ])
    designer, _, _ = _designer({
        "name": "sr_pine", "title": "t", "description": "d", "logic": "l",
        "executor": "price_action",
        "params": {"mode": "breakout", "breakout_pct": 0.001, "volume_confirm": 1.2,
                   "rsi_ob": 72, "stop_loss_pct": 0.02, "take_profit_pct": 0.04,
                   "use_sr_stop": True, "size_pct": 0.5},
        "risk_tips": [], "pine_code": pine,
    }, monkeypatch)

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert spec["params"]["breakout_pct"] == pytest.approx(0.0025)
    assert spec["params"]["use_sr_stop"] is False


# ---------------- 5. 注册前信号密度门 ----------------

def _density_result(trades, available=True):
    """密度门替身结果：达标与否走**真实**的归一化判定（`sd._judge_density`）。

    替身自己再写一套判据会跟真门漂移：L2 把门槛从"绝对 5 笔"改成"≥15 笔/1000 根"时，
    写死的 `trades >= sd._DENSITY_MIN_TRADES` 让 9/12 笔的替身突然变成"不过"，
    把 4 个链路测试一起带崩——链路测试关心的是"过 / 不过之后怎么办"，不是门槛数值。
    """
    j = sd._judge_density(trades if available else 0, sd._DENSITY_CANDLES)
    return {"available": available, "passed": (not available) or j["passed"],
            "trades": trades if available else None,
            "candles": sd._DENSITY_CANDLES, "required_candles": sd._DENSITY_CANDLES,
            "min_trades": j["min_trades"], "trades_per_1000": j["trades_per_1000"],
            "min_trades_per_1000": sd._DENSITY_MIN_TRADES_PER_1000,
            "symbol": "BTC/USDT",
            "timeframe": "1h", "reason": ""}


def _fake_density(monkeypatch, outcomes):
    """把密度门换成可控替身：真实门要跑回测，链路测试只关心"过 / 不过之后怎么办"。"""
    calls = []

    async def _fake(executor, params, snap, guard_candles=None):
        idx = min(len(calls), len(outcomes) - 1)
        calls.append({"executor": executor, "params": dict(params), "result": outcomes[idx]})
        return dict(outcomes[idx])

    monkeypatch.setattr(sd, "_check_signal_density", _fake)
    return calls


PA_RESULT = {
    "name": "sr_dead", "title": "关键位突破", "description": "d", "logic": "l",
    "executor": "price_action",
    "params": {"mode": "breakout", "breakout_pct": 0.002, "volume_confirm": 2.5,
               "rsi_ob": 75, "stop_loss_pct": 0.02, "take_profit_pct": 0.04,
               "use_sr_stop": True, "size_pct": 0.5},
    "risk_tips": [], "pine_code": "",
}


async def test_zero_signal_design_is_not_registered(monkeypatch):
    designer, client, _ = _designer(PA_RESULT, monkeypatch)
    _fake_density(monkeypatch, [_density_result(0)])

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert spec["blocked_by_signal_density"] is True
    # 关键：不出信号的设计不能进注册表，否则会被策略库与后续优化当成"可用策略"
    assert "sr_dead" not in strategies._DYNAMIC
    # 用完全部重试额度后才放弃
    assert len(client.calls) == sd._DENSITY_MAX_RETRIES + 1


async def test_blocked_design_is_logged_for_audit(monkeypatch):
    designer, _, _ = _designer(PA_RESULT, monkeypatch)
    _fake_density(monkeypatch, [_density_result(0)])

    await designer.design(SNAP, [], [], strategy_type="breakout")

    rows = [o for o in designer._db.added if getattr(o, "kind", "") == "design_gate"]
    assert len(rows) == 1
    assert "0 笔" in rows[0].summary
    assert "breakout_pct" in rows[0].suggestion  # 记下该执行器的真实旋钮，供人工复盘


async def test_density_failure_is_fed_back_with_real_knobs(monkeypatch):
    designer, client, _ = _designer(PA_RESULT, monkeypatch)
    _fake_density(monkeypatch, [_density_result(0)])

    await designer.design(SNAP, [], [], strategy_type="breakout")

    first_user = client.calls[0][1]["content"]
    retry_user = client.calls[1][1]["content"]
    assert "未通过注册前校验" not in first_user
    assert "未通过注册前校验" in retry_user
    assert "只成交 0 笔" in retry_user
    # 反馈必须点名该执行器真实存在的旋钮，而不是让 AI 再猜一轮
    assert "breakout_pct" in retry_user and "volume_confirm" in retry_user


async def test_retry_that_passes_the_gate_gets_registered(monkeypatch):
    designer, client, calls = _designer(PA_RESULT, monkeypatch)
    _fake_density(monkeypatch, [_density_result(0), _density_result(16)])

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert "blocked_by_signal_density" not in spec
    assert strategies._DYNAMIC["sr_dead"]["executor"] == "price_action"
    assert len(client.calls) == 2
    # 被门拦下的那一版根本不该浪费数百次回测去过拟合守卫
    assert len(calls) == 1


async def test_gate_measures_the_final_params_not_the_ai_echo(monkeypatch):
    """门要量"最终会注册的那份参数"：Pine 回读改掉的值必须体现在测量里。"""
    pine = "\n".join([
        "//@version=5", 'strategy("sr", overlay=true)',
        'mode        = input.string("pullback", "入场模式")',
        'breakoutPct = input.float(0.0005, "突破幅度")',
        'volConfirm  = input.float(0.6, "放量倍数")',
        'rsiOB       = input.float(85, "RSI超买")',
        'slPct       = input.float(0.02, "止损")',
        'tpPct       = input.float(0.04, "止盈")',
        'useSRStop   = input.bool(true, "止损参考关键位")',
        'sizePct     = input.float(50.0, "仓位")',
    ])
    result = {**PA_RESULT, "name": "sr_live", "pine_code": pine}
    designer, _, calls = _designer(result, monkeypatch)
    _fake_density(monkeypatch, [_density_result(16)])

    await designer.design(SNAP, [], [], strategy_type="breakout")

    measured = calls[0]["params"]
    assert measured["breakout_pct"] == pytest.approx(0.0005)
    assert measured["mode"] == "pullback"
    assert measured == strategies._DYNAMIC["sr_live"]["params"]


async def test_insufficient_history_warns_but_does_not_block(monkeypatch):
    designer, client, _ = _designer(PA_RESULT, monkeypatch)
    _fake_density(monkeypatch, [_density_result(None, available=False)])

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert "blocked_by_signal_density" not in spec
    assert spec["signal_density"]["available"] is False
    # 没数据不是 AI 的错，不该据此把它重试到死
    assert len(client.calls) == 1


def _closes(prices):
    ts0 = 1700000000000
    return [[ts0 + i * 3600000, p, p + 1, p - 1, p, 10.0] for i, p in enumerate(prices)]


def test_measure_signal_density_needs_full_window():
    res = sd._measure_signal_density("dual_ma", {}, _closes([100.0] * 500), "BTC/USDT", "1h")
    assert res["available"] is False and res["passed"] is True
    assert res["trades"] is None


def test_measure_signal_density_counts_real_trades():
    import math
    params = {"fast_period": 3, "slow_period": 8, "size_pct": 0.5,
              "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
    candles = _closes([100 + 5 * math.sin(i / 5) for i in range(1200)])
    res = sd._measure_signal_density("dual_ma", params, candles, "BTC/USDT", "1h")
    # 只量最近 1000 根，多给的历史不算数
    assert res["candles"] == sd._DENSITY_CANDLES
    assert res["available"] is True and res["passed"] is True
    assert res["trades"] >= sd._DENSITY_MIN_TRADES


def test_measure_signal_density_catches_a_dead_strategy():
    """横盘无波动的双均线：真实回测 0 笔 —— 这正是过去能混进策略库的那一类。"""
    params = {"fast_period": 5, "slow_period": 20, "size_pct": 0.5,
              "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
    res = sd._measure_signal_density("dual_ma", params, _closes([100.0] * 1200),
                                     "BTC/USDT", "1h")
    assert res["available"] is True
    assert res["trades"] == 0 and res["passed"] is False


# ---------------- 6. 过拟合守卫的第三种结论 ----------------

def _designer_with_verdict(monkeypatch, verdicts):
    """密度门给过，只让守卫返回指定结论（verdicts 可传列表按调用次数轮转）：
    真实守卫要跑数百次回测，链路测试只关心"过 / 不过之后怎么办"。"""
    verdicts = [verdicts] if isinstance(verdicts, str) else list(verdicts)
    calls = []

    async def _fake_guard(name, params, snap, executor="price_action", guard_candles=None,
                          **_kw):
        idx = min(len(calls), len(verdicts) - 1)
        calls.append({"name": name, "executor": executor, "params": dict(params)})
        return {"verdict": verdicts[idx], "score": 50.9, "oos_ret": 0.0, "pbo": None,
                "decay": None, "n_folds": 0}

    monkeypatch.setattr(sd, "_guard_ai_strategy", _fake_guard)
    _fake_density(monkeypatch, [_density_result(16)])
    return sd.StrategyDesigner(_FakeClient(PA_RESULT), _FakeDb(), None), calls


async def test_inconclusive_guard_registers_but_bars_auto_takeover(monkeypatch):
    """"无法判定"＝证据不足，不是有罪：可以注册供人工启用，但打标禁止无人接管实盘。"""
    designer, _ = _designer_with_verdict(monkeypatch, "无法判定")

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert spec["overfit"]["verdict"] == "无法判定"
    assert spec["overfit_inconclusive"] is True
    assert "blocked_by_overfit" not in spec
    assert "sr_dead" in strategies._DYNAMIC


async def test_severe_overfit_guard_registers_but_flags(monkeypatch):
    """P4-D8：严重过拟合先回炉重试（_OVERFIT_MAX_RETRIES 次仍过拟合）才注册打标——
    兜底保证策略仍入库供人工启用，保留 blocked_by_overfit 标记
    （前端警告 + 引擎自动接管门拦截，见 test_ai_guard_data）。"""
    designer, guard_calls = _designer_with_verdict(monkeypatch, "严重过拟合")

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    assert spec["blocked_by_overfit"] is True
    assert spec["overfit"]["verdict"] == "严重过拟合"
    # 回炉消耗了全部重试次数（guard 每次都被调）
    assert len(guard_calls) == sd._OVERFIT_MAX_RETRIES + 1
    # 仍注册（用户可手动启用），但带标记
    assert "sr_dead" in strategies._DYNAMIC
    assert strategies._DYNAMIC["sr_dead"].get("blocked_by_overfit") is True
    # 判得清楚（通过）时不该留下任何拦截标记
    ok_designer, _ = _designer_with_verdict(monkeypatch, "通过")
    ok_spec = await ok_designer.design(SNAP, [], [], strategy_type="breakout")
    assert "overfit_inconclusive" not in ok_spec and "blocked_by_overfit" not in ok_spec


# ---------------- 6.5 P4-D8：过拟合回炉（检测报告喂回 AI 重设计） ----------------

def _client_with_results(results):
    """按调用次数返回不同 AI 结果的 fake client（首版严重过拟合，回炉版通过）。"""
    class _SeqClient:
        def __init__(self, results):
            self.results = list(results)
            self.calls = []
            self.ctx = None
            self.messages = None

        async def chat_json_validated(self, messages, feature="", ctx=None):
            self.messages = messages
            self.calls.append(messages)
            self.ctx = ctx
            result = dict(self.results[min(len(self.calls) - 1, len(self.results) - 1)])
            return result

    return _SeqClient(results)


async def test_severe_overfit_first_then_pass_registers_clean(monkeypatch):
    """回炉生效：首版严重过拟合 → 检测报告喂回 AI → 第二版通过 → 正常注册、无拦截标记。"""
    # 建议分两版：第一版被判严重过拟合，第二版「AI 修正后」通过
    client = _client_with_results([PA_RESULT, {
        **PA_RESULT, "name": "sr_v2", "params": {
            **PA_RESULT["params"], "volume_confirm": 1.5, "rsi_ob": 70.0}}])
    verdicts = ["严重过拟合", "通过"]
    calls = []

    async def _flaky_guard(name, params, snap, executor="price_action", guard_candles=None,
                           **_kw):
        idx = min(len(calls), len(verdicts) - 1)
        calls.append({"name": name, "executor": executor, "params": dict(params)})
        return {"verdict": verdicts[idx], "score": 50.9, "oos_ret": 0.0, "pbo": None,
                "decay": None, "n_folds": 0}

    monkeypatch.setattr(sd, "_guard_ai_strategy", _flaky_guard)
    _fake_density(monkeypatch, [_density_result(16)])
    designer = sd.StrategyDesigner(client, _FakeDb(), None)

    spec = await designer.design(SNAP, [], [], strategy_type="breakout")

    # 第二版通过 → 干净注册（无任何拦截标记）
    assert "blocked_by_overfit" not in spec and "overfit_inconclusive" not in spec
    assert "sr_v2" in strategies._DYNAMIC
    assert not strategies._DYNAMIC["sr_v2"].get("blocked_by_overfit", False)
    # 两版各喂了守卫一轮（首版被测、修正后被重测）
    assert len(calls) == 2
    # 第二版 prompt 携有过拟合回炉反馈（P4-D8：把检测报告喂回 AI）
    fb = "".join(m["content"] for m in client.calls[1])
    assert "过拟合检测未通过" in fb or "上一版" in fb
    assert "score" in fb or "PBO" in fb or "过拟合" in fb


async def test_overfit_feedback_mentions_metrics_and_cures(monkeypatch):
    """_overfit_feedback 必须讲清检测指标 + 修正方向（降自由度/收敛/简化），
    否则 AI 无从知道上一版错在哪。"""
    report = {"verdict": "严重过拟合", "score": 30.0, "is_ret": 0.12, "oos_ret": -0.08,
              "decay": 0.9, "pbo": 0.8, "n_folds": 4,
              "flags": [{"msg": "样本外收益为负，过拟合信号强"}]}
    fb = sd._overfit_feedback("price_action", {"mode": "breakout"}, report)

    assert "严重过拟合" in fb
    assert "-0.08" in fb and "0.9" in fb and "0.8" in fb   # oos_ret / decay / pbo 都进入反馈
    assert any(k in fb for k in ("自由度", "收敛", "简化"))
    assert "上一版参数" in fb and "mode" in fb
