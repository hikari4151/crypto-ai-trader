"""关键要求：任何策略的 Pine 代码贴到 TradingView 图表上，必须画出买入/卖出位置。

持续进化训练出来的策略此前完全不满足这条要求，实测三处叠加：
1. 注册的 rl_evolve / meta_controller spec 不带 pine_code（ai_strategies 表实测
   pine_code 长度为 0）→ 前端 buildPine 无条件套用 price_action 模板，UI 上显示
   的是一段与训练模型无关的默认参数 S/R 策略（假代码）
2. 进化训练 config 为 state_window=20 + hidden=[64,64] → 21184 个权重（≈230KB），
   而导出器写死拒绝 state_window>1 → 进化模型永远导不出 Pine
3. 导出器/模板的可视化：只有 plot(target)（0~1 画在 overlay=true 的价格轴上等于
   看不见）与单个"做多信号" plotshape；止损/止盈/减仓离场一个标记都没有

本文件把这条要求固化为回归测试。
"""
import json
import re

import numpy as np
import pytest

from drl.agent import ACAgent
from indicators.technical import SR_MIN_TOUCHES, SR_WINDOW
from strategies.pine_utils import (TRADE_MARKERS_PINE, build_price_action_pine,
                                   ensure_trade_markers, has_trade_markers,
                                   resolve_pine)


def _markers(code: str) -> tuple[bool, bool]:
    """返回 (有买入标记, 有卖出标记)：按成交方向判定，不依赖各策略自己的信号变量。"""
    up = "triangleup" in code or "arrowup" in code
    down = "triangledown" in code or "arrowdown" in code
    return up, down


def _agent(state_dim: int, n_actions: int = 5, hidden=(16, 16),
           state_window: int = 1) -> ACAgent:
    """默认 hidden=(16,16)：与进化默认 config 同形（能装进 Pine 的真实模型）。"""
    return ACAgent(state_dim=state_dim, n_actions=n_actions, hidden=hidden,
                   seed=7, state_window=state_window)


# ---------------- A. 通用成交标记块 ----------------

def test_price_action_pine_marks_both_buy_and_sell():
    code = build_price_action_pine({"name": "sr_x", "params": {"breakout_pct": 0.004}})
    buy, sell = _markers(code)
    assert buy and sell, "price_action Pine 必须同时画买入与卖出标记"
    # 标记基于真实成交（持仓变化），而不是只标入场信号
    assert "strategy.position_size" in code


def test_price_action_pine_does_not_expose_engine_fixed_sr_config_as_input():
    """摆动窗口/触碰次数/RSI 超卖阈值由引擎常量固定，本地执行器不读取策略参数。
    把它们写成 input.* 会让用户在图表端调了却对实盘无效 —— 两端结果分叉。"""
    code = build_price_action_pine({"name": "sr_cfg", "params": {}})
    for decl in ("srWindow    = input", "srWindow   = input",
                 "minTouches  = input", "minTouches = input", "rsiOS       = input"):
        assert decl not in code, f"关键位口径不应作为可调 input 导出：{decl.strip()}"
    # 变量仍要存在（下游 ta.pivothigh / touchRes 判定引用它们），只是取引擎常量
    assert f"srWindow   = {SR_WINDOW}" in code
    assert f"minTouches = {SR_MIN_TOUCHES}" in code


def test_ensure_trade_markers_appends_marker_less_code():
    """AI 生成/TradingView 导入的 Pine 常见形态：有下单语句、图上零标记。"""
    ai_code = ("//@version=5\nstrategy(\"AI\", overlay=true)\n"
               "if cond\n    strategy.entry(\"L\", strategy.long)\n"
               "strategy.exit(\"XL\", from_entry=\"L\", stop=100)\n")
    assert has_trade_markers(ai_code) is False
    out = ensure_trade_markers(ai_code)
    assert out != ai_code
    buy, sell = _markers(out)
    assert buy and sell
    assert out.startswith("//@version=5")          # 版本声明仍在首行
    assert out.startswith(ai_code)                  # 原代码不被改写


def test_ensure_trade_markers_is_idempotent():
    once = ensure_trade_markers("strategy(\"x\")\nplot(close)\n")
    twice = ensure_trade_markers(once)
    assert twice == once
    assert twice.count("aiTradeBuy") == once.count("aiTradeBuy")


def test_ensure_trade_markers_keeps_code_that_already_marks_both_sides():
    marked = ("strategy(\"x\")\nplotshape(a, style=shape.triangleup)\n"
              "plotshape(b, style=shape.triangledown)\n")
    assert has_trade_markers(marked) is True
    assert ensure_trade_markers(marked) == marked


def test_marker_block_is_valid_pine_statements():
    """标记块自身的最低语法自检：括号配对、无 Python 残留。"""
    assert TRADE_MARKERS_PINE.count("(") == TRADE_MARKERS_PINE.count(")")
    assert "None" not in TRADE_MARKERS_PINE and "True" not in TRADE_MARKERS_PINE


# ---------------- B. DRL 神经引擎导出 ----------------

def test_drl_pine_has_visible_markers():
    from drl.pine_export import build_drl_pine
    code = build_drl_pine(_agent(15), model_name="rl_evolve")
    buy, sell = _markers(code)
    assert buy and sell, "DRL Pine 必须在图上给出买卖点"
    # 目标仓位画在 overlay 价格轴上看不见：必须缩放到价格量纲或独立 pane
    for m in re.finditer(r"^plot\((target)[^)]*", code, re.M):
        assert "close" in m.group(0) or "pane" in m.group(0), m.group(0)


def test_drl_pine_supports_state_window_stacking():
    """state_window>1 曾直接 raise（"Pine 不支持窗口堆叠"）——特征都是 series，
    用 f?[k] 历史引用即可堆叠，顺序与 env._state_feats 一致（旧→新）。"""
    from drl.pine_export import _state_refs, build_drl_pine
    w = 4
    refs = _state_refs(w, has_factor=False)
    assert len(refs) == 13 * w + 2, "状态分量总数必须等于 env 的观测维度"
    assert refs[0] == "nz(f0[3])" and refs[13] == "nz(f0[2])", "最旧一根偏移最大，逐根递减"
    assert refs[13 * (w - 1)] == "f0", "最新一根不应有历史偏移"
    assert refs[-2:] == ["posRatio", "pnlRatio"], "账户状态列排在特征之后"
    assert _state_refs(2, has_factor=True)[-1] == "factorVal"
    code = build_drl_pine(_agent(13 * w + 2, state_window=w), model_name="rl_evolve")
    assert all(r in code for r in refs), "状态数组拼装必须逐分量落地到脚本"
    assert "posRatio, pnlRatio)" in code


def test_drl_pine_target_ternary_is_valid_pine():
    """target 档位映射必须是合法三元链（曾因拼接符误用 ' ? ' 产出非法语句）。"""
    from drl.pine_export import build_drl_pine
    line = next(l for l in build_drl_pine(_agent(15)).splitlines() if l.startswith("target ="))
    assert line == ("target = bestIdx == 0 ? 0.0 : bestIdx == 1 ? 0.25"
                    " : bestIdx == 2 ? 0.5 : bestIdx == 3 ? 0.75 : 1.0")
    assert line.count("?") == line.count(":")


def test_drl_pine_features_use_training_normalization():
    """特征口径必须与 env._precompute_features 一致（P4-B4 tanh 有界 + clip）。"""
    from drl.pine_export import build_drl_pine
    code = build_drl_pine(_agent(15))
    assert "math.tanh" in code, "ret/vol 特征缺 tanh 归一化（训练端已 tanh 压缩）"
    assert "/ 5.0" in code and "/ 3.0" in code, "缺 _TANH_RET/_TANH_VOL 同口径缩放"
    assert "math.min(5.0" in code, "vol_ratio 缺 clip(0,5)/5"
    assert "math.min(10.0" in code or "math.min(10," in code, "macd 缺 clip(-10,10)/10"


def test_drl_pine_rejects_net_too_large_for_pine():
    """262→64→64→5（进化旧 config）= 21184 权重 ≈230KB，Pine 装不下：必须显式拒绝，
    而不是产出一段永远无法编译的代码，更不能回落到别的策略模板。"""
    from drl.pine_export import PineExportError, build_drl_pine
    with pytest.raises(PineExportError) as ei:
        build_drl_pine(_agent(13 * 20 + 2, hidden=(64, 64), state_window=20))
    assert "KB" in str(ei.value) or "权重" in str(ei.value)


def test_try_build_drl_pine_returns_note_instead_of_raising():
    from drl.pine_export import try_build_drl_pine
    code, note = try_build_drl_pine(_agent(15))
    assert code and note == ""
    code2, note2 = try_build_drl_pine(_agent(13 * 20 + 2, hidden=(64, 64), state_window=20))
    assert code2 == "" and note2, "不可导出时必须给出原因说明（供 UI 显式提示）"


def test_drl_pine_rejects_cascade_factor_column_with_clear_reason():
    """级联因子列（factor_miner 预计算值）Pine 端无法重建：必须点名因子，
    而不是含糊报"维度不一致"（用户会以为导出器算错了窗口）。"""
    from drl.pine_export import PineExportError, build_drl_pine
    w = 4
    with pytest.raises(PineExportError) as ei:
        build_drl_pine(_agent(13 * w + 3, state_window=w))   # 多出的 1 维 = 级联因子
    assert "因子" in str(ei.value)


def test_export_pine_from_model_roundtrip(tmp_path):
    """注册策略统一从磁盘模型文件导出 Pine（进化 / DRL 手动训练共用这一入口）。"""
    from drl.pine_export import export_pine_from_model
    w = 4
    path = tmp_path / "strategy_drl.json"
    _agent(13 * w + 2, state_window=w).save(str(path))
    code, note = export_pine_from_model(path, "rl_evolve")
    assert note == "" and code
    buy, sell = _markers(code)
    assert buy and sell, "导出的模型代码必须在图上给出买卖点"
    code2, note2 = export_pine_from_model(tmp_path / "missing.json", "rl_evolve")
    assert code2 == "" and "不存在" in note2


def test_ensure_trade_markers_skips_indicator_scripts():
    """标记块读 strategy.position_size，indicator() 型脚本没有该变量：
    追加标记会让整份脚本编译不过，这类代码原样返回。"""
    ind = "//@version=5\nindicator(\"RSI\")\nplot(ta.rsi(close, 14))\n"
    assert ensure_trade_markers(ind) == ind


def test_evolve_train_cfg_produces_exportable_model():
    """持续进化的训练 config 必须产出可导出 Pine 的模型，否则关键要求永远落空。

    断言复用导出器自己的体积守门（不再写死数字），并用实际生成的脚本字节数兜底，
    防止估算器与真实脚本体积再度脱节。
    """
    from drl.evolve_engine import _strategy_drl_train_cfg
    from drl.pine_export import PINE_MAX_BYTES, build_drl_pine, estimate_pine_bytes
    cfg = _strategy_drl_train_cfg(episodes=8)
    w = int(cfg["state_window"])
    hidden = [int(h) for h in cfg["hidden"]]
    in_dim = 13 * w + 2
    assert estimate_pine_bytes(in_dim, hidden, 5) <= PINE_MAX_BYTES, \
        f"进化模型 {in_dim}→{'→'.join(map(str, hidden))}→5 装不进 Pine"
    code = build_drl_pine(_agent(in_dim, hidden=tuple(hidden), state_window=w),
                          model_name="rl_evolve")
    assert len(code.encode()) <= PINE_MAX_BYTES, \
        f"实际脚本 {len(code.encode()) // 1024}KB 超预算"


@pytest.mark.asyncio
async def test_register_rl_evolve_attaches_real_pine(tmp_path):
    """进化产物注册为策略时必须带真实 Pine（从模型现场导出），不能再是假模板。"""
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    import strategies

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    w = 4
    agent = _agent(13 * w + 2, state_window=w)
    eng.zoo.save_agent_best(agent, "strategy_drl", meta={"fitness": 0.1})
    await eng._register_rl_evolve()
    spec = strategies.get_dynamic("rl_evolve")
    assert spec.get("pine_code"), "rl_evolve 必须带 pine_code"
    buy, sell = _markers(spec["pine_code"])
    assert buy and sell
    assert "rl_adaptive" in spec["pine_code"] or "DRL" in spec["pine_code"]


@pytest.mark.asyncio
async def test_register_rl_evolve_records_note_when_not_exportable(tmp_path):
    """不可导出时 pine_code 留空但必须写明原因（前端据此显式提示，不偷套别的模板）。"""
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from tests.test_drl_optimizations import _MetaDB
    import strategies

    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    agent = _agent(13 * 20 + 2, hidden=(64, 64), state_window=20)
    eng.zoo.save_agent_best(agent, "strategy_drl", meta={"fitness": 0.1})
    await eng._register_rl_evolve()
    spec = strategies.get_dynamic("rl_evolve")
    assert not spec.get("pine_code")
    assert spec.get("pine_note")


# ---------------- C. AI 生成通道的要求下达 ----------------

def test_ai_prompts_require_trade_markers():
    from ai.prompts import design_strategy_messages
    msgs = design_strategy_messages({"price": 100.0, "change_24h": 0.1}, [])
    sys = msgs[0]["content"]
    assert "plotshape" in sys, "AI pine_code 规则必须明确要求画买卖点标记"
    assert "strategy.position_size" in sys or "买入" in sys


# ---------------- D. API 通道：真实 Pine 必须能到达前端 ----------------

def _register_pine_specs():
    import strategies
    strategies.register_dynamic("iter_real_pine", {
        "title": "有真实代码", "executor": "price_action",
        "params": {"volume_confirm": 1.5}, "created_by": "ai_iteration",
        "pine_code": build_price_action_pine(
            {"name": "iter_real_pine", "params": {"volume_confirm": 1.5}}),
        "pine_note": "",
    })
    strategies.register_dynamic("rl_no_pine", {
        "title": "不可导出", "executor": "rl_adaptive", "params": {},
        "created_by": "evolve_engine", "pine_code": "",
        "pine_note": "模型无法导出 Pine：网络超出图表脚本体积上限",
    })


@pytest.mark.asyncio
async def test_repo_all_exposes_pine_availability(monkeypatch):
    """/all 曾把 pine_code / pine_note 全部丢掉 → 前端无从判断，只能无条件套
    price_action 模板，于是进化策略显示成假代码。列表只带「有没有真实代码 + 原因」，
    代码本体按需单独取（一份 DRL 脚本约 44KB，几十份内嵌会撑爆列表响应）。"""
    from web.api.strategy_repo import repo_all
    _register_pine_specs()

    async def _noop(db):
        return 0
    monkeypatch.setattr("strategies.dynamic_store.restore_from_db", _noop)
    rows = {r["name"]: r for r in (await repo_all(db=None))["strategies"]}

    assert rows["iter_real_pine"]["has_pine"] is True
    assert "pine_code" not in rows["iter_real_pine"], "列表不内嵌代码本体"
    assert rows["rl_no_pine"]["has_pine"] is False
    assert "体积上限" in rows["rl_no_pine"]["pine_note"]


@pytest.mark.asyncio
async def test_repo_pine_endpoint_returns_real_code(monkeypatch):
    from fastapi import HTTPException
    from web.api.strategy_repo import repo_pine
    _register_pine_specs()

    async def _noop(db):
        return 0
    monkeypatch.setattr("strategies.dynamic_store.restore_from_db", _noop)

    out = await repo_pine("iter_real_pine")
    buy, sell = _markers(out["pine_code"])
    assert buy and sell, "按策略名取到的必须是带买卖点的真实代码"
    out2 = await repo_pine("rl_no_pine")
    assert out2["pine_code"] == "" and out2["pine_note"], "不可导出时给显式原因，不套别的模板"
    with pytest.raises(HTTPException):
        await repo_pine("no_such_strategy_xyz")


# ---------------- E. 读取侧兜底：历史老记录也要有买卖点 ----------------

def test_legacy_pine_code_gets_markers_on_read():
    """标记规则是后加的：库里老记录存的 pine_code 零标记（实测 simple_sr_breakout
    1472 字节无任何 plotshape）。用户不该为了看到买卖点重新设计一遍，
    所以补齐放在读取侧，历史记录立刻生效。"""
    legacy = ("//@version=5\nstrategy(\"legacy\", overlay=true)\n"
              "if longCond\n    strategy.entry(\"L\", strategy.long)\n")
    code, note = resolve_pine({"name": "legacy", "executor": "price_action",
                               "params": {}, "pine_code": legacy})
    assert note == ""
    assert code.startswith(legacy)
    buy, sell = _markers(code)
    assert buy and sell


def test_resolve_pine_falls_back_to_executor_template():
    """AI 设计当时未落库 pine_code 的 price_action 策略：按 spec 参数生成，
    生成的代码同样必须带买卖点。

    夹具故意带 sr_window —— 库里历史记录（simple_sr_breakout 等）的 params 仍存着
    这个已废弃的键，生成 Pine 时必须忽略它，不能把死参数暴露成图表可调项。
    """
    code, note = resolve_pine({"name": "sr_only", "executor": "price_action",
                               "params": {"sr_window": 9}})
    assert note == "" and "sr_window" not in code
    buy, sell = _markers(code)
    assert buy and sell


def test_resolve_pine_refuses_borrowed_template_for_other_executors():
    """rl_adaptive / dual_ma 等无等价模板的执行器：只能返回原因，
    返回 price_action 代码等于给图上画一堆与真实逻辑无关的假买卖点。"""
    code, note = resolve_pine({
        "name": "rl_evolve", "executor": "rl_adaptive", "params": {},
        "pine_note": "模型装不进 Pine",
    })
    assert code == "" and note == "模型装不进 Pine"
    # 调用方（repo_pine）为内置策略显式带上 executor；spec 缺 executor 时按
    # price_action 处理——历史 AI 设计记录都是该执行器，名字却是自定义的
    code2, note2 = resolve_pine({"name": "dual_ma", "executor": "dual_ma", "params": {}})
    assert code2 == "" and "dual_ma" in note2


# ---------------- F. /api/ai/strategies 通道（前端「最新设计/最近迭代」数据源） ----------------

@pytest.mark.asyncio
async def test_ai_strategies_endpoint_always_returns_code_or_reason(tmp_path):
    """这个端点曾只做 ensure_trade_markers：库里没存 pine_code 的记录（进化注册、
    导入失败）返回空字符串，前端连「为什么没有代码」都拿不到，只能显示空白面板。
    统一走 resolve_pine 后三条分支都要落地：补标记 / 按执行器生成 / 给出原因。"""
    from core.database import AiStrategy, Database
    from web.api.ai import list_ai_strategies

    legacy = ('//@version=5\nstrategy("old", overlay=true)\n'
              'if c\n    strategy.entry("L", strategy.long)\n')
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 't.db'}")
    await db.init()
    async with db.session() as s:
        s.add(AiStrategy(name="legacy_no_marker", spec_json=json.dumps(
            {"name": "legacy_no_marker", "executor": "price_action", "params": {},
             "pine_code": legacy})))
        s.add(AiStrategy(name="iter_no_pine_code", spec_json=json.dumps(
            {"name": "iter_no_pine_code", "executor": "price_action",
             "params": {"breakout_pct": 0.006}})))
        s.add(AiStrategy(name="meta_controller", spec_json=json.dumps(
            {"name": "meta_controller", "executor": "rl_adaptive", "params": {},
             "pine_note": "模型装不进 Pine"})))
        await s.commit()

    rows = {r["name"]: r["spec"] for r in await list_ai_strategies(db)}
    await db.close()

    buy, sell = _markers(rows["legacy_no_marker"]["pine_code"])
    assert buy and sell, "老记录读取时补齐买卖点"
    buy, sell = _markers(rows["iter_no_pine_code"]["pine_code"])
    assert buy and sell, "没存代码但执行器有等价模板：必须生成而不是留空"
    assert rows["meta_controller"]["pine_code"] == ""
    assert rows["meta_controller"]["pine_note"] == "模型装不进 Pine", \
        "无模板执行器：空代码必须配显式原因，前端才能给出解释而不是空白"
