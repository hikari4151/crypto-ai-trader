"""提示词模板（推倒重做版）：市场解读 / 参数优化 / AI 策略设计 / 策略迭代 / 交易复盘。

每套提示词统一包含五大要件：
1. **角色 + 能力要求**：明确专家身份与必须掌握的方法
2. **反幻觉规则**：只允许使用输入数据，禁止编造价格/指标数字；每条结论必须可追溯到输入
3. **量化规则预告**：预告与 ai.validator 一致的输出规则（方向一致性/范围/逻辑），让 AI 一次通过
4. **价格行为知识**：优先用价格行为（AL Brooks）框架，其他策略次之
5. **反思正循环**：把历史表现/上代结果反馈注入，让每代改进有据可依（正向迭代）

配合 ai/client.py 的按功能参数预设（温度/token）与 chat_json_validated 校验重试机制。
"""
from datetime import datetime
from typing import Any, Optional

# ============ 输出纪律（所有任务共用） ============
_OUTPUT_RULE = (
    "【输出纪律】\n"
    "1. 只输出一个合法 JSON 对象，不要 Markdown 代码块、不要前后缀解释。\n"
    "2. 字段名严格按下述 schema，不要增删改名。\n"
    "3. 数值类型严格符合：字符串用引号、数字不用引号、布尔用 true/false、数组用 []。\n"
)

# ============ 反幻觉规则（所有任务共用） ============
_NO_HALLUCINATION = (
    "【反幻觉规则（必须遵守）】\n"
    "1. 只能使用输入中提供的数据（价格/指标/成交/回测）。禁止编造输入里不存在的数字。\n"
    "2. 每条结论必须能追溯到输入中的具体证据；没有证据就降低置信度或明确说不确定。\n"
    "3. 不要假设输入未提供的背景信息（如未提供的宏观事件、未显示的K线）。\n"
    "4. 输出中的任何价格位（如触发价）必须与输入快照中的价格处于同一量级。\n"
    "5. 宁可明确说「证据不足」，也不要编造一个听起来合理的结论。\n"
)

# ============ 数据块 ============
def _snapshot_block(snap: dict) -> str:
    ind = snap.get("indicators", {})
    candles = snap.get("candles", [])[-30:]
    tf = snap.get("timeframe", "")
    lines = [
        f"- 周期: {tf or '未知'}（每根K线={tf}，时间窗口 {len(candles)} 根）",
        f"- 最新收盘价: {ind.get('close')}",
        f"- MA10/MA30: {ind.get('ma_fast')} / {ind.get('ma_slow')}",
        f"- MACD(DIF/DEA/HIST): {ind.get('macd')} / {ind.get('macd_signal')} / {ind.get('macd_hist')}",
        f"- RSI(14): {ind.get('rsi')}",
        f"- 布林带 上/中/下: {ind.get('bb_upper')} / {ind.get('bb_mid')} / {ind.get('bb_lower')}",
        f"- %B位置: {ind.get('bb_position')}  带宽: {ind.get('bb_width')}%",
        f"- 波动率 ATR%: {ind.get('atr_pct')}  均线斜率(MA10/ATR): {ind.get('ma_slope')}",
        f"- 最新一根成交量: {ind.get('volume')}  量比: {ind.get('vol_ratio')}",
        f"- 最近 {len(candles)} 根K线(OHLCV):",
    ]
    for c in candles[-10:]:
        # K线时间戳转 ISO（原始毫秒对 LLM 不友好，曾直接打印毫秒）
        try:
            ts = datetime.fromtimestamp(c[0] / 1000).strftime("%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            ts = str(c[0])
        lines.append(f"  [{ts}] O={c[1]} H={c[2]} L={c[3]} C={c[4]} V={c[5]}")
    return "\n".join(lines)


def _sr_pa_block(snap: dict) -> str:
    ind = snap.get("indicators", {})
    sr, pa = ind.get("sr", {}), ind.get("pa", {})
    return (f"- 支撑位: {sr.get('support')}  阻力位: {sr.get('resistance')}\n"
            f"- 距支撑: {sr.get('distance_to_support')}  距阻力: {sr.get('distance_to_resistance')}\n"
            f"- 近期支撑带: {sr.get('support_levels')}  近期阻力带: {sr.get('resistance_levels')}\n"
            f"- 最新K线: {pa.get('last_candle')}, 实体比{pa.get('body_ratio')}, "
            f"上影{pa.get('upper_wick')}, 下影{pa.get('lower_wick')}\n"
            f"- 量能比: {pa.get('volume_ratio')}x, 连续方向: {pa.get('streak')}, "
            f"突破新高: {pa.get('breakout_high')}, 突破新低: {pa.get('breakout_low')}, 波幅: {pa.get('range_pct')}%")


def _backtests_block(backtests: list) -> str:
    """把历史回测结果整理成 AI 可读的参考文本（反思正循环：反馈真实表现）。"""
    if not backtests:
        return ""
    lines = ["【历史回测参考（真实跑过的策略表现，供反思迭代用）】"]
    for b in backtests:
        lines.append(
            f"- 策略[{b.get('strategy')}] {b.get('symbol')} {b.get('timeframe')} "
            f"总收益{b.get('total_return')} 年化{b.get('annual_return')} "
            f"回撤{b.get('max_drawdown')} 夏普{b.get('sharpe')} "
            f"胜率{b.get('win_rate')} 盈亏比{b.get('profit_factor')} 交易{b.get('total_trades')}笔"
        )
    return "\n".join(lines) + "\n"


# ============ 1. 市场解读（温度 0.2 / token 4096） ============
def _program_direction_block(program: dict) -> str:
    """程序方向复算结果 → 提示词引导块。"""
    if not program or not program.get("enough_data"):
        return ""
    d = {"long": "看多", "short": "看空", "neutral": "中性"}.get(program["direction"], "中性")
    lines = [f"【程序方向参考（K线结构复算）】方向 {d}，评分 {program['score']:+d}，"
             f"置信度 {program['confidence']:.2f}"]
    for s in program.get("signals", [])[:5]:
        lines.append(f"- {s['name']}: {'多' if s['sign']>0 else ('空' if s['sign']<0 else '平')}（{s['reason']}）")
    lines.append("（若你的 bias 与此矛盾，必须在 summary 中用明确证据说明反转理由）")
    return "\n".join(lines)


def _continuity_block(cont: dict) -> str:
    """决策连续性上下文 → 提示词引导块。

    cont 可能来自：
      - 提示词阶段 hint：{has_prev, prev_bias, prev_confidence, prev_regime}
      - 输出后 continuity：{flipped, flip_conflict, note, ...}
    只做信息性引导（方向一致优先），硬性约束由校验器在输出后强制。
    """
    if not cont or not cont.get("has_prev"):
        return ""
    if cont.get("flip_conflict"):
        return f"⚠️【决策连续性】{cont['note']}"
    if cont.get("flipped"):
        return f"【决策连续性】{cont['note']}"
    prev_bias = cont.get("prev_bias")
    if prev_bias in ("long", "short"):
        d = "看多" if prev_bias == "long" else "看空"
        prev_conf = cont.get("prev_confidence")
        conf_txt = f"（置信度 {prev_conf:.0%}）" if isinstance(prev_conf, (int, float)) else ""
        return (f"【决策连续性】上一轮分析判断 {d}{conf_txt}。若本轮方向与之相反，"
                f"必须用明确的结构反转证据说明，否则程序会强制压低置信度。")
    return ""


def market_analysis_messages(snap: dict, position: float, recent_trades: list[dict],
                             program: Optional[dict] = None,
                             continuity: Optional[dict] = None) -> list[dict]:
    from .price_action_kb import add_price_action_knowledge
    sys = (
        "你是资深加密货币量化分析师，精通价格行为（Price Action）趋势/震荡/拐点识别，擅长多空证据权衡。\n"
        "你的输出会被自动校验：方向与置信度必须匹配、信号与方向必须一致、结论必须可追溯到输入。\n\n"
        "【思考框架（内部推理，不必输出）】\n"
        "1. 观察：价格 vs 均线、MACD 柱/金叉死叉、RSI 区间、布林带收口/张口、量能、支撑阻力距离\n"
        "2. 归类：用价格行为判断当前是趋势/震荡/拐点（HH/HL、趋势线突破、高潮、关键位测试）\n"
        "3. 推演：给出 2-3 种情景与触发条件（突破/跌破/盘整），每种都给具体价位\n"
        "4. 校准：置信度必须匹配证据强度；证据不足给 0.3-0.5 并明确说不确定\n\n"
        "【输出 schema】\n"
        '{"regime": "趋势|震荡|拐点|未知", "bias": "long|short|neutral",\n'
        ' "confidence": 0.0-1.0,\n'
        ' "summary": "不超过120字中文解读，含关键证据链",\n'
        ' "warnings": ["风险点，每条一句话"],\n'
        ' "signals": ["可执行建议，含具体触发价"],\n'
        ' "trade_plan": 可选结构化方案（决定给出具体入场/止损/止盈时才填）{\n'
        '   "side": "long|short", "entry": 入场价, "stop_loss": 止损价,\n'
        '   "take_profit": 止盈价, "win_rate": 0.0-1.0 预估胜率}\n'
        '  —— trade_plan 给定时：side 必须与 bias 一致；做多要求 stop<entry<tp，做空要求 stop>entry>tp；\n'
        '     RR=(|tp-entry|/|entry-stop|)>=1.0 且 win_rate×reward > (1-win_rate)×risk，否则方案会被程序拒绝\n'
        '  —— 没有把握给出三价时省略 trade_plan，不要硬填}\n'
        ' "bull_strength": 可选 0-100 多头力量评分, "bear_strength": 可选 0-100 空头力量评分\n'
        ' "risk_level": 可选 "low|medium|high" 风险总评（与 warnings 并存）\n'
        ' "outlook_24h": 可选 24小时展望 {"scenario": "情景描述", "probability": 0-1, "trigger": "触发条件"}\n'
        '  —— bull_strength/bear_strength/risk_level/outlook_24h 均为可选字段，有依据才填，不硬凑}\n'
        "【量化规则（不满足会被拒绝）】\n"
        "- bias=long/short 时 confidence 必须 >=0.6\n"
        "- signals 方向必须与 bias 一致（bias=long 不允许建议做空）\n"
        "- summary 必须引用价格行为证据（HH/HL、突破成败、高潮、关键位测试）\n"
        "- 触发价必须与快照价格同一量级（反幻觉）\n"
        "- trade_plan 存在时须通过交易者方程校验（见上）\n\n"
        + _NO_HALLUCINATION + "\n"
        + _OUTPUT_RULE
    )
    sys = add_price_action_knowledge(sys, compact=True)
    blocks = [f"【市场快照】\n{_snapshot_block(snap)}",
              f"【关键位与价格行为】\n{_sr_pa_block(snap)}"]
    pblock = _program_direction_block(program)
    if pblock:
        blocks.append(pblock)
    cblock = _continuity_block(continuity)
    if cblock:
        blocks.append(cblock)
    blocks.append(f"【当前持仓数量】{position}")
    blocks.append(f"【近期交易】\n{recent_trades if recent_trades else '暂无'}")
    user = "\n\n".join(blocks) + "\n\n请先深度思考再输出 JSON。"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============ 2. 参数优化（温度 0.2 / token 4096） ============
def optimize_params_messages(strategy_name: str, schema: dict, current_params: dict,
                             performance: dict, snap: dict) -> list[dict]:
    sys = (
        "你是量化策略参数优化器，只做有证据的微调，反对无依据乱改。\n"
        "你的输出会被自动校验：参数必须落在 schema 内、不允许改 schema 外的参数。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 诊断：近期表现问题（胜率/盈亏比/回撤/交易频率）映射到哪个参数环节\n"
        "2. 对照：当前市场 regime 与该策略适配度（用价格行为判断）\n"
        "3. 假设：为什么改这个参数、改多少，给出一句话逻辑\n"
        "4. 权衡：避免过拟合，一次只调整 1-3 个最有把握的参数；没把握就不改\n\n"
        "【输出 schema】\n"
        '{"params": {参数名: 新值}, "reason": "中文理由(120字内，说明证据与预期效果)"}\n'
        f"参数 schema（type/min/max 约束）: {schema}\n"
        "【量化规则】\n"
        "- 只允许改 schema 内存在的参数，新值必须落在范围内\n"
        "- params 不能为空（若判断不需要改，给出最小的合理调整并说明理由）\n"
        "- reason 必须引用近期表现或市场数据，禁止无依据修改\n\n"
        + _NO_HALLUCINATION + "\n"
        + _OUTPUT_RULE
    )
    user = (f"策略: {strategy_name}\n"
            f"当前参数: {current_params}\n"
            f"近期表现: {performance}\n"
            f"市场快照: {_snapshot_block(snap)}\n"
            f"关键位与价格行为: {_sr_pa_block(snap)}\n\n"
            "请先深度思考再输出 JSON。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============ 3. 价格行为全自动优化（温度 0.2 / token 4096） ============
def price_action_optimize_messages(strategy, performance: dict, snap: dict, focus: str,
                                   backtests: Optional[list] = None) -> list[dict]:
    from .price_action_kb import add_price_action_knowledge
    sys = (
        "你是量化策略参数优化器，专精价格行为学与关键位(S/R)分析。\n"
        "你的输出会被自动校验：参数必须落在 schema 内、止盈必须大于止损、超买必须大于超卖。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 评估当前价格行为：突破还是回调、量能是否确认、假突破风险、上下影线含义\n"
        "2. 诊断错配：哪个参数最影响近期表现（关键位识别/突破确认/RSI过滤/止损止盈）\n"
        "3. 假设验证：调整方向、幅度、与 S/R 距离的关系\n"
        "4. 风险优先：止损止盈必须与关键位距离匹配，先控风险再谈收益\n\n"
        "【输出 schema】\n"
        '{"params": {参数名: 新值}, "reason": "中文理由(150字内，含证据链)", "focus": "优化重点中文说明"}\n'
        f"参数 schema: {strategy.param_schema}\n"
        "【量化规则】\n"
        "- 只改 schema 内的参数，新值必须在范围内；params 不能为空\n"
        "- take_profit_pct 必须 > stop_loss_pct；rsi_ob 必须 > rsi_os\n"
        "- 回测若显示某参数组合失效，必须优先调整而不是维持\n\n"
        + _NO_HALLUCINATION + "\n"
        + _OUTPUT_RULE
    )
    sys = add_price_action_knowledge(sys, compact=True)
    user = (f"策略: {strategy.name}\n"
            f"当前参数: {strategy.params}\n"
            f"近期表现: {performance}\n"
            f"{_backtests_block(backtests or [])}"
            f"本次优化重点: {focus}\n"
            f"市场快照: {_snapshot_block(snap)}\n"
            f"关键位与价格行为: {_sr_pa_block(snap)}\n\n"
            "请先深度思考再输出 JSON。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============ 4. AI 策略设计（温度 0.7 / token 8192） ============
_STRATEGY_TYPE_GUIDE = {
    "trend": "趋势跟踪：顺势做多做空，用均线/趋势线/动量过滤，捕捉大趋势。",
    "breakout": "突破交易：识别关键位（S/R/通道/布林带），放量突破后顺势入场。",
    "mean_reversion": "均值回归：超买超卖后回归（RSI/乖离/布林带位置），逆势接飞刀需严格风控。",
    "grid": "网格震荡：区间低吸高抛，适合横盘。",
    "custom": "自定义要求：严格按用户给出的具体要求设计。",
}


def design_strategy_messages(snap: dict, recent_trades: list[dict],
                             backtests: Optional[list] = None,
                             strategy_type: str = "",
                             custom_requirement: str = "") -> list[dict]:
    from .price_action_kb import add_price_action_knowledge
    sys = (
        "你是顶级量化策略架构师，优先使用价格行为（Price Action）框架设计策略，"
        "参数必须可执行、风控必须完整。\n"
        "你的输出会被自动校验：参数必须落在 schema 内、止盈>止损、超买>超卖、名称合法。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 市场判定：当前是突破市/回调市/震荡市？最适合哪种入场模式\n"
        "2. 关键位：如何识别支撑/阻力，距离、被触及次数多少才有效\n"
        "3. 入场确认：突破/回调的价量确认标准（幅度、量能倍数、RSI 过滤）\n"
        "4. 出场与风控：止损止盈与关键位的关系、仓位控制\n"
        "5. 逻辑优先价格行为：S/R 突破/回调、信号棒确认、趋势过滤；其他策略思路须明确说明\n"
        "6. 用户指定了策略类型或自定义要求时，必须优先满足用户要求\n\n"
        "【输出 schema】\n"
        '{"name": "英文小写下划线策略名(如 sr_breakout)", "title": "中文标题(<=15字)",\n'
        ' "description": "中文描述(60字内)", "logic": "入场出场逻辑中文说明",\n'
        ' "params": {参数名: 值}, "risk_tips": ["中文风控提示数组"]}\n'
        "params 只能使用以下键(值必须落在标注范围内)：\n"
        "mode(\"breakout\"或\"pullback\"), sr_window(int 5-60), min_touches(int 1-8), "
        "breakout_pct(float 0.0001-0.02), volume_confirm(float 0.5-3.0), "
        "rsi_ob(float 60-90), rsi_os(float 10-40), stop_loss_pct(float 0.001-0.1), "
        "take_profit_pct(float 0.001-0.3), use_sr_stop(bool), size_pct(float 0.05-1.0)。\n"
        "【量化规则】\n"
        "- take_profit_pct 必须 > stop_loss_pct；rsi_ob 必须 > rsi_os\n"
        "- logic 必须引用输入中的 S/R/量能 等真实特征，禁止凭空编造\n"
        "- 优先设计价格行为策略；若采用其他思路须在 logic 里写明\n\n"
        + _NO_HALLUCINATION
    )
    sys = add_price_action_knowledge(sys, compact=True) + "\n" + _OUTPUT_RULE

    # 用户要求块：策略类型 + 自定义要求
    req_lines = []
    if strategy_type and strategy_type in _STRATEGY_TYPE_GUIDE:
        req_lines.append(f"【用户指定策略类型】{_STRATEGY_TYPE_GUIDE[strategy_type]}")
    if custom_requirement and custom_requirement.strip():
        req_lines.append(f"【用户自定义要求（必须严格执行）】{custom_requirement.strip()}")
    req_block = "\n".join(req_lines) + "\n\n" if req_lines else ""

    user = (f"{req_block}【当前市场快照】\n{_snapshot_block(snap)}\n\n"
            f"【关键位与价格行为】\n{_sr_pa_block(snap)}\n\n"
            f"{_backtests_block(backtests or [])}"
            f"【近期交易】\n{recent_trades if recent_trades else '暂无'}\n\n"
            "请先深度思考再设计策略并输出 JSON。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============ 5. AI 策略迭代（温度 0.6 / token 8192） ============
def _previous_block(previous: Optional[list]) -> str:
    """前代迭代记录块：让 AI 看到历史迭代的 critique/改进点与回测指标，形成正循环。"""
    if not previous:
        return ""
    lines = ["【前代迭代记录（上一代说了什么/改了什么，请先对照其回测指标检验效果）】"]
    for p in previous:
        bt = p.get("backtest") or {}
        bt_txt = ""
        if bt:
            new_m = bt.get("new") or {}
            old_m = bt.get("old") or {}
            bt_txt = (f"  注册回测: 收益 {new_m.get('total_return')} vs 旧 {old_m.get('total_return')}, "
                      f"回撤 {new_m.get('max_drawdown')}, 胜率 {new_m.get('win_rate')} "
                      f"（{'超越旧策略' if bt.get('improved') else '未超越'}）")
        lines.append(
            f"- 版本 {p.get('version', '?')} [{p.get('name', '')}]:\n"
            f"  批判: {p.get('critique', '')}\n"
            f"  改进点: {'；'.join(p.get('improvements', []) or [])}\n"
            f"  总结: {p.get('summary', '')}"
            f"{bt_txt}"
        )
    return "\n".join(lines) + "\n"


def iteration_messages(strategy, performance: dict, snap: dict,
                       backtests: Optional[list] = None,
                       previous: Optional[list] = None,
                       goal: str = "") -> list[dict]:
    from .price_action_kb import add_price_action_knowledge
    sys = (
        "你是顶级量化策略评审与迭代专家，对现有策略做批判性深度反思并给出改进版。\n"
        "你的输出会被自动校验：参数必须落在给定 schema 内、止盈>止损、超买>超卖、改进必须引用真实表现。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 复述：先复述原策略核心逻辑，确认理解正确\n"
        "2. 批判：基于近期表现与回测数据，找出最可能的失效点（入场模糊/止损不合理/过滤不足/与市场不匹配）\n"
        "3. 假设：提出 1-2 个最有价值的改进点，说明为什么能提升表现（证据驱动）\n"
        "4. 落地：将改进落实到具体参数调整，并明确新旧差异\n"
        "5. 价格行为：优先用价格行为视角改进（S/R 联动、信号棒确认、趋势过滤）\n\n"
        "【输出 schema】\n"
        '{"name": "必须严格沿用原策略名，不要改名", "title": "中文标题(<=15字)",\n'
        ' "description": "中文描述(60字内)", "logic": "改进后的入场出场逻辑说明",\n'
        ' "params": {参数名: 新值}, "risk_tips": ["中文风控提示数组"],\n'
        ' "critique": "对原策略的批判性分析(150字内)", "improvements": ["改进点数组"],\n'
        ' "summary": "本次迭代总结(80字内)"}\n'
        "【量化规则】\n"
        "- 只允许改下方给定 schema 内存在的参数，新值必须落在范围内；params 不能为空\n"
        "- take_profit_pct 必须 > stop_loss_pct；rsi_ob 必须 > rsi_os（若 schema 含这些键）\n"
        "- critique 必须诚实指出原策略真正的问题（基于数据），不要空泛\n"
        "- improvements 必须与 critique 对应\n"
        "- 若历史回测显示原策略样本外失效，改进必须针对该问题\n"
        "- 若提供【前代迭代记录】，必须检查上代改进在回测中是否真的生效：\n"
        "  上代改进有效→保持并深化；上代改进无效或回撤→本次必须调整该方向，禁止重复同样错误\n\n"
        + _NO_HALLUCINATION
    )
    sys = add_price_action_knowledge(sys, compact=True) + "\n" + _OUTPUT_RULE
    goal_txt = f"\n【本次迭代目标（用户指定，请优先围绕它改进）】{goal}\n" if goal else ""
    user = (f"【当前策略】名称: {strategy.name}\n"
            f"描述: {strategy.description}\n"
            f"参数schema: {strategy.param_schema}\n"
            f"当前参数: {strategy.params}\n"
            f"【近期表现】{performance}\n"
            f"{goal_txt}"
            f"{_previous_block(previous)}"
            f"{_backtests_block(backtests or [])}"
            f"【市场快照】{_snapshot_block(snap)}\n"
            f"【关键位与价格行为】{_sr_pa_block(snap)}\n"
            "请深度思考，批判性评估该策略并给出改进版本，输出严格 JSON。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============ 6. 交易复盘（温度 0.2 / token 4096） ============
def review_messages(trades: list[dict], summary: dict) -> list[dict]:
    sys = (
        "你是量化交易复盘教练，帮助交易者从实盘中提炼可复用的改进。\n"
        "你的评分会被自动校验：亏损且胜率低时不能打高分（禁止谄媚）。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 量化评估：胜率/盈亏比/交易频率，评估整体质量（score 依据必须与实际数据一致）\n"
        "2. 归因：盈利单和亏损单分别有什么共同特征（入场时机/持仓时间/止损执行/是否逆势）\n"
        "3. 问题定位：最大亏损来源是策略问题还是执行问题\n"
        "4. 改进：给出 2-4 条可落地行动项，每条具体到行为\n\n"
        "【输出 schema】\n"
        '{"score": 0-100, "strengths": ["做得好的数组"], "problems": ["问题数组"],\n'
        ' "action_items": ["具体可执行改进，每条一句话"], "summary": "中文总结120字内"}\n'
        "【量化规则】\n"
        "- score 必须与实际胜率/盈亏比/回撤匹配：亏损+低胜率时 score 不得超过 60\n"
        "- action_items 必须具体（什么条件下做什么），禁止空话\n"
        "- problems 必须引用实际交易中的具体特征（如\"连续亏损单都发生在回调入场\"）\n\n"
        + _NO_HALLUCINATION + "\n"
        + _OUTPUT_RULE
    )
    user = (f"交易总数: {summary.get('total_trades', 0)}\n"
            f"胜率: {summary.get('win_rate')}\n"
            f"总盈亏: {summary.get('total_pnl')}\n"
            f"交易记录(最多100条): {trades[:100]}\n\n"
            "请先深度思考再输出 JSON。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
