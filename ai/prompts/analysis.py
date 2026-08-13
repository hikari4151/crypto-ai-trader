"""市场解读提示词（温度 0.2 / token 4096）。"""
from typing import Optional

from .common import _NO_HALLUCINATION, _OUTPUT_RULE, _snapshot_block, _sr_pa_block


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
