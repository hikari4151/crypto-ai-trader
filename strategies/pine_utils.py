"""Pine Script v5 参数解析与工具（供导入端点与 AI 策略设计共用）。

AI 设计策略产出完整 Pine Script v5 代码后，需把其中的 input.* 参数解析回来，
映射到 price_action 执行器的参数字典（与前端 buildPine 导出的格式互逆）。

本项目的关键要求：任何策略的 Pine 代码贴到 TradingView 图表上，必须在 K 线上
给出买入与卖出位置。各策略模板共用本模块的 TRADE_MARKERS_PINE / ensure_trade_markers
落地这条要求（AI 生成与 TradingView 导入的代码用后者兜底补标记）。
"""
from __future__ import annotations

import re

from indicators.technical import SR_MIN_TOUCHES, SR_WINDOW


# 图表成交标记块：按真实持仓变化（而非策略信号变量）画买卖点，
# 因此覆盖止损/止盈/关键位离场/分批调仓等全部成交路径。
# 前端 index.html buildPine 有同结构的镜像实现，改动需两处同步。
TRADE_MARKERS_PINE = """// ───── 成交标记：图上给出买入/卖出位置 ─────
// 按真实持仓变化标记：建仓/加仓 → ▲买入；减仓/平仓 → ▼卖出
aiPosPrev = nz(strategy.position_size[1])
aiTradeBuy = strategy.position_size > aiPosPrev
aiTradeSell = strategy.position_size < aiPosPrev
plotshape(aiTradeBuy, title="买入", style=shape.triangleup, location=location.belowbar, color=color.new(#30d158, 0), size=size.small, text="买入", textcolor=color.new(#30d158, 0))
plotshape(aiTradeSell, title="卖出", style=shape.triangledown, location=location.abovebar, color=color.new(#ff453a, 0), size=size.small, text="卖出", textcolor=color.new(#ff453a, 0))"""


def has_trade_markers(code: str) -> bool:
    """Pine 代码是否已同时标出买入与卖出（两个方向都要有才算满足要求）。"""
    buy = "triangleup" in code or "arrowup" in code
    sell = "triangledown" in code or "arrowdown" in code
    return buy and sell


def ensure_trade_markers(code: str) -> str:
    """缺买卖点标记时追加通用标记块（幂等；已有双向标记的代码原样返回）。

    标记块读的是 strategy.position_size——只有 strategy() 型脚本才有持仓变量，
    给 indicator() 型代码补一段会让整份脚本编译不过，这类脚本原样返回。
    """
    if not code or not code.strip() or has_trade_markers(code):
        return code
    if not re.search(r"\bstrategy\s*\(", code):
        return code
    return code.rstrip("\n") + "\n\n" + TRADE_MARKERS_PINE + "\n"


# 有等价 Pine 模板的执行器：只有这些能按 spec 忠实翻译成图表代码。
# 其余执行器（rl_adaptive / meta_controller / dual_ma / factor_signal / grid）
# 套 price_action 模板会得到一段与真实逻辑无关的假代码。
PINE_TEMPLATE_EXECUTORS = frozenset({"price_action"})


def resolve_pine(spec: dict) -> tuple[str, str]:
    """取某个策略 spec 的真实 Pine，返回 (代码, 不可用原因)。

    注册时已存的代码优先（进化/DRL 产物是从模型导出的），其次才按执行器模板翻译。
    拿不到代码一定给原因：前端据此显式提示，而不是静默换一套别的策略逻辑。
    """
    code = (spec.get("pine_code") or "").strip()
    if code:
        # 库里老记录可能存的是零标记代码：读取侧兜底补齐，买卖点要求立刻生效
        return ensure_trade_markers(code), ""
    executor = spec.get("executor") or "price_action"
    if executor in PINE_TEMPLATE_EXECUTORS:
        try:
            return build_price_action_pine(spec), ""
        except Exception as e:  # noqa: BLE001
            return "", f"按执行器模板生成 Pine 失败：{e}"
    return "", (spec.get("pine_note") or "").strip() or f"执行器 {executor} 无等价 Pine 模板"


# Pine 变量名 → price_action 参数（与前端 buildPine 生成的 input.* 一一对应）
# 不含摆动窗口/触碰次数/RSI 超卖：它们不是执行器参数（关键位口径由引擎常量固定，
# RSI 侧只用超买过滤），暴露成可调项会让"图表可调、本地执行器无效"两边分叉。
PINE_PARAM_TYPES = {
    "breakoutPct": ("breakout_pct", float), "volConfirm": ("volume_confirm", float),
    "rsiOB": ("rsi_ob", float),
    "slPct": ("stop_loss_pct", float), "tpPct": ("take_profit_pct", float),
    "useSRStop": ("use_sr_stop", bool), "sizePct": ("size_pct", float),
}


def parse_pine_params(code: str) -> dict:
    """解析 Pine input.*() 参数与 default_qty_value，映射为执行器参数。

    仅识别本软件导出格式的变量名；非本软件来源的 Pine 只提取同名变量，其余忽略。
    """
    params: dict = {}
    m = re.search(r'mode\s*=\s*input\.string\(\s*"([^"]+)"', code)
    if m and m.group(1) in ("breakout", "pullback"):
        params["mode"] = m.group(1)
    for m in re.finditer(r'(\w+)\s*=\s*input\.(int|float|bool)\(\s*(true|false|[-\d.]+)', code):
        var, _typ, raw = m.group(1), m.group(2), m.group(3)
        if var not in PINE_PARAM_TYPES:
            continue
        key, cast = PINE_PARAM_TYPES[var]
        if cast is bool:
            params[key] = raw == "true"
        else:
            try:
                params[key] = cast(raw)
            except ValueError:
                continue
    # 仓位：本软件导出为 default_qty_value=<size_pct*100>
    m = re.search(r'default_qty_value=([\d.]+)', code)
    if m:
        try:
            params["size_pct"] = round(float(m.group(1)) / 100.0, 4)
        except ValueError:
            pass
    return params


def pine_shorttitle(code: str) -> str:
    m = re.search(r'shorttitle\s*=\s*"([^"]+)"', code)
    return m.group(1) if m else ""


# price_action 执行器的 Pine 参数默认值（与 strategies/price_action.py 一致）
_PA_DEFAULTS = {
    "mode": "breakout", "breakout_pct": 0.001,
    "volume_confirm": 1.2, "rsi_ob": 72, "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04, "use_sr_stop": True, "size_pct": 0.5,
}


def _pn(v: float) -> str:
    return f"{v:.6g}"


def build_price_action_pine(spec: dict) -> str:
    """把 price_action 策略规格（name/title/desc/logic/params）渲染为 Pine Script v5。

    与前端 index.html 的 buildPine 模板同结构（参数→input.*，入场/离场与执行器一致），
    供 AI 迭代/导入等后端路径生成可运行 Pine 代码。
    """
    P = {**_PA_DEFAULTS, **(spec.get("params") or {})}
    L = []
    push = lambda s: L.append(s)
    push("//@version=5")
    push('// ════════════════════════════════════════════════════════════════')
    push(f'//  AI 策略 · {spec.get("name", "ai_strategy")}'
         + (f' — {spec.get("title", "")}' if spec.get("title") else ""))
    if spec.get("description"):
        push('//  说明: ' + str(spec["description"]).replace("\n", " "))
    if spec.get("logic"):
        push('//  逻辑: ' + str(spec["logic"]).replace("\n", " "))
    push('// ════════════════════════════════════════════════════════════════')
    push(f'strategy("AI · {str(spec.get("title") or spec.get("name", "ai"))}", '
         f'shorttitle="{str(spec.get("name", "ai"))}", overlay=true,')
    push('     initial_capital=10000,')
    push(f'     default_qty_type=strategy.percent_of_equity, default_qty_value={_pn(float(P["size_pct"]) * 100)},')
    push('     commission_type=strategy.commission.percent, commission_value=0.1)')
    push('')
    push('// ───── ① 参数（AI 优化值 · 图表设置中可微调） ─────')
    push(f'mode        = input.string("{P["mode"]}", "入场模式", options=["breakout", "pullback"])')
    push(f'breakoutPct = input.float({_pn(P["breakout_pct"])}, "突破确认幅度", step=0.0001)')
    push(f'volConfirm  = input.float({_pn(P["volume_confirm"])}, "放量确认倍数", step=0.1)')
    push(f'rsiOB       = input.float({_pn(P["rsi_ob"])}, "RSI 超买阈值", minval=60, maxval=90)')
    push(f'slPct       = input.float({_pn(P["stop_loss_pct"])}, "止损比例", step=0.001)')
    push(f'tpPct       = input.float({_pn(P["take_profit_pct"])}, "止盈比例", step=0.001)')
    push(f'useSRStop   = input.bool({"true" if P["use_sr_stop"] else "false"}, "止损参考关键位")')
    push('')
    push('// ───── ② 关键位：摆动高低点 + 触碰计数 ─────')
    push('// 口径与本地回测/实盘执行器同源（indicators.technical 引擎常量），故不作 input：')
    push('// 图表端改了不会同步到实盘，暴露成可调项只会造成两边分叉')
    push(f'srWindow   = {SR_WINDOW}')
    push(f'minTouches = {SR_MIN_TOUCHES}')
    push('ph = ta.pivothigh(high, srWindow, srWindow)')
    push('pl = ta.pivotlow(low, srWindow, srWindow)')
    push('var float resistance = na')
    push('var float support    = na')
    push('var int touchRes = 0')
    push('var int touchSup = 0')
    push('if not na(ph)')
    push('    resistance := ph')
    push('    touchRes := 0')
    push('if not na(pl)')
    push('    support := pl')
    push('    touchSup := 0')
    push('if not na(resistance) and math.abs(close - resistance) / resistance <= 0.003')
    push('    touchRes += 1')
    push('if not na(support) and math.abs(close - support) / support <= 0.003')
    push('    touchSup += 1')
    push('')
    push('// ───── ③ 指标 ─────')
    push('rsi      = ta.rsi(close, 14)')
    push('volMa    = ta.sma(volume, 20)')
    push('volRatio = volMa > 0 ? volume / volMa : 1.0')
    push('prevC    = close[1]')
    push('')
    push('// ───── ④ 入场信号（突破 / 回调，与执行器逐条对应） ─────')
    push('justBroke = not na(resistance) and prevC <= resistance and close > resistance')
    push('     and close >= resistance * (1 + breakoutPct)')
    push('longBreak = mode == "breakout" and justBroke and touchRes >= minTouches')
    push('     and volRatio >= volConfirm and rsi < rsiOB')
    push('atSupport = not na(support) and close >= support and close <= support * 1.01')
    push('longPull  = mode == "pullback" and atSupport and volRatio < volConfirm')
    push('')
    push('warmup    = bar_index > 30')
    push('longCond  = (longBreak or longPull) and warmup and strategy.position_size == 0')
    push('')
    push('// ───── ⑤ 持仓管理：止损/止盈（参考关键位，与执行器一致） ─────')
    push('avg     = strategy.position_avg_price')
    push('srDist  = not na(support) and avg > support ? (avg - support) / avg : na')
    push('stopPct = useSRStop and not na(srDist) ? math.min(slPct, math.max(0.001, srDist * 0.5)) : slPct')
    push('if strategy.position_size > 0')
    push('    strategy.exit("XL", from_entry="L", stop=avg * (1 - stopPct), limit=avg * (1 + tpPct))')
    push('    if useSRStop and not na(resistance) and close >= resistance')
    push('        strategy.close("L", comment="触及阻力位获利了结")')
    push('')
    push('if longCond')
    push('    strategy.entry("L", strategy.long, comment=longBreak ? "突破阻力" : "回调支撑企稳")')
    push('')
    push('// ───── ⑥ 可视化：关键位 + 信号 ─────')
    push('plot(resistance, title="阻力位", color=color.new(#ff453a, 25), linewidth=2, style=plot.style_linebr)')
    push('plot(support, title="支撑位", color=color.new(#30d158, 25), linewidth=2, style=plot.style_linebr)')
    push('plot(ta.sma(close, 20), title="MA20", color=color.new(#98989f, 40))')
    push(TRADE_MARKERS_PINE)
    return "\n".join(L)