"""提示词共用块：输出纪律、反幻觉规则、市场快照/关键位/历史回测 数据块。"""

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
    lines = [
        f"- 最新收盘价: {ind.get('close')}",
        f"- MA10/MA30: {ind.get('ma_fast')} / {ind.get('ma_slow')}",
        f"- MACD(DIF/DEA/HIST): {ind.get('macd')} / {ind.get('macd_signal')} / {ind.get('macd_hist')}",
        f"- RSI(14): {ind.get('rsi')}",
        f"- 布林带 上/中/下: {ind.get('bb_upper')} / {ind.get('bb_mid')} / {ind.get('bb_lower')}",
        f"- 近5根成交量合计: {ind.get('volume')}",
        f"- 最近 {len(candles)} 根K线(OHLCV):",
    ]
    for c in candles[-10:]:
        lines.append(f"  [{c[0]}] O={c[1]} H={c[2]} L={c[3]} C={c[4]} V={c[5]}")
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
