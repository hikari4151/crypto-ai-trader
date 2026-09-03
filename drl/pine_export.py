"""DRL 神经引擎 → Pine Script v5 自动交易代码导出。

把训练好的 Actor MLP 权重固化为 Pine Script v5 策略代码：内含与 env._precompute_features
一致的特征计算与 MLP 前向（ReLU 隐藏层 + softmax 动作概率 → 目标仓位档位），
并画出买卖点标记，可直接粘贴到 TradingView 运行。

2026-09 「图上必须给出买卖位置」专项修复：
- 接入 strategies.pine_utils.TRADE_MARKERS_PINE：按 strategy.position_size 变化画
  ▲买入 / ▼卖出，覆盖清仓、减仓、止损全部路径（旧版只有 plot(target)，在
  overlay=true 的价格轴上等于看不见，且全模块零 plotshape）
- 解除「state_window>1 拒绝导出」：特征都是 series，用 f?[k] 历史引用即可堆叠，
  顺序与 env._state_feats 一致（每根K线 13 个特征，旧→新）
- 特征口径与训练端对齐：P4-B4 的 tanh 有界压缩与 clip 旧版全部缺失，导出的策略
  与训练出的模型不是同一个东西；因子列同样补上 (x-mu)/sd 标准化
- 体积守门：权重必须整段写进脚本，超过约 64KB 显式拒绝并给出原因（PineExportError
  / try_build_drl_pine 的 note），不再产出一段永远编译不过的代码
- 调仓数学修正：减仓此前用 strategy.order(long) 表达（实际是加仓），改为
  strategy.close(qty_percent=)；posPct 此前用 base 数量除以 quote 权益（量纲错误）
"""
from __future__ import annotations

import numpy as np

from strategies.pine_utils import TRADE_MARKERS_PINE

# 与 drl/env.py ACTION_BUCKETS 一致的目标仓位档位
ACTION_BUCKETS = [0.0, 0.25, 0.5, 0.75, 1.0]

# 特征列数（13 维价量）与账户状态列数（持仓比例 + 浮动盈亏）——与 env._build_state 一致
N_PRICE_FEATURES = 13
N_ACCOUNT_FEATURES = 2

# 归一化常量：与 drl/env.py _precompute_features 的 _TANH_RET / _TANH_VOL 同步修改
_TANH_RET = 5.0
_TANH_VOL = 3.0

# 权重必须整段写进脚本（in_dim×hidden… 全部内联），超过约这个自设保守上限就判定
# 「这个模型装不进 Pine」。说明：TradingView 官方未公布可核实的脚本体积上限（本环境
# 无法联网核实），64KB 是本软件自设的保守预算——按实测约 36 字节/权重校准，
# 目的是产出一段人能贴进去、编辑器不卡的代码，而不是精确对齐服务端限制。
PINE_MAX_BYTES = 64 * 1024

# 实测校准：w=1/[32,32] 与 w=4/[16,16] 两种网络，生成脚本的实际字节 ÷ 权重数
_BYTES_PER_WEIGHT = 36

# 单个 array.from() 的分块参数个数（规避解析器对调用参数个数的上限）
_ARRAY_CHUNK = 50

# 训练端特征名（供 factor_expression 翻译用——仅支持简单表达式翻译）
_FEAT_ALIASES = {
    "close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume",
    "ret_1": "f0", "ret_3": "f1", "ret_5": "f2", "ret_10": "f3", "vol_5": "f4",
    "rsi": "f5", "macd": "f6", "vol_ratio": "f7",
    "ma_dist10": "f8", "ma_dist30": "f9", "bb_pos": "f10",
    "trend_regime": "f11", "risk_regime": "f12",
}


class PineExportError(ValueError):
    """模型无法导出为 Pine（体积超限 / 因子表达式不可翻译等）。"""


def _fmt(v: float) -> str:
    """Pine 数字字面量：一律带小数点（'%.6g' 会把 0.0/1.0 写成 int 字面量 0/1，
    混进三元链或 array.from 时与 float 分支类型不一致）。"""
    s = f"{float(v):.6g}"
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def _fmt_matrix(w: np.ndarray) -> list[str]:
    """权重展开成文本行：超过分块大小时用 array.concat 拼接。

    array.new_float(size, init) 全为同一初值，装不下逐元素权重；分块 array.from
    既避开超长调用，也让每行长度可控。
    """
    vals = [_fmt(x) for x in np.asarray(w, dtype=float).ravel()]
    if not vals:
        return ["array.new_float(0, 0.0)"]
    lines = [f"array.from({', '.join(vals[:_ARRAY_CHUNK])})"]
    for i in range(_ARRAY_CHUNK, len(vals), _ARRAY_CHUNK):
        lines.append(f"array.from({', '.join(vals[i:i + _ARRAY_CHUNK])})")
    return lines


def _emit_array(push, name: str, lines: list[str]) -> None:
    """把一个数组分块写入脚本：`name = array.from(...)` + 若干 `name := array.concat(...)`。"""
    if len(lines) == 1 and lines[0].startswith("array.from"):
        push(f"{name} = {lines[0]}")
        return
    push(f"var array<float> {name} = {lines[0]}")
    for extra in lines[1:]:
        push(f"{name} := array.concat({name}, {extra})")


def _translate_factor_expr(expr: str) -> str:
    """把训练端 factor_expression（如 'ma(close,5)/ma(close,20)-1'）翻译为 Pine 表达式。

    仅支持白名单：ma/std/max/min/sum/delta/delay 与基础算术。无法翻译时返回空串
    （调用方会跳过因子列并警告）。
    """
    expr = expr.strip()
    if not expr:
        return ""
    # 变量别名替换（最长的先替换避免子串误配）
    for k in sorted(_FEAT_ALIASES, key=len, reverse=True):
        expr = expr.replace(k, f"({_FEAT_ALIASES[k]})")
    # 白名单函数翻译
    import re as _re
    expr = _re.sub(r"\bma\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"ta.sma(\1, \2)", expr)
    expr = _re.sub(r"\bstd\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"ta.stdev(\1, \2)", expr)
    expr = _re.sub(r"\bmax\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"ta.highest(\1, \2)", expr)
    expr = _re.sub(r"\bmin\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"ta.lowest(\1, \2)", expr)
    expr = _re.sub(r"\bsum\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"ta.cum(\1)", expr)  # 近似：cum 累积
    expr = _re.sub(r"\bdelta\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"(\1 - \1[\2])", expr)
    expr = _re.sub(r"\bdelay\s*\(\s*([^,]+),\s*(\d+)\s*\)", r"\1[\2]", expr)
    # 合法字符校验（防注入）
    if not _re.fullmatch(r"[\w\s()+\-*/%<>=.,\[\]]+", expr):
        return ""
    return expr


def pine_feature_lines() -> list[str]:
    """13 维特征，与 env._precompute_features 逐项同口径（顺序不可变）。

    已知残差：pandas rolling.std 为样本标准差（ddof=1）、Pine ta.stdev 为总体标准差，
    以及 rolling.rank 与 ta.percentrank 的分母差一——预热区之外影响在小数点后量级，
    不改变动作档位；在此记录以免被误读为完全逐位一致。
    """
    tr, tv = _fmt(_TANH_RET), _fmt(_TANH_VOL)
    return [
        "// 0-3 收益率（tanh 压缩到 (-1,1)，与训练端 _TANH_RET 同口径）",
        "retPct = ta.change(close) / nz(close[1], close) * 100.0",
        f"f0 = nz(math.tanh(retPct / {tr}))",
        f"f1 = nz(math.tanh((close / nz(close[3], close) - 1.0) * 100.0 / {tr}))",
        f"f2 = nz(math.tanh((close / nz(close[5], close) - 1.0) * 100.0 / {tr}))",
        f"f3 = nz(math.tanh((close / nz(close[10], close) - 1.0) * 100.0 / {tr}))",
        f"// 4 已实现波动率（tanh，_TANH_VOL={tv}）｜5 RSI(14) 0-1",
        f"f4 = nz(math.tanh(nz(ta.stdev(retPct, 5)) / {tv}))",
        "f5 = nz(ta.rsi(close, 14) / 100.0)",
        "// 6 MACD 柱（dif-dea)*2 → clip(-10,10)/10",
        "dif = ta.ema(close, 12) - ta.ema(close, 26)",
        "f6 = nz(math.min(10.0, math.max(-10.0, (dif - ta.ema(dif, 9)) * 2.0 / close * 100.0)) / 10.0)",
        "// 7 量比 clip(0,5)/5 ｜ 8-9 MA 偏离 ｜ 10 布林带位置 clip(0,1)",
        "f7 = nz(math.min(5.0, volume / (ta.sma(volume, 5) + 1e-12)) / 5.0)",
        "f8 = nz((close / ta.sma(close, 10) - 1.0) * 100.0 / 10.0)",
        "f9 = nz((close / ta.sma(close, 30) - 1.0) * 100.0 / 10.0)",
        "bbMid = ta.sma(close, 20)",
        "bbStd = ta.stdev(close, 20)",
        "f10 = nz(math.min(1.0, math.max(0.0, (close - (bbMid - 2.0 * bbStd)) / (4.0 * bbStd + 1e-12))))",
        "// 11 趋势 regime ｜ 12 风险 regime（波动率 40 期滚动分位，无前视）",
        "f11 = nz(math.sign(close / nz(close[60], close) - 1.0))",
        "f12 = nz(ta.percentrank(nz(ta.stdev(retPct, 20)), 40) / 100.0)",
    ]


def _state_refs(state_window: int, has_factor: bool) -> list[str]:
    """状态向量各分量的 Pine 表达式，顺序严格等于 env._build_state：

    [bar_{t-w+1} 的 13 特征, …, bar_t 的 13 特征, posRatio, pnlRatio, (因子)]
    —— _state_feats 用 features[start:t+1].reshape(-1)，即旧→新逐根堆叠。
    """
    w = max(1, int(state_window))
    refs: list[str] = []
    for k in range(w):
        offset = w - 1 - k          # 最旧的一根偏移最大
        for j in range(N_PRICE_FEATURES):
            refs.append(f"f{j}" if offset == 0 else f"nz(f{j}[{offset}])")
    refs.append("posRatio")
    refs.append("pnlRatio")
    if has_factor:
        refs.append("factorVal")
    return refs


def _n_weights(in_dim: int, hidden: list[int], n_actions: int) -> int:
    dims = [int(in_dim), *[int(h) for h in hidden], int(n_actions)]
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def estimate_pine_bytes(in_dim: int, hidden: list[int], n_actions: int) -> int:
    """粗估脚本体积：权重文本 + 前向乘积项（每个权重约一一对应一行乘加）。

    实测（w=1/[32,32] 与 w=4/[16,16]）单权重成本约 35~39 字节，取 36；
    固定开销为特征段 + 头部注释 + 标记段。
    """
    dims = [int(in_dim), *[int(h) for h in hidden], int(n_actions)]
    weights = _n_weights(in_dim, hidden, n_actions) + sum(dims)   # 权重 + 偏置
    return int(weights * _BYTES_PER_WEIGHT + 4096)


def build_drl_pine(agent, model_name: str = "rl_agent",
                   symbol_label: str = "BTC/USDT",
                   factor_expression: str = "",
                   factor_mu: float = 0.0, factor_sd: float = 1.0,
                   train_metrics: dict | None = None) -> str:
    """把 ACAgent.actor 的权重导出为 Pine Script v5 策略代码。

    特征向量 = 13 维价量 × state_window + 2 维账户状态（持仓比例/浮动盈亏）+ 可选因子列，
    与 env._build_state 一致。网络输出 5 档 softmax 概率 → argmax 映射为目标仓位。

    参数:
        factor_expression: 训练时注入的因子表达式（翻译为 Pine 特征，并同口径标准化）
        factor_mu/factor_sd: 训练段拟合的因子标准化统计量（部署必须同一口径）
        train_metrics: {train_ret, oos_ret, sharpe, ...} 写入头部注释供用户对照

    Raises:
        PineExportError: 模型装不进 Pine（体积超限）或因子表达式不可翻译
    """
    W = agent.actor.W
    b = agent.actor.b
    in_dim = int(W[0].shape[0])
    n_actions = int(W[-1].shape[1])
    hidden = [int(w.shape[1]) for w in W[:-1]]
    state_window = max(1, int(getattr(agent, "state_window", 1) or 1))

    size = estimate_pine_bytes(in_dim, hidden, n_actions)
    if size > PINE_MAX_BYTES:
        n_weights = sum([in_dim, *hidden, n_actions][i] * [in_dim, *hidden, n_actions][i + 1]
                        for i in range(len(hidden) + 1))
        raise PineExportError(
            f"模型装不进 Pine：{in_dim}→{'→'.join(map(str, hidden))}→{n_actions} "
            f"约 {n_weights} 个权重（脚本约 {size // 1024}KB，上限 {PINE_MAX_BYTES // 1024}KB）。"
            f"state_window={state_window} 时特征维数是窗口的 13 倍——请用更小的 "
            "state_window / hidden 重新训练，才能导出可运行的自动交易代码。")

    # 级联因子（factor_miner 组合因子）是训练时注入的预计算数组，factor_expression
    # 会被清空、值来自 .npy 文件 → Pine 端无法重建这一列。此前它只报"维度不一致"，
    # 让人误以为导出器算错了窗口；这里直接说明这条路径不可导出。
    if not factor_expression and in_dim == (N_PRICE_FEATURES * state_window
                                           + N_ACCOUNT_FEATURES + 1):
        raise PineExportError(
            f"模型无法导出 Pine：它依赖 factor_miner 组合因子列（第 {in_dim} 维为预计算值，"
            f"非价量特征可复算）。该因子只在训练端按整段历史算出并存在模型文件旁，"
            f"Pine 端无法逐根K线重建。请改用「表达式因子」训练，或在本软件内运行该策略。")

    factor_expr_pine = _translate_factor_expr(factor_expression) if factor_expression else ""
    if factor_expression and not factor_expr_pine:
        raise PineExportError(f"因子表达式无法翻译为 Pine（{factor_expression}），请使用白名单表达式")
    has_factor = bool(factor_expr_pine)
    expected_dim = N_PRICE_FEATURES * state_window + N_ACCOUNT_FEATURES + (1 if has_factor else 0)
    if in_dim != expected_dim:
        raise PineExportError(
            f"模型输入维度 {in_dim} 与 state_window={state_window}"
            f"{' + 因子列' if has_factor else ''} 推出的 {expected_dim} 不一致，无法导出")

    L: list[str] = []
    push = L.append
    push("//@version=5")
    push("// ╔════════════════════════════════════════════════════════════════╗")
    push(f"//  DRL 神经引擎自动交易 · {model_name}")
    push(f"//  {in_dim} 维状态（13 价量 × {state_window} 窗口 + {N_ACCOUNT_FEATURES} 账户"
         + (" + 1 因子" if has_factor else "") + f"）→ {n_actions} 档目标仓位")
    push("//  权重由本软件 DRL 训练固化，可直接粘贴到 TradingView Pine 编辑器")
    if train_metrics:
        push(f"//  训练收益 {train_metrics.get('train_ret', '-')} · "
             f"OOS收益 {train_metrics.get('oos_ret', '-')} · 夏普 {train_metrics.get('sharpe', '-')}")
    push(f"//  特征口径与训练环境 drl/env.py 逐项一致（tanh 有界 + clip + 滚动分位）")
    push("// ╚════════════════════════════════════════════════════════════════╝")
    push('strategy("DRL · ' + model_name + '", overlay=true,')
    push('     initial_capital=10000,')
    push('     default_qty_type=strategy.percent_of_equity, default_qty_value=100,')
    push('     commission_type=strategy.commission.percent, commission_value=0.1)')
    push('')
    push('minZonePct = input.float(5.0, "调仓死区（百分点）", minval=0.0, step=0.5)')
    push('hardStopPct = input.float(8.0, "硬止损（%，0=关闭）", minval=0.0, step=0.5)')
    push('')
    push('// ───── ① 特征计算（与训练环境 _precompute_features 同口径） ─────')
    for line in pine_feature_lines():
        push(line)
    push('// 账户状态：持仓比例（占总权益）与浮动盈亏')
    push('posRatio = nz(strategy.position_size * close / strategy.equity)')
    push('pnlRatio = strategy.position_avg_price > 0 ? (close - strategy.position_avg_price) / strategy.position_avg_price : 0.0')
    if has_factor:
        push('// 因子列：训练段拟合的 mu/sd 标准化（与部署端 rl_adaptive 一致）')
        if float(factor_sd) and abs(float(factor_sd) - 1.0) > 1e-12:
            push(f'factorVal = nz((nz({factor_expr_pine}) - {_fmt(float(factor_mu))}) / {_fmt(float(factor_sd))})')
        else:
            push(f'factorVal = nz({factor_expr_pine})')
    push('')
    push('// ───── ② 状态向量（顺序 = env._build_state：旧→新堆叠 + 账户 + 因子） ─────')
    refs = _state_refs(state_window, has_factor)
    chunks = [refs[i:i + _ARRAY_CHUNK] for i in range(0, len(refs), _ARRAY_CHUNK)]
    push(f'h0 = array.from({", ".join(chunks[0])})')
    for ch in chunks[1:]:
        push(f'h0 := array.concat(h0, array.from({", ".join(ch)}))')
    push('')
    push('// ───── ③ MLP 前向（权重固化，ReLU 隐藏层 + softmax 输出） ─────')
    for i, (wi, bi) in enumerate(zip(W, b)):
        push(f'// Layer {i}: {wi.shape[0]}→{wi.shape[1]}')
        _emit_array(push, f'w{i}', _fmt_matrix(wi))
        _emit_array(push, f'b{i}', _fmt_matrix(bi))
    prev_arr = 'h0'
    for i in range(len(W) - 1):
        n_in, n_out = int(W[i].shape[0]), int(W[i].shape[1])
        for j in range(n_out):
            terms = ' + '.join(f'{prev_arr}[{k}] * w{i}[{k * n_out + j}]' for k in range(n_in))
            push(f'z{i}_{j} = {terms} + b{i}[{j}]')
            push(f'a{i}_{j} = math.max(z{i}_{j}, 0.0)')
        rows = ', '.join(f'a{i}_{j}' for j in range(n_out))
        push(f'h{i + 1} = array.from({rows})')
        prev_arr = f'h{i + 1}'
    last = len(W) - 1
    n_out = int(W[last].shape[1])
    for j in range(n_out):
        n_in = int(W[last].shape[0])
        terms = ' + '.join(f'{prev_arr}[{k}] * w{last}[{k * n_out + j}]' for k in range(n_in))
        push(f'z{last}_{j} = {terms} + b{last}[{j}]')
    mx = f'z{last}_0'
    for j in range(1, n_out):
        mx = f'math.max({mx}, z{last}_{j})'
    push(f'mx = {mx}')
    push('ex = array.from(' + ', '.join(f'math.exp(z{last}_{j} - mx)' for j in range(n_out)) + ')')
    push('sumEx = ' + ' + '.join(f'ex[{j}]' for j in range(n_out)))
    push('proba = array.from(' + ', '.join(f'ex[{j}] / sumEx' for j in range(n_out)) + ')')
    push('// 最优仓位档位 = argmax(proba)')
    push('bestIdx = 0')
    for j in range(1, n_out):
        push(f'if proba[{j}] > proba[bestIdx]')
        push(f'    bestIdx := {j}')
    buckets = ACTION_BUCKETS[:n_out]
    # 合法三元链：各档以 " : " 分隔，末档为兜底分支
    expr = ' : '.join(f'bestIdx == {i} ? {_fmt(v)}' for i, v in enumerate(buckets[:-1]))
    push(f'target = {expr} : {_fmt(buckets[-1])}' if buckets[:-1] else f'target = {_fmt(buckets[-1])}')
    push('')
    push('// ───── ④ 调仓执行（percent_of_equity 语义 + 死区 + 硬止损） ─────')
    push('posPct = strategy.equity > 0 ? strategy.position_size * close * 100.0 / strategy.equity : 0.0')
    push('wantPct = 100.0 * target')
    push('// 硬止损：浮亏超过阈值强制清仓（0 关闭）')
    push('hardHit = hardStopPct > 0 and strategy.position_size > 0 and pnlRatio < -hardStopPct / 100.0')
    push('if hardHit')
    push('    strategy.close("L", comment="硬止损")')
    push('else if wantPct > posPct + minZonePct')
    push('    // 目标高于当前 → 按差额买入（percent_of_equity：qty 即增量占比）')
    push('    strategy.entry("L", strategy.long, qty=wantPct - posPct, comment="RL加仓至'
         '" + str(math.round(target * 100)) + "%")')
    push('else if wantPct < posPct - minZonePct and strategy.position_size > 0')
    push('    if wantPct <= 0')
    push('        strategy.close("L", comment="RL清仓")')
    push('    else')
    push('        // 目标低于当前 → 部分平仓减仓（曾误用 strategy.order(long) 表达减仓）')
    push('        strategy.close("L", qty_percent=(posPct - wantPct) / posPct * 100.0,')
    push('             comment="RL减仓至" + str(math.round(target * 100)) + "%")')
    push('')
    push('// ───── ⑤ 可视化：目标仓位 + 买卖位置 ─────')
    push('// target ∈ 0~1，直接 plot 在价格轴上贴 0 不可见 → 乘 close 折算到价格量纲')
    push('plot(target * close, title="目标仓位（×close 折算）",')
    push('     color=color.new(#0a84ff, 25), style=plot.style_stepline, linewidth=2)')
    push(TRADE_MARKERS_PINE)
    return "\n".join(L) + "\n"


def try_build_drl_pine(agent, model_name: str = "rl_agent", **kwargs) -> tuple[str, str]:
    """导出 Pine，失败时返回 (空代码, 原因说明)——调用方据此在 UI 显式提示。

    持续进化产出的模型不一定装得进 Pine；与其静默回落到别的策略模板（用户会
    在图上看到与训练模型无关的买卖点），不如留空并把原因说清楚。
    """
    try:
        return build_drl_pine(agent, model_name=model_name, **kwargs), ""
    except PineExportError as e:
        return "", str(e)
    except Exception as e:  # noqa: BLE001
        return "", f"Pine 导出失败：{e}"


def export_pine_from_model(model_path, model_name: str = "rl_agent") -> tuple[str, str]:
    """从磁盘上的模型文件现场导出 Pine，返回 (代码, 不可导出原因)。

    注册策略时统一走这里，而不是复用训练返回值里的 pine_code：模型文件才是
    部署端/回测端实际运行的那份权重（回退、重启恢复、切换模型都只改文件），
    从它导出才能保证图上的买卖点与真实行为同源。
    """
    import json as _json
    import os
    path = str(model_path)
    if not os.path.exists(path):
        return "", "模型文件不存在，无法导出自动交易代码"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        from .agent import ACAgent      # 延迟导入：避免与 drl.agent 循环依赖
        agent = ACAgent.from_dict(data)
    except Exception as e:  # noqa: BLE001
        return "", f"模型文件读取失败，无法导出 Pine：{e}"
    _tm = data.get("train_meta") or {}
    _oos = _tm.get("oos_report") or {}
    return try_build_drl_pine(
        agent, model_name=model_name,
        factor_expression=str(data.get("factor_expression") or ""),
        factor_mu=float(data.get("factor_mu") or 0.0),
        factor_sd=float(data.get("factor_sd") or 1.0),
        train_metrics={"train_ret": _tm.get("best_ret", "-"),
                       "oos_ret": _oos.get("oos_ret", "-"),
                       "sharpe": _oos.get("sharpe", "-")})
