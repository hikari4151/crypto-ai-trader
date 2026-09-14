"""智能因子挖掘与合成。

1. **表达式沙箱**：AI 生成的因子表达式经 AST 白名单校验后安全执行
   （只允许受控的数学运算 + 白名单函数 + OHLCV 变量，杜绝任意代码执行）。
2. **因子合成**：按 IC/ICIR 加权合成组合因子，或等权合成；支持相关性去冗余。
3. **AI 挖掘流程**：给定市场快照与现有因子表现，AI 输出候选因子表达式，
   在历史数据上验证 IC / 分层收益，只收录有效的因子。
"""
import ast
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from .analysis import (DEFAULT_GATES, factor_ic_table, factor_quality_gate,
                       factor_turnover)
from .engine import compute_factor_matrix

log = logging.getLogger(__name__)

# ============ 1. 表达式沙箱 ============

# 允许的单变量函数（输入 pd.Series，输出 pd.Series，保持索引一致）
_WHITELIST_FUNCS = {
    "abs": lambda s: s.abs(),
    "log": lambda s: np.log(s.clip(lower=1e-10)),
    "sign": lambda s: np.sign(s),
    "sqrt": lambda s: np.sqrt(s.clip(lower=0.0)),
    "rank": lambda s: s.rank(pct=True),
    "normalize": lambda s: (s - s.mean()) / (s.std() + 1e-12),
}

# 允许的窗口函数（输入 pd.Series，返回同索引 Series）
_WINDOW_FUNCS = {
    "ma": lambda s, n: s.rolling(int(n)).mean(),
    "std": lambda s, n: s.rolling(int(n)).std(),
    "max": lambda s, n: s.rolling(int(n)).max(),
    "min": lambda s, n: s.rolling(int(n)).min(),
    "sum": lambda s, n: s.rolling(int(n)).sum(),
    "median": lambda s, n: s.rolling(int(n)).median(),
    "delta": lambda s, n: s - s.shift(int(n)),
    "delay": lambda s, n: s.shift(int(n)),
    "pct_change": lambda s, n: s.pct_change(int(n)),
}

# 允许的变量名：OHLCV 列 + 内置因子 key + 常量
_VARIABLES = {"open", "high", "low", "close", "volume", "v", "c"}

_ALLOWED_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                ast.UAdd, ast.USub, ast.FloorDiv}
_ALLOWED_COMPARE = {ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq}
_ALLOWED_BOOL = {ast.And, ast.Or, ast.Not}


class FactorExprError(ValueError):
    """因子表达式非法（含非白名单操作）。"""


def validate_expression(expr: str) -> bool:
    """校验表达式是否可安全执行（仅白名单）。非法则抛 FactorExprError。"""
    _parse_expr(expr)
    return True


@lru_cache(maxsize=256)
def _parse_expr(expr: str) -> ast.AST:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FactorExprError(f"表达式语法错误: {e}")
    _check_node(tree.body)
    return tree.body


def _check_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_OPS:
            raise FactorExprError(f"不允许的操作符: {type(node.op).__name__}")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        # UAdd/USub 数值正负；Not 逻辑取反（Python 中 not 是 UnaryOp，
        # 曾只放行 UAdd/USub，导致提示词宣称支持的 not 必被拒）
        if type(node.op) not in {ast.UAdd, ast.USub, ast.Not}:
            raise FactorExprError(f"不允许的一元操作: {type(node.op).__name__}")
        _check_node(node.operand)
    elif isinstance(node, ast.BoolOp):
        if type(node.op) not in _ALLOWED_BOOL:
            raise FactorExprError(f"不允许的逻辑运算: {type(node.op).__name__}")
        for v in node.values:
            _check_node(v)
    elif isinstance(node, ast.Compare):
        if not all(type(op) in _ALLOWED_COMPARE for op in node.ops):
            raise FactorExprError("不允许的比较操作符")
        _check_node(node.left)
        for c in node.comparators:
            _check_node(c)
    elif isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name not in _WHITELIST_FUNCS and name not in _WINDOW_FUNCS:
            raise FactorExprError(f"不允许的函数: {name}")
        for a in node.args:
            _check_node(a)
        if name in _WINDOW_FUNCS and len(node.args) != 2:
            raise FactorExprError(f"窗口函数 {name} 需要 2 个参数（序列, 窗口）")
        if name in _WHITELIST_FUNCS and len(node.args) != 1:
            raise FactorExprError(f"函数 {name} 需要 1 个参数")
    elif isinstance(node, ast.Name):
        if node.id not in _VARIABLES:
            raise FactorExprError(f"不允许的变量: {node.id}")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise FactorExprError(f"不允许的常量: {node.value!r}")
    elif isinstance(node, ast.IfExp):
        _check_node(node.test)
        _check_node(node.body)
        _check_node(node.orelse)
    else:
        raise FactorExprError(f"不允许的语法节点: {type(node).__name__}")


class FactorExecutor:
    """把经过校验的表达式绑定到具体 DataFrame 执行。"""

    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.tree = _parse_expr(expr)
        self._used_vars = _collect_vars(self.tree)

    def eval(self, df: pd.DataFrame) -> pd.Series:
        """在 df 上执行表达式，返回 Series（与 df 同索引）。"""
        missing = [v for v in self._used_vars if v not in df.columns and v not in ("v", "c")]
        if missing:
            raise FactorExprError(f"缺少计算所需的列: {missing}")
        return _eval_node(self.tree, df)


def _collect_vars(node: ast.AST) -> set[str]:
    """收集表达式引用的数据变量（排除函数名与关键字）。"""
    found = set()
    # 函数调用中的 func 名不是变量
    func_names = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name)}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in func_names:
            found.add(n.id)
    return found


def _eval_node(node: ast.AST, df: pd.DataFrame) -> pd.Series:
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, df)
        right = _eval_node(node.right, df)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / (right.replace(0.0, np.nan))
        if op is ast.Mod:
            return left % right
        if op is ast.Pow:
            return left ** right
        if op is ast.FloorDiv:
            return left // right
    elif isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, df)
        if isinstance(node.op, ast.Not):
            return (~v.astype(bool)).astype(float)
        return -v if isinstance(node.op, ast.USub) else v
    elif isinstance(node, ast.Compare):
        left = _eval_node(node.left, df)
        right = _eval_node(node.comparators[0], df)
        op = node.ops[0]
        mask = {
            ast.Lt: left < right, ast.LtE: left <= right,
            ast.Gt: left > right, ast.GtE: left >= right,
            ast.Eq: left == right, ast.NotEq: left != right,
        }[type(op)]
        return mask.astype(float)
    elif isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, df).astype(bool) for v in node.values]
        if isinstance(node.op, ast.And):
            out = vals[0]
            for v in vals[1:]:
                out = out & v
        elif isinstance(node.op, ast.Or):
            out = vals[0]
            for v in vals[1:]:
                out = out | v
        else:
            out = ~vals[0]
        return out.astype(float)
    elif isinstance(node, ast.Call):
        name = node.func.id
        if name in _WINDOW_FUNCS:
            s = _eval_node(node.args[0], df)
            n = int(_const(node.args[1]))
            return _WINDOW_FUNCS[name](s, n)
        arg = _eval_node(node.args[0], df)
        return _WHITELIST_FUNCS[name](arg)
    elif isinstance(node, ast.Name):
        v = node.id
        if v in df.columns:
            return df[v]
        if v in ("v", "c"):
            return df["volume"] if v == "v" else df["close"]
        raise FactorExprError(f"未知变量: {v}")
    elif isinstance(node, ast.Constant):
        return pd.Series(node.value, index=df.index, dtype=float)
    elif isinstance(node, ast.IfExp):
        cond = _eval_node(node.test, df).astype(bool)
        a = _eval_node(node.body, df)
        b = _eval_node(node.orelse, df)
        return pd.Series(np.where(cond, a, b), index=df.index)
    raise FactorExprError(f"无法执行节点: {type(node).__name__}")


def _const(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    raise FactorExprError("窗口参数必须为数字常量")


# ============ 2. 因子合成 ============

def composite_factor(mat: pd.DataFrame, weights: dict[str, float],
                     method: str = "ic", z_mode: str = "full",
                     z_map: Optional[dict[str, pd.Series]] = None) -> pd.Series:
    """合成组合因子。

    mat: 因子矩阵（列=因子）
    weights: {key: 权重}；method="ic" 时若未传权重则用 |rank_ic| 加权
    z_mode: "full"=全样本标准化（默认，历史行为逐位不变）；
            "expanding"=只用截至当期 t 的统计量标准化（无前视，与
            dynamic_composite 的 expanding 口径一致——RL 因子挖掘奖励路径使用，
            曾全样本 mean/std 让 fitness 携带验证段未来分布信息）
    z_map: 可选预计算 z（P2-12，仅 z_mode="expanding" 时生效）——调用方在
           初始化时按同一 expanding 公式一次性算好全部列的 z，避免每步全量重算
           统计量；为 None 时按 z_mode 现场计算（行为逐位不变）
    """
    # 只用有权重（或 method=ic 时全部）的列
    cols = list(weights.keys()) if weights else [c for c in mat.columns]
    cols = [c for c in cols if c in mat.columns and mat[c].notna().sum() > 5]
    if not cols:
        raise ValueError("没有可用于合成的因子")
    z = {}
    for c in cols:
        s = mat[c]
        if z_map is not None and z_mode == "expanding" and c in z_map:
            z[c] = z_map[c]
        elif z_mode == "expanding":
            # 无前视：z 只用截至合成时点 t 的统计量（同 dynamic_composite 口径）；
            # 早期样本不足（expanding std 为 NaN）按 0（均值）填充
            z[c] = ((s - s.expanding().mean()) / (s.expanding().std() + 1e-12)).fillna(0.0)
        else:
            z[c] = (s - s.mean()) / (s.std() + 1e-12)
    # 权重归一化（按绝对值比例）
    total = sum(abs(weights.get(c, 0.0)) for c in cols)
    w = {c: weights.get(c, 0.0) / total for c in cols} if total else {c: 1.0 / len(cols) for c in cols}
    out = pd.Series(0.0, index=mat.index)
    for c in cols:
        out = out + z[c] * w[c]
    return out


class CompositeFactorEvaluator:
    """实盘复算 factor_miner 级联组合因子（训练口径不变式）。

    背景：持续进化挖掘的组合因子在训练端由 composite_factor(z_mode="expanding")
    在整段历史因子矩阵上算出，train_drl 再用训练段拟合的 mu/sd 二次标准化后
    作为策略状态特征。部署端拿不到整段历史，此前把预计算数组打进模型文件、
    声称"rl_adaptive 按索引取用"——但数组无法覆盖实盘新K线，实盘因子列实际
    全为 None，带级联因子的模型每根K线都走"跳过本根"分支、永不交易。

    本类在滚动缓冲上按**同一公式**复算：选中因子的序列用 compute_factor_matrix
    （与训练同函数同口径）计算，expanding z 只用截至当下的统计量（无前视、
    早样本填 0，与 composite_factor/z_mode="expanding" 逐位一致），权重按
    参与合成列的绝对值归一，最后用模型内训练段统计量 mu/sd 做二次标准化
    （与 train_drl 注入口径一致）。滚动缓冲长度与训练窗口不同会造成数值
    漂移（与 factor_signal 局部 z 同类限制），但方向与量级信息保持有效。
    """

    def __init__(self, weights: dict, mu: Optional[float] = None,
                 sd: Optional[float] = None) -> None:
        self.weights = {str(k): float(w) for k, w in (weights or {}).items() if w}
        self.mu = float(mu) if mu is not None else None
        self.sd = float(sd) if sd else None

    def eval(self, df: pd.DataFrame) -> Optional[float]:
        """在 df（标准 OHLCV，越长的滚动历史越接近训练口径）上复算最新组合因子值。"""
        if not self.weights or df is None or len(df) < 5:
            return None
        keys = [k for k in self.weights if k]
        try:
            mat = compute_factor_matrix(df, keys=keys)
        except (ValueError, TypeError, KeyError):
            return None
        cols = [c for c in mat.columns if mat[c].notna().sum() > 5]
        if not cols:
            return None
        total = sum(abs(self.weights[c]) for c in cols)
        if total <= 0:
            return None
        # expanding z（无前视）：与 composite_factor(z_mode="expanding") 同式，
        # 早样本不足填 0（均值）；权重按参与合成列的绝对值归一
        z = ((mat[cols] - mat[cols].expanding().mean())
             / (mat[cols].expanding().std() + 1e-12)).fillna(0.0)
        w = {c: self.weights[c] / total for c in cols}
        out = sum(float(z[c].iloc[-1]) * w[c] for c in cols)
        # 二次标准化（模型内训练段统计量；无统计量的旧模型返回原始 z 合成值）
        if self.mu is not None and self.sd:
            out = (out - self.mu) / self.sd
        return float(out) if out == out else None


def ic_weighted_composite(mat: pd.DataFrame, close: pd.Series, h: int = 1,
                          top_n: int = 8, min_abs_ic: float = 0.03) -> dict:
    """IC 加权合成：选 |rank_ic| 最高且方向明确的 top_n 个因子，按 |rank_ic| 加权。

    返回 {composite: Series, weights: {key: 权重}, ics: 选中因子的 rank_ic}
    """
    tbl = factor_ic_table(mat, close, h=h, method="rank")
    sel = [r for r in tbl if r["n"] >= 20 and abs(r.get("rank_ic", 0.0)) >= min_abs_ic]
    sel = sel[:top_n]
    if not sel:
        raise ValueError("没有达到 IC 门槛的因子，无法合成")
    weights = {}
    for r in sel:
        # 方向：rank_ic 为负则反向暴露（乘 -1）
        weights[r["key"]] = r["rank_ic"]
    combo = composite_factor(mat, weights, method="ic")
    return {
        "composite": combo,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "selected": [{"key": r["key"], "rank_ic": r.get("rank_ic"), "icir": r.get("icir")} for r in sel],
    }


def dynamic_composite(mat: pd.DataFrame, close: pd.Series, h: int = 1,
                      ic_window: int = 120, min_abs_ic: float = 0.03,
                      top_n: int = 8) -> dict:
    """动态权重合成组合因子（借鉴 AlphaForge 的时序表现动态调权）。

    与 ic_weighted_composite（全程单一权重）不同：权重随因子近期表现滚动更新——
    因子衰减时自动降权，新晋有效的因子自动入选，对抗因子衰减与市场风格切换。

    防前视（关键）：
    - 每个因子在非重叠窗口上算滚动 IC（analysis._rolling_ic，样本独立）
    - 时点 t 的权重只能用"截至 t-h 已实现"的滚动 IC（t 期因子 IC 需要 t+h 收益，
      因此 shift(h) 后才可用）；早期无 IC 时权重为 0（不参与合成）

    返回 {composite: Series, weight_history: DataFrame(列=因子), coverage: 参与合成的时点占比}
    """
    from .analysis import _rolling_ic

    # 滚动 IC 序列（非重叠窗口，索引处有值、其余 NaN）
    ics = {col: _rolling_ic(mat[col], close, h=h, method="rank", window=ic_window)
           for col in mat.columns}
    ic_matrix = pd.DataFrame(ics, index=mat.index)
    # 前视防护：t 时点可见的 IC 截止到 t-h（shift 后 ffill 取最近已实现值）
    ic_avail = ic_matrix.shift(h).ffill()

    # 逐时点权重：|IC|≥门槛 的 top_n，按 |IC| 加权（负 IC 反向暴露）
    n = len(mat)
    weight_rows = np.zeros((n, len(mat.columns)))
    # 标准化用 expanding（截至当期）统计量：曾用全样本 mean/std（含未来段），
    # 组合值携带未来分布信息且与实盘滚动缓冲（factor_signal 300 根局部 z）口径漂移
    z = (mat - mat.expanding().mean()) / (mat.expanding().std() + 1e-12)
    z = z.fillna(0.0).to_numpy(float)
    cols = list(mat.columns)
    for t in range(n):
        row = ic_avail.iloc[t]
        # 保留符号：负 IC 因子反向暴露（权重为负）。筛选/排序/归一按 |IC|，
        # 写入权重带原符号——与 ic_weighted_composite 的方向语义一致。
        sel = [(c, v) for c, v in row.items() if v == v and abs(v) >= min_abs_ic]
        if not sel:
            continue
        sel.sort(key=lambda x: -abs(x[1]))
        sel = sel[:top_n]
        total = sum(abs(v) for _, v in sel)
        for c, v in sel:
            weight_rows[t, cols.index(c)] = v / total if total > 0 else 0.0

    weights_df = pd.DataFrame(weight_rows, index=mat.index, columns=cols)
    composite = pd.Series((weights_df.to_numpy() * z).sum(axis=1), index=mat.index)
    coverage = float((weights_df.abs().sum(axis=1) > 0).mean())
    return {"composite": composite, "weight_history": weights_df,
            "coverage": round(coverage, 4)}


# ============ 2.5 程序化进化变异（借鉴 QuantaAlpha 轨迹进化 / alpha-mining 遗传编程） ============

# 窗口参数变异倍数：对表达式中的窗口期做 ± 扰动生成变体
_EVOLVE_WINDOW_SCALES = (0.5, 0.8, 1.2, 1.5)


def _evolve_expression(expr: str, window: int, new_window: int) -> str:
    """把表达式中的指定窗口参数替换为新值（带括号边界保护）。"""
    import re
    # 匹配 ", N)" 或 ",N)"（窗口参数在窗口函数第二参数位）
    return re.sub(r",\s*" + str(window) + r"\s*\)", f", {new_window})", expr)


def evolve_factors(valid_factors: list[dict], df: pd.DataFrame, h: int = 1,
                   oos_ratio: float = 0.25, gates: Optional[dict] = None) -> list[dict]:
    """对 AI 挖掘出的有效因子做窗口参数进化变异，生成 IC 提升且样本外稳健的变体。

    借鉴遗传编程的参数变异算子 + 防过拟合要求（BRAIN IS/OOS 双验证）：
    - 解析因子表达式中的窗口参数（ma/std/max/min 等的第二参数）
    - 对窗口期做 ±50%/-20%/+20%/+50% 扰动生成变体
    - 按时间顺序切分 IS/OOS 两段分别验证：
        IS 段 |rank_ic| 须超过原因子 IS 段 |rank_ic|×1.05 且 ≥ 门槛（选出提升）
        OOS 段 |rank_ic| 须 ≥ 门槛 且方向与 IS 段一致（防样本内过拟合）
        OOS 段换手须 ≤ 上限（高换手变体即使 IC 提升也不可取）
    - OOS 段不足（<50 根）时退化为仅 IS 验证，并标记 oos_skipped=True
    - 纯程序计算（无额外 AI 调用），成本极低

    返回：原始有效因子（附 OOS 报告）+ 进化出的更优变体（合并后按 |IS rank_ic| 排序去重）
    """
    import re
    from .analysis import factor_ic

    g = {**DEFAULT_GATES, **(gates or {})}
    close = df["close"]
    n = len(df)
    n_oos = int(n * oos_ratio)
    # OOS 段或 IS 段太短时无法做独立样本外验证 → 退化为仅 IS 提升校验
    oos_skipped = n_oos < 50 or (n - n_oos) < 100
    is_df = df.iloc[: n - n_oos] if not oos_skipped else df
    oos_df = df.iloc[n - n_oos:] if not oos_skipped else df.iloc[:0]
    if oos_skipped:
        log.warning("[factor] 数据量不足（%d 根），进化变异跳过独立 OOS 验证", n)

    # 原因子在 IS 段的 |rank_ic|（与变体同口径对比，避免全量/分段混用）
    base_is_ic: dict[str, float] = {}
    for f in valid_factors:
        try:
            s = FactorExecutor(f["expression"]).eval(is_df).astype(float)
            base_is_ic[f["name"]] = abs(factor_ic(s, close.loc[is_df.index], h=h, method="rank")["rank_ic"])
        except Exception:  # noqa: BLE001
            base_is_ic[f["name"]] = 0.0

    # 原因子统一为 IS 段口径 + 补 OOS 报告（与进化变体同口径对比，展示/排序才公平）
    for f in valid_factors:
        f["oos_skipped"] = oos_skipped
        f["oos_rank_ic"] = None
        try:
            s_is = FactorExecutor(f["expression"]).eval(is_df).astype(float)
            if s_is.notna().sum() >= 20:
                ic_is = factor_ic(s_is, close.loc[is_df.index], h=h, method="rank")
                f["rank_ic"] = ic_is["rank_ic"]
                f["ic"] = ic_is["ic"]
                f["icir"] = ic_is["icir"]
        except Exception:  # noqa: BLE001
            pass
        if oos_skipped:
            continue
        try:
            s_oos = FactorExecutor(f["expression"]).eval(oos_df).astype(float)
            if s_oos.notna().sum() >= 20:
                f["oos_rank_ic"] = factor_ic(s_oos, close.loc[oos_df.index], h=h, method="rank")["rank_ic"]
        except Exception:  # noqa: BLE001
            pass

    evolved = []
    seen_exprs = {f["expression"] for f in valid_factors}

    for f in valid_factors:
        expr = f["expression"]
        # 提取窗口参数（形如 func(..., N)）
        windows = re.findall(r",\s*(\d{1,3})\s*\)", expr)
        for w_str in windows:
            w = int(w_str)
            if w < 3 or w > 200:
                continue
            for scale in _EVOLVE_WINDOW_SCALES:
                new_w = max(2, int(round(w * scale)))
                if new_w == w:
                    continue
                new_expr = _evolve_expression(expr, w, new_w)
                if new_expr in seen_exprs:
                    continue
                try:
                    fx = FactorExecutor(new_expr)
                    s_is = fx.eval(is_df).astype(float)
                    if s_is.notna().sum() < 20:
                        continue
                    ic_is = factor_ic(s_is, close.loc[is_df.index], h=h, method="rank")
                    base = base_is_ic.get(f["name"], 0.0)
                    # IS 提升门槛：须明显超过原因子且超过最低 IC 门槛
                    if abs(ic_is["rank_ic"]) <= base * 1.05 or abs(ic_is["rank_ic"]) <= g["min_abs_ic"]:
                        continue
                    # 换手门槛（IS 段）
                    if factor_turnover(s_is) > g["max_turnover"]:
                        continue
                    # OOS 验证：|rank_ic| 达标 + 方向与 IS 一致
                    oos_rank_ic = None
                    if not oos_skipped:
                        s_oos = fx.eval(oos_df).astype(float)
                        if s_oos.notna().sum() < 20:
                            continue
                        ic_oos = factor_ic(s_oos, close.loc[oos_df.index], h=h, method="rank")
                        oos_rank_ic = ic_oos["rank_ic"]
                        if abs(oos_rank_ic) < g["min_abs_ic"]:
                            continue
                        if np.sign(oos_rank_ic) != np.sign(ic_is["rank_ic"]):
                            continue
                    seen_exprs.add(new_expr)
                    evolved.append({
                        **{k: v for k, v in f.items() if k not in ("rank_ic", "ic", "icir", "samples", "oos_rank_ic")},
                        "name": f"{f['name']}_e{new_w}",
                        "title": f"{f.get('title', f['name'])}进化",
                        "expression": new_expr,
                        "logic": f"{f.get('logic', '')}（窗口 {w}→{new_w} 进化，IS+OOS 验证）",
                        "valid": True, "evolved_from": f["name"],
                        "rank_ic": ic_is["rank_ic"], "ic": ic_is["ic"],
                        "icir": ic_is["icir"], "samples": int(s_is.notna().sum()),
                        "oos_rank_ic": oos_rank_ic, "oos_skipped": oos_skipped,
                    })
                except Exception:  # noqa: BLE001
                    continue

    # 合并原因子 + 进化变体，按 |IS rank_ic| 排序
    merged = list(valid_factors) + evolved
    merged.sort(key=lambda r: -abs(r.get("rank_ic", 0.0)))
    return merged


# ============ 3. AI 因子挖掘 ============

def mine_prompt_system(prev_feedback: Optional[list] = None) -> str:
    """AI 挖掘因子的系统提示词。

    prev_feedback: 上一轮 AI 挖掘因子的 IC 表现反馈（借鉴 QuantaAlpha 自进化：
    让 AI 知道哪些方向有效、哪些无效，从而迭代优化表达式而非每次从头猜）。
    """
    base = (
        "你是一位顶尖量化因子研究员。请基于给定市场数据特征与现有因子，设计新的候选因子表达式。\n\n"
        "【思考框架（内部推理）】\n"
        "1. 分析当前市场：趋势/震荡/波动率水平，哪些信息尚未被现有因子覆盖\n"
        "2. 提出 3-5 个有经济逻辑的因子假设（动量/反转/量价背离/波动率调节等）\n"
        "3. 将每个假设转化为数学表达式，用白名单函数组合\n"
        "4. 若提供【上轮因子表现】，必须据此调整：表现好的方向深化，表现差的方向避免\n\n"
        "【可用变量】close, high, low, open, volume\n"
        "【可用函数】\n"
        "- 单变量: abs, log, sign, sqrt, rank, normalize\n"
        "- 窗口(序列, n): ma, std, max, min, sum, median, delta, delay, pct_change\n"
        "- 运算: + - * / % **，比较 > < >= <= == !=，逻辑 and or not，条件 a if cond else b\n\n"
        "【输出 schema】\n"
        '{"factors": [{"name": "英文小写下划线因子名(如 vol_adj_mom)", "title": "中文名",\n'
        '  "category": "momentum|volatility|volume|trend|mean_reversion|macro",\n'
        '  "expression": "只用白名单的数学表达式", "logic": "中文经济逻辑(60字内)",\n'
        '  "expected_behavior": "该因子在什么市场条件下有效"}]}\n'
        "要求：表达式必须只使用上述白名单符号；每个因子逻辑要自洽、可解释；"
        "尽量设计彼此互补（不同类别）的因子；避免与现有因子重复（包括窗口期相近的）。\n\n"
        "【输出纪律】只输出一个合法 JSON 对象，不要代码块、不要解释。"
    )
    if prev_feedback:
        fb = "\n【上轮因子表现（IC 反馈，用于自进化）】\n"
        for p in prev_feedback:
            fb += (f"- {p.get('name','?')} [{p.get('expression','')[:50]}]: "
                   f"rank_ic={p.get('rank_ic')} icir={p.get('icir')} "
                   f"({'有效' if abs(p.get('rank_ic',0))>=0.01 else '无效'})\n")
        fb += "请参考这些结果优化本轮表达式：深化有效方向、调整或避开无效方向。\n"
        return base + fb
    return base


def mine_prompt_user(market_desc: str, existing_factors: list[dict]) -> str:
    return (f"【市场数据特征】\n{market_desc}\n\n"
            f"【现有因子（避免重复）】\n{existing_factors}\n\n"
            "请深度思考后输出 3-5 个新的候选因子 JSON。")


def parse_ai_factors(data: dict) -> list[dict]:
    """解析 AI 返回的因子列表，校验表达式合法性。"""
    items = data.get("factors") or []
    out = []
    for it in items:
        expr = str(it.get("expression", "")).strip()
        try:
            validate_expression(expr)
        except FactorExprError as e:
            log.warning("[factor] AI 因子表达式非法: %s -> %s", expr, e)
            continue
        out.append({
            "name": str(it.get("name", "")).strip(),
            "title": str(it.get("title", "")).strip(),
            "category": str(it.get("category", "momentum")),
            "expression": expr,
            "logic": str(it.get("logic", "")),
            "expected_behavior": str(it.get("expected_behavior", "")),
        })
    return out


def evaluate_mined_factors(candidates: list[dict], df: pd.DataFrame,
                           h: int = 1, gates: Optional[dict] = None,
                           cross_validate_data: Optional[dict[str, pd.DataFrame]] = None) -> list[dict]:
    """在历史数据上验证挖掘出的因子：计算序列 + IC 分析 + 安检门槛。

    返回每个因子的 {name, title, category, expression, logic, series_keys,
    rank_ic, ic, icir, turnover, fitness, valid}；
    无效因子 valid=False 并说明原因（样本过少 / |rank_ic| / |icir| / 换手 任一不过即无效）。

    cross_validate_data: 可选跨品种验证数据，{symbol: OHLCV DataFrame}，
        同周期、≥200 根K线。表达式在每只品种上计算 IC/ICIR，方向一致且
        均值 ≥0.01 才算通过。None 时每项追加 cross_validation="not_configured"。
    """
    out = []
    close = df["close"]
    for cand in candidates:
        try:
            fx = FactorExecutor(cand["expression"])
            s = fx.eval(df).astype(float)
            # 跨品种验证：传递 fx.eval 作为 cross_factor_fn
            cross_fn = fx.eval if cross_validate_data else None
            gate = factor_quality_gate(s, close, h=h, gates=gates,
                                       cross_validate_data=cross_validate_data,
                                       cross_factor_fn=cross_fn)
            result = {
                **cand, "valid": gate["valid"],
                "rank_ic": gate["rank_ic"], "ic": gate["ic"], "icir": gate["icir"],
                "ic_win_rate": gate["ic_win_rate"],
                "turnover": gate["turnover"], "fitness": gate["fitness"],
                "spread": gate["spread"], "monotonic": gate["monotonic"],
                "samples": gate["samples"],
                "reason": gate["reason"],
                "series_key": None,
            }
            # 跨品种标记（gate 返回 None 时填充默认值）
            result["cross_validation"] = gate.get("cross_validation", "not_configured")
            result["cross_detail"] = gate.get("cross_detail", [])
            out.append(result)
        except Exception as e:  # noqa: BLE001
            log.warning("[factor] 挖掘因子执行失败 %s: %s", cand.get("expression"), e)
            result = {**cand, "valid": False, "reason": str(e)[:80]}
            result["cross_validation"] = "not_configured"
            result["cross_detail"] = []
            out.append(result)
    return out
