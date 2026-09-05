"""元策略控制器：统一多个子策略信号的智能调度器。

两种模式：
1. **ensemble（集成模式）**：置信度加权投票，按子策略的置信度加权合成最终信号。
   无需训练，立即可用。
2. **drl（强化学习模式）**：训练一个 DRL 智能体，根据市场状态、子策略信号和
   历史表现动态选择最优子策略。需要先训练。

设计目标：
- 所有现有策略（dual_ma / grid / price_action / factor_signal / rl_adaptive）
  都作为"子策略"被管理
- 元策略在子策略之上做决策，不修改子策略本身
- 回测引擎只需要运行元策略一个策略实例
"""
import copy
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

from .base import Signal, Strategy

log = logging.getLogger(__name__)

# ============ 元控制器状态向量：训练端与实盘端的单一事实来源 ============
# 曾两侧各拼各的：实盘算真实持仓比/浮盈，训练端 MetaControllerEnv._build_state
# 把这三个全局特征写死 [0,0,0]，且喂给子策略的是 open=high=low=close=price、
# volume=0 的退化快照。维度对得上、数值全不对，训出来的 DRL 模型部署即失配
# （实盘 len(state) != state_dim 就静默回退集成），所谓"智能调度"从未真正生效。
# 因此状态拼装只允许走 build_meta_state 这一条路径。

_META_FEAT_PER_STRATEGY = 4      # 方向 + 置信度 + 胜率 + 近期平均收益
_META_GLOBAL_FEATS = 3           # 持仓比 + 浮动盈亏 + 实现波动率
_DEFAULT_META_WINDOW = 20        # 子策略表现跟踪窗口（两侧同默认值）
_META_VOL_WINDOW = 20            # 实现波动率回看K线数
_META_VOL_SCALE = 0.02           # 归一尺度：2% 的 bar 收益标准差记满 1.0


def meta_state_dim(n_strategies: int) -> int:
    """子策略个数 → 状态维度（不含可选因子列）。"""
    return int(n_strategies) * _META_FEAT_PER_STRATEGY + _META_GLOBAL_FEATS


def realized_volatility(closes, window: int = _META_VOL_WINDOW) -> float:
    """近端 bar 收益率标准差（原始量纲）。

    实盘 ctx["indicators"] 里根本没有 volatility 这个键（全仓库无人产出），
    曾按 ind.get("volatility", 0.0) 取值 → 该特征实盘恒为 0。改由收盘序列
    自算，训练/实盘同一函数、同一窗口。
    """
    arr = np.asarray(closes, dtype=float)
    if arr.size < 3:
        return 0.0
    arr = arr[-(window + 1):]
    rets = np.diff(arr) / np.maximum(np.abs(arr[:-1]), 1e-9)
    return float(np.std(rets))


def meta_vol_feature(closes, window: int = _META_VOL_WINDOW) -> float:
    """实现波动率特征，归一到 [0,1]。"""
    return float(min(realized_volatility(closes, window) / _META_VOL_SCALE, 1.0))


def meta_performance(trades, window: int) -> tuple[float, float]:
    """子策略近期表现 → (胜率, 近期平均收益率)。"""
    trades = list(trades or [])
    if not trades:
        return 0.0, 0.0
    win_rate = sum(1.0 for p in trades if p > 0) / max(len(trades), 1)
    recent_avg = float(np.mean(trades[-window:])) if window > 0 else 0.0
    return float(win_rate), recent_avg


def record_meta_trade_result(trades: list, pnl: float, window: int) -> None:
    """追加一次平仓收益并裁剪到 2×window（胜率与近期均值都只看这段）。"""
    trades.append(float(pnl))
    keep = max(1, int(window)) * 2
    if len(trades) > keep:
        del trades[:len(trades) - keep]


def build_meta_state(strategy_names, signals: dict, trades_by_name: dict, *,
                     meta_window: int, pos_ratio: float, pnl_ratio: float,
                     vol_closes) -> np.ndarray:
    """拼装元控制器状态向量（训练 MetaControllerEnv 与实盘 MetaController 共用）。

    signals: {子策略名: {"direction": ±1/0, "confidence": float}}
    trades_by_name: {子策略名: [已平仓收益率, ...]}
    vol_closes: 截至当前K线的收盘序列尾部（含当前根）
    """
    parts: list[float] = []
    for name in strategy_names:
        info = signals.get(name) or {}
        direction = float(info.get("direction", 0.0) or 0.0)
        conf = float(info.get("confidence", 0.0) or 0.0)
        win_rate, recent_avg = meta_performance(trades_by_name.get(name), meta_window)
        parts.extend([direction, conf, win_rate, recent_avg])
    parts.extend([float(pos_ratio), float(pnl_ratio), meta_vol_feature(vol_closes)])
    return np.asarray(parts, dtype=float)


def resolve_sub_executors(names) -> list[str]:
    """子策略名 → executor 名（动态策略查注册表，内置策略同名）。

    引擎据此判断要不要注入 S/R 与价格行为特征：price_action 缺 sr 时
    broken_resistance/support 恒为 None，作为子策略时一个信号都发不出来。
    """
    from . import get_dynamic
    out = []
    for n in names or []:
        spec = get_dynamic(n) or {}
        out.append(str(spec.get("executor") or n))
    return out


def _resolve_sub_strategies(names) -> tuple[list[str], list[Strategy]]:
    """按名去重实例化子策略，返回 (可用名, 实例)。全不可用时抛 ValueError。

    训练端每个 MetaControllerEnv 都调它自建实例：子策略是带内部状态的
    （prev MA、_entry…），共享实例在并行轨迹收集中会互相踩状态。
    """
    from . import get_strategy  # 延迟导入避免循环依赖
    uniq = [n.strip() for n in (names or []) if n and n.strip()]
    resolved: list[str] = []
    insts: list[Strategy] = []
    for name in dict.fromkeys(uniq):
        try:
            st = get_strategy(name)
            st.reset()
        except ValueError as e:
            log.warning("[meta] 子策略 %s 不可用: %s", name, e)
            continue
        resolved.append(name)
        insts.append(st)
    if not insts:
        raise ValueError("没有可用的子策略，无法训练元控制器")
    return resolved, insts


class MetaController(Strategy):
    """元策略控制器：在多个子策略间动态选择/加权合成。

    参数:
        sub_strategies: 逗号分隔的子策略名列表（如 "dual_ma,price_action,factor_signal"）
        mode: "ensemble" 置信度加权投票 | "drl" DRL 动态选择
        model_path: DRL 模式下的模型文件路径
        meta_window: 子策略近期表现跟踪窗口（用于 DRL 状态）
        drl_ensemble_threshold: DRL 模式下，选择置信度低于此阈值时回退到集成投票
        ensemble_buy_threshold: 集成模式买入阈值（加权和 > 此值做多）
        ensemble_sell_threshold: 集成模式卖出阈值（加权和 < 此值做空）
    """
    name = "meta_controller"
    description = "元策略控制器：在多个子策略（趋势/突破/因子/RL）间动态选择或加权合成，统一所有策略信号"
    default_params = {
        "sub_strategies": "dual_ma,price_action,factor_signal",
        "mode": "ensemble",  # ensemble | drl
        "model_path": "",
        "meta_window": _DEFAULT_META_WINDOW,
        "drl_ensemble_threshold": 0.3,
        "ensemble_buy_threshold": 0.3,
        "ensemble_sell_threshold": -0.3,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "size_pct": 0.5,
    }
    param_schema = {
        "sub_strategies": {"type": "str", "label": "子策略列表（逗号分隔，如 dual_ma,price_action）"},
        "mode": {"type": "str", "choices": ["ensemble", "drl"], "label": "模式(集成/DRL)"},
        "model_path": {"type": "str", "label": "DRL 模型路径（drl 模式需要）"},
        "meta_window": {"type": "int", "min": 5, "max": 100, "label": "表现跟踪窗口"},
        "drl_ensemble_threshold": {"type": "float", "min": 0.0, "max": 1.0, "label": "DRL 回退阈值"},
        "ensemble_buy_threshold": {"type": "float", "min": -1.0, "max": 1.0, "label": "集成买入阈值"},
        "ensemble_sell_threshold": {"type": "float", "min": -1.0, "max": 1.0, "label": "集成卖出阈值"},
        "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "止损比例"},
        "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.5, "label": "止盈比例"},
        "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
    }

    def reset(self) -> None:
        """重置所有子策略和元控制器状态。"""
        self._sub_strategies: list[Strategy] = []
        self._strategy_names: list[str] = []
        self._entry: Optional[float] = None
        # 子策略近期表现跟踪
        self._strategy_trades: dict[str, list[float]] = {}  # name -> [pnl, ...]
        self._strategy_signals: dict[str, Optional[Signal]] = {}  # 当前K线各子策略信号
        # DRL 模式
        self._meta_agent: Any = None
        self._meta_state_dim = 0
        self._last_signals: list[float] = []  # 最近信号记录（用于 DRL 状态）
        # 外部因子列（可选，与 rl_adaptive 一致）
        self._factor_mu: Optional[float] = None
        self._factor_sd: Optional[float] = None
        self._factor_expr: str = ""
        self._closes: list[float] = []
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []

    def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """热更新参数；子策略列表/模式/模型路径变更时重建运行态。

        基类只写 params，而子策略实例在首次 on_candle 时才按 sub_strategies 构建——
        不重建的话改了 sub_strategies 仍跑旧的那几个策略。_entry 保留，
        避免调参时丢掉持仓的止损止盈跟踪。
        """
        old_subs = str(self.params.get("sub_strategies", ""))
        old_mode = str(self.params.get("mode", ""))
        old_model = str(self.params.get("model_path", ""))
        applied = super().update_params(params)
        if str(applied.get("sub_strategies", "")) != old_subs:
            self._init_sub_strategies()
            self._strategy_trades = {n: [] for n in self._strategy_names}
            self._strategy_signals = {}
            self._last_signals = []
        if (str(applied.get("mode", "")) != old_mode
                or str(applied.get("model_path", "")) != old_model):
            self._meta_agent = None
            self._meta_state_dim = 0
        return applied

    def _init_sub_strategies(self) -> None:
        """解析子策略名列表并实例化。"""
        raw = str(self.params.get("sub_strategies", "") or "").strip()
        if not raw:
            self._strategy_names = []
            self._sub_strategies = []
            return
        names = [n.strip() for n in raw.split(",") if n.strip()]
        self._strategy_names = list(dict.fromkeys(names))  # 去重但保持顺序
        self._sub_strategies = []
        from . import get_strategy  # 延迟导入避免循环依赖
        for name in self._strategy_names:
            try:
                st = get_strategy(name)
                st.reset()
                self._sub_strategies.append(st)
                # 初始化表现跟踪
                if name not in self._strategy_trades:
                    self._strategy_trades[name] = []
                log.info("[meta] 子策略 %s 已加载", name)
            except ValueError as e:
                log.warning("[meta] 子策略 %s 加载失败: %s", name, e)

    @property
    def managed_executors(self) -> list[str]:
        """子策略对应的 executor 名（从 params 推导，不依赖已实例化的子策略）。"""
        raw = str(self.params.get("sub_strategies", "") or "").strip()
        names = list(dict.fromkeys(n.strip() for n in raw.split(",") if n.strip()))
        return resolve_sub_executors(names)

    def _meta_window(self) -> int:
        """表现跟踪窗口（胜率/近期均值）；非法值回退默认，下界与 param_schema 一致。"""
        try:
            return max(5, int(self.params.get("meta_window", _DEFAULT_META_WINDOW)))
        except (TypeError, ValueError):
            return _DEFAULT_META_WINDOW

    def reload_model(self) -> bool:
        """元模型文件被回退/替换后重载运行实例（返回是否加载成功）。

        清空 DRL 模型缓存与状态维度后从当前 model_path 重新加载；
        失败时 _meta_agent 保持 None，让上层按运行时失败处理。
        """
        self._meta_agent = None
        self._meta_state_dim = 0
        return self._load_meta_agent()

    def _load_meta_agent(self) -> bool:
        """加载 DRL 元模型（仅 drl 模式）。"""
        path = str(self.params.get("model_path", "") or "").strip()
        if not path:
            return False
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        if not os.path.exists(path):
            log.warning("[meta] 元模型不存在: %s", path)
            return False
        try:
            from drl.agent import ACAgent
            self._meta_agent = ACAgent.load(path)
            self._meta_state_dim = self._meta_agent.state_dim
            return True
        except Exception as e:
            log.warning("[meta] 元模型加载失败: %s", e)
            return False

    def _factor_value(self) -> Optional[float]:
        """计算因子信号列（与 rl_adaptive 同口径）。"""
        if not self._factor_expr or len(self._closes) < 5:
            return None
        from factors.mining import FactorExecutor
        import pandas as pd
        df = pd.DataFrame({
            "open": self._opens, "high": self._highs, "low": self._lows,
            "close": self._closes, "volume": self._volumes,
        })
        try:
            s = FactorExecutor(self._factor_expr).eval(df).astype(float)
            v = s.iloc[-1]
            if v is None or v != v:
                return None
            v = float(v)
            if self._factor_mu is not None and self._factor_sd:
                v = (v - self._factor_mu) / self._factor_sd
            return v
        except Exception:
            return None

    def _collect_sub_signals(self, ctx: dict) -> dict[str, Any]:
        """收集所有子策略的当前信号，返回 {name: signal}。"""
        result: dict[str, Any] = {}
        self._strategy_signals = {}
        for i, name in enumerate(self._strategy_names):
            st = self._sub_strategies[i]
            try:
                sig = st.on_candle(ctx)
                self._strategy_signals[name] = sig
                result[name] = {
                    "signal": sig,
                    "confidence": sig.confidence if sig else 0.0,
                    "direction": 1.0 if sig and sig.side == "buy" else (-1.0 if sig and sig.side == "sell" else 0.0),
                }
            except Exception as e:
                log.warning("[meta] 子策略 %s 信号异常: %s", name, e)
                self._strategy_signals[name] = None
                result[name] = {"signal": None, "confidence": 0.0, "direction": 0.0}
        return result

    def _compute_ensemble_signal(self, sub_results: dict) -> tuple[float, float, str]:
        """置信度加权投票合成最终信号。

        返回 (weighted_sum, total_confidence, detail_str)。
        """
        total = 0.0
        total_conf = 0.0
        details = []
        for name, info in sub_results.items():
            direction = info["direction"]
            conf = info["confidence"]
            total += direction * conf
            total_conf += conf
            if direction != 0:
                details.append(f"{name}({direction:.0f}×{conf:.2f})")
        detail_str = " + ".join(details) if details else "无信号"
        return total, total_conf, detail_str

    def _build_meta_state(self, ctx: dict, sub_results: dict) -> Optional[np.ndarray]:
        """构建 DRL 元策略状态向量（与训练端共用 build_meta_state，逐元素同构）。

        状态组成：
        - 每个子策略的：当前信号方向(-1/0/1) + 置信度 + 近期胜率 + 近期平均收益
        - 全局：当前持仓比 + 浮动盈亏 + 实现波动率
        - 可选因子列
        """
        price = float(ctx["price"])
        position = float(ctx.get("position", 0.0))
        cash = float(ctx.get("cash", 0.0))
        equity = cash + position * price
        pos_ratio = (position * price) / equity if equity > 0 else 0.0
        pnl_ratio = 0.0
        if self._entry and self._entry > 0:
            pnl_ratio = (price - self._entry) / self._entry

        state = build_meta_state(self._strategy_names, sub_results, self._strategy_trades,
                                 meta_window=self._meta_window(), pos_ratio=pos_ratio,
                                 pnl_ratio=pnl_ratio, vol_closes=self._closes)

        # 因子列（可选）
        if self._meta_agent and self._meta_agent.state_dim > len(state):
            fv = self._factor_value()
            if fv is not None:
                state = np.concatenate([state, [fv]])
            else:
                state = np.concatenate([state, [0.0]])

        return state

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        price = float(ctx["price"])
        ind = ctx.get("indicators", {})
        position = float(ctx.get("position", 0.0))
        cash = float(ctx.get("cash", 0.0))
        symbol = ctx["symbol"]
        equity = cash + position * price

        # 累计OHLCV历史
        close = float(ind.get("close", price) or price)
        volume = float(ind.get("volume", 0.0) or 0.0)
        high = float(ind.get("high", price) or price)
        low = float(ind.get("low", price) or price)
        open_ = float(ind.get("open", price) or price)
        self._closes.append(close)
        self._opens.append(open_)
        self._volumes.append(volume)
        self._highs.append(high)
        self._lows.append(low)
        if len(self._closes) > 400:
            self._closes = self._closes[-400:]
            self._opens = self._opens[-400:]
            self._volumes = self._volumes[-400:]
            self._highs = self._highs[-400:]
            self._lows = self._lows[-400:]

        # 首次运行：初始化子策略
        if not self._sub_strategies:
            self._init_sub_strategies()
        if not self._strategy_names:
            return None

        # 止损止盈
        if position > 0 and self._entry:
            chg = (price - self._entry) / self._entry
            if chg <= -self.params["stop_loss_pct"]:
                self._entry = None
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"元策略止损 {chg:.2%}")
            if chg >= self.params["take_profit_pct"]:
                self._entry = None
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"元策略止盈 {chg:.2%}")

        # 收集子策略信号
        sub_results = self._collect_sub_signals(ctx)

        mode = str(self.params.get("mode", "ensemble"))

        if mode == "drl":
            # ---- DRL 模式：用元模型动态选择子策略 ----
            if self._meta_agent is None:
                if not self._load_meta_agent():
                    # 模型加载失败，回退到集成模式
                    log.warning("[meta] DRL 模型未加载，回退到集成模式")
                    mode = "ensemble"
                else:
                    # 加载因子元数据（从模型文件）
                    try:
                        model_path = str(self.params.get("model_path", "") or "").strip()
                        if model_path and os.path.exists(model_path):
                            import json
                            with open(model_path, "r", encoding="utf-8") as f:
                                md = json.load(f)
                            self._factor_mu = md.get("factor_mu")
                            self._factor_sd = md.get("factor_sd")
                            self._factor_expr = str(md.get("factor_expression", "") or "")
                    except Exception:
                        pass

            if self._meta_agent is not None:
                state = self._build_meta_state(ctx, sub_results)
                if state is not None and len(state) == self._meta_state_dim:
                    action = self._meta_agent.greedy_action(state)
                    # 如果 DRL 选择置信度太低，回退到集成
                    proba = self._meta_agent.actor.predict_proba(state.reshape(1, -1))[0]
                    action_conf = float(proba[action])
                    drl_threshold = float(self.params.get("drl_ensemble_threshold", 0.3))
                    if action_conf < drl_threshold:
                        # 回退到集成
                        weighted_sum, total_conf, _ = self._compute_ensemble_signal(sub_results)
                        action = 1 if weighted_sum > self.params["ensemble_buy_threshold"] else (
                            -1 if weighted_sum < self.params["ensemble_sell_threshold"] else 0
                        )
                        reason = f"元策略DRL回退(conf={action_conf:.2f}), 加权={weighted_sum:.2f}"
                    else:
                        # DRL 选择：0=清仓, 1=买入, 2=卖出
                        action_map = {0: 0, 1: 1, 2: -1}
                        mapped = action_map.get(action, 0)
                        reason = f"元策略DRL选择子策略#{action}(conf={action_conf:.2f})"
                        action = mapped
                else:
                    weighted_sum, total_conf, _ = self._compute_ensemble_signal(sub_results)
                    action = 1 if weighted_sum > self.params["ensemble_buy_threshold"] else (
                        -1 if weighted_sum < self.params["ensemble_sell_threshold"] else 0
                    )
                    reason = f"元策略状态不匹配，回退集成"
            else:
                return None
        else:
            # ---- 集成模式：置信度加权投票 ----
            weighted_sum, total_conf, detail = self._compute_ensemble_signal(sub_results)
            action = 1 if weighted_sum > self.params["ensemble_buy_threshold"] else (
                -1 if weighted_sum < self.params["ensemble_sell_threshold"] else 0
            )
            reason = f"元策略集成({detail}), 加权={weighted_sum:.2f}"

        # ---- 执行决策 ----
        if action == 1 and equity > 0 and cash > 0:
            # 买入
            size = float(self.params.get("size_pct", 0.5))
            buy_value = size * equity
            qty = min(buy_value / price, cash / price)
            if qty > 0:
                return Signal(symbol, "buy", size_pct=size, qty=qty,
                              strategy=self.name, reason=reason)
        elif action == -1 and position > 0:
            # 卖出
            return Signal(symbol, "sell", 1.0, strategy=self.name, reason=reason)

        return None

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        # 成交回报透传子策略：它们各自的 _entry 决定止损止盈/关键位离场是否生效
        for st in self._sub_strategies:
            try:
                st.on_fill(symbol, side, price)
            except Exception as e:  # noqa: BLE001
                log.warning("[meta] 子策略 on_fill 异常: %s", e)
        if side == "buy":
            self._entry = price
        else:
            # 卖出成交后，记录本次交易的收益到各子策略的胜率跟踪
            # 注意：这里是简化的跟踪，每个子策略都记同一个收益
            if self._entry and self._entry > 0:
                pnl = (price - self._entry) / self._entry
                window = self._meta_window()
                for name in self._strategy_names:
                    record_meta_trade_result(
                        self._strategy_trades.setdefault(name, []), pnl, window)
            self._entry = None


# ============ DRL 元策略训练 ============

class MetaControllerEnv:
    """元策略训练环境（gym 风格）：闭环推进，与实盘 MetaController 同构。

    在历史数据上逐K线运行子策略，让 DRL 智能体学习"在什么市场状态下
    应该信任哪个子策略"。三条同构约束（都曾不满足，导致模型部署即失效）：

    1. 子策略实例由环境自建，绝不跨环境共享——它们是带内部状态的有状态对象，
       并行收集时共享实例会互相污染 prev 值/入场价。
    2. 喂给子策略的是真实指标快照（precompute_indicator_series + snapshot_at），
       曾在构造时用 open=high=low=close=price、volume=0 的退化 ctx 预扫描全量K线。
    3. position/cash 用环境自己的账户实测值（闭环）。曾恒传 0/满仓现金，而
       dual_ma/grid/price_action 的卖出分支都要求 position>0 → 训练里只见过买入信号。

    注：本环境以收盘价成交（DRL 环境的通用简化），撮合口径与回测内核
    （次根开盘成交）的差异属 P1-6 范围，此处不改动。
    """

    def __init__(self, df, strategy_names,
                 *,
                 warmup: int = 60,
                 start_cash: float = 10000.0,
                 fee_rate: float = 0.001,
                 meta_window: int = _DEFAULT_META_WINDOW,
                 symbol: str = "META",
                 timeframe: str = ""):
        self.df = df
        self.strategy_names, self.sub_strategies = _resolve_sub_strategies(strategy_names)
        self.n_strategies = len(self.strategy_names)
        self.warmup = warmup
        self.start_cash = start_cash
        self.fee_rate = fee_rate
        self.meta_window = max(5, int(meta_window))
        self.symbol = symbol
        self.timeframe = timeframe
        self.n = len(df)

        # 状态维度：每子策略 (方向 + 置信度 + 胜率 + 近期均值) + 全局(持仓比+盈亏+波动率)
        self._state_dim = meta_state_dim(self.n_strategies)
        # 动作：0=清仓, 1=买入, 2=卖出
        self._n_actions = 3
        # 运行态（reset 会重置；先建好，避免 _collect 读到未定义属性）
        self._cash = start_cash
        self._qty = 0.0
        self._entry_price = None
        self._t = min(warmup, max(0, self.n - 2))
        self._strategies_trades: dict[str, list[float]] = {n: [] for n in self.strategy_names}
        self._sigs_t = -1
        self._sigs = None

        # 指标序列向量化预计算一次（与回测引擎同一函数、同一 MA 口径）
        self._closes = df["close"].to_numpy(float)
        opens = df["open"].to_numpy(float) if "open" in df.columns else self._closes
        highs = df["high"].to_numpy(float) if "high" in df.columns else self._closes
        lows = df["low"].to_numpy(float) if "low" in df.columns else self._closes
        vols = df["volume"].to_numpy(float) if "volume" in df.columns else np.zeros(self.n)
        from indicators.vectorized import precompute_indicator_series, snapshot_at
        self._snapshot_at = snapshot_at
        self._series = precompute_indicator_series(self._closes, opens, highs, lows, vols)
        # price_action 子策略缺 sr/pa 特征时恒不产出信号（突破/回调判据全靠关键位）
        self._sr_series = self._pa_series = None
        if "price_action" in resolve_sub_executors(self.strategy_names):
            from indicators.technical import (SR_MIN_TOUCHES, SR_WINDOW,
                                              price_action_features_series,
                                              support_resistance_series)
            self._sr_series = support_resistance_series(highs, lows, self._closes,
                                                        window=SR_WINDOW,
                                                        min_touches=SR_MIN_TOUCHES)
            self._pa_series = price_action_features_series(highs, lows, self._closes,
                                                           opens, vols)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def _collect(self, t: int) -> dict[str, dict]:
        """在 t 根K线收盘处调用各子策略（ctx 与回测/实盘同形状，无前视）。"""
        price = float(self._closes[t])
        ind = self._snapshot_at(self._series, t)
        if self._sr_series is not None:
            ind["sr"] = self._sr_series[t]
            ind["pa"] = self._pa_series[t]
        ctx = {
            "symbol": self.symbol, "price": price,
            "position": self._qty, "cash": self._cash,
            "indicators": ind, "timeframe": self.timeframe,
        }
        signals: dict[str, dict] = {}
        for name, st in zip(self.strategy_names, self.sub_strategies):
            try:
                sig = st.on_candle(ctx)
                signals[name] = {
                    "direction": 1.0 if sig and sig.side == "buy" else (
                        -1.0 if sig and sig.side == "sell" else 0.0),
                    "confidence": sig.confidence if sig else 0.0,
                }
            except Exception as e:  # noqa: BLE001
                log.warning("[meta_env] 子策略 %s 信号异常: %s", name, e)
                signals[name] = {"direction": 0.0, "confidence": 0.0}
        return signals

    def _signals_at(self, t: int) -> dict[str, dict]:
        """推进子策略到 t 并返回该处信号。

        子策略有状态且只按K线顺序推进，故 t 必须单调递增（本环境满足：
        reset 落在 warmup，step 每次 +1）。暖机段（t < warmup）同样要喂，
        否则 MA/RSI 的 prev 值缺失，与实盘从第 0 根起跑的口径不一致。
        """
        if self._sigs is not None and self._sigs_t == t:
            return self._sigs
        while self._sigs_t < t:
            self._sigs_t += 1
            self._sigs = self._collect(self._sigs_t)
        return self._sigs or {}

    def _build_state(self, t: int) -> np.ndarray:
        """构建 t 时的状态（与实盘 MetaController._build_state 同一构造函数）。"""
        sigs = self._signals_at(t)
        price = float(self._closes[t])
        equity = self._equity(price)
        parts_price = self._qty * price
        pnl_ratio = ((price - self._entry_price) / self._entry_price
                     if self._entry_price and self._entry_price > 0 else 0.0)
        return build_meta_state(self.strategy_names, sigs, self._strategies_trades,
                                meta_window=self.meta_window,
                                pos_ratio=(parts_price / equity) if equity > 0 else 0.0,
                                pnl_ratio=pnl_ratio,
                                vol_closes=self._closes[max(0, t - _META_VOL_WINDOW):t + 1])

    def _notify_fill(self, side: str, price: float) -> None:
        """成交回报透传子策略（与 MetaController.on_fill 同构）。"""
        for st in self.sub_strategies:
            try:
                st.on_fill(self.symbol, side, float(price))
            except Exception:  # noqa: BLE001
                pass

    def _close_position(self, price_now: float) -> None:
        """按现价清仓，并把这笔收益记进各子策略的表现跟踪。"""
        sell_value = self._qty * price_now
        self._cash += sell_value - sell_value * self.fee_rate
        if self._entry_price and self._entry_price > 0:
            pnl = (price_now - self._entry_price) / self._entry_price
            for name in self.strategy_names:
                record_meta_trade_result(self._strategies_trades.setdefault(name, []),
                                         pnl, self.meta_window)
        self._qty = 0.0
        self._entry_price = None
        self._notify_fill("sell", price_now)

    def _equity(self, price: float) -> float:
        return float(self._cash + self._qty * price)

    def _pos_ratio(self) -> float:
        price = float(self._closes[self._t])
        equity = self._equity(price)
        return float(self._qty * price / equity) if equity > 0 else 0.0

    def reset(self) -> np.ndarray:
        for st in self.sub_strategies:
            st.reset()
        self._t = self.warmup
        self._cash = self.start_cash
        self._qty = 0.0
        self._entry_price: Optional[float] = None
        self._prev_equity = self.start_cash
        self._final_equity = self.start_cash
        self._final_position_ratio = 0.0
        self._done = False
        self._strategies_trades: dict[str, list[float]] = {n: [] for n in self.strategy_names}
        # 子策略推进游标：暖机段（0..warmup-1）也要按顺序喂进去，状态才与实盘一致
        self._sigs_t = -1
        self._sigs = None
        self._last_trade_time = 0
        return self._build_state(self._t)

    def step(self, action: int) -> tuple:
        """执行动作：0=清仓, 1=买入, 2=卖出。"""
        assert not self._done, "环境已终止"

        price_now = self.df["close"].iloc[self._t]
        price_next = self.df["close"].iloc[self._t + 1] if self._t + 1 < self.n else price_now

        # 执行动作
        if action == 1 and self._qty <= 0:  # 买入
            buy_value = self._cash * 0.5
            cost = buy_value * self.fee_rate
            self._cash -= (buy_value + cost)
            self._qty = buy_value / price_now
            self._entry_price = price_now
            self._notify_fill("buy", price_now)
        elif action == 2 and self._qty > 0:  # 卖出（清仓）
            self._close_position(price_now)
        elif action == 0:  # 清仓
            if self._qty > 0:
                self._close_position(price_now)

        # 推进到下一根K线
        self._t += 1
        if self._t >= self.n - 1:
            self._t = self.n - 2
            self._done = True

        equity_next = self._cash + self._qty * price_next
        ret = (equity_next / max(self._prev_equity, 1e-9)) - 1.0
        self._final_equity = float(equity_next)
        self._final_position_ratio = float(self._qty * price_next / max(equity_next, 1e-9))
        # 奖励 = 收益率（bp），软裁剪
        reward = float(150.0 * np.tanh(ret * 10000.0 / 150.0))
        self._prev_equity = equity_next

        info = {
            "equity": equity_next,
            "ret_pct": ret * 100.0,
        }
        return self._build_state(self._t), reward, self._done, info


def train_meta_controller(df, cfg: dict,
                          on_progress: Optional[callable] = None) -> dict:
    """训练 DRL 元策略控制器（训练/验证/OOS 三段切分）。

    在历史数据上运行所有子策略，让 DRL 智能体学会在它们之间动态选择。
    产出管理子策略的元级智能体模型。

    三段口径与 drl.agent.train_drl 一致（进化引擎共用同一道部署硬门）：
    训练段拟合策略、验证段按确定性收益选最优模型、OOS 段做无人见过行情
    的独立验收。此前元控制器只在整段数据上取"训练集内均值收益的最大值"
    作为 best_ret 并直接注册为实盘策略，等于零样本外证据上线。

    cfg:
        sub_strategies: 子策略名列表（逗号分隔）
        episodes: PPO 训练轮数
        hidden: 网络隐藏层
        lr_actor / lr_critic: 学习率
        seed: 随机种子（None=每轮随机）
        fee_rate / start_cash / meta_window / warmup
        train_ratio / val_ratio: 切分比例（其余给 OOS）
        val_eval_interval: 每几轮做一次验证评估
        oos_hard_gate: OOS 衰减过半是否拦截自动部署（默认 True）
        symbol / timeframe: 透传给子策略 ctx
    返回新增：best_val_ret / final_ret / oos_report / deployment_blocked /
    split / （数据不足时）split_skipped
    """
    t0 = time.time()
    from drl.agent import ACAgent, _equity_drawdown, _equity_sharpe, _oos_decay
    from drl.env import run_episode

    episodes = max(1, int(cfg.get("episodes", 60)))
    hidden = tuple(cfg.get("hidden", [32, 32]))
    n_episodes = max(1, int(cfg.get("n_episodes", 4)))
    ppo_epochs = max(1, int(cfg.get("ppo_epochs", 4)))
    mini_batch_size = max(8, int(cfg.get("mini_batch_size", 64)))
    seed_cfg = cfg.get("seed", 42)
    seed = int(seed_cfg) if seed_cfg is not None else random.randint(0, 99999)
    meta_window = max(5, int(cfg.get("meta_window", _DEFAULT_META_WINDOW)))
    fee_rate = float(cfg.get("fee_rate", 0.001))
    start_cash = float(cfg.get("start_cash", 10000.0))
    warmup = int(cfg.get("warmup", 60))

    # 子策略可用性校验；实例由每个环境自建（见 MetaControllerEnv 同构约束 1）
    sub_names = [n.strip() for n in str(cfg.get("sub_strategies", "dual_ma,price_action")).split(",") if n.strip()]
    sub_names, _ = _resolve_sub_strategies(sub_names)

    # ---- 训练 / 验证 / OOS 三段切分（互不重叠，杜绝前视） ----
    n_total = len(df)
    min_seg = max(80, warmup + 40)
    if n_total < min_seg:
        raise ValueError(f"元控制器训练数据不足：需要至少 {min_seg} 根K线，实际 {n_total}")
    n_train = int(n_total * float(cfg.get("train_ratio", 0.6)))
    n_val = int(n_total * float(cfg.get("val_ratio", 0.2)))
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    oos_df = df.iloc[n_train + n_val:]
    split_skipped = False
    if min(len(train_df), len(val_df), len(oos_df)) < min_seg:
        # 数据不足以切三段：整段训练，无验证选模、无样本外证据。
        # 不静默放行——oos_report.enabled=False 会被部署硬门拦下（同 train_drl）
        train_df = val_df = df
        oos_df = df.iloc[:0]
        split_skipped = True
        log.warning("[meta_train] 数据不足（%d 根），三段切分退化为整段训练，"
                    "模型将因缺少样本外证据被部署硬门拦截", n_total)

    env_kwargs = dict(warmup=warmup, start_cash=start_cash, fee_rate=fee_rate,
                      meta_window=meta_window, symbol=str(cfg.get("symbol", "META")),
                      timeframe=str(cfg.get("timeframe", "") or ""))

    def _make_env(seg):
        return MetaControllerEnv(seg, sub_names, **env_kwargs)

    state_dim = meta_state_dim(len(sub_names))
    agent = ACAgent(state_dim, 3, hidden=hidden,
                    lr_actor=float(cfg.get("lr_actor", 3e-3)),
                    lr_critic=float(cfg.get("lr_critic", 6e-3)),
                    gamma=float(cfg.get("gamma", 0.99)), seed=seed,
                    entropy_coef=float(cfg.get("entropy_coef", 0.05)))
    # 验证环境全程复用：run_episode 内部 reset 会把子策略退回起点重推
    val_env = None if split_skipped else _make_env(val_df)

    # 训练循环（PPO 批量收集 + 多轮 mini-batch）
    history: list[dict] = []
    best_train_ret = -1e9
    best_val_ret = -1e9
    best_agent: Optional[ACAgent] = None
    val_eval_interval = max(1, int(cfg.get("val_eval_interval", 5)))
    total_ret = 0.0

    with ThreadPoolExecutor(max_workers=min(n_episodes, 4)) as pool:
        for ep in range(1, episodes + 1):
            agent.set_episode_epsilon(ep - 1)

            def _collect_one(args: tuple) -> dict:
                ep_i, worker_i = args
                local_env = _make_env(train_df)
                local_rng = np.random.default_rng(seed + ep_i * 1000 + worker_i)
                return agent.collect_episode(local_env, rng=local_rng)

            trajs = list(pool.map(_collect_one, [(ep, i) for i in range(n_episodes)]))
            states = np.concatenate([t["states"] for t in trajs])
            actions = np.concatenate([t["actions"] for t in trajs])
            rewards = np.concatenate([t["rewards"] for t in trajs])
            old_log_probs = np.concatenate([t["old_log_probs"] for t in trajs])
            stats = agent.train_batch(states, actions, rewards, old_log_probs,
                                      ppo_epochs=ppo_epochs,
                                      mini_batch_size=min(mini_batch_size, len(states)),
                                      segment_lengths=[t["steps"] for t in trajs],
                                      actor_grad_clip=float(cfg.get("actor_grad_clip", 1.0)))
            total_ret = float(np.mean([t["total_ret"] for t in trajs]))
            if total_ret > best_train_ret:
                best_train_ret = total_ret

            row = {
                "episode": ep,
                "total_ret": round(total_ret, 6),
                "policy_loss": round(stats["policy_loss"], 6),
                "value_loss": round(stats["value_loss"], 6),
                "epsilon": stats["epsilon"],
            }
            if split_skipped:
                row["val_skipped"] = True

            # 周期性用确定性策略在验证段评估，并按验证收益挑选最优模型
            # （训练集收益带探索噪声，按它选优等于选到"运气最好的那条随机轨迹"）
            if val_env is not None and (ep % val_eval_interval == 0 or ep == episodes):
                try:
                    vtraj = run_episode(val_env, lambda s: agent.greedy_action(s))
                    val_ret = float(vtraj["total_ret"])
                    row["val_ret"] = round(val_ret, 6)
                    v_eq = [i.get("equity") for i in vtraj.get("infos", [])
                            if i.get("equity") is not None]
                    if len(v_eq) > 2:
                        row["val_sharpe"] = round(_equity_sharpe(v_eq), 4)
                        row["val_max_dd"] = round(_equity_drawdown(v_eq), 4)
                    if val_ret > best_val_ret:
                        best_val_ret = val_ret
                        # P2：JSON 往返改 deepcopy（快照不经过全量序列化）
                        best_agent = copy.deepcopy(agent)
                except Exception as e:  # noqa: BLE001
                    log.warning("[meta_train] 验证失败: %s", e)

            row["best_ret"] = round(max(best_train_ret, best_val_ret), 6)
            history.append(row)

            if on_progress:
                try:
                    on_progress({
                        "episode": ep, "episodes": episodes,
                        "total_ret": round(total_ret, 6),
                        "best_ret": row["best_ret"],
                        "val_ret": row.get("val_ret"),
                        "policy_loss": stats["policy_loss"],
                        "elapsed_sec": round(time.time() - t0, 1),
                    })
                except Exception as e:
                    log.warning("[meta_train] 进度回调异常: %s", e)

    final_agent = best_agent or agent
    final_agent.epsilon = final_agent.epsilon_min

    # ============ OOS 独立评估（无人见过的一段行情，部署硬门的依据） ============
    oos_report: dict = {"enabled": False}
    deployment_blocked = False
    if len(oos_df) >= min_seg:
        try:
            oos_env = _make_env(oos_df)
            otraj = run_episode(oos_env, lambda s: final_agent.greedy_action(s))
            oos_ret = float(otraj["total_ret"])
            train_ref = best_train_ret if best_train_ret > -1e8 else total_ret
            decay = _oos_decay(train_ref, oos_ret)
            o_eq = [i.get("equity") for i in otraj.get("infos", [])
                    if i.get("equity") is not None]
            oos_report = {
                "enabled": True,
                "oos_ret": round(oos_ret, 6),
                "train_ret": round(train_ref, 6),
                "val_ret": round(best_val_ret, 6) if best_val_ret > -1e8 else None,
                "decay": round(decay, 4),
                "overfit_likely": bool(decay > 0.5),
                "oos_equity_final": round(float(otraj["final_equity"]), 2),
                "oos_position_ratio": round(float(otraj["final_position_ratio"]), 4),
                "oos_sharpe": round(_equity_sharpe(o_eq), 4) if len(o_eq) > 2 else 0.0,
                "oos_max_drawdown": round(_equity_drawdown(o_eq), 4) if len(o_eq) > 2 else 0.0,
            }
            if bool(cfg.get("oos_hard_gate", True)) and oos_report["overfit_likely"]:
                oos_report["hard_rejected"] = True
                oos_report["reason"] = (
                    f"OOS 收益 {oos_ret:.4f} 较训练段 {train_ref:.4f} 衰减 {decay:.1%}，"
                    f"疑似过拟合，已拦截自动部署（可关闭 oos_hard_gate 强制产出）")
                deployment_blocked = True
        except Exception as e:  # noqa: BLE001
            log.warning("[meta_train] OOS 评估失败: %s", e)
            oos_report = {"enabled": False}

    best_ret = max(best_train_ret, best_val_ret)
    result = {
        "agent": final_agent,
        "history": history,
        "best_ret": round(best_ret, 6),
        "best_val_ret": round(best_val_ret, 6) if best_val_ret > -1e8 else None,
        "final_ret": round(total_ret, 6),
        "sub_strategies": sub_names,
        "state_dim": state_dim,
        "n_actions": 3,
        "algo": "ppo",
        "episodes": episodes,
        "split": {"train": len(train_df), "val": len(val_df), "oos": len(oos_df),
                  "warmup": warmup},
        "oos_report": oos_report,
        "deployment_blocked": deployment_blocked,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if split_skipped:
        result["split_skipped"] = True
    return result