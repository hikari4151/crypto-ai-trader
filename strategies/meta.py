"""元策略控制器：统一多个子策略信号的智能调度器。

两种模式：
1. **ensemble（集成模式）**：置信度加权投票，按子策略的置信度加权合成最终信号。
   无需训练，立即可用。
2. **drl（强化学习模式）**：加载**既有的**已训练元控制器模型，根据市场状态、
   子策略信号和历史表现动态选择最优子策略。

注：元策略**训练**功能已移除（2026-09-28）——训练环境 `MetaControllerEnv`、
`_meta_env_kwargs`、`train_meta_controller` 及进化引擎的训练管线、手动触发端点
均已删除。drl 模式仍可加载此前训练好的模型文件（model_path）运行。

设计目标：
- 所有现有策略（dual_ma / grid / price_action / factor_signal / rl_adaptive）
  都作为"子策略"被管理
- 元策略在子策略之上做决策，不修改子策略本身
- 回测引擎只需要运行元策略一个策略实例
"""
import logging
import os
from typing import Any, Optional

import numpy as np

from .base import Signal, Strategy

log = logging.getLogger(__name__)

# ============ 元控制器状态向量（单一事实来源） ============
# 历史上训练端（已随元策略训练于 2026-09-28 移除）曾把这三个全局特征写死 [0,0,0]，
# 且喂给子策略的是 open=high=low=close=price、volume=0 的退化快照。维度对得上、
# 数值全不对，训出来的 DRL 模型部署即失配（实盘 len(state) != state_dim 就静默
# 回退集成），所谓"智能调度"从未真正生效。因此状态拼装只允许走 build_meta_state
# 这一条路径——该约束现在只约束实盘端。

_META_VOL_WINDOW = 20            # 实现波动率回看K线数
_META_VOL_SCALE = 0.02           # 归一尺度：2% 的 bar 收益标准差记满 1.0
_DEFAULT_META_WINDOW = 20        # 子策略表现跟踪窗口


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
                     vol_closes, vol_feature: Optional[float] = None) -> np.ndarray:
    """拼装元控制器状态向量。

    历史上由训练环境（已随元策略训练于 2026-09-28 移除）与实盘 MetaController
    共用；现仅实盘端调用。

    signals: {子策略名: {"direction": ±1/0, "confidence": float}}
    trades_by_name: {子策略名: [已平仓收益率, ...]}
    vol_closes: 截至当前K线的收盘序列尾部（含当前根）
    vol_feature: 预计算的波动率特征值。训练端整列预计算后查表传入的做法
        已随训练功能移除；实盘端不传，回落 meta_vol_feature(vol_closes)
        （.optim/probe_meta_vol.py：5000 点含边界 bitwise PASS）。
    """
    parts: list[float] = []
    for name in strategy_names:
        info = signals.get(name) or {}
        direction = float(info.get("direction", 0.0) or 0.0)
        conf = float(info.get("confidence", 0.0) or 0.0)
        win_rate, recent_avg = meta_performance(trades_by_name.get(name), meta_window)
        parts.extend([direction, conf, win_rate, recent_avg])
    if vol_feature is None:
        vol_feature = meta_vol_feature(vol_closes)
    parts.extend([float(pos_ratio), float(pnl_ratio), float(vol_feature)])
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
        "per_strategy_trades": {"type": "bool", "label": "按子策略独立记战绩（训练/部署同构）"},
        "signal_novelty_filter": {"type": "bool", "label": "子策略信号新颖性滤波（训练/部署同构；密集信号池必需）"},
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
        self._factor_composite: Optional[dict] = None  # 级联组合因子配方（factor_miner 挖掘）
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
        """解析子策略名列表并实例化。

        剔除 meta_controller 自身是硬约束——元控制器把自己当子策略会无限递归。
        该护栏原先在训练/注册路径（该路径已随元策略训练于 2026-09-28 移除），
        现收敛到本运行时唯一入口。
        """
        raw = str(self.params.get("sub_strategies", "") or "").strip()
        if not raw:
            self._strategy_names = []
            self._sub_strategies = []
            return
        names = [n.strip() for n in raw.split(",")
                 if n.strip() and n.strip() != "meta_controller"]
        self._strategy_names = list(dict.fromkeys(names))  # 去重但保持顺序
        self._sub_strategies = []
        from . import get_strategy  # 延迟导入避免循环依赖
        loaded = []
        for name in list(self._strategy_names):
            try:
                st = get_strategy(name)
                st.reset()
                self._sub_strategies.append(st)
                # 初始化表现跟踪
                if name not in self._strategy_trades:
                    self._strategy_trades[name] = []
                loaded.append(name)
                log.info("[meta] 子策略 %s 已加载", name)
            except ValueError as e:
                # run24 修复：加载失败的名字若不剔除会留在 _strategy_names，
                # 而 _sub_strategies 只含成功项 → _collect_sub_signals 按序号
                # 取实例时错位/IndexError，前几个名字张冠李戴污染集成信号
                log.warning("[meta] 子策略 %s 加载失败，剔除: %s", name, e)
        self._strategy_names = loaded
        self._last_sig_dir = {n: 0.0 for n in self._strategy_names}  # 信号流滤波记忆

    @property
    def managed_executors(self) -> list[str]:
        """子策略对应的 executor 名（从 params 推导，不依赖已实例化的子策略）。"""
        raw = str(self.params.get("sub_strategies", "") or "").strip()
        names = list(dict.fromkeys(
            n.strip() for n in raw.split(",")
            if n.strip() and n.strip() != "meta_controller"))
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
        self._factor_composite = None  # 换模型后清空旧配方，防跨模型残留
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
        """计算因子信号列（与 rl_adaptive 同口径：配方优先，其次表达式）。"""
        import pandas as pd
        if self._factor_composite:
            from factors.mining import CompositeFactorEvaluator
            ev = CompositeFactorEvaluator(
                self._factor_composite.get("weights") or {},
                mu=self._factor_mu, sd=self._factor_sd)
            df = pd.DataFrame({
                "open": self._opens, "high": self._highs, "low": self._lows,
                "close": self._closes, "volume": self._volumes,
            })
            return ev.eval(df)
        if not self._factor_expr or len(self._closes) < 5:
            return None
        from factors.mining import FactorExecutor
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
        # 信号流新颖性滤波：方向未变化的重复信号抑制为无信号。历史实验
        # （.optim/exp_training/confirm_signal_filter.py，脚本已随元策略训练移除）
        # 曾验证它能逃出密集信号坍缩。默认关；开启时训练/部署须同步开启同名 cfg
        # 才能保持状态口径一致（训练侧已随该功能移除，此约束现仅约束部署侧）。
        use_filter = bool(self.params.get("signal_novelty_filter", False))
        last_dirs = getattr(self, "_last_sig_dir", {})
        for i, name in enumerate(self._strategy_names):
            st = self._sub_strategies[i]
            try:
                sig = st.on_candle(ctx)
                self._strategy_signals[name] = sig
                direction = 1.0 if sig and sig.side == "buy" else (
                    -1.0 if sig and sig.side == "sell" else 0.0)
                if use_filter and direction != 0.0 and direction == last_dirs.get(name, 0.0):
                    # 重复方向抑制：方向清零、置信度清零（原始 signal 保留供日志）
                    direction, conf = 0.0, 0.0
                else:
                    conf = sig.confidence if sig else 0.0
                last_dirs[name] = 1.0 if sig and sig.side == "buy" else (
                    -1.0 if sig and sig.side == "sell" else 0.0)
                result[name] = {
                    "signal": sig,
                    "confidence": conf,
                    "direction": direction,
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
        """构建 DRL 元策略状态向量（走 build_meta_state 单一路径）。

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
                            fc = md.get("factor_composite")
                            if isinstance(fc, dict) and fc.get("weights"):
                                self._factor_composite = fc
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
                        # DRL 选择：0=清仓, 1=买入, 2=卖出（原训练环境的 action 0 与
                        # 2 均触发清仓；实盘执行段只有 1(买)/-1(卖) 分支，0 曾无映射 →
                        # 持仓中选 0 实盘不动作，训练/部署语义不一致。0 映射到 -1
                        # （卖出清仓）以对齐原训练语义——该环境已随元策略训练于
                        # 2026-09-28 移除，此处映射保留，保证既有模型文件仍可加载执行）
                        action_map = {0: -1, 1: 1, 2: -1}
                        mapped = action_map.get(action, -1)
                        reason = f"元策略DRL选择子策略#{action}(conf={action_conf:.2f})"
                        action = mapped
                else:
                    weighted_sum, total_conf, _ = self._compute_ensemble_signal(sub_results)
                    action = 1 if weighted_sum > self.params["ensemble_buy_threshold"] else (
                        -1 if weighted_sum < self.params["ensemble_sell_threshold"] else 0
                    )
                    reason = "元策略状态不匹配，回退集成"
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
            # 卖出成交后，记录本次交易的收益到各子策略的胜率跟踪。
            # 默认（per_strategy_trades=False）：每个子策略都记同一个收益（简化跟踪）；
            # 开启后按子策略各自信号定向记账（原训练环境同款口径，该环境已随元策略
            # 训练于 2026-09-28 移除）：顺向信号记真实收益、逆向记 0、
            # 无信号不记（缺席≠错误），让状态里的 win_rate 具备子策略区分度。
            if self._entry and self._entry > 0:
                pnl = (price - self._entry) / self._entry
                window = self._meta_window()
                if bool(self.params.get("per_strategy_trades", False)):
                    for name in self._strategy_names:
                        sig = (self._strategy_signals or {}).get(name)
                        if sig is None:
                            continue  # 无信号不记
                        d = (1.0 if sig.side == "buy"
                             else (-1.0 if sig.side == "sell" else 0.0))
                        if d > 0:
                            record_meta_trade_result(
                                self._strategy_trades.setdefault(name, []), pnl, window)
                        elif d < 0:
                            record_meta_trade_result(
                                self._strategy_trades.setdefault(name, []), 0.0, window)
                else:
                    for name in self._strategy_names:
                        record_meta_trade_result(
                            self._strategy_trades.setdefault(name, []), pnl, window)
            self._entry = None
