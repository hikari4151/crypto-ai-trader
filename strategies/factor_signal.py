"""因子择时策略：把量化因子（动量/RSI/MACD/量比/布林位置/自定义表达式/RL组合）转为可执行的买卖信号。

这是"因子使用场景"的落地实现——高 IC 因子不再只是分析图表，而是：
- 因子值 → z-score/阈值 → 当前买卖/中性信号
- 直接接入回测引擎跑绩效（配合过拟合检测验证有效性）

因子来源：
- 内置因子：复用 fast_engine 已预计算的指标序列（ind 快照），不重复计算
- 自定义因子：factor="custom" + expression 参数，用因子表达式沙箱（FactorExecutor，
  factors/mining.py）在策略自维护的滚动 OHLCV 缓冲上实时计算——AI 挖掘的因子、
  进化变体、手动表达式都能直接拿来当信号源
- RL 组合因子：factor="combo" + combo_spec 参数（JSON：{因子key: 权重}，来自
  /api/drl/mine-factor 的输出）——RL 因子挖掘选出的组合在策略端实时重建
支持的因子 key 与信号逻辑：
- mom_ma10/mom_ma30: 均线偏离动量（趋势跟随，>0 做多）
- macd_hist: MACD 柱方向（动量强弱）
- rsi_osc: RSI 超买超卖反转（<超卖做多，>超买做空）
- vol_break: 量比 + 价格突破（放量突破做多）
- bb_pos: 布林带位置（均值回归）
- custom: 任意白名单表达式（如 close / ma(close, 30) - 1）
- combo: RL 挖掘的组合因子（combo_spec 指定因子与权重）
"""
from typing import Any, Optional

import logging

from .base import Signal, Strategy

log = logging.getLogger(__name__)

# 滚动缓冲上限（与 rl_adaptive 一致；表达式窗口最大 200，留足余量）
_BUFFER_MAX = 300


class FactorSignalStrategy(Strategy):
    name = "factor_signal"
    description = "因子择时策略：将量化因子（动量/RSI/MACD/量比/布林位置/自定义表达式/RL组合）转为买卖信号，让高IC因子可回测可实盘"
    default_params = {
        "factor": "macd_hist",
        "mode": "trend",            # trend=顺势 / reversal=反转
        "buy_threshold": 0.0,       # 入场阈值（趋势>0买；反转<阈值买）
        "sell_threshold": 0.0,      # 离场阈值
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "size_pct": 0.5,
        "expression": "",           # factor="custom" 时的因子表达式（沙箱白名单）
        "combo_spec": "",           # factor="combo" 时的因子组合 JSON（{key: 权重}，来自 RL 挖掘）
    }
    param_schema = {
        "factor": {"type": "str", "choices": ["mom_ma10", "mom_ma30", "macd_hist", "rsi_osc", "vol_break", "bb_pos", "custom", "combo"],
                   "label": "因子"},
        "mode": {"type": "str", "choices": ["trend", "reversal"], "label": "模式(顺势/反转)"},
        "buy_threshold": {"type": "float", "min": -100.0, "max": 100.0, "label": "买入阈值"},
        "sell_threshold": {"type": "float", "min": -100.0, "max": 100.0, "label": "卖出阈值"},
        "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.2, "label": "止损比例"},
        "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.5, "label": "止盈比例"},
        "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
        "expression": {"type": "str", "label": "自定义因子表达式（factor=custom 时生效）"},
        "combo_spec": {"type": "str", "label": "RL组合因子 JSON（factor=combo 时生效，如 {\"vol_ratio\":-0.04}）"},
    }

    def reset(self) -> None:
        self._entry = None
        self._closes: list[float] = []
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []

    def _push_ohlcv(self, ctx: dict) -> None:
        """把当前K线追加到滚动缓冲（供自定义表达式使用）。"""
        ind = ctx.get("indicators", {})
        price = float(ctx["price"])
        self._closes.append(float(ind.get("close", price) or price))
        self._opens.append(float(ind.get("open", price) or price))
        self._highs.append(float(ind.get("high", price) or price))
        self._lows.append(float(ind.get("low", price) or price))
        self._volumes.append(float(ind.get("volume", 0.0) or 0.0))
        if len(self._closes) > _BUFFER_MAX:
            self._closes = self._closes[-_BUFFER_MAX:]
            self._opens = self._opens[-_BUFFER_MAX:]
            self._highs = self._highs[-_BUFFER_MAX:]
            self._lows = self._lows[-_BUFFER_MAX:]
            self._volumes = self._volumes[-_BUFFER_MAX:]

    def _custom_factor_value(self) -> Optional[float]:
        """在滚动缓冲上执行自定义因子表达式，返回最新因子值（非法/异常返回 None）。"""
        expr = str(self.params.get("expression", "")).strip()
        if not expr:
            return None
        if len(self._closes) < 5:
            return None
        import pandas as pd
        from factors.mining import FactorExecutor
        df = pd.DataFrame({
            "open": self._opens, "high": self._highs, "low": self._lows,
            "close": self._closes, "volume": self._volumes,
        })
        try:
            s = FactorExecutor(expr).eval(df).astype(float)
            v = s.iloc[-1]
            return None if v is None or v != v else float(v)  # NaN 视为无效
        except Exception:  # noqa: BLE001
            return None

    def _combo_factor_value(self) -> Optional[float]:
        """计算 RL 挖掘的组合因子（combo_spec：{因子key: 权重}）。

        与训练时的 composite_factor（全量 z-score）同构，但基于策略滚动缓冲：
        逐因子用因子库 compute 计算序列（library.py 纯 pandas），缓冲内 z-score 后
        按权重加权求和（负权重=反向暴露）。
        """
        spec = str(self.params.get("combo_spec", "")).strip()
        if not spec:
            return None
        if len(self._closes) < 20:
            return None
        try:
            import json
            weights = json.loads(spec)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(weights, dict) or not weights:
            return None
        import pandas as pd
        from factors.library import get_factor
        df = pd.DataFrame({
            "open": self._opens, "high": self._highs, "low": self._lows,
            "close": self._closes, "volume": self._volumes,
        })
        total = 0.0
        n_used = 0
        for key, w in weights.items():
            f = get_factor(str(key))
            if f is None:
                continue
            # 跳过已下线因子（IC 衰变自动下线）
            try:
                from factors.library import factor_is_live
                if not factor_is_live(str(key)):
                    log.debug("[factor_signal] 组合因子 %s 已下线，跳过", key)
                    continue
            except Exception:
                pass
            try:
                s = f.series(df).astype(float)
                v = s.iloc[-1]
                if v is None or v != v:
                    continue
                mu = s.mean()
                sd = s.std()
                total += float(w) * ((float(v) - mu) / (sd + 1e-12))
                n_used += 1
            except Exception:  # noqa: BLE001
                continue
        return total if n_used > 0 else None

    def _factor_value(self, ind: dict) -> float:
        """从指标快照计算当前因子值（与 fast_engine 预计算一致）。"""
        p = self.params
        f = p["factor"]
        if f == "mom_ma10":
            return ind.get("close", 0.0) - ind.get("ma_fast", 0.0)
        if f == "mom_ma30":
            return ind.get("close", 0.0) - ind.get("ma_slow", 0.0)
        if f == "macd_hist":
            return ind.get("macd_hist", 0.0)
        if f == "rsi_osc":
            return ind.get("rsi", 50.0)
        if f == "vol_break":
            # 量比 + 价格相对布林带中轨（无量纲化：价格偏离用百分比）
            close = ind.get("close", 0.0)
            bb_mid = ind.get("bb_mid", 0.0)
            return (ind.get("vol_ratio", 1.0) - 1.0) + ((close - bb_mid) / bb_mid if bb_mid else 0.0)
        if f == "bb_pos":
            # 价格在布林带中的位置（0=下轨 1=上轨）
            up = ind.get("bb_upper", 0.0)
            lo = ind.get("bb_lower", 0.0)
            c = ind.get("close", 0.0)
            return (c - lo) / (up - lo + 1e-9)
        if f == "custom":
            v = self._custom_factor_value()
            return v if v is not None else 0.0
        if f == "combo":
            v = self._combo_factor_value()
            return v if v is not None else 0.0
        return 0.0

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        p = self.params
        ind = ctx.get("indicators", {})
        price = ctx["price"]
        position = ctx.get("position", 0.0)
        symbol = ctx["symbol"]
        # 暖机守卫：指标未就绪（NaN 被快照置 0）时不得交易，
        # 否则 RSI=0 等在最初 30 根内产生大量假信号
        if int(ind.get("candles_count", 999)) < 30:
            return None
        # 因子下线隔离：库内注册因子因 IC 衰变下线时跳过该策略（仅调试日志，不抛异常）
        # 仅对因果子库 key 生效（bb_pos 在库内，mom_ma10/macd_hist 等不在库内不受影响）
        f_name = p.get("factor", "")
        if f_name not in ("custom", "combo"):
            try:
                from factors.library import factor_is_live, get_factor
                if get_factor(f_name) is not None and not factor_is_live(f_name):
                    log.debug("[factor_signal] 因子 %s 已下线，跳过信号生成", f_name)
                    return None
            except Exception:
                pass
        self._push_ohlcv(ctx)
        f = self._factor_value(ind)
        mode = p.get("mode", "trend")

        # ---- 持仓管理：止损止盈 ----
        if position > 0 and self._entry:
            chg = (price - self._entry) / self._entry
            if chg <= -p["stop_loss_pct"]:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止损 {chg:.2%}")
            if chg >= p["take_profit_pct"]:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止盈 {chg:.2%}")
            # 信号反向时离场
            if mode == "trend" and f < p["sell_threshold"]:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"因子转弱离场(f={f:.2f})")
            if mode == "reversal" and f > (p["sell_threshold"] if p["sell_threshold"] else 70):
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"超买离场(f={f:.2f})")
            return None

        if position > 0:
            return None

        # ---- 空仓入场 ----
        if mode == "trend":
            if f > p["buy_threshold"]:
                return Signal(symbol, "buy", p["size_pct"], strategy=self.name,
                              reason=f"因子转强(f={f:.2f})")
        else:  # reversal
            # RSI 类因子：低于阈值超卖买入
            if f < p["buy_threshold"]:
                return Signal(symbol, "buy", p["size_pct"], strategy=self.name,
                              reason=f"超卖买入(f={f:.2f})")
        return None

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        self._entry = price if side == "buy" else None
