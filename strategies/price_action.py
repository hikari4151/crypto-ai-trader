"""关键位 + 价格行为策略：支撑/阻力突破或回调入场，量能与 RSI 过滤，止损止盈参考关键位。"""
from typing import Any, Optional

from .base import Signal, Strategy


class PriceActionStrategy(Strategy):
    name = "price_action"
    description = ("关键位+价格行为：基于摆动高低点识别支撑/阻力，突破(breakout)或回调(pullback)入场，"
                   "量能确认 + RSI 过滤，止损止盈参考关键位")
    default_params = {
        "mode": "breakout",
        "sr_window": 10,
        "min_touches": 2,
        "breakout_pct": 0.001,
        "volume_confirm": 1.2,
        "rsi_ob": 72,
        "rsi_os": 28,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "use_sr_stop": True,
        "size_pct": 0.5,
    }
    param_schema = {
        "mode": {"type": "str", "choices": ["breakout", "pullback"], "label": "入场模式(突破/回调)"},
        "sr_window": {"type": "int", "min": 5, "max": 60, "label": "摆动高低点窗口"},
        "min_touches": {"type": "int", "min": 1, "max": 8, "label": "关键位最少触碰次数"},
        "breakout_pct": {"type": "float", "min": 0.0001, "max": 0.02, "label": "突破确认幅度"},
        "volume_confirm": {"type": "float", "min": 0.5, "max": 3.0, "label": "放量确认倍数"},
        "rsi_ob": {"type": "float", "min": 60, "max": 90, "label": "RSI超买阈值"},
        "rsi_os": {"type": "float", "min": 10, "max": 40, "label": "RSI超卖阈值"},
        "stop_loss_pct": {"type": "float", "min": 0.001, "max": 0.1, "label": "止损比例"},
        "take_profit_pct": {"type": "float", "min": 0.001, "max": 0.3, "label": "止盈比例"},
        "use_sr_stop": {"type": "bool", "label": "止损参考关键位"},
        "size_pct": {"type": "float", "min": 0.05, "max": 1.0, "label": "下单比例"},
    }

    def reset(self) -> None:
        self._entry = None
        self._prev_close: Optional[float] = None  # 前一根收盘价（判断"刚突破"用）

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        p = self.params
        ind = ctx["indicators"]
        price = ctx["price"]
        position = ctx.get("position", 0.0)
        # 暖机守卫：RSI/SR 在窗口期前为 NaN（快照置 0），0 值会误触 RSI 过滤放行
        if int(ind.get("candles_count", 999)) < 30:
            return None
        symbol = ctx["symbol"]
        rsi = ind.get("rsi", 50.0)
        pa = ind.get("pa") or {}
        sr = ind.get("sr") or {}
        vol_ratio = pa.get("volume_ratio", 1.0)

        # ---- 持仓管理：止盈/止损/触及关键位离场 ----
        if position > 0 and self._entry:
            chg = (price - self._entry) / self._entry
            if p.get("use_sr_stop") and sr.get("support"):
                sr_dist = (price - sr["support"]) / price
                stop = min(p["stop_loss_pct"], max(0.001, sr_dist * 0.5))
            else:
                stop = p["stop_loss_pct"]
            if chg <= -stop:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止损 {chg:.2%}(参考关键位)")
            if chg >= p["take_profit_pct"]:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason=f"止盈 {chg:.2%}")
            if p.get("use_sr_stop") and sr.get("resistance") and price >= sr["resistance"]:
                return Signal(symbol, "sell", 1.0, strategy=self.name, reason="触及阻力位获利了结")
            return None

        # ---- 空仓入场 ----
        if position > 0:
            return None
        prev_close = self._prev_close
        self._prev_close = price
        mode = p.get("mode", "breakout")
        if mode == "breakout":
            # 突破语义修正：判断"价格刚上穿近期阻力位"。
            # 旧逻辑用"上方最近的阻力"（technical 的 resistance 恒 > 当前价），
            # 条件 price >= res*(1+pct) 数学上恒假 → 突破模式永远不开仓。
            # 正确判据：broken_resistance（低于当前价的最高阻力，即刚被突破的位）
            # + 前一根收盘价仍在阻力下方（确认"刚刚上穿"而非早已突破）。
            broken = sr.get("broken_resistance")
            if (broken and prev_close is not None
                    and prev_close <= broken < price
                    and price >= broken * (1 + p["breakout_pct"])):
                if vol_ratio >= p["volume_confirm"] and rsi < p["rsi_ob"]:
                    return Signal(symbol, "buy", p["size_pct"], strategy=self.name,
                                  reason=f"突破阻力{broken}(量能{vol_ratio:.1f}x)")
        else:  # pullback 回调入场
            sup = sr.get("support")
            if sup and price >= sup and price <= sup * 1.01 and vol_ratio < p["volume_confirm"]:
                return Signal(symbol, "buy", p["size_pct"], strategy=self.name,
                              reason=f"回调至支撑{sup}企稳(缩量)")
        return None

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        if side == "buy":
            self._entry = price
        else:
            self._entry = None