"""可插拔策略基类。用户自定义策略继承 Strategy 并实现 on_candle 即可。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


def strategy_ma_periods(strategy) -> tuple[int, int]:
    """从策略参数推导 MA 快/慢线周期（回测引擎与实时行情中枢共用，保证同口径）。

    仅对声明了 fast_period/slow_period 参数的策略（目前为 dual_ma）生效；
    其余策略（factor_signal/grid/price_action 等）读 ma_fast/ma_slow 时保持
    默认 10/30 —— 与 factor 库中 bias_10/bias_30 的口径一致。
    """
    p = getattr(strategy, "params", None) or {}
    fast = p.get("fast_period")
    slow = p.get("slow_period")
    if fast is None or slow is None:
        return 10, 30
    try:
        return max(1, int(float(fast))), max(1, int(float(slow)))
    except (TypeError, ValueError):
        return 10, 30


@dataclass
class Signal:
    symbol: str
    side: str                       # buy / sell
    size_pct: float = 0.5
    qty: Optional[float] = None
    order_type: str = "market"
    limit_price: Optional[float] = None
    reason: str = ""
    strategy: str = ""
    # 信号置信度（0.0~1.0，默认 1.0 兼容旧策略）。DRL 策略用策略分布熵倒数/方差
    # 填充，供元策略/多策略合成加权使用。旧策略不设置则视为满置信。
    confidence: float = 1.0


class Strategy(ABC):
    name: str = "base"
    description: str = ""
    param_schema: dict[str, dict[str, Any]] = {}
    default_params: dict[str, Any] = {}

    def __init__(self) -> None:
        self.params: dict[str, Any] = dict(self.default_params)
        # 立即初始化内部状态：曾仅依赖引擎/回测流程显式调用 reset()，
        # 工厂创建（make_rl_strategy/register_dynamic）后直接 on_candle 会 AttributeError
        self.reset()

    def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """热更新参数（含 AI 优化结果），按 schema 校验类型与范围，返回实际生效参数。"""
        schema = self.param_schema
        for k, v in params.items():
            if k not in schema:
                continue
            spec = schema[k]
            try:
                t = spec.get("type")
                if t == "int":
                    v = int(round(float(v)))
                elif t == "float":
                    v = float(v)
                elif t == "bool":
                    v = bool(v)
                elif t == "str":
                    choices = spec.get("choices") or []
                    if choices and str(v) not in choices:
                        continue
                    v = str(v)
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                self.params[k] = v
            except (TypeError, ValueError):
                continue
        return dict(self.params)

    def reset(self) -> None:
        """回测/重启时重置内部状态。"""
        pass

    @abstractmethod
    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        """处理一根新K线，返回信号（可空）。"""

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        """成交回调（用于记录入场价等状态）。"""

    def protective_levels(self, ctx: dict[str, Any]) -> Optional[dict]:
        """返回当前持仓的触价保护位：{"stop": 价格, "take_profit": 价格} 或 None。

        P0-1 intrabar 止损/止盈建模：回测撮合内核拿到该保护位后，会用K线的
        high/low 判断盘中是否触及并在触价位成交（而非仅收盘价判断），
        使回测/纸面/实盘的止损口径一致（实盘 protective stop 本就是触价单）。
        默认返回 None（策略内部自行处理止损止盈，回测维持旧行为）。
        """
        return None