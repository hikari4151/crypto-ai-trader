"""深度强化学习自适应策略：用训练好的智能体实时决策。

策略在每根K线：
1. 维护滚动 OHLCV 历史
2. 按与训练一致的 _precompute_features 计算当前状态特征
3. 叠加持仓比例 / 浮动盈亏 → 完整状态
4. 若训练时注入了因子信号列（factor_expression），用同一表达式实时计算并拼接
5. 智能体输出目标仓位档位（清仓/轻仓/半仓/重仓/满仓）
6. 按当前权益调整到目标仓位：目标>当前→买入补仓，目标<当前→卖出减仓

通过 strategy_params["model_path"] 指定训练产物模型文件。
被回测引擎（fast_engine / engine）与实盘引擎（TradingEngine）共用。
"""
import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import Signal, Strategy

log = logging.getLogger(__name__)

try:
    from drl.env import ACTION_BUCKETS, _precompute_features
    from drl.agent import ACAgent
    _DRL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _DRL_AVAILABLE = False
    log.warning("[rl] DRL 模块导入失败，rl_adaptive 策略不可用")


class RLAdaptiveStrategy(Strategy):
    name = "rl_adaptive"
    description = "深度强化学习自适应策略：MDP建模+策略梯度训练，按市场状态动态调仓（波动率上升降仓、趋势明朗加仓），可选注入因子信号列"
    default_params = {
        "model_path": "",             # 训练好的智能体模型文件（JSON）
        "buy_zone": 0.02,             # 与目标仓位差异>该比例才调仓（减少无效交易）
        "factor_expression": "",      # 训练时注入的因子表达式（部署必须一致，留空=无因子列）
    }
    param_schema = {
        "model_path": {"type": "str", "label": "智能体模型路径"},
        "buy_zone": {"type": "float", "min": 0.0, "max": 0.2, "label": "调仓死区比例"},
        "factor_expression": {"type": "str", "label": "因子表达式（须与训练一致）"},
    }

    def reset(self) -> None:
        self._closes: list[float] = []
        self._opens: list[float] = []
        self._volumes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._entry = None
        self._agent = None
        self._factor_mu: Optional[float] = None   # 模型文件内的因子标准化统计量
        self._factor_sd: Optional[float] = None
        self._factor_expr_from_model: str = ""    # 模型文件内记录的训练表达式
        self._state_dim = 15  # 13 特征 + 持仓比例 + 浮动盈亏（+因子列时动态调整）
        self._n_actions = len(ACTION_BUCKETS) if _DRL_AVAILABLE else 5

    def _load_agent(self) -> bool:
        path = self.params.get("model_path", "")
        if not path:
            return False
        if not os.path.isabs(path):
            # 相对项目根目录
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        if not os.path.exists(path):
            log.warning("[rl] 模型不存在: %s", path)
            return False
        try:
            self._agent = ACAgent.load(path)
            self._state_dim = self._agent.state_dim
            self._n_actions = self._agent.n_actions
            # 读取模型文件内的因子标准化统计量（训练段拟合）——
            # 部署端必须用同一 mu/sd 标准化因子值，与训练状态分布一致
            self._factor_mu = None
            self._factor_sd = None
            self._factor_expr_from_model = ""
            self._trade_zone = None   # 训练死区（模型内记录），None 时用 params.buy_zone 兜底
            try:
                with open(path, "r", encoding="utf-8") as f:
                    md = json.load(f)
                if "factor_mu" in md and "factor_sd" in md:
                    self._factor_mu = float(md["factor_mu"])
                    self._factor_sd = float(md["factor_sd"])
                self._factor_expr_from_model = str(md.get("factor_expression", "") or "")
                if "min_trade_zone" in md:
                    self._trade_zone = float(md["min_trade_zone"])
            except Exception as e:  # noqa: BLE001
                log.warning("[rl] 模型元数据读取失败: %s", e)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("[rl] 模型加载失败: %s", e)
            return False

    def _factor_value(self) -> Optional[float]:
        """在滚动缓冲上计算因子表达式，返回标准化后的最新值。

        标准化统计量取模型文件（训练段拟合的 mu/sd）；旧模型无统计量时
        返回原始值（保持旧行为）。表达式优先取模型内记录的（部署一致性）。
        """
        expr = str(self.params.get("factor_expression", "") or "").strip() or self._factor_expr_from_model
        if not expr:
            return None
        if len(self._closes) < 5:
            return None
        from factors.mining import FactorExecutor
        df = pd.DataFrame({
            "open": self._opens, "high": self._highs, "low": self._lows,
            "close": self._closes, "volume": self._volumes,
        })
        try:
            s = FactorExecutor(expr).eval(df).astype(float)
            v = s.iloc[-1]
            if v is None or v != v:  # NaN 视为无效
                return None
            v = float(v)
            # 模型带训练统计量 → 同一口径标准化（旧模型无统计量 → 原始值）
            if self._factor_mu is not None and self._factor_sd:
                v = (v - self._factor_mu) / self._factor_sd
            return v
        except Exception:  # noqa: BLE001
            return None

    def on_candle(self, ctx: dict[str, Any]) -> Optional[Signal]:
        price = float(ctx["price"])
        ind = ctx.get("indicators", {})
        position = float(ctx.get("position", 0.0))
        cash = float(ctx.get("cash", 0.0))
        symbol = ctx["symbol"]

        # 外部平仓（止损/其他策略逻辑）后清空入场价，避免成本基残留
        if position <= 0 and self._entry is not None:
            self._entry = None

        # 累积历史（真实 open/high/low；曾误用 highs 冒充 open，已修复）
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

        if not _DRL_AVAILABLE or len(self._closes) < 60:
            return None

        if self._agent is None and not self._load_agent():
            return None

        # 构造与训练一致的 DataFrame → 特征（open 为真实数据，与 drl.env 口径一致）
        df = pd.DataFrame({
            "open": self._opens, "high": self._highs, "low": self._lows,
            "close": self._closes, "volume": self._volumes,
        })
        try:
            feats = _precompute_features(df)
        except Exception as e:  # noqa: BLE001
            log.warning("[rl] 特征计算失败: %s", e)
            return None
        # 状态窗口：与训练一致的时序堆叠（模型 state_window>1 时堆叠最近 N 根，
        # 早期不足零填充——与 drl.env._state_feats 同口径）
        w = getattr(self._agent, "state_window", 1) if self._agent else 1
        if w > 1 and len(feats) > 0:
            seg = feats[-w:] if len(feats) >= w else np.pad(feats, ((w - len(feats), 0), (0, 0)))
            state_feat = seg.reshape(-1)
        else:
            state_feat = feats[-1] if len(feats) else np.zeros(13)

        # 账户状态
        equity = cash + position * price
        pos_ratio = (position * price) / equity if equity > 0 else 0.0
        pnl_ratio = 0.0
        if self._entry and self._entry > 0:
            pnl_ratio = (price - self._entry) / self._entry
        state = np.concatenate([state_feat, [pos_ratio, pnl_ratio]])

        # 因子信号列（训练时注入过才需要；与训练同表达式、同标准化口径）
        if self._agent and self._agent.state_dim > len(state):
            fv = self._factor_value()
            if fv is None:
                log.warning("[rl] 因子信号计算失败（表达式与训练不一致？），跳过本根K线")
                return None
            state = np.concatenate([state, [fv]])

        # 维度校验：状态维度必须与模型一致（模型 state_dim 含因子列时部署须配同表达式）
        if self._agent and len(state) != self._state_dim:
            log.warning("[rl] 状态维度 %d 与模型 %d 不匹配（模型含因子列时须配置 factor_expression）",
                        len(state), self._state_dim)
            return None

        action = self._agent.greedy_action(state)
        target = float(ACTION_BUCKETS[action])
        # 调仓死区：优先取模型内训练值（部署与训练严格一致），params.buy_zone 兜底
        zone = float(self._trade_zone if self._trade_zone is not None
                     else self.params.get("buy_zone", 0.02))
        diff = target - pos_ratio

        if diff > zone and equity > 0 and cash > 0:
            # 目标仓位差 diff（占总权益比例）→ 按现金折算下单比例。
            # 曾误用 diff/zone（同量纲相除）：zone=0.02 时差 0.02 就下 50% 现金、
            # 差 0.04 就满仓，目标仓位档位形同虚设（0%~100% 反复打满）。
            # 现为"补足一半仓位差"：差 20% 仓位 → 补 10% 现金。
            buy_value = min(diff * equity, cash)
            size = max(min(1.0, buy_value / max(cash, 1e-9)), 0.05)
            # 显式传 qty：回测引擎卖出忽略 size_pct（恒全平），实盘/纸面按比例——
            # 传 qty 保证回测与实盘部分加/减仓口径一致
            return Signal(symbol, "buy", qty=buy_value / price, size_pct=size,
                          strategy=self.name,
                          reason=f"RL加仓至{int(target*100)}%")
        if diff < -zone and position > 0:
            sell_value = min(abs(diff) * equity, position * price)
            qty = sell_value / price
            return Signal(symbol, "sell", qty=max(qty, 0.0), size_pct=1.0,
                          strategy=self.name,
                          reason=f"RL减仓至{int(target*100)}%")
        return None

    def on_fill(self, symbol: str, side: str, price: float) -> None:
        if side == "buy":
            # 加仓/新仓：更新入场价（分批加仓用最新成交价作为成本参考）
            self._entry = price
        # 卖出（分批减仓）不修改入场价：部分减仓后持仓成本不变。
        # 曾在此用卖出价覆盖 entry，导致止损卖出后成本基失真、浮动盈亏虚高；
        # 全仓离场由 on_candle 的 position<=0 检测清空。


def make_rl_strategy(name: str, model_path: str) -> RLAdaptiveStrategy:
    """工厂：用指定模型创建 RL 策略实例（用于注册为 AI 训练产物策略）。"""
    st = RLAdaptiveStrategy()
    st.name = name
    st.update_params({"model_path": model_path})
    return st
