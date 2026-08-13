"""风险管理：单笔最大亏损、每日最大亏损、频率限制、最小订单金额、连续亏损冷却等。

借鉴成熟量化项目（freqtrade）的 Protection 机制：
- 连续亏损冷却：连续 N 笔亏损后暂停交易 M 分钟，防止策略失效期持续爆仓
- 记录最近交易盈亏，供冷却判断
- 闪崩/暴涨保护：检测异常价格波动，自动暂停交易
- 成交量异常保护：检测流动性危机信号
- 每日盈亏持久化：跨重启保留日盈亏状态
"""
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from core.database import Database

log = logging.getLogger(__name__)

DEFAULT_RULES = {
    "max_loss_per_trade_usd": 50.0,
    "max_daily_loss_usd": 200.0,
    "min_order_value_usd": 10.0,
    "max_trades_per_hour": 20,
    "max_position_pct": 0.9,
    "max_consecutive_losses": 3,      # 连续亏损 N 笔
    "cooldown_minutes": 15,           # 触发后冷却 M 分钟
    # 闪崩/暴涨保护
    "flash_crash_5min_drop_pct": 0.10,   # 5分钟跌幅超过 10% → 暂停
    "flash_crash_24h_drop_pct": 0.25,    # 24小时跌幅超过 25% → 暂停
    "flash_rally_5min_rise_pct": 0.20,   # 5分钟涨幅超过 20% → 暂停（防追高）
    "flash_cooldown_minutes": 30,        # 闪崩触发后冷却 30 分钟
    # 成交量异常保护
    "max_vol_ratio": 3.0,               # 成交量超过均量 3 倍 → 暂停
}

# 风控规则上下界（update_rules 强制钳制，防止误配/恶意配置关闭风控）
_RULE_BOUNDS = {
    "max_loss_per_trade_usd": (1.0, 1_000_000.0),
    "max_daily_loss_usd": (1.0, 1_000_000.0),
    "min_order_value_usd": (0.0, 1_000_000.0),
    "max_trades_per_hour": (1.0, 1000.0),
    "max_position_pct": (0.05, 1.0),
    "max_consecutive_losses": (1.0, 100.0),
    "cooldown_minutes": (0.0, 1440.0),
    "flash_crash_5min_drop_pct": (0.0, 1.0),
    "flash_crash_24h_drop_pct": (0.0, 1.0),
    "flash_rally_5min_rise_pct": (0.0, 2.0),
    "flash_cooldown_minutes": (0.0, 1440.0),
    "max_vol_ratio": (1.0, 100.0),
}


@dataclass
class RiskManager:
    db: Database
    _recent_pnls: list[dict] = None  # 最近平仓盈亏记录 [{ts, pnl, reason}]
    _cooldown_until: float = 0.0     # 冷却截止时间戳
    # 闪崩保护状态
    _flash_cooldown_until: float = 0.0
    _price_history: list[tuple[float, float]] = None  # [(timestamp, price)]
    # 每日盈亏持久化
    _daily_pnl: float = 0.0
    _daily_pnl_date: str = ""
    _daily_pnl_loaded: bool = False
    # 规则缓存（check() 在每根 K 线热路径，避免每次信号检查都打 DB）
    _rules_cache: Optional[dict] = None
    _rules_cache_ts: float = 0.0

    def __post_init__(self) -> None:
        if self._recent_pnls is None:
            self._recent_pnls = []
        if self._price_history is None:
            self._price_history = []
        # 每日盈亏在引擎启动时通过 get_daily_pnl() 异步加载（DB 是 async 接口）

    async def get_daily_pnl(self) -> float:
        """加载并返回今日累计盈亏（跨重启持久化恢复）。"""
        if not self._daily_pnl_loaded:
            try:
                today = time.strftime("%Y-%m-%d")
                stored = await self.db.kv_get(f"daily_pnl_{today}", None)
                if stored:
                    parts = str(stored).split("|")
                    if len(parts) == 2:
                        self._daily_pnl_date = parts[0]
                        self._daily_pnl = float(parts[1])
                        log.info("[risk] 恢复每日盈亏: date=%s pnl=%.2f", self._daily_pnl_date, self._daily_pnl)
            except Exception as e:  # noqa: BLE001
                log.warning("[risk] 加载每日盈亏失败: %s", e)
            self._daily_pnl_loaded = True
        return self._daily_pnl

    async def add_daily_pnl(self, pnl: float) -> None:
        """累计每日盈亏；跨日时归档昨日并重置；每次更新持久化（跨重启保留）。"""
        today = time.strftime("%Y-%m-%d")
        if self._daily_pnl_date and self._daily_pnl_date != today:
            # 跨日：归档昨日并重置今日
            try:
                await self.db.kv_set(f"daily_pnl_{self._daily_pnl_date}",
                                     f"{self._daily_pnl_date}|{self._daily_pnl:.4f}")
            except Exception as e:  # noqa: BLE001
                log.warning("[risk] 归档昨日盈亏失败: %s", e)
            self._daily_pnl_date = today
            self._daily_pnl = 0.0
        self._daily_pnl += pnl
        try:
            await self.db.kv_set(f"daily_pnl_{today}", f"{today}|{self._daily_pnl:.4f}")
        except Exception as e:  # noqa: BLE001
            log.warning("[risk] 保存每日盈亏失败: %s", e)

    async def record_trade(self, pnl: float, reason: str = "") -> None:
        """记录一笔平仓盈亏，供连续亏损冷却判断。"""
        self._recent_pnls.append({"ts": time.time(), "pnl": pnl, "reason": reason})
        if len(self._recent_pnls) > 20:
            self._recent_pnls = self._recent_pnls[-20:]
        # 每日盈亏累计（含盈利与亏损，跨重启持久化）——曾只累计亏损，
        # 重启后当日盈利清零、风控"每日最大亏损"基准被低估
        await self.add_daily_pnl(pnl)

    def _consecutive_losses(self) -> int:
        """连续亏损笔数（从最近一笔往前数）。"""
        count = 0
        for t in reversed(self._recent_pnls):
            if t["pnl"] < 0:
                count += 1
            else:
                break
        return count

    async def get_rules(self) -> dict:
        # 5s TTL 缓存：check() 在每根 K 线热路径上（实盘/纸面/模拟），
        # 每次都 kv_json_get 会引入一次 DB 往返；update_rules 时主动失效
        now = time.time()
        if self._rules_cache is not None and now - self._rules_cache_ts < 5.0:
            return self._rules_cache
        rules = await self.db.kv_json_get("risk_rules")
        if not rules:
            rules = dict(DEFAULT_RULES)
            await self.db.kv_json_set("risk_rules", rules)
        # 合并默认值：旧配置缺新字段时补默认
        changed = False
        for k, v in DEFAULT_RULES.items():
            if k not in rules:
                rules[k] = v
                changed = True
        if changed:
            await self.db.kv_json_set("risk_rules", rules)
        self._rules_cache = rules
        self._rules_cache_ts = now
        return rules

    async def update_rules(self, rules: dict) -> dict:
        merged = await self.get_rules()
        for k in ("max_loss_per_trade_usd", "max_daily_loss_usd", "min_order_value_usd",
                  "max_trades_per_hour", "max_position_pct",
                  "max_consecutive_losses", "cooldown_minutes",
                  "flash_crash_5min_drop_pct", "flash_crash_24h_drop_pct",
                  "flash_rally_5min_rise_pct", "flash_cooldown_minutes",
                  "max_vol_ratio"):
            if k not in rules:
                continue
            try:
                v = float(rules[k])
            except (TypeError, ValueError):
                log.warning("[risk] 忽略非法规则值 %s=%r", k, rules[k])
                continue
            # JSON NaN/Infinity 字面量可绕过 max/min 钳制（比较恒 False → 钳到上界），
            # 直接把每日亏损/单笔亏损/闪崩保护全部失效；非有限值一律拒绝
            if not math.isfinite(v):
                log.warning("[risk] 忽略非有限规则值 %s=%r", k, rules[k])
                continue
            lo, hi = _RULE_BOUNDS.get(k, (None, None))
            if lo is not None:
                v = max(lo, min(hi, v))
            merged[k] = v
        await self.db.kv_json_set("risk_rules", merged)
        # 更新后立即失效缓存，check() 下根 K 线即用新规则
        self._rules_cache = merged
        self._rules_cache_ts = time.time()
        return merged

    def cooldown_status(self) -> dict:
        """当前冷却状态（供前端/API 展示）。"""
        remaining = self._cooldown_until - time.time()
        flash_remaining = self._flash_cooldown_until - time.time()
        return {
            "active": remaining > 0,
            "remaining_sec": max(0, int(remaining)),
            "consecutive_losses": self._consecutive_losses(),
            "flash_cooling": flash_remaining > 0,
            "flash_remaining_sec": max(0, int(flash_remaining)),
        }

    def _add_price_sample(self, price: float) -> None:
        """记录价格样本，用于闪崩/24h 跌幅检测。

        - 节流：至少间隔 5 秒采一个点（ticker 事件约每秒一次，避免列表无限膨胀）
        - 保留 48 小时：支撑 24h 窗口检测（此前只留 2h，1h 周期下样本永远凑不满）
        """
        now = time.time()
        if self._price_history and now - self._price_history[-1][0] < 5.0:
            return
        self._price_history.append((now, price))
        cutoff = now - 172800
        self._price_history = [(t, p) for t, p in self._price_history if t >= cutoff]

    def _check_flash_crash(self, rules: dict) -> Optional[str]:
        """检测闪崩/暴涨，返回原因字符串（异常则返回），正常返回 None。"""
        if len(self._price_history) < 5:
            return None
        now = time.time()
        # 5分钟前的价格
        cutoff_5m = now - 300
        old_prices = [p for t, p in self._price_history if t <= cutoff_5m]
        recent_prices = [p for t, p in self._price_history if t > cutoff_5m]
        if not old_prices or not recent_prices:
            return None
        old_price = old_prices[-1]
        new_price = recent_prices[-1]
        if old_price <= 0:
            return None
        drop_pct = (old_price - new_price) / old_price
        rise_pct = (new_price - old_price) / old_price

        threshold_drop = rules.get("flash_crash_5min_drop_pct", 0.10)
        threshold_rise = rules.get("flash_rally_5min_rise_pct", 0.20)

        if drop_pct >= threshold_drop:
            return f"闪崩检测：5分钟跌幅 {drop_pct:.1%} >= {threshold_drop:.0%}"
        if rise_pct >= threshold_rise:
            return f"暴涨检测：5分钟涨幅 {rise_pct:.1%} >= {threshold_rise:.0%}"
        return None

    def _check_vol_spike(self, vol_ratio: float, rules: dict) -> Optional[str]:
        """检测成交量异常放大。"""
        max_ratio = rules.get("max_vol_ratio", 3.0)
        if vol_ratio > max_ratio:
            return f"成交量异常：量比 {vol_ratio:.1f}x 超过阈值 {max_ratio:.1f}x"
        return None

    async def check(self, signal, price: float, cash: float, position_value: float,
                    daily_pnl: float, trade_times: list[float], entry_price: Optional[float],
                    vol_ratio: float = 1.0) -> tuple[bool, str]:
        """返回 (是否放行, 原因)。

        vol_ratio: 当前成交量与均量比值，用于成交量异常检测。
        """
        rules = await self.get_rules()

        # 平仓（卖出持仓）豁免所有"防新开仓"的冷却类规则：
        # 连续亏损冷却/每日亏损上限/闪崩冷却/成交量异常/频率限制/24h 跌幅，
        # 否则止损单会被冷却规则拦截，亏损继续扩大（违背止损本意）。
        closing = signal.side == "sell" and position_value > 0

        # 0) 更新价格历史（用于闪崩检测）
        self._add_price_sample(price)

        # 0.1) 闪崩/暴涨保护
        if self._flash_cooldown_until > time.time() and not closing:
            remaining = int(self._flash_cooldown_until - time.time())
            return False, f"闪崩冷却中，剩余 {remaining}s"
        flash_reason = self._check_flash_crash(rules)
        if flash_reason and not closing:
            cooldown_min = int(rules.get("flash_cooldown_minutes", 30))
            self._flash_cooldown_until = time.time() + cooldown_min * 60
            log.warning("[risk] %s，触发 %d 分钟冷却", flash_reason, cooldown_min)
            return False, flash_reason + f"，触发 {cooldown_min} 分钟冷却"

        # 0.2) 成交量异常保护
        vol_reason = self._check_vol_spike(vol_ratio, rules)
        if vol_reason and not closing:
            log.warning("[risk] %s", vol_reason)
            return False, vol_reason

        # 1) 订单金额：qty 明确时 = qty*price；比例下单时按 cash*size_pct 估算实际金额
        if signal.qty is not None:
            order_value = signal.qty * price
        else:
            order_value = cash * float(signal.size_pct or 0.5)

        # 2) 连续亏损冷却保护（仅限新开仓）
        if self._cooldown_until > time.time() and not closing:
            remaining = int(self._cooldown_until - time.time())
            return False, f"连续亏损冷却中，剩余 {remaining}s（暂停新开仓）"
        losses = self._consecutive_losses()
        if losses >= int(rules.get("max_consecutive_losses", 3)) and not closing:
            self._cooldown_until = time.time() + int(rules.get("cooldown_minutes", 15)) * 60
            log.warning("[risk] 连续 %d 笔亏损，触发 %s 分钟冷却", losses, rules.get("cooldown_minutes", 15))
            return False, f"连续 {losses} 笔亏损，触发冷却 {rules.get('cooldown_minutes', 15)} 分钟"

        # 3) 最小订单金额（仅限新开仓：小仓位止损单必须能离场）
        if order_value < rules["min_order_value_usd"] and not closing:
            return False, f"订单金额 ${order_value:.2f} 低于最小限额 ${rules['min_order_value_usd']:.2f}"

        # 4) 每日最大亏损（仅限新开仓，保证持仓可随时离场）
        if daily_pnl <= -rules["max_daily_loss_usd"] and not closing:
            return False, f"今日已亏损 ${-daily_pnl:.2f}，触发每日最大亏损限制"

        # 5) 交易频率限制（每小时；仅限新开仓）
        cutoff = time.time() - 3600
        recent = [t for t in trade_times if t >= cutoff]
        if len(recent) >= int(rules["max_trades_per_hour"]) and not closing:
            return False, f"1小时内交易次数已达上限 {rules['max_trades_per_hour']}"

        # 6) 单笔最大亏损（卖出止盈/止损场景）
        # 注意：绝不拦截止损单——拒绝执行止损只会让亏损继续扩大。
        # 超过上限时记录告警日志（供复盘），放行成交。
        if signal.side == "sell" and entry_price and "止损" in signal.reason:
            qty = signal.qty or (position_value / price if price > 0 else 0.0)
            est_loss = (entry_price - price) * qty
            if est_loss > rules["max_loss_per_trade_usd"]:
                log.warning("[risk] 止损预估亏损 $%.2f 超过单笔上限 $%.2f（止损放行）",
                            est_loss, rules["max_loss_per_trade_usd"])

        # 7) 仓位上限
        if signal.side == "buy" and position_value + order_value > (cash + position_value) * rules["max_position_pct"]:
            return False, "超过最大仓位比例"

        # 7) 24小时跌幅检测（仅限新开仓）
        now = time.time()
        cutoff_24h = now - 86400
        prices_24h = [p for t, p in self._price_history if t >= cutoff_24h]
        if len(prices_24h) >= 10 and not closing:
            drop_24h = (prices_24h[0] - prices_24h[-1]) / prices_24h[0]
            threshold_24h = rules.get("flash_crash_24h_drop_pct", 0.25)
            if drop_24h >= threshold_24h:
                cooldown_min = int(rules.get("flash_cooldown_minutes", 30))
                self._flash_cooldown_until = time.time() + cooldown_min * 60
                log.warning("[risk] 24小时跌幅 %.1f%% >= %.0f%%，触发冷却", drop_24h * 100, threshold_24h * 100)
                return False, f"24小时跌幅 {drop_24h:.1%}，触发 {cooldown_min} 分钟冷却"

        return True, ""
