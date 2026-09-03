"""风险管理：单笔最大亏损、每日最大亏损、频率限制、最小订单金额、连续亏损冷却等。

借鉴成熟量化项目（freqtrade）的 Protection 机制：
- 连续亏损冷却：连续 N 笔亏损后暂停交易 M 分钟，防止策略失效期持续爆仓
- 记录最近交易盈亏，供冷却判断
- 闪崩/暴涨保护：检测异常价格波动，自动暂停交易
- 成交量异常保护：检测流动性危机信号
- 每日盈亏持久化：跨重启保留日盈亏状态
"""
import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

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
    # 熔断人工确认恢复（T3）
    "risk_manual_recovery": 0,          # 0=自动恢复, 1=冷却到期后需人工确认
    # 波动率目标仓位（volatility targeting）：高波动自动缩仓
    "vol_target_enabled": 0,            # 0=关闭, 1=开启（只缩仓不加杠杆）
    "vol_target_annual_pct": 0.40,      # 目标年化波动率（仓位贡献的波动）
    "vol_target_lookback_hours": 24.0,  # 已实现波动率回看窗口（小时）
}

# 风控冷却状态落库键（连亏记录/冷却到期时间/人工确认标记）
RISK_STATE_KEY = "risk_state"

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
    "vol_target_enabled": (0.0, 1.0),
    "vol_target_annual_pct": (0.05, 5.0),
    "vol_target_lookback_hours": (1.0, 168.0),
}


@dataclass
class RiskManager:
    db: Database
    _recent_pnls: list[dict] = None  # 最近平仓盈亏记录 [{ts, pnl, reason}]
    _cooldown_until: float = 0.0     # 冷却截止时间戳
    # 闪崩保护状态
    _flash_cooldown_until: float = 0.0
    _price_history: deque = None  # deque of (timestamp, price); 保留 48h，O(1) 两端操作
    # 每日盈亏持久化
    _daily_pnl: float = 0.0
    _daily_pnl_date: str = ""
    _daily_pnl_loaded: bool = False
    # 规则缓存（check() 在每根 K 线热路径，避免每次信号检查都打 DB）
    _rules_cache: Optional[dict] = None
    _rules_cache_ts: float = 0.0
    _rules_updated_at: float = 0.0     # 规则最后更新（DB/内存）时间戳，供健康信号展示
    # 熔断人工确认恢复（T3）
    _manual_recovery: bool = False
    # 冷却状态是否已从 KV 恢复（restore_state 幂等）
    _state_restored: bool = False

    def __post_init__(self) -> None:
        if self._recent_pnls is None:
            self._recent_pnls = []
        if self._price_history is None:
            self._price_history = deque()
        # 每日盈亏在引擎启动时通过 get_daily_pnl() 异步加载（DB 是 async 接口）

    async def _rollover(self, today: str) -> None:
        """跨日归档：昨日累计写回 KV，今日清零。

        曾只在 add_daily_pnl（即有成交）时归档——零成交的隔天，昨日亏损
        一直留在内存，「每日最大亏损」熔断触发后永不解除。
        """
        if self._daily_pnl_date == today:
            return
        if self._daily_pnl_date:
            try:
                await self.db.kv_set(f"daily_pnl_{self._daily_pnl_date}",
                                     f"{self._daily_pnl_date}|{self._daily_pnl:.4f}")
            except (ValueError, KeyError) as e:  # 归档昨日盈亏：值/键错误属于已知异常类型
                log.warning("[risk] 归档昨日盈亏失败: %s", e)
            log.info("[risk] 每日盈亏跨日重置: %s(%.2f) → %s(0)",
                     self._daily_pnl_date, self._daily_pnl, today)
        # 日期为空（今日尚无成交）时同样要对齐，否则隔天进来仍判不出跨日
        self._daily_pnl_date = today
        self._daily_pnl = 0.0

    async def get_daily_pnl(self) -> float:
        """加载并返回**今日**累计盈亏（跨重启持久化恢复 + 跨日自动归零）。"""
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
            except (ValueError, KeyError) as e:  # 加载每日盈亏：值/键错误属于已知异常类型
                log.warning("[risk] 加载每日盈亏失败: %s", e)
            self._daily_pnl_loaded = True
        await self._rollover(time.strftime("%Y-%m-%d"))
        return self._daily_pnl

    async def add_daily_pnl(self, pnl: float) -> None:
        """累计每日盈亏；跨日时归档昨日并重置；每次更新持久化（跨重启保留）。"""
        # 确保先加载持久化值：曾直接覆盖当日 KV，若 record_trade 先于
        # get_daily_pnl 调用会把历史累计清零（丢数据）
        if not self._daily_pnl_loaded:
            await self.get_daily_pnl()
        today = time.strftime("%Y-%m-%d")
        await self._rollover(today)
        self._daily_pnl += pnl
        try:
            await self.db.kv_set(f"daily_pnl_{today}", f"{today}|{self._daily_pnl:.4f}")
        except (ValueError, KeyError) as e:  # 保存每日盈亏：值/键错误属于已知异常类型
            log.warning("[risk] 保存每日盈亏失败: %s", e)

    async def record_trade(self, pnl: float, reason: str = "") -> None:
        """记录一笔平仓盈亏，供连续亏损冷却判断。"""
        self._recent_pnls.append({"ts": time.time(), "pnl": pnl, "reason": reason})
        if len(self._recent_pnls) > 20:
            self._recent_pnls = self._recent_pnls[-20:]
        # 每日盈亏累计（含盈利与亏损，跨重启持久化）——曾只累计亏损，
        # 重启后当日盈利清零、风控"每日最大亏损"基准被低估
        await self.add_daily_pnl(pnl)
        await self.persist_state()

    async def persist_state(self) -> None:
        """连亏/冷却状态落库（跨重启保留）。

        曾全在内存：重启即清零，"连续亏损 N 笔后暂停交易"这道防线只要重启一次
        就形同作废（策略失效期继续下单），待人工确认的熔断也会凭空解除。
        """
        try:
            await self.db.kv_json_set(RISK_STATE_KEY, {
                "recent_pnls": self._recent_pnls[-20:],
                "cooldown_until": self._cooldown_until,
                "flash_cooldown_until": self._flash_cooldown_until,
                "manual_recovery": self._manual_recovery,
            })
        except Exception as e:  # noqa: BLE001  # 落库失败只降级为"重启丢状态"，不得影响风控判定
            log.warning("[risk] 冷却状态落库失败: %s", e)

    async def restore_state(self) -> None:
        """启动时恢复连亏/冷却状态（幂等）。"""
        if self._state_restored:
            return
        self._state_restored = True
        try:
            raw = await self.db.kv_json_get(RISK_STATE_KEY)
        except Exception as e:  # noqa: BLE001
            log.warning("[risk] 冷却状态读取失败: %s", e)
            return
        if not isinstance(raw, dict):
            return
        rows = raw.get("recent_pnls")
        if isinstance(rows, list):
            clean: list[dict] = []
            for r in rows:
                try:
                    clean.append({"ts": float(r["ts"]), "pnl": float(r["pnl"]),
                                  "reason": str(r.get("reason", ""))})
                except (TypeError, KeyError, ValueError):
                    continue
            self._recent_pnls = clean[-20:]
        try:
            self._cooldown_until = float(raw.get("cooldown_until") or 0.0)
            self._flash_cooldown_until = float(raw.get("flash_cooldown_until") or 0.0)
        except (TypeError, ValueError):
            pass
        self._manual_recovery = bool(raw.get("manual_recovery"))
        if self._recent_pnls or self._cooldown_until > time.time():
            log.info("[risk] 恢复风控状态: 最近平仓 %d 笔，连亏冷却剩余 %.0fs",
                     len(self._recent_pnls), max(0.0, self._cooldown_until - time.time()))

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
        try:
            rules = await self.db.kv_json_get("risk_rules")
        except Exception as e:
            log.warning("[risk] DB 读取风控规则失败，使用默认规则: %s", e)
            rules = None
        if not rules:
            rules = dict(DEFAULT_RULES)
            # DB 不可用时只读不写，避免重复异常
            try:
                await self.db.kv_json_set("risk_rules", rules)
            except Exception as e:
                log.warning("[risk] DB 写入风控规则失败（忽略）: %s", e)
        # 合并默认值：旧配置缺新字段时补默认
        changed = False
        for k, v in DEFAULT_RULES.items():
            if k not in rules:
                rules[k] = v
                changed = True
        if changed:
            try:
                await self.db.kv_json_set("risk_rules", rules)
            except Exception as e:
                log.warning("[risk] DB 写入合并后风控规则失败（忽略）: %s", e)
        self._rules_cache = rules
        self._rules_cache_ts = now
        # 首次从 DB 加载时记录更新时间（用于健康信号；DB 更新走 update_rules 单独记录）
        if self._rules_updated_at <= 0:
            self._rules_updated_at = now
        return rules

    async def update_rules(self, rules: dict) -> dict:
        merged = await self.get_rules()
        for k in ("max_loss_per_trade_usd", "max_daily_loss_usd", "min_order_value_usd",
                  "max_trades_per_hour", "max_position_pct",
                  "max_consecutive_losses", "cooldown_minutes",
                  "flash_crash_5min_drop_pct", "flash_crash_24h_drop_pct",
                  "flash_rally_5min_rise_pct", "flash_cooldown_minutes",
                  "max_vol_ratio", "risk_manual_recovery",
                  "vol_target_enabled", "vol_target_annual_pct", "vol_target_lookback_hours"):
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
        manual = self._manual_recovery
        awaiting = False
        if self._cooldown_until > 0 and remaining <= 0 and manual:
            awaiting = True  # 冷却时长已到，但需人工确认
        return {
            "active": (remaining > 0) or awaiting,
            "remaining_sec": max(0, int(remaining)),
            "consecutive_losses": self._consecutive_losses(),
            "flash_cooling": flash_remaining > 0,
            "flash_remaining_sec": max(0, int(flash_remaining)),
            "manual_recovery": manual,
            "awaiting_clear": awaiting,
        }

    def clear_cooldown(self) -> None:
        """人工确认解除冷却（冷却到期后 awaiting_clear 状态可调用）。

        人工确认视为接受当前连亏状态：同时重置连亏计数，避免解除后
        下一次 check 立即因历史连亏再次触发冷却（模块 C T3 验证路径）。
        """
        self._cooldown_until = 0.0
        self._manual_recovery = False
        self._recent_pnls = []
        log.info("[risk] 冷却已人工解除（连亏计数已重置）")

    def _add_price_sample(self, price: float) -> None:
        """记录价格样本，用于闪崩/24h 跌幅检测。

        - 节流：至少间隔 5 秒采一个点（ticker 事件约每秒一次，避免列表无限膨胀）
        - 保留 48 小时：支撑 24h 窗口检测（此前只留 2h，1h 周期下样本永远凑不满）
        - 使用 deque 实现 O(1) 两端操作，避免 O(n) 全量重建
        """
        now = time.time()
        if self._price_history and now - self._price_history[-1][0] < 5.0:
            return
        self._price_history.append((now, price))
        cutoff = now - 172800
        while self._price_history and self._price_history[0][0] < cutoff:
            self._price_history.popleft()

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
                    daily_pnl: float, trade_times: Iterable[float], entry_price: Optional[float],
                    vol_ratio: float = 1.0) -> tuple[bool, str]:
        """返回 (是否放行, 原因)。

        vol_ratio: 当前成交量与均量比值，用于成交量异常检测。
        daily_pnl: 调用方应传 await get_daily_pnl()（本类维护、跨日归零），
        自持副本会让「每日最大亏损」熔断跨日不解封。

        fail-safe：任何内部异常 → 按拦截处理（绝不放行新单），
        避免风控检查崩溃被事件总线吞掉后"引擎看似运行实则已死"
        （每次风控检查必炸，下单/止损/止盈静默停止）。
        """
        try:
            before = (self._cooldown_until, self._flash_cooldown_until, self._manual_recovery)
            result = await self._check(signal, price, cash, position_value,
                                       daily_pnl, trade_times, entry_price, vol_ratio)
            # 冷却/闪崩/人工确认状态变更时落库（稀有事件，值比较本身零成本）
            if (self._cooldown_until, self._flash_cooldown_until, self._manual_recovery) != before:
                await self.persist_state()
            return result
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("[risk] 风控检查异常，按拦截处理: %s", e)
            return False, f"风控检查异常（拦截下单）: {e}"


    def _realized_vol_annualized(self, lookback_hours: float) -> Optional[float]:
        """已实现年化波动率：价格历史对数收益 std × 年化系数。

        采样自 ticker（5s 节流），样本充足；不足 30 个样本或跨度 < 15 分钟
        时返回 None（不缩仓，避免用噪声估计）。纯 Python 实现（样本 ≤ 1.7 万，
        仅在信号时调用，不引 numpy 依赖）。
        """
        if len(self._price_history) < 30:
            return None
        cutoff = time.time() - lookback_hours * 3600.0
        pts = [(t, p) for t, p in self._price_history if t >= cutoff and p > 0]
        if len(pts) < 30 or pts[-1][0] - pts[0][0] < 900:
            return None
        rets = []
        for i in range(1, len(pts)):
            if pts[i][1] > 0 and pts[i - 1][1] > 0:
                rets.append(math.log(pts[i][1] / pts[i - 1][1]))
        if len(rets) < 20:
            return None
        n = len(rets)
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / max(n - 1, 1)
        sd = math.sqrt(var)
        # 年化：按样本平均间隔缩放（秒 → 年）
        span = pts[-1][0] - pts[0][0]
        avg_dt = max(span / n, 1.0)
        return sd * math.sqrt(31536000.0 / avg_dt)

    def _apply_vol_target(self, signal, rules: dict) -> None:
        """波动率目标仓位：按 目标波动/已实现波动 比例缩放 size_pct。

        只缩不加（scale ≤ 1.0）：低波动时不放大仓位，避免杠杆化；
        仅作用于比例开仓（size_pct）的买单，qty 明确的单不干预。
        """
        if signal.side != "buy" or signal.qty is not None:
            return
        if not rules.get("vol_target_enabled"):
            return
        lookback = float(rules.get("vol_target_lookback_hours", 24.0))
        target = float(rules.get("vol_target_annual_pct", 0.40))
        if target <= 0:
            return
        rv = self._realized_vol_annualized(lookback)
        if rv is None or rv <= 0:
            return
        scale = min(1.0, target / rv)
        if scale >= 0.98:
            return
        old_pct = float(signal.size_pct or 0.5)
        new_pct = max(0.01, old_pct * scale)
        signal.size_pct = new_pct
        log.info("[risk] 波动率目标仓位：已实现波动 %.1f%% > 目标 %.1f%%，仓位 %.0f%% → %.0f%%",
                 rv * 100, target * 100, old_pct * 100, new_pct * 100)

    async def _check(self, signal, price: float, cash: float, position_value: float,
                     daily_pnl: float, trade_times: Iterable[float], entry_price: Optional[float],
                     vol_ratio: float = 1.0) -> tuple[bool, str]:
        """风控检查主体（check() 的 fail-safe 包装内执行）。"""
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
            try:
                from core.notify import notify
                await notify(self.db, "风控熔断：闪崩保护",
                             f"{flash_reason}，触发 {cooldown_min} 分钟冷却")
            except Exception:  # noqa: BLE001
                pass
            return False, flash_reason + f"，触发 {cooldown_min} 分钟冷却"

        # 0.2) 成交量异常保护
        vol_reason = self._check_vol_spike(vol_ratio, rules)
        if vol_reason and not closing:
            log.warning("[risk] %s", vol_reason)
            return False, vol_reason

        # 0.3) 波动率目标仓位：高波动时自动缩仓（只缩不加）
        self._apply_vol_target(signal, rules)

        # 1) 订单金额：qty 明确时 = qty*price；比例下单时按 cash*size_pct 估算实际金额
        if signal.qty is not None:
            order_value = signal.qty * price
        else:
            order_value = cash * float(signal.size_pct or 0.5)

        # 2) 连续亏损冷却保护（仅限新开仓）
        # closing（平仓豁免）：止损/止盈单即使处于冷却或人工确认模式也绝不拦截
        if not closing and self._cooldown_until > 0:
            remaining = int(self._cooldown_until - time.time())
            if remaining > 0:
                return False, f"连续亏损冷却中，剩余 {remaining}s（暂停新开仓）"
            # 冷却时长已到：开启人工确认模式时等待 clear_cooldown()，否则自动恢复
            if self._manual_recovery:
                return False, "连续亏损冷却已到期，等待人工确认解除（暂停新开仓）"
            self._manual_recovery = False
        losses = self._consecutive_losses()
        if losses >= int(rules.get("max_consecutive_losses", 3)) and not closing:
            self._cooldown_until = time.time() + int(rules.get("cooldown_minutes", 15)) * 60
            self._manual_recovery = bool(rules.get("risk_manual_recovery", False))
            log.warning("[risk] 连续 %d 笔亏损，触发 %s 分钟冷却%s",
                        losses, rules.get("cooldown_minutes", 15),
                        "（人工确认模式）" if self._manual_recovery else "")
            try:
                from core.notify import notify
                await notify(self.db, "风控熔断：连续亏损",
                             f"连续 {losses} 笔亏损，冷却 {rules.get('cooldown_minutes', 15)} 分钟"
                             + ("（需人工确认解除）" if self._manual_recovery else ""))
            except Exception:  # noqa: BLE001
                pass
            return False, f"连续 {losses} 笔亏损，触发冷却 {rules.get('cooldown_minutes', 15)} 分钟"

        # 3) 最小订单金额（仅限新开仓：小仓位止损单必须能离场）
        if order_value < rules["min_order_value_usd"] and not closing:
            return False, f"订单金额 ${order_value:.2f} 低于最小限额 ${rules['min_order_value_usd']:.2f}"

        # 4) 每日最大亏损（仅限新开仓，保证持仓可随时离场）
        if daily_pnl <= -rules["max_daily_loss_usd"] and not closing:
            try:
                from core.notify import notify
                await notify(self.db, "风控熔断：每日亏损上限",
                             f"今日已亏损 ${-daily_pnl:.2f}，达上限 ${rules['max_daily_loss_usd']}，暂停新开仓")
            except Exception:  # noqa: BLE001
                pass
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
        # 最小时间跨度约束：曾只数样本数（5s 节流 50 秒就凑够 10 个），
        # 刚启动的新实例会把"2 分钟跌 25%"误判为 24 小时跌幅触发冷却
        # 时间跨度取自 _price_history 元组首尾 ts（_price_history 保留 48h）——
        # 曾对 float 元素做 prices_24h[-1][0] 下标 → TypeError 风控静默失效
        span_ok = len(prices_24h) >= 10 and (self._price_history[-1][0] - self._price_history[0][0]) >= 2 * 3600
        if span_ok and not closing:
            drop_24h = (prices_24h[0] - prices_24h[-1]) / prices_24h[0]
            threshold_24h = rules.get("flash_crash_24h_drop_pct", 0.25)
            if drop_24h >= threshold_24h:
                cooldown_min = int(rules.get("flash_cooldown_minutes", 30))
                self._flash_cooldown_until = time.time() + cooldown_min * 60
                log.warning("[risk] 24小时跌幅 %.1f%% >= %.0f%%，触发冷却", drop_24h * 100, threshold_24h * 100)
                try:
                    from core.notify import notify
                    await notify(self.db, "风控熔断：24小时跌幅",
                                 f"24h 跌幅 {drop_24h:.1%} ≥ {threshold_24h:.0%}，触发 {cooldown_min} 分钟冷却")
                except Exception:  # noqa: BLE001
                    pass
                return False, f"24小时跌幅 {drop_24h:.1%}，触发 {cooldown_min} 分钟冷却"

        return True, ""
