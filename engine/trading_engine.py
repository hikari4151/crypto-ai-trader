"""总调度引擎：行情→策略→风控→下单，调度 AI 任务 + AI 策略工坊。"""
import asyncio
import json
import logging
import math
import re
import time
from collections import deque
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from ai import AIClient, MarketAnalyst, ParamOptimizer, StrategyIteration, TradeReviewer
from ai.strategy_designer import StrategyDesigner
from drl.evolve_engine import EvolveEngine
from backtest.metrics import compute_metrics
from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.manager import ExchangeManager
from exchange.paper import PaperAccount
from exchange.symbols import base_currency, parse_symbol, quote_currency
from exchange.ws_market import MarketDataHub
from strategies import get_strategy, list_strategies
from strategies.base import Signal
from .order_manager import OrderManager
from .portfolio import PortfolioManager
from .risk import RiskManager

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, db: Database, bus: EventBus) -> None:
        self.db = db
        self.bus = bus
        self.paper = settings.paper_trading
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self.exchange_id = settings.default_exchange
        self.symbol = settings.default_symbol
        self.timeframe = settings.default_timeframe
        self.start_cash = settings.default_start_cash
        self.paper_fee_rate = 0.001
        self.paper_slippage = settings.paper_slippage
        # 成交时点：True=信号在下一根K线开盘成交（与回测一致），False=当前K线收盘即时成交
        self.trade_on_open = bool(settings.trade_on_open)
        # 待成交信号（trade_on_open 时：上一根收盘产生、待下一根开盘成交的信号，(signal, 挂单尝试次数)）
        self._pending_signal: Optional[tuple] = None

        self.paper_account = PaperAccount(start_cash=settings.default_start_cash) if self.paper else None
        self.exchange: Optional[ExchangeManager] = None
        self.hub: Optional[MarketDataHub] = None
        self.order_manager = OrderManager(db, bus, self.paper, self.paper_account)
        self.portfolio = PortfolioManager(db, bus, self.paper, self.paper_account, None)
        self.risk = RiskManager(db)
        self.ai_client = AIClient(db, settings.model_dump())
        self.analyst = MarketAnalyst(self.ai_client, db, bus)
        self.optimizer = ParamOptimizer(self.ai_client, db)
        self.reviewer = TradeReviewer(self.ai_client, db)
        self.designer = StrategyDesigner(self.ai_client, db, bus)
        self.iteration = StrategyIteration(self.ai_client, db)
        # 持续进化引擎（DRL 因子挖掘 + 策略训练自动持续学习）
        self.evolve = EvolveEngine(db, bus, settings.data_dir / "models")

        self.strategy = get_strategy(settings.default_strategy)
        # 当前策略的 MA 快/慢线周期（dual_ma 的 fast_period/slow_period 生效，
        # 回测与实时关键路径同一来源 strategies.base.strategy_ma_periods）
        self._ma_fast_period, self._ma_slow_period = self._strategy_ma_periods()
        self._trade_times: deque = deque()
        # 每日盈亏不在此处副本：单一来源在 RiskManager（跨日归档 + KV 持久化）
        self._paper_equity = settings.default_start_cash
        self._last_analysis = 0.0
        self._last_optimize = 0.0
        self._last_review = 0.0
        # 生命周期互斥锁：防止 start/stop 并发竞态（重复下单风险）
        self._lifecycle_lock = asyncio.Lock()
        # K线处理互斥锁：事件总线并发分发时（重连补历史等场景），
        # 多个 MARKET_CANDLE 事件可能同时进入 _on_candle——风控检查与下单
        # 之间存在 await 点，并发执行会导致重复下单，必须串行化
        self._candle_lock = asyncio.Lock()
        # 策略切换/参数热更新 vs K 线处理互斥（必须在 __init__ 创建：
        # 曾只在 start 时创建，引擎未启动时 select_strategy 等直接 AttributeError 500）
        self._strategy_lock = asyncio.Lock()
        # 主事件循环引用：后台 AI worker 线程经 run_coroutine_threadsafe 把
        # 参数应用调度回主循环（持 _strategy_lock），避免跨循环用锁
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        # 实盘余额缓存（避免每根K线打 REST）
        self._live_balance: dict = {}
        self._live_balance_ts: float = 0.0
        # 每日对账状态（T2）
        self._daily_reconciled_today: bool = False
        self._last_reconcile_date: str = ""
        # 断链平仓状态（T5）
        self._stale_alerted: bool = False

        bus.subscribe(EventType.MARKET_CANDLE, self._on_candle)
        # 高频价格采样：供风控闪崩/暴涨检测使用（hub 约每秒发布一次 ticker）
        bus.subscribe(EventType.MARKET_TICKER, self._on_ticker)
        # 成交事件：对账回灌的迟到成交（late=True）走公共记账 _record_fill；
        # 普通成交事件（即时路径已直接记账）忽略
        bus.subscribe(EventType.ORDER_FILL, self._on_order_fill)

    async def _load_live_config(self) -> None:
        """从数据库 KV 读取实盘/模拟启动配置（交易所/交易对/周期/初始资金/手续费）。"""
        try:
            mode = await self.db.kv_get("trading_mode", "paper")
            self.trading_mode = mode
            cfg = await self.db.kv_json_get("live_trading_config")
            if cfg:
                self.exchange_id = cfg.get("exchange") or self.exchange_id
                self.symbol = cfg.get("symbol") or self.symbol
                self.timeframe = cfg.get("timeframe") or self.timeframe
                # 启动策略：实盘启动参数可指定策略名，启动时应用（无效策略名回退默认，不阻断启动）
                # 必须位于 if cfg 守卫内：全新数据库无该 KV 时 cfg 为 None
                try:
                    strat = cfg.get("strategy") or ""
                    if strat and strat != self.strategy.name:
                        await self.select_strategy(strat)
                        log.info("[engine] 应用实盘启动策略: %s", strat)
                except (ValueError, KeyError) as e:
                    log.warning("[engine] 应用实盘启动策略 %s 失败（保持默认 %s）: %s",
                                cfg.get("strategy"), self.strategy.name, e)
            # 模拟实盘：用真实行情 + 模拟账户撮合（paper=True，不真实下单）
            # 实盘：真实下单（paper=False）
            self.paper = (mode in ("paper", "simulated"))
            # 初始资金按模式取来源：
            # - live/simulated（模拟实盘=用实盘参数做模拟验证）：用实盘配置的资金
            # - paper：用模拟配置的资金（曾 simulated 只读模拟配置，
            #   用户在"实盘配置"窗口设置的资金不生效）
            # 防御：DB 中可能残留历史 NaN 值（校验被绕过写入），非有限值回退默认
            if mode in ("live", "simulated") and (cfg or {}).get("start_cash"):
                v = float(cfg["start_cash"])
                if math.isfinite(v) and v > 0:
                    self.start_cash = v
            # 模拟模式配置：手续费率（纸面/模拟实盘共用可自定义费率）；
            # 纸面模式的资金也来自本配置
            paper_cfg = await self.db.kv_json_get("paper_trading_config")
            if paper_cfg:
                if mode == "paper":
                    v = float(paper_cfg.get("start_cash") or self.start_cash)
                    if math.isfinite(v) and v > 0:
                        self.start_cash = v
                f = float(paper_cfg.get("fee_rate") or self.paper_fee_rate)
                if math.isfinite(f) and 0 <= f <= 0.1:
                    self.paper_fee_rate = f
                s = float(paper_cfg.get("slippage") or self.paper_slippage)
                if math.isfinite(s) and 0 <= s <= 0.1:
                    self.paper_slippage = s
                if "trade_on_open" in paper_cfg:
                    self.trade_on_open = bool(paper_cfg["trade_on_open"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:  # 读取配置：JSON 解析/键缺失/类型错误属于已知异常类型
            log.warning("[engine] 读取配置失败: %s", e)

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        async with self._lifecycle_lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        if self.running:
            return
        await self._load_dynamic_strategies()
        await self._load_live_config()  # 读取实盘/模拟配置
        mode = getattr(self, "trading_mode", "paper")
        # 快照启动前状态：启动失败时复位（P1-5——失败后引擎状态干净、可重试）
        saved = (self.paper, self.paper_account, self.order_manager,
                 self.portfolio, self.exchange, self.hub)
        tasks_created: list[asyncio.Task] = []
        try:
            # 纸面 / 模拟实盘：用模拟账户撮合（不下真实单）
            if mode in ("paper", "simulated"):
                self.paper_account = PaperAccount(start_cash=self.start_cash, fee_rate=self.paper_fee_rate,
                                                  slippage=self.paper_slippage)
                self.order_manager = OrderManager(self.db, self.bus, True, self.paper_account)
                self.portfolio = PortfolioManager(self.db, self.bus, True, self.paper_account, None)
            # 读取交易所密钥（解密失败降级为空串，避免纸面/模拟模式因密钥损坏无法启动）
            try:
                api_key = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_api_key") or ""
                secret = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_secret") or ""
                password = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_password") or ""
            except Exception as e:  # noqa: BLE001
                log.warning("[engine] 读取交易所密钥失败（降级为空）: %s", e)
                api_key = secret = password = ""

            # 实盘：真实下单。模拟实盘：若配了密钥也连交易所（验证连接/看真实余额），但订单仍模拟
            if mode == "live":
                self.exchange = ExchangeManager(self.exchange_id)
                await self.exchange.start(api_key, secret, password)
                # 关键：重建订单/组合管理器走真实下单——否则沿用纸面模式下创建的
                # OrderManager（paper 标志为 True），实盘订单会被静默撮合进纸面账户
                self.paper_account = None
                self.paper = False
                self.order_manager = OrderManager(self.db, self.bus, False, None)
                self.order_manager.attach_exchange(self.exchange)
                # 兜底止损的触发深度跟随策略软止损（热更新参数后自动生效）
                self.order_manager.set_stop_pct_source(
                    lambda: getattr(self.strategy, "params", {}).get("stop_loss_pct"))
                self.portfolio = PortfolioManager(self.db, self.bus, False, None, self.exchange)
            elif mode == "simulated" and api_key and secret:
                self.exchange = ExchangeManager(self.exchange_id)
                await self.exchange.start(api_key, secret, password)
                log.info("[engine] 模拟实盘：已连接 %s（仅展示余额，订单走模拟撮合）", self.exchange_id)

            # 预热风控的每日盈亏（跨重启恢复 + 跨日归档），引擎不再持有副本
            try:
                await self.risk.get_daily_pnl()
            except (ValueError, KeyError) as e:  # 恢复每日盈亏：值/键错误属于已知异常类型
                log.warning("[engine] 恢复每日盈亏失败: %s", e)
            # 连亏/冷却/人工确认状态同样要跨重启保留：曾纯内存，
            # 重启一次就把"连续亏损暂停交易"这道防线清零
            await self.risk.restore_state()
            if mode == "live":
                # 实盘持仓状态回放（FIFO 成本队列 + 策略入口价），否则重启后无止损
                await self._restore_live_position_state()

            self.hub = MarketDataHub(self.exchange_id, [self.symbol], [self.timeframe], self.bus,
                                     ma_fast_period=self._ma_fast_period, ma_slow_period=self._ma_slow_period)
            self.running = True
            # 启动时清空上一轮残留的待成交信号（跨重启不携带）
            self._pending_signal = None
            # 逐个收集局部列表：列表推导式中途 create_task 抛异常会丢失已建任务引用
            # （孤儿任务不被取消，下次 start 新旧任务并存重复下单）
            for name, coro in (
                ("hub", self.hub.run()),
                ("portfolio", self._portfolio_loop()),
                ("ai_scheduler", self._ai_scheduler()),
                ("event_bus", self.bus.run()),  # 启动事件分发循环
            ):
                tasks_created.append(asyncio.create_task(coro, name=name))
            self._tasks = tasks_created
            # 启动持续进化引擎（后台训练循环，不阻塞主流程）
            try:
                await self.evolve.start()
            except Exception as e:  # noqa: BLE001
                log.warning("[engine] 启动持续进化引擎失败: %s", e)
            await self.bus.publish(Event(EventType.ENGINE_STATE, {"state": "running"}, source="engine"))
            log.info("交易引擎启动: paper=%s exchange=%s symbol=%s strategy=%s",
                     self.paper, self.exchange_id, self.symbol, self.strategy.name)
        except Exception:  # noqa: BLE001
            # P1-5 失败路径：清理新建资源 + 恢复快照 + running=False + 取消已建任务，
            # 然后 re-raise——main.py lifespan 降级逻辑与 /api/trading/start 的 500 行为保持不变
            log.exception("[engine] 引擎启动失败，复位状态")
            if self.exchange is not None:
                try:
                    await self.exchange.close()
                except Exception:  # noqa: BLE001
                    pass
            (self.paper, self.paper_account, self.order_manager,
             self.portfolio, self.exchange, self.hub) = saved
            self.running = False
            self._tasks = []
            for t in tasks_created:
                t.cancel()
            if tasks_created:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_created, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            raise

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if not self.running:
                return
            self.running = False
            tasks = self._tasks
            self._tasks = []
            # 清空待成交信号，避免重启后误成交上一轮的信号
            self._pending_signal = None
            for t in tasks:
                t.cancel()
            if self.hub:
                self.hub.stop()
            # 等待所有任务真正退出，避免快速重启时新旧任务并存导致重复下单。
            # 带超时：live 模式下取消中的 _on_candle 可能卡在交易所请求上，
            # 曾无 timeout 极端时 stop 永久挂起（孤儿单由交易所侧对账兜底）
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
                except asyncio.TimeoutError:
                    log.warning("[engine] 任务退出超时，强制继续停止流程")
            if self.exchange:
                await self.exchange.close()
            # 关闭订单对账任务（防泄漏到下次启动；订单注册表随 stop 清空，
            # 交易所侧残留挂单由人工/交易所兜底，stop 不强行撤单）
            try:
                await self.order_manager.close()
            except Exception as e:  # noqa: BLE001
                log.warning("[engine] 关闭订单对账任务失败: %s", e)
            # 停止持续进化引擎
            try:
                await self.evolve.stop()
            except Exception as e:  # noqa: BLE001
                log.warning("[engine] 停止持续进化引擎失败: %s", e)
            await self.bus.publish(Event(EventType.ENGINE_STATE, {"state": "stopped"}, source="engine"))
            try:
                from core.notify import notify
                await notify(self.db, "引擎已停止",
                             f"{self.exchange_id} {self.symbol} {self.timeframe} · 策略 {self.strategy.name}")
            except Exception:  # noqa: BLE001
                pass
            log.info("交易引擎已停止")

    def status(self) -> dict:
        # 追加 degraded/bus 键（现有 6 键顺序与取值不变）：
        # /api/health 经 web/main.py 原样带出 engine.degraded（零改动接线）
        b = self.bus.health()
        out = {"running": self.running, "paper": self.paper,
               "mode": getattr(self, "trading_mode", "paper" if self.paper else "live"),
               "exchange": self.exchange_id,
               "symbol": self.symbol, "timeframe": self.timeframe, "strategy": self.strategy.name,
               "degraded": b["degraded"], "bus": b["handlers"]}
        prot = getattr(self.order_manager, "protective", None)
        out["protective_stop"] = prot.status() if prot is not None else None
        return out

    async def _load_dynamic_strategies(self) -> None:
        """启动时从数据库恢复 AI 设计的策略（与 web 启动路径同一实现）。"""
        from strategies.dynamic_store import restore_from_db
        count = await restore_from_db(self.db)
        log.info("已加载 %d 个 AI 设计策略", count)

    # ---------------- 行情处理 ----------------
    async def _on_ticker(self, event: Event) -> None:
        """高频价格采样：供风控闪崩/暴涨检测使用（hub 约每秒发布一次）。"""
        if not self.running:
            return
        if event.payload.get("symbol") != self.symbol:
            return
        ticker = event.payload.get("ticker") or {}
        price = float(ticker.get("last") or 0.0)
        if price > 0:
            self.risk._add_price_sample(price)

    async def _live_balance_cached(self) -> dict:
        """实盘余额缓存（30 秒刷新），避免每根K线都打 REST。

        失败时保留上次成功的缓存：曾清空为 {}，导致 position=0 → 风控
        "平仓豁免"判定失效，止损卖单在冷却规则下再次被拦截（行情剧烈、
        交易所限流概率最高的场景）。缓存陈旧的风险远小于丢持仓口径。
        """
        now = time.time()
        if now - self._live_balance_ts > 30:
            try:
                self._live_balance = await self.exchange.fetch_balance() if self.exchange else {}
                self._live_balance_ts = now
            except Exception as e:  # noqa: BLE001
                # 保留上次成功缓存，不清空
                log.warning("[engine] 获取实盘余额失败(保留旧缓存): %s", e)
        return self._live_balance

    async def _on_candle(self, event: Event) -> None:
        # 串行化：防止并发 MARKET_CANDLE 事件绕过风控/频率检查导致重复下单
        async with self._candle_lock:
            await self._on_candle_locked(event)

    async def _on_candle_locked(self, event: Event) -> None:
        # 策略锁：select_strategy/update_strategy_params 与 K 线处理整段互斥。
        # 曾无锁——切换策略可在 on_candle 与 on_fill 之间发生：旧策略信号配新策略
        # on_fill（_entry 成本基被污染），参数热更新读到混合参数
        async with self._strategy_lock:
            await self._on_candle_locked_inner(event)

    async def _on_candle_locked_inner(self, event: Event) -> None:
        snap = event.payload.get("snapshot", {})
        if not snap or not self.running:
            return
        symbol = snap.get("symbol")
        if symbol != self.symbol:
            return
        indicators = snap.get("indicators", {})
        price = float(indicators.get("close") or 0.0)
        vol_ratio = float(indicators.get("vol_ratio") or 1.0)

        # ---- 成交时点：trade_on_open=True 时，先在当前K线开盘成交上一根收盘产生的信号 ----
        # （与回测引擎口径一致：信号在第 i 根收盘产生、第 i+1 根开盘成交，杜绝前视）
        if self.trade_on_open and self._pending_signal is not None:
            candles = snap.get("candles") or []
            if candles:
                open_ = float(candles[-1][1])
                high_ = float(candles[-1][2])
                low_ = float(candles[-1][3])
            else:
                open_ = high_ = low_ = 0.0
            sig, tries = self._pending_signal
            self._pending_signal = None
            fill_ref: Optional[float] = None
            lim = getattr(sig, "limit_price", None)
            if getattr(sig, "order_type", "market") == "limit" and lim is not None:
                lp = float(lim)
                if low_ <= lp <= high_:
                    # 限价触及成交（与 backtest _limit_fill_ratio 的 OHLC 触碰逻辑同口径）
                    fill_ref = lp
                else:
                    # 未触及：挂单最多重试 3 根K线（与 backtest _LIMIT_MAX_TRIES 一致），超时取消
                    tries += 1
                    if tries < 3:
                        self._pending_signal = (sig, tries)
                    else:
                        log.info("[engine] 限价单挂单超时取消（3根K线未触及）: %s %s", sig.side, sig.symbol)
            elif open_ > 0:
                fill_ref = open_
            if fill_ref is not None:
                await self._execute_signal(sig, fill_ref, vol_ratio)

        # ---- 当前K线收盘：生成信号 ----
        position, cash_now, pos_value, entry_price = await self._current_state(symbol, price)
        signal = self.strategy.on_candle({
            "symbol": symbol, "price": price, "position": position,
            "cash": cash_now,
            "indicators": indicators, "timeframe": self.timeframe,
        })
        if not signal:
            return
        if self.trade_on_open:
            # 延迟到下一根K线开盘成交（先记信号，不立即下单）
            self._pending_signal = (signal, 0)
            return
        # 默认（向后兼容）：当前收盘即时执行
        await self._execute_signal(signal, price, vol_ratio)

    async def _current_state(self, symbol: str, price: float) -> tuple[float, float, float, Optional[float]]:
        """计算当前持仓/现金/持仓价值/策略入场价（成交与信号生成共用，保持同口径）。

        price: 估值参考价（成交场景传成交参考价：开盘/收盘；信号场景传收盘价）。
        """
        self.portfolio.update_prices({symbol: price})
        if self.paper and self.paper_account:
            self._paper_equity = self.paper_account.cash + self.paper_account.positions_value({symbol: price})
        position = 0.0
        cash_now = 0.0
        pos_value = 0.0
        if self.paper and self.paper_account:
            position = self.paper_account.positions.get(symbol, {}).get("qty", 0.0)
            cash_now = self.paper_account.cash
            pos_value = self.paper_account.positions_value({symbol: price})
        elif self.exchange and self.exchange.has_private:
            # 实盘：用真实余额参与风控，避免 0 余额导致仓位上限恒拦截买入
            bal = await self._live_balance_cached()
            base, quote, _settle = parse_symbol(symbol)
            position = float(bal.get(base, {}).get("total", 0.0) or 0.0)
            cash_now = float(bal.get(quote, {}).get("free", 0.0) or 0.0)
            pos_value = position * price
        # 策略入口价属性名不统一：price_action/rl_adaptive 用 _entry，dual_ma 用 _entry_price，两者都兼容
        entry_price = getattr(self.strategy, "_entry", None)
        if entry_price is None:
            entry_price = getattr(self.strategy, "_entry_price", None)
        return position, cash_now, pos_value, entry_price

    async def _restore_live_position_state(self) -> None:
        """实盘重启：回放 Trade 重建 FIFO 成本队列并回填策略入口价。

        两者都只在内存里，重启即丢：入口价丢失后 price_action/dual_ma 的
        止损判据 `position > 0 and self._entry` 恒不成立，交易所里的真实持仓
        再无止损；FIFO 队列丢失后卖出成本走兜底、Trade.pnl 失真。
        """
        try:
            state = await self.order_manager.restore_live_state(self.symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 实盘持仓状态回放失败（止损/成本将走兜底）: %s", e)
            return
        qty = float(state.get("qty") or 0.0)
        entry = float(state.get("entry") or 0.0)
        if qty <= 0 or entry <= 0:
            return
        # 策略入口价属性名不统一（price_action/rl_adaptive 用 _entry，dual_ma 用 _entry_price）
        for attr in ("_entry", "_entry_price"):
            if hasattr(self.strategy, attr):
                setattr(self.strategy, attr, entry)
        log.warning("[engine] 重启回填实盘持仓: %s qty=%s entry=%s（止损/止盈恢复生效）",
                    self.symbol, qty, entry)
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "position_state_restored", "symbol": self.symbol, "qty": qty, "entry": entry,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, source="engine"))

    async def _execute_signal(self, signal, exec_price: float, vol_ratio: float,
                              bypass_risk: bool = False) -> None:
        """风控检查 → 发布 SIGNAL → 下单 → 记账（即时成交与延迟成交共用执行路径）。

        bypass_risk=True 用于保护性平仓：止损/断连平仓不能被冷却、频率或
        亏损上限拦住（否则最需要平仓时反而平不掉），但仍照常记账。

        调用方须在 _strategy_lock 内（_on_candle_locked 已持；延迟路径同在锁内）。
        exec_price: 市价单的成交参考价（延迟路径=当前K线开盘，即时路径=当前K线收盘），
        滑点由 order_manager 按模式叠加（纸面叠加 paper_slippage，实盘以交易所真实成交价为准）。
        """
        symbol = signal.symbol
        position, cash_now, pos_value, entry_price = await self._current_state(symbol, exec_price)
        if bypass_risk:
            allowed, reason = True, ""
        else:
            # 每日盈亏向风控现取：只有它会在日历跨日时归零。引擎旧副本
            # 全程只增不减，熔断一旦触发就要等到重启才解除
            allowed, reason = await self.risk.check(signal, exec_price, cash_now, pos_value,
                                                    await self.risk.get_daily_pnl(),
                                                    self._trade_times, entry_price,
                                                    vol_ratio=vol_ratio)
        if not allowed:
            await self.bus.publish(Event(EventType.RISK_BLOCKED, {"signal": signal.__dict__, "reason": reason}, source="risk"))
            log.warning("[risk] 拦截信号 %s: %s", signal.side, reason)
            return

        await self.bus.publish(Event(EventType.SIGNAL, signal.__dict__, source="engine"))
        fill = await self.order_manager.place(signal, exec_price, self.strategy.name)
        if fill:
            # 成交后处理走公共记账方法（即时路径；回灌迟到成交走 _on_order_fill）
            await self._record_fill(symbol, signal.side, fill, self.strategy.name,
                                    entry_price, reason=signal.reason, signal_dict=signal.__dict__)

    async def _on_order_fill(self, event: Event) -> None:
        """成交回灌处理：仅处理 order_manager 对账回灌的迟到成交（late=True）。

        普通成交事件（即时路径）已由 _on_candle_locked_inner 直接记账，这里
        忽略（订阅前无人消费，等价无操作，避免重复记账）。整体持 _strategy_lock
        与即时路径互斥（on_fill 写策略状态）。
        """
        if not event.payload.get("late"):
            return
        p = event.payload
        async with self._strategy_lock:
            # 策略入口价属性名不统一（price_action/rl_adaptive 用 _entry，
            # dual_ma 用 _entry_price，两者都兼容）——与即时路径同口径
            entry_price = getattr(self.strategy, "_entry", None)
            if entry_price is None:
                entry_price = getattr(self.strategy, "_entry_price", None)
            await self._record_fill(
                p.get("symbol", ""), p.get("side", ""),
                {"price": p.get("price", 0.0), "qty": p.get("qty", 0.0),
                 "fee": p.get("fee", 0.0), "trade_id": p.get("trade_id"),
                 "cost_price": p.get("cost_price")},
                p.get("strategy_name", ""), entry_price=entry_price)

    async def _record_fill(self, symbol: str, side: str, fill: dict, strategy_name: str,
                           entry_price: Optional[float], reason: str = "",
                           signal_dict: Optional[dict] = None) -> None:
        """成交后处理公共记账（即时成交与对账回灌迟到成交共用一套逻辑）：
        _trade_times 记录、strategy.on_fill、卖出时 risk.record_trade（每日盈亏
        唯一来源）/update_trade_pnl 同口径更新、TRADE 事件发布。

        调用方须保证在 _strategy_lock 内（即时路径天然在锁内，回灌路径
        _on_order_fill 显式持锁）。
        """
        # P2-2：入队时剪枝——只保留最近 1 小时内的成交时间（risk.check 的频率
        # 限制只用 cutoff=now-3600 窗口过滤），防止列表无限增长、每次风控检查
        # 的 O(n) 过滤越拖越慢（append 单调递增，头部弹出即正确）
        _now = time.time()
        self._trade_times.append(_now)
        while self._trade_times and self._trade_times[0] < _now - 3600:
            self._trade_times.popleft()
        self.strategy.on_fill(symbol, side, fill["price"])
        if side == "sell":
            # M1：成本基准与回测引擎对齐——FIFO 逐笔记账（_place_paper 从
            # PaperAccount 的 lots 队列摊销，_place_live 从 _live_lots 队列
            # 摊销，成交时已随 fill 返回 cost_price，与回测 lots.pop(0) 同款
            # 口径）；entry_price 仅作兜底（单点，会被最后一次买入价覆盖）
            cost_price = fill.get("cost_price")
            if not cost_price:
                # 0.0 同样兜底（队列无消费/无成本参考时 FIFO 返回 0.0 / None）
                cost_price = entry_price
            # 无入场价跟踪的策略（如 grid 不维护 _entry）无法算真实盈亏：
            # 置 0 而非记 -fee，否则每日盈亏被手续费污染、连续亏损冷却误触发
            if cost_price is None:
                pnl = 0.0
            else:
                pnl = (fill["price"] - cost_price) * fill["qty"] - fill["fee"]
            # 记录平仓盈亏给风控（连续亏损冷却判断 + 每日盈亏累计持久化）
            await self.risk.record_trade(pnl, reason)
            # 平仓盈亏回写 Trade 行：曾只累加内存/风控 KV，性能页的
            # win_rate/total_pnl/avg_win/avg_loss 与 AI 复盘按 pnl=0 聚合全部失真
            if fill.get("trade_id"):
                try:
                    await self.db.update_trade_pnl(fill["trade_id"], pnl)
                except (SQLAlchemyError, asyncio.TimeoutError) as e:  # 平仓盈亏落库：DB/超时错误属于已知异常类型
                    log.warning("[engine] 平仓盈亏落库失败: %s", e)
        if signal_dict is None:
            # 回灌路径无原始 Signal：构造最小 dict（TRADE 事件仅作展示/留痕）
            signal_dict = {"symbol": symbol, "side": side, "qty": fill.get("qty", 0.0),
                           "reason": reason, "strategy": strategy_name}
        await self.bus.publish(Event(EventType.TRADE, {"fill": fill, "signal": signal_dict}, source="engine"))

    # ---------------- 定时任务 ----------------
    async def _analysis_context(self, snap: dict) -> tuple[float, list]:
        """行情分析上下文：当前持仓数量 + 近期平仓记录（供 AI 解读联动账户状态）。"""
        symbol = snap.get("symbol", self.symbol)
        pos = 0.0
        if self.paper and self.paper_account:
            pos = self.paper_account.positions.get(symbol, {}).get("qty", 0.0)
        elif self.exchange and self.exchange.has_private:
            try:
                bal = await self._live_balance_cached()
                pos = float(bal.get(base_currency(symbol), {}).get("total", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                pass
        recent = await self._recent_trades(10)
        return float(pos), recent

    async def _portfolio_loop(self) -> None:
        while self.running:
            try:
                await self.portfolio.snapshot()
                await self._daily_reconcile_if_due()   # T2
                await self._check_market_stale()        # T5
            except Exception as e:  # noqa: BLE001
                log.warning("[portfolio] 快照失败: %s", e)
            await asyncio.sleep(15)

    # ---------------- T2 每日对账 ----------------
    async def _daily_reconcile_if_due(self) -> None:
        """每日 0 点做一次交易所全量持仓对账（仅 live 模式）。

        放宽为"0 点后 5 分钟内补跑一次"（_daily_reconciled_today 防重复），
        跨 0 点重置。此为刻意容错。
        """
        today = time.strftime("%Y-%m-%d")
        if self._last_reconcile_date != today:
            self._daily_reconciled_today = False
            self._last_reconcile_date = today
        if self._daily_reconciled_today:
            return
        trading_mode = getattr(self, "trading_mode", "paper")
        if trading_mode != "live":
            return  # 纸面/模拟无真实余额可比
        from datetime import datetime
        now = datetime.now()
        if not (now.hour == 0 and 0 <= now.minute < 5):
            return
        self._daily_reconciled_today = True  # 标记已处理（今天不再重跑）
        await self._reconcile_positions()

    async def _reconcile_positions(self) -> None:
        """全量对比本地持仓与交易所钱包，发布 SYSTEM 事件。"""
        try:
            api_key = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_api_key") or ""
            secret = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_secret") or ""
            password = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_password") or ""
            if not api_key or not secret:
                log.warning("[reconcile] 未配置交易所密钥，跳过每日对账")
                return
            bal = await self.portfolio.fetch_live_balance(api_key, secret, password)
            # 交易所钱包持仓（asset -> qty）
            exchange_pos: dict[str, float] = {}
            for asset, qty in bal.get("total", {}).items():
                try:
                    q = float(qty or 0.0)
                    if q > 0:
                        exchange_pos[asset] = q
                except (TypeError, ValueError):
                    pass
        except Exception as e:  # noqa: BLE001
            log.warning("[reconcile] 每日对账失败（交易所不可达，改日重试）: %s", e)
            self._daily_reconciled_today = False  # 失败改日重试
            return

        # ---- 本地持仓快照 ----
        local_pos: dict[str, float] = {}
        if self.exchange and self.exchange.has_private:
            try:
                lb = await self._live_balance_cached()
                for asset, qty in lb.get("total", {}).items():
                    try:
                        q = float(qty or 0.0)
                        if q > 0:
                            local_pos[asset] = q
                    except (TypeError, ValueError):
                        pass
            except Exception:  # noqa: BLE001
                pass
        elif self.paper and self.paper_account:
            for sym, p in self.paper_account.positions.items():
                local_pos[base_currency(sym)] = p["qty"]
        else:
            log.warning("[reconcile] 本地持仓来源不可用，跳过")
            return

        # ---- 排除计价币（USDT 等 quote 币不视为持仓） ----
        quote = quote_currency(self.symbol) or "USDT"
        local_pos.pop(quote, None)
        exchange_pos.pop(quote, None)

        # ---- 差异判定与告警 + SYSTEM 事件 ----
        syms = set(local_pos) | set(exchange_pos)
        for asset in syms:
            lq, eq = local_pos.get(asset, 0.0), exchange_pos.get(asset, 0.0)
            if lq > 0 and eq <= 0:
                await self._emit_reconcile("MISSING_POSITION", f"{asset}/{quote}", lq, eq)
            elif eq > 0 and lq <= 0:
                await self._emit_reconcile("UNKNOWN_POSITION", f"{asset}/{quote}", lq, eq)
            elif lq > 0 and eq > 0:
                diff = abs(eq - lq) / max(lq, 1e-12)
                if diff > 0.05:
                    await self._emit_reconcile("QTY_MISMATCH", f"{asset}/{quote}", lq, eq, round(diff, 4))

    async def _emit_reconcile(self, rtype: str, symbol: str, lq: float, eq: float, diff: float = 0.0) -> None:
        """发布对账差异 SYSTEM 事件（架构 §2.2 契约）。"""
        log.warning("[reconcile] %s %s: local=%s exchange=%s diff_pct=%s",
                    rtype, symbol, lq, eq, diff)
        from datetime import datetime, timezone
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "reconcile", "type": rtype, "symbol": symbol,
            "local_qty": round(lq, 8), "exchange_qty": round(eq, 8),
            "diff_pct": diff, "ts": datetime.now(timezone.utc).isoformat(),
        }, source="engine"))

    # ---------------- T5 断链检测 ----------------
    def _stale_threshold(self) -> int:
        """断流阈值秒数：未配置时按当前周期取 2 根 K 线；<=0 表示显式关闭。"""
        cfg = settings.max_stale_seconds
        if cfg is None:
            from backtest.data_loader import timeframe_seconds
            return timeframe_seconds(self.timeframe) * 2
        return int(cfg)

    async def _check_market_stale(self) -> None:
        """断链检测：K 线超过阈值未更新 → live 平仓 / 其他告警。

        恢复后发布 recovered 事件，且不自动重新开仓（需人工）。
        """
        threshold = self._stale_threshold()
        if not self.hub or threshold <= 0:
            return
        age = self.hub.last_kline_age(self.symbol, self.timeframe)
        stale = age > threshold
        # 无任何K线数据时 age=inf：视为断链（历史数据尚未到达）
        if age == float("inf"):
            stale = True
            age_repr = -1
        else:
            age_repr = int(age)
        live = getattr(self, "trading_mode", "paper") == "live"
        if stale and not self._stale_alerted:
            self._stale_alerted = True
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "market_stale", "symbol": self.symbol,
                "timeframe": self.timeframe,
                "stale_sec": age_repr,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, source="engine"))
            log.warning("[ws] 行情断流 %ss，超过阈值 %ss%s",
                        age_repr, threshold,
                        "（live 模式：触发自动平仓）" if live else "（非 live：仅告警）")
            closed_qty, close_err = (0.0, "")
            if live:
                closed_qty, close_err = await self._force_close_stale_position()
            try:
                from core.notify import notify
                if closed_qty > 0:
                    tail = f"，已市价平仓 {closed_qty}"
                elif live:
                    tail = f"，自动平仓失败（持仓裸奔）：{close_err}" if close_err else "，无持仓可平"
                else:
                    tail = ""
                await notify(self.db, "行情断流告警",
                             f"{self.symbol} {self.timeframe} K线已 {age_repr}s 未更新"
                             f"（阈值 {threshold}s）" + tail)
            except Exception:  # noqa: BLE001
                pass
        elif not stale and self._stale_alerted:
            self._stale_alerted = False
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "market_recovered", "symbol": self.symbol,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, source="engine"))
            log.info("[ws] 行情恢复（后续开仓需人工确认）")
            try:
                from core.notify import notify
                await notify(self.db, "行情恢复", f"{self.symbol} 行情已恢复，后续开仓需人工确认")
            except Exception:  # noqa: BLE001
                pass

    async def _force_close_stale_position(self) -> tuple[float, str]:
        """断连保护性平仓：真下市价单，返回 (平仓量, 错误详情)。

        曾只 publish EventType.SIGNAL —— 全仓库无 SIGNAL 订阅者，实际不会下单，
        而告警文案已写"已触发自动平仓"，用户据此以为裸仓已受保护。
        """
        position = self._current_position_qty()
        if position <= 0:
            return 0.0, ""
        price = float((self.hub.last_price(self.symbol) if self.hub else None) or 0.0)
        signal = Signal(self.symbol, "sell", qty=position, order_type="market",
                        reason="交易所断连自动平仓", strategy=self.strategy.name)
        try:
            async with self._strategy_lock:
                await self._execute_signal(signal, price, 1.0, bypass_risk=True)
        except Exception as e:  # noqa: BLE001
            log.exception("[engine] 断连自动平仓失败: %s", e)
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "stale_close_failed", "symbol": self.symbol, "qty": position,
                "detail": str(e), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, source="engine"))
            return 0.0, str(e)
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "stale_close", "symbol": self.symbol, "qty": position,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, source="engine"))
        return position, ""

    def _current_position_qty(self) -> float:
        """返回当前持仓数量（供断链平仓使用）。"""
        if self.paper and self.paper_account:
            return self.paper_account.positions.get(self.symbol, {}).get("qty", 0.0)
        if self.exchange and self.exchange.has_private:
            bal = self._live_balance if hasattr(self, "_live_balance") else {}
            base = base_currency(self.symbol)
            return float(bal.get(base, {}).get("total", 0.0)) if bal else 0.0
        return 0.0

    async def _ai_scheduler(self) -> None:
        """AI 定时任务：市场解读 / 全自动优化(价格行为) / 复盘。

        市场解读由自动分析开关控制（ai_auto_analysis_enabled，可在 Web 界面切换）。
        循环体整体 try/except：任一非 AI 异常（DB locked、类型错误等）不得杀死任务。
        """
        while self.running:
            try:
                now = time.time()
                snap = await self._current_snapshot()
                if snap:
                    # 自动行情解读：受开关控制
                    auto_on = (await self.db.kv_get("ai_auto_analysis_enabled", "1")) != "0"
                    if auto_on and now - self._last_analysis >= settings.ai_market_analysis_interval:
                        self._last_analysis = now
                        # 注入真实持仓（曾恒传 0/空，提示词持仓死数据）
                        pos, _recent = await self._analysis_context(snap)
                        # safe_analyze 签名 (snap, position)——recent_trades 仅 analyze
                        # 支持、safe_analyze 不转发，曾传 recent_trades=… 必抛 TypeError，
                        # 自动分析每次触发即失败被循环体吞掉（"看似运行实则已死"）
                        await self.analyst.safe_analyze(snap, position=pos)
                    if now - self._last_optimize >= settings.ai_optimize_interval:
                        self._last_optimize = now
                        perf = await self._recent_performance()
                        # 锁内仅快照策略名与参数（短持锁；AI 网络调用最坏数分钟，
                        # 不得阻塞 K 线处理——_on_candle_locked 需要同一把锁）
                        async with self._strategy_lock:
                            target = self.strategy
                            name, params_before = target.name, dict(target.params)
                        # 锁外执行 AI 网络调用（apply=False：内部不写策略参数）
                        result = await self.optimizer.optimize_price_action(
                            target, perf, snap, "价格行为与关键位", apply=False)
                        if result and result.get("params"):
                            # AI 参数热更新验证门：先回测 + 过拟合校验，通过才应用。
                            # 否则 AI 纯凭一张快照改实盘参数，无样本外验证（曾直接 apply）。
                            if settings.ai_optimize_validate:
                                ok, vinfo = await self._validate_param_update(
                                    name, result["params"], params_before)
                                if ok:
                                    # 锁内应用（短持锁；按 name 重新获取，容忍锁外策略被切换）
                                    await self.apply_strategy_params(name, result["params"])
                                log.info("[ai] 参数优化验证 %s: %s",
                                         "通过并应用" if ok else "未通过，保留当前参数",
                                         vinfo.get("reason", ""))
                            else:
                                await self.apply_strategy_params(name, result["params"])
                    if now - self._last_review >= settings.ai_review_interval:
                        self._last_review = now
                        trades = await self._recent_trades(limit=100)
                        if trades:
                            # 用真实交易累计构建权益曲线（曾传假曲线 [x]*2，
                            # 导致 total_return/sharpe/max_drawdown 恒 0，复盘数据失真）
                            # P2-11：_recent_trades 按 ts 倒序查询，必须先反转再累计，
                            # 否则曲线按时间倒序构造，total_return/sharpe/回撤全部失真
                            equity = [self._paper_equity]
                            for t in reversed(trades):
                                equity.append(equity[-1] + (t.get("pnl") or 0.0))
                            summary = compute_metrics(equity, trades, self.timeframe, self._paper_equity)
                            # validator 反谄媚校验依赖 total_pnl（compute_metrics 无此键，补上）
                            summary["total_pnl"] = round(float(summary.get("final_equity", 0.0)) - self._paper_equity, 4)
                            await self.reviewer.review(trades, summary)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("[ai_scheduler] AI 定时任务异常，30s 后继续: %s", e)
            await asyncio.sleep(30)

    # ---------------- AI 策略工坊 ----------------
    async def _current_snapshot(self) -> Optional[dict]:
        """取当前行情快照（含 S/R 与价格行为）。

        数据源优先级：
        1. WebSocket hub 快照（引擎运行时）
        2. Binance 官方 .vision 数据域（国内可直连，无需代理）
        3. ccxt REST（带代理）
        """
        if self.hub:
            snap = self.hub.snapshot(self.symbol, self.timeframe)
            if snap:
                return snap
        # 优先用国内可直连的 Binance .vision 数据域（已验证无需代理）
        ohlcv = await self._fetch_vision_ohlcv(limit=150)
        if ohlcv is None:
            # 兜底：ccxt REST（自动走本地代理）
            try:
                from exchange.manager import ExchangeManager
                mgr = ExchangeManager(self.exchange_id)
                await mgr.start()
                try:
                    ohlcv = await mgr.fetch_ohlcv(self.symbol, self.timeframe, limit=120)
                finally:
                    await mgr.close()
            except Exception as e:
                log.warning("[engine] ccxt 快照兜底失败: %s", e)
                return None
        if not ohlcv:
            return None
        from indicators.technical import compute_latest
        ohlcv = [list(c) for c in ohlcv]
        ind = compute_latest(ohlcv)
        return {"symbol": self.symbol, "timeframe": self.timeframe, "candles": ohlcv,
                "closes": [c[4] for c in ohlcv], "indicators": ind}

    async def _fetch_vision_ohlcv(self, limit: int = 150, symbol: Optional[str] = None,
                                  timeframe: Optional[str] = None) -> Optional[list]:
        """从 Binance 官方 .vision 数据域拉取K线（国内可直连，无需代理）。

        symbol/timeframe 可选：默认用引擎当前配置；传参时用指定品种/周期。
        """
        import time as _time
        import httpx
        from config.settings import settings
        sym = (symbol or self.symbol).replace("/", "").upper()
        tf = timeframe or self.timeframe
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": sym, "interval": tf, "limit": limit}
        try:
            # .vision 大陆可直连：不挂代理（本地代理故障时不被拖死）
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                rows = r.json()
                out = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in rows]
                log.info("[engine] 从 .vision 拉取 %s %s 共 %d 根K线", sym, tf, len(out))
                return out
        except Exception as e:
            log.warning("[engine] .vision 拉取失败: %s", e)
            return None

    # 注册前守卫（信号密度门 + 过拟合前推）所需的历史K线根数。
    # 密度门只看最近 1000 根，但过拟合 CSCV/PBO 分折要更多样本才不至于"无法判定"。
    GUARD_CANDLES = 3000

    async def _fetch_history_ohlcv(self, target: int, symbol: Optional[str] = None,
                                   timeframe: Optional[str] = None) -> list:
        """取最近 target 根真实K线：本地库优先，缺口分页回填。

        不能直接把 _fetch_vision_ohlcv 的 limit 调到 3000：/api/v3/klines 单次上限
        1000 根，更大的 limit 会被静默截断 —— 守卫因此长期"数据不足→无法判定→等于放行"。
        """
        from backtest import kline_store
        sym = symbol or self.symbol
        tf = timeframe or self.timeframe
        rows = await asyncio.to_thread(kline_store.load_rows, "binance", sym, tf)
        if len(rows) < target:
            fetched = await self._fetch_klines_pages(target, sym, tf)
            if fetched:
                await asyncio.to_thread(kline_store.save_rows, "binance", sym, tf, fetched)
                rows = await asyncio.to_thread(kline_store.load_rows, "binance", sym, tf)
        if len(rows) < target:
            log.warning("[engine] %s %s 历史K线仅 %d 根（需 %d 根），守卫按可得数据判定",
                        sym, tf, len(rows), target)
        return [list(r) for r in rows[-target:]]

    async def _fetch_klines_pages(self, target: int, sym: str, tf: str) -> list:
        """REST 向历史回退分页（endTime 逐步前移），每页 ≤1000 根。"""
        import time as _time
        import httpx
        url = "https://data-api.binance.vision/api/v3/klines"
        api_sym = sym.replace("/", "").upper()
        out: list = []
        end = int(_time.time() * 1000)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                for _ in range((target + 999) // 1000):
                    r = await client.get(url, params={"symbol": api_sym, "interval": tf,
                                                      "limit": 1000, "endTime": end})
                    r.raise_for_status()
                    page = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]),
                             float(c[4]), float(c[5])] for c in r.json()]
                    if not page or page[0][0] >= end:  # 无数据 / 无进展 → 停，防死循环
                        break
                    out = page + out
                    end = page[0][0] - 1
                    if len(page) < 1000:
                        break  # 已触到该品种历史起点
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 分页拉取K线失败: %s", e)
            return []
        log.info("[engine] 分页拉取 %s %s 共 %d 根K线", sym, tf, len(out))
        return out

    async def _recent_backtests(self, limit: int = 3) -> list[dict]:
        """读取最近的回测结果摘要，供 AI 策略设计/优化/迭代参考。"""
        try:
            from core.database import BacktestResult
            from sqlalchemy import select
            async with self.db.session() as s:
                rows = (await s.execute(
                    select(BacktestResult).order_by(BacktestResult.id.desc()).limit(limit)
                )).scalars().all()
                out = []
                for r in rows:
                    try:
                        m = json.loads(r.metrics_json or "{}")
                        cfg = json.loads(r.config_json or "{}")
                        if m.get("status") == "running" or m.get("error"):
                            continue
                        out.append({
                            "id": r.id,
                            "strategy": cfg.get("strategy_name", ""),
                            "symbol": cfg.get("symbol", ""),
                            "timeframe": cfg.get("timeframe", ""),
                            "total_return": m.get("total_return"),
                            "annual_return": m.get("annual_return"),
                            "max_drawdown": m.get("max_drawdown"),
                            "sharpe": m.get("sharpe"),
                            "win_rate": m.get("win_rate"),
                            "profit_factor": m.get("profit_factor"),
                            "total_trades": m.get("total_trades"),
                            "elapsed_sec": m.get("elapsed_sec"),
                        })
                    except Exception:  # noqa: BLE001
                        continue
                return out
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 读取历史回测失败: %s", e)
            return []

    async def _previous_iterations(self, based_on: str, limit: int = 3) -> list[dict]:
        """读取基于同一基础策略的历史 AI 迭代记录（critique/改进点/版本）。

        用于迭代正循环：AI 迭代时能看到上一代说了什么、改了什么、效果如何，
        从而避免重复同样错误、真正做到"每一代正向提升"。
        """
        try:
            from core.database import AiStrategy
            from sqlalchemy import select
            async with self.db.session() as s:
                rows = (await s.execute(
                    select(AiStrategy).order_by(AiStrategy.id.desc()).limit(200)
                )).scalars().all()
            out = []
            # 基础名归一化：对 dual_ma_v2 再迭代时 based_on 是 "dual_ma_v2"，
            # 但存储的记录 based_on="dual_ma"——曾精确匹配拉不到任何前代历史
            base = re.sub(r"_v\d+$", "", based_on)
            for r in rows:
                try:
                    spec = json.loads(r.spec_json or "{}")
                    if spec.get("created_by") != "ai_iteration":
                        continue
                    if spec.get("based_on") not in (based_on, base):
                        continue
                    out.append({
                        "name": spec.get("name", ""),
                        "version": spec.get("version", ""),
                        "critique": spec.get("critique", ""),
                        "improvements": spec.get("improvements", []),
                        "summary": spec.get("summary", ""),
                        "params": spec.get("params", {}),
                        # 注册时的回测验证指标（迭代验证门持久化，AI 可见"上一代真实表现"）
                        "backtest": spec.get("backtest"),
                    })
                    if len(out) >= limit:
                        break
                except Exception:  # noqa: BLE001
                    continue
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 读取历史迭代失败: %s", e)
            return []

    async def design_strategy(self, symbol: str = "", timeframe: str = "",
                              strategy_type: str = "", custom_requirement: str = "") -> dict:
        """AI 设计策略：可指定品种/周期/策略类型/自定义要求。

        symbol/timeframe 传参时按指定品种拉取行情快照（否则用引擎当前配置）。
        strategy_type: trend/breakout/mean_reversion/grid/custom
        custom_requirement: 用户自由打字的设计要求
        """
        use_symbol = symbol or self.symbol
        use_tf = timeframe or self.timeframe
        snap = await self._current_snapshot()
        if not snap:
            raise RuntimeError("无法获取行情快照，请先启动引擎或稍后重试")
        # 指定了品种或周期时，拉取对应品种/周期的K线构造快照
        if (symbol or timeframe) and (symbol != self.symbol or timeframe != self.timeframe):
            ohlcv = await self._fetch_vision_ohlcv(limit=150, symbol=use_symbol, timeframe=use_tf)
            if ohlcv:
                from indicators.technical import compute_latest
                ohlcv = [list(c) for c in ohlcv]
                snap = {"symbol": use_symbol, "timeframe": use_tf, "candles": ohlcv,
                        "closes": [c[4] for c in ohlcv], "indicators": compute_latest(ohlcv)}
            else:
                log.warning("[engine] 指定品种 %s/%s 拉取失败，回退引擎当前快照", use_symbol, use_tf)
        trades = await self._recent_trades(50)
        backtests = await self._recent_backtests(3)
        guard_candles = await self._fetch_history_ohlcv(self.GUARD_CANDLES, symbol=use_symbol,
                                                        timeframe=use_tf) \
            or snap.get("candles", [])
        return await self.designer.design(snap, trades, backtests, guard_candles=guard_candles,
                                          strategy_type=strategy_type,
                                          custom_requirement=custom_requirement)

    async def _recent_trades(self, limit: int = 100) -> list[dict]:
        from core.database import Trade
        async with self.db.session() as s:
            rows = (await s.execute(select(Trade).order_by(Trade.ts.desc()).limit(limit))).scalars().all()
            return [{"ts": t.ts.isoformat(), "symbol": t.symbol, "side": t.side, "price": t.price,
                     "qty": t.qty, "fee": t.fee, "pnl": t.pnl, "strategy": t.strategy, "reason": t.reason} for t in rows]

    async def _recent_performance(self) -> dict:
        trades = await self._recent_trades(50)
        pnls = [t["pnl"] for t in trades if t["side"] == "sell" and t["pnl"]]
        wins = [p for p in pnls if p > 0]
        return {"total_trades": len(pnls), "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
                "total_pnl": round(sum(pnls), 4), "recent_pnl": round(sum(pnls[-10:]), 4)}

    # ---------------- AI 参数热更新验证门 ----------------
    async def _validation_df(self, bars: int):
        """取用于验证的历史K线 DataFrame（优先 .vision 数据域；失败回退 hub 内存缓存）。

        bars 保持在 1000 以内：本路径是单次 REST 拉取，更大的 limit 会被静默截断。
        """
        from backtest.data_loader import _to_df
        ohlcv = await self._fetch_vision_ohlcv(limit=bars)
        if (not ohlcv or len(ohlcv) < 200) and self.hub:
            snap = self.hub.snapshot(self.symbol, self.timeframe)
            ohlcv = (snap or {}).get("candles") or ohlcv
        if not ohlcv or len(ohlcv) < 200:
            return None
        return _to_df([list(c) for c in ohlcv])

    async def _validate_param_update(self, name: str, proposed: dict, current: dict) -> tuple[bool, dict]:
        """AI 参数热更新验证门：在最近窗口上跑当前/建议参数的回测 + 过拟合检测，
        通过才允许应用（否则保留当前参数，保守优先）。

        返回 (ok, info)。数据不足或无法验证时一律不应用（ok=False）。
        回测/过拟合为 CPU 密集，放 asyncio.to_thread 避免阻塞事件循环。
        """
        df = await self._validation_df(settings.ai_optimize_validation_bars)
        if df is None or len(df) < 200:
            return False, {"reason": f"验证数据不足（<200 根），跳过应用", "applied": False}

        def _compute() -> tuple[bool, dict]:
            from backtest.engine import BacktestConfig
            from backtest.fast_engine import run_backtest_fast

            def _run(params: dict, data=None) -> dict | None:
                cfg = BacktestConfig(symbol=self.symbol, timeframe=self.timeframe,
                                     strategy_name=name, strategy_params=dict(params),
                                     start_cash=self.start_cash, fee_rate=self.paper_fee_rate,
                                     slippage=self.paper_slippage)
                try:
                    return run_backtest_fast(data if data is not None else df, cfg,
                                             bootstrap=False)["metrics"]  # P2-13 验证门只需基础指标
                except Exception as e:  # noqa: BLE001
                    log.warning("[ai] 参数验证回测失败（%s）: %s", name, e)
                    return None

            # P1-8 纯 OOS 预留：整段末尾 ~10% 作为从不参与拟合/过拟合检测的
            # 纯样本外段，建议参数必须在纯 OOS 上也不劣于当前参数才应用。
            # OOS 段过短（<60 根）时降级为普通回测对比，不硬性拦截。
            oos_len = max(0, len(df) // 10)
            oos_df = df.iloc[-oos_len:] if oos_len >= 60 else None
            fit_df = df.iloc[:-oos_len] if oos_len >= 60 else df

            cur_m = _run(current)
            prop_m = _run(proposed)
            if not cur_m or not prop_m:
                return False, {"reason": "回测失败，跳过应用", "current": cur_m, "proposed": prop_m}
            # 建议参数至少要有真实交易，否则应用后策略停摆
            if int(prop_m.get("total_trades", 0)) < 1:
                return False, {"reason": f"建议参数回测无交易（{prop_m.get('total_trades', 0)} 笔），跳过应用",
                               "current": cur_m, "proposed": prop_m}

            # 过拟合检测（数据够长才做；失败不阻断，仅降级为回测对比）。
            # 只在 fit 段上做：walk-forward 末折 OOS 与纯 OOS 段重叠时，
            # 建议参数会"提前见过"尾部行情，纯 OOS 校验就失去意义。
            overfit = None
            if len(fit_df) >= 400:
                try:
                    from backtest.overfit import OverfitConfig, detect_overfit
                    rep = detect_overfit(fit_df, name, proposed, OverfitConfig())
                    overfit = {"verdict": rep.verdict, "score": rep.score,
                               "oos_ret": rep.oos_ret, "pbo": rep.pbo,
                               "decay": rep.decay, "n_folds": rep.n_folds}
                    if rep.verdict in ("严重过拟合", "无法判定"):
                        # 无法判定＝证据不足（样本外交易太少）：可以留在策略库里人工用，
                        # 但不允许 AI 无人值守地接管实盘
                        return False, {"reason": f"过拟合检测未通过（{rep.verdict}，score={rep.score}）",
                                       "overfit": overfit, "current": cur_m, "proposed": prop_m}
                except Exception as e:  # noqa: BLE001
                    log.warning("[ai] 参数验证过拟合检测失败（降级为回测对比）: %s", e)

            prop_ret = float(prop_m.get("total_return", -1.0) or -1.0)
            cur_ret = float(cur_m.get("total_return", -1.0) or -1.0)
            prop_dd = float(prop_m.get("max_drawdown", 1.0) or 1.0)
            cur_dd = float(cur_m.get("max_drawdown", 0.0) or 0.0)
            # 门槛：收益不比当前差太多（容忍 0.5%），且回撤不明显恶化
            not_worse = prop_ret >= cur_ret - 0.005
            dd_ok = prop_dd <= max(0.5, cur_dd + 0.05)
            if not (not_worse and dd_ok):
                return False, {"reason": f"建议参数未通过收益/回撤门槛（ret {cur_ret:.4f}→{prop_ret:.4f}，dd {cur_dd:.4f}→{prop_dd:.4f}）",
                               "overfit": overfit, "current": cur_m, "proposed": prop_m}

            # ---- P1-8 纯 OOS 硬校验：建议参数在预留尾部 OOS 上不得劣于当前 ----
            oos_info: dict = {"enabled": False, "reason": "OOS 段过短，降级为普通对比"}
            if oos_df is not None:
                cur_oos = _run(current, oos_df)
                prop_oos = _run(proposed, oos_df)
                if not cur_oos or not prop_oos:
                    oos_info = {"enabled": True,
                                "reason": "纯 OOS 回测失败，跳过（保守）"}
                elif int(prop_oos.get("total_trades", 0)) < 1 or int(cur_oos.get("total_trades", 0)) < 1:
                    # OOS 段交易过少 → 收益噪音大，不硬拦但如实标注
                    oos_info = {"enabled": True,
                                "reason": f"纯 OOS 交易过少（cur {cur_oos.get('total_trades', 0)} / prop {prop_oos.get('total_trades', 0)}），不构成可靠证据",
                                "oos_bars": oos_len,
                                "cur_oos_ret": cur_oos.get("total_return", 0.0),
                                "prop_oos_ret": prop_oos.get("total_return", 0.0)}
                else:
                    cur_oos_ret = float(cur_oos.get("total_return", -1.0) or -1.0)
                    prop_oos_ret = float(prop_oos.get("total_return", -1.0) or -1.0)
                    oos_info = {"enabled": True, "oos_bars": oos_len,
                                "cur_oos_ret": cur_oos_ret, "prop_oos_ret": prop_oos_ret}
                    # 纯 OOS 门槛（容忍 0.5%）
                    if prop_oos_ret < cur_oos_ret - 0.005:
                        return False, {"reason": f"建议参数在纯 OOS 段劣于当前（ret {cur_oos_ret:.4f}→{prop_oos_ret:.4f}），跳过应用",
                                       "overfit": overfit, "oos": oos_info,
                                       "current": cur_m, "proposed": prop_m}

            return True, {"reason": f"回测通过（ret {cur_ret:.4f}→{prop_ret:.4f}，dd {cur_dd:.4f}→{prop_dd:.4f}）",
                          "overfit": overfit, "oos": oos_info,
                          "current": cur_m, "proposed": prop_m}

        return await asyncio.to_thread(_compute)

    async def validate_and_apply_ai_params(self, name: str, proposed: dict,
                                           current: Optional[dict] = None) -> tuple[bool, dict]:
        """验证 AI 候选参数，且仅在验证通过后执行一次热更新。"""
        if not isinstance(proposed, dict) or not proposed:
            return False, {"reason": "AI 未返回可应用的参数", "applied": False}
        async with self._strategy_lock:
            if self.strategy.name != name:
                return False, {"reason": f"策略已切换为 {self.strategy.name}，跳过应用",
                                "applied": False}
            snapshot = dict(current if current is not None else self.strategy.params)
        if not bool(getattr(settings, "ai_optimize_validate", True)):
            applied = await self.apply_strategy_params(name, proposed)
            return True, {"reason": "已关闭 AI 参数性能验证门", "applied": True,
                          "validation_bypassed": True, "applied_params": applied}
        ok, info = await self._validate_param_update(name, proposed, snapshot)
        info = dict(info or {})
        if not ok:
            info["applied"] = False
            return False, info
        async with self._strategy_lock:
            if self.strategy.name != name or dict(self.strategy.params) != snapshot:
                return False, {**info, "reason": "验证期间策略参数已变化，跳过应用",
                                "applied": False}
        applied = await self.apply_strategy_params(name, proposed)
        info["applied"] = True
        info["applied_params"] = applied
        return True, info

    # ---------------- 策略管理 ----------------
    def _strategy_ma_periods(self) -> tuple[int, int]:
        """当前策略的 MA 快/慢线周期（与回测引擎同源 strategies.base.strategy_ma_periods）。"""
        from strategies.base import strategy_ma_periods
        return strategy_ma_periods(self.strategy)

    def _sync_hub_ma_periods(self) -> None:
        """策略切换/参数热更新后，把新 MA 周期同步到行情中枢（实时指标跟随策略参数）。"""
        if self.hub is None:
            return
        fast_p, slow_p = self._strategy_ma_periods()
        try:
            self.hub.set_ma_periods(fast_p, slow_p)
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 同步行情中枢 MA 周期失败（继续使用旧周期）: %s", e)

    async def select_strategy(self, name: str) -> dict:
        async with self._strategy_lock:
            self.strategy = get_strategy(name)
            # 清空待成交信号：旧策略的信号不得在新策略下成交（否则 on_fill 污染新策略状态）
            self._pending_signal = None
            self._ma_fast_period, self._ma_slow_period = self._strategy_ma_periods()
            log.info("切换策略: %s", name)
            # 锁内同步（hub.set_ma_periods 非 async，无提交点）
            self._sync_hub_ma_periods()
            return {"name": name, "params": self.strategy.params}

    async def update_strategy_params(self, params: dict) -> dict:
        async with self._strategy_lock:
            return {"params": self.strategy.update_params(params)}

    async def apply_strategy_params(self, name: str, params: dict) -> dict:
        """带策略锁应用参数——后台 AI worker 线程经 run_coroutine_threadsafe 调回主循环。

        曾 worker 直接 strategy.update_params 绕过锁，与 _on_candle_locked 持锁期间
        并发写 params（多键非原子），信号读到混合参数。
        曾 get_strategy(name) 每次返回新实例，参数更新在无人引用的实例上——
        "热更新"从不生效；改为 name 与当前策略一致时直接作用于 self.strategy
        （容忍锁外策略被切换：不一致时仅记录，参数不生效但错误可见）。
        """
        async with self._strategy_lock:
            if self.strategy.name == name:
                applied = self.strategy.update_params(params)
                # 参数热更新后同步 MA 周期到行情中枢（dual_ma 周期实际生效）
                self._ma_fast_period, self._ma_slow_period = self._strategy_ma_periods()
                self._sync_hub_ma_periods()
                log.info("[engine] 参数热更新（主循环持锁）: %s", applied)
                return applied
            st = get_strategy(name)
            applied = st.update_params(params)
            log.warning("[engine] 参数热更新跳过（策略已切换为 %s，目标 %s）: %s",
                        self.strategy.name, name, applied)
            return applied

    def list_strategies(self) -> list[dict]:
        return list_strategies()
