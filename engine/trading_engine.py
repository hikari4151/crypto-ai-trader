"""总调度引擎：行情→策略→风控→下单，调度 AI 任务 + AI 策略工坊。"""
import asyncio
import json
import logging
import math
import time
from typing import Any, Optional

from sqlalchemy import select

from ai import AIClient, MarketAnalyst, ParamOptimizer, StrategyIteration, TradeReviewer
from ai.strategy_designer import StrategyDesigner
from backtest.metrics import compute_metrics
from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from exchange.manager import ExchangeManager
from exchange.paper import PaperAccount
from exchange.ws_market import MarketDataHub
from strategies import get_strategy, list_strategies, register_dynamic
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

        self.strategy = get_strategy(settings.default_strategy)
        self._trade_times: list[float] = []
        self._daily_pnl = 0.0
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

        bus.subscribe(EventType.MARKET_CANDLE, self._on_candle)
        # 高频价格采样：供风控闪崩/暴涨检测使用（hub 约每秒发布一次 ticker）
        bus.subscribe(EventType.MARKET_TICKER, self._on_ticker)

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
            # 模拟实盘：用真实行情 + 模拟账户撮合（paper=True，不真实下单）
            # 实盘：真实下单（paper=False）
            self.paper = (mode in ("paper", "simulated"))
            # 初始资金按模式取来源：
            # - live/simulated（模拟实盘=用实盘参数做模拟验证）：用实盘配置的资金
            # - paper：用模拟配置的资金（曾 simulated 只读模拟配置，
            #   用户在"实盘配置"窗口设置的资金不生效）
            # 防御：DB 中可能残留历史 NaN 值（校验被绕过写入），非有限值回退默认
            if mode in ("live", "simulated") and cfg.get("start_cash"):
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
        except Exception as e:  # noqa: BLE001
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
        # 纸面 / 模拟实盘：用模拟账户撮合（不下真实单）
        if mode in ("paper", "simulated"):
            self.paper_account = PaperAccount(start_cash=self.start_cash, fee_rate=self.paper_fee_rate)
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
            self.portfolio = PortfolioManager(self.db, self.bus, False, None, self.exchange)
        elif mode == "simulated" and api_key and secret:
            self.exchange = ExchangeManager(self.exchange_id)
            await self.exchange.start(api_key, secret, password)
            log.info("[engine] 模拟实盘：已连接 %s（仅展示余额，订单走模拟撮合）", self.exchange_id)

        # 恢复今日累计盈亏（风控每日最大亏损跨重启保留）
        try:
            self._daily_pnl = await self.risk.get_daily_pnl()
        except Exception as e:  # noqa: BLE001
            log.warning("[engine] 恢复每日盈亏失败: %s", e)

        self.hub = MarketDataHub(self.exchange_id, [self.symbol], [self.timeframe], self.bus)
        self.running = True
        self._tasks = [
            asyncio.create_task(self.hub.run(), name="hub"),
            asyncio.create_task(self._portfolio_loop(), name="portfolio"),
            asyncio.create_task(self._ai_scheduler(), name="ai_scheduler"),
            asyncio.create_task(self.bus.run(), name="event_bus"),  # 启动事件分发循环
        ]
        await self.bus.publish(Event(EventType.ENGINE_STATE, {"state": "running"}, source="engine"))
        log.info("交易引擎启动: paper=%s exchange=%s symbol=%s strategy=%s",
                 self.paper, self.exchange_id, self.symbol, self.strategy.name)

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if not self.running:
                return
            self.running = False
            tasks = self._tasks
            self._tasks = []
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
            await self.bus.publish(Event(EventType.ENGINE_STATE, {"state": "stopped"}, source="engine"))
            log.info("交易引擎已停止")

    def status(self) -> dict:
        return {"running": self.running, "paper": self.paper,
                "mode": getattr(self, "trading_mode", "paper" if self.paper else "live"),
                "exchange": self.exchange_id,
                "symbol": self.symbol, "timeframe": self.timeframe, "strategy": self.strategy.name}

    async def _load_dynamic_strategies(self) -> None:
        """启动时从数据库恢复 AI 设计的策略。"""
        from core.database import AiStrategy
        try:
            async with self.db.session() as s:
                rows = (await s.execute(select(AiStrategy))).scalars().all()
            count = 0
            for r in rows:
                try:
                    spec = json.loads(r.spec_json or "{}")
                    register_dynamic(r.name, spec)
                    count += 1
                except Exception:
                    continue
            log.info("已加载 %d 个 AI 设计策略", count)
        except Exception as e:
            log.warning("加载 AI 设计策略失败: %s", e)

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
            base, quote = symbol.split("/")[0], symbol.split("/")[1]
            position = float(bal.get(base, {}).get("total", 0.0) or 0.0)
            cash_now = float(bal.get(quote, {}).get("free", 0.0) or 0.0)
            pos_value = position * price
        # 策略入口价属性名不统一：price_action/rl_adaptive 用 _entry，dual_ma 用 _entry_price，两者都兼容
        entry_price = getattr(self.strategy, "_entry", None)
        if entry_price is None:
            entry_price = getattr(self.strategy, "_entry_price", None)

        signal = self.strategy.on_candle({
            "symbol": symbol, "price": price, "position": position,
            "cash": cash_now,
            "indicators": indicators, "timeframe": self.timeframe,
        })
        if not signal:
            return

        allowed, reason = await self.risk.check(signal, price, cash_now, pos_value,
                                                self._daily_pnl, self._trade_times, entry_price,
                                                vol_ratio=vol_ratio)
        if not allowed:
            await self.bus.publish(Event(EventType.RISK_BLOCKED, {"signal": signal.__dict__, "reason": reason}, source="risk"))
            log.warning("[risk] 拦截信号 %s: %s", signal.side, reason)
            return

        await self.bus.publish(Event(EventType.SIGNAL, signal.__dict__, source="engine"))
        fill = await self.order_manager.place(signal, price, self.strategy.name)
        if fill:
            self._trade_times.append(time.time())
            self.strategy.on_fill(symbol, signal.side, fill["price"])
            if signal.side == "sell":
                # 纸面模式用账户加权平均成本（分批建仓/部分减仓 pnl 准确）；
                # 曾用策略单点 entry_price（最后一次买入价被覆盖后 pnl 虚高，
                # 污染每日盈亏/连续亏损冷却/AI 复盘），与回测 FIFO 口径不一致
                cost_price = None
                if self.paper and self.paper_account:
                    cost_price = self.paper_account.positions.get(symbol, {}).get("avg_price")
                if cost_price is None:
                    cost_price = entry_price
                # 无入场价跟踪的策略（如 grid 不维护 _entry）无法算真实盈亏：
                # 置 0 而非记 -fee，否则每日盈亏被手续费污染、连续亏损冷却误触发
                if cost_price is None:
                    pnl = 0.0
                else:
                    pnl = (fill["price"] - cost_price) * fill["qty"] - fill["fee"]
                self._daily_pnl += pnl
                # 记录平仓盈亏给风控（连续亏损冷却判断 + 每日盈亏持久化）
                await self.risk.record_trade(pnl, signal.reason)
                # 平仓盈亏回写 Trade 行：曾只累加内存/风控 KV，性能页的
                # win_rate/total_pnl/avg_win/avg_loss 与 AI 复盘按 pnl=0 聚合全部失真
                if fill.get("trade_id"):
                    try:
                        await self.db.update_trade_pnl(fill["trade_id"], pnl)
                    except Exception as e:  # noqa: BLE001
                        log.warning("[engine] 平仓盈亏落库失败: %s", e)
            await self.bus.publish(Event(EventType.TRADE, {"fill": fill, "signal": signal.__dict__}, source="engine"))

    # ---------------- 定时任务 ----------------
    async def _portfolio_loop(self) -> None:
        while self.running:
            try:
                await self.portfolio.snapshot()
            except Exception as e:  # noqa: BLE001
                log.warning("[portfolio] 快照失败: %s", e)
            await asyncio.sleep(15)

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
                        await self.analyst.safe_analyze(snap)
                    if now - self._last_optimize >= settings.ai_optimize_interval:
                        self._last_optimize = now
                        perf = await self._recent_performance()
                        # 持策略锁执行：optimizer 内部会 update_params 当前策略，
                        # 与 _on_candle 的读参/信号段互斥（同循环 asyncio.Lock 安全）
                        async with self._strategy_lock:
                            await self.optimizer.optimize_price_action(self.strategy, perf, snap, "价格行为与关键位")
                    if now - self._last_review >= settings.ai_review_interval:
                        self._last_review = now
                        trades = await self._recent_trades(limit=100)
                        if trades:
                            # 用真实交易累计构建权益曲线（曾传假曲线 [x]*2，
                            # 导致 total_return/sharpe/max_drawdown 恒 0，复盘数据失真）
                            equity = [self._paper_equity]
                            for t in trades:
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
            async with httpx.AsyncClient(timeout=10, proxy=settings.resolved_proxy or None) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                rows = r.json()
                out = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in rows]
                log.info("[engine] 从 .vision 拉取 %s %s 共 %d 根K线", sym, tf, len(out))
                return out
        except Exception as e:
            log.warning("[engine] .vision 拉取失败: %s", e)
            return None

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
            for r in rows:
                try:
                    spec = json.loads(r.spec_json or "{}")
                    if spec.get("created_by") != "ai_iteration":
                        continue
                    if spec.get("based_on") != based_on:
                        continue
                    out.append({
                        "name": spec.get("name", ""),
                        "version": spec.get("version", ""),
                        "critique": spec.get("critique", ""),
                        "improvements": spec.get("improvements", []),
                        "summary": spec.get("summary", ""),
                        "params": spec.get("params", {}),
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
        guard_candles = await self._fetch_vision_ohlcv(limit=800, symbol=use_symbol, timeframe=use_tf) \
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

    # ---------------- 策略管理 ----------------
    async def select_strategy(self, name: str) -> dict:
        async with self._strategy_lock:
            self.strategy = get_strategy(name)
            log.info("切换策略: %s", name)
            return {"name": name, "params": self.strategy.params}

    async def update_strategy_params(self, params: dict) -> dict:
        async with self._strategy_lock:
            return {"params": self.strategy.update_params(params)}

    async def apply_strategy_params(self, name: str, params: dict) -> dict:
        """带策略锁应用参数——后台 AI worker 线程经 run_coroutine_threadsafe 调回主循环。

        曾 worker 直接 strategy.update_params 绕过锁，与 _on_candle_locked 持锁期间
        并发写 params（多键非原子），信号读到混合参数。
        """
        async with self._strategy_lock:
            st = get_strategy(name)
            applied = st.update_params(params)
            log.info("[engine] 参数热更新（主循环持锁）: %s", applied)
            return applied

    def list_strategies(self) -> list[dict]:
        return list_strategies()
