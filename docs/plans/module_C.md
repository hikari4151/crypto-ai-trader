# Module C: 实盘安全 — 实施计划

> 依赖架构契约：docs/plans/architecture.md §2.2（每日对账 SYSTEM payload）、§2.3（断链自动平仓 settings 键）、§2.9（熔断人工确认）
> 独占文件：`engine/trading_engine.py`、`engine/portfolio.py`、`web/api/portfolio.py`、`engine/risk.py`、`web/api/risk.py`、`exchange/ws_market.py`、`config/settings.py`
> 任务：🔴3 每日对账 + 🟠4 熔断人工确认 + 🟠5 断链自动平仓

---

## T1. 🔴3 portfolio.py 抽取公共钱包拉取方法

**Files**：`engine/portfolio.py`（:27-82 `snapshot`、:84-89 `_safe_ticker`）、`web/api/portfolio.py`（:53-87 `_fetch_live_balance`、:116-149 `_do_fetch`）

**Interfaces**：
- 产出：`PortfolioManager.fetch_live_balance(api_key: str, secret: str, password: str) -> dict`（公共方法，供引擎侧每日对账 + API 侧复用）

**实现**：
1. `engine/portfolio.py` `PortfolioManager` 新增公共方法（复用 `web/api/portfolio.py` `_do_fetch` 的逻辑，参数为密钥）：
   ```python
   async def fetch_live_balance(self, api_key: str, secret: str, password: str) -> dict:
       """拉取交易所真实钱包（供每日对账与 API 复用；close 由调用方负责）。"""
       if self.exchange is None:
           return {"total": {}}
       # 复用即时平衡缓存逻辑：30s TTL，异常保留旧缓存
       now = time.time()
       if now - self._live_balance_ts < 30 and self._live_balance_cache:
           return self._live_balance_cache
       bal = await self.exchange.fetch_balance()
       self._live_balance_cache = bal
       self._live_balance_ts = now
       return bal
   ```
2. `PortfolioManager.__init__` 增加字段：`self._live_balance_cache: dict = {}`、`self._live_balance_ts: float = 0.0`
3. `web/api/portfolio.py` 侧：保持现有 `_do_fetch`/`_fetch_live_balance`（API 轮询路径独立），**不改其行为**。若复用需要，可改为调用 `PortfolioManager.fetch_live_balance`——但**不强制**，两处逻辑同构即可。

> 设计决策：每日对账（引擎内）与 API 轮询（无引擎或引擎未连交易所时）所需上下文不同，引擎侧对账用 `self.exchange`（已连接），API 侧动态创建 manager。两者不强行合并，只保证**余额字段口径一致**（`total`/`used`/`free`）。

**防回归红线**：
- `snapshot()`（:27-82）的 paper 分支与 live 分支行为零改动
- `web/api/portfolio.py` 的 `_do_fetch`/`_fetch_live_balance`/`_socket_probe` 全部不动（API 响应结构 `{equity, cash, positions_value, positions, source/error}` 不变）
- 对账拉取失败（交易所不可达）不抛异常——返回空 dict + warning，对账逻辑见 T2

**验证**：
- `python -m pytest tests/test_portfolio_equity.py -q`（现有组合测试回归）

---

## T2. 🔴3 trading_engine 每日对账分支

**Files**：`engine/trading_engine.py`（`_portfolio_loop` :458-464、`__init__` :31-80、SYSTEM 事件发布）

**Interfaces**：
- 消费：`PortfolioManager.fetch_live_balance()`
- 产出：`SYSTEM` 事件（payload 见 architecture.md §2.2）、`logs` 告警格式 `[reconcile] ...`

**实现**：
1. `__init__`（:31-80）新增状态：`self._daily_reconciled_today: bool = False`、`self._last_reconcile_date: str = ""`
2. `_portfolio_loop`（:458-464）改造（**保持 15s 循环与 snapshot 行为不变，只加每日分支**）：
   ```python
   async def _portfolio_loop(self) -> None:
       while self.running:
           try:
               await self.portfolio.snapshot()
               await self._daily_reconcile_if_due()   # 新增
           except Exception as e:
               log.warning("[portfolio] 快照失败: %s", e)
           await asyncio.sleep(15)
   ```
3. 新增方法：
   ```python
   async def _daily_reconcile_if_due(self) -> None:
       """每日 0 点做一次交易所全量持仓对账（仅 live 模式）。"""
       today = time.strftime("%Y-%m-%d")
       if self._last_reconcile_date != today:
           self._daily_reconciled_today = False   # 跨日重置
           self._last_reconcile_date = today
       if self._daily_reconciled_today:
           return
       if getattr(self, "trading_mode", "paper") != "live":
           return   # 纸面/模拟无真实余额可比
       now = datetime.now()   # 引擎通常跨 0 点运行；为便于验证，可在 0 点后任意时刻补跑一次
       if not (now.hour == 0 and 0 <= now.minute < 5):
           return
       self._daily_reconciled_today = True           # 标记已处理（今天不再重跑）
       await self._reconcile_positions()
   ```
   > 说明：QUANT_ADVICE 的判据是"0 点 0 分且秒<15"，但 15s 轮询周期可能错过该瞬间；放宽为"0 点后 5 分钟内补跑一次"（`_daily_reconciled_today` 防重复），跨 0 点重置。此为刻意容错，PM 已认可。
4. 新增对账方法：
   ```python
   async def _reconcile_positions(self) -> None:
       try:
           api_key = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_api_key") or ""
           secret = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_secret") or ""
           password = await self.db.kv_get_secret(f"exchange_{self.exchange_id}_password") or ""
           if not api_key or not secret:
               log.warning("[reconcile] 未配置交易所密钥，跳过每日对账")
               return
           bal = await self.portfolio.fetch_live_balance(api_key, secret, password)
           exchange_pos = {k.split("/")[0]: float(v.get("total") or 0.0)
                           for k, v in bal.get("total", {}).items()}
       except Exception as e:
           log.warning("[reconcile] 每日对账失败（交易所不可达，改日重试）: %s", e)
           self._daily_reconciled_today = False   # 失败改日重试
           return
       # ---- 本地持仓快照 ----
       local_pos = {}
       if self.exchange and self.exchange.has_private:
           try:
               lb = await self._live_balance_cached()
               for asset, v in lb.get("total", {}).items():
                   if float(v or 0.0) > 0:
                       local_pos[asset] = float(v)
           except Exception:
               pass
       elif self.paper and self.paper_account:
           for sym, p in self.paper_account.positions.items():
               local_pos[sym.split("/")[0]] = p["qty"]
       else:
           log.warning("[reconcile] 本地持仓来源不可用，跳过")
           return
       # ---- 差异判定与告警 + SYSTEM 事件 ----
       syms = set(local_pos) | set(exchange_pos)
       for asset in syms:
           lq, eq = local_pos.get(asset, 0.0), exchange_pos.get(asset, 0.0)
           if lq > 0 and eq <= 0:
               self._emit_reconcile("MISSING_POSITION", f"{asset}/USDT", lq, eq)
           elif eq > 0 and lq <= 0:
               self._emit_reconcile("UNKNOWN_POSITION", f"{asset}/USDT", lq, eq)
           elif lq > 0 and eq > 0:
               diff = abs(eq - lq) / max(lq, 1e-12)
               if diff > 0.05:
                   self._emit_reconcile("QTY_MISMATCH", f"{asset}/USDT", lq, eq, round(diff, 4))
   async def _emit_reconcile(self, rtype: str, symbol: str, lq: float, eq: float, diff: float = 0.0) -> None:
       log.warning("[reconcile] %s %s: local=%s exchange=%s diff_pct=%s",
                   rtype, symbol, lq, eq, diff)
       await self.bus.publish(Event(EventType.SYSTEM, {
           "kind": "reconcile", "type": rtype, "symbol": symbol,
           "local_qty": round(lq, 8), "exchange_qty": round(eq, 8),
           "diff_pct": diff, "ts": datetime.now(datetime.timezone.utc).isoformat(),
       }, source="engine"))
   ```

**防回归红线**：
- `_portfolio_loop` 的 15s 周期、`snapshot()` 调用与落库行为不变
- `_live_balance_cached()`（:286-301）现有实现不动（30s 缓存、失败保留旧缓存）
- 对账只读不写：不改本地持仓、不自动补平（仅告警——QUANT_ADVICE 语义）

**验证**：
- 模拟盘（simulated）启动引擎，临时把 `_daily_reconcile_if_due` 的判据改为"立即执行一次"（开发期验证用），观察日志出现 `[reconcile]`；完成后再还原
- 手插一条测试 Trade 改变本地持仓 → 观察 MISSING_POSITION/UNKNOWN_POSITION/QTY_MISMATCH 告警

---

## T3. 🟠4 risk.py 冷却人工确认恢复 + 端点

**Files**：`engine/risk.py`（`cooldown_status` :188-198、`check` 冷却段 :304-312）、`web/api/risk.py`（:21-42 路由）、`config/settings.py`（`Settings` :61-110）

**Interfaces**（契约见 architecture.md §2.9）：
- 产出：`RiskManager.cooldown_status()` 返回追加 `manual_recovery`、`awaiting_clear` 键
- 产出：`RiskManager.clear_cooldown() -> None`（同步方法）
- 产出：`POST /api/risk/clear-cooldown` 端点
- 产出：`Settings.risk_manual_recovery: bool = False`（KV 键 `risk_manual_recovery`）

**实现**：
1. `risk.py` `RiskManager` 新增字段：`_manual_recovery: bool = False`（在 __post_init__ 或类属性）
2. `cooldown_status`（:188-198）改造：
   ```python
   def cooldown_status(self) -> dict:
       remaining = self._cooldown_until - time.time()
       flash_remaining = self._flash_cooldown_until - time.time()
       manual = self._manual_recovery
       awaiting = False
       if self._cooldown_until > 0 and remaining <= 0 and manual:
           awaiting = True      # 冷却时长已到，但需人工确认
       return {
           "active": (remaining > 0) or awaiting,
           "remaining_sec": max(0, int(remaining)),
           "consecutive_losses": self._consecutive_losses(),
           "flash_cooling": flash_remaining > 0,
           "flash_remaining_sec": max(0, int(flash_remaining)),
           "manual_recovery": manual,
           "awaiting_clear": awaiting,
       }
   ```
3. `_check` 冷却段（:304-312）：冷却后新开仓拦截逻辑不变；在触发冷却/恢复时读取并设置 manual 标志：
   ```python
   # 在 :304-312 冷却判断中，读取规则 manual_recovery 标志
   manual = bool(rules.get("risk_manual_recovery", False))
   self._manual_recovery = manual and self._cooldown_until > time.time()
   ```
   核心改动点：`if self._cooldown_until > time.time() and not closing:` 保持拦截；冷却到期后 `remaining <= 0` 但 `manual` 为真时**仍然拦截**（awaiting_clear）：
   ```python
   if not closing:
       manual = bool(rules.get("risk_manual_recovery", False))
       remaining = self._cooldown_until - time.time()
       if self._cooldown_until > 0 and (remaining > 0 or (remaining <= 0 and manual)):
           reason = f"连续亏损冷却中" + (f"，剩余 {int(remaining)}s" if remaining > 0 else "（人工确认模式）")
           return False, reason
   ```
4. 新增方法：
   ```python
   def clear_cooldown(self) -> None:
       """人工确认解除冷却（冷却到期后 awaiting_clear 状态可调用）。"""
       self._cooldown_until = 0
       self._manual_recovery = False
       log.info("[risk] 冷却已人工解除")
   ```
5. `get_rules`/`update_rules`（:136-186）：`DEFAULT_RULES` 增加 `"risk_manual_recovery": 0`（0/1 布尔，_RULE_BOUNDS 不约束布尔）；`update_rules` 的白名单键集合（:160-165）加入 `"risk_manual_recovery"`：
   ```python
   DEFAULT_RULES = {..., "risk_manual_recovery": 0}
   # update_rules 键列表加入 "risk_manual_recovery"
   ```
6. `web/api/risk.py` 新增端点（:42 后）：
   ```python
   @router.post("/clear-cooldown")
   async def clear_cooldown(engine=Depends(get_engine)):
       """人工确认解除连续亏损冷却（冷却到期且开启人工确认模式时）。"""
       engine.risk.clear_cooldown()
       return {"ok": True, "cooldown": engine.risk.cooldown_status()}
   ```

**防回归红线**：
- `check()` 签名不变；`closing`（平仓豁免）逻辑不变——**止损单即使等待人工确认也绝不拦截**
- `cooldown_status` 现有 5 键（active/remaining_sec/consecutive_losses/flash_cooling/flash_remaining_sec）值语义不变，仅追加 2 键
- 默认 `risk_manual_recovery=0` → 现有自动恢复行为零变化（awaiting_clear 恒 False）
- `flash_cooldown` 相关逻辑零改动

**验证**：
- 测试：设置 `risk_manual_recovery=1`、连续亏损触发冷却 → 冷却到期后 `cooldown_status()["awaiting_clear"]=True` 且 `check()` 仍拦截新开仓 → `clear_cooldown()` 后 `check()` 放行
- 默认配置（0）回归：冷却到期自动恢复（现有 tests/test_risk_guards.py 全绿）

---

## T4. 🟠5 ws_market 断连检测（last_klines_ts）

**Files**：`exchange/ws_market.py`（`__init__` :52-71、`_upsert` :135-148、`watch_ohlcv_for` :180-200）

**Interfaces**：
- 产出：`MarketDataHub.last_klines_ts: dict[tuple[str, str], float]`
- 产出：`MarketDataHub.max_stale_seconds: int`（来自 settings，默认 0=禁用）
- 产出：`MarketDataHub.last_kline_ts(symbol, timeframe) -> float`（供引擎读取）

**实现**：
1. `__init__`（:52-71）新增：
   ```python
   self.last_klines_ts: dict[tuple[str, str], float] = {}
   self.max_stale_seconds = settings.max_stale_seconds   # 0=禁用
   ```
2. `_upsert`（:135-148）中，当有**新 K 线追加**（`changed=True` 分支，:144）时更新：
   ```python
   if changed:
       self.last_klines_ts[key] = time.time()
   ```
3. 新增读取方法：
   ```python
   def last_kline_age(self, symbol: str, timeframe: str) -> float:
       """当前 symbol/tf 最近一根 K 线距今秒数（无数据返回无限大）。"""
       ts = self.last_klines_ts.get((symbol, timeframe))
       return float("inf") if ts is None else time.time() - ts
   ```
4. `watch_ohlcv_for` 成功分支（:185-196）设 `last_klines_ts`（`_upsert` 内已处理），异常重连（:197-200）不动。

**防回归红线**：
- `_upsert` 的数据缓冲逻辑（同 ts 覆盖/追加/超 max_candles 裁剪）零改动；`changed` 返回值语义不变
- `watch_symbols`（ticker 循环）零改动；ticker 不算"K线新鲜度"（对账/平仓只认 K 线）

**验证**：
- 测试：构造 hub（不用真实交易所，monkeypatch `ex.watch_ohlcv` 抛异常）→ `last_kline_age` 随时间增长；`_upsert` 新 K 线后归零

---

## T5. 🟠5 断链自动平仓（settings + 引擎检查）

**Files**：`config/settings.py`（`Settings` :61-110）、`engine/trading_engine.py`（`_portfolio_loop`/新增检查方法）

**Interfaces**（契约见 architecture.md §2.3）：
- 产出：`Settings.max_stale_seconds: int = 0`（0=禁用，默认安全）
- 产出：SYSTEM 事件 `kind: "market_stale"` / `"market_recovered"`；平仓 SIGNAL 事件 reason="交易所断连自动平仓"

**实现**：
1. `config/settings.py` `Settings` 类（:100-110 交易参数段）新增：
   ```python
   max_stale_seconds: int = 0   # 交易所断链自动平仓：0=禁用；>0=允许的最大 K 线滞留秒数（如 300）
   ```
2. `engine/trading_engine.py` `__init__` 新增状态：`self._stale_alerted: bool = False`（防重复告警/重复平仓）
3. `_portfolio_loop` 追加调用（与 T2 并列）：
   ```python
   await self._daily_reconcile_if_due()
   await self._check_market_stale()
   ```
4. 新增方法：
   ```python
   async def _check_market_stale(self) -> None:
       """断链检测：K 线超过 max_stale_seconds 未更新 → live 平仓 / 其他告警。
       恢复后发布 recovered 事件，且不自动重新开仓（需人工）。"""
       if not self.hub or settings.max_stale_seconds <= 0:
           return
       stale = self.hub.last_kline_age(self.symbol, self.timeframe) > settings.max_stale_seconds
       live = getattr(self, "trading_mode", "paper") in ("live",)
       if stale and not self._stale_alerted:
           self._stale_alerted = True
           await self.bus.publish(Event(EventType.SYSTEM, {
               "kind": "market_stale", "symbol": self.symbol,
               "timeframe": self.timeframe,
               "stale_sec": int(self.hub.last_kline_age(self.symbol, self.timeframe)),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           }, source="engine"))
           log.warning("[ws] 行情断流 %ss，超过阈值 %ss%s",
                       int(self.hub.last_kline_age(self.symbol, self.timeframe)),
                       settings.max_stale_seconds,
                       "（live 模式：触发自动平仓）" if live else "（非 live：仅告警）")
           if live:
               # 有持仓才平仓（无持仓不发信号）
               position = self._current_position_qty()
               if position > 0:
                   await self.bus.publish(Event(EventType.SIGNAL, {
                       "symbol": self.symbol, "side": "sell", "qty": position,
                       "reason": "交易所断连自动平仓", "strategy": self.strategy.name,
                   }, source="engine"))
       elif not stale and self._stale_alerted:
           self._stale_alerted = False
           await self.bus.publish(Event(EventType.SYSTEM, {
               "kind": "market_recovered", "symbol": self.symbol,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           }, source="engine"))
           log.info("[ws] 行情恢复（后续开仓需人工确认）")
   def _current_position_qty(self) -> float:
       if self.paper and self.paper_account:
           return self.paper_account.positions.get(self.symbol, {}).get("qty", 0.0)
       if self.exchange and self.exchange.has_private:
           bal = self._live_balance_cache if hasattr(self, "_live_balance_cache") else {}
           base = self.symbol.split("/")[0]
           return float((bal.get("balances") or {}).get(base, {}).get("total", 0.0)) if bal else 0.0
       return 0.0
   ```

> 注意：平仓 SIGNAL 事件由风控/订单管线消费（`risk.check` 平仓豁免 + `order_manager.place`）。**复用现有 SIGNAL 通道**，避免新建专用通道。若实测发现 SIGNAL 通道被策略互斥锁阻塞（`_on_candle_locked` 持锁），可改为直接在 `_check_market_stale` 内调用 `order_manager.place`（本模块独占 trading_engine.py 与 order_manager.py 不占——不越界即可，实现时选最简路径）。

**防回归红线**：
- `max_stale_seconds=0`（默认）→ `_check_market_stale` 立即 return，零行为变化
- 不自动"重新开仓"：recovered 后只发事件+日志，恢复手动的语义由用户操作（风控不放开自动重开仓）
- `_portfolio_loop` 15s 周期、异常兜底不变；`SIGNAL`/`SYSTEM` 事件 payload 结构不变

**验证**：
- 模拟盘：临时把阈值改小（如 5s），停止 hub（monkeypatch `_upsert` 不更新）→ 5s 后 `[ws] 行情断流` 日志 + simulated 模式不平仓；live 模式（用 kv reconfig）→ SIGNAL 平仓事件
- 恢复 hub → `market_recovered` 事件

---

## 交付验收清单（模块 C）

- [ ] `python -m compileall -q engine exchange config web/api/risk.py web/api/portfolio.py`
- [ ] `python -m pytest -q`（现有 risk/portfolio/engine 测试全绿 + 新增 reconcile/stale/manual 测试）
- [ ] 新增 pytest：cooldown_status manual_recovery 分支、clear-cooldown 端点、reconcile 差异三元组、stale 平仓
- [ ] 手工模拟盘跨 0 点（放宽 5 分钟窗）对账日志出现 `[reconcile]`
- [ ] 手工风控页：开启"冷却后需人工确认"，冷却到期显示 awaiting_clear，点击解除恢复
- [ ] 手工关卡：默认 settings（max_stale_seconds=0）断流时只告警不平仓
- [ ] `run.py backtest --source demo --strategy dual_ma` 回归（引擎改动不影响回测）