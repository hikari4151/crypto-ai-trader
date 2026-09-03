# M5 数据与风控增强 — 实施计划

> 独占文件：`backtest/data_loader.py`、`backtest/metrics.py`
> 共享文件：`engine/risk.py`（与 M6 共享，但改动区域不同）
> 任务：历史数据分页拉取 + 风控规则 DB fallback + 基准对比添加交易成本

---

## 问题分析

### P1. 历史数据分页拉取

**当前**：`load_from_exchange()`（`backtest/data_loader.py` 行 34-50）默认 `limit=1000`，使用 `ccxt.fetch_ohlcv` 一次性拉取。

**问题**：交易所 API 通常有单次请求条数上限（如 Binance 最多 1000 根），无法获取更长时间范围的历史数据。

**目标**：实现分页循环拉取，通过 `since` 参数逐页向后翻，直到拉满所需数据量或到达当前时间。

### P2. 风控规则 DB 不可用时 fallback

**当前**：`RiskManager.get_rules()`（`engine/risk.py` 行 140-160）调用 `self.db.kv_json_get("risk_rules")`，若 DB 不可用（抛出异常），异常会传播到 `check()` 方法，被 `except Exception` 捕获后返回 `(False, "风控检查异常")`——所有交易被拦截。

**问题**：DB 临时不可用不应导致交易完全停止，应 fallback 到默认规则。

**目标**：在 `get_rules()` 中添加 try/except，DB 异常时返回 `DEFAULT_RULES` 并告警。

### P3. 基准对比添加交易成本

**当前**：`compute_benchmark()`（`backtest/metrics.py` 行 10-32）使用原始收盘价序列计算买入持有收益，**不含手续费和滑点**。

**问题**：基准收益不含交易成本，策略收益含交易成本，两者对比时基准被高估，超额收益被低估。

**目标**：在 `compute_benchmark()` 中添加可选参数 `fee_rate` 和 `slippage`，计算考虑交易成本后的基准收益。但注意：买入持有策略只交易两次（买入+卖出），成本影响很小。

---

## T1. 历史数据分页拉取

**Files**：`backtest/data_loader.py`

### 改动方案

```python
async def load_from_exchange(exchange_id: str, symbol: str, timeframe: str = "1h",
                             since: Optional[datetime] = None, 
                             limit: int = 1000,
                             max_candles: int = 10000) -> pd.DataFrame:
    """通过交易所公开 REST 接口拉取历史K线，支持分页自动翻页。
    
    max_candles: 最大拉取条数（默认 10000，超过此值停止）。
    limit: 单次请求条数（默认 1000，交易所上限）。
    """
    import ccxt.async_support as ccxt
    proxy = settings.resolved_proxy or None
    cfg: dict = {"enableRateLimit": True}
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    ex = getattr(ccxt, exchange_id)(cfg)
    try:
        await ex.load_markets()
        all_rows = []
        since_ms = int(since.timestamp() * 1000) if since else None
        fetch_limit = min(limit, max_candles)
        while len(all_rows) < max_candles:
            rows = await ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=fetch_limit)
            if not rows:
                break
            # 去重：防止最后一条与上一页最后一条重叠
            new_rows = [r for r in rows if not all_rows or r[0] > all_rows[-1][0]]
            if not new_rows:
                break
            all_rows.extend(new_rows)
            # 下一页 from 最后一根K线的时间戳
            since_ms = rows[-1][0] + 1
            if len(rows) < fetch_limit:
                break  # 已拉取到最新数据
            if len(all_rows) >= max_candles:
                all_rows = all_rows[:max_candles]
                break
        log.info("[回测] 从 %s 分页拉取 %s %s 共 %d 根K线", exchange_id, symbol, timeframe, len(all_rows))
        return _to_df(all_rows)
    finally:
        await ex.close()
```

### 接口变更

```python
# 函数签名新增参数
async def load_from_exchange(exchange_id, symbol, timeframe="1h",
                              since=None, limit=1000, max_candles=10000) -> pd.DataFrame:
```

---

## T2. 风控规则 DB fallback

**Files**：`engine/risk.py`

### 改动方案

```python
async def get_rules(self) -> dict:
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
        # DB 不可用时不写回（避免重复异常）
        try:
            await self.db.kv_json_set("risk_rules", rules)
        except Exception as e:
            log.warning("[risk] DB 写入风控规则失败（忽略）: %s", e)
    # 合并默认值...
    self._rules_cache = rules
    self._rules_cache_ts = now
    return rules
```

---

## T3. 基准对比添加交易成本

**Files**：`backtest/metrics.py`

### 改动方案

当前 `compute_benchmark` 使用原始 close 序列，不含交易成本。

买入持有策略只交易两次（期初买入、期末卖出），成本为：
- 买入：`start_cash * fee_rate`
- 卖出：`final_value * fee_rate`

```python
def compute_benchmark(closes: np.ndarray, start_cash: float, timeframe: str = "1h",
                      fee_rate: float = 0.0, slippage: float = 0.0) -> dict:
    """买入持有基准权益曲线与本段区间收益。
    
    fee_rate: 手续费率（双向），默认 0.0 保持向后兼容。
    slippage: 滑点（单向），默认 0.0。
    """
    eq = np.asarray(closes, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {
            "buy_hold_ret": 0.0, "excess_return": 0.0,
            "information_ratio": 0.0, "excess_max_drawdown": 0.0,
            "bench_equity_curve": [round(start_cash, 4)],
        }
    # 买入成本
    buy_cost = start_cash * fee_rate
    effective_cash = start_cash - buy_cost
    initial_units = effective_cash / eq[0] * (1 - slippage)  # 买入滑点
    # 期末价值
    final_value = initial_units * eq[-1]
    sell_cost = final_value * fee_rate
    final_value -= sell_cost
    # 基准权益曲线
    bench_equity = start_cash * eq / eq[0]  # 不含成本的权益曲线（用于超额收益计算）
    buy_hold_ret = final_value / start_cash - 1.0
    return {
        "buy_hold_ret": round(buy_hold_ret, 6),
        "excess_return": 0.0,
        "information_ratio": 0.0,
        "excess_max_drawdown": 0.0,
        "bench_equity_curve": [round(float(x), 4) for x in bench_equity],
    }
```

### 向后兼容

`compute_benchmark` 新增 `fee_rate=0.0` 和 `slippage=0.0` 默认参数，不传时行为与修改前完全一致。

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| `backtest/data_loader.py` | `load_from_exchange` 新增 `max_candles` 参数；实现分页循环拉取 | 无冲突 |
| `engine/risk.py` | `get_rules()` 中 `kv_json_get` 添加 try/except，DB 异常时返回 DEFAULT_RULES | 与 M6 共享（except 收紧），需协调 |
| `backtest/metrics.py` | `compute_benchmark` 新增 `fee_rate`/`slippage` 参数；计算含成本的基准收益 | 无冲突 |

---

## 验收标准

- [ ] `load_from_exchange` 分页拉取能返回超过 1000 根 K 线
- [ ] `load_from_exchange` 分页拉取不会重复/遗漏数据
- [ ] `risk.py` 的 `get_rules()` 在 DB 不可用时返回 `DEFAULT_RULES` 并告警，不抛异常
- [ ] `risk.py` 的 `get_rules()` 在 DB 正常时行为不变
- [ ] `compute_benchmark(fee_rate=0.001)` 的 `buy_hold_ret` 略低于 `compute_benchmark(fee_rate=0.0)`
- [ ] 不传 `fee_rate`/`slippage` 时，`compute_benchmark` 行为与修改前一致
- [ ] `python -m compileall -q backtest engine`