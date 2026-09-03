# crypto_ai_trader 量化改进架构设计：模块拆分与接口契约

> 版本：v2.0（2026-08-15）
> 依据：docs/QUANT_ADVICE.md（10 项量化改进，P0-P2 优先级）
> 本文档为 5 个并行开发模块的**唯一契约来源**；module_A.md ~ module_E.md 为各模块执行计划。
> 原则：既有代码库**原地修改**，禁止新建 modules/ 目录；用文件所有权隔离并行开发。

---

## 1. 模块拆分表

| 模块 | 独占文件 | 任务（来自 QUANT_ADVICE.md） | 风险点 |
|---|---|---|---|
| **A 回测增强** | `backtest/metrics.py`、`backtest/engine.py`、`backtest/fast_engine.py`、`web/api/backtest.py`、`backtest/data_loader.py` | 🔴1 基准对比（metrics.py 新增 compute_benchmark + 回测结果含 benchmark 键 + 前端权益曲线双线）+ 🟡7 限价单模拟（BacktestConfig 新增 limit_order_model 参数 + 部分成交模拟） | 基准计算需传入原始 close 序列（非 equity_curve，因 equity_curve 已含手续费/滑点）；限价单模拟默认关闭（limit_order_model="none"），开启后回测结果会变——网格策略需验证；两引擎（engine.py + fast_engine.py）必须同步修改 |
| **B 因子系统** | `factors/mining.py`、`factors/model_factor.py`、`factors/library.py`、`factors/analysis.py`、`web/api/factors.py`、`indicators/technical.py` | 🔴2 跨品种验证（mining.py factor_quality_gate + model_factor 元数据 + 端点参数）+ 🟠6 IC 衰变自动下线（library.py 因子 alive 标记 + 滚动 IC 检测 + 策略隔离）+ 🟡10 因子扩展（library.py 注册 5 个新因子，含动量/波动率/成交量异常/价格位置/乖离率） | 跨品种数据需同周期同长度；IC 衰变检测的滚动 IC 窗口（30 期）和判定阈值（3 周期持续）需谨慎调参；`bias_20` 和 `vol_20` 已在 library.py 存在——新因子名需避免重复；策略隔离需要改 `strategies/factor_signal.py` 的 `on_candle` 方法（不在 B 独占文件内，需 E 或额外协调） |
| **C 实盘安全** | `engine/trading_engine.py`、`engine/portfolio.py`、`web/api/portfolio.py`、`engine/risk.py`、`web/api/risk.py`、`exchange/ws_market.py`、`config/settings.py` | 🔴3 每日对账（portfolio_loop 每日 0 点 fetch_balance 全量对比 + 告警）+ 🟠4 熔断（risk.py 冷却后人工确认恢复 + 端点）+ 🟠5 断链平仓（ws_market 断连检测 + 超时自动平仓 + settings 配置） | 每日对账的 `_do_fetch` 需抽取为公共方法供引擎侧和 API 侧复用；断链自动平仓是极端风控措施，默认关闭（max_stale_seconds=0）；平仓逻辑需保证在 live 模式下才触发，simulated 模式只告警；熔断的 manual_recovery 字段改变了 cooldown_status 返回结构，前端需同步更新 |
| **D DRL+AI** | `web/api/drl.py`、`drl/env.py`、`ai/optimizer.py`、`web/api/ai.py` | 🟡8 DRL 默认值（TrainIn 默认值调整 + env 配置日志）+ 🟡9 AI+网格串联（optimizer 返回 grid_center + auto-optimize 触发局部网格扫描→背靠背调 run_grid_scan，不碰 backtest.py） | DRL 默认值变更后现有模型的训练行为会变（reward 不同），但不影响推理，属 forward change；AI+网格串联中的 grid-scan 端点签名已有（`/api/backtest/grid-scan`），auto-optimize 完成后调用它需注意并发上限（`_BT_SEM` 限制 2 个并发回测） |
| **E 前端** | `web/static/index.html`（唯一文件） | 配合 A/C/D 的 API 变化：权益曲线加基准线、风控冷却解除按钮、DRL 训练提示文字、因子库页面扩展 | 244KB 单体文件，必须与各模块 API 响应契约保持一致；不删功能不破渲染 |

---

## 2. 跨模块接口契约

### 2.1 基准数据的 API 响应结构（🔴1 基准对比）

**回测结果字典**（`/api/backtest/run` 的进度缓存 `_PROGRESS[task_id]` 以及 `/api/backtest/results/{id}` 的 `metrics` 键）新增 `benchmark` 键：

```python
# 回测引擎返回的顶层 dict（engine.py/fast_engine.py 的 return 语句）
{
    "metrics": {..., "benchmark": {  # ← 新增
        "buy_hold_ret": 0.123456,         # 买入持有总收益率
        "excess_return": 0.045678,        # 策略收益 - 基准收益
        "information_ratio": 0.8765,      # 年化信息比率
        "excess_max_drawdown": 0.089123,  # 超额收益最大回撤
        "bench_equity_curve": [10000.0, 10123.45, ...]  # 基准每日权益序列（前端绘图用）
    }},
    "equity_curve": [...],
    ...
}
```

- `bench_equity_curve` 长度与 `equity_curve` 一致（每根 K 线一个点），首点为 `start_cash`
- 前端 `renderEquity` 函数接收 `bench_equity_curve` 作为第二条线（虚线，灰色）
- `compute_benchmark(closes, start_cash)` 在 `metrics.py` 中新增，入参为原始 close 价格数组（numpy array）

### 2.2 每日对账的 SYSTEM 事件 payload（🔴3 每日全量对账）

通过 `core.bus.EventBus` 发布 `EventType.SYSTEM` 事件：

```python
# 对账发现的差异事件
{
    "kind": "reconcile",
    "type": "MISSING_POSITION" | "UNKNOWN_POSITION" | "QTY_MISMATCH",
    "symbol": "BTC/USDT",
    "local_qty": 0.5,          # 本地持仓量（仅 QTY_MISMATCH/MISSING_POSITION 有）
    "exchange_qty": 0.3,        # 交易所持仓量（仅 QTY_MISMATCH/UNKNOWN_POSITION 有）
    "diff_pct": 0.4,            # 偏差百分比（仅 QTY_MISMATCH 有）
    "ts": "2026-08-15T00:00:00Z",
}
```

- `EventType.SYSTEM = "system"` 已在 `core/events.py:20` 定义，**无需修改**
- 对账仅在 `trading_mode == "live"` 时执行全量拉取，`paper`/`simulated` 跳过（纸面无真实余额可对比）
- 日志格式：`[reconcile] MISSING POSITION BTC/USDT: local=0.5 exchange=0.0`

### 2.3 断链自动平仓的 settings 配置键名（🟠5 断链平仓）

在 `config/settings.py` 的 `Settings` 类中新增：

```python
# 交易所断链自动平仓（默认关闭，0=禁用；大于 0 表示最大允许的 stale 秒数）
max_stale_seconds: int = 0
```

- `max_stale_seconds = 0` 表示禁用断链自动平仓（默认安全）
- 用户需在 `config/config.yaml` 或 `.env` 中显式设置：`max_stale_seconds=300`（5 分钟）
- `ws_market.py` 的 `MarketDataHub` 新增 `last_klines_ts: dict[tuple[str, str], float]` 记录每个 `(symbol, timeframe)` 最后一根 K 线的时间戳
- `trading_engine.py` 的 `_portfolio_loop` 中检查 `hub.last_klines_ts`，超时则发布 `SYSTEM` 事件 + 自动平仓

### 2.4 IC 衰变标记的因子库元数据格式（🟠6 因子滚动 IC 衰变自动下线）

`Factor` dataclass 新增 `alive` 字段，`meta()` 方法新增 `ic_decay` 键：

```python
# factors/base.py Factor 类新增字段
@dataclass
class Factor:
    ...
    alive: bool = True                     # 是否活跃（False=自动下线）
    ic_decay: dict = field(default_factory=lambda: {  # IC 衰变状态
        "rolling_ic": [],                  # 最近 30 期滚动 IC 值
        "rolling_icir": 0.0,              # 滚动 ICIR
        "last_updated": None,             # 最后更新时间
        "auto_offline": False,            # 是否因 IC 衰变而自动下线
        "auto_online": False,             # 是否因 IC 恢复而自动上线
    })

# meta() 的输出新增
{
    "key": "mom_20",
    "name": "20期动量",
    "alive": True,                         # ← 新增
    "ic_decay": {"rolling_ic": [...], ...},  # ← 新增
    ...
}
```

- `factor_signal.py` 的 `on_candle` 方法需检查 `f.alive`（在 `_factor_value` 中跳过 `alive=False` 的因子）
- `web/api/factors.py` 的 `/factors/library` 端点返回 `alive` 状态
- 下线的因子不删除，只标记 `alive=False`，保留历史记录

### 2.5 限价单模拟的 BacktestConfig 字段（🟡7 限价单部分成交模拟）

```python
# backtest/engine.py BacktestConfig 新增字段
@dataclass
class BacktestConfig:
    ...
    limit_order_model: str = "none"  # "none" | "partial" | "probabilistic"
```

- `"none"`：保持现有行为（市价成交），默认值
- `"partial"`：部分成交模拟（基于 K 线 OHLC 触碰逻辑）
- `"probabilistic"`：概率模型（无 tick 数据时与 `"partial"` 等价）

### 2.6 DRL 默认值变更（🟡8 DRL reward 默认推荐值）

```python
# web/api/drl.py TrainIn 默认值变更
entropy_coef: float = 0.05    # 原 0.03
reward_trend_align: float = 0.1   # 原 0.0
reward_dd_penalty: float = 0.5    # 原 0.0
reward_losing_penalty: float = 0.2   # 原 0.0
```

### 2.7 AI 优化器返回 grid_center（🟡9 AI+网格串联）

```python
# ai/optimizer.py optimize_price_action apply=False 时返回
{
    "params": {...},
    "reason": "...",
    "focus": "...",
    "grid_center": {...},   # ← 新增 = params 的副本（AI 建议参数作为网格扫描中心）
}
```

### 2.8 跨品种验证的 API 参数（🔴2 跨品种验证）

```python
# web/api/factors.py FactorMineIn 新增字段
class FactorMineIn(BaseModel):
    ...
    cross_symbols: str = ""  # 逗号分隔的交易对，如 "ETH/USDT,BNB/USDT"
```

### 2.9 熔断人工确认的端点与字段（🟠4 连续亏损熔断 + 人工确认恢复）

```python
# engine/risk.py cooldown_status 返回结构新增字段
{
    "active": True,
    "remaining_sec": 0,
    "consecutive_losses": 5,
    "flash_cooling": False,
    "flash_remaining_sec": 0,
    "manual_recovery": True,       # ← 新增
    "awaiting_clear": True,        # ← 新增（冷却时间已到，等待人工确认）
}

# web/api/risk.py 新增端点
# POST /api/risk/clear-cooldown → 调用 engine.risk.clear_cooldown()
```

---

## 3. 跨模块依赖关系

```
A (回测增强) ← 无依赖，纯增量
B (因子系统) ← D 的 DRL 因子挖掘产出写入 factor_custom，依赖 B 的 library 结构
C (实盘安全) ← 无依赖，纯增量
D (DRL+AI)   ← 依赖 C 的 trading_engine._portfolio_loop（AI scheduler 在其中）
               依赖 A 的 grid-scan 端点（auto-optimize 完成后触发）
E (前端)      ← 依赖 A/C/D 的 API 响应结构变化
```

**关键依赖**：D 模块的 `auto-optimize` 调用 `grid-scan` 端点（`/api/backtest/grid-scan`），但 A 模块的 `limit_order_model` 参数不改变 `grid-scan` 的输入/输出结构，因此无冲突。D 模块**不碰 `backtest.py`**，只通过 HTTP 调用 grid-scan（或直接调用 `run_backtest_fast` 的局部扫描）。

---

## 3.5 分工外的越界协调项（PM 仲裁）

以下改动点不在任何模块的独占文件清单内，但被 QUANT_ADVICE 任务引用。各模块计划已采用**防御式兼容**方案避免直接越界，是否纳入最终实现由 PM 统一协调：

| # | 越界点 | 相关任务 | 模块计划中的处理 | 建议归属 |
|---|---|---|---|---|
| 1 | `strategies/base.py` 的 `Signal` 类无 `order_type`/`limit_price` 字段 | 🟡7 限价单模拟 | A 用 `getattr(sig, "order_type", None)` 防御读取，字段缺失时 partial 模式退化为市价 + 告警 | PM 协调：如需完整支持给 Signal 加两个可选字段（影响所有策略构造点），或暂缓限价单实信号支持（仅预留框架） |
| 2 | `strategies/factor_signal.py` 的 `on_candle` 未检查 `alive` | 🟠6 IC 衰变策略隔离 | B 交付 `library.factor_is_live(key)` 辅助函数 + 测试；是否让 factor_signal 调用由 PM 协调 | PM 协调（factor_signal.py 不在任何模块独占清单） |
| 3 | `engine/trading_engine.py` 的 `_ai_scheduler` 不做周期网格扫描 | 🟡9 每周网格挂载 | D 交付 `_run_nearby_grid_scan` 纯函数；挂到 `_ai_scheduler` 由模块 C 协调（C 独占 trading_engine.py），周期沿用 `ai_optimize_interval` | 模块 C（在 T6 的 `_ai_scheduler` 内叠网格扫描） |
| 4 | `engine/trading_engine.py` 的 `_ai_scheduler` 周期性调用因子滚动 IC 刷新 | 🟠6 滚动 IC 计算 | B 交付 `library.periodic_ic_refresh` 幂等纯函数；定时调用由模块 C 在 `_portfolio_loop`/`_ai_scheduler` 内挂载 | 模块 C（C 独占 trading_engine.py） |
| 5 | `strategies/base.py`/`strategies/repository.py` 内置策略注册表 | 🟡10 因子扩展可用性 | B 只注册因子，factor_signal 策略回测可用性由因子库自动覆盖；无需登记策略表 | 无需改动 |
| 6 | `core/events.py` | 🔴3/🟠5 SYSTEM 事件 | `EventType.SYSTEM` 已存在（:20），**零改动复用** | 无需改动 |

---

## 4. 防回归红线（全局）

| 行为 | 现状锚点 | 保持内容 |
|---|---|---|
| 回测结果结构 | engine.py/fast_engine.py 的 return dict | 现有 `metrics`/`equity_curve`/`trades`/`symbol`/`timeframe`/`strategy`/`params` 键不可删除或改名；`benchmark` 为新增键 |
| 回测进度上报 | backtest.py `_make_progress_cb` 写入 `_PROGRESS[task_id]` | 键结构（i/n/pct/kline_history/last_price/last_item/running/done/metrics/elapsed_sec/backend）不变 |
| 风控 check 签名 | risk.py `check(signal, price, cash, position_value, daily_pnl, trade_times, entry_price, vol_ratio)` | 参数顺序和类型不可变；返回 `(bool, str)` 格式不变 |
| 因子库接口 | `/api/factors/library` 返回 `{"factors": [Factor.meta()]}` | 现有 `key`/`name`/`category`/`description`/`default_params` 键不可删；`alive`/`ic_decay` 为新增键 |
| DRL 训练进度 | drl.py `_TRAINING[task_id]` 键结构 | `running`/`episode`/`episodes`/`done`/`error`/`model_name`/`history` 等键不可变 |
| 引擎状态 | engine.status() 返回 dict | `running`/`paper`/`mode`/`exchange`/`symbol`/`timeframe`/`strategy`/`degraded` 键不可变 |
| 前端路由 | index.html 的 `renderEquity` 签名 | `renderEquity(chart, data, key)` 三个参数不变；新增 `benchmark` 数据参数为第四个可选参数 |