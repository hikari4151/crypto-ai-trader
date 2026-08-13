# Crypto AI Trader

模块化、可扩展的加密货币 AI 量化交易机器人。**所有 AI 能力都通过用户自填的 AI API（OpenAI / DeepSeek / 任意 chat/completions 兼容服务）实现，程序不内置任何 API 密钥。**

## 功能总览

- **多交易所**：Binance / OKX / Bybit / Bitget（ccxt 统一封装，读写权限分离）
- **实时行情**：内置「行情」视图——免密钥实时K线（MA5/10/20+成交量，多周期，缩放滑块）、热门币种、24h 涨跌榜，Binance 走官方 `.vision` 数据域国内可直连，其余交易所自动走本地代理
- **本地代理支持**：`PROXY_URL` 显式配置或自动探测 Clash/V2Ray 端口（7890/1080 等），OKX/Bitget/Bybit 等受限交易所连接自动注入代理
- **策略引擎**：可插拔框架，内置双均线、网格示例策略，支持自定义
- **AI 持续优化**：市场状态理解、策略参数动态优化（热更新）、每日/每周反思复盘
- **历史回测**：事件驱动引擎，计入手续费+滑点，输出收益/年化/回撤/夏普/胜率/盈亏比/权益曲线，前端图表展示
- **实时行情指标**：ccxt.pro WebSocket 订阅 + MA/MACD/RSI/布林带指标
- **AI 行情解读**：一键或定时把实时指标快照发给你的 AI API
- **钱包与持仓**：定时快照，USDT 折算，浮动盈亏
- **战绩看板**：每笔成交入库（SQLite/PostgreSQL），累计盈亏曲线、胜率、交易明细
- **AI 设置界面**：Web 页面填写 Provider/Endpoint/Key/模型/温度/Token，Key 加密存储、掩码展示
- **风控**：单笔最大亏损、每日最大亏损、频率限制、最小订单金额、仓位上限，界面动态调整
- **安全**：Fernet 加密密钥落盘、日志脱敏、绝不明文传输/记录密钥

## 架构

```
Web(FastAPI+Vue3) ──> AI设置/回测/战绩/风控 API
        │
        ▼
TradingEngine（事件驱动调度）
        │ 订阅
EventBus ◄── MarketDataHub(ccxt.pro WS)   ← 实时K线/指标
        │                │
        ▼                ▼
Strategy(on_candle)  AI Scheduler(解读/优化/复盘)
        │                │
        ▼                ▼
RiskManager ──> OrderManager ──> ccxt REST / PaperAccount ──> SQLite/Postgres
```

## 快速开始

### 1. 安装

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

要完全复现开发环境，可用锁定版本清单：

```bash
pip install -r requirements-lock.txt
```

### 2. 配置

```bash
cp .env.example .env      # 编辑：MASTER_KEY / DATABASE_URL / 模式等
# config/config.yaml 里可调默认交易对、周期、AI 任务频率、默认风控
```

- **纸面模式（默认）**：`PAPER_TRADING=true`，无需任何交易所密钥即可体验全流程。
- **实盘模式**：`PAPER_TRADING=false`，并在 Web 界面「AI 设置」页填写交易所密钥（加密存储），或在 `.env` 中注入。

### 3. 启动

```bash
python run.py web          # 打开 http://localhost:8000
```

> 默认只监听 `127.0.0.1`。如需局域网访问，显式配置 `WEB_HOST=0.0.0.0` 并自行确认访问控制。

### 4. 配置 AI

打开「AI 设置」页：

1. 选提供商：OpenAI / DeepSeek / 自定义兼容接口
2. 填 Endpoint（如 `https://api.deepseek.com/v1`）、模型（如 `deepseek-chat`）、API Key、温度、最大 Token
3. 点「测试连接」确认，再点「保存」；之后：
   - 仪表盘「立即解读」→ AI 行情解读
   - 引擎运行后 → 定时市场解读 / 每日参数优化 / 每周复盘（间隔在 `config/config.yaml` 调整）

### 5. 回测

- 网页端：「历史回测」页选择数据源（演示数据 / 交易所 / CSV）与策略，运行后查看图表与报告。
- 命令行：

```bash
python run.py backtest --source demo --strategy dual_ma
python run.py backfill --exchange binance --symbol BTC/USDT --timeframe 1h --limit 1000
python run.py backtest --source csv --csv data/binance_BTCUSDT_1h.csv
```

## 自定义策略

在 `strategies/` 新增文件，继承 `Strategy` 并实现 `on_candle`，然后在 `strategies/__init__.py` 注册：

```python
class MyStrategy(Strategy):
    name = "my_strategy"
    default_params = {"threshold": 0.5}
    param_schema = {"threshold": {"type": "float", "min": 0.0, "max": 1.0}}

    def on_candle(self, ctx):        # ctx: symbol/price/position/cash/indicators
        ...
        return Signal(self.symbol, "buy", size_pct=0.5, reason="...")
```

注册后即可被引擎、回测、前端、AI 参数优化器自动发现。

## AI 提示词与 JSON 约定

所有 AI 调用都走 `ai/client.py`，统一处理：错误重试（指数退避）、速率限制（滑动窗口）、JSON 稳健解析、密钥绝不落日志。市场解读/参数优化/复盘三套提示词在 `ai/prompts.py`，可直接定制。

## 测试

单元/集成测试位于 `tests/`，运行：

```bash
pytest
```

## 安全说明

- 交易所与 AI 密钥用 Fernet 加密后存入数据库（主密钥来自 `MASTER_KEY` 或 `data/master.key`）。
- 所有 API 响应只返回掩码（`sk-****abcd`），前端永不接触明文。
- 日志过滤器自动脱敏疑似密钥串。
- 敏感文件（`.env`、`data/master.key`、数据库、日志）已列入 `.gitignore`，提交前请确认。
- 实盘前请务必：小资金验证、核对币对精度/最小下单量、在交易所开启相应交易权限并妥善保管密钥。

## 目录速览

| 目录 | 说明 |
|---|---|
| `config/` | pydantic-settings 配置 + YAML 非敏感默认值 |
| `core/` | 事件总线、加密、数据库、日志 |
| `exchange/` | ccxt REST 封装、ccxt.pro WebSocket 行情、纸面账户 |
| `indicators/` | 技术指标计算（含向量化实现） |
| `strategies/` | 策略框架与内置策略 |
| `factors/` | 因子库、因子引擎与因子挖掘 |
| `ai/` | AI 调用管理器、提示词、解读/优化/复盘 |
| `drl/` | 深度强化学习（环境、网络、PPO Agent） |
| `backtest/` | 数据加载、回测引擎（事件驱动+向量化）、绩效指标、过拟合检查 |
| `engine/` | 交易调度、风控、下单、资产 |
| `web/` | FastAPI 后端 + Vue3 前端 |
| `scripts/` | 数据生成等辅助脚本 |
| `tests/` | 单元/集成测试 |
| `data/` | 运行时数据（日志、数据库、模型，已被 gitignore） |

> 免责声明：本项目仅供学习与研究，加密货币交易风险极高，使用前请充分测试并自担风险。
