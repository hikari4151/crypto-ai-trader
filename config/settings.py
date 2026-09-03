"""统一配置：从 .env + config/config.yaml 加载（yaml 键名与字段同名，扁平结构）。

非敏感默认值可被 Web 界面覆盖。优先级：环境变量 > .env > config.yaml > 代码默认。
"""
import logging
import socket
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 常见本地代理端口（Clash / V2Ray / FClash 等），用于自动探测
_COMMON_PROXY_PORTS = (7890, 7891, 7892, 7897, 7898, 1080, 10808, 10809, 8888, 33210, 2080, 2081, 8080)

# P2-4：代理探测结果进程级缓存（探测是阻塞 socket 连接，最坏
# 13 端口 × 0.4s ≈ 5s；缓存后仅首次访问触发，lifespan 异步预热避开请求路径）
_PROXY_CACHE: Optional[str] = None
_PROXY_READY: bool = False
_PROXY_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def probe_local_proxy() -> str:
    """自动探测本机正在监听的 HTTP(S) 代理端口，返回如 http://127.0.0.1:7890，未找到返回空串。

    探测结果是进程级缓存，只做一次端口连通性检查（耗时约几十毫秒）。
    """
    for port in _COMMON_PROXY_PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                url = f"http://127.0.0.1:{port}"
                log.info("自动探测到本地代理: %s", url)
                return url
        except OSError:
            continue
    return ""


class _Utf8YamlSource(YamlConfigSettingsSource):
    """UTF-8 读取 config.yaml：pydantic-settings 默认以系统编码（中文 Windows=GBK）
    文本模式打开，UTF-8 中文注释直接 UnicodeDecodeError（启动即崩）。
    """

    def _read_file(self, file_path):
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        yaml_file=str(ROOT / "config" / "config.yaml"),
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        """注册 config/config.yaml 为配置源（曾仅声明 yaml_file 无 source，文件被静默忽略）。"""
        return (init_settings, env_settings, dotenv_settings,
                _Utf8YamlSource(settings_cls), file_secret_settings)

    # 安全
    master_key: str = ""       # 可选：固定主密钥（.env 配置），留空则自动生成 data/master.key
    # Web API 访问令牌：留空则启动时自动生成并打印到日志（前端通过 Cookie 自动携带）
    api_token: str = ""
    # 数据库（默认绝对路径，避免从其他目录启动时数据分裂到 CWD）
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'trader.db'}"
    # AI
    ai_provider: str = "openai"
    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_max_retries: int = 3
    ai_rate_limit_rpm: int = 30
    ai_rate_limit_min_interval: float = 2.0
    # P1：结构化输出用真 JSON Schema（response_format=json_schema）而非自由文本描述。
    # 默认关（向后兼容）：开启后仅对"支持 json_schema 的模型"生效，
    # 不支持/报错时自动回退 json_object。DeepSeek 等不支持 json_schema 的模型不受影响。
    ai_use_json_schema: bool = False
    # P5：按功能路由模型（feature -> 模型名），如 {"market_analysis": "gpt-4o", "factor_mine": "gpt-4o-mini"}。
    # 高价值决策用强模型、批量化用便宜模型；可同时在数据库 KV `ai_model_map` 覆盖。
    ai_model_map: dict[str, str] = {}
    # 交易
    paper_trading: bool = True
    log_level: str = "INFO"
    log_dir: str = str(ROOT / "data" / "logs")
    # Web
    web_host: str = "127.0.0.1"  # 默认仅本机访问；需要局域网访问时用 --host 0.0.0.0 或改 .env
    web_port: int = 8000
    # 网络代理（用于访问 OKX / Bitget / Bybit 等受限交易所）
    # 留空则自动探测本机常见 Clash/V2Ray 端口；也可显式指定如 http://127.0.0.1:7890
    proxy_url: str = ""
    # 默认
    default_exchange: str = "binance"
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "1h"
    default_start_cash: float = 10000.0
    trading_auto_start: bool = False
    ai_market_analysis_interval: int = 600
    ai_optimize_interval: int = 86400
    ai_review_interval: int = 604800
    default_strategy: str = "dual_ma"
    # 交易所断链自动平仓：None=按当前周期自动取 2 根 K 线时长；0 或负数=显式关闭；>0=自定义秒数
    max_stale_seconds: Optional[int] = None
    # 纸面/模拟成交滑点（默认与回测引擎一致 0.0005，避免纸面比回测更乐观；
    # 仅对市价单生效，限价单按限价成交不叠加滑点）
    paper_slippage: float = 0.0005
    # 成交时点：True = 信号在下一根K线开盘成交（与回测一致，杜绝前视、更保守）；
    # False = 当前K线收盘即时成交（默认，向后兼容）。
    trade_on_open: bool = False
    # 实盘服务端兜底止损（仅 live 模式生效）：成交后在交易所挂触发式市价卖单，
    # 进程崩溃/断网/机器休眠时交易所侧仍有一道防线
    protective_stop_enabled: bool = True
    # 策略 params 里没有 stop_loss_pct 时使用的止损距离（相对持仓均价）
    protective_stop_default_pct: float = 0.07
    # 在引擎软止损基础上再往深处让出的幅度：保证正常情况下由引擎先平仓，
    # 兜底单只在本地链路真的失效时才成交
    protective_stop_buffer_pct: float = 0.02
    # AI 参数热更新验证门：应用 AI 建议参数前先跑回测 + 过拟合校验，通过才应用
    ai_optimize_validate: bool = True
    # 验证用历史K线根数（建议 >= 400 以便过拟合检测可执行）
    ai_optimize_validation_bars: int = 800

    # 持续进化引擎（默认开启，间隔秒）
    evolve_enabled: bool = True
    evolve_factor_miner_interval: int = 60     # 因子挖掘训练间隔：60s（连续训练）
    evolve_strategy_drl_interval: int = 60      # 策略 DRL 训练间隔：60s（连续训练）
    evolve_meta_interval: int = 600             # 元策略训练间隔
    evolve_rolling_window: int = 5000             # 滚动窗口K线数
    evolve_factor_miner_episodes: int = 8         # 每轮因子挖掘训练轮数（连续训练取小值）
    evolve_strategy_drl_episodes: int = 8          # 每轮策略 DRL 训练轮数（连续训练取小值）
    # 策略 DRL 网络规模：导出 Pine 时权重必须整段内联，故默认取「装得进 Pine」的组合。
    # 旧配置 state_window=20 + hidden=[64,64] = 21184 权重 ≈700KB 脚本 → 进化产物
    # 永远没有可运行的自动交易代码（图上买卖点无从谈起）。改回大网络即放弃 Pine 导出。
    evolve_strategy_drl_state_window: int = 4
    evolve_strategy_drl_hidden: list[int] = [16, 16]
    evolve_meta_episodes: int = 8                 # 每轮元策略训练轮数（连续训练取小值）
    evolve_rollback_threshold: float = 0.95       # 新模型低于最佳模型此比例时回退（仅旧 fitness 为正时生效）
    evolve_max_versions: int = 10                 # 保留的历史版本数
    evolve_symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # 训练标的池（留单个=固定品种训练）
    evolve_timeframe: str = ""                  # 训练时间周期（空则跟随 default_timeframe；指定如 "4h" 则独立训练周期）
    evolve_oos_min_bars: int = 250                # OOS 安检最小样本数（<此值视为无样本外证据）
    evolve_vol_penalty: float = 20.0              # 策略 DRL 波动率惩罚（与 train_drl 默认对齐，原 0.5 偏小压不住波动）
    evolve_cross_symbol_oos: bool = True          # 是否启用跨标的传导 OOS（P1-8）
    evolve_min_new_bars: int = 2                  # 增量训练触发阈值：新增 K 线达此根数才训练（1h 周期 2 根≈2 小时）

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    def _resolve_proxy_sync(self) -> str:
        """实际探测逻辑（阻塞 socket 连接，最坏 13 端口 × 0.4s ≈ 5s）。
        仅首次经 _ensure_proxy 调用，结果进程级缓存。"""
        if self.proxy_url and self.proxy_url.strip():
            url = self.proxy_url.strip()
            try:
                parts = urlsplit(url if "://" in url else "http://" + url)
                host = parts.hostname or "127.0.0.1"
                port = parts.port or 80
                with socket.create_connection((host, int(port)), timeout=0.4):
                    return url
            except (OSError, ValueError):
                log.warning("显式代理 %s 不可达（按 host 探测），回退自动探测", url)
                return probe_local_proxy()
            return url
        return probe_local_proxy()

    def _ensure_proxy(self) -> str:
        """缓存化访问：探测结果进程级只算一次（线程安全），后续 O(1) 返回。"""
        global _PROXY_CACHE, _PROXY_READY
        with _PROXY_LOCK:
            if not _PROXY_READY:
                try:
                    _PROXY_CACHE = self._resolve_proxy_sync()
                except Exception as e:  # noqa: BLE001
                    log.warning("[settings] 代理探测失败，置空: %s", e)
                    _PROXY_CACHE = ""
                _PROXY_READY = True
            return _PROXY_CACHE or ""

    @property
    def resolved_proxy(self) -> str:
        """实际生效的代理地址（探测结果进程级缓存，仅首次可能阻塞）：
        - 显式配置：按 URL 的 host 探测（曾写死 127.0.0.1，远程代理被静默丢弃）；
          可达性检查失败仅告警并回退自动探测，但显式配置仍优先返回。
        - 留空：自动探测本机常见代理端口。
        - P2-4：结果缓存一次（首次调用最坏 ~5s），lifespan 经 prewarm_proxy
          异步预热，请求路径不再在事件循环上同步阻塞。
        """
        return self._ensure_proxy()

    async def prewarm_proxy(self) -> None:
        """异步预热：把阻塞 socket 探测放线程池（asyncio.to_thread），
        服务启动时调用一次，首个行情请求不再同步阻塞数秒。幂等。"""
        import asyncio
        try:
            await asyncio.to_thread(self._ensure_proxy)
        except Exception as e:  # noqa: BLE001
            log.warning("[settings] 代理预热失败: %s", e)


settings = Settings()
