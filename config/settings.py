"""统一配置：从 .env + config/config.yaml 加载，非敏感默认值可被 Web 界面覆盖。"""
import logging
import socket
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 常见本地代理端口（Clash / V2Ray / FClash 等），用于自动探测
_COMMON_PROXY_PORTS = (7890, 7891, 7892, 7897, 7898, 1080, 10808, 10809, 8888, 33210, 2080, 2081, 8080)


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        yaml_file=str(ROOT / "config" / "config.yaml"),
        extra="ignore",
        case_sensitive=False,
    )

    # 安全
    master_key: str = ""
    # Web API 访问令牌：留空则启动时自动生成并打印到日志（前端通过 Cookie 自动携带）
    api_token: str = ""
    # 数据库
    database_url: str = "sqlite+aiosqlite:///./data/trader.db"
    # AI
    ai_provider: str = "openai"
    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_temperature: float = 0.7
    ai_max_tokens: int = 2048
    ai_max_retries: int = 3
    ai_rate_limit_rpm: int = 30
    ai_rate_limit_min_interval: float = 2.0
    # 交易
    paper_trading: bool = True
    log_level: str = "INFO"
    log_dir: str = "./data/logs"
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
    ai_analysis_max_candles: int = 60
    default_strategy: str = "dual_ma"
    # 风控默认值
    risk_max_loss_per_trade_usd: float = 50.0
    risk_max_daily_loss_usd: float = 200.0
    risk_min_order_value_usd: float = 10.0
    risk_max_trades_per_hour: int = 20
    risk_max_position_pct: float = 0.9
    # 回测默认值
    bt_default_fee_rate: float = 0.001
    bt_default_slippage: float = 0.0005
    bt_default_start_cash: float = 10000.0

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def resolved_proxy(self) -> str:
        """实际生效的代理地址：
        - 若 .env 显式配置且端口可连，直接使用；
        - 否则自动探测本机常见代理端口。
        """
        if self.proxy_url and self.proxy_url.strip():
            url = self.proxy_url.strip()
            port = url.rsplit(":", 1)[-1]
            if port.isdigit():
                try:
                    with socket.create_connection(("127.0.0.1", int(port)), timeout=0.4):
                        return url
                except OSError:
                    log.warning("显式代理 %s 不可达，尝试自动探测", url)
                    return probe_local_proxy()
            return url
        return probe_local_proxy()


settings = Settings()