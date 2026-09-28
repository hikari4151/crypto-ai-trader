"""分级日志 + 持久化 + 密钥脱敏（绝不落盘 API Key）。"""
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SENSITIVE = re.compile(r"(sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._-]{8,}|api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{6,})", re.I)
# 高熵串：交易所纯字母数字密钥（Binance/OKX/Bybit 无前缀无标签）——长度 ≥20 且大小写字母+数字混合
_HIGH_ENTROPY = re.compile(r"\b(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{20,}\b")


def _redact(value):
    """对任意日志参数脱敏：字符串直接替换，dict/list 递归（曾只处理 str，dict 整体跳过）。"""
    if isinstance(value, str):
        v = _SENSITIVE.sub("***REDACTED***", value)
        return _HIGH_ENTROPY.sub("***REDACTED***", v)
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


class RedactFilter(logging.Filter):
    """在写日志前把疑似密钥替换为 ***，防止误落盘。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = _redact(record.args)
                else:
                    # 数值参数必须保持原类型，否则 %d/%.2f 等格式符会抛 TypeError
                    record.args = tuple(_redact(a) for a in record.args)
        except Exception:
            pass
        return True


class ResilientRotatingFileHandler(RotatingFileHandler):
    """容错轮转文件 handler：轮转失败（多实例并发持锁日志文件）静默降级。

    2026-09-28 真实启动实测：第二个实例起服务时 trader.log 被并行实例持锁，
    基类 doRollover 抛 PermissionError → logging 的 emit 捕获后调用 handleError，
    把整段 traceback 打到 stderr，污染启动日志——用户无法分辨是启动失败还是
    仅"日志轮转没轮成"。这里吞掉轮转异常只记录一次 warning 并降级为继续写
    当前文件（宁可日志文件不轮转，也不能让日志系统干扰启动/可观测性）。
    """

    def doRollover(self) -> None:  # noqa: N802（基类同名）
        try:
            super().doRollover()
        except OSError as e:  # PermissionError(32) 文件被占 / FileNotFoundError
            logging.getLogger("logging_config").warning(
                "日志轮转失败，继续写当前文件（可能多实例并存）: %s", e)


# 交易事件 logger 前缀：下单/成交/撤单/对账/风控熔断/保护止损（GAP-07 日志分级）
_TRADING_LOGGER_PREFIXES = ("engine.",)
_TRADING_LOGGER_EXACT = ("core.bus",)


class TradingEventFilter(logging.Filter):
    """只放行交易事件日志（engine.* / core.bus），把 HTTP 访问日志挡在 trading.log 外。"""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        return (name.startswith(_TRADING_LOGGER_PREFIXES)
                or name in _TRADING_LOGGER_EXACT)


def uvicorn_log_config(log_dir: str = "./data/logs") -> dict:
    """uvicorn 日志配置：access/error 写入 server.log（RotatingFileHandler 轮转）。

    解决：run.py 直跑 uvicorn 时 access 日志默认只进 stdout（被 supervisor
    重定向成 server.log 直写、无轮转）；这里接管为带轮转的 file handler。
    disable_existing_loggers=False：不重置应用已配置的 root handlers
    （trader.log/trading.log 保持 setup_logging 的配置）。
    """
    server_log = str(Path(log_dir) / "server.log")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S"},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
            "server_file": {
                "class": "core.logging_config.ResilientRotatingFileHandler",
                "filename": server_log, "maxBytes": 5 * 1024 * 1024, "backupCount": 5,
                "encoding": "utf-8", "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "server_file"], "level": "INFO",
                        "propagate": False},
            "uvicorn.error": {"handlers": ["console", "server_file"], "level": "INFO",
                              "propagate": False},
            "uvicorn.access": {"handlers": ["server_file"], "level": "INFO",
                               "propagate": False},
        },
    }


def setup_logging(level: str = "INFO", log_dir: str = "./data/logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # 幂等：run.py 与 web.main 各调用一次，不去重会让每条日志在 trader.log 落两遍
    # （排障时会误判为「两个引擎实例在并行跑」）
    for h in list(root.handlers):
        if getattr(h, "_trader_handler", False):
            root.removeHandler(h)
            h.close()

    handler = ResilientRotatingFileHandler(
        Path(log_dir) / "trader.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    # 交易事件单独成文件（GAP-07④）：运维只需盯 trading.log 就能看到
    # 下单/成交/撤单/对账差异/风控熔断，不被 HTTP 轮询日志淹没
    trading_handler = ResilientRotatingFileHandler(
        Path(log_dir) / "trading.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    trading_handler.setFormatter(fmt)
    trading_handler.addFilter(TradingEventFilter())
    for h in (handler, console, trading_handler):
        # 幂等标记：上面按 getattr(h, "_trader_handler") 识别本模块安装的 handler，
        # 动态属性用 setattr 赋值（三者的具体类型不同，直接属性赋值 mypy 过不了 union）。
        setattr(h, "_trader_handler", True)  # noqa: B010
        h.addFilter(RedactFilter())
        root.addHandler(h)
    root.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
