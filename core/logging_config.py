"""分级日志 + 持久化 + 密钥脱敏（绝不落盘 API Key）。"""
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SENSITIVE = re.compile(r"(sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._-]{8,}|api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{6,})", re.I)


class RedactFilter(logging.Filter):
    """在写日志前把疑似密钥替换为 ***，防止误落盘。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _SENSITIVE.sub("***REDACTED***", str(record.msg))
            if record.args:
                # 只对字符串参数脱敏；数值参数必须保持原类型，否则 %d/%.2f 等格式符会抛 TypeError
                record.args = tuple(
                    _SENSITIVE.sub("***REDACTED***", a) if isinstance(a, str) else a
                    for a in record.args
                )
        except Exception:
            pass
        return True


def setup_logging(level: str = "INFO", log_dir: str = "./data/logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = RotatingFileHandler(
        Path(log_dir) / "trader.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    for h in (handler, console):
        h.addFilter(RedactFilter())
        root.addHandler(h)
    root.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)