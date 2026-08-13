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