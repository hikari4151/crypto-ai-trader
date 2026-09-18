"""GAP-07④ 回归测试：交易事件日志分级（trading.log vs trader.log / HTTP 访问日志）。

要求：
- engine.* 与 core.bus 的交易事件写入 data/logs/trading.log；
- HTTP 访问日志（uvicorn.access 等）与普通 API 日志**不**进 trading.log；
- 全部日志仍进 trader.log（不丢信息）。
"""
import logging

import pytest

from core.logging_config import setup_logging


@pytest.fixture
def log_dir(tmp_path):
    """隔离日志目录；测试后移除本次安装的 handler（防污染其他测试）。"""
    root = logging.getLogger()
    saved = list(root.handlers)
    level = root.level
    setup_logging("INFO", str(tmp_path))
    try:
        yield tmp_path
    finally:
        for h in list(root.handlers):
            if getattr(h, "_trader_handler", False):
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
        for h in saved:
            root.addHandler(h)
        root.setLevel(level)


def _flush():
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass


def test_trading_events_isolated_to_trading_log(log_dir):
    logging.getLogger("engine.order_manager").warning("下单事件 ORDER_PLACED")
    logging.getLogger("engine.risk").warning("风控熔断 RISK_BLOCKED")
    logging.getLogger("core.bus").error("事件处理失败 BUS_FAIL")
    logging.getLogger("uvicorn.access").info("GET /api/liveness 200")
    logging.getLogger("web.api.trading").info("普通 API 日志")
    _flush()

    trading = (log_dir / "trading.log").read_text(encoding="utf-8")
    trader = (log_dir / "trader.log").read_text(encoding="utf-8")

    assert "ORDER_PLACED" in trading
    assert "RISK_BLOCKED" in trading
    assert "BUS_FAIL" in trading
    assert "GET /api/liveness" not in trading, "HTTP 访问日志不得进 trading.log"
    assert "普通 API 日志" not in trading, "普通 API 日志不得进 trading.log"

    # trader.log 仍保留全量（不丢信息）
    for token in ("ORDER_PLACED", "RISK_BLOCKED", "BUS_FAIL",
                  "GET /api/liveness", "普通 API 日志"):
        assert token in trader, f"trader.log 应保留 {token}"


def test_setup_logging_idempotent_no_duplicate_handlers(log_dir):
    """重复 setup_logging 不得叠加 handler（旧行为曾让每条日志落两遍）。"""
    before = [h for h in logging.getLogger().handlers if getattr(h, "_trader_handler", False)]
    setup_logging("INFO", str(log_dir))
    after = [h for h in logging.getLogger().handlers if getattr(h, "_trader_handler", False)]
    assert len(before) == len(after) == 3, f"应为 trader/console/trading 三个 handler，实际 {len(after)}"