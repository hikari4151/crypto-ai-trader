"""P0-3 日志轮转回归：uvicorn server.log 走 RotatingFileHandler。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_uvicorn_log_config_structure():
    """uvicorn_log_config 必须用 RotatingFileHandler 写 server.log。"""
    from core.logging_config import uvicorn_log_config

    c = uvicorn_log_config("./data/logs")
    assert c["disable_existing_loggers"] is False, "不得重置应用 root handlers"
    h = c["handlers"]["server_file"]
    assert h["class"] == "core.logging_config.ResilientRotatingFileHandler", \
        "server.log 须用容错轮转 handler（多实例并发持锁时静默降级，不刷 stderr）"
    assert h["maxBytes"] == 5 * 1024 * 1024 and h["backupCount"] == 5
    assert "console" in c["handlers"]
    assert set(c["loggers"]) == {"uvicorn", "uvicorn.error", "uvicorn.access"}
    # access 不进 console（避免与 server_file 双写/刷屏）；error 双通道保留可见性
    assert c["loggers"]["uvicorn.access"]["handlers"] == ["server_file"]
    assert "console" in c["loggers"]["uvicorn.error"]["handlers"]


def test_uvicorn_log_config_does_not_touch_root(tmp_path, monkeypatch):
    """应用 setup_logging 后，uvicorn 配置不得清掉 root 的 trader.log handler。"""
    from core.logging_config import setup_logging, uvicorn_log_config
    import logging.config

    setup_logging(log_dir=str(tmp_path / "logs"))
    root_before = len(logging.getLogger().handlers)
    logging.config.dictConfig(uvicorn_log_config(str(tmp_path / "logs")))
    root_after = len(logging.getLogger().handlers)
    assert root_after >= root_before, "dictConfig 不得移除应用 root handlers"
    assert any(getattr(h, "_trader_handler", False) for h in logging.getLogger().handlers)


def test_run_py_passes_log_config():
    """run.py 的 uvicorn.run 必须传 log_config（防回退到 stdout 直写）。"""
    src = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "log_config=uvicorn_log_config" in src, "run.py 未接入 server.log 轮转配置"


def test_rotation_survives_locked_log_file(tmp_path, monkeypatch, capsys):
    """多实例并发持锁日志文件时，轮转失败必须静默降级（继续写，不刷 stderr）。

    2026-09-28 真实启动实测：第二个实例起服务时 trader.log 被并行实例持锁，
    doRollover 抛 PermissionError → logging 回退 handleError 打印整段 traceback
    到 stderr，污染启动日志（用户无法分辨是启动失败还是仅日志轮转问题）。"""
    from core.logging_config import setup_logging
    import logging
    from logging.handlers import RotatingFileHandler

    setup_logging(log_dir=str(tmp_path / "logs"))
    trader = next(h for h in logging.getLogger().handlers
                  if getattr(h, "_trader_handler", False)
                  and isinstance(h, RotatingFileHandler))

    # 模拟另一实例持锁导致轮转失败
    monkeypatch.setattr(trader, "doRollover", lambda: (_ for _ in ()).throw(
        PermissionError("[WinError 32] 另一个程序正在使用此文件")))

    log = logging.getLogger("rotation_lock_test")
    log.info("concurrent instance writes after rotation failure")
    err = capsys.readouterr().err
    # console handler 本就写 stderr（正常日志文本会出现在那）；真正的病征是
    # logging 的 handleError 把轮转 traceback 刷到 stderr。断言只盯 traceback 字样。
    for bad in ("Traceback (most recent call last)", "WinError 32", "doRollover"):
        assert bad not in err, f"轮转失败不得刷 stderr（出现: {bad}）"
    # 但日志本身仍应通过 console 通道可见（不静默吞掉应用日志）
    assert "concurrent instance writes" in err, "应用日志不得被容错逻辑吞掉"
    # 日志仍写入文件（降级为不轮转，继续追加）
    text = (tmp_path / "logs" / "trader.log").read_text(encoding="utf-8")
    assert "concurrent instance writes" in text
