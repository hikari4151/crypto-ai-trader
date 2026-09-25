"""P0-3 日志轮转回归：uvicorn server.log 走 RotatingFileHandler。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_uvicorn_log_config_structure():
    """uvicorn_log_config 必须用 RotatingFileHandler 写 server.log。"""
    from core.logging_config import uvicorn_log_config

    c = uvicorn_log_config("./data/logs")
    assert c["disable_existing_loggers"] is False, "不得重置应用 root handlers"
    h = c["handlers"]["server_file"]
    assert h["class"] == "logging.handlers.RotatingFileHandler"
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
