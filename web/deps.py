"""FastAPI 依赖注入：共享数据库与引擎单例。"""
from pathlib import Path

from fastapi import Request

from config.settings import settings
from core.database import Database
from engine.trading_engine import TradingEngine


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_engine(request: Request) -> TradingEngine:
    return request.app.state.engine


def resolve_data_path(path: str, subdir: str = "") -> Path:
    """校验并返回 data/ 目录下的安全路径，防止任意文件读取/路径遍历。

    - 相对路径基于 data/（或 data/<subdir>）解析
    - 绝对路径必须位于 data/ 目录内，否则拒绝
    - 抛 ValueError 表示非法路径
    """
    base = settings.data_dir / subdir if subdir else settings.data_dir
    base = base.resolve()
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    p = p.resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"路径必须在 data 目录内: {path}")
    return p