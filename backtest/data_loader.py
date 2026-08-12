"""回测数据加载：CSV 或交易所历史K线（自定义周期）。"""
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import settings

log = logging.getLogger(__name__)

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_csv(path: str) -> pd.DataFrame:
    """加载 CSV：列名支持 timestamp/open/high/low/close/volume 或 时间/开盘/最高/最低/收盘/成交量。"""
    df = pd.read_csv(path)
    rename = {"时间": "timestamp", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    df = df.rename(columns=rename)
    for col in _COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少列: {col}（需要 {_COLUMNS}）")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**6
    return _to_df(df[_COLUMNS].values.tolist())


async def load_from_exchange(exchange_id: str, symbol: str, timeframe: str = "1h",
                             since: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
    """通过交易所公开 REST 接口拉取历史K线（自动注入本地代理）。"""
    import ccxt.async_support as ccxt
    proxy = settings.resolved_proxy or None
    cfg: dict = {"enableRateLimit": True}
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    ex = getattr(ccxt, exchange_id)(cfg)
    try:
        await ex.load_markets()
        since_ms = int(since.timestamp() * 1000) if since else None
        rows = await ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        log.info("[回测] 从 %s 拉取 %s %s 共 %d 根K线", exchange_id, symbol, timeframe, len(rows))
        return _to_df(rows)
    finally:
        await ex.close()


def generate_demo(n: int = 2000, start_price: float = 100.0, timeframe: str = "1h",
                  vol: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """生成合成K线（几何随机游走），便于离线演示回测。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    step_s = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 3600)
    start_ts = int(datetime.now(timezone.utc).timestamp()) // step_s * step_s - n * step_s
    rets = rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    ohlc = np.column_stack([
        np.roll(close, 1), close * (1 + rng.normal(0, vol / 3, n)),
        close * (1 + rng.normal(0, vol / 3, n)), close,
    ])
    rows = [[start_ts + i * step_s * 1000, ohlc[i, 0], ohlc[i, 1], ohlc[i, 2], ohlc[i, 3],
             float(abs(rng.normal(100, 20)))] for i in range(n)]
    return _to_df(rows)