"""回测数据加载：CSV 或交易所历史K线（自定义周期）。

数据获取策略：
- binance：优先走币安公开数据域 data-api.binance.vision（httpx 直连，
  大陆可直连、免代理、免 ccxt；此前走 ccxt binance 主域经常连不上，
  持续进化引擎因此反复告警）。失败再回退 ccxt。
- 其他交易所：ccxt（自动注入本地代理）。
- 成功拉取自动写入本地K线库（kline_store）；load_klines_cached 本地优先。
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import settings

log = logging.getLogger(__name__)


_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# 币安公开数据域（仅行情只读，大陆直连，与 web/api/market.py 同源）
_BINANCE_PUBLIC = "https://data-api.binance.vision/api/v3"
_BINANCE_TF = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
_TF_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
               "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
               "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000}


def timeframe_seconds(timeframe: str) -> int:
    """周期对应的秒数（未知周期按 1h 处理）。"""
    return _TF_SECONDS.get(timeframe, 3600)


def _binance_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "").replace("/", "")


async def _fetch_binance_public(symbol: str, timeframe: str,
                                since_ms: Optional[int], limit: int,
                                end_ms: Optional[int] = None) -> list:
    """币安公开数据域拉取K线（httpx 直连，无代理），自动分页。

    - since_ms 给定：从该时间正向翻页拉到最新（limit 根为止）
    - since_ms 为空且 limit>1000：先取最新1000根，再用 endTime 向前翻页补足
    返回升序 [[ts,o,h,l,c,v],...]；与 web/api/market.py 的 .vision 端点同源。
    """
    import httpx
    url = f"{_BINANCE_PUBLIC}/klines"
    sym = _binance_symbol(symbol)
    if timeframe not in _BINANCE_TF:
        raise ValueError(f"币安不支持周期: {timeframe}")
    out: list = []
    cur_since = since_ms
    cur_end = end_ms
    fetch_limit = min(limit, 1000)
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
        while len(out) < limit:
            params: dict = {"symbol": sym, "interval": timeframe, "limit": fetch_limit}
            if cur_since is not None:
                params["startTime"] = int(cur_since)
            if cur_end is not None:
                params["endTime"] = int(cur_end)
            last_exc = None
            rows = None
            for attempt in range(3):
                try:
                    r = await client.get(url, params=params)
                    if r.status_code == 429 or r.status_code >= 500:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    r.raise_for_status()
                    rows = r.json()
                    break
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    await asyncio.sleep(0.5 * (2 ** attempt))
            if rows is None:
                raise RuntimeError(f"币安公开数据域请求失败: {last_exc}")
            if not rows:
                break
            # [ts, open, high, low, close, volume, close_ts, ...] → 取前6列
            new_rows = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]),
                         float(c[4]), float(c[5])] for c in rows]
            if cur_since is not None:
                # 正向分页：跳过与已有尾部重叠的时间戳
                new_rows = [r for r in new_rows if not out or r[0] > out[-1][0]]
                if not new_rows:
                    break
                out.extend(new_rows)
                cur_since = rows[-1][0] + 1
                if len(rows) < fetch_limit:
                    break
            else:
                # 向前翻页：新页整体在已有序列之前，去重后前置
                new_rows = [r for r in new_rows if not out or r[0] < out[0][0]]
                if not new_rows:
                    break
                out = new_rows + out
                cur_end = new_rows[0][0] - 1
                if len(rows) < fetch_limit:
                    break
    # since 给定 → 从 since 起的前 limit 根；否则 → 最新 limit 根
    return out[:limit] if since_ms is not None else out[-limit:]


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


async def _fetch_ccxt(exchange_id: str, symbol: str, timeframe: str,
                      since_ms: Optional[int], limit: int, max_candles: int) -> list:
    """ccxt 交易所拉取（自动注入本地代理），支持分页。"""
    import ccxt.async_support as ccxt
    proxy = settings.resolved_proxy or None
    cfg: dict = {"enableRateLimit": True}
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    ex_cls = getattr(ccxt, exchange_id, None)
    if ex_cls is None:
        raise ValueError(f"不支持的交易所: {exchange_id}")
    ex = ex_cls(cfg)
    try:
        await ex.load_markets()

        # 向后兼容：max_candles=0 或 max_candles <= limit 时不分页
        if max_candles <= 0 or max_candles <= limit:
            rows = await ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            log.info("[回测] 从 %s 拉取 %s %s 共 %d 根K线", exchange_id, symbol, timeframe, len(rows))
            return rows

        # 分页循环拉取
        all_rows = []
        fetch_limit = min(limit, max_candles)
        while len(all_rows) < max_candles:
            rows = await ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=fetch_limit)
            if not rows:
                break
            # 去重：跳过与上一页最后一条时间戳重叠的K线
            new_rows = [r for r in rows if not all_rows or r[0] > all_rows[-1][0]]
            if not new_rows:
                break
            all_rows.extend(new_rows)
            # 下一页从最后一根K线的时间戳 +1ms 开始
            since_ms = rows[-1][0] + 1
            if len(rows) < fetch_limit:
                break  # 已拉取到最新数据
            if len(all_rows) >= max_candles:
                all_rows = all_rows[:max_candles]
                break
        log.info("[回测] 从 %s 分页拉取 %s %s 共 %d 根K线", exchange_id, symbol, timeframe, len(all_rows))
        return all_rows
    finally:
        await ex.close()


async def load_from_exchange(exchange_id: str, symbol: str, timeframe: str = "1h",
                             since: Optional[datetime] = None, limit: int = 1000,
                             max_candles: int = 0) -> pd.DataFrame:
    """拉取交易所历史K线并写入本地K线库。

    binance 优先走公开数据域（直连免代理）；失败回退 ccxt。
    """
    since_ms = int(since.timestamp() * 1000) if since else None
    rows: Optional[list] = None

    if exchange_id == "binance":
        try:
            rows = await _fetch_binance_public(symbol, timeframe, since_ms,
                                               limit=max_candles or limit)
            log.info("[回测] 币安公开数据域拉取 %s %s 共 %d 根K线", symbol, timeframe, len(rows))
        except Exception as e:  # noqa: BLE001
            log.warning("[回测] 币安公开数据域失败(%s)，回退 ccxt", e)
            rows = None
    if rows is None:
        rows = await _fetch_ccxt(exchange_id, symbol, timeframe, since_ms, limit, max_candles)

    # 写入本地K线库（失败不影响返回）
    try:
        from . import kline_store
        await asyncio.to_thread(kline_store.save_rows, exchange_id, symbol, timeframe, rows)
    except Exception as e:  # noqa: BLE001
        log.warning("[回测] K线入库失败（不影响返回）: %s", e)
    return _to_df(rows)


async def load_klines_cached(exchange_id: str, symbol: str, timeframe: str = "1h",
                             limit: int = 1000, max_candles: int = 0) -> pd.DataFrame:
    """本地优先K线加载：本地已覆盖且新鲜直接返回；否则增量补尾部 + 历史回填。

    新鲜度：最新一根K线开始时间距今 < max(1.5 周期, 5 分钟) 视为新鲜
    （未收盘K线重复拉取时按时间戳覆盖，不会脏读）。
    数量不足时：binance 用 endTime 向前翻页回填历史；其他交易所返回现有。
    """
    from . import kline_store
    n_want = int(max_candles or limit)
    now_ms = time.time() * 1000
    period_ms = timeframe_seconds(timeframe) * 1000
    fresh_window = max(int(1.5 * period_ms), 300_000)

    cov = await asyncio.to_thread(kline_store.coverage, exchange_id, symbol, timeframe)
    if not cov["count"]:
        # 本地为空：全量拉取（binance 公开域自动向后翻页补足 n_want 根）
        return await load_from_exchange(exchange_id, symbol, timeframe, limit=n_want)
    if cov["count"] >= n_want and cov["max_ts"] and now_ms - cov["max_ts"] < fresh_window:
        return await asyncio.to_thread(kline_store.load_df, exchange_id, symbol,
                                       timeframe, n_want)

    # 1) 增量补尾部（本地最新时间戳之后到 now）
    if cov["max_ts"]:
        since_dt = datetime.fromtimestamp((cov["max_ts"] + 1) / 1000, tz=timezone.utc)
        try:
            await load_from_exchange(exchange_id, symbol, timeframe,
                                     since=since_dt, limit=1000, max_candles=100_000)
        except Exception as e:  # noqa: BLE001
            log.warning("[klines] 增量拉取失败（用本地现有数据）: %s", e)
        cov = await asyncio.to_thread(kline_store.coverage, exchange_id, symbol, timeframe)

    # 2) 数量不足 → 历史回填（仅 binance 支持 endTime 向前翻页）
    if cov["count"] < n_want and exchange_id == "binance":
        need = n_want - cov["count"]
        end_ms = (cov["min_ts"] - 1) if cov["min_ts"] else None
        try:
            rows = await _fetch_binance_public(symbol, timeframe, None, need, end_ms=end_ms)
            await asyncio.to_thread(kline_store.save_rows, exchange_id, symbol, timeframe, rows)
        except Exception as e:  # noqa: BLE001
            log.warning("[klines] 历史回填失败: %s", e)

    return await asyncio.to_thread(kline_store.load_df, exchange_id, symbol,
                                   timeframe, n_want)


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


class OHLCVSanitizer:
    """轻量 OHLCV 数据清洗：异常检测 + 缺失值填充。

    P1-9 说明：默认 mode="mark"（标记不清洗）——只修复结构损坏
    （high < max(open,close) 等）与缺失值，对真实行情异常值只记录标记
    到 df.attrs["clean_flags"]，不做全样本中位数替换。旧实现用整段样本的
    中位数替换异常值：(1) 用未来数据改过去真实K线（前视），(2) 极端行情
    （如闪崩真实大幅波动）被抹平会系统性低估回撤与风险。需要清洗时
    显式传 mode="replace"（保留旧行为）。

    用法：
        sanitizer = OHLCVSanitizer(z_threshold=6.0)
        df = sanitizer.clean(df)              # 默认标记制
        df = sanitizer.clean(df, mode="replace")  # 中位数替换（旧行为）
    """

    def __init__(self, z_threshold: float = 6.0, ffill_limit: int = 3):
        self.z_threshold = z_threshold
        self.ffill_limit = ffill_limit

    def clean(self, df: pd.DataFrame, mode: str = "mark") -> pd.DataFrame:
        """清洗 OHLCV DataFrame，返回副本。

        mode="mark": 只修结构损坏 + 填充缺失，异常值仅标记不替换。
        mode="replace": 异常值用中位数替换（旧行为）。
        """
        result = df.copy()
        # 1) 缺失值前向填充（最多 ffill_limit 根）
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = result[col].ffill(limit=self.ffill_limit)
        # 2) 结构完整性修复（无论何种模式都必须做）：high 必须 >= max(open,close)、
        #    low 必须 <= min(open,close)、volume >= 0。这类损坏是数据源错误而非行情，
        #    修复它不构成前视。
        import numpy as np
        for col in ["open", "high", "low", "close"]:
            if col not in result.columns:
                continue
            arr = result[col].to_numpy(float).copy()
            arr = np.where(np.isfinite(arr), arr, np.nan)
            result[col] = arr
        flags: dict[str, int] = {}
        if "high" in result.columns and "open" in result.columns and "close" in result.columns:
            hc = np.maximum(result["open"].to_numpy(float), result["close"].to_numpy(float))
            bad_high = result["high"].to_numpy(float) < hc
            if bad_high.any():
                flags["high_repaired"] = int(bad_high.sum())
                result.loc[bad_high, "high"] = hc[bad_high]
        if "low" in result.columns and "open" in result.columns and "close" in result.columns:
            lc = np.minimum(result["open"].to_numpy(float), result["close"].to_numpy(float))
            bad_low = result["low"].to_numpy(float) > lc
            if bad_low.any():
                flags["low_repaired"] = int(bad_low.sum())
                result.loc[bad_low, "low"] = lc[bad_low]
        if "volume" in result.columns:
            bad_vol = result["volume"].to_numpy(float) < 0
            if bad_vol.any():
                flags["negative_volume"] = int(bad_vol.sum())
                result.loc[bad_vol, "volume"] = 0.0
        # 3) 异常值检测（mode="mark" 只标记，mode="replace" 中位数替换）
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in result.columns:
                continue
            series = result[col].to_numpy(float).copy()
            if len(series) < 10:
                continue
            median = np.median(series)
            mad = np.median(np.abs(series - median)) + 1e-12
            z = np.abs(series - median) / mad
            outliers = (z > self.z_threshold) & np.isfinite(series)
            if outliers.any():
                flags.setdefault(f"outliers_{col}", int(outliers.sum()))
                if mode == "replace":
                    series[outliers] = median
                    result[col] = series
        result.attrs["clean_flags"] = flags
        return result
