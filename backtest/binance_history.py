"""币安官方公开历史数据下载（data.binance.vision 月度K线 zip）。

- 数据源：https://data.binance.vision/data/spot/monthly/klines/{SYM}/{tf}/{SYM}-{tf}-{YYYY}-{MM}.zip
- 官方口径、免费、无 API 限流（普通对象存储下载），单月 1h K线约几百 KB
- 2025-01 起部分文件 open_time 为微秒：>1e14 自动 /1000 归一到毫秒
- 下载后解压 CSV → 入 kline_store 本地库；404（月份未发布/币种不存在）跳过
- 月度文件只到上一个完整月；当月缺口由 REST 增量补齐（load_klines_cached）
"""
import asyncio
import csv
import io
import logging
import zipfile
from typing import Callable, Optional

import httpx


log = logging.getLogger(__name__)

_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# 允许的K线周期（与币安月度文件目录一致）
ALLOWED_TF = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}


def _norm_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "").replace("/", "")


def _month_iter(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _norm_ts(ts: int) -> int:
    """微秒时间戳（2025-01 起部分文件）归一到毫秒。"""
    return ts // 1000 if ts > 10**14 else ts


async def download_months(symbol: str, timeframe: str,
                          start: tuple[int, int], end: tuple[int, int],
                          proxy: Optional[str] = None,
                          on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """批量下载月度K线 zip 并入本地库。

    start/end: (year, month) 闭区间。返回 {downloaded_months, skipped, saved_rows, errors}。
    on_progress({month, done, total, saved}): 每个月完成后的进度回调。
    """
    from backtest import kline_store

    if timeframe not in ALLOWED_TF:
        raise ValueError(f"不支持的周期: {timeframe}（可选 {sorted(ALLOWED_TF)}）")
    sym = _norm_symbol(symbol)
    months = list(_month_iter(start, end))
    total = len(months)
    saved_rows = 0
    downloaded = 0
    skipped = 0
    errors: list[str] = []
    # 直连（data.binance.vision 大陆可直连）；显式代理仅在用户配置时使用
    proxy_url = proxy or None
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0),
                                 proxy=proxy_url, follow_redirects=True) as client:
        for i, (y, m) in enumerate(months):
            label = f"{y}-{m:02d}"
            url = f"{_BASE}/{sym}/{timeframe}/{sym}-{timeframe}-{label}.zip"
            try:
                r = await client.get(url)
                if r.status_code == 404:
                    skipped += 1
                    log.info("[history] %s %s %s 不存在（跳过）", symbol, timeframe, label)
                else:
                    r.raise_for_status()
                    rows = _parse_zip(r.content)
                    if rows:
                        await asyncio.to_thread(kline_store.save_rows, "binance", symbol,
                                                 timeframe, rows)
                        saved_rows += len(rows)
                        downloaded += 1
                        log.info("[history] %s %s %s 入库 %d 根", symbol, timeframe, label, len(rows))
                    else:
                        skipped += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}: {e}")
                log.warning("[history] %s %s %s 失败: %s", symbol, timeframe, label, e)
            if on_progress:
                on_progress({"month": label, "done": i + 1, "total": total, "saved": saved_rows})
    return {"downloaded_months": downloaded, "skipped": skipped,
            "saved_rows": saved_rows, "errors": errors}


def _parse_zip(content: bytes) -> list:
    """解压月度K线 zip → [[ts,o,h,l,c,v],...]（升序去重）。"""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            rows: list = []
            for rec in reader:
                if not rec or len(rec) < 6:
                    continue
                try:
                    ts = _norm_ts(int(rec[0]))
                    rows.append([ts, float(rec[1]), float(rec[2]), float(rec[3]),
                                 float(rec[4]), float(rec[5])])
                except (ValueError, IndexError):
                    continue  # 跳过表头/坏行
    rows.sort(key=lambda r: r[0])
    # 去重（相邻同 ts 保留后者）
    return [r for i, r in enumerate(rows) if i == 0 or r[0] != rows[i - 1][0]]
