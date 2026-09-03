"""K线本地持久化：SQLite 存储，供回测/DRL训练/因子分析本地优先读取。

设计：
- 独立 SQLite 文件 data/klines.db（与 trader.db 分离，高频写入互不干扰）
- 复合主键 (exchange, symbol, timeframe, ts) + WITHOUT ROWID，天然去重
- INSERT OR REPLACE：未收盘K线重复拉取时按时间戳覆盖
- 线程局部连接：多线程安全；大量写入可走 asyncio.to_thread
"""
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import ROOT

log = logging.getLogger(__name__)

DB_PATH = ROOT / "data" / "klines.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS klines (
  exchange  TEXT NOT NULL,
  symbol    TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  ts        INTEGER NOT NULL,
  open      REAL NOT NULL,
  high      REAL NOT NULL,
  low       REAL NOT NULL,
  close     REAL NOT NULL,
  volume    REAL NOT NULL,
  PRIMARY KEY (exchange, symbol, timeframe, ts)
) WITHOUT ROWID
"""

_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(_SCHEMA)
        conn.commit()
        _local.conn = conn
    return conn


def save_rows(exchange: str, symbol: str, timeframe: str, rows: list) -> int:
    """写入K线（覆盖同时间戳），rows=[[ts,o,h,l,c,v],...]，返回写入条数。"""
    if not rows:
        return 0
    conn = _conn()
    data = [(exchange, symbol, timeframe, int(r[0]), float(r[1]), float(r[2]),
             float(r[3]), float(r[4]), float(r[5])) for r in rows]
    conn.executemany(
        "INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?,?,?,?)", data)
    conn.commit()
    return len(data)


def load_rows(exchange: str, symbol: str, timeframe: str,
              limit: int = 0, start_ms: Optional[int] = None,
              end_ms: Optional[int] = None) -> list:
    """读取K线（升序）。limit>0 取最新 limit 根。"""
    sql = "SELECT ts,open,high,low,close,volume FROM klines WHERE exchange=? AND symbol=? AND timeframe=?"
    args: list = [exchange, symbol, timeframe]
    if start_ms is not None:
        sql += " AND ts>=?"
        args.append(int(start_ms))
    if end_ms is not None:
        sql += " AND ts<=?"
        args.append(int(end_ms))
    # 取末尾 n 根：必须先 DESC 再 LIMIT（把 DESC 拼在子查询括号外会被 SQLite 静默
    # 忽略，等价于 ASC LIMIT n —— 取回的是最早 n 根，且末尾时间戳随数据追加不动）
    tail = bool(limit and limit > 0)
    if tail:
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
    else:
        sql += " ORDER BY ts"
    cur = _conn().execute(sql, args)
    rows = [list(r) for r in cur.fetchall()]
    if tail:
        rows.reverse()  # 调用方（指标/回测/增量门）一律假定升序
    return rows


def coverage(exchange: str, symbol: str, timeframe: str) -> dict:
    """本地覆盖情况：{count, min_ts, max_ts}（无数据 count=0）。"""
    cur = _conn().execute(
        "SELECT COUNT(*), MIN(ts), MAX(ts) FROM klines WHERE exchange=? AND symbol=? AND timeframe=?",
        (exchange, symbol, timeframe))
    n, lo, hi = cur.fetchone()
    return {"count": int(n or 0), "min_ts": lo, "max_ts": hi}


def delete_range(exchange: str, symbol: str, timeframe: str,
                 start_ms: int, end_ms: int) -> int:
    """删除区间内K线（历史重下时清旧数据），返回删除条数。"""
    conn = _conn()
    cur = conn.execute(
        "DELETE FROM klines WHERE exchange=? AND symbol=? AND timeframe=? AND ts>=? AND ts<=?",
        (exchange, symbol, timeframe, int(start_ms), int(end_ms)))
    conn.commit()
    return cur.rowcount


def load_df(exchange: str, symbol: str, timeframe: str,
            limit: int = 0, start_ms: Optional[int] = None,
            end_ms: Optional[int] = None) -> pd.DataFrame:
    """读取为 DataFrame（与 data_loader._to_df 同构：DatetimeIndex + float 列）。"""
    rows = load_rows(exchange, symbol, timeframe, limit=limit,
                     start_ms=start_ms, end_ms=end_ms)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


def all_series() -> list[dict]:
    """本地已有的全部K线序列清单（前端数据管理展示用）。"""
    cur = _conn().execute(
        "SELECT exchange, symbol, timeframe, COUNT(*), MIN(ts), MAX(ts) "
        "FROM klines GROUP BY exchange, symbol, timeframe ORDER BY symbol")
    out = []
    for ex, sym, tf, n, lo, hi in cur.fetchall():
        out.append({"exchange": ex, "symbol": sym, "timeframe": tf,
                    "count": int(n),
                    "start": pd.to_datetime(lo, unit="ms", utc=True).isoformat() if lo else None,
                    "end": pd.to_datetime(hi, unit="ms", utc=True).isoformat() if hi else None})
    return out
