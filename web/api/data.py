"""数据管理 API：本地K线库查询 + 币安官方历史数据批量下载。

- GET  /api/data/series          本地已有K线序列清单（交易所/品种/周期/覆盖范围）
- POST /api/data/download        后台启动月度 zip 批量下载（data.binance.vision）
- GET  /api/data/download/status 下载进度查询
- DELETE /api/data/series        删除指定序列的本地数据
"""
import asyncio
import logging
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backtest import kline_store
from backtest.binance_history import ALLOWED_TF

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data"])

# 下载任务状态（进程内单任务：同一时间只允许一个下载，避免重复打源）
_DL_LOCK = threading.Lock()
_DL_STATUS: dict = {"running": False, "symbol": "", "timeframe": "",
                    "done": 0, "total": 0, "current_month": "",
                    "saved_rows": 0, "errors": [], "finished_at": None,
                    "last_message": ""}


@router.get("/series")
async def list_series():
    # 同步 SQLite 查询放线程池，避免阻塞事件循环
    return {"series": await asyncio.to_thread(kline_store.all_series)}


class DownloadRequest(BaseModel):
    symbol: str = Field(..., description="交易对，如 BTC/USDT")
    timeframe: str = Field("1h", description="K线周期")
    start_year: int = Field(..., ge=2017, le=2100)
    start_month: int = Field(..., ge=1, le=12)
    end_year: int = Field(..., ge=2017, le=2100)
    end_month: int = Field(..., ge=1, le=12)


@router.post("/download")
async def start_download(body: DownloadRequest):
    if (body.start_year, body.start_month) > (body.end_year, body.end_month):
        raise HTTPException(status_code=400, detail="起始月份不能晚于结束月份")
    if body.timeframe not in ALLOWED_TF:
        raise HTTPException(status_code=400, detail=f"不支持的周期: {body.timeframe}")
    with _DL_LOCK:
        if _DL_STATUS["running"]:
            raise HTTPException(status_code=409, detail="已有下载任务进行中，请等待完成")
        _DL_STATUS.update({"running": True, "symbol": body.symbol, "timeframe": body.timeframe,
                           "done": 0, "total": 0, "current_month": "", "saved_rows": 0,
                           "errors": [], "finished_at": None, "last_message": ""})

    async def _task():
        from backtest.binance_history import download_months
        try:
            result = await download_months(
                body.symbol, body.timeframe,
                (body.start_year, body.start_month), (body.end_year, body.end_month),
                on_progress=lambda p: _DL_STATUS.update(
                    {"done": p["done"], "total": p["total"],
                     "current_month": p["month"], "saved_rows": p["saved"]}))
            _DL_STATUS["last_message"] = (
                f"完成：下载 {result['downloaded_months']} 个月，"
                f"入库 {result['saved_rows']} 根K线，跳过 {result['skipped']}，"
                f"失败 {len(result['errors'])}")
            if result["errors"]:
                _DL_STATUS["errors"] = result["errors"][:20]
            log.info("[data] %s", _DL_STATUS["last_message"])
        except Exception as e:  # noqa: BLE001
            _DL_STATUS["last_message"] = f"下载失败: {e}"
            log.exception("[data] 历史数据下载失败")
        finally:
            _DL_STATUS["running"] = False
            _DL_STATUS["finished_at"] = True

    asyncio.get_event_loop().create_task(_task())
    return {"ok": True, "message": "下载已启动"}


@router.get("/download/status")
async def download_status():
    return dict(_DL_STATUS)


class DeleteRequest(BaseModel):
    exchange: str = "binance"
    symbol: str
    timeframe: str


@router.delete("/series")
async def delete_series(body: DeleteRequest):
    from backtest.data_loader import timeframe_seconds
    cov = await asyncio.to_thread(kline_store.coverage, body.exchange, body.symbol, body.timeframe)
    if not cov["count"]:
        raise HTTPException(status_code=404, detail="本地无该序列数据")
    # delete 前把 WAL 落盘并截断（TRUNCATE），保证删除作用于完整数据视图
    await asyncio.to_thread(kline_store.checkpoint)
    n = await asyncio.to_thread(
        kline_store.delete_range, body.exchange, body.symbol, body.timeframe,
        0, cov["max_ts"] + timeframe_seconds(body.timeframe) * 2000)
    return {"ok": True, "deleted": n}
