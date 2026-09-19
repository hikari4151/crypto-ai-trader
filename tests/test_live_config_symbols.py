"""GAP-06 多标的配置 API 回归：symbols 逗号字符串与 list 双形态保存 → 读回为 list。

后端 /api/live/config 规范化：前端可能发逗号分隔字符串（带空格）或 list，
保存后统一为 list；GET 回读 symbols 数组；引擎侧由 multi_symbol_engine 测试覆盖。
"""
import pytest

from core.database import Database


@pytest.mark.asyncio
async def test_live_config_symbols_string_normalized(tmp_path):
    from web.api.live import LiveConfigIn, get_live_config, save_live_config

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'live.db'}")
    await db.init()
    try:
        # 逗号分隔字符串（带空格）→ 保存为 list
        body = LiveConfigIn(exchange="binance", symbol="SOL/USDT", timeframe="1h",
                            start_cash=10000.0, symbols="SOL/USDT, ETH/USDT")
        res = await save_live_config(body=body, db=db)
        assert res["symbols"] == ["SOL/USDT", "ETH/USDT"], res
        d = await get_live_config(db=db)
        assert d["symbols"] == ["SOL/USDT", "ETH/USDT"], "GET 应回读标准化后的 list"
        assert d["symbol"] == "SOL/USDT" and d["timeframe"] == "1h"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_config_symbols_list_roundtrip(tmp_path):
    from web.api.live import LiveConfigIn, get_live_config, save_live_config

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'live2.db'}")
    await db.init()
    try:
        body = LiveConfigIn(exchange="binance", symbol="BTC/USDT", timeframe="1h",
                            start_cash=10000.0, symbols=["BTC/USDT", "ETH/USDT"])
        await save_live_config(body=body, db=db)
        d = await get_live_config(db=db)
        assert d["symbols"] == ["BTC/USDT", "ETH/USDT"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_config_symbols_empty_means_single(tmp_path):
    from web.api.live import LiveConfigIn, get_live_config, save_live_config

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'live3.db'}")
    await db.init()
    try:
        body = LiveConfigIn(exchange="binance", symbol="BTC/USDT", timeframe="1h",
                            start_cash=10000.0)  # symbols 缺省 = 单标的
        await save_live_config(body=body, db=db)
        d = await get_live_config(db=db)
        assert d["symbols"] == [], "空 = 单标的（引擎回退 symbol 字段）"
    finally:
        await db.close()