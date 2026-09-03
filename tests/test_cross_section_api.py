"""横截面因子 API 端点测试：demo/local 数据源 + 参数校验。

local 源用真实 kline_store（data/klines.db）——若本地无 BTC/ETH/SOL 1h 数据
则跳过（CI/新环境无数据时不应误报）。
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api():
    from fastapi import FastAPI
    from web.api.factors import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(api, **kw):
    body = {"data_source": "demo", "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "timeframe": "1h", "lookback": 24, "horizon": 4, **kw}
    return api.post("/api/factors/cross-section", json=body)


def test_demo_source_full_report(api):
    r = _post(api)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True and d["n_symbols"] == 3
    assert len(d["factors"]) == 5
    # demo 为独立同分布合成序列 → 必然带"数字仅演示"的诚实性提示
    assert d["warning"] is not None and "演示数据" in d["warning"]
    # demo 模式去掉快照与曲线（时间戳无意义）
    assert "spread_curves" not in d
    assert all("latest" not in f for f in d["factors"])
    # IC 字段齐全且有界
    for f in d["factors"]:
        assert -1.0 <= f["ic_mean"] <= 1.0
        assert f["n_ic_obs"] > 0 and "total_ret" in f["long_short"]


def test_too_few_symbols_rejected(api):
    r = _post(api, symbols=["BTC/USDT", "ETH/USDT"])
    assert r.status_code == 400 and "3-12" in r.json()["detail"]


def test_duplicate_symbols_rejected(api):
    r = _post(api, symbols=["BTC/USDT", "BTC/USDT", "ETH/USDT"])
    assert r.status_code == 400 and "重复" in r.json()["detail"]


def test_too_many_symbols_rejected(api):
    r = _post(api, symbols=[f"S{i}/USDT" for i in range(13)])
    assert r.status_code == 400


def test_bad_lookback_rejected(api):
    r = _post(api, lookback=3)
    assert r.status_code == 422  # pydantic ge=5


def test_unknown_source_rejected(api):
    r = _post(api, data_source="csv")
    assert r.status_code == 400


def test_local_source_with_real_store(api):
    """local 源读真实本地库（BTC/ETH/SOL 1h 已在库时）。"""
    from backtest import kline_store
    have = {s["symbol"] for s in kline_store.all_series()
            if s["timeframe"] == "1h" and s["count"] >= 30}
    symbols = [s for s in ["BTC/USDT", "ETH/USDT", "SOL/USDT"] if s in have]
    if len(symbols) < 3:
        pytest.skip(f"本地K线库品种不足（{sorted(have)}），跳过 local 源测试")
    r = _post(api, data_source="local", symbols=symbols, horizon=1)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_symbols"] == 3 and d["data_source"] == "local"
    # local 模式含快照与曲线
    assert "spread_curves" in d
    mom = next(f for f in d["factors"] if f["key"] == "xs_mom")
    assert all("rank" in s and "value" in s for s in mom["latest"])


def test_local_source_missing_symbol_hint(api):
    """本地缺品种 → 400 且提示去数据管理页下载。"""
    r = _post(api, data_source="local",
              symbols=["BTC/USDT", "ETH/USDT", "NOSUCH/USDT"])
    assert r.status_code == 400
    assert "NOSUCH/USDT" in r.json()["detail"]
    assert "数据管理" in r.json()["detail"]
