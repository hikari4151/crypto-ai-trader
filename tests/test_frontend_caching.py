"""前端缓存 hash 化回归：vendor immutable 头 + ops 版本字段。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_has_vendor_immutable_route():
    """main.py 必须定义 /static/vendor/{filename} 路由且带 immutable 缓存头。"""
    src = (ROOT / "web" / "main.py").read_text(encoding="utf-8")
    assert '/static/vendor/{filename}' in src, "缺 vendor 路由"
    assert "immutable" in src, "vendor 必须 immutable 长缓存"
    assert "max-age=31536000" in src, "vendor 缓存应为一年的 max-age"
    # 路由须在 app.mount("/static") 之前注册（优先级：先路由后 mount）
    idx_route = src.index('/static/vendor/{filename}')
    idx_mount = src.index('app.mount("/static"')
    assert idx_route < idx_mount, "vendor 路由必须先于 /static mount 注册"


def test_ops_frontend_version_field(tmp_path, monkeypatch):
    """_frontend_version 读取 build_manifest 的 sha256 前 8 位。"""
    from web.api.ops import _frontend_version

    m = tmp_path / "build_manifest.json"
    m.write_text(json.dumps({"sha256": "abcdef1234567890", "built_at": "2026-01-01T00:00:00Z"}),
                 encoding="utf-8")
    monkeypatch.setattr("web.api.ops._MANIFEST", m)
    r = _frontend_version()
    assert r["version"] == "abcdef12" and r["manifest_exists"] is True
    assert r["built_at"] == "2026-01-01T00:00:00Z"


def test_ops_frontend_version_missing(tmp_path, monkeypatch):
    """清单缺失/损坏 → 返回 manifest_exists=False，不抛异常。"""
    from web.api.ops import _frontend_version

    monkeypatch.setattr("web.api.ops._MANIFEST", tmp_path / "nope.json")
    r = _frontend_version()
    assert r["manifest_exists"] is False and r["version"] is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr("web.api.ops._MANIFEST", bad)
    r2 = _frontend_version()
    assert r2["manifest_exists"] is False


def test_ops_overview_includes_frontend(tmp_path, monkeypatch):
    """overview 响应含 frontend 字段（端到端注入）。"""
    import asyncio
    from web.api.ops import ops_overview
    from core.database import Database
    from core.bus import EventBus
    from engine.trading_engine import TradingEngine

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ops.db'}")
    asyncio.run(db.init())
    try:
        eng = TradingEngine(db, EventBus())
        eng.running = False
        m = tmp_path / "build_manifest.json"
        m.write_text(json.dumps({"sha256": "feedbeef" * 4}), encoding="utf-8")
        monkeypatch.setattr("web.api.ops._MANIFEST", m)
        out = asyncio.run(ops_overview(engine=eng, db=db))
        assert "frontend" in out and out["frontend"]["version"] == "feedbeef"
    finally:
        asyncio.run(db.close())
