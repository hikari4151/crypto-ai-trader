"""AI 配置持久化回归（历史 bug：界面默认显示 gpt 而不是已保存的自定义模型）。

两个层面的锁：
1. **后端契约**：`/api/ai/settings` 的读取逻辑必须返回**数据库 KV 里保存的值**
   （provider/base_url/model），而不是 config.settings 的默认值
   （openai / api.openai.com / gpt-4o-mini）。
2. **前端守卫**：index.html 不得再出现"模型不在服务商列表就替换成列表第一个"的
   破坏性写法；且必须保留加载阶段不回调默认模型（aiSettingsLoaded）的守卫。

直接调用端点协程（不经 ASGI/TestClient）：aiosqlite 连接与事件循环绑定，
TestClient 在别的线程跑 loop 会超时。
"""
from pathlib import Path

import pytest

from core.database import Database

REPO = Path(__file__).resolve().parent.parent


class _FakeAIClient:
    def invalidate_cache(self):
        pass


class _FakeEngine:
    ai_client = _FakeAIClient()


@pytest.mark.asyncio
async def test_ai_settings_returns_saved_kv_not_defaults(tmp_path):
    """保存自定义提供商后，读取必须回显保存值（而不是 gpt-4o-mini 默认值）。"""
    from web.api.ai import get_ai_settings

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ai.db'}")
    await db.init()
    try:
        await db.kv_set("ai_provider", "custom")
        await db.kv_set("ai_base_url", "https://api.agnes-ai.cn/v1")
        await db.kv_set("ai_model", "agnes-2.5-flash")
        d = await get_ai_settings(db=db)
        assert d["provider"] == "custom"
        assert d["base_url"] == "https://api.agnes-ai.cn/v1"
        assert d["model"] == "agnes-2.5-flash", \
            f"必须回显保存的自定义模型，实际 {d['model']!r}（默认 gpt-4o-mini 是历史 bug）"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ai_settings_roundtrip_save_then_read(tmp_path):
    """保存 → 读回：自定义模型不得被默认值吞掉。"""
    from web.api.ai import AISettingsIn, get_ai_settings, save_ai_settings

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ai.db'}")
    await db.init()
    try:
        res = await save_ai_settings(
            body=AISettingsIn(provider="custom", base_url="https://api.agnes-ai.cn/v1",
                              model="agnes-3.0-flash", api_key="sk-test-1234567890"),
            db=db, engine=_FakeEngine())
        assert res["ok"] is True
        d = await get_ai_settings(db=db)
        assert d["model"] == "agnes-3.0-flash"
        assert d["base_url"] == "https://api.agnes-ai.cn/v1"
        assert d["has_api_key"] is True
    finally:
        await db.close()


def test_frontend_no_longer_clobbers_saved_model():
    """前端不得再把"模型不在 /models 列表"当作替换理由（会吃掉自定义模型名）。"""
    html = (REPO / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert "!d.models.includes(ai.model)) ai.model = d.models[0]" not in html, \
        "破坏性回填仍在：会把已保存的自定义模型替换成服务商列表第一个"
    assert "aiSettingsLoaded" in html, "缺少加载阶段守卫（provider watch 会覆盖已保存模型）"
    assert "只在模型为空时自动选中第一个" in html, "缺少非破坏性回填逻辑"