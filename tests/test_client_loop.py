"""AIClient 按 loop 的 HTTP 客户端生命周期测试（模块 C P1-4）。

覆盖 2026-08-14 修复：
- `_http_clients` 从 dict[id(loop)] 改为 weakref.WeakKeyDictionary（loop 键），
  死循环的条目随 GC 自动淘汰（原 id 键永不淘汰 → 每个后台任务泄漏一个 client）
- 新增 `close_loop(loop)` 交付接口（B/D 模块 worker finally 调用）：
  关闭指定 loop 的 client 并移除条目，幂等
- `close()` 全量关闭语义不变；`_client_for_loop` 同 loop 复用不变

注：asyncio.run 结束后其内部 loop 强引用立即释放，条目可能同步被弱引用
回调清除——因此需要持有一个 loop 强引用的测试用 holder 保证断言确定性。
"""
import asyncio
import gc
import weakref

from ai.client import AIClient


class FakeDB:
    """内存 KV，模拟 Database 的 kv 接口（无需真 SQLite）。"""

    def __init__(self):
        self.kv = {}

    async def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    async def kv_get_secret(self, key, default=None):
        return self.kv.get(key, default)


def make_client() -> AIClient:
    return AIClient(FakeDB())


# ============ 数据结构 ============

def test_http_clients_is_weakkeydict():
    """键从 id(loop) 改为 loop 弱引用对象（架构契约：WeakKeyDictionary）。"""
    client = make_client()
    assert isinstance(client._http_clients, weakref.WeakKeyDictionary)


def test_same_loop_reuses_client():
    """同一 loop 两次取客户端必须拿到同一实例（连接复用回归红线）。"""
    client = make_client()
    holder = {}

    async def run():
        holder["loop"] = asyncio.get_running_loop()
        c1 = client._client_for_loop()
        c2 = client._client_for_loop()
        return c1, c2

    c1, c2 = asyncio.run(run())
    assert c1 is c2
    # loop 强引用存活期间条目存在且键为 loop 本身（非 id）
    assert len(client._http_clients) == 1
    assert next(iter(client._http_clients)) is holder["loop"]


def test_different_loops_get_different_clients():
    """不同 loop 各自独立的 client（httpx client 绑定创建时的事件循环）。"""
    client = make_client()

    async def run():
        return client._client_for_loop()

    c1 = asyncio.run(run())
    c2 = asyncio.run(run())
    assert c1 is not c2


# ============ close_loop（交付给 B/D 的接口） ============

def test_close_loop_closes_and_removes_entry():
    """close_loop：关闭该 loop 的 client 并移除条目。"""
    client = make_client()

    async def run():
        loop = asyncio.get_running_loop()
        c = client._client_for_loop()
        await client.close_loop(loop)
        return loop, c

    loop, c = asyncio.run(run())
    assert c.is_closed
    assert loop not in client._http_clients
    assert len(client._http_clients) == 0


def test_close_loop_idempotent():
    """close_loop 重复调用不炸（B/D worker finally 可能多次触发）。"""
    client = make_client()

    async def run():
        loop = asyncio.get_running_loop()
        client._client_for_loop()
        await client.close_loop(loop)
        await client.close_loop(loop)   # 第二次：条目已不存在，静默返回
        return loop

    loop = asyncio.run(run())
    assert loop not in client._http_clients


def test_close_loop_foreign_loop_noop():
    """close_loop 传入从未注册的 loop：静默返回，不炸，不影响自己的条目。"""
    client = make_client()
    holder = {}

    async def run():
        holder["loop"] = asyncio.get_running_loop()
        client._client_for_loop()
        other = asyncio.new_event_loop()
        await client.close_loop(other)
        other.close()

    asyncio.run(run())
    # 自己的条目不受影响（loop 强引用存活）
    assert len(client._http_clients) == 1
    assert next(iter(client._http_clients)) is holder["loop"]


def test_close_loop_after_close_still_safe():
    """先 close() 全量关闭，再对具体 loop close_loop：安全且幂等。"""
    client = make_client()

    async def run():
        loop = asyncio.get_running_loop()
        client._client_for_loop()
        await client.close()
        await client.close_loop(loop)
        await client.close_loop(loop)
        return loop

    loop = asyncio.run(run())
    assert len(client._http_clients) == 0


# ============ close() 全量语义不变 ============

def test_close_closes_all_and_clears():
    """close() 关闭全部 client 并清空（web/main.py lifespan 依赖，语义不可变）。"""
    client = make_client()

    async def run():
        c1 = client._client_for_loop()
        await client.close()
        return c1

    c1 = asyncio.run(run())
    assert c1.is_closed
    assert len(client._http_clients) == 0


def test_rebuild_after_close():
    """close() 后再次调用 _client_for_loop：惰性重建新 client（is_closed 兜底）。"""
    client = make_client()

    async def run():
        c1 = client._client_for_loop()
        await client.close()
        c2 = client._client_for_loop()
        return c1, c2

    c1, c2 = asyncio.run(run())
    assert c1 is not c2
    assert not c2.is_closed


# ============ 死循环条目自动淘汰（WeakKeyDictionary 核心收益） ============

def test_dead_loop_entry_gc_collected():
    """loop 死亡后条目必须自动淘汰（原 dict[id] 永不淘汰 → 每任务泄漏一 client）。

    验证链：持有强引用 → 条目存在；解除强引用 → 弱引用回调清除条目。
    """
    client = make_client()
    holder = {}

    async def touch():
        holder["loop"] = asyncio.get_running_loop()
        client._client_for_loop()

    asyncio.run(touch())
    assert len(client._http_clients) == 1  # 强引用存活 → 条目存在
    holder.clear()
    gc.collect()
    assert len(client._http_clients) == 0  # 唯一强引用解除 → 条目自动清除


def test_many_loops_no_leak():
    """连续 10 个独立 loop 后字典不增长（模拟 10 次后台 AI 任务）。"""
    client = make_client()

    async def touch():
        client._client_for_loop()

    for _ in range(10):
        asyncio.run(touch())
    gc.collect()
    assert len(client._http_clients) == 0
