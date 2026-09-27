"""网络加速（应用级代理）配置 API 测试（2026-09-24）。

功能：用户提供 VPN/代理 URL（网页配置），仅作用于本软件出站请求；
设置持久化 data/proxy_config.json，重启恢复；清空回退自动探测。
"""
import pytest

from config.settings import settings
from web.api import proxy as proxy_api


@pytest.mark.asyncio
async def test_set_proxy_updates_and_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_api, "OVERRIDE_FILE", tmp_path / "proxy_config.json")
    monkeypatch.setattr(settings, "_resolve_proxy_sync",
                        lambda: "http://127.0.0.1:17890", raising=False)
    r = await proxy_api.set_proxy(proxy_api.ProxyConfigIn(proxy_url="http://127.0.0.1:17890"))
    assert r["ok"] is True
    assert r["proxy_url"] == "http://127.0.0.1:17890"
    assert r["resolved_proxy"] == "http://127.0.0.1:17890"
    assert r["auto_detected"] is False
    # 持久化文件已写（重启恢复）
    assert (tmp_path / "proxy_config.json").exists()
    assert "http://127.0.0.1:17890" in (tmp_path / "proxy_config.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_clear_proxy_auto_detect(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_api, "OVERRIDE_FILE", tmp_path / "proxy_config.json")
    monkeypatch.setattr(settings, "_resolve_proxy_sync",
                        lambda: "http://127.0.0.1:17890", raising=False)
    await proxy_api.set_proxy(proxy_api.ProxyConfigIn(proxy_url="http://127.0.0.1:17890"))
    r = await proxy_api.set_proxy(proxy_api.ProxyConfigIn(proxy_url="   "))
    assert r["proxy_url"] == ""
    assert r["auto_detected"] is True
    assert (tmp_path / "proxy_config.json").read_text(encoding="utf-8").strip() != ""
    assert '"proxy_url": ""' in (tmp_path / "proxy_config.json").read_text(encoding="utf-8")


def test_reset_proxy_cache_revalidates(monkeypatch):
    """reset_proxy_cache 后 resolved_proxy 重新探测（动态更新生效）。"""
    calls = {"n": 0}

    def _fake_resolve():
        calls["n"] += 1
        return f"http://127.0.0.1:{9000 + calls['n']}"
    monkeypatch.setattr(settings, "_resolve_proxy_sync", _fake_resolve, raising=False)
    settings.reset_proxy_cache()
    first = settings.resolved_proxy
    second = settings.resolved_proxy  # 缓存命中，不变
    assert first == second and calls["n"] == 1
    settings.reset_proxy_cache()
    third = settings.resolved_proxy
    assert third != first and calls["n"] == 2, "reset 后应重新探测"

# 2026-09-25：手动代理 URL 校验（拒绝订阅链接/非法协议，防"网络加速无效"）
def test_validate_proxy_url_rejects_subscription():
    from web.api.proxy import _validate_proxy_url
    ok, reason = _validate_proxy_url("https://x.example/api/v1/client/subscribe?token=abc")
    assert ok is False and "扫描全部节点" in reason
    ok, _ = _validate_proxy_url("https://x.example/sub?url=1")
    assert ok is False


def test_validate_proxy_url_rejects_bad_scheme():
    from web.api.proxy import _validate_proxy_url
    ok, reason = _validate_proxy_url("vmess://1.2.3.4:443")
    assert ok is False and "socks5" in reason
    ok, _ = _validate_proxy_url("socks5://1.2.3.4:7890")
    assert ok is True
    ok, _ = _validate_proxy_url("")
    assert ok is True


# 2026-09-27（第二轮）：软件内仍报「网络问题」——根因是 ccxt 异步的代理写法
# ① ccxt async 的 REST 走 aiohttp，`aiohttp_proxy` 只认 http/https；socks 必须用
#    `socksProxy`（内部走 aiohttp_socks，且识别 socks5h 远端 DNS）。
#    误用 aiohttp_proxy → WinError 64 → /api/exchanges/test 502（用户实测）。
#    两者同时设置 ccxt 会抛 NotSupported('you have set multiple proxies')。
def test_ccxt_async_proxy_cfg_picks_scheme():
    from config.settings import ccxt_async_proxy_cfg
    assert ccxt_async_proxy_cfg("socks5h://127.0.0.1:7891") == {"socksProxy": "socks5h://127.0.0.1:7891"}
    assert ccxt_async_proxy_cfg("socks5://10.0.0.5:1080") == {"socksProxy": "socks5://10.0.0.5:1080"}
    assert ccxt_async_proxy_cfg("http://127.0.0.1:7890") == {"aiohttp_proxy": "http://127.0.0.1:7890"}
    assert ccxt_async_proxy_cfg(None) == {}
    assert ccxt_async_proxy_cfg("") == {}


def test_ccxt_async_proxy_cfg_never_sets_both():
    """同时设 socksProxy 与 aiohttp_proxy 会被 ccxt 直接拒绝。"""
    from config.settings import ccxt_async_proxy_cfg
    for px in ("socks5h://127.0.0.1:7891", "http://127.0.0.1:7890", None):
        cfg = ccxt_async_proxy_cfg(px)
        assert not (("socksProxy" in cfg) and ("aiohttp_proxy" in cfg))


@pytest.mark.asyncio
async def test_market_fetch_retries_transport_errors(monkeypatch):
    """传输层错误（代理瞬时 Host unreachable）也要重试，而非直接 502。

    实测：binance tickers 经节点首连偶发 ProxyError、第 2 次即成功 1.9MB。
    """
    import httpx
    from web.api import market as market_api

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class _Flaky:
        def __init__(self, fails):
            self.fails = fails
            self.calls = 0

        async def get(self, url, params=None):
            self.calls += 1
            if self.calls <= self.fails:
                raise httpx.ConnectError("proxy unreachable")
            return _Resp()

    flaky = _Flaky(fails=1)
    monkeypatch.setattr(market_api, "_get_client", lambda: flaky)
    out = await market_api._fetch_json("https://example.com/x")
    assert out == {"ok": True}
    assert flaky.calls == 2, "第 1 次传输失败应重试"


@pytest.mark.asyncio
async def test_market_fetch_error_mentions_exception_type(monkeypatch):
    """错误体不能是空串（用户看到"数据源不可达: "无法定位）。"""
    import httpx
    from fastapi import HTTPException
    from web.api import market as market_api

    class _Dead:
        def __init__(self):
            self.calls = 0

        async def get(self, url, params=None):
            self.calls += 1
            raise httpx.ConnectError("")

    dead = _Dead()
    monkeypatch.setattr(market_api, "_get_client", lambda: dead)
    with pytest.raises(HTTPException) as ei:
        await market_api._fetch_json("https://example.com/x")
    detail = ei.value.detail
    assert "ConnectError" in detail, f"错误体应含异常类型，实际: {detail!r}"
    assert dead.calls == 3, "重试耗尽后才失败"


# 2026-09-27（第一轮）：内置加速失效根因（三连）
# ① 本机 DNS 被污染（www.okx.com -> 169.254.0.2），socks5:// 在本地解析域名，
#    经隧道连到废 IP 必然超时；socks5h:// 由节点侧解析，实测 OKX 1.5s 通。
# ② hysteria 节点自签证书 + 只写 pin ⇒ 客户端秒退（详见 test_proxy_client）。
# ③ 代理不可达时接口把死地址报成 resolved_proxy，界面显示"生效"却在直连。
def test_validate_proxy_url_accepts_socks5h():
    from web.api.proxy import _validate_proxy_url
    ok, reason = _validate_proxy_url("socks5h://127.0.0.1:7891")
    assert ok is True, reason


def test_resolved_proxy_upgrades_socks5_to_socks5h():
    from config.settings import normalize_proxy_url
    assert normalize_proxy_url("socks5://1.2.3.4:7890") == "socks5h://1.2.3.4:7890"
    assert normalize_proxy_url("SOCKS5://1.2.3.4:7890") == "socks5h://1.2.3.4:7890"
    assert normalize_proxy_url("socks5h://1.2.3.4:1080") == "socks5h://1.2.3.4:1080"
    assert normalize_proxy_url("http://127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert normalize_proxy_url("socks4://1.2.3.4:1080") == "socks4://1.2.3.4:1080"
    assert normalize_proxy_url("") == ""


def test_resolve_proxy_sync_applies_normalization(monkeypatch):
    """显式配置 socks5:// 且可达 → resolved 必须返回 socks5h://（远端 DNS）。"""
    from config import settings as settings_mod

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(settings_mod, "probe_local_proxy", lambda: "")
    monkeypatch.setattr(settings_mod.socket, "create_connection", lambda *a, **k: _Conn())
    monkeypatch.setattr(settings, "proxy_url", "socks5://127.0.0.1:7891")
    assert settings._resolve_proxy_sync() == "socks5h://127.0.0.1:7891"
    monkeypatch.setattr(settings, "proxy_url", "http://127.0.0.1:7890")
    assert settings._resolve_proxy_sync() == "http://127.0.0.1:7890"


@pytest.mark.asyncio
async def test_select_node_does_not_report_dead_proxy_as_resolved(monkeypatch, tmp_path):
    """节点不可达时不许把死地址报成生效（否则界面显示"生效"却在直连）。"""
    monkeypatch.setattr(proxy_api, "OVERRIDE_FILE", tmp_path / "proxy_config.json")
    monkeypatch.setattr(settings, "_resolve_proxy_sync", lambda: "", raising=False)
    settings.reset_proxy_cache()
    try:
        r = await proxy_api.select_proxy_node(proxy_api.ProxySelectIn(
            node={"type": "socks5", "url": "socks5://1.2.3.4:7890"}))
        assert r["ok"] is True
        assert r["resolved_proxy"] == "", "不可达时 resolved_proxy 必须为空"
        assert r["unreachable"] is True, "必须显式告知不可达"
    finally:
        settings.reset_proxy_cache()


@pytest.mark.asyncio
async def test_get_proxy_rechecks_liveness_when_client_died(monkeypatch):
    """resolved_proxy 是启动时探测的进程级缓存：本地客户端中途挂掉后必须按实测置空。

    否则界面显示"生效"、实际请求连到死端口（用户实测"还是报网络问题"的一种）。
    """
    monkeypatch.setattr(settings, "proxy_url", "socks5h://127.0.0.1:7999")
    settings.reset_proxy_cache()
    monkeypatch.setattr(settings, "_resolve_proxy_sync", lambda: "socks5h://127.0.0.1:7999",
                        raising=False)
    settings.reset_proxy_cache()
    try:
        monkeypatch.setattr(proxy_api, "_probe_alive", lambda url, timeout=0.4: False)
        r = await proxy_api.get_proxy()
        assert r["resolved_proxy"] == "", "端口没人监听时 resolved 必须为空"
        assert r["unreachable"] is True
    finally:
        settings.reset_proxy_cache()


@pytest.mark.asyncio
async def test_get_proxy_keeps_resolved_when_alive(monkeypatch):
    monkeypatch.setattr(settings, "proxy_url", "socks5h://127.0.0.1:7998")
    monkeypatch.setattr(settings, "_resolve_proxy_sync", lambda: "socks5h://127.0.0.1:7998",
                        raising=False)
    settings.reset_proxy_cache()
    try:
        monkeypatch.setattr(proxy_api, "_probe_alive", lambda url, timeout=0.4: True)
        r = await proxy_api.get_proxy()
        assert r["resolved_proxy"] == "socks5h://127.0.0.1:7998"
        assert r["unreachable"] is False
    finally:
        settings.reset_proxy_cache()


def test_probe_alive_connects_to_host_port():
    """_probe_alive：连不上返回 False，连得上返回 True，无法解析按 True（不误报）。"""
    assert proxy_api._probe_alive("socks5h://127.0.0.1:1", timeout=0.2) is False
    assert proxy_api._probe_alive("", timeout=0.2) is True
    assert proxy_api._probe_alive("not-a-url", timeout=0.2) is True


# 2026-09-25：扫描端点拉取失败应返回具体 error（前端可显示，而非"未解析到节点"）
@pytest.mark.asyncio
async def test_scan_returns_error_on_fetch_failure(monkeypatch):
    from web.api import proxy as proxy_api
    async def _bad(url):  # noqa: A002
        raise ValueError("订阅拉取失败: Read timed out")
    monkeypatch.setattr("net.proxy_nodes.resolve_subscription", _bad)
    r = await proxy_api.scan_proxy_nodes(proxy_api.ProxyNodesIn(source="https://x.example/sub"))
    assert r["ok"] is False
    assert "Read timed out" in r["error"]


@pytest.mark.asyncio
async def test_scan_returns_format_hint_when_empty(monkeypatch):
    from web.api import proxy as proxy_api
    async def _empty(url):  # noqa: A002
        return []
    monkeypatch.setattr("net.proxy_nodes.resolve_subscription", _empty)
    r = await proxy_api.scan_proxy_nodes(proxy_api.ProxyNodesIn(source="https://x.example/sub"))
    assert r["count"] == 0
    assert "未解析到节点" in r.get("error", "")
