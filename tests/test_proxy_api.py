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


# 2026-09-27：内置加速失效根因（三连）
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
