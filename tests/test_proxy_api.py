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
