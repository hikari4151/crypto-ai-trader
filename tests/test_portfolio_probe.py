"""实盘仪表盘「网络不可达」误报回归（2026-09-28）。

用户现象：切到实盘模式后，仪表盘仍显示「无网络」，但交易所测试连接是通的。

根因：`_socket_probe` 用**裸 socket 直连**交易所 host:443 做可达性探测，完全绕过
应用级代理（网络加速）。本机直连 `www.okx.com:443` 必然超时（实测 1521ms），
于是探测恒为 False ⇒ `/api/portfolio/summary` 直接返回「okx 网络不可达」，
**根本没走到真正走代理的钱包请求**——而那条路径（ccxt + socksProxy）实测 500ms 通。

修法：配置了代理 → 探**代理 host:port**（保留 1s 快速失败的价值）；
未配置 → 维持原直连探测。错误文案同步指向内置「网络加速」。
"""
import pytest

from web.api import portfolio as pf


def _set_proxy(monkeypatch, url: str) -> None:
    from config.settings import settings
    monkeypatch.setattr(settings, "_resolve_proxy_sync", lambda: url, raising=False)
    settings.reset_proxy_cache()


@pytest.fixture(autouse=True)
def _clean_proxy_cache():
    from config.settings import settings
    settings.reset_proxy_cache()
    yield
    settings.reset_proxy_cache()


@pytest.mark.asyncio
async def test_probe_targets_proxy_endpoint_when_proxy_configured(monkeypatch):
    """配置代理后必须探代理端口——裸连交易所 host 在被墙网络必然超时。"""
    _set_proxy(monkeypatch, "socks5h://127.0.0.1:7899")
    seen = {}

    def _fake_connect(host, port):
        seen["hp"] = (host, port)

    monkeypatch.setattr(pf, "_socket_connect", _fake_connect)
    ok = await pf._socket_probe("okx")
    assert ok is True
    assert seen.get("hp") == ("127.0.0.1", 7899), \
        f"应探测代理端口，实际探测了 {seen.get('hp')}"


@pytest.mark.asyncio
async def test_probe_keeps_direct_when_no_proxy(monkeypatch):
    """未配置代理时保持原直连探测（回归保护：不要把直连也改掉）。"""
    _set_proxy(monkeypatch, "")
    seen = {}

    def _fake_connect(host, port):
        seen["hp"] = (host, port)

    monkeypatch.setattr(pf, "_socket_connect", _fake_connect)
    ok = await pf._socket_probe("okx")
    assert ok is True
    assert seen.get("hp") == ("www.okx.com", 443)
    # 其它交易所的 host 映射同样不被破坏
    seen.clear()
    await pf._socket_probe("binance")
    assert seen.get("hp") == ("data-api.binance.vision", 443)


@pytest.mark.asyncio
async def test_probe_false_when_proxy_port_dead(monkeypatch):
    """代理端口没人监听 ⇒ 探测失败（快失败仍然成立）。"""
    _set_proxy(monkeypatch, "socks5h://127.0.0.1:7899")

    def _boom(host, port):
        raise OSError("connection refused")

    monkeypatch.setattr(pf, "_socket_connect", _boom)
    assert await pf._socket_probe("okx") is False


@pytest.mark.asyncio
async def test_unparsable_proxy_does_not_block(monkeypatch):
    """代理串解析不出 host/port 时不拦截：交给真实请求去报具体错误。"""
    _set_proxy(monkeypatch, "socks5h://")

    def _boom(host, port):
        raise AssertionError("不应发起连接")

    monkeypatch.setattr(pf, "_socket_connect", _boom)
    assert await pf._socket_probe("okx") is True


@pytest.mark.asyncio
async def test_unreachable_error_points_to_builtin_accel(monkeypatch):
    """探测失败的提示要指向内置「网络加速」，而不是只让用户去开 FlClash。"""
    _set_proxy(monkeypatch, "socks5h://127.0.0.1:7899")

    async def _no(*a, **k):
        return False

    monkeypatch.setattr(pf, "_socket_probe", _no)
    res = await pf._fetch_live_balance("okx", "k", "s", "")
    assert res["equity"] is None
    assert "网络加速" in res["error"], f"提示应指向内置加速，实际: {res['error']!r}"


@pytest.mark.asyncio
async def test_unreachable_error_without_proxy_still_explains(monkeypatch):
    _set_proxy(monkeypatch, "")

    async def _no(*a, **k):
        return False

    monkeypatch.setattr(pf, "_socket_probe", _no)
    res = await pf._fetch_live_balance("okx", "k", "s", "")
    assert "okx" in res["error"] and "网络" in res["error"]


class _FakeMgr:
    closed = 0

    def __init__(self, *a, **k):
        pass

    async def close(self):
        _FakeMgr.closed += 1


@pytest.mark.asyncio
async def test_live_balance_retries_transient_network_error(monkeypatch):
    """瞬时网络错误（实测：重启后首次经节点拉取偶发 Host unreachable）重试一次。

    不重试的话：失败结果还会被 5 秒缓存 ⇒ 用户重启后第一眼必然看到红条。
    """
    import exchange.manager as mgr_mod

    _set_proxy(monkeypatch, "socks5h://127.0.0.1:7899")
    _FakeMgr.closed = 0

    async def _ok(*a, **k):
        return True

    monkeypatch.setattr(pf, "_socket_probe", _ok)
    monkeypatch.setattr(mgr_mod, "ExchangeManager", _FakeMgr)
    monkeypatch.setattr(pf, "_RETRY_BACKOFF", 0)   # 测试不等真实退避

    calls = {"n": 0}

    async def _flaky(mgr, key, sec, pwd):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Host unreachable")
        # _do_fetch 返回的是折算后的 portfolio 口径（不是原始 balance）
        return {"equity": 5.0, "cash": 5.0, "positions_value": 0.0, "positions": []}

    monkeypatch.setattr(pf, "_do_fetch", _flaky)
    res = await pf._fetch_live_balance("okx", "k", "s", "")
    assert calls["n"] == 2, f"第 1 次瞬时失败后应重试，实际调用 {calls['n']} 次"
    assert res.get("error") in (None, ""), f"重试成功后不应带 error: {res}"
    assert res["equity"] == 5.0


@pytest.mark.asyncio
async def test_live_balance_does_not_retry_permanent_error(monkeypatch):
    """非网络类错误（如密钥无效）不重试——避免把 4xx 拖成两倍时延。"""
    import exchange.manager as mgr_mod

    _set_proxy(monkeypatch, "socks5h://127.0.0.1:7899")
    _FakeMgr.closed = 0

    async def _ok(*a, **k):
        return True

    monkeypatch.setattr(pf, "_socket_probe", _ok)
    monkeypatch.setattr(mgr_mod, "ExchangeManager", _FakeMgr)
    monkeypatch.setattr(pf, "_RETRY_BACKOFF", 0)

    calls = {"n": 0}

    async def _auth_fail(mgr, key, sec, pwd):
        calls["n"] += 1
        raise Exception("AuthenticationError: invalid api key")

    monkeypatch.setattr(pf, "_do_fetch", _auth_fail)
    res = await pf._fetch_live_balance("okx", "k", "s", "")
    assert calls["n"] == 1, "认证类错误不应重试"
    assert "拉取失败" in res["error"]
