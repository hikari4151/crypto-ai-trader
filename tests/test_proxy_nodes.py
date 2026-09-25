"""网络加速节点扫描解析测试（2026-09-24）。

用户提供订阅 URL / 节点文本（Clash YAML、socks5/http/ss 行格式）→
解析全部节点（名称/类型/地址/端口/是否可直接用作代理 URL），供网页选择。
"""
import pytest

from net.proxy_nodes import (
    node_to_url,
    parse_nodes_text,
    probe_node,
    resolve_subscription,
)


_CLASH_YAML = """proxies:
  - name: HK1
    type: socks5
    server: 1.2.3.4
    port: 7890
  - name: JP2
    type: http
    server: 5.6.7.8
    port: 8080
  - name: US3
    type: ss
    server: 9.9.9.9
    port: 8388
    cipher: aes-256-gcm
    password: secret
  - name: SG4
    type: vmess
    server: 8.8.8.8
    port: 443
"""


def test_parse_clash_yaml_nodes():
    nodes = parse_nodes_text(_CLASH_YAML)
    assert len(nodes) == 4
    by_name = {n["name"]: n for n in nodes}
    assert by_name["HK1"]["supported"] is True
    assert by_name["HK1"]["url"] == "socks5://1.2.3.4:7890"
    assert by_name["JP2"]["supported"] is True
    assert by_name["JP2"]["url"] == "http://5.6.7.8:8080"
    # ss/vmess 无法直接作为 http/socks5 代理 URL → 标记不支持但保留供展示
    assert by_name["US3"]["supported"] is False
    assert by_name["SG4"]["supported"] is False


def test_parse_line_urls():
    text = "socks5://1.2.3.4:7890\nhttp://5.6.7.8:8080\nhttps://9.9.9.9:443"
    nodes = parse_nodes_text(text)
    assert len(nodes) == 3
    assert all(n["supported"] for n in nodes)
    urls = {n["url"] for n in nodes}
    assert "socks5://1.2.3.4:7890" in urls
    assert "http://5.6.7.8:8080" in urls


def test_parse_ss_share_link():
    # ss://base64(method:password@server:port)#name
    import base64
    userinfo = base64.urlsafe_b64encode(b"aes-256-gcm:secret@9.9.9.9:8388").decode().rstrip("=")
    text = f"ss://{userinfo}#US-SS"
    nodes = parse_nodes_text(text)
    assert len(nodes) == 1
    n = nodes[0]
    assert n["name"] == "US-SS"
    assert n["server"] == "9.9.9.9" and n["port"] == 8388
    assert n["supported"] is False  # ss 需本地客户端转换，不能直接作代理 URL


def test_node_to_url_and_probe():
    n = {"server": "127.0.0.1", "port": 7890, "type": "socks5", "name": "t"}
    assert node_to_url(n) == "socks5://127.0.0.1:7890"
    # 探测本地不存在的端口 → (False, None)
    ok, ms = probe_node("http://127.0.0.1:9", timeout=0.3)
    assert ok is False


@pytest.mark.asyncio
async def test_resolve_subscription_returns_parsed():
    """订阅 URL 拉取后解析（mock 拉取函数）。"""
    class _FakeFetcher:
        async def fetch(self, url):  # noqa: A002
            return _CLASH_YAML
    nodes = await resolve_subscription("https://example.com/sub", fetcher=_FakeFetcher())
    assert len(nodes) == 4


# 2026-09-25：常见订阅格式覆盖（base64 编码 / vmess/trojan/ssr 行）
def test_parse_base64_encoded_subscription():
    import base64
    raw = "socks5://1.2.3.4:7890\nss://FakePass@5.6.7.8:8388"
    b64 = base64.b64encode(raw.encode()).decode()
    nodes = parse_nodes_text(b64)
    assert len(nodes) >= 2
    socks = [n for n in nodes if n["type"] == "socks5"]
    assert socks and socks[0]["supported"] is True


def test_parse_vmess_link():
    import base64
    import json
    payload = json.dumps({"v": "2", "ps": "HK-VM", "add": "1.2.3.4", "port": "443", "id": "x"})
    b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    nodes = parse_nodes_text(f"vmess://{b64}")
    assert len(nodes) == 1
    n = nodes[0]
    assert n["name"] == "HK-VM" and n["server"] == "1.2.3.4" and n["port"] == 443
    assert n["supported"] is False  # vmess 需本地客户端转换


def test_parse_trojan_and_ssr_lines():
    nodes = parse_nodes_text("trojan://pass@8.8.8.8:443#US-TJ\nssr://abcDEF123")
    assert len(nodes) == 2
    tj = [n for n in nodes if n["type"] == "trojan"][0]
    assert tj["server"] == "8.8.8.8" and tj["port"] == 443 and tj["name"] == "US-TJ"
    assert tj["supported"] is False
    ssr = [n for n in nodes if n["type"] == "ssr"][0]
    assert ssr["supported"] is False
