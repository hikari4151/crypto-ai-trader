"""Quantumult X / Loon 风格订阅行解析测试（2026-09-25）。

格式：`协议 = host:port, 节点名`（socks5/http/trojan/vmess/ss 等），
机场常见导出格式之一。
"""
from net.proxy_nodes import parse_nodes_text


def test_parse_quantumultx_lines():
    text = (
        "socks5 = 1.2.3.4:7890, HK-A\n"
        "http = 5.6.7.8:8080, JP-B\n"
        "trojan = 8.8.8.8:443, US-C, password=secret\n"
        "vmess = 9.9.9.9:443, SG-D, method=aes"
    )
    nodes = parse_nodes_text(text)
    assert len(nodes) == 4
    by = {n["name"]: n for n in nodes}
    assert by["HK-A"]["type"] == "socks5" and by["HK-A"]["supported"] is True
    assert by["HK-A"]["url"] == "socks5://1.2.3.4:7890"
    assert by["JP-B"]["type"] == "http" and by["JP-B"]["supported"] is True
    assert by["US-C"]["type"] == "trojan" and by["US-C"]["supported"] is False
    assert by["SG-D"]["type"] == "vmess" and by["SG-D"]["supported"] is False


def test_quantumultx_not_parsed_for_plain_urls():
    # 常规 URL 行不应被误判
    nodes = parse_nodes_text("socks5://1.2.3.4:7890\nhttp://5.6.7.8:8080")
    assert len(nodes) == 2
    assert all(n["supported"] is True for n in nodes)
