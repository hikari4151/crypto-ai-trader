"""hysteria2 / tuic 等新协议订阅行测试（2026-09-25）。

真实机场订阅内容为 base64 编码的 hysteria2:// 行——此前 base64 识别标记
不含新协议，导致解析空（"未解析到节点"）。
"""
import base64

from net.proxy_nodes import parse_nodes_text


def test_base64_hysteria2_subscription():
    inner = (
        "hysteria2://c0f9ee58-ce11-4e63-9115-d3d0887a3626@203.10.98.189:443/"
        "?insecure=false&sni=www.bing.com&pinSHA256=1d7995901a93bada6d17f10289441793e8ec54ee314d9f04f3a9d05daa622331#"
        "香港-01"
    )
    b64 = base64.b64encode(inner.encode()).decode()
    nodes = parse_nodes_text(b64)
    assert len(nodes) == 1
    n = nodes[0]
    assert n["type"] == "hysteria2"
    assert n["name"] == "香港-01"
    assert n["server"] == "203.10.98.189" and n["port"] == 443
    assert n["supported"] is False  # hysteria2 需本地客户端转换


def test_hysteria_and_tuic_lines():
    text = (
        "hysteria2://uuid@1.2.3.4:443/?sni=x#HK-H2\n"
        "tuic://uuid@5.6.7.8:8443?sni=y#JP-TUI\n"
        "hysteria://1.1.1.1:36711#SG-HY"
    )
    nodes = parse_nodes_text(text)
    assert len(nodes) == 3
    types = {n["type"] for n in nodes}
    assert types == {"hysteria2", "tuic", "hysteria"}
    assert all(n["supported"] is False for n in nodes)