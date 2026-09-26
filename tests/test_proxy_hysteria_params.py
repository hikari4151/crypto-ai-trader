"""hysteria2 节点参数解析测试（2026-09-25）：节点需携带 auth/sni 等参数，
供程序内置 hysteria 客户端生成配置并本地启动 SOCKS5（软件直接支持 hysteria2）。
"""
from net.proxy_nodes import parse_nodes_text


def test_hysteria2_params_extracted():
    line = (
        "hysteria2://c0f9ee58-ce11-4e63-9115-d3d0887a3626@203.10.98.189:443/"
        "?insecure=false&sni=www.bing.com&pinSHA256=1d7995901a93bada6d17f10289441793e8ec54ee314d9f04f3a9d05daa622331#"
        "日本专线"
    )
    nodes = parse_nodes_text(line)
    assert len(nodes) == 1
    n = nodes[0]
    params = n.get("params", {})
    assert params.get("auth") == "c0f9ee58-ce11-4e63-9115-d3d0887a3626"
    assert params.get("sni") == "www.bing.com"
    assert params.get("insecure") == "false"
    assert params.get("pinSHA256") == "1d7995901a93bada6d17f10289441793e8ec54ee314d9f04f3a9d05daa622331"


def test_hysteria2_without_query():
    line = "hysteria2://uuid@1.2.3.4:443#HK-H2"
    n = parse_nodes_text(line)[0]
    params = n.get("params", {})
    assert params.get("auth") == "uuid"
    assert params.get("sni") == ""
