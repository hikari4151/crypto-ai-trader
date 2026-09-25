"""网络加速节点扫描（2026-09-24）。

用户提供订阅 URL / 节点文本（Clash YAML、socks5/http/ss 行格式），
解析全部节点供网页选择；选中节点即作为应用级代理（settings.resolved_proxy）。
- supported=True：可直接作为 http/socks5 代理 URL（数据层直接使用）
- supported=False（ss/vmess/trojan 等）：需本地客户端转换，列出但提示不可直用
"""
import base64
import logging
import re
import socket
import time
from typing import Any, Optional
from urllib.parse import urlparse

import yaml  # type: ignore

log = logging.getLogger("proxy_nodes")

# 可直接作为代理 URL 的类型（数据层/ccxt 支持）
_DIRECT_TYPES = {"socks5", "socks4", "http", "https"}


def _norm_type(t: str) -> str:
    return (t or "").strip().lower()


def _parse_ss_link(line: str) -> dict:
    """解析 ss:// 分享链接（base64 userinfo 或明文），返回节点 dict（不支持直用）。"""
    raw = line[len("ss://"):].split("#")[0]
    name = ""
    m = re.search(r"#([^#\s]+)$", line)
    if m:
        name = m.group(1).strip()
    server = port = ""
    userinfo = raw
    if "@" in raw:
        userinfo, hostpart = raw.rsplit("@", 1)
        hp = hostpart.split(":")
        server = hp[0]
        port = int(hp[1]) if len(hp) > 1 and hp[1].isdigit() else None
    else:
        # 纯 base64(method:password@server:port)
        try:
            pad = "=" * (-len(userinfo) % 4)
            decoded = base64.urlsafe_b64decode(userinfo + pad).decode(errors="ignore")
            if "@" in decoded:
                _, hostpart = decoded.rsplit("@", 1)
                hp = hostpart.split(":")
                server = hp[0]
                port = int(hp[1]) if len(hp) > 1 and hp[1].isdigit() else None
        except Exception:  # noqa: BLE001
            server, port = "", None
    return {
        "name": name or f"{server or 'ss'}:{port or ''}",
        "type": "ss", "server": server, "port": port,
        "url": "", "supported": False,
    }


def parse_nodes_text(text: str) -> list[dict]:
    """解析节点文本（Clash YAML 或每行 URL），返回节点列表。"""
    text = (text or "").strip()
    if not text:
        return []
    nodes: list[dict] = []
    # 1) Clash YAML：含 proxies 段（顶层或缩进）
    yaml_nodes = _parse_clash_yaml(text)
    if yaml_nodes:
        nodes.extend(yaml_nodes)
    # 2) 行格式 URL
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("proxies"):
            continue
        low = line.lower()
        if low.startswith(("socks5://", "socks4://", "http://", "https://")):
            u = urlparse(line)
            port = u.port or (443 if u.scheme == "https" else 80)
            nodes.append({
                "name": f"{u.hostname}:{port}",
                "type": u.scheme, "server": u.hostname, "port": port,
                "url": line.rstrip("/"), "supported": True,
            })
        elif low.startswith("ss://"):
            nodes.append(_parse_ss_link(line))
    return nodes


def _parse_clash_yaml(text: str) -> list[dict]:
    """Clash YAML：提取 proxies 列表（仅标准键）。"""
    try:
        data = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, dict):
        return []
    proxies = data.get("proxies")
    if not isinstance(proxies, list):
        return []
    out: list[dict] = []
    for p in proxies:
        if not isinstance(p, dict):
            continue
        typ = _norm_type(p.get("type", ""))
        server = str(p.get("server", "")).strip()
        port = p.get("port")
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = None
        name = str(p.get("name") or f"{server}:{port}").strip()
        # 注意：and 短路链返回最后一个真值——必须 bool() 包裹
        supported = bool(typ in _DIRECT_TYPES and server and port)
        url = f"{typ}://{server}:{port}" if supported else ""
        out.append({
            "name": name, "type": typ, "server": server, "port": port,
            "url": url, "supported": supported,
        })
    return out


def node_to_url(node: dict) -> str:
    """supported 节点 → 代理 URL。"""
    typ = _norm_type(node.get("type", ""))
    server = node.get("server")
    port = node.get("port")
    if typ in _DIRECT_TYPES and server and port:
        return f"{typ}://{server}:{port}"
    return ""


def probe_node(url: str, timeout: float = 0.4) -> tuple[bool, Optional[float]]:
    """探测代理节点可达性：返回 (可达, 延迟ms)。"""
    if not url:
        return False, None
    try:
        u = urlparse(url if "://" in url else "http://" + url)
        host = u.hostname or "127.0.0.1"
        port = u.port or 80
        t0 = time.perf_counter()
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, int((time.perf_counter() - t0) * 1000)
    except OSError:
        return False, None


async def resolve_subscription(url: str, fetcher: Any = None) -> list[dict]:
    """拉取订阅 URL 内容并解析节点。

    fetcher: 可选对象，须有 async fetch(url)->str（测试注入；默认用 requests 线程化）。
    """
    if fetcher is not None:
        text = await fetcher.fetch(url)
        return parse_nodes_text(text)
    import asyncio
    import requests
    try:
        resp = await asyncio.to_thread(requests.get, url, timeout=10)
        resp.raise_for_status()
        return parse_nodes_text(resp.text)
    except Exception as e:  # noqa: BLE001
        log.warning("[proxy] 订阅拉取失败 %s: %s", url, e)
        return []