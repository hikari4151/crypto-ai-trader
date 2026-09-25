"""网络加速节点扫描（2026-09-24）。

用户提供订阅 URL / 节点文本（Clash YAML、socks5/http/ss 行格式），
解析全部节点供网页选择；选中节点即作为应用级代理（settings.resolved_proxy）。
- supported=True：可直接作为 http/socks5 代理 URL（数据层直接使用）
- supported=False（ss/vmess/trojan 等）：需本地客户端转换，列出但提示不可直用
"""
import base64
import json
import logging
import re
import socket
import time
from typing import Any, Optional
from urllib.parse import unquote, urlparse

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
        name = unquote(m.group(1).strip())
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




_NODE_MARKERS = ("ss://", "vmess://", "trojan://", "ssr://",
                 "hysteria2://", "hysteria://", "tuic://", "wireguard://",
                 "socks5://", "socks4://", "http://", "https://", "proxies:")


def _maybe_base64_decode(text: str) -> Optional[str]:
    """订阅内容常见为整段 base64（节点行编码）——自动解码识别。"""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 40:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return None
    try:
        pad = "=" * (-len(compact) % 4)
        decoded = base64.b64decode(compact + pad).decode(errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    return decoded if any(m in decoded for m in _NODE_MARKERS) else None


def _parse_vmess_link(line: str) -> dict:
    """vmess://base64(JSON) → 节点（需本地客户端转换，仅提取展示信息）。"""
    raw = line[len("vmess://"):].split("#")[0]
    server = port = ""
    name = ""
    try:
        pad = "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw + pad).decode(errors="ignore"))
        name = unquote(str(payload.get("ps") or ""))
        server = str(payload.get("add") or "")
        port = int(payload.get("port")) if str(payload.get("port", "")).isdigit() else None
    except Exception:  # noqa: BLE001
        pass
    return {"name": name or f"{server or 'vmess'}:{port or ''}",
            "type": "vmess", "server": server, "port": port,
            "url": "", "supported": False}


def _parse_trojan_link(line: str) -> dict:
    """trojan://pass@host:port#name（需本地客户端转换）。"""
    body = line[len("trojan://"):]
    name = ""
    m = re.search(r"#([^#\s]+)$", body)
    if m:
        name = unquote(m.group(1).strip())
    server = port = ""
    m2 = re.search(r"@([^:/#]+):(\d+)", body)
    if m2:
        server, port = m2.group(1), int(m2.group(2))
    return {"name": name or f"{server or 'trojan'}:{port or ''}",
            "type": "trojan", "server": server, "port": port or None,
            "url": "", "supported": False}


def _parse_ssr_link(line: str) -> dict:
    """ssr://base64(host:port:...)（需本地客户端转换）。"""
    raw = line[len("ssr://"):].split("#")[0]
    server = port = ""
    name = ""
    m = re.search(r"#([^#\s]+)$", line)
    if m:
        name = unquote(m.group(1).strip())
    try:
        pad = "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(raw + pad).decode(errors="ignore")
        parts = decoded.split(":")
        if len(parts) >= 2:
            server, port = parts[0], int(parts[1])
    except Exception:  # noqa: BLE001
        pass
    return {"name": name or f"{server or 'ssr'}:{port or ''}",
            "type": "ssr", "server": server, "port": port or None,
            "url": "", "supported": False}





_QX_TYPES = {"socks5", "socks4", "http", "https", "ss", "vmess", "trojan", "ssr"}




_NEW_PROTO_TYPES = {"hysteria2", "hysteria", "tuic", "wireguard"}


def _parse_generic_link(line: str, scheme: str) -> dict:
    """新协议行：scheme://(user@)host:port(/?params)#名称（需本地客户端转换）。"""
    body = line.split("#", 1)[0]
    name = ""
    m = re.search(r"#([^#\s]+)$", line)
    if m:
        name = unquote(m.group(1).strip())
    server, port = "", None
    if "@" in body:
        hostpart = body.rsplit("@", 1)[1]
    else:
        hostpart = body.split("://", 1)[-1] if "://" in body else body
    hostpart = hostpart.split("?")[0].split("/")[0]
    hp = hostpart.split(":")
    if hp and hp[0]:
        server = hp[0]
        if len(hp) > 1 and hp[1].isdigit():
            port = int(hp[1])
    return {"name": name or f"{server or scheme}:{port or ''}",
            "type": scheme, "server": server, "port": port,
            "url": "", "supported": False}
def _parse_qx_line(line: str):
    """Quantumult X / Loon 风格行：`协议 = host:port, 节点名`（机场常见导出）。
    socks5/http 可直接用作代理 URL；trojan/vmess/ss 仅提取展示（需本地转换）。
    """
    if "=" not in line:
        return None
    typ, rest = line.split("=", 1)
    typ = typ.strip().lower()
    if typ not in _QX_TYPES:
        return None
    m = re.match(r"\s*([^:\s]+):(\d+)", rest)
    if not m:
        return None
    server, port = m.group(1), int(m.group(2))
    name = ""
    nm_field = re.search(r"\bname\s*=\s*([^,]+)", rest)
    if nm_field:
        name = unquote(nm_field.group(1).strip())
    else:
        # host:port 后第一个逗号字段即节点名（避免把 password= 等尾段当名字）
        m2 = re.search(r":\d+\s*,\s*(.*?)(?:,|\s*$)", rest)
        if m2:
            name = unquote(m2.group(1).strip())
    supported = typ in _DIRECT_TYPES
    url = f"{typ}://{server}:{port}" if supported else ""
    return {"name": name or f"{server}:{port}", "type": typ,
            "server": server, "port": port, "url": url, "supported": supported}
def extract_server_error(text: str) -> str:
    """订阅内容若为服务器错误响应（如 {"message":"token is error"}）→ 提取可读信息。"""
    if not (text or "").strip():
        return ""
    try:
        data = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(data, dict):
        return ""
    for k in ("message", "msg", "error", "detail"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
def parse_nodes_text(text: str, _json_depth: int = 0) -> list[dict]:
    """解析节点文本（Clash YAML 或每行 URL），返回节点列表。"""
    text = (text or "").strip()
    if not text:
        return []
    decoded = _maybe_base64_decode(text)
    if decoded:
        return parse_nodes_text(decoded)
    nodes: list[dict] = []
    # 1) Clash YAML：含 proxies 段（顶层或缩进）
    yaml_nodes = _parse_clash_yaml(text)
    if yaml_nodes:
        nodes.extend(yaml_nodes)
    # 2) 行格式 URL（清洗行首序号/项目符号/引号：如 "1. ss://..."、'"vmess://..."')
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^(?:[-\*]\s*)?(?:\d+[\.\)]\s*)?", "", line).strip()
        if line.startswith(('"', "'")) and len(line) >= 2 and line.endswith(line[0]):
            line = line[1:-1].strip()
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
        elif low.startswith("vmess://"):
            nodes.append(_parse_vmess_link(line))
        elif low.startswith("trojan://"):
            nodes.append(_parse_trojan_link(line))
        elif low.startswith("ssr://"):
            nodes.append(_parse_ssr_link(line))
        elif low.startswith(("hysteria2://", "hysteria://", "tuic://", "wireguard://")):
            scheme = line.split("://", 1)[0].lower()
            nodes.append(_parse_generic_link(line, scheme))
        else:
            qx = _parse_qx_line(line)
            if qx is not None:
                nodes.append(qx)
    if not nodes and _json_depth <= 2:
        # JSON 包装的订阅（如 {"status":"success","data":"base64..."}）——递归提取字符串值
        nodes = _extract_from_json(text, _json_depth)
    return nodes


def _extract_from_json(text: str, depth: int = 0) -> list[dict]:
    """从 JSON 结构递归收集含节点标记/base64 的字符串值并解析（防递归深度≤2）。"""
    if depth > 2:
        return []
    try:
        data = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, (dict, list)):
        return []
    found: list[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            s = v.strip()
            if any(m in s for m in _NODE_MARKERS):
                found.append(s)
            else:
                dec = _maybe_base64_decode(s)
                if dec:
                    found.append(dec)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, list):
            for x in v:
                _walk(x)
    _walk(data)
    out: list[dict] = []
    for s in found:
        out.extend(parse_nodes_text(s, _json_depth=depth + 1))
    return out


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
        if not (text or "").strip():
            raise ValueError("订阅返回空内容")
        return parse_nodes_text(text)
    import asyncio
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cat-trading/1.0"}
    last_err: Optional[Exception] = None
    for attempt in range(2):  # 一次重试（部分订阅服务器首连慢/校验 UA）
        try:
            resp = await asyncio.to_thread(requests.get, url, timeout=15, headers=headers)
            resp.raise_for_status()
            if not (resp.text or "").strip():
                raise ValueError("订阅返回空内容")
            return parse_nodes_text(resp.text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("[proxy] 订阅拉取失败(第%d次) %s: %s", attempt + 1, url, e)
    # 失败：向上抛具体原因（scan 端点转为用户可见 error，而非静默空列表）
    raise ValueError(f"订阅拉取失败: {last_err}" if last_err else "订阅拉取失败")
