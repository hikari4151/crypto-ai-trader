"""网络加速（应用级代理）配置 API（2026-09-24 功能：VPN 加速仅作用于本软件）。

用户提供代理/VPN URL（http/socks5，如 http://127.0.0.1:7890），网页可配置；
设置仅影响本软件出站请求（settings.resolved_proxy 被数据层/交易所/通知复用），
不影响系统其他程序。设置持久化到 data/proxy_config.json（重启自动恢复）。
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.settings import settings

log = logging.getLogger("api.proxy")
router = APIRouter(prefix="/api/proxy", tags=["proxy"])

OVERRIDE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "proxy_config.json"


class ProxyConfigIn(BaseModel):
    proxy_url: str = Field(default="", max_length=300)  # 空=清空（回退自动探测）


def _persist(url: str) -> None:
    """持久化覆盖（网页设置重启保留）。"""
    OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_FILE.write_text(json.dumps({"proxy_url": url}, ensure_ascii=False),
                             encoding="utf-8")


@router.get("")
async def get_proxy() -> dict:
    """读取当前代理配置与生效地址。"""
    resolved = settings.resolved_proxy or ""
    from net.proxy_client import find_hysteria_bin
    proxy_url = (settings.proxy_url or "").strip()
    return {
        "proxy_url": proxy_url,
        "resolved_proxy": resolved,
        "unreachable": bool(proxy_url and not resolved),
        "auto_detected": not bool(proxy_url),
        "hy_client_installed": bool(find_hysteria_bin()),
    }


_SUBSCRIPTION_HINTS = ("subscribe", "token=", "sub?")
_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h", "socks4"}


def _validate_proxy_url(url: str) -> tuple[bool, str]:
    """校验手动代理 URL：拒绝订阅链接（应走扫描节点）与不支持的协议。"""
    if not url:
        return True, ""
    low = url.lower()
    if any(h in low for h in _SUBSCRIPTION_HINTS):
        return False, "这是订阅链接（节点列表），请使用「扫描全部节点」选择节点，而非手动代理 URL"
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme not in _PROXY_SCHEMES:
        return False, "不支持的代理格式（支持 http/https/socks5/socks4）"
    return True, ""


@router.put("")
async def set_proxy(body: ProxyConfigIn) -> dict:
    """设置代理/VPN URL（仅本软件生效）；空=清空回退自动探测。"""
    url = body.proxy_url.strip()
    ok, reason = _validate_proxy_url(url)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=reason)
    settings.proxy_url = url
    settings.reset_proxy_cache()
    resolved = settings.resolved_proxy or ""
    if not resolved and url:
        log.warning("[proxy] 配置的代理 %s 不可达（回退自动探测）", url)
    _persist(url)
    return {
        "ok": True,
        "proxy_url": url,
        "resolved_proxy": resolved,
        # 2026-09-27：不可达时如实标记——此前前端拿 proxy_url 兜底显示"生效"，
        # 实际流量在直连，用户看到"加速中"却完全没加速。
        "unreachable": bool(url and not resolved),
        "auto_detected": not bool(url),
        "note": "仅作用于本软件出站请求（交易所/K线/通知/AI）；不影响系统其他程序",
    }


class ProxyNodesIn(BaseModel):
    source: str = Field(default="", max_length=20000)  # 订阅 URL 或节点文本（Clash YAML/每行 URL）


@router.post("/nodes/scan")
async def scan_proxy_nodes(body: ProxyNodesIn) -> dict:
    """扫描节点：订阅 URL 或粘贴文本 → 解析全部节点（名称/类型/地址/可达性）。

    supported=True 的节点可直接选作当前代理（http/socks5）；
    ss/vmess/trojan 列出但标记不支持直用（需本地客户端转换）。
    """
    from net.proxy_nodes import parse_nodes_text, probe_node, resolve_subscription

    src = body.source.strip()
    if not src:
        return {"ok": False, "error": "source 为空"}
    low = src.lower()
    try:
        if low.startswith(("http://", "https://")):
            nodes = await resolve_subscription(src)
        else:
            nodes = parse_nodes_text(src)
    except Exception as e:  # noqa: BLE001  # 订阅拉取失败（超时/网络）→ 具体原因可见
        log.warning("[proxy] 扫描失败 %s: %s", src[:80], e)
        return {"ok": False, "error": str(e), "nodes": [], "count": 0}
    if not nodes:
        from net.proxy_nodes import extract_server_error
        server_err = extract_server_error(src)
        if server_err:
            return {"ok": True, "nodes": [], "count": 0,
                    "error": f"订阅服务器返回错误：{server_err}——订阅链接 token 可能无效/过期，请到机场面板重新复制订阅链接"}
        return {"ok": True, "nodes": [], "count": 0,
                "error": "未解析到节点：请确认是 Clash YAML / 每行 socks5:// http:// ss:// 格式；若订阅内容为 base64 编码，请先解码后粘贴"}
    for n in nodes:
        n["reachable"], n["delay_ms"] = probe_node(n.get("url", ""))
    return {"ok": True, "nodes": nodes, "count": len(nodes)}


class ProxySelectIn(BaseModel):
    node: dict = Field(default_factory=dict)


@router.post("/select")
async def select_proxy_node(body: ProxySelectIn) -> dict:
    """选择节点：socks5/http 直接设为代理；hysteria2/hysteria 内置客户端本地启动。"""
    from net.proxy_client import select_node, stop_local_proxy

    node = body.node or {}
    if not node.get("type"):
        return {"ok": False, "error": "node 为空"}
    scheme = str(node.get("type", "")).lower()
    if scheme in ("hysteria2", "hysteria"):
        stop_local_proxy()  # 切换节点前停旧客户端
    ok, url, err = select_node(node)
    if not ok:
        return {"ok": False, "error": err, "node": node.get("name", "")}
    # 设置并持久化（本地 socks5 URL / 直连 URL）
    v_ok, v_reason = _validate_proxy_url(url)
    if not v_ok:
        return {"ok": False, "error": v_reason}
    settings.proxy_url = url
    settings.reset_proxy_cache()
    _persist(url)
    resolved = settings.resolved_proxy or ""
    return {"ok": True, "proxy_url": url,
            "resolved_proxy": resolved,
            "unreachable": bool(url and not resolved),
            "selector": "local_client" if scheme in ("hysteria2", "hysteria") else "direct",
            "note": "仅作用于本软件出站请求"}


@router.post("/client/download")
async def download_hysteria_client_api():
    """一键下载官方 hysteria 客户端二进制到 bin/（hysteria2 节点直用）。"""
    import asyncio
    from net.proxy_client import download_hysteria_client, find_hysteria_bin

    if find_hysteria_bin():
        return {"ok": True, "installed": True, "bin": find_hysteria_bin()}
    ok, dest, err = await asyncio.to_thread(download_hysteria_client)
    if not ok:
        return {"ok": False, "error": err}
    return {"ok": True, "installed": True, "bin": dest,
            "note": "hysteria 客户端已下载，可直接选择 hysteria2 节点"}
