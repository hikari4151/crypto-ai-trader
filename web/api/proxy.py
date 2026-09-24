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
    return {
        "proxy_url": (settings.proxy_url or "").strip(),
        "resolved_proxy": resolved,
        "auto_detected": not bool((settings.proxy_url or "").strip()),
    }


@router.put("")
async def set_proxy(body: ProxyConfigIn) -> dict:
    """设置代理/VPN URL（仅本软件生效）；空=清空回退自动探测。"""
    url = body.proxy_url.strip()
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
        "auto_detected": not bool(url),
        "note": "仅作用于本软件出站请求（交易所/K线/通知/AI）；不影响系统其他程序",
    }
