"""事件通知推送：Telegram / 钉钉 / 邮件。

- 配置存 KV（notify_config），Web 设置页可视化配置
- notify(db, title, message)：fire-and-forget，永不抛错、绝不阻塞调用方
- 每通道独立 try/except：一个渠道失败不影响其他渠道
- 消息防抖：同标题消息 60s 内只发一次（风控连续触发场景）
"""
import asyncio
import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Optional

log = logging.getLogger(__name__)

_DEBOUNCE: dict[str, float] = {}
_DEBOUNCE_SEC = 60.0


async def get_notify_config(db) -> dict:
    """读取通知配置（DB 不可用时返回空配置，绝不抛错）。"""
    try:
        cfg = await db.kv_json_get("notify_config") or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return cfg


async def save_notify_config(db, cfg: dict) -> None:
    await db.kv_json_set("notify_config", cfg)


async def notify(db, title: str, message: str, force: bool = False) -> None:
    """向所有已启用渠道推送通知（fire-and-forget，安全可重入）。

    在没有运行事件循环的上下文（同步线程）调用时静默丢弃并记录。
    """
    if not force:
        last = _DEBOUNCE.get(title, 0)
        if time.time() - last < _DEBOUNCE_SEC:
            return
        _DEBOUNCE[title] = time.time()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.debug("[notify] 无事件循环，丢弃通知: %s", title)
        return
    loop.create_task(_send_all(db, title, message))


async def _send_all(db, title: str, message: str) -> None:
    cfg = await get_notify_config(db)
    text = f"【Crypto AI Trader】{title}\n{message}"
    results = await asyncio.gather(
        _send_telegram(cfg.get("telegram") or {}, text),
        _send_dingtalk(cfg.get("dingtalk") or {}, title, message),
        _send_email(cfg.get("email") or {}, title, message),
        return_exceptions=True,
    )
    sent = sum(1 for r in results if r is True)
    for r in results:
        if isinstance(r, Exception):
            log.warning("[notify] 渠道发送失败: %s", r)


async def _send_telegram(cfg: dict, text: str) -> bool:
    """Telegram Bot API（大陆网络需代理：自动复用 settings.resolved_proxy）。"""
    if not cfg.get("enabled") or not cfg.get("bot_token") or not cfg.get("chat_id"):
        return False
    import httpx
    from config.settings import settings
    url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
    proxy = settings.resolved_proxy or None
    async with httpx.AsyncClient(timeout=15.0, proxy=proxy) as client:
        r = await client.post(url, json={"chat_id": cfg["chat_id"], "text": text})
        r.raise_for_status()
    return True


def _ding_sign(secret: str, ts: int) -> str:
    """钉钉加签：HMAC-SHA256(timestamp\\nsecret) → base64 → urlencode。"""
    string_to_sign = f"{ts}\n{secret}"
    hmac_code = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"),
                         digestmod=hashlib.sha256).digest()
    return urllib.parse.quote_plus(base64.b64encode(hmac_code).decode("utf-8"))


async def _send_dingtalk(cfg: dict, title: str, message: str) -> bool:
    """钉钉群机器人 webhook（markdown 消息；配置了 secret 自动加签）。"""
    if not cfg.get("enabled") or not cfg.get("webhook"):
        return False
    import httpx
    url: str = cfg["webhook"]
    if cfg.get("secret"):
        ts = int(time.time() * 1000)
        url += f"&timestamp={ts}&sign={_ding_sign(cfg['secret'], ts)}"
    payload = {"msgtype": "markdown",
               "markdown": {"title": title,
                            "text": f"### {title}\n\n{message}"}}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") not in (0, None):
            raise RuntimeError(f"钉钉返回错误: {data}")
    return True


async def _send_email(cfg: dict, title: str, message: str) -> bool:
    """SMTP 邮件（阻塞调用放线程池）。"""
    if not cfg.get("enabled") or not cfg.get("smtp_host") or not cfg.get("to"):
        return False
    return await asyncio.to_thread(_send_email_sync, cfg, title, message)


def _send_email_sync(cfg: dict, title: str, message: str) -> bool:
    import smtplib
    from email.header import Header
    from email.mime.text import MIMEText
    host = cfg["smtp_host"]
    port = int(cfg.get("smtp_port") or 465)
    user = cfg.get("username") or ""
    password = cfg.get("password") or ""
    to_list = [x.strip() for x in str(cfg["to"]).replace(",", ";").split(";") if x.strip()]
    msg = MIMEText(message, "plain", "utf-8")
    msg["Subject"] = Header(f"[Crypto AI Trader] {title}", "utf-8")
    msg["From"] = user
    msg["To"] = ";".join(to_list)
    if cfg.get("use_ssl", True) or port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=15) as srv:
            if user:
                srv.login(user, password)
            srv.sendmail(user, to_list, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=15) as srv:
            srv.starttls()
            if user:
                srv.login(user, password)
            srv.sendmail(user, to_list, msg.as_string())
    return True


async def test_channels(db, cfg: Optional[dict] = None) -> dict:
    """测试各渠道连通性，返回 {telegram: ok/error, dingtalk: ..., email: ...}。"""
    if cfg is None:
        cfg = await get_notify_config(db)
    text = "【Crypto AI Trader】测试通知\n通知渠道配置成功。"
    out: dict[str, Any] = {}
    for name, coro in (("telegram", _send_telegram(cfg.get("telegram") or {}, text)),
                       ("dingtalk", _send_dingtalk(cfg.get("dingtalk") or {}, "测试通知", "通知渠道配置成功。")),
                       ("email", _send_email(cfg.get("email") or {}, "测试通知", "通知渠道配置成功。"))):
        try:
            ok = await coro
            out[name] = "ok" if ok else "未启用"
        except Exception as e:  # noqa: BLE001
            out[name] = f"失败: {e}"
    return out
