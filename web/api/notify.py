"""通知推送配置 API：Telegram / 钉钉 / 邮件。

- GET  /api/notify/config  读取通知渠道配置
- POST /api/notify/config  保存配置
- POST /api/notify/test    向已启用渠道发送测试消息
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.notify import get_notify_config, save_notify_config, test_channels
from web.deps import get_db

router = APIRouter(prefix="/api/notify", tags=["notify"])


class NotifyConfigIn(BaseModel):
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    dingtalk_webhook: str = ""
    dingtalk_secret: str = ""
    dingtalk_enabled: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 465
    email_username: str = ""
    email_password: str = ""
    email_to: str = ""
    email_use_ssl: bool = True
    email_enabled: bool = False


@router.get("/config")
async def read_config(db=Depends(get_db)):
    cfg = await get_notify_config(db)
    return {
        "telegram": cfg.get("telegram") or {},
        "dingtalk": cfg.get("dingtalk") or {},
        "email": cfg.get("email") or {},
    }


@router.post("/config")
async def write_config(body: NotifyConfigIn, db=Depends(get_db)):
    cfg = {
        "telegram": {
            "bot_token": body.telegram_bot_token.strip(),
            "chat_id": body.telegram_chat_id.strip(),
            "enabled": body.telegram_enabled,
        },
        "dingtalk": {
            "webhook": body.dingtalk_webhook.strip(),
            "secret": body.dingtalk_secret.strip(),
            "enabled": body.dingtalk_enabled,
        },
        "email": {
            "smtp_host": body.email_smtp_host.strip(),
            "smtp_port": body.email_smtp_port,
            "username": body.email_username.strip(),
            "password": body.email_password,
            "to": body.email_to.strip(),
            "use_ssl": body.email_use_ssl,
            "enabled": body.email_enabled,
        },
    }
    await save_notify_config(db, cfg)
    return {"ok": True}


@router.post("/test")
async def send_test(db=Depends(get_db)):
    results = await test_channels(db)
    return {"ok": True, "results": results}
