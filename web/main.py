"""FastAPI 主应用：初始化数据库/引擎、挂载路由、托管前端静态文件。"""
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.logging_config import setup_logging
from engine.trading_engine import TradingEngine

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

# API 访问令牌：.env 配置 API_TOKEN，未配置则启动时自动生成（同源前端经 Cookie 自动携带）
API_TOKEN = settings.api_token or secrets.token_hex(16)
TOKEN_COOKIE = "zx_api_token"


class AuthMiddleware(BaseHTTPMiddleware):
    """API 鉴权：除 /api/health 外的所有 /api/* 请求必须携带 X-API-Token 头。

    令牌通过响应 Cookie 下发（SameSite=Lax、非 HttpOnly），前端 JS 读取后附加到
    请求头。跨站请求无法读取该 Cookie，也无法在无 CORS 白名单下自定义该头，形成 CSRF 防线。
    """

    async def dispatch(self, request: Request, call_next):
        import hmac
        path = request.url.path
        if path.startswith("/api/") and path != "/api/health":
            # 恒定时间比较（防时序侧信道）；header 缺失时直接拒绝
            if not hmac.compare_digest(request.headers.get("X-API-Token", ""), API_TOKEN):
                return JSONResponse(status_code=401, content={"detail": "未授权：缺少或错误的 X-API-Token"})
        response = await call_next(request)
        # 下发给同源前端（页面加载后即可读到）
        response.set_cookie(TOKEN_COOKIE, API_TOKEN, path="/", samesite="lax", httponly=False)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level, settings.log_dir)
    db = Database(settings.database_url)
    await db.init()
    app.state.db = db
    app.state.bus = EventBus()
    app.state.engine = TradingEngine(db, app.state.bus)
    if settings.trading_auto_start:
        await app.state.engine.start()
    if settings.api_token:
        log.info("API 令牌来自 .env API_TOKEN（前 N 位 %s****）", API_TOKEN[:6])
    else:
        log.warning("未配置 API_TOKEN，已自动生成（前 N 位 %s****）；建议在 .env 中固定配置", API_TOKEN[:6])
    log.info("Web 服务已启动: http://%s:%s", settings.web_host, settings.web_port)
    yield
    await app.state.engine.stop()
    # 释放 AI 长连接（loop-keyed HTTP 客户端）
    try:
        await app.state.engine.ai_client.close()
    except Exception:  # noqa: BLE001
        pass
    await db.close()


app = FastAPI(title="Crypto AI Trader", version="1.0.0", lifespan=lifespan)
# 不启用跨域 CORS：前端由本服务同源托管，无需跨域；移除通配 CORS 可阻止恶意网站读取 API。
# 若需从其它来源访问，请显式配置允许的源（勿用 *）。
app.add_middleware(AuthMiddleware)

from web.api import (  # noqa: E402
    ai_router, backtest_router, drl_router, exchanges_router, factors_router, live_router,
    market_router, performance_router, portfolio_router, risk_router, strategy_repo_router,
    trading_router,
)

for r in (ai_router, exchanges_router, trading_router, backtest_router,
          portfolio_router, performance_router, risk_router, market_router, live_router,
          strategy_repo_router, factors_router, drl_router):
    app.include_router(r)


@app.get("/api/health")
async def health():
    return {"ok": True, "engine": app.state.engine.status()}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
