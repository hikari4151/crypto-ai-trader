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
TOKEN_FILE = Path(settings.data_dir) / "api_token.txt"


def _load_or_create_token() -> str:
    """API 令牌：.env 配置优先；否则持久化到 data/api_token.txt（重启复用）。

    曾每次启动重新生成 → 重启后浏览器旧页面 Cookie 里的旧令牌全部 401，
    用户表现为"运行报错"（所有接口未授权）。
    """
    if settings.api_token:
        return settings.api_token
    try:
        TOKEN_FILE.parent.mkdir(exist_ok=True)
        if TOKEN_FILE.exists():
            tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if len(tok) >= 16:
                return tok
        tok = secrets.token_hex(16)
        TOKEN_FILE.write_text(tok, encoding="utf-8")
        return tok
    except Exception as e:  # noqa: BLE001
        log.warning("[main] 令牌持久化失败，使用临时令牌: %s", e)
        return secrets.token_hex(16)


# API 访问令牌：.env 配置 API_TOKEN，未配置则自动生成并持久化（同源前端经 Cookie 自动携带）
API_TOKEN = _load_or_create_token()
TOKEN_COOKIE = "zx_api_token"


class AuthMiddleware(BaseHTTPMiddleware):
    """API 鉴权：除 /api/health 外的所有 /api/* 请求必须携带 X-API-Token 头。

    令牌通过响应 Cookie 下发（SameSite=Lax、非 HttpOnly），前端 JS 读取后附加到
    请求头。跨站请求无法读取该 Cookie，也无法在无 CORS 白名单下自定义该头，形成 CSRF 防线。
    """

    async def dispatch(self, request: Request, call_next):
        import hmac
        path = request.url.path
        # CORS 预检请求（OPTIONS）直接放行，否则浏览器预检会被 401 拦截
        if request.method == "OPTIONS":
            response = await call_next(request)
            response.set_cookie(TOKEN_COOKIE, API_TOKEN, path="/", samesite="lax", httponly=False)
            return response
        if path.startswith("/api/") and path.rstrip("/") != "/api/health":
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
    # 动态策略必须在 TradingEngine 构造前恢复：引擎 __init__ 会 get_strategy(default_strategy)，
    # 而默认策略可能是 AI 动态策略；未恢复则冷启动后回测/实盘看不到任何 AI 策略
    from strategies.dynamic_store import restore_from_db
    restored = await restore_from_db(db)
    if restored:
        log.info("[main] 已恢复 %d 个动态策略（回测/实盘/AI 页共用同一注册表）", restored)
    app.state.engine = TradingEngine(db, app.state.bus)
    # 持续进化引擎独立启动（不依赖交易引擎状态）：即使交易引擎未启动，
    # 因子挖掘和策略训练仍可用演示数据运行，用户可随时查看训练进度
    try:
        await app.state.engine.evolve.start()
        log.info("[main] 持续进化引擎已启动（后台训练循环，因子挖掘15s后开始，策略DRL 30s后开始）")
    except Exception as e:  # noqa: BLE001
        log.warning("[main] 持续进化引擎启动失败（不影响主服务）: %s", e)
    # P2-4：代理探测异步预热（阻塞 socket 探测放线程池，启动即缓存，
    # 首个行情请求不再在事件循环上同步阻塞数秒）
    try:
        await settings.prewarm_proxy()
    except Exception as e:  # noqa: BLE001
        log.warning("[main] 代理预热失败: %s", e)
    if settings.trading_auto_start:
        # 自动启动失败（实盘密钥错误/网络不可达等）仅告警降级：
        # 曾直接抛异常使整个 Web UI 起不来，连修复配置的页面都不可用
        try:
            await app.state.engine.start()
        except Exception as e:  # noqa: BLE001
            log.exception("[main] 交易引擎自动启动失败，降级为停止状态（可在界面修复配置后手动启动）: %s", e)
    if settings.api_token:
        log.info("API 令牌来自 .env API_TOKEN（前 N 位 %s****）", API_TOKEN[:6])
    else:
        log.info("API 令牌已自动生成并持久化（data/api_token.txt，前 N 位 %s****）；重启保持不变，建议在 .env 固定配置", API_TOKEN[:6])
    log.info("Web 服务已启动: http://%s:%s", settings.web_host, settings.web_port)
    yield
    await app.state.engine.stop()
    # 释放 AI 长连接（loop-keyed HTTP 客户端）
    try:
        await app.state.engine.ai_client.close()
    except Exception as e:  # noqa: BLE001
        log.warning("[main] 关闭 AI 客户端失败: %s", e, exc_info=True)
    # 关闭行情全局 HTTP 客户端（模块 C 独占此接线，幂等）
    try:
        from web.api.market import close_client as _close_market
        await _close_market()
    except Exception as e:  # noqa: BLE001
        log.warning("[main] 关闭行情客户端失败: %s", e, exc_info=True)
    await db.close()


app = FastAPI(title="Crypto AI Trader", version="1.0.0", lifespan=lifespan)
# 不启用跨域 CORS：前端由本服务同源托管，无需跨域；移除通配 CORS 可阻止恶意网站读取 API。
# 若需从其它来源访问，请显式配置允许的源（勿用 *）。
app.add_middleware(AuthMiddleware)

from web.api import (  # noqa: E402
    ai_router, backtest_router, data_router, drl_router, evolve_router, exchanges_router,
    factors_router, live_router, market_router, notify_router, performance_router,
    portfolio_router, risk_router, strategy_repo_router, trading_router,
)

for r in (ai_router, exchanges_router, trading_router, backtest_router,
          portfolio_router, performance_router, risk_router, market_router, live_router,
          strategy_repo_router, factors_router, drl_router, evolve_router, data_router,
          notify_router):
    app.include_router(r)


@app.get("/api/health")
async def health():
    return {"ok": True, "engine": app.state.engine.status()}


@app.get("/")
async def index():
    # no-cache：浏览器每次 revalidate（etag/304，代价极低），确保前端页面始终是最新版——
    # 曾因浏览器沿用旧缓存页面，出现"新功能不生效/点击无反应"的用户环境问题
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
