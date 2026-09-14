"""SQLite（默认）/ PostgreSQL（可选）异步 ORM + 加密 KV 存储。"""
import asyncio
import json
import logging
import weakref
from datetime import datetime
from typing import Any, Callable, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .security import decrypt, encrypt

log = logging.getLogger(__name__)

# 未关闭的 Database 实例（弱引用集合）：测试/退出路径可据此清理未显式 close 的
# 引擎——未 dispose 的 aiosqlite 连接会在事件循环关闭后让 worker 线程崩溃
_OPEN_DATABASES: "weakref.WeakSet[Database]" = weakref.WeakSet()


async def _run_on_main(db, coro_factory: Callable[[], Any]) -> Any:
    """兼容辅助：db 提供 on_main（真实 Database）则经主循环桥接（worker 跨 loop
    安全）；未提供（测试替身 _FakeDb 等）则直接执行——行为与旧版一致。
    """
    runner = getattr(db, "on_main", None)
    if runner is None:
        return await coro_factory()
    return await runner(coro_factory)


class Base(DeclarativeBase):
    pass


class KV(Base):
    """加密/普通键值存储：交易所与 AI 密钥加密存放于此。"""
    __tablename__ = "kv"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    is_secret: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True, default=func.now())
    exchange: Mapped[str] = mapped_column(String(32), default="")
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))                      # buy / sell
    price: Mapped[float] = mapped_column(Float)
    qty: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    fee_currency: Mapped[str] = mapped_column(String(16), default="USDT")
    pnl: Mapped[float] = mapped_column(Float, default=0.0)            # 平仓盈亏
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)   # 已实现盈亏（累计口径）
    strategy: Mapped[str] = mapped_column(String(64), default="")
    order_id: Mapped[str] = mapped_column(String(64), default="")
    reason: Mapped[str] = mapped_column(String(255), default="")

class AiStrategy(Base):
    """AI 设计的策略（动态注册，重启后自动恢复）。"""
    __tablename__ = "ai_strategies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    spec_json: Mapped[str] = mapped_column(Text, default="{}")

class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True, default=func.now())
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float, default=0.0)


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    equity_curve_json: Mapped[str] = mapped_column(Text, default="[]")
    trades_json: Mapped[str] = mapped_column(Text, default="[]")


class Analysis(Base):
    """AI 市场解读记录。"""
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    symbol: Mapped[str] = mapped_column(String(32), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    meta_json: Mapped[str] = mapped_column(Text, default="{}")


class OptimizationLog(Base):
    """AI 优化/复盘日志。"""
    __tablename__ = "optimization_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="")         # optimize / review
    summary: Mapped[str] = mapped_column(Text, default="")
    suggestion: Mapped[str] = mapped_column(Text, default="")
    params_json: Mapped[str] = mapped_column(Text, default="{}")


class EvolveRound(Base):
    """持续进化引擎训练轮次落库（P2-13）：每轮训练结果归档，支撑 fitness 曲线回看。

    selected_factors 为因子挖掘选中的因子表达式列表 JSON；status 记录该轮结局
    （ok / oos_rejected / cross_rejected / rollback / demo_blocked）。
    """
    __tablename__ = "evolve_rounds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    model: Mapped[str] = mapped_column(String(32), index=True)        # factor_miner / strategy_drl / meta_controller
    symbol: Mapped[str] = mapped_column(String(32), default="")
    timeframe: Mapped[str] = mapped_column(String(16), default="")
    data_source: Mapped[str] = mapped_column(String(16), default="exchange")
    round_no: Mapped[int] = mapped_column(Integer, default=0)
    fitness: Mapped[float] = mapped_column(Float, default=0.0)
    oos_ret: Mapped[float] = mapped_column(Float, default=0.0)
    decay: Mapped[float] = mapped_column(Float, default=0.0)
    position_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    selected_factors: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(32), default="ok")
    # Task 5：人工回退等操作审计明细（目标/源版本、原因、运行时结果）JSON
    audit_json: Mapped[str] = mapped_column(Text, default="{}")


class Database:
    def __init__(self, url: str) -> None:
        # 仅 SQLite 需要 busy timeout（aiosqlite 的 PRAGMA）；PG/其他方言忽略
        connect_args = {"timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_async_engine(url, echo=False, future=True, connect_args=connect_args)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        # 未关闭实例登记（弱引用）：供测试/退出路径清理未显式 close 的引擎——
        # 未 dispose 的 aiosqlite 连接在事件循环关闭后会让 worker 线程崩溃
        _OPEN_DATABASES.add(self)
        # run26 R3：主事件循环引用——SQLAlchemy 异步引擎连接池绑定首次使用它的
        # 事件循环；AI 后台 worker 线程（web/api/ai.py _run_ai_task_background）在
        # 独立事件循环里直接 await db.kv_get()/db.session() 会跨 loop 复用连接池
        # → 间歇性 RuntimeError。init() 在主循环（web lifespan）调用，捕获该循环，
        # 之后所有公开异步方法经 on_main() 自动调度回主循环执行。
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def _is_cross_loop(self) -> bool:
        """当前是否运行在非主循环（worker 线程的独立事件循环）。"""
        if self._main_loop is None:
            return False
        try:
            return asyncio.get_running_loop() is not self._main_loop
        except RuntimeError:
            # 无运行中事件循环（同步上下文）：无需桥接
            return False

    async def on_main(self, coro_factory: Callable[[], Any]) -> Any:
        """在绑定主循环上执行 coro_factory()（worker 跨 loop 安全）。

        已在主循环/无绑定 → 直接执行；否则 run_coroutine_threadsafe 调度回主
        循环并阻塞等待结果（worker 线程语义；timeout 语义与 web/api/backtest.py
        的 _run_on_loop 一致：超时后协程仍会迟到完成）。
        """
        if not self._is_cross_loop():
            return await coro_factory()
        fut = asyncio.run_coroutine_threadsafe(coro_factory(), self._main_loop)
        # 取走迟到协程的最终异常（超时后无人 result()，否则噪音日志）；
        # 主循环关闭导致 future 取消时 f.exception() 会抛 CancelledError，需吞掉
        def _drain_cancelled(f) -> None:
            try:
                f.exception()
            except Exception:  # noqa: BLE001 (CancelledError 等)
                pass
        fut.add_done_callback(_drain_cancelled)
        return fut.result(timeout=30.0)

    async def init(self) -> None:
        # run26 R3：主循环引用在 init 时捕获（init 由 web lifespan 在主循环调用；
        # 测试中 asyncio.run(db.init()) 亦捕获测试循环，on_main 桥接自然失效无害）
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
        # PRAGMA 仅 SQLite 方言：WAL 允许读写并发，synchronous=NORMAL 降低 fsync 开销。
        # 曾无条件执行，切换文档化的 postgresql+asyncpg 后 init() 直接语法错误
        if self.engine.url.drivername.startswith("sqlite"):
            async with self.engine.begin() as conn:
                await conn.execute(text("PRAGMA journal_mode=WAL"))
                await conn.execute(text("PRAGMA synchronous=NORMAL"))
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # P2：evolve_rounds 查询热路径是 WHERE model=? ORDER BY id/round_no DESC
        # （fitness 曲线 + 重启 streak 恢复），单列索引走不了复合覆盖。
        # 已存在的表 create_all 不会补索引，用幂等 DDL 显式建（SQLite 语义）。
        if self.engine.url.drivername.startswith("sqlite"):
            async with self.engine.begin() as conn:
                await conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_evolve_rounds_model_id "
                    "ON evolve_rounds (model, id DESC)"))
                await conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_evolve_rounds_model_round_no "
                    "ON evolve_rounds (model, round_no DESC)"))
                # Task 5：既有库缺 audit_json 列时幂等补列（新库由 create_all 建全）
                cols = (await conn.execute(text("PRAGMA table_info(evolve_rounds)"))).fetchall()
                if cols and "audit_json" not in {c[1] for c in cols}:
                    await conn.execute(text(
                        "ALTER TABLE evolve_rounds ADD COLUMN audit_json TEXT NOT NULL DEFAULT '{}'"))
        log.info("数据库初始化完成: %s", self.engine.url.render_as_string(hide_password=True))

    async def close(self) -> None:
        """关闭数据库引擎并回收全部连接。

        aiosqlite 的 `_connection_worker_thread` 在连接关闭后仍会处理队列剩余项，
        并通过 `future.get_loop().call_soon_threadsafe(...)` 回调；若此时事件循环
        已关闭，回调抛 RuntimeError('Event loop is closed') 使 worker 线程崩溃
        （表现为 "Exception in thread ..._connection_worker_thread"，且会掩盖真实错误、
        丢失迟到落库）。dispose 会等连接回收（含 worker 队列排空）后再返回，因此
        必须在事件循环关闭前 await 本方法——测试与 web lifespan 均应显式调用。
        """
        await self.engine.dispose()
        _OPEN_DATABASES.discard(self)

    def session(self) -> AsyncSession:
        return self.session_factory()

    async def update_trade_pnl(self, trade_id: int, pnl: float) -> None:
        """回写成交记录的平仓盈亏（卖出成交后由引擎调用）。"""
        async with self.session() as s:
            row = await s.get(Trade, trade_id)
            if row is not None:
                row.pnl = pnl
                await s.commit()

    # ---------- KV（加密） ----------
    async def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return await self.on_main(lambda: self._kv_get(key, default))

    async def _kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        async with self.session() as s:
            row = (await s.execute(select(KV).where(KV.key == key))).scalar_one_or_none()
            return row.value if row else default

    async def kv_get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return await self.on_main(lambda: self._kv_get_secret(key, default))

    async def _kv_get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        async with self.session() as s:
            row = (await s.execute(select(KV).where(KV.key == key))).scalar_one_or_none()
            return decrypt(row.value) if row and row.value else default

    async def kv_set(self, key: str, value: str, is_secret: bool = False) -> None:
        await self.on_main(lambda: self._kv_set(key, value, is_secret))

    async def _kv_set(self, key: str, value: str, is_secret: bool = False) -> None:
        # 原子 UPSERT：曾先 SELECT 后 INSERT/UPDATE 两段式，并发写同 key
        # （引擎 check 与 /api/risk/rules 同拍 seeding）可能主键冲突 IntegrityError
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stored = encrypt(value) if is_secret else value
        stmt = sqlite_insert(KV).values(key=key, value=stored, is_secret=is_secret)
        stmt = stmt.on_conflict_do_update(
            index_elements=[KV.key],
            set_={"value": stored, "is_secret": is_secret},
        )
        async with self.session() as s:
            await s.execute(stmt)
            await s.commit()

    async def kv_json_get(self, key: str, default: Any = None) -> Any:
        raw = await self.kv_get(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    async def kv_json_set(self, key: str, obj: Any) -> None:
        await self.kv_set(key, json.dumps(obj, ensure_ascii=False))