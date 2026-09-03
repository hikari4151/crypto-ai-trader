"""SQLite（默认）/ PostgreSQL（可选）异步 ORM + 加密 KV 存储。"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .security import decrypt, encrypt

log = logging.getLogger(__name__)


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


class Database:
    def __init__(self, url: str) -> None:
        # 仅 SQLite 需要 busy timeout（aiosqlite 的 PRAGMA）；PG/其他方言忽略
        connect_args = {"timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_async_engine(url, echo=False, future=True, connect_args=connect_args)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        # PRAGMA 仅 SQLite 方言：WAL 允许读写并发，synchronous=NORMAL 降低 fsync 开销。
        # 曾无条件执行，切换文档化的 postgresql+asyncpg 后 init() 直接语法错误
        if self.engine.url.drivername.startswith("sqlite"):
            async with self.engine.begin() as conn:
                await conn.execute(text("PRAGMA journal_mode=WAL"))
                await conn.execute(text("PRAGMA synchronous=NORMAL"))
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("数据库初始化完成: %s", self.engine.url.render_as_string(hide_password=True))

    async def close(self) -> None:
        await self.engine.dispose()

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
        async with self.session() as s:
            row = (await s.execute(select(KV).where(KV.key == key))).scalar_one_or_none()
            return row.value if row else default

    async def kv_get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        async with self.session() as s:
            row = (await s.execute(select(KV).where(KV.key == key))).scalar_one_or_none()
            return decrypt(row.value) if row and row.value else default

    async def kv_set(self, key: str, value: str, is_secret: bool = False) -> None:
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