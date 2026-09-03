"""历史战绩与绩效看板。"""
from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select

from core.database import Database, EquitySnapshot, Trade
from web.deps import get_db

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/summary")
async def summary(db: Database = Depends(get_db)):
    async with db.session() as s:
        # 用 SQL 聚合替代全表加载，交易量大时大幅提速
        total_row = (await s.execute(
            select(func.count(Trade.id), func.coalesce(func.sum(case((Trade.side == "sell", 1), else_=0)), 0)))).one()
        total_trades, closed_trades = total_row[0], total_row[1] or 0

        # 平仓盈亏聚合（CASE WHEN 兼容 SQLite/PostgreSQL）
        win_sum = func.coalesce(func.sum(case((Trade.pnl > 0, Trade.pnl), else_=0.0)), 0.0)
        loss_sum = func.coalesce(func.sum(case((Trade.pnl < 0, -Trade.pnl), else_=0.0)), 0.0)
        win_cnt = func.coalesce(func.sum(case((Trade.pnl > 0, 1), else_=0)), 0)
        pnl_row = (await s.execute(
            select(win_sum, loss_sum, win_cnt).where(Trade.side == "sell"))).one()
        gross_profit, gross_loss, win_count = pnl_row[0], pnl_row[1], pnl_row[2] or 0

        total_pnl = gross_profit - gross_loss
        avg_win = gross_profit / win_count if win_count else 0.0
        loss_count = max(closed_trades - win_count, 0)
        avg_loss = gross_loss / loss_count if loss_count else 0.0

        equity_row = (await s.execute(
            select(EquitySnapshot).order_by(EquitySnapshot.ts.desc()).limit(1))).scalar_one_or_none()
        return {
            "total_trades": total_trades,
            "closed_trades": closed_trades,
            "win_rate": round(win_count / closed_trades, 4) if closed_trades else 0.0,
            "total_pnl": round(total_pnl, 4),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "latest_equity": equity_row.equity if equity_row else 0.0,
        }


@router.get("/trades")
async def trades(limit: int = 200, db: Database = Depends(get_db)):
    async with db.session() as s:
        rows = (await s.execute(select(Trade).order_by(Trade.ts.desc()).limit(limit))).scalars().all()
        return [{"id": t.id, "ts": t.ts.isoformat(), "symbol": t.symbol, "side": t.side,
                 "price": t.price, "qty": t.qty, "fee": t.fee, "pnl": t.pnl,
                 "strategy": t.strategy, "reason": t.reason} for t in rows]


@router.get("/equity")
async def equity(limit: int = 500, db: Database = Depends(get_db)):
    async with db.session() as s:
        # 先取最新的 N 条（降序），再升序返回，保证图表显示最新数据
        # 注意：ORDER BY ts ASC LIMIT 会取最旧的 N 条，此处用子查询方式取最新 N 条升序
        rows = (await s.execute(
            select(EquitySnapshot).order_by(EquitySnapshot.ts.desc()).limit(limit))).scalars().all()
        rows.reverse()
        return [{"ts": r.ts.isoformat(), "equity": r.equity} for r in rows]