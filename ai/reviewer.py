"""反思与复盘：每日/每周把历史交易与盈亏发给 AI，获得改进建议并记录日志。"""
import json
import logging
from typing import Optional

from core.database import Database, _run_on_main
from .client import AIClient, AICallError, AINotConfigured
from .prompts import review_messages

log = logging.getLogger(__name__)


class TradeReviewer:
    def __init__(self, client: AIClient, db: Database) -> None:
        self._client = client
        self._db = db

    async def _previous_review(self) -> Optional[dict]:
        """读取上一次复盘的行动项（kind=review 最新一条），供本轮跟进核验。

        复盘不再是孤立的一次性报告：上轮提出的行动项是否被执行、是否有效，
        本轮 AI 必须对照核验，形成跨轮反思闭环。
        """
        try:
            from core.database import OptimizationLog
            from sqlalchemy import select

            async def _query() -> Optional[dict]:
                async with self._db.session() as s:
                    row = (await s.execute(
                        select(OptimizationLog)
                        .where(OptimizationLog.kind == "review")
                        .order_by(OptimizationLog.id.desc()).limit(1))).scalar_one_or_none()
                    if row is None:
                        return None
                    sug = json.loads(row.suggestion or "{}") if isinstance(row.suggestion, str) else {}
                    action_items = sug.get("action_items") if isinstance(sug, dict) else None
                    if not isinstance(action_items, list) or not action_items:
                        return None
                    return {"ts": row.ts.isoformat() if getattr(row, "ts", None) else None,
                            "score": sug.get("score"),
                            "action_items": [str(a) for a in action_items[:6]]}
            # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
            return await _run_on_main(self._db, _query)
        except Exception:  # noqa: BLE001
            return None

    async def review(self, trades: list[dict], summary: dict) -> Optional[dict]:
        try:
            previous_review = await self._previous_review()
            result = await self._client.chat_json_validated(
                review_messages(trades, summary, previous_review=previous_review),
                feature="trade_review",
                ctx={"summary": summary, "trades": trades})
        except (AINotConfigured, AICallError) as e:
            log.warning("[ai] 复盘跳过: %s", e)
            return None

        async def _persist_review() -> None:
            from core.database import OptimizationLog
            async with self._db.session() as s:
                s.add(OptimizationLog(
                    kind="review",
                    summary=result.get("summary", "交易复盘"),
                    suggestion=json.dumps({
                        "score": result.get("score"),
                        "strengths": result.get("strengths", []),
                        "problems": result.get("problems", []),
                        "action_items": result.get("action_items", []),
                    }, ensure_ascii=False),
                ))
                await s.commit()
        # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
        await _run_on_main(self._db, _persist_review)
        log.info("[ai] 复盘完成: %s", result.get("summary", ""))
        return result