"""反思与复盘：每日/每周把历史交易与盈亏发给 AI，获得改进建议并记录日志。"""
import json
import logging
from typing import Any, Optional

from core.database import Database
from .client import AIClient, AICallError, AINotConfigured
from .prompts import review_messages

log = logging.getLogger(__name__)


class TradeReviewer:
    def __init__(self, client: AIClient, db: Database) -> None:
        self._client = client
        self._db = db

    async def review(self, trades: list[dict], summary: dict) -> Optional[dict]:
        try:
            result = await self._client.chat_json_validated(
                review_messages(trades, summary), feature="trade_review",
                ctx={"summary": summary, "trades": trades})
        except (AINotConfigured, AICallError) as e:
            log.warning("[ai] 复盘跳过: %s", e)
            return None
        async with self._db.session() as s:
            from core.database import OptimizationLog
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
        log.info("[ai] 复盘完成: %s", result.get("summary", ""))
        return result