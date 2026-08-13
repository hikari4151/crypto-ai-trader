"""策略参数动态优化：AI 依据近期表现与市场特征给出新参数，程序热更新。"""
import json
import logging
from typing import Any, Optional

from core.database import Database
from strategies.base import Strategy
from .client import AIClient, AICallError, AINotConfigured
from .prompts import price_action_optimize_messages

log = logging.getLogger(__name__)


class ParamOptimizer:
    def __init__(self, client: AIClient, db: Database) -> None:
        self._client = client
        self._db = db

    async def optimize_price_action(self, strategy: Strategy, performance: dict, snap: dict,
                                    focus: str = "价格行为与关键位",
                                    backtests: Optional[list] = None) -> Optional[dict]:
        """全自动优化：重点针对关键位识别与价格行为确认，返回 {params, reason, focus}。"""
        try:
            result = await self._client.chat_json_validated(
                price_action_optimize_messages(strategy, performance, snap, focus, backtests or []),
                feature="param_optimize",
                ctx={"param_schema": strategy.param_schema, "performance": performance, "snap": snap})
        except (AINotConfigured, AICallError) as e:
            log.warning("[ai] 价格行为优化跳过: %s", e)
            return None

        new_params = result.get("params") or {}
        applied = strategy.update_params(new_params)
        async with self._db.session() as s:
            from core.database import OptimizationLog
            s.add(OptimizationLog(
                kind="auto_optimize",
                summary=f"策略[{strategy.name}] 全自动优化（{focus}）",
                suggestion=json.dumps({"reason": result.get("reason", ""), "focus": result.get("focus", "")}, ensure_ascii=False),
                params_json=json.dumps(applied, ensure_ascii=False),
            ))
            await s.commit()
        log.info("[ai] 策略 %s 已按关键位/价格行为优化: %s", strategy.name, applied)
        return {"params": applied, "reason": result.get("reason", ""), "focus": result.get("focus", "")}