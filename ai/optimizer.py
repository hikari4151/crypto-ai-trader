"""策略参数动态优化：AI 依据近期表现与市场特征给出新参数，程序热更新。"""
import json
import logging
from typing import Optional

from core.database import Database, _run_on_main
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
                                    backtests: Optional[list] = None,
                                    apply: bool = True) -> Optional[dict]:
        """全自动优化：重点针对关键位识别与价格行为确认，返回 {params, reason, focus}。

        apply=False 时只返回参数不应用（由调用方持策略锁应用——
        后台 worker 线程直接 update_params 会绕过引擎的策略锁，与 K 线处理竞态）。
        """
        try:
            result = await self._client.chat_json_validated(
                price_action_optimize_messages(strategy, performance, snap, focus, backtests or []),
                feature="param_optimize",
                ctx={"param_schema": strategy.param_schema, "performance": performance, "snap": snap})
        except (AINotConfigured, AICallError) as e:
            log.warning("[ai] 价格行为优化跳过: %s", e)
            return None

        new_params = result.get("params") or {}
        # apply=True：直接热更新策略参数（旧行为）；apply=False：只返回 AI 建议参数，
        # 由调用方持策略锁应用（曾错误返回 dict(strategy.params) 旧参数快照，
        # 导致 AI 优化建议从不生效——仅写 OptimizationLog）
        applied = strategy.update_params(new_params) if apply else new_params

        async def _persist_optimize() -> None:
            from core.database import OptimizationLog
            async with self._db.session() as s:
                s.add(OptimizationLog(
                    kind="auto_optimize",
                    summary=f"策略[{strategy.name}] 全自动优化（{focus}）",
                    suggestion=json.dumps({"reason": result.get("reason", ""), "focus": result.get("focus", "")}, ensure_ascii=False),
                    params_json=json.dumps(applied, ensure_ascii=False),
                ))
                await s.commit()
        # run26 R3：session 使用经主循环调度（worker 跨 loop 连接池安全）
        await _run_on_main(self._db, _persist_optimize)
        log.info("[ai] 策略 %s 已按关键位/价格行为优化: %s", strategy.name, applied)
        out = {"params": applied, "reason": result.get("reason", ""),
               "focus": result.get("focus", "")}
        if not apply:
            out["grid_center"] = dict(new_params)   # AI 建议参数即网格扫描中心
        return out