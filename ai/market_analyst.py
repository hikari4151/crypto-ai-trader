"""市场状态理解：定时/手动将K线+指标+交易打包为提示词，获取 AI 解读。"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from core.bus import EventBus
from core.database import Analysis, Database
from core.events import Event, EventType
from .client import AIClient, AICallError, AINotConfigured
from .context_engine import build_continuity_context, parse_prev_analysis
from .prompts import market_analysis_messages

log = logging.getLogger(__name__)


def _ref_price(snap: dict) -> Optional[float]:
    """取快照参考价（用于交易者方程的价格量级反幻觉）。"""
    ind = snap.get("indicators") or {}
    for k in ("close", "ma_fast", "ma_slow"):
        v = ind.get(k)
        if isinstance(v, (int, float)) and v:
            return float(v)
    return None


class MarketAnalyst:
    def __init__(self, client: AIClient, db: Database, bus: EventBus) -> None:
        self._client = client
        self._db = db
        self._bus = bus

    async def analyze(self, snap: dict, position: float = 0.0,
                      recent_trades: Optional[list] = None,
                      source: str = "manual") -> dict:
        """执行一次 AI 行情解读，返回结构化结果并持久化。

        程序先独立复算方向与读取上一轮结论，注入提示词并在校验器中强制检查：
          - 方向一致性：AI bias 与程序方向矛盾 → 校验打回（强确认时）
          - 交易者方程：trade_plan 三价须 RR≥1.0 且胜率×回报>败率×风险
          - 决策连续性：短时反手 → 强制降置信度上限
        """
        # 程序方向复算（从 K 线独立计算，不依赖 AI）
        program = None
        candles = snap.get("candles") or []
        if len(candles) >= 8:
            try:
                from .direction_engine import compute_direction
                program = compute_direction(candles)
            except Exception as e:  # noqa: BLE001
                log.warning("[ai] 方向复算失败: %s", e)
        program_bias = (program or {}).get("direction")

        # 决策连续性：查同品种上一轮分析（反手判定在 AI 输出后做）
        prev_analysis = None
        try:
            from sqlalchemy import select as _sel
            from core.database import Analysis as _Analysis
            async with self._db.session() as s:
                row = (await s.execute(
                    _sel(_Analysis).where(_Analysis.symbol == snap.get("symbol", ""))
                    .order_by(_Analysis.ts.desc()).limit(1))).scalar_one_or_none()
            if row is not None:
                prev_analysis = parse_prev_analysis(row.content)
        except Exception as e:  # noqa: BLE001
            log.warning("[ai] 上一轮分析读取失败: %s", e)

        # 提示词阶段：仅携带上一轮 bias 供参考（具体反手规则由校验器在输出后强制）
        continuity_hint = None
        if prev_analysis and prev_analysis.get("bias") in ("long", "short"):
            continuity_hint = {
                "has_prev": True, "prev_bias": prev_analysis.get("bias"),
                "prev_confidence": prev_analysis.get("confidence"),
                "prev_regime": prev_analysis.get("regime"),
            }

        messages = market_analysis_messages(snap, position, recent_trades or [],
                                            program=program, continuity=continuity_hint)
        # 校验 ctx：程序方向（方向一致性）、上轮结论（反手置信度上限）、参考价（方程量级）
        ctx_snap = dict(snap)
        if program_bias:
            ctx_snap["program_direction"] = program_bias
            ctx_snap["program_score"] = (program or {}).get("score", 0)
        result = await self._client.chat_json_validated(
            messages, feature="market_analysis",
            ctx={"snap": ctx_snap,
                 "continuity": prev_analysis,
                 "trade_plan_ref_price": _ref_price(snap)})

        # 决策连续性：AI 输出后判定反手
        continuity = None
        if prev_analysis:
            continuity = build_continuity_context(
                prev_analysis, str(result.get("bias") or "neutral"), snap.get("symbol", ""))

        # 附上程序上下文，前端可展示方向对齐/冲突
        result["_program"] = {
            "direction": program_bias,
            "score": (program or {}).get("score", 0),
            "confidence": (program or {}).get("confidence", 0.0),
            "reasons": (program or {}).get("reasons", []),
        }
        result["_continuity"] = continuity
        # 关键：补程序时间戳——"短时反手"判定依赖上一轮分析的 ts 字段，
        # AI 输出 schema 不含 ts，此前依赖 AI 自愿输出导致防线永远失效
        result["ts"] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(result, ensure_ascii=False)
        async with self._db.session() as s:
            s.add(Analysis(
                symbol=snap.get("symbol", ""), content=content,
                meta_json=json.dumps({"source": source, "has_program": bool(program)})))
            await s.commit()
        await self._bus.publish(Event(EventType.AI_ANALYSIS, {"snapshot": snap, "result": result}, source="ai.market_analyst"))
        return result

    async def safe_analyze(self, snap: dict, position: float = 0.0) -> Optional[dict]:
        """容错版：AI 未配置或失败时返回 None，由调用方降级处理。"""
        try:
            return await self.analyze(snap, position, source="auto")
        except AINotConfigured as e:
            log.warning("[ai] %s", e)
            return None
        except AICallError as e:
            log.warning("[ai] 解读失败: %s", e)
            return None