"""AI 策略迭代：重新思考现有策略，批判性反思并给出改进版本。

与"参数优化"的区别：
- 参数优化：在原有框架内微调参数
- 策略迭代：重新审视策略逻辑本身（入场/出场/风控假设），可能调整参数+逻辑思路，
  产出改进版说明，供用户选择是否应用
"""
import json
import logging
from typing import Any, Optional

from core.database import Database
from strategies.base import Strategy
from .client import AIClient, AICallError, AINotConfigured
from .prompts import iteration_messages

log = logging.getLogger(__name__)


class StrategyIteration:
    """AI 重新思考迭代现有策略。"""

    def __init__(self, client: AIClient, db: Database) -> None:
        self._client = client
        self._db = db

    @staticmethod
    def _base_name(name: str) -> str:
        """剥离迭代后缀：dual_ma_v2 -> dual_ma。用于基于同一基础策略累加序号。"""
        import re
        return re.sub(r"_v\d+$", "", name)

    async def _next_version(self, based_on: str) -> int:
        """返回下一迭代序号（基于同一基础策略）：1 → 2 → 3 …（对应 _v1/_v2/_v3）。"""
        try:
            key = f"ai_iter_count_{based_on}"
            count = int(await self._db.kv_get(key) or 0)
            new_count = count + 1
            await self._db.kv_set(key, str(new_count))
            return new_count
        except Exception:  # noqa: BLE001
            return 1

    async def iterate(self, strategy: Strategy, performance: dict, snap: dict,
                      backtests: Optional[list] = None,
                      previous: Optional[list] = None) -> Optional[dict]:
        """对现有策略做一次深度反思迭代（参考历史回测与前代迭代记录）。失败时返回 None。"""
        try:
            result = await self._client.chat_json_validated(
                iteration_messages(strategy, performance, snap, backtests or [], previous or []),
                feature="strategy_iterate",
                ctx={"param_schema": strategy.param_schema})
        except (AINotConfigured, AICallError) as e:
            log.warning("[ai] 策略迭代跳过: %s", e)
            return None

        # 保留原策略的执行器与参数 schema：用原策略类校验 AI 给出的参数，
        # 避免迭代 dual_ma/grid 后 executor 被替换成 price_action（逻辑脱节）。
        from strategies import get_strategy as _get_strategy
        from strategies import get_dynamic as _get_dynamic
        _dyn = _get_dynamic(strategy.name) or {}
        executor = _dyn.get("executor") or strategy.name
        stub = _get_strategy(strategy.name)  # 同执行器类的新实例，参数仅保留 schema 内合法项
        applied = stub.update_params(result.get("params") or {})

        # 计算迭代序号：基于同一基础策略的累计迭代次数 → 1、2、3…（对应 _v1/_v2/_v3）
        base_name = self._base_name(strategy.name)
        iter_no = await self._next_version(base_name)
        new_name = f"{base_name}_v{iter_no}"

        # 迭代生成新策略名（基础名_v序号），不显示 v1.0 版本徽标，改由策略名体现代次
        spec = {
            "name": new_name,
            "title": result.get("title", "迭代后策略"),
            "description": result.get("description", ""),
            "logic": result.get("logic", ""),
            "params": applied,
            "risk_tips": result.get("risk_tips", []),
            "critique": result.get("critique", ""),
            "improvements": result.get("improvements", []),
            "summary": result.get("summary", ""),
            "created_by": "ai_iteration",
            "based_on": base_name,
            "version": "",
            "iter_no": iter_no,
        }
        # 过拟合守卫：迭代出的策略同样要过前推验证，防止"把样本内噪声当规律"
        try:
            from .strategy_designer import _guard_ai_strategy
            guard_candles = snap.get("candles", [])
            overfit_report = await _guard_ai_strategy(spec["name"], applied, snap,
                                                      executor=executor,
                                                      guard_candles=guard_candles)
            if overfit_report:
                spec["overfit"] = overfit_report
            if overfit_report and overfit_report.get("verdict") == "严重过拟合":
                log.warning("[ai] 迭代策略 %s 过拟合未通过，拒绝注册", spec["name"])
                spec["blocked_by_overfit"] = True
                # 仍记录日志但不注册
                async with self._db.session() as s:
                    from core.database import OptimizationLog
                    s.add(OptimizationLog(
                        kind="iteration",
                        summary=f"策略[{strategy.name}] 迭代 [{spec['name']}] 因过拟合被拦截",
                        suggestion=json.dumps({"overfit": overfit_report}, ensure_ascii=False),
                        params_json=json.dumps(applied, ensure_ascii=False),
                    ))
                    await s.commit()
                return spec
        except Exception:  # noqa: BLE001
            log.warning("[ai] 迭代过拟合守卫跳过", exc_info=True)

        # 注册为可用策略（新策略名 = 基础名_v序号，如 dual_ma_v1）
        from strategies import register_dynamic
        iter_spec = {
            "name": new_name,
            "title": spec.get("title", "迭代策略"),
            "description": spec.get("description", ""),
            "logic": spec.get("logic", ""),
            "params": applied,
            "risk_tips": spec.get("risk_tips", []),
            "created_by": "ai_iteration",
            "based_on": base_name,
            "version": "",
            "iter_no": iter_no,
            "critique": spec.get("critique", ""),
            "improvements": spec.get("improvements", []),
            "executor": executor,          # 保留原策略执行器，注册时不再默认 price_action
        }
        register_dynamic(new_name, iter_spec)
        # 持久化到 AiStrategy（重启后仍可见，全部策略列表能显示迭代策略）
        from core.database import AiStrategy
        from sqlalchemy import select
        async with self._db.session() as s:
            from core.database import OptimizationLog
            s.add(OptimizationLog(
                kind="iteration",
                summary=f"策略[{strategy.name}] AI 深度反思迭代 -> [{new_name}]",
                suggestion=json.dumps({
                    "critique": spec["critique"],
                    "improvements": spec["improvements"],
                    "summary": spec["summary"],
                    "overfit": overfit_report if overfit_report else None,
                }, ensure_ascii=False),
                params_json=json.dumps(applied, ensure_ascii=False),
            ))
            existing = (await s.execute(select(AiStrategy).where(AiStrategy.name == new_name))).scalar_one_or_none()
            if existing:
                existing.spec_json = json.dumps(iter_spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=new_name, spec_json=json.dumps(iter_spec, ensure_ascii=False)))
            await s.commit()
        log.info("[ai] 策略迭代完成: %s -> %s", strategy.name, new_name)
        return spec
