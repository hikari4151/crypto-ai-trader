"""AI 策略设计师：基于关键位+价格行为生成可执行策略，注册为动态策略并持久化。"""
import asyncio
import json
import logging
import re
from typing import Any, Optional

from sqlalchemy import select

from core.bus import EventBus
from core.database import Database
from .client import AIClient, AICallError, AINotConfigured
from .prompts import design_strategy_messages

log = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", (name or "ai_strategy").lower().strip())
    return name[:48] or "ai_strategy"


async def _guard_ai_strategy(name: str, params: dict, snap: dict,
                             executor: str = "price_action",
                             guard_candles: Optional[list] = None) -> Optional[dict]:
    """AI 策略过拟合守卫：用历史数据跑前推验证 + PBO。

    - 数据源：guard_candles（引擎主动拉的足量真实K线）或快照K线。
      合成数据无法让 price_action 产生可靠交易，会导致守卫误判，
      故数据不足时返回"无法判定"而非拦截。
    - executor：AI 策略在执行器上回测（AI 策略本身尚未注册，直接用执行器类）
    返回过拟合报告 dict；无法判定时返回 None（不拦截）。
    """
    import pandas as pd
    from backtest.overfit import OverfitConfig, detect_overfit

    try:
        candles = guard_candles or snap.get("candles") or []
        if len(candles) < 600:
            return {
                "verdict": "无法判定",
                "score": None,
                "data_source": "数据不足",
                "flags": [{"level": "info",
                           "msg": f"历史K线仅 {len(candles)} 根，不足 600 根，过拟合检测跳过。"
                                  "策略落地前建议先用足量历史数据跑过拟合检测。"}],
            }
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
        timeframe = snap.get("timeframe", "1h")
        cfg = OverfitConfig(symbol=snap.get("symbol", "BTC/USDT"),
                            timeframe=timeframe, n_folds=4, fold_ratio=0.2, min_is_len=200)
        # 用执行器名回测（AI 策略尚未注册，无法用 AI 名）。
        # CSCV/PBO 计算量大（数百次回测），放后台线程避免阻塞主事件循环
        report = await asyncio.to_thread(detect_overfit, df, executor, params, cfg)
        return {
            "verdict": report.verdict, "score": report.score,
            "is_ret": report.is_ret, "oos_ret": report.oos_ret,
            "decay": report.decay, "pbo": report.pbo,
            "oos_sharpe": report.oos_sharpe, "stability": report.stability,
            "flags": report.flags, "n_folds": report.n_folds,
            "data_source": "真实K线",
        }
    except Exception as e:  # noqa: BLE001
        log.warning("[ai] 过拟合守卫跳过: %s", e)
        return None


class StrategyDesigner:
    def __init__(self, client: AIClient, db: Database, bus: EventBus) -> None:
        self._client = client
        self._db = db
        self._bus = bus

    async def design(self, snap: dict, recent_trades: Optional[list] = None,
                     backtests: Optional[list] = None,
                     guard_candles: Optional[list] = None,
                     strategy_type: str = "",
                     custom_requirement: str = "") -> dict:
        """让 AI 基于行情快照（含 S/R 与价格行为）设计策略。返回策略规格 dict。

        backtests: 历史回测结果摘要，供 AI 参考（避免重蹈覆辙）。
        guard_candles: 用于过拟合守卫的足量历史K线（引擎传入）。
        strategy_type: 用户指定策略类型（trend/breakout/mean_reversion/grid/custom）。
        custom_requirement: 用户自定义设计要求（自由打字）。
        """
        from strategies import register_dynamic
        from strategies.price_action import PriceActionStrategy

        result = await self._client.chat_json_validated(
            design_strategy_messages(snap, recent_trades or [], backtests or [],
                                     strategy_type=strategy_type,
                                     custom_requirement=custom_requirement),
            feature="strategy_design",
            ctx={"param_schema": PriceActionStrategy.param_schema})
        name = sanitize_name(result.get("name"))
        if name in ("dual_ma", "grid", "price_action"):
            name = "ai_" + name

        # 用 price_action 的 schema 校验/限制 AI 给出的参数，保证可直接执行
        stub = PriceActionStrategy()
        applied = stub.update_params(result.get("params") or {})

        spec = {
            "name": name,
            "title": result.get("title", "AI 设计策略"),
            "description": result.get("description", ""),
            "logic": result.get("logic", ""),
            "params": applied,
            "risk_tips": result.get("risk_tips", []),
            "created_by": "ai",
        }
        # 过拟合守卫：用足量历史K线跑前推验证，识别 AI 在样本内"作弊"的策略
        overfit_report = await _guard_ai_strategy(name, applied, snap, guard_candles=guard_candles)
        if overfit_report:
            spec["overfit"] = overfit_report
        if overfit_report and overfit_report.get("verdict") == "严重过拟合":
            log.warning("[ai] 策略 %s 过拟合检测未通过（%s 分），拒绝注册", name, overfit_report.get("score"))
            spec["blocked_by_overfit"] = True
            return spec
        register_dynamic(name, spec)

        from core.database import AiStrategy
        async with self._db.session() as s:
            existing = (await s.execute(select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
            if existing:
                existing.spec_json = json.dumps(spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=name, spec_json=json.dumps(spec, ensure_ascii=False)))
            await s.commit()
        log.info("[ai] 新策略已设计并注册: %s", name)
        return spec