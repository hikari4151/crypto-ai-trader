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
                      previous: Optional[list] = None,
                      validation_df=None, validation_symbol: str = "BTC/USDT",
                      validation_timeframe: str = "1h",
                      validation_df_alt=None,
                      validation_alt_symbol: Optional[str] = None,
                      validation_alt_timeframe: Optional[str] = None,
                      require_improve: bool = True,
                      goal: str = "") -> Optional[dict]:
        """对现有策略做一次深度反思迭代（参考历史回测与前代迭代记录）。失败时返回 None。

        validation_df：同数据回测验证门（先回测后注册，绩效差拒绝）——
        由调用方传入快照 K 线构造的 DataFrame；None 时跳过验证直接注册。
        validation_df_alt：备选验证窗口（如更大周期/更长历史）——主窗口 0 成交时
        自动用备选窗口重验同一版参数，避免"验证窗口恰好横盘→网格 0 成交→
        四指标全 0→误判绩效下降"的死锁（grid_pct 超出窗口波动时窗口触发不了）。
        """
        # 保留原策略的执行器与参数 schema：用原策略类校验 AI 给出的参数，
        # 避免迭代 dual_ma/grid 后 executor 被替换成 price_action（逻辑脱节）。
        from strategies import get_strategy as _get_strategy
        from strategies import get_dynamic as _get_dynamic
        _dyn = _get_dynamic(strategy.name) or {}
        executor = _dyn.get("executor") or strategy.name

        # 计算迭代序号：基于同一基础策略的累计迭代次数 → 1、2、3…（对应 _v1/_v2/_v3）
        base_name = self._base_name(strategy.name)
        iter_no = await self._next_version(base_name)
        new_name = f"{base_name}_v{iter_no}"

        # ---- 过拟合回炉（P4-D8，与设计侧一致）：检测照判，报告喂回 AI 重迭代 ----
        # 严重过拟合时把检测指标（IS/OOS、衰减、PBO、flags）作为 feedback 注入下一次
        # 迭代 prompt，让 AI 降自由度/收敛参数/简化逻辑；最多 _OVERFIT_MAX_RETRIES 次，
        # 重试用尽仍过拟合才注册打标（P4-D7 兜底：人工可启用，自动接管被拦）。
        from .strategy_designer import _guard_ai_strategy, _overfit_feedback, _OVERFIT_MAX_RETRIES
        guard_candles = snap.get("candles", [])
        spec: dict = {}
        applied: dict = {}
        overfit_report = None
        overfit_feedback = ""
        for retry in range(_OVERFIT_MAX_RETRIES + 1):
            try:
                result = await self._client.chat_json_validated(
                    iteration_messages(strategy, performance, snap, backtests or [],
                                       previous or [], goal=goal,
                                       overfit_feedback=overfit_feedback),
                    feature="strategy_iterate",
                    ctx={"param_schema": strategy.param_schema})
            except (AINotConfigured, AICallError) as e:
                log.warning("[ai] 策略迭代跳过: %s", e)
                return None

            stub = _get_strategy(strategy.name)  # 同执行器类的新实例，参数仅保留 schema 内合法项
            applied = stub.update_params(result.get("params") or {})

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
                overfit_report = await _guard_ai_strategy(spec["name"], applied, snap,
                                                          executor=executor,
                                                          guard_candles=guard_candles)
            except Exception:  # noqa: BLE001
                overfit_report = None
                log.warning("[ai] 迭代过拟合守卫跳过", exc_info=True)
            if overfit_report:
                spec["overfit"] = overfit_report
            if overfit_report and overfit_report.get("verdict") == "无法判定":
                # 证据不足不拦注册，但必须标出"未证明可用"：自动接管实盘的通道据此拒绝
                spec["overfit_inconclusive"] = True
                log.warning("[ai] 迭代策略 %s 过拟合守卫无法判定，允许注册但禁止自动接管实盘",
                            spec["name"])
                break
            if overfit_report and overfit_report.get("verdict") == "严重过拟合":
                if retry < _OVERFIT_MAX_RETRIES:
                    overfit_feedback = _overfit_feedback(executor, applied, overfit_report)
                    log.warning("[ai] 迭代策略 %s 过拟合未通过（%s 分），回炉重迭代（第 %s/%s 次）",
                                spec["name"], overfit_report.get("score"),
                                retry + 1, _OVERFIT_MAX_RETRIES)
                    continue
                # 重试用尽 → P4-D7 兜底：仍注册但打标（前端警告、自动接管门拦截）
                log.warning("[ai] 迭代策略 %s 过拟合回炉 %s 次仍未通过（%s 分），注册供人工启用（自动接管被拦）",
                            spec["name"], _OVERFIT_MAX_RETRIES, overfit_report.get("score"))
                spec["blocked_by_overfit"] = True
            break

        # ---- 回测验证门（先回测后注册）：同数据新旧对比，绩效差拒绝注册 ----
        # 曾对比回测在注册后执行、仅展示——劣质迭代照样入库污染策略库。
        comparison = None
        if validation_df is not None and len(validation_df) >= 200:
            try:
                from backtest.engine import BacktestConfig
                from backtest.fast_engine import run_backtest_fast

                async def _run_pair(df, sym, tf):
                    """跑同一份验证数据上的新旧策略对比，返回 (旧指标, 新指标)。"""
                    old_bt = run_backtest_fast(df, BacktestConfig(
                        symbol=sym, timeframe=tf,
                        strategy_name=strategy.name, strategy_params=dict(strategy.params or {})),
                        bootstrap=False)["metrics"]  # P2-13 验证门只需基础指标
                    new_bt = run_backtest_fast(df, BacktestConfig(
                        symbol=sym, timeframe=tf,
                        strategy_name=executor, strategy_params=applied),
                        bootstrap=False)["metrics"]
                    return old_bt, new_bt

                def _make_comparison(old_bt, new_bt, note: str) -> dict:
                    return {
                        "old": {"total_return": old_bt["total_return"], "sharpe": old_bt["sharpe"],
                                "max_drawdown": old_bt["max_drawdown"], "win_rate": old_bt["win_rate"],
                                "total_trades": int(old_bt.get("total_trades") or 0)},
                        "new": {"total_return": new_bt["total_return"], "sharpe": new_bt["sharpe"],
                                "max_drawdown": new_bt["max_drawdown"], "win_rate": new_bt["win_rate"],
                                "total_trades": int(new_bt.get("total_trades") or 0)},
                        "improved": bool(new_bt["total_return"] > old_bt["total_return"]
                                         and new_bt["max_drawdown"] < old_bt["max_drawdown"] * 1.3),
                        "note": note,
                    }

                old_bt, new_bt = await _run_pair(validation_df, validation_symbol, validation_timeframe)
                comparison = _make_comparison(
                    old_bt, new_bt, "同数据快速回测对比（含手续费/滑点）")
                spec["comparison"] = comparison

                # 0 成交识别：主验证窗口没有触发任何交易（网格/间距类参数超出窗口
                # 实际波动、或窗口恰好横盘）→ 四指标全 0 会误报"绩效下降"。
                # 此时若有备选窗口（更大周期/更长历史），用同一版参数自动重验一次，
                # 避免因窗口选择而误杀合理迭代。
                old_trades = int(comparison["old"]["total_trades"])
                new_trades = int(comparison["new"]["total_trades"])
                no_trade = old_trades == 0 or new_trades == 0
                if no_trade and validation_df_alt is not None and len(validation_df_alt) >= 200:
                    try:
                        alt_old, alt_new = await _run_pair(
                            validation_df_alt,
                            validation_alt_symbol or validation_symbol,
                            validation_alt_timeframe or validation_timeframe)
                        alt_old_trades = int(alt_old.get("total_trades") or 0)
                        alt_new_trades = int(alt_new.get("total_trades") or 0)
                        if alt_old_trades > 0 and alt_new_trades > 0:
                            comparison = _make_comparison(
                                alt_old, alt_new,
                                "主窗口无成交，改用备选验证窗口（"
                                f"{validation_alt_timeframe or validation_timeframe}）后重验对比")
                            spec["comparison"] = comparison
                            log.info("[ai] 迭代 %s 主窗口无成交，备选窗口(%s)重验："
                                     "旧 %d 笔/新 %d 笔",
                                     spec["name"], validation_alt_timeframe,
                                     alt_old_trades, alt_new_trades)
                    except Exception:  # noqa: BLE001
                        log.warning("[ai] 迭代备选窗口重验失败，沿用主窗口判定: %s",
                                    exc_info=True)

                old_trades = int(comparison["old"]["total_trades"])
                new_trades = int(comparison["new"]["total_trades"])
                no_trade = old_trades == 0 or new_trades == 0

                if require_improve and not comparison["improved"]:
                    spec["registered"] = False
                    if no_trade:
                        # 明确区别"窗口无成交"与"绩效下降"，不再给 0%→0% 的误导结论
                        log.warning("[ai] 迭代策略 %s 验证窗口无成交触发（旧 %d 笔/新 %d 笔），拒绝注册",
                                    spec["name"], old_trades, new_trades)
                        spec["no_trade"] = True
                        spec["rejected_reason"] = (
                            f"验证窗口无成交触发（旧 {old_trades} 笔 / 新 {new_trades} 笔）——"
                            "疑似参数间距（如 grid_pct）超出窗口实际波动或窗口恰好横盘，"
                            "非绩效下降；请调整参数间距后重试，或等待行情波动放大")
                        summary = (f"策略[{strategy.name}] 迭代 [{spec['name']}] "
                                   f"验证窗口无成交（{old_trades}/{new_trades} 笔）被拒绝")
                    else:
                        log.warning("[ai] 迭代策略 %s 回测未超越旧策略，拒绝注册: %s",
                                    spec["name"], comparison["improved"])
                        spec["rejected_reason"] = (
                            f"同数据回测未超越上一代（新收益 {new_bt['total_return']:.2%} "
                            f"vs 旧 {old_bt['total_return']:.2%}），保留旧策略")
                        summary = (f"策略[{strategy.name}] 迭代 [{spec['name']}] "
                                   f"因回测未提升被拒绝")
                    async with self._db.session() as s:
                        from core.database import OptimizationLog
                        s.add(OptimizationLog(
                            kind="iteration_blocked",
                            summary=summary,
                            suggestion=json.dumps({"comparison": comparison,
                                                    "no_trade": no_trade},
                                                   ensure_ascii=False),
                            params_json=json.dumps(applied, ensure_ascii=False),
                        ))
                        await s.commit()
                    return spec
            except Exception:  # noqa: BLE001
                log.warning("[ai] 迭代回测验证失败（降级为直接注册）: %s", exc_info=True)

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
            "backtest": comparison,        # 注册时的回测验证指标（下一代迭代可见）
            # P4-D7：过拟合照判但注册——标记与检测报告随 spec 落库，供前端显示警告、
            # 引擎自动接管门拦截（verdict 判定不变，只是不再一票否决注册）
            "overfit": spec.get("overfit"),
            "blocked_by_overfit": spec.get("blocked_by_overfit", False),
        }
        # C3：迭代策略也生成 Pine Script 代码（与前端 buildPine 同源模板），
        # 保证"最新迭代"页签展示的是可运行 Pine 而非仅参数 JSON。
        # 但模板只对 price_action 执行器成立：曾无条件套用，rl_adaptive / dual_ma
        # 迭代出来的策略在图上显示的买卖点与其真实逻辑无关（假代码）。
        from strategies.pine_utils import PINE_TEMPLATE_EXECUTORS
        if executor in PINE_TEMPLATE_EXECUTORS:
            try:
                from strategies.pine_utils import build_price_action_pine
                iter_spec["pine_code"] = build_price_action_pine(iter_spec)
            except Exception as e:  # noqa: BLE001
                log.warning("[ai] 迭代 Pine 代码生成失败（不影响注册）: %s", e)
        else:
            iter_spec["pine_note"] = (
                f"执行器 {executor} 无等价 Pine 模板，未生成图表代码"
                f"（不套用其他策略模板，避免图上买卖点与真实逻辑脱节）")
        register_dynamic(new_name, iter_spec)
        spec["registered"] = True
        spec["comparison"] = comparison
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
