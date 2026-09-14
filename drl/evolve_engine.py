"""持续进化引擎：管理因子挖掘和策略训练的持续循环，实现"越训练越强"。

核心流程：
1. 后台循环按间隔检查是否有新数据
2. 加载上次最佳模型继续训练（而非从头开始）
3. 滚动窗口训练（最近 N 根 K 线）
4. 验证段评估新模型 vs 最佳模型，退化时自动回退
5. 保存最佳模型到 ModelZoo

使用方式：
    evolve = EvolveEngine(db, bus, settings.data_dir / "models")
    await evolve.start()
    # ... 运行中 ...
    await evolve.pause()
    await evolve.resume()
    await evolve.stop()
"""
import asyncio
import concurrent.futures
import json
import logging
import math
import random
import threading
import time
from datetime import datetime, timezone
from collections.abc import Awaitable
from typing import Any, Callable, Optional

from backtest.data_loader import generate_demo
from config.settings import settings
from core.bus import EventBus
from core.database import Database
from core.events import Event, EventType
from drl.agent import train_drl
from drl.factor_miner import train_factor_miner
from factors.engine import compute_factor_matrix
from strategies.meta import train_meta_controller
from strategies import get_dynamic, register_dynamic
from strategies.rl_adaptive import RLAdaptiveStrategy

from .model_zoo import ModelZoo

log = logging.getLogger(__name__)

# 元控制器默认子策略池（用户未在「统一策略」中指定时）。
# P2-9：默认含 rl_evolve（策略DRL 的持续进化产出）——让
# "因子→策略DRL→元控制器"进化链路默认闭合，而非只在三个内置策略间组合。
DEFAULT_META_SUBS = "dual_ma,factor_signal,price_action,rl_evolve"

# P4-E1：K线数据缓存键上限（symbol×timeframe 组合只增不减，超限淘汰最旧）
_DATA_CACHE_MAX_KEYS = 16


class InsufficientDataError(RuntimeError):
    """数据不足的业务分支（非崩溃）：_run_loop 用 warning 记日志、不带全栈。

    此前以裸 RuntimeError 表达，落在通用 except 里 log.exception 每 5 分钟
    刷一屏 traceback，且把「数据不足」与真正的训练崩溃混在同一语义。
    """


def _strategy_drl_train_cfg(episodes: int) -> dict:
    """持续进化「策略 DRL」的训练配置（单一定义处：训练与 Pine 导出口径同源）。

    state_window / hidden 默认取「装得进 Pine」的组合：神经引擎导出自动交易代码时
    权重整段内联，旧配置 state_window=20 + hidden=[64,64] 有 21184 个权重
    （脚本约 700KB），导出器只能显式拒绝——进化产物于是永远没有 Pine 代码，
    图上也就永远没有买卖点（本次关键要求的症结）。窗口从 20 降到 4 牺牲部分
    时序记忆，同时也在压低过拟合风险；要恢复旧规模改 config.yaml 两项即可。
    """
    return {
        "episodes": episodes,
        "n_episodes": 4,
        "ppo_epochs": 4,
        "mini_batch_size": 128,
        "hidden": [int(h) for h in getattr(settings, "evolve_strategy_drl_hidden", [16, 16])],
        "seed": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "fee_rate": 0.001,
        "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
        "cost_scale": 1.0,
        "min_trade_zone": 0.05,
        "state_window": int(getattr(settings, "evolve_strategy_drl_state_window", 4)),
        "lr_actor": 3e-3,
        "lr_critic": 6e-3,
        "gamma": 0.95,
        "entropy_coef": 0.05,
        "val_eval_interval": 10,
        "early_stop_patience": 0,
        "reward_dd_penalty": 0.0,
        "reward_losing_penalty": 0.0,
        "reward_trend_align": 0.0,
        # P1-5：OOS 安检门槛（settings.evolve_oos_min_bars，默认 250）
        # 与多段评估（3 段取中位数，降单段运气噪声）
        "oos_min_bars": getattr(settings, "evolve_oos_min_bars", 250),
        "oos_segments": 3,
    }


def _should_rollback(new_fit: float, old_fit: Optional[float], threshold: float) -> bool:
    """回退判据（P0-4）：旧 fitness 为正时按比例阈值比较；旧 fitness 非正时
    “×threshold”会让阈值更宽松（方向反转），改为新值更低即回退。

    run24 修复：NaN 旧/新 fitness 必须视为不可比（NaN 与任何数比较恒 False，
    会让回退保护永久失效——NaN 锚点污染后永不回退/永不拦截）。
    """
    if old_fit is None or not math.isfinite(old_fit) or not math.isfinite(new_fit):
        return False
    if old_fit <= 0:
        return new_fit < old_fit
    return new_fit < old_fit * threshold


def _oos_gate_passed(report: Optional[dict]) -> tuple[bool, str]:
    """因子挖掘的 OOS 安检门：判据与 /api/drl/mine-factor 注册前完全一致。

    进化引擎此前把组合因子无条件写进 _cascade_*，策略 DRL 下一轮就直接继承，
    等于绕过了 API 路径上的 valid 检查。
    """
    rep = report or {}
    if not rep.get("enabled"):
        return False, "OOS 段样本不足，无样本外证据"
    if not rep.get("valid"):
        return False, str(rep.get("reason") or "OOS 安检未通过")
    return True, ""


def _strategy_deploy_gate(result: dict) -> tuple[bool, str]:
    """新模型能否接管实盘（策略 DRL / 元控制器共用）：先过 OOS 硬门，再看样本外是否赚钱。

    train_drl 的硬门只在「训练段盈利且 OOS 衰减过半」时判过拟合，训练段本身亏损时
    decay=0（不可比、不判定）；而进化引擎旧逻辑连这个硬门都没读，只要数据源不是
    demo 就注册 rl_evolve 并用新权重覆盖实盘模型文件——未通过样本外检验的模型会被
    自动上线。

    P1-6：判据统一为 OOS 收益（与回退保护同口径）。
    P1-7：regime 剧变时把原因写清楚，供用户判断是否行情状态切换造成的假阳性。
    """
    oos = result.get("oos_report") or {}
    if result.get("deployment_blocked") or oos.get("hard_rejected"):
        return False, str(oos.get("reason") or "OOS 硬门拦截（疑似过拟合）")
    if not oos.get("enabled"):
        return False, "OOS 段样本不足（<最小样本数），无样本外证据不得自动接管实盘"
    try:
        oos_ret = float(oos.get("oos_ret", 0.0))
    except (TypeError, ValueError):
        oos_ret = 0.0
    if not math.isfinite(oos_ret):
        # run24 修复：NaN/Inf oos_ret 曾绕过硬门（float("nan") <= 0 为 False → 部署）。
        # 行情数据缺口导致 NaN 指标时不得上线："无有效 OOS 收益"按非正拦截。
        return False, f"OOS 段收益无效（{oos_ret}，非有限数值），数据缺口，暂不部署"
    if oos_ret <= 0:
        _regime_note = ("；且训练/样本外行情状态剧变(regime_shift)"
                        if oos.get("regime_shift") else "")
        # P2-14：区分"样本外无信号"与"真亏损"。
        # P4-E2：优先用真实交易次数（oos_trades，元控制器/新报告提供）判断是否
        # 在样本外行动过——段末持仓比例会误伤"交易后已平仓"的合理策略（段末空仓
        # → 比例 0 被误判为从未交易）。报告缺 oos_trades 时回退到段末持仓比例
        #（老报告/精简报告），再缺失视为未知，按真亏损口径给"非正"。
        raw_trades = oos.get("oos_trades")
        if raw_trades is not None:
            try:
                n_trades = int(raw_trades)
            except (TypeError, ValueError):
                n_trades = None
            if n_trades is not None and n_trades <= 0:
                return False, (f"OOS 段模型从未发出交易指令(oos_trades={n_trades})，"
                               f"样本外未交易，暂不部署{_regime_note}")
        else:
            raw_pos = oos.get("oos_position_ratio")
            if raw_pos is not None:
                try:
                    pos_ratio = float(raw_pos)
                except (TypeError, ValueError):
                    pos_ratio = None
                if pos_ratio is not None and pos_ratio < 0.05:
                    return False, (f"OOS 段几乎无信号(position_ratio={pos_ratio:.3f})，"
                                   f"样本外未交易，暂不部署{_regime_note}")
        return False, f"OOS 段收益 {oos_ret:.4f} 非正，样本外不具备盈利能力{_regime_note}"
    return True, ""


def resolve_meta_sub_strategies(spec_params: dict | None) -> str:
    """解析元控制器的子策略池：用户选定优先，剔除 meta_controller 自身与
    不存在的策略，空则回退默认池（同样过滤未注册项）。

    剔除自身是硬约束——元控制器把自己当子策略会无限递归。
    P2-9：默认池含 rl_evolve，若 rl_evolve 尚未注册（首启）则自动过滤。
    """
    from strategies import list_strategies
    available = {s["name"] for s in list_strategies()}
    raw = str((spec_params or {}).get("sub_strategies") or "")
    names = [n.strip() for n in raw.split(",")
             if n.strip() and n.strip() != "meta_controller" and n.strip() in available]
    if names:
        return ",".join(dict.fromkeys(names))
    defaults = [n for n in DEFAULT_META_SUBS.split(",")
                if n in available and n != "meta_controller"]
    return ",".join(dict.fromkeys(defaults)) or "dual_ma,factor_signal"


def _training_identity(model: str, symbol: str, timeframe: str, *,
                       state_window: int = 1, factor_signature: str = "",
                       config: Optional[dict] = None,
                       window_tail_ts: Optional[int] = None) -> dict:
    """当前训练轮的可比较身份：回退判据只在身份一致时生效。

    不同 symbol/timeframe/state_window/训练配置产出的模型互相比较没有意义
    （BTC 锚点不该压住 ETH 候选；1h 锚点不该与 15m 候选比）。config_fingerprint
    用稳定排序后的结构化 JSON 做 SHA-256，只覆盖影响状态/奖励/切分/部署门的口径。
    """
    stable = config or {}
    import hashlib
    canonical = json.dumps(
        stable, sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "model": model,
        "symbol": symbol,
        "timeframe": timeframe,
        "state_window": int(state_window),
        "factor_signature": str(factor_signature or ""),
        "config_fingerprint": fingerprint,
        "window_tail_ts": window_tail_ts,
    }


def _identity_mismatch(anchor_meta: dict, identity: dict) -> Optional[str]:
    """锚点与候选的身份差异。返回 None 表示可比；否则给出不可比原因。

    锚点缺身份字段（旧版本）视为 unknown：不进行跨口径回退比较，
    但保留其权重作为可运行部署（comparable 仍由 demo 口径控制）。
    """
    for key in ("model", "symbol", "timeframe", "state_window",
                "factor_signature", "config_fingerprint"):
        a = anchor_meta.get(key)
        b = identity.get(key)
        if a is None:
            return f"锚点缺少身份字段 {key}（旧版本，不可比）"
        if str(a) != str(b):
            return f"{key} 不一致（锚点 {a} vs 候选 {b}）"
    return None


class EvolveEngine:
    """持续进化引擎：管理因子挖掘和策略训练的持续循环。"""

    def __init__(self, db: Database, bus: EventBus, models_dir) -> None:
        self.db = db
        self.bus = bus
                # P4-A5：版本保留数从 settings 传入（此前 ModelZoo 硬编码 _MAX_VERSIONS=10，
        # settings.evolve_max_versions 是死配置，用户修改无效）
        self.zoo = ModelZoo(models_dir,
                            max_versions=int(getattr(settings, "evolve_max_versions", 10)))
        self._running = False
        self._paused = False
        self._tasks: list[asyncio.Task] = []
        # 保活（watchdog）：_run_loop 只能兜住"逐轮训练"里的异常。一旦异常从
        # 循环体之外逸出，那条 asyncio 任务会直接结束，而 _running/_enabled 仍是
        # True——引擎对外依然显示"运行中"，实际已永久停训，只能重启进程或手动
        # 关开一次开关。下面这组状态供 _supervise_loop 周期巡检并重建死循环。
        self._loop_names = ("evolve_factor_miner", "evolve_strategy_drl",
                            "evolve_meta_controller")
        self._restarts: dict[str, int] = {n: 0 for n in self._loop_names}
        self._last_restart: dict[str, float] = {n: 0.0 for n in self._loop_names}
        self._supervise_interval = 30.0
        # 同一管线两次自愈的最小间隔：防"一重建就崩"演变成重启风暴烧 CPU
        self._restart_min_gap = 20.0
        self._supervise_status: dict[str, Any] = {
            "alive": False, "last_check": 0.0, "checks": 0, "restarts": 0,
            "interval": self._supervise_interval, "last_restart_at": 0.0,
            "last_restart_name": "",
        }
        # P2-10：全局训练锁——三条训练管线串行执行（因子挖掘/策略DRL/元策略
        # 同时训练会吃满 CPU 拖垮实时引擎），手动触发与周期循环共用同一把锁。
        self._train_lock = asyncio.Lock()
        # 部署后后台回测摘要任务引用（防 GC；done 回调里弹出）
        self._backtest_tasks: dict[str, asyncio.Task] = {}
        # P2-11：数据增量门——记录各管线最近一次训练时的数据末根时间戳；
        # 数据无新增 K 线时跳过本轮（避免 60s 间隔 + 5 分钟数据缓存下的空转重训）
        self._last_tail_ts: dict[str, float] = {}
        # 数据缓存：5 分钟内不重复拉取交易所（避免频繁限流）
        self._data_cache: dict[str, tuple[float, Any]] = {}  # key -> (timestamp, df)
        self._data_cache_ttl = 300  # 5 分钟
        # P4-E1：缓存键上限（symbol×timeframe 组合只增不减，超限淘汰最旧）
        # 训练状态（_demo_streak 为内部字段，不对外展示：记录连续 demo 训练次数，
        # 供 _run_loop 在交易所持续不可达时自动降频，避免 60s 一次全量烧 CPU）
        self._factor_miner_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                     "last_error": "", "next_run": 0, "last_success": False,
                                     "oos_rejected": "", "_demo_streak": 0,
                                     "loop_beat": 0.0, "restarts": 0,
                                     # 进化组合因子部署状态（Task：进化因子使用场景；combo_ 前缀
                                     # 避开 _pipeline_view/_mark_deployment_status 的 deployed_* 键覆盖）
                                     "combo_strategy": "", "combo_version": "",
                                     "combo_deployed_at": 0, "combo_deploy_error": "",
                                     "combo_backtest": None, "combo_backtest_error": ""}
        self._strategy_drl_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                     "last_error": "", "next_run": 0, "last_success": False,
                                     "oos_rejected": "", "_demo_streak": 0,
                                     "loop_beat": 0.0, "restarts": 0,
                                     # P0-族群：冠军/挑战者轮次计数与当前谱系（展示用）
                                     "_pop_round": 0, "lineage": "main"}
        self._meta_controller_status = {"active": False, "last_run": 0, "episode": 0, "fitness": 0.0,
                                        "last_error": "", "next_run": 0, "last_success": False,
                                        "oos_rejected": "", "_demo_streak": 0,
                                        "loop_beat": 0.0, "restarts": 0,
                                        "sub_strategies": DEFAULT_META_SUBS}
        # 锁死自检：连续多少轮没有任何模型被接受。进化引擎只看 episode/fitness 时
        # "一直在训"和"训了但全被锚点拒绝"长得一模一样（实测 72/72 全 rollback，
        # 前端毫无异常提示），所以这里按管线记连续拒绝数，供 status 判 stalled。
        self._models = ("factor_miner", "strategy_drl", "meta_controller")
        self._reject_streak: dict[str, int] = {m: 0 for m in self._models}
        self._rounds_total: dict[str, int] = {m: 0 for m in self._models}
        self._stall_rounds = max(2, int(getattr(settings, "evolve_stall_rounds", 10)))
        # 从配置读取间隔
        self._factor_miner_interval = getattr(settings, "evolve_factor_miner_interval", 60)
        self._strategy_drl_interval = getattr(settings, "evolve_strategy_drl_interval", 60)
        self._rolling_window = getattr(settings, "evolve_rolling_window", 5000)
        self._rollback_threshold = getattr(settings, "evolve_rollback_threshold", 0.95)
        self._enabled = getattr(settings, "evolve_enabled", True)
        # 训练标的池（用户指定：仅 BTC/ETH/SOL），训练时轮换使用
        self._symbols = list(getattr(settings, "evolve_symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]))
        if not self._symbols:
            self._symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        # P4-A1：三管线共享同一个"当前训练标的"（不再各自轮换）。
        # 此前 factor_miner/strategy_drl/meta_controller 各自调用 _next_symbol 独立递增，
        # 导致级联因子（BTC 上挖出）被注入 ETH 训练——长度校验通过但语义错配，
        # 整个"因子→策略DRL→元策略"链路断裂。现在：三管线同标训练，
        # 全部完成一个 cycle 后才切换到下一标的。
        self._symbol_idx = 0
        # 当前标的已训练完成的管线集合（三管线齐全后切换标的）
        self._cycle_pipelines_done: set[str] = set()
        # 专用训练线程池（单 worker）：训练本体串行执行；stop/set_enabled(False)
        # 通过 _training_done 事件等待在跑的训练跑完——取消协程只会中断 await，
        # executor 线程会继续烧 CPU，等待后"禁用/停止"语义才名副其实
        self._train_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="evolve-train")
        self._training_done = threading.Event()
        self._training_done.set()
        # 拉数串行锁：三管线 + 跨标 OOS 并发打交易所会放大 429，统一串行化
        self._fetch_lock = asyncio.Lock()
        # 部署编排：可选的运行时 reload 回调（由 TradingEngine 注入）。
        # 未配置时部署仍完成文件与注册表层，runtime_reload 标记 skipped。
        self._deployment_reload: Optional[Callable[[str, int, dict], Awaitable[Optional[dict]]]] = None
        # 部署/候选状态：candidate 是本轮训练结果，deployed 是当前 best 快照。
        # 回退轮 candidate_fitness 与 deployed_fitness 分离后，UI 不再误把
        # 被回退的候选值当成"当前模型收益"。
        self._deploy_status: dict[str, dict] = {
            m: {
                "candidate_fitness": None, "deployed_fitness": None,
                "deployed_version": None, "deployed_source": "",
                "last_outcome": "", "anchor_mismatch": "",
                "runtime_reload": "skipped",
            } for m in ("factor_miner", "strategy_drl", "meta_controller")
        }

    def set_deployment_reload(
        self,
        callback: Optional[Callable[[str, int, dict], Awaitable[Optional[dict]]]],
    ) -> None:
        """注入运行时模型热加载回调（TradingEngine 在构造完成后调用）。"""
        self._deployment_reload = callback

    async def start(self) -> None:
        """启动后台训练循环。"""
        if self._running:
            return
        self._running = True
        self._paused = False
        # 启动恢复：只要训练产物（flat 模型文件）存在就注册对应策略，
        # 保证「由持续进化训练的策略」在重启/多轮回退后依然在其他模块可选可用
        try:
            await self._restore_evolve_strategies()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 启动恢复策略注册失败: %s", e)
        # 重启后恢复各管线已完成的训练轮次（episode），避免「重新进入软件后
        # 训练轮次显示归零」——轮次是累计事实，应以数据库落库为准接续显示
        try:
            await self._restore_episode_counts()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 启动恢复训练轮次失败: %s", e)
        if self._enabled:
            # 首次训练在启动 15s 后进行（等待系统初始化完成），之后按间隔定期训练
            self._spawn_loops()
            # 保活监督：三条循环任一条意外退出，都由巡检发现并重建
            self._spawn_supervisor()
            self._supervise_status["alive"] = True
            log.info("[evolve] 持续进化引擎已启动（因子挖掘=%ds，策略DRL=%ds，元策略=%ds，"
                     "首次训练将在15s后开始；保活巡检间隔=%ds）",
                     self._factor_miner_interval, self._strategy_drl_interval,
                     getattr(settings, "evolve_meta_interval", 600),
                     int(self._supervise_interval))
        else:
            log.info("[evolve] 持续进化引擎已禁用（evolve_enabled=false）")

    async def stop(self) -> None:
        """停止训练循环。"""
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._supervise_status["alive"] = False
        # 等在跑的 to_thread 训练跑完（取消只停协程；executor 线程会跑到底）。
        # 用 to_thread 等 threading.Event，避免阻塞事件循环；超时仅告警不阻断。
        await self._wait_training_done()
        log.info("[evolve] 持续进化引擎已停止")

    async def _wait_training_done(self, timeout: float = 180.0) -> None:
        """等待专用训练线程池里当前训练结束（无训练在跑时立即返回）。"""
        try:
            await asyncio.to_thread(self._training_done.wait, timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            log.warning("[evolve] 等待训练线程结束超时（>%.0fs），可能仍在后台训练", timeout)

    # ---- 策略注册（训练成功 / 启动恢复共用同一套规格，保证各模块看到的口径一致） ----

    async def _upsert_ai_strategy(self, name: str, spec: dict) -> None:
        """动态策略注册结果落库（AiStrategy 表），重启后由 dynamic_store.restore_from_db 恢复。"""
        import json as _json
        from core.database import AiStrategy
        from sqlalchemy import select
        async with self.db.session() as s:
            row = (await s.execute(select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
            if row:
                row.spec_json = _json.dumps(spec, ensure_ascii=False)
            else:
                s.add(AiStrategy(name=name, spec_json=_json.dumps(spec, ensure_ascii=False)))
            await s.commit()

    def _export_rl_evolve_pine(self) -> tuple[str, str]:
        """从磁盘上的 strategy_drl 最佳模型现场导出 Pine，返回 (代码, 不可导出原因)。

        为什么注册时读文件重导，而不是复用训练返回值里的 pine_code：
        1. 回退分支在写 pine_code 之前就 return 了，注册的却是历史最佳模型；
        2. save_agent_best 会重写 flat 文件，训练返回值可能已不是当前最佳；
        3. 启动恢复路径（_restore_evolve_strategies）根本没有训练返回值。
        flat 文件才是"部署端与回测端实际跑的那个模型"，从它导出才能保证
        图上买卖点与实盘行为同源。装不进 Pine 时留空 + 写明原因，
        由前端显式提示——绝不回落到别的策略模板（那是假代码）。
        """
        from .pine_export import export_pine_from_model
        return export_pine_from_model(self.zoo._flat_path("strategy_drl"), "rl_evolve")

    async def _register_rl_evolve(self) -> None:
        """把当前 strategy_drl 最佳（flat）模型注册为 rl_evolve 策略并落库。"""
        episodes = getattr(settings, "evolve_strategy_drl_episodes", 8)
        # Pine 导出是同步文件读 + 字符串拼装（潜在几百 KB），放线程池防阻塞事件循环
        pine_code, pine_note = await asyncio.to_thread(self._export_rl_evolve_pine)
        if pine_note:
            log.warning("[evolve] rl_evolve 无法导出 Pine：%s", pine_note)
        spec = {
            "name": "rl_evolve",
            "title": f"RL持续进化·{settings.default_symbol}",
            "description": f"持续进化引擎训练的强化学习策略（{episodes}轮/轮）",
            "logic": "MDP建模+策略梯度训练，持续进化自动续训",
            "executor": "rl_adaptive",
            "param_schema": RLAdaptiveStrategy.param_schema,
            "params": {"model_path": str(self.zoo._flat_path("strategy_drl")), "buy_zone": 0.02},
            "risk_tips": ["持续进化模型，自动回退保护"],
            "created_by": "evolve_engine",
            "version": f"v{self.zoo.best_version('strategy_drl') or 0}",
            "base_symbol": settings.default_symbol,
            "base_timeframe": self._effective_timeframe(),
            "pine_code": pine_code,
            "pine_note": pine_note,
        }
        register_dynamic("rl_evolve", spec)
        await self._upsert_ai_strategy("rl_evolve", spec)
        log.info("[evolve] 策略 %s 已注册（pine %s）",
                 "rl_evolve", f"{len(pine_code)}B" if pine_code else "不可导出")

    # ---- 进化组合因子策略部署（Task：进化因子使用场景）----

    def _combo_strategy_name(self, symbol: str) -> str:
        """进化组合因子策略名：evolve_combo_<symbol>，/ 替换为 _，与级联文件命名一致）。"""
        return f"evolve_combo_{symbol.replace('/', '_')}"

    async def _deploy_factor_strategy(self, symbol: str, weights: dict,
                                      report: Optional[dict],
                                      meta: Optional[dict] = None) -> dict:
        """把进化产出的组合因子权重部署为 factor_signal 组合策略（幂等覆盖）。
        spec 落 AiStrategy 表（重启后由 dynamic_store.restore_from_db 恢复）；
        同标的重复部署 = 覆盖更新同名 spec，version 递增。返回 {name, version, spec}。
        """
        from strategies.factor_signal import FactorSignalStrategy
        name = self._combo_strategy_name(symbol)
        prev = get_dynamic(name)
        prev_version = 0
        if prev and prev.get("version"):
            try:
                prev_version = int(str(prev["version"]).lstrip("v"))
            except ValueError:
                prev_version = 0
        version = f"v{prev_version + 1}"
        spec = {
            "name": name,
            "title": f"进化组合因子·{symbol}",
            "description": f"持续进化引擎因子挖掘（factor_miner）产出的组合因子策略（{symbol}）",
            "logic": "RL因子挖掘选出因子组合，按IC方向加权合成，z-score后与阈值比较产生买卖信号",
            "executor": "factor_signal",
            "param_schema": FactorSignalStrategy.param_schema,
            "params": {
                "factor": "combo",
                "combo_spec": json.dumps(weights, ensure_ascii=False),
                "mode": "trend",
                "buy_threshold": 0.0,
                "sell_threshold": 0.0,
            },
            "risk_tips": ["因子IC可能衰减，因子库下线后该因子自动跳过"],
            "created_by": "evolve_engine",
            "version": version,
            "base_symbol": symbol,
            "base_timeframe": self._effective_timeframe(),
            "evolve_meta": {
                "fitness": (meta or {}).get("fitness"),
                "oos_report": report,
                "selected_factors": (meta or {}).get("selected_factors", []),
                "deployed_at": time.time(),
                "round_no": (meta or {}).get("round_no"),
                "backtest": None,
            },
        }
        register_dynamic(name, spec)
        await self._upsert_ai_strategy(name, spec)
        log.info("[evolve] 进化组合因子策略 %s v%s 已部署（%d 个因子）",
                 name, version, len(weights))
        return {"name": name, "version": version, "spec": spec}

    async def _refresh_factor_strategy_backtest(self, name: str, symbol: str,
                                                df, version: str) -> None:
        """后台跑一次部署策略的回测，把摘要写回 factor_miner 状态（仅展示用）。

        用训练轮同一份 df（rolling_window 根K线）+ 部署策略参数回测。
        跨轮竞态守卫：写回时校验 combo_strategy/combo_version 仍是本轮部署
        （防旧轮慢任务清掉新一轮部署错误、用旧摘要覆盖新摘要）；不归属则丢弃。
        异常自吞，只记独立键 combo_backtest_error（与部署语义 combo_deploy_error 分离）。
        """

        def _owned_now() -> bool:
            # 状态必须在写回时刻读取：任务可能跨过多轮部署，开头读一次不够
            cur_name = self._factor_miner_status.get("combo_strategy")
            cur_ver = self._factor_miner_status.get("combo_version")
            return cur_name == name and cur_ver == version

        try:
            from backtest.engine import BacktestConfig, run_backtest
            spec = get_dynamic(name) or {}
            cfg = BacktestConfig(
                symbol=symbol,
                timeframe=self._effective_timeframe(),
                strategy_name=name,
                strategy_params=dict(spec.get("params", {})),
                start_cash=10000.0, fee_rate=0.001, slippage=0.0005,
            )
            res = await asyncio.to_thread(run_backtest, df, cfg)
            m = res.get("metrics") or {}
            summary = {
                "total_ret": float(m.get("total_return", 0.0)),
                "max_drawdown": float(m.get("max_drawdown", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "trades": int(m.get("total_trades", 0)),
                "benchmark_ret": float((res.get("benchmark") or {}).get("buy_hold_ret", 0.0)),
                "at": time.time(),
            }
            if _owned_now():
                self._factor_miner_status["combo_backtest"] = summary
                self._factor_miner_status["combo_backtest_error"] = ""
                log.info("[evolve] %s 回测摘要已刷新: %s", name, summary)
            else:
                log.debug("[evolve] %s v%s 回测摘要过期（当前部署非本轮），丢弃", name, version)
        except Exception as e:  # noqa: BLE001
            if _owned_now():
                self._factor_miner_status["combo_backtest_error"] = f"回测摘要失败: {e}"
                log.warning("[evolve] %s 回测摘要失败: %s", name, e)
            else:
                log.debug("[evolve] %s v%s 回测失败但部署已换代，丢弃错误", name, version)

    async def _restore_evolve_strategies(self) -> None:
        """启动时按已训练模型补齐动态策略注册（幂等）。

        此前 rl_evolve 只在「训练成功且未被回退/拦截」时注册——一旦多轮回退
        （新模型 fitness 低于历史最佳×阈值即回退，是常态），训练产物永远不出现在
        策略库，其他模块（回测/实盘/统一策略）无法使用持续进化成果。
        """
        from strategies import get_dynamic
        if self.zoo._flat_path("strategy_drl").exists() and get_dynamic("rl_evolve") is None:
            await self._register_rl_evolve()

    async def _restore_episode_counts(self) -> None:
        """重启后从 evolve_rounds 恢复各管线已完成轮次（episode 计数）。

        训练轮次是累计事实：持续进化跨会话/跨重启保持训练时，轮次显示应接续
        而不是归零。数据库 EvolveRound 表以 round_no 顺序落库每轮结果，这里取
        各模型的最大 round_no 作为已完成的轮次数（无记录则保持 0）。
        """
        from sqlalchemy import func, select
        from core.database import EvolveRound
        try:
            async with self.db.session() as s:
                rows = (await s.execute(
                    select(EvolveRound.model, func.max(EvolveRound.round_no))
                    .group_by(EvolveRound.model))).all()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 读取历史轮次失败（轮次从 0 开始）: %s", e)
            return
        counts = {model: (max_round or 0) for model, max_round in rows}
        self._factor_miner_status["episode"] = counts.get("factor_miner", 0)
        self._strategy_drl_status["episode"] = counts.get("strategy_drl", 0)
        self._meta_controller_status["episode"] = counts.get("meta_controller", 0)
        # 连续拒绝数同样要跨重启接续：锁死是"上一轮还没解开"的状态，
        # 重启不该让它看起来像刚恢复正常
        try:
            # 锚点重置是账界线：重置前的连续拒绝针对的是已作废的基线，重启后不该把它
            # 们算回来——否则横幅会显示一个"没人能追平"的 streak（现场实测 35/10）。
            # EvolveRound.ts 是 SQLite CURRENT_TIMESTAMP（naive UTC），故这里转成同基准。
            cutoffs: dict[str, Any] = {}
            for m in self._models:
                rst = (self.zoo.best_info(m).get("reset") or {})
                if rst.get("timestamp"):
                    cutoffs[m] = datetime.fromtimestamp(
                        rst["timestamp"], tz=timezone.utc).replace(tzinfo=None)
            async with self.db.session() as s:
                # P2-9：按模型分查 + LIMIT（连续 streak 在第 1 条 ok/锚点重置处即终止，
                # 每模型最多需要 stall_rounds+1 条；此前无 WHERE/LIMIT 全表载入，长期
                # 运行后重启耗时会随 evolve_rounds 表增长）
                hist: list[Any] = []
                for m in self._models:
                    rows_m = (await s.execute(
                        select(EvolveRound.model, EvolveRound.status, EvolveRound.round_no,
                               EvolveRound.ts)
                        .where(EvolveRound.model == m)
                        .order_by(EvolveRound.round_no.desc())
                        .limit(int(self._stall_rounds) + 1))).all()
                    hist.extend(rows_m)
            seen: set[str] = set()
            streaks: dict[str, int] = {m: 0 for m in self._models}
            for model, st, _rn, ts in hist:  # 已按 round_no 倒序
                if model not in streaks or model in seen:
                    continue
                cutoff = cutoffs.get(model)
                if cutoff is not None and ts is not None and ts < cutoff:
                    seen.add(model)
                    continue
                if st == "ok":
                    seen.add(model)
                    continue
                if st == "demo_blocked":
                    # run24 修复：与运行时 _log_round 口径一致——demo_blocked
                    # 行不计入连续拒绝（P1-3 修复只覆盖了运行时路径，重启恢复
                    # 曾把交易所故障期计入 streak，误报 stalled/诱导重置锚点）
                    seen.add(model)
                    continue
                streaks[model] += 1
            self._reject_streak.update(streaks)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 读取连续拒绝历史失败: %s", e)
        log.info("[evolve] 已恢复训练轮次: 因子挖掘=%d 策略DRL=%d 元策略=%d"
                 "（连续未接受: %s）",
                 self._factor_miner_status["episode"],
                 self._strategy_drl_status["episode"],
                 self._meta_controller_status["episode"],
                 "/".join(str(self._reject_streak.get(m, 0)) for m in self._models))

    async def set_enabled(self, enabled: bool) -> None:
        """启动/禁用持续进化（运行时热切换）。"""
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled and self._running:
            # 从禁用状态切换为启用：补齐缺失的训练循环。
            # P4-A2：三条管线都要恢复（此前遗漏 meta_controller_loop → 热切换后
            # 元策略控制器永久停训，直到进程重启）。现在改为"缺哪条补哪条"，
            # 判定不再依赖 `not self._tasks`——保活监督自身也占一个任务槽，
            # 拿列表空否当开关会让热切换漏建整组循环。
            created = self._spawn_loops()
            self._spawn_supervisor()
            self._supervise_status["alive"] = True
            if created:
                log.info("[evolve] 持续进化已启用（补齐 %d 条训练循环，保活监督已就位）", created)
        elif not enabled:
            # 禁用：取消所有训练循环（正在训练的任务会被取消，但 executor 线程
            # 会跑完——等待其结束再返回，避免"已禁用仍在后台烧 CPU"）
            for t in self._tasks:
                t.cancel()
            self._tasks.clear()
            self._supervise_status["alive"] = False
            await self._wait_training_done()
            log.info("[evolve] 持续进化已禁用")

    async def pause(self) -> None:
        """暂停训练（当前轮完成后停止新轮）。"""
        self._paused = True
        log.info("[evolve] 持续训练已暂停")

    async def resume(self) -> None:
        """恢复训练。"""
        self._paused = False
        log.info("[evolve] 持续训练已恢复")

    async def trigger_factor_miner(self) -> dict:
        """手动触发一次因子挖掘训练（同步执行，不阻塞太久需容忍）。"""
        if not self._running:
            return {"ok": False, "reason": "持续进化引擎未运行"}
        # P2：忙态守卫——另一条训练正在进行时直接拒绝（此前连点会在
        # _train_lock 上排队，连续执行数轮完整训练）
        if self._train_lock.locked():
            return {"ok": False, "busy": True, "reason": "另一条训练正在进行中，请稍后再试"}
        async with self._train_lock:
            self._factor_miner_status["active"] = True
            try:
                self._factor_miner_status["last_error"] = ""
                _res = await self._train_factor_miner_once(force=True)
                self._factor_miner_status["last_run"] = time.time()
                if _res == "demo":
                    # 演示数据轮：降级不算成功（last_error 已写明故障）
                    self._factor_miner_status["last_success"] = False
                    return {"ok": True, "model": "factor_miner", "demo": True}
                self._factor_miner_status["last_success"] = True
                return {"ok": True, "model": "factor_miner"}
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)[:200]
                self._factor_miner_status["last_error"] = err
                self._factor_miner_status["last_success"] = False
                return {"ok": False, "error": err}
            finally:
                self._factor_miner_status["active"] = False

    async def trigger_strategy_drl(self) -> dict:
        """手动触发一次策略 DRL 训练（同步执行）。"""
        if not self._running:
            return {"ok": False, "reason": "持续进化引擎未运行"}
        # P2：忙态守卫——另一条训练正在进行时直接拒绝（防连点排队执行多轮）
        if self._train_lock.locked():
            return {"ok": False, "busy": True, "reason": "另一条训练正在进行中，请稍后再试"}
        async with self._train_lock:
            self._strategy_drl_status["active"] = True
            try:
                self._strategy_drl_status["last_error"] = ""
                _res = await self._train_strategy_drl_once(force=True)
                self._strategy_drl_status["last_run"] = time.time()
                if _res == "demo":
                    self._strategy_drl_status["last_success"] = False
                    return {"ok": True, "model": "strategy_drl", "demo": True}
                self._strategy_drl_status["last_success"] = True
                return {"ok": True, "model": "strategy_drl"}
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)[:200]
                self._strategy_drl_status["last_error"] = err
                self._strategy_drl_status["last_success"] = False
                return {"ok": False, "error": err}
            finally:
                self._strategy_drl_status["active"] = False

    async def trigger_meta_controller(self) -> dict:
        """手动触发一次元策略控制器训练（同步执行）。"""
        if not self._running:
            return {"ok": False, "reason": "持续进化引擎未运行"}
        # P2：忙态守卫——另一条训练正在进行时直接拒绝（防连点排队执行多轮）
        if self._train_lock.locked():
            return {"ok": False, "busy": True, "reason": "另一条训练正在进行中，请稍后再试"}
        async with self._train_lock:
            self._meta_controller_status["active"] = True
            try:
                self._meta_controller_status["last_error"] = ""
                _res = await self._train_meta_controller_once(force=True)
                self._meta_controller_status["last_run"] = time.time()
                self._meta_controller_status["last_success"] = _res is not False
                return {"ok": True, "model": "meta_controller"}
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)[:200]
                self._meta_controller_status["last_error"] = err
                self._meta_controller_status["last_success"] = False
                return {"ok": False, "error": err}
            finally:
                self._meta_controller_status["active"] = False

    def _pipeline_view(self, model: str, base: dict, anchor: dict) -> dict:
        """管线状态 + 锁死自检（episode 只说明"训了几轮"，不说明"有没有一轮被接受"）。

        P4-E1：anchor 由 status() 预取传入——每个模型每轮 status() 只调一次
        best_info，此前 _pipeline_view 内又读一次（前端秒级轮询时重复磁盘 I/O）。
        """
        out = dict(base)
        streak = int(self._reject_streak.get(model, 0))
        out["reject_streak"] = streak
        out["rounds_total"] = int(self._rounds_total.get(model, 0))
        stalled = streak >= self._stall_rounds
        out["stalled"] = stalled
        # 真锁死 = 连续未接受 且 被一份"不可能追上"的基线压住。
        # 只有这种情况该给「重置锚点」；同口径下的连续拒绝是正常保护，
        # 而无锚点时回退守卫根本不生效（此时 streak 只是重置前的历史）。
        out["anchor_locked"] = bool(stalled and anchor.get("fitness") is not None
                                    and not anchor.get("comparable"))
        out["stall_reason"] = ""
        # 部署/候选画像：candidate vs deployed 分离（Task 2 契约）
        _ds = self._deploy_status.get(model, {})
        out["candidate_fitness"] = _ds.get("candidate_fitness")
        out["deployed_fitness"] = _ds.get("deployed_fitness")
        out["deployed_version"] = _ds.get("deployed_version")
        out["deployed_source"] = _ds.get("deployed_source") or ""
        out["last_outcome"] = _ds.get("last_outcome") or ""
        out["anchor_mismatch"] = _ds.get("anchor_mismatch") or ""
        out["runtime_reload"] = _ds.get("runtime_reload") or "skipped"
        if stalled:
            _anchor_txt = ("无锚点" if anchor.get("fitness") is None
                           else f"fitness={anchor['fitness']}"
                                f"({'demo 合成数据' if anchor.get('data_source') == 'demo' else '真实数据'})")
            if anchor.get("fitness") is None:
                out["stall_reason"] = (f"当前无回退基线（已重置或尚无模型通过检验），"
                                       f"此前连续 {streak} 轮未接受不计入锁死判定；"
                                       f"下一轮起重新建立锚点")
            elif not anchor.get("comparable"):
                out["stall_reason"] = (f"连续 {streak} 轮无模型被接受：{anchor.get('reason') or '锚点不可比'}"
                                       f"（当前锚点 {_anchor_txt}）")
            else:
                out["stall_reason"] = (f"连续 {streak} 轮无模型被接受，均低于当前锚点"
                                       f"（{_anchor_txt}）或被 OOS 硬门拦截")
        return out

    def _mark_deployment_status(self, model: str, *, outcome: str = "",
                                candidate_fitness: Optional[float] = None,
                                runtime_reload: str = "skipped",
                                anchor_mismatch: str = "") -> None:
        """同步管线部署画像：deployed_* 一律取自 best 快照（zoo 为准），
        candidate_fitness 由训练轮单独写入，二者分离防止回退轮状态污染。"""
        status = self._deploy_status.setdefault(model, {
            "candidate_fitness": None, "deployed_fitness": None,
            "deployed_version": None, "deployed_source": "",
            "last_outcome": "", "anchor_mismatch": "",
            "runtime_reload": "skipped",
        })
        if outcome:
            status["last_outcome"] = outcome
        if candidate_fitness is not None:
            status["candidate_fitness"] = candidate_fitness
        if runtime_reload != "skipped":
            status["runtime_reload"] = runtime_reload
        if anchor_mismatch:
            status["anchor_mismatch"] = anchor_mismatch
        info = self.zoo.best_info(model)
        status["deployed_version"] = info.get("version")
        status["deployed_fitness"] = info.get("fitness")
        status["deployed_source"] = info.get("data_source") or ""

    async def _register_deployed_model(self, name: str) -> None:
        """注册当前部署模型为动态策略（与部署快照同源）。"""
        if name == "strategy_drl":
            await self._register_rl_evolve()
        elif name == "meta_controller":
            from strategies import get_dynamic
            sub_strategies = resolve_meta_sub_strategies(
                (get_dynamic("meta_controller") or {}).get("params"))
            from strategies.meta import MetaController as _MetaControllerCls
            spec = {
                "name": "meta_controller",
                "title": "元策略控制器",
                "description": "DRL 元控制器：在子策略间按学习到的置信度动态选择",
                "logic": "元级 PPO，状态含子策略信号+置信度+历史胜率",
                "executor": "meta_controller",
                "param_schema": _MetaControllerCls.param_schema,
                "params": {
                    "sub_strategies": sub_strategies,
                    "mode": ((get_dynamic("meta_controller") or {})
                             .get("params") or {}).get("mode") or "drl",
                    "model_path": str(self.zoo._flat_path("meta_controller")),
                },
                "risk_tips": ["元策略依赖子策略质量，建议定期重训"],
                "created_by": "evolve_engine",
                "version": f"v{self.zoo.best_version('meta_controller') or 0}",
                "base_symbol": self._effective_symbol() if hasattr(self, "_effective_symbol") else "",
                "base_timeframe": self._effective_timeframe(),
                "pine_code": "",
                "pine_note": ("元策略控制器在子策略间动态切换，无等价 Pine 模板；"
                              "如需图上买卖点请对单个子策略导出 Pine"),
            }
            register_dynamic("meta_controller", spec)
            await self._upsert_ai_strategy("meta_controller", spec)

    async def _audit_manual_rollback(self, name: str, version: int, *, reason: str,
                                     runtime_reload: dict, registered: bool) -> None:
        """手动回退的审计轮次：目标/源版本、原因、运行时结果（不阻断主流程）。"""
        try:
            await self._log_round(
                name, "", self._effective_timeframe(), "manual", 0.0,
                status="manual_rollback",
                audit={"target_version": version,
                       "reason": reason or "",
                       "runtime_reload": runtime_reload.get("status", "skipped"),
                       "registered": registered})
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 手动回退审计落库失败(%s v%d): %s", name, version, e)

    async def deploy_model(self, name: str, version: int, *,
                           outcome: str, symbol: str = "",
                           timeframe: str = "", reason: str = "") -> dict:
        """统一部署入口：磁盘快照恢复 → 动态策略注册 → 运行时 reload。

        自动接受、自动回退、手动回退共用本入口。任何一层失败都不伪报成功：
        reload 失败时尝试恢复部署前的 best 快照，仍失败则保留现场并返回错误。
        _train_lock 由调用方在必要时持有；本方法不重复加锁。
        """
        if name not in self._models:
            return {"ok": False, "error": f"未知模型 {name}，可选：{'/'.join(self._models)}",
                    "version": version, "meta": {}, "runtime_reload": {},
                    "registered": False}
        zoo = self.zoo
        old_best = zoo.best_version(name)
        try:
            restored = await asyncio.to_thread(
                zoo.restore_deployment, name, version,
                reason=reason or outcome, outcome=outcome)
        except Exception as e:  # noqa: BLE001
            self._mark_deployment_status(name, outcome="failed", runtime_reload="failed")
            log.warning("[evolve] 部署快照恢复失败(%s v%d): %s", name, version, e)
            return {"ok": False, "version": version, "meta": {},
                    "runtime_reload": {"status": "failed", "error": str(e)},
                    "registered": False, "error": str(e)}
        deployed_meta = restored.get("meta") or {}
        registered = True
        try:
            await self._register_deployed_model(name)
        except Exception as e:  # noqa: BLE001
            registered = False
            log.warning("[evolve] 部署后策略注册失败(%s): %s", name, e)

        if self._deployment_reload is None:
            self._mark_deployment_status(name, outcome=outcome,
                                         runtime_reload="skipped")
            if outcome == "manual_rollback":
                await self._audit_manual_rollback(
                    name, version, reason=reason,
                    runtime_reload={"status": "skipped"}, registered=registered)
            log.info("[evolve] 部署完成(%s v%d, 无运行时回调, outcome=%s)",
                     name, version, outcome)
            return {"ok": True, "version": version, "meta": deployed_meta,
                    "runtime_reload": {"status": "skipped"},
                    "registered": registered, "error": None}
        try:
            result = await self._deployment_reload(name, version, deployed_meta)
        except Exception as e:  # noqa: BLE001
            result = {"status": "failed", "error": str(e)}
        if result and result.get("status") == "reloaded":
            self._mark_deployment_status(name, outcome=outcome,
                                         runtime_reload="reloaded")
            if outcome == "manual_rollback":
                await self._audit_manual_rollback(
                    name, version, reason=reason,
                    runtime_reload=result, registered=registered)
            log.info("[evolve] 部署完成(%s v%d, 运行实例已热加载, outcome=%s)",
                     name, version, outcome)
            return {"ok": True, "version": version, "meta": deployed_meta,
                    "runtime_reload": result, "registered": registered,
                    "error": None}
        # reload 失败：恢复到部署前的版本，不留下"文件已回退、运行时未回退"
        # 的半成功状态。恢复失败时保留现场并如实报告。
        runtime_status = dict(result or {"status": "failed"})
        if old_best and old_best != version:
            try:
                await asyncio.to_thread(
                    zoo.restore_deployment, name, old_best,
                    reason=f"reload失败恢复({runtime_status.get('error', '')})",
                    outcome=outcome)
            except Exception as e:  # noqa: BLE001
                log.error("[evolve] reload 失败后恢复 %s v%d 也失败: %s",
                          name, old_best, e)
        self._mark_deployment_status(name, outcome="failed",
                                     runtime_reload="failed")
        if outcome == "manual_rollback":
            await self._audit_manual_rollback(
                name, version, reason=reason,
                runtime_reload=runtime_status, registered=registered)
        log.warning("[evolve] 部署失败(%s v%d): reload 未成功 %s",
                    name, version, runtime_status)
        return {"ok": False, "version": version, "meta": deployed_meta,
                "runtime_reload": runtime_status, "registered": registered,
                "error": runtime_status.get("error") or "运行时模型热加载未完成"}

    def status(self) -> dict:
        """返回当前训练状态。"""
        tasks_alive = {t.get_name(): not t.done() for t in self._tasks}
        # P4-E1：锚点画像只读一次（_pipeline_view 与 anchor 字段共用），
        # 消除秒级轮询下的重复 meta.json 磁盘读
        anchors = {m: self.zoo.best_info(m) for m in self._models}
        # 保活视角：应当存活的训练循环里，有哪几条已经不在了。
        # 「引擎开着但已停训」此前完全不可观测（tasks_alive 算了却无人消费），
        # 这里把它升级为带判定的健康信号，前端/AI 都能据此判断是否真的在跑。
        _now = time.time()
        loops_alive: dict[str, bool] = {}
        for _n in self._loop_names:
            _t = self._find_task(_n)
            loops_alive[_n] = bool(_t is not None and not _t.done())
        _expected = bool(self._running and self._enabled)
        _last_check = float(self._supervise_status.get("last_check") or 0.0)
        supervisor = {
            **self._supervise_status,
            "loops_alive": loops_alive,
            "restarts_by_loop": dict(self._restarts),
            "heartbeat_age": (_now - _last_check) if _last_check else None,
            "expected_running": _expected,
            "degraded": bool(_expected and not all(loops_alive.values())),
        }
        return {
            "running": self._running,
            "paused": self._paused,
            "enabled": self._enabled,
            "symbols": self._symbols,
            "current_symbol": self._symbols[self._symbol_idx % len(self._symbols)],
            "timeframe": self._effective_timeframe(),
            "factor_miner": self._pipeline_view("factor_miner", self._factor_miner_status, anchors["factor_miner"]),
            "strategy_drl": self._pipeline_view("strategy_drl", self._strategy_drl_status, anchors["strategy_drl"]),
            "meta_controller": self._pipeline_view("meta_controller", self._meta_controller_status, anchors["meta_controller"]),
            # 锚点（回退基线）画像：fitness 数值 + 口径凭证，前端据此解释"为什么不接受"
            "anchor": anchors,
            "stall_rounds": self._stall_rounds,
            "factor_miner_interval": self._factor_miner_interval,
            "strategy_drl_interval": self._strategy_drl_interval,
            "meta_controller_interval": getattr(settings, "evolve_meta_interval", 600),
            "rolling_window": self._rolling_window,
            "rollback_threshold": self._rollback_threshold,
            "tasks_alive": tasks_alive,
            # 保活状态：巡检心跳、各循环存活情况、自愈次数与降级判定
            "supervisor": supervisor,
        }

    async def reset_anchor(self, name: str, reason: str = "") -> dict:
        """重置锚点（归档式，不删权重）：解锁被不可比基线锁死的进化循环。

        P4-E1：与训练写路径互斥（_train_lock）——否则 reset_anchor 读 meta 后
        后台训练又写回 meta（追加版本记录），旧快照 replace 覆盖导致版本记录
        丢失更新；zoo.reset_anchor 含 gzip/复制写盘，放 to_thread 防阻塞事件循环。
        """
        if name not in self._models:
            return {"ok": False, "error": f"未知模型 {name}，可选：{'/'.join(self._models)}"}
        async with self._train_lock:
            try:
                out = await asyncio.to_thread(self.zoo.reset_anchor, name, reason=reason)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 锚点重置失败(%s): %s", name, e)
                return {"ok": False, "error": str(e)}
        self._reject_streak[name] = 0
        try:
            from core.notify import notify
            cleared = out.get("cleared") or {}
            await notify(self.db, "进化引擎：锚点已重置",
                         f"{name} 的回退基线已清除（原 fitness={cleared.get('fitness')} "
                         f"来源={cleared.get('data_source') or '未知'}），"
                         f"下一条通过 OOS 检验的模型将重建基线。留档：{out.get('stamp')}")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, **out, "anchor": self.zoo.best_info(name)}

    @property
    def config_keys(self) -> dict:
        """当前可配置项（P2-12 超参面板 + 「指定品种/周期」落地。训练循环读的是
        self 缓存属性与 settings 字段，这里统一收口作为面板编辑的数据源）。"""
        return {
            "symbols": list(self._symbols),
            "timeframe": getattr(settings, "evolve_timeframe", "") or settings.default_timeframe,
            "factor_miner_interval": int(self._factor_miner_interval),
            "strategy_drl_interval": int(self._strategy_drl_interval),
            "rolling_window": int(self._rolling_window),
            "rollback_threshold": float(self._rollback_threshold),
            "factor_miner_episodes": int(getattr(settings, "evolve_factor_miner_episodes", 8)),
            "strategy_drl_episodes": int(getattr(settings, "evolve_strategy_drl_episodes", 8)),
            "meta_episodes": int(getattr(settings, "evolve_meta_episodes", 8)),
            "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
            "oos_min_bars": int(getattr(settings, "evolve_oos_min_bars", 250)),
            "cross_symbol_oos": bool(getattr(settings, "evolve_cross_symbol_oos", True)),
            "min_new_bars": int(getattr(settings, "evolve_min_new_bars", 2)),
            "train_time_budget": float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0),
            "min_train_gap_sec": int(getattr(settings, "evolve_min_train_gap_sec", 0) or 0),
            "train_on_demo": bool(getattr(settings, "evolve_train_on_demo", False)),
            "anchor_max_fitness": float(getattr(settings, "evolve_anchor_max_fitness", 50.0) or 0.0),
            "strategy_drl_population": bool(getattr(settings, "evolve_strategy_drl_population", True)),
        }

    async def apply_config(self, changes: dict) -> dict:
        """热更新可配置项（不重启生效）：写引擎缓存属性 + settings 字段，
        并持久化回 config.yaml（用户在面板上改的配置重启后保留）。

        'symbols'：允许单元素列表 → 固定品种训练（替代默认轮换）。
        'timeframe'：训练时间周期（空/缺省时回落 settings.default_timeframe）。
        返回 {ok, applied: {...}, persisted: bool}。

        事务性（P1-1/P1-2 修复）：先对全部变更做校验（含 timeframe 白名单），
        全部通过后才一次性写入 settings/引擎属性并持久化；任何一项失败返回
        错误且不产生副作用——此前 symbols/timeframe 先赋值、后续字段校验失败
        不回滚，造成「前端提示失败但引擎已在用新配置」的状态分裂。
        """
        import shutil
        from backtest.data_loader import _BINANCE_TF  # 合法周期白名单（唯一权威来源）
        applied: dict = {}
        allowed = set(self.config_keys)
        unknown = sorted(set(changes) - allowed)
        if unknown:
            return {"ok": False, "error": f"未知配置项: {', '.join(unknown)}",
                    "applied": applied}
        # 1) 全部校验 → pending 清单（不产生任何副作用）
        pending: list[tuple[str, Any, Optional[str] | None]] = []  # (settings_attr, value, engine_attr)

        # 训练标的池：支持"单元素 = 固定品种训练"。
        # P4-E1：空数组视为非法输入（清空标的池会让三管线无从取数），
        # 明确报错而不是静默 no-op；此前 `if changes["symbols"]:` 把空池整体跳过，
        # 前端却提示"保存成功"。
        if "symbols" in changes:
            raw_symbols = changes["symbols"]
            if not isinstance(raw_symbols, list) or not raw_symbols or len(raw_symbols) > 50:
                return {"ok": False, "error": "symbols 必须是 1-50 个交易对的列表",
                        "applied": applied}
            syms = []
            for raw_symbol in raw_symbols:
                if not isinstance(raw_symbol, str):
                    return {"ok": False, "error": "symbols 必须全部是字符串",
                            "applied": applied}
                symbol = raw_symbol.strip().upper()
                if not symbol or "/" not in symbol or len(symbol) > 40:
                    return {"ok": False, "error": f"非法交易对: {raw_symbol!r}",
                            "applied": applied}
                syms.append(symbol)
            pending.append(("evolve_symbols", syms, "_symbols"))
        # 训练时间周期：independent of default_timeframe（P2-12 用户主诉求落地）。
        # P4-E1：timeframe 为 null/空串 = 前端「跟随默认」——清除独立周期、回落
        # default_timeframe。此前 str(None)="None" 恒真值，把 "none" 写进 settings，
        # 非法周期导致拉数失败退化 demo 且面板永久显示错误周期。
        # P1-2：非法周期在这里用 _BINANCE_TF 白名单拦截（此前注释声称"持久化前
        # 兜底"但全仓库根本不存在该白名单，非法值被静默持久化，重启后整个进化
        # 链路持续 demo 空转）。
        if "timeframe" in changes:
            tf_raw = changes.get("timeframe")
            tf = str(tf_raw).strip().lower() if tf_raw not in (None, "") else ""
            if tf:
                if tf not in _BINANCE_TF:
                    return {"ok": False,
                            "error": f"不支持的训练时间周期 {tf!r}，可选：{'/'.join(sorted(_BINANCE_TF))}",
                            "applied": applied}
                pending.append(("evolve_timeframe", tf, None))
            else:
                # 显式清除：回落默认周期
                pending.append(("evolve_timeframe", "", None))

        def _check(name: str, settings_attr: str, engine_attr: str = None,
                   conv=None, min_v: float = None, max_v: float = None) -> None:
            if name not in changes or changes[name] is None:
                return
            val = changes[name]
            if isinstance(val, bool) and conv is not None:
                raise ValueError(f"{name} 必须是数字，收到: {changes[name]!r}")
            if conv is not None:
                try:
                    val = conv(val)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} 必须是数字，收到: {changes[name]!r}")
            if min_v is not None and val < min_v:
                raise ValueError(f"{name} 不能小于 {min_v}，收到: {val}")
            if max_v is not None and val > max_v:
                raise ValueError(f"{name} 不能大于 {max_v}，收到: {val}")
            pending.append((settings_attr, val, engine_attr))

        try:
            _check("factor_miner_interval", "evolve_factor_miner_interval", "_factor_miner_interval",
                   conv=int, min_v=1)
            _check("strategy_drl_interval", "evolve_strategy_drl_interval", "_strategy_drl_interval",
                   conv=int, min_v=1)
            _check("rolling_window", "evolve_rolling_window", "_rolling_window",
                   conv=lambda v: max(1000, int(v)), min_v=1000)
            _check("rollback_threshold", "evolve_rollback_threshold", "_rollback_threshold",
                   conv=float, min_v=0.0, max_v=1.0)
            _check("factor_miner_episodes", "evolve_factor_miner_episodes", conv=int, min_v=1, max_v=500)
            _check("strategy_drl_episodes", "evolve_strategy_drl_episodes", conv=int, min_v=1, max_v=500)
            _check("meta_episodes", "evolve_meta_episodes", conv=int, min_v=1, max_v=500)
            _check("vol_penalty", "evolve_vol_penalty", conv=float, min_v=0.0, max_v=1000.0)
            _check("oos_min_bars", "evolve_oos_min_bars", conv=int, min_v=50, max_v=100000)
            _check("min_new_bars", "evolve_min_new_bars", conv=int, min_v=0, max_v=100000)
            _check("train_time_budget", "evolve_train_time_budget", conv=float, min_v=0.0, max_v=86400.0)
            _check("min_train_gap_sec", "evolve_min_train_gap_sec", conv=int, min_v=0, max_v=604800)
            _check("anchor_max_fitness", "evolve_anchor_max_fitness", conv=float, min_v=0.0, max_v=1e9)
            if "train_on_demo" in changes and changes["train_on_demo"] is not None:
                if not isinstance(changes["train_on_demo"], bool):
                    raise ValueError("train_on_demo 必须是布尔值")
                pending.append(("evolve_train_on_demo", changes["train_on_demo"], None))
            if "strategy_drl_population" in changes and changes["strategy_drl_population"] is not None:
                if not isinstance(changes["strategy_drl_population"], bool):
                    raise ValueError("strategy_drl_population 必须是布尔值")
                pending.append(("evolve_strategy_drl_population", changes["strategy_drl_population"], None))
            if "cross_symbol_oos" in changes and changes["cross_symbol_oos"] is not None:
                if not isinstance(changes["cross_symbol_oos"], bool):
                    raise ValueError("cross_symbol_oos 必须是布尔值")
                pending.append(("evolve_cross_symbol_oos", changes["cross_symbol_oos"], None))
        except ValueError as e:
            log.warning("[evolve] 配置校验拒绝: %s", e)
            return {"ok": False, "error": str(e), "applied": applied}

        # 2) 全部通过后一次性生效
        for settings_attr, val, engine_attr in pending:
            setattr(settings, settings_attr, val)
            if engine_attr is not None:
                setattr(self, engine_attr, val)
            applied[settings_attr] = val
        if "evolve_symbols" in applied:
            # 标的池变更：重置轮换起点与 cycle 记账（P4-A1）
            self._symbol_idx = 0
            self._cycle_pipelines_done.clear()
            self._last_tail_ts.clear()  # 旧键对新池无意义，防跨池误判
        if "evolve_timeframe" in applied:
            self._last_tail_ts.clear()  # 增量门键含周期，切周期后旧键作废

        # 3) 持久化回 config.yaml（扁平结构，键名与 Settings 字段同名）：
        # 逐键更新顶层文档，备份后原子写入（失败不致命，仅内存生效）。
        # P1-1：此前 open("w") 直写非原子，写一半崩溃会损坏配置；改 tmp+rename。
        persisted = False
        from config.settings import ROOT as _SETTINGS_ROOT
        yaml_cfg_path = _SETTINGS_ROOT / "config" / "config.yaml"
        if yaml_cfg_path.exists():
            import yaml
            try:
                with open(str(yaml_cfg_path), "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                doc.update(applied)
                try:
                    shutil.copy2(str(yaml_cfg_path), str(yaml_cfg_path) + ".bak")
                except Exception:  # noqa: BLE001
                    pass
                tmp_path = yaml_cfg_path.with_name(yaml_cfg_path.name + ".tmp")
                with open(str(tmp_path), "w", encoding="utf-8") as f:
                    yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
                tmp_path.replace(yaml_cfg_path)
                persisted = True
                log.info("[evolve] 配置已持久化到 config.yaml: %s", applied)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 配置持久化失败（仅内存生效）: %s", e)
        else:
            log.warning("[evolve] 未找到 config.yaml，配置仅内存生效")

        log.info("[evolve] 应用配置变更: %s", applied)
        return {"ok": True, "applied": applied, "persisted": persisted}

    def _effective_timeframe(self) -> str:
        """训练时间周期：面板独立配置优先，缺省回落 default_timeframe。"""
        return getattr(settings, "evolve_timeframe", "") or settings.default_timeframe

    # ---- 通用训练循环模板 ----

    # 训练失败后的快速重试间隔（秒）：避免失败后等 7 天/1 天才重试
    _RETRY_INTERVAL = 300  # 5 分钟

    def _run_train(self, train_callable: Callable) -> asyncio.Future:
        """在专用单线程 executor 里执行训练本体，并置位训练完成事件。

        - 单 worker 保证即便手动触发与周期循环并发，训练也绝不并行（双保险，
          与 _train_lock 同一目的）；取消协程不会中断 executor 线程（线程无法
          被 kill），训练+写盘必然完整执行，事件标志在 finally 里置位，
          stop()/set_enabled(False) 据此等待真正的训练结束。
        """
        self._training_done.clear()

        def _wrap():
            try:
                return train_callable()
            finally:
                self._training_done.set()

        return asyncio.get_running_loop().run_in_executor(self._train_executor, _wrap)

    async def _run_loop(self, status: dict, train_fn: Callable, interval: int, name: str,
                        start_delay: int = 15, pipeline: str = "",
                        yield_to: tuple = ()) -> None:
        """通用训练循环：启动 start_delay 秒后首次训练，之后按间隔（失败时 5 分钟重试）。

        pipeline: 管线标识（factor_miner/strategy_drl/meta_controller），传入后
        在每次真实训练尝试后调用 _mark_pipeline_done——三管线同标训练完成一轮才
        切换标的（P4-A1 同步修复）。增量门跳过（train_fn 返回 False，无训练发生）
        时不推进；异常也算一次尝试（防单管线长期失败导致标的永卡）。

        yield_to: 本管线应向哪些更高优先级管线让出锁（P4-C3）。拿锁前若任一
        更高优先级管线 active（正在训练），先让出重试——保证链路源头（因子挖掘）
        不被下游挤占，避免训练耗时波动时低优先级管线挨饿。
        """
        # 首次训练前等待：等待引擎初始化、WebSocket 建立连接、基础数据就位
        # 两个训练循环错峰启动，避免同时开始吃满 CPU
        try:
            await asyncio.sleep(start_delay)
        except asyncio.CancelledError:
            raise
        status["next_run"] = time.time() + start_delay
        status["loop_beat"] = time.time()
        while self._running:
            # 每轮先给 retry_interval 兜底初值：此前它只在 try 内赋值，若异常在
            # 赋值之前逸出，循环尾部的 `time.time() + retry_interval` 会抛
            # UnboundLocalError——而异常处理器已在 try 内，兜不住尾部，
            # 整条任务就此终结。这是"训练循环静默死亡"的一条真实路径。
            retry_interval = interval
            try:
                if self._paused:
                    await asyncio.sleep(10)
                    if not self._running:
                        break
                    continue
                # P4-C3：让出给更高优先级管线（因子挖掘 > 策略DRL > 元策略）。
                # 锁本身无优先级，若不主动让出，间隔相同 + 训练耗时波动时
                # 可能长时间拿不到锁。
                _yielded = False
                if yield_to:
                    for _prio in range(12):  # 最多让 12×5s=60s
                        if not self._running:
                            break
                        higher_active = any(
                            getattr(self, f"_{p}_status", {}).get("active", False)
                            for p in yield_to)
                        if not higher_active:
                            break
                        _yielded = True
                        await asyncio.sleep(5)
                # P2-10：全局训练锁——同一时刻仅一条训练管线在跑
                async with self._train_lock:
                    status["active"] = True
                    status["last_error"] = ""
                    # P2-11：train_fn 返回 False 表示"数据无新增 K 线，本轮跳过"
                    # （增量门），不更新上次训练时间/成功标志，保持等待。
                    # "demo" = 交易所故障期的演示数据轮：训练发生了但属降级，
                    # 不记 last_success（与 last_error 并存自相矛盾），也不把
                    # demo 轮算进 reject_streak（_log_round 对 demo_blocked 已特殊处理）。
                    _res = await train_fn()
                    if _res is False:
                        log.debug("[evolve] %s：无新增K线，跳过本轮", name)
                    else:
                        status["last_run"] = time.time()
                        if _res == "demo":
                            status["last_success"] = False
                            log.warning("[evolve] %s：交易所故障，本轮为演示数据训练（不记成功）", name)
                        else:
                            status["last_success"] = True
                        # 真实训练尝试完成 → 标记当前标的该管线已训练（P4-A1）
                        if pipeline:
                            self._mark_pipeline_done(pipeline)
                # 训练轮正常结束：恢复常规间隔（失败分支在上面各自改写）
                # P4-C1：交易所持续不可达（连续 demo 训练）时自动降频——
                # 故障 1 小时若仍每 60s 全量训练合成数据，纯烧 CPU 拖垮实盘。
                # 连续 demo ≥2 次后放大到 10 分钟；恢复正常（streak 清零）即恢复。
                _demo_streak = int(status.get("_demo_streak", 0) or 0)
                if _demo_streak >= 2:
                    retry_interval = max(interval * 10, 600)
                    log.info("[evolve] %s：交易所不可达连续 %d 次，训练降频至 %ds",
                             name, _demo_streak, retry_interval)
                # P4-C4：连续失败指数退避（防"长期故障下每 5 分钟空转一次"）——
                # 失败重试随连续失败次数增长：5min → 10min → 20min → 封顶 30min；
                # 成功或跳过即清零。
                status["_fail_streak"] = 0
            except asyncio.CancelledError:
                raise
            except InsufficientDataError as e:
                # 数据不足是预期业务分支：不带全栈刷屏，仍按失败退避节奏重试
                err = str(e)[:200]
                log.warning("[evolve] %s 数据不足，进入失败退避: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                _fs = int(status.get("_fail_streak", 0) or 0) + 1
                status["_fail_streak"] = _fs
                retry_interval = min(self._RETRY_INTERVAL * (2 ** min(_fs - 1, 3)), 1800)
                log.info("[evolve] %s 连续数据不足 %d 次，%d 秒后重试", name, _fs, retry_interval)
                # 异常也算一次训练尝试（防单管线长期失败导致标的永卡，P4-A1）
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                log.exception("[evolve] %s 训练异常: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                # P4-C4：连续失败退避（封顶 30 分钟），避免长期故障下高频空转
                _fs = int(status.get("_fail_streak", 0) or 0) + 1
                status["_fail_streak"] = _fs
                retry_interval = min(self._RETRY_INTERVAL * (2 ** min(_fs - 1, 3)), 1800)
                log.info("[evolve] %s 连续失败 %d 次，%d 秒后重试", name, _fs, retry_interval)
                # 异常也算一次训练尝试（防某管线长期失败导致标的永卡，P4-A1）
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, (SystemExit, KeyboardInterrupt, GeneratorExit)):
                    raise  # 真正的退出请求不该被吞掉
                # 兜底：训练本体可能抛出非 Exception 基类的错误。此前这类异常会
                # 直接终结整条循环任务，而 _running 仍是 True——外部完全看不出
                # 停训，只能靠保活监督事后重建。这里先降级为"一轮失败"，
                # 多数情况不必动用监督。
                err = f"{type(e).__name__}: {e}"[:200]
                log.error("[evolve] %s 循环遭遇非 Exception 异常，已兜底保活: %s", name, err)
                status["last_error"] = err
                status["last_success"] = False
                retry_interval = max(interval, self._RETRY_INTERVAL)
                if pipeline:
                    self._mark_pipeline_done(pipeline)
            finally:
                status["active"] = False
            status["next_run"] = time.time() + retry_interval
            status["loop_beat"] = time.time()
            await asyncio.sleep(retry_interval)

    async def _factor_miner_loop(self) -> None:
        """因子挖掘持续训练循环（最高优先级：链路源头，下游依赖其产出）。"""
        await self._run_loop(self._factor_miner_status, self._train_factor_miner_once,
                             self._factor_miner_interval, "因子挖掘", pipeline="factor_miner")

    async def _strategy_drl_loop(self) -> None:
        """策略 DRL 持续训练循环（首个训练延迟 30s；因子挖掘优先，见 yield_to）。"""
        await self._run_loop(self._strategy_drl_status, self._train_strategy_drl_once,
                             self._strategy_drl_interval, "策略DRL",
                             start_delay=30, pipeline="strategy_drl",
                             yield_to=("factor_miner",))

    async def _meta_controller_loop(self) -> None:
        """元策略控制器持续训练循环（间隔较长；因子挖掘/策略DRL 优先，见 yield_to）。"""
        meta_interval = getattr(settings, "evolve_meta_interval", 600)
        await self._run_loop(self._meta_controller_status, self._train_meta_controller_once,
                             meta_interval, "元策略", start_delay=60, pipeline="meta_controller",
                             yield_to=("factor_miner", "strategy_drl"))

    # ---- 保活（watchdog）：训练循环的统一创建与自愈 ----
    # 背景：三条训练循环是 asyncio 任务。任务一旦结束就再也不会自己回来，而
    # _running/_enabled 仍是 True，status() 照样报 running=true —— 「看起来在跑、
    # 其实已经停训」是最难发现的一类故障。此前 status() 虽然算了 tasks_alive，
    # 但全仓无人消费它（只读诊断），也没有任何重建逻辑，只能重启进程恢复。

    def _loop_factories(self) -> dict:
        """训练循环名 → 协程工厂。清单单一来源，避免多处硬编码走偏。"""
        return {
            "evolve_factor_miner": self._factor_miner_loop,
            "evolve_strategy_drl": self._strategy_drl_loop,
            "evolve_meta_controller": self._meta_controller_loop,
        }

    def _status_for(self, name: str) -> Optional[dict]:
        """训练循环名 → 对应管线的状态字典。"""
        return {
            "evolve_factor_miner": self._factor_miner_status,
            "evolve_strategy_drl": self._strategy_drl_status,
            "evolve_meta_controller": self._meta_controller_status,
        }.get(name)

    def _find_task(self, name: str) -> Optional[asyncio.Task]:
        """按任务名查任务；_tasks 里可能同时残留已终结的旧任务。"""
        for t in self._tasks:
            if t.get_name() == name:
                return t
        return None

    def _spawn_loops(self) -> int:
        """补齐缺失的训练循环任务（幂等），返回新建条数。

        start() / set_enabled(True) / 保活监督三条路径共用同一入口，消除
        "哪条路径漏建了哪条管线"这类只在热切换时才暴露的漏配。
        """
        factories = self._loop_factories()
        created = 0
        for name in self._loop_names:
            t = self._find_task(name)
            if t is not None:
                if not t.done():
                    continue
                # 清掉已终结的旧任务，让 _tasks 与现实保持一致
                self._tasks.remove(t)
            self._tasks.append(asyncio.create_task(factories[name](), name=name))
            created += 1
        return created

    def _spawn_supervisor(self) -> None:
        """启动保活监督任务（幂等）。"""
        t = self._find_task("evolve_supervisor")
        if t is not None:
            if not t.done():
                return
            self._tasks.remove(t)
        self._tasks.append(asyncio.create_task(self._supervise_loop(),
                                               name="evolve_supervisor"))

    async def _supervise_loop(self) -> None:
        """保活监督：周期巡检三条训练循环，发现已退出的就地重建。"""
        log.info("[evolve] 保活监督已启动（巡检间隔 %.0fs，自愈最小间隔 %.0fs）",
                 self._supervise_interval, self._restart_min_gap)
        while True:
            try:
                await asyncio.sleep(self._supervise_interval)
            except asyncio.CancelledError:
                raise
            self._supervise_status["last_check"] = time.time()
            self._supervise_status["checks"] = int(self._supervise_status["checks"]) + 1
            self._supervise_status["alive"] = True
            if not self._running:
                break
            if not self._enabled:
                continue  # 用户主动禁用：不重建
            try:
                for name in self._loop_names:
                    t = self._find_task(name)
                    if t is not None and not t.done():
                        continue
                    now = time.time()
                    if now - self._last_restart.get(name, 0.0) < self._restart_min_gap:
                        # 自愈最小间隔未到：留给下一轮，防"一重建就崩"变成重启风暴
                        continue
                    await self._restart_loop(name, now)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("[evolve] 保活巡检异常（下一轮继续）")
        log.info("[evolve] 保活监督已停止")

    async def _restart_loop(self, name: str, now: float) -> None:
        """重建一条已退出的训练循环，记录自愈次数并通知（通知已去重）。"""
        dead = self._find_task(name)
        detail = "任务已丢失"
        if dead is not None:
            self._tasks.remove(dead)
            if dead.cancelled():
                detail = "任务被取消"
            elif dead.done():
                try:
                    exc = dead.exception()
                except asyncio.CancelledError:
                    exc = None
                detail = f"异常退出：{exc!r}" if exc is not None else "任务正常结束"
        try:
            self._tasks.append(
                asyncio.create_task(self._loop_factories()[name](), name=name))
        except Exception as e:  # noqa: BLE001
            log.exception("[evolve] 保活：重建 %s 失败: %s", name, e)
            return
        self._restarts[name] = int(self._restarts.get(name, 0)) + 1
        self._last_restart[name] = now
        self._supervise_status["restarts"] = sum(self._restarts.values())
        self._supervise_status["last_restart_at"] = now
        self._supervise_status["last_restart_name"] = name
        status = self._status_for(name)
        if status is not None:
            status["active"] = False
            status["restarts"] = self._restarts[name]
        log.error("[evolve] 保活：训练循环 %s 已自动重建（第 %d 次，%s）",
                  name, self._restarts[name], detail)
        try:
            from core.notify import notify
            await notify(self.db, f"进化引擎自愈：{name}",
                         f"训练循环曾意外退出（{detail}），已自动重建，"
                         f"累计自愈 {self._restarts[name]} 次。")
        except Exception:  # noqa: BLE001
            pass  # 通知不可用不阻断保活主流程

    async def _train_factor_miner_once(self, force: bool = False) -> Optional[bool]:
        """执行一次因子挖掘训练（轮换训练标的：BTC/ETH/SOL）。"""
        # 追踪实际使用的数据源（真实交易所 or 演示兜底）
        data_source_used = "exchange"
        # 轮换选择下一个训练标的
        symbol = self._next_symbol()
        self._factor_miner_status["symbol"] = symbol

        # 1. 获取最新数据（交易所不可达时按配置决定：跳过本轮 或 用演示数据兜底）
        df = await self._fetch_latest_data(symbol=symbol)
        if df is None or len(df) < self._rolling_window * 0.1:
            if df is None:
                # P4-C1：递增连续 demo 计数（供 _run_loop 降频）
                self._factor_miner_status["_demo_streak"] = int(self._factor_miner_status.get("_demo_streak", 0)) + 1
                # P1：evolve_train_on_demo=False（默认）时故障期直接跳过——
                # demo 模型永远不会被部署，故障期每 60s 全量重训纯烧 CPU，
                # 合成数据的 OOS 还必为负、徒增回退噪声。跳过同样返回 False
                # （与增量门同语义），_run_loop 的降频逻辑照常生效。
                if not getattr(settings, "evolve_train_on_demo", False):
                    self._factor_miner_status["last_error"] = f"交易所数据不可达，已暂停训练（{symbol}）"
                    log.warning("[evolve] 因子挖掘（%s）：交易所数据不可达，跳过本轮（演示训练已禁用）", symbol)
                    return False
                data_source_used = "demo"
                self._factor_miner_status["last_error"] = f"交易所数据不可达，已用合成数据兜底（{symbol}）"
                log.warning("[evolve] 因子挖掘（%s）：交易所数据不可达，使用合成数据兜底", symbol)
                df = self._generate_symbol_demo(symbol, n=self._rolling_window, timeframe=self._effective_timeframe())
            else:
                self._factor_miner_status["last_error"] = f"数据不足（{len(df)} < {int(self._rolling_window * 0.1)}），5分钟后重试"
                log.info("[evolve] 因子挖掘（%s）：数据不足（%d < %d），5分钟后重试", symbol, len(df), int(self._rolling_window * 0.1))
                raise InsufficientDataError(f"数据不足: {len(df)} < {int(self._rolling_window * 0.1)}")
        else:
            # 真实交易所数据：清零连续 demo 计数（恢复正常训练节奏）
            self._factor_miner_status["_demo_streak"] = 0

        # 2. 截取滚动窗口
        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]

        # P2-11 增量门：数据无新增 K 线（末根时间戳未变化）则跳过本轮，
        # 避免 60s 间隔 + 5 分钟数据缓存下的空转重训。demo 轮直接放行且不写 prev。
        if not force and not self._data_incremented("factor_miner", df, symbol=symbol,
                                                    is_demo=(data_source_used == "demo")):
            return False

        # 3. 加载上次最佳模型
        agent = self.zoo.load_agent("factor_miner")
        best_agent = agent  # 训练前保存当前最佳

        # 4. 计算因子矩阵（P4-E1：包 to_thread——对最多 rolling_window 根K线
        # 计算全部内置因子，纯 pandas 向量化数百毫秒~秒级，留在主协程会阻塞事件循环）
        mat = await asyncio.to_thread(compute_factor_matrix, df)

        # 5. 训练（P0：续训——把已加载的 best_agent 传给训练器作为起点。
        # 此前 load_agent 只用于回退对照，训练本体每次随机初始化重训，
        # 8 轮 PPO 从零起步学不到东西 → 每轮都被锚点回退，形成死循环。
        # 现在每轮在旧模型上小步增量，成本大幅下降、模型真正进化。
        # 续训只换起点：OOS 安检、回退保护、demo 轮短路等防线全部照常生效。)
        def _train():
            return train_factor_miner(
                df, mat=mat,
                cfg={
                    "episodes": getattr(settings, "evolve_factor_miner_episodes", 8),
                    "n_episodes": 8,
                    "ppo_epochs": 4,
                    "mini_batch_size": 64,
                    "hidden": [64, 64],
                    "seed": None,  # 随机种子，每次不同
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "h": 1,
                    "ic_window": 120,
                    "max_steps": 6,
                    "corr_threshold": 0.85,
                    # E-EXP：动作掩码（强制每步选新因子，组合恒为 max_steps 个）。
                    # .optim/exp_training/sweep_factor.py + verify_factor_win.py
                    # 实测：OOS 秩IC 0.0088→0.0356、续训 OOS 门 0/3→3/3（3 seed）。
                    "action_masking": True,
                    "base_agent": best_agent,   # 续训起点（None=首轮从零）
                    "time_budget": float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0),
                },
                # 不传 on_progress 避免回调跨线程
            )

        result = await self._run_train(_train)
        new_fitness = result["history"][-1]["best_fitness"]
        self._factor_miner_status["episode"] += 1
        self._factor_miner_status["fitness"] = round(new_fitness, 5)

        # 6.0 P4-E1：demo 合成数据轮短路——不回退比较、不写 best/版本、不触发通知。
        # 此前 demo 轮走 save_agent(is_best=False) 占版本名额（高失败率下版本历史
        # 被合成模型污染、还可能裁剪掉 best 锚点快照），且 _log_round("ok") 会把
        # _reject_streak 清零掩盖真实锁死状态；通知文本还引用未绑定的 old_fitness
        # （锚点不可比路径 NameError 被吞，通知永远发不出）。
        if data_source_used == "demo":
            log.warning("[evolve] 因子挖掘（%s）demo 数据轮：跳过回退/落库，存档留证", symbol)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], "factor_miner",
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": "demo", "demo_blocked": True})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 因子挖掘 demo 存档失败: %s", e)
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked")
            # demo 轮不发布组合因子（防污染下游），状态里标记
            self._factor_miner_status["oos_rejected"] = "演示数据训练，禁止级联发布"
            return "demo"

        # 6. 最佳模型回退保护（P0-4：旧 fitness 非正时“×阈值”方向反转，统一走 _should_rollback）
        # 锚点口径不可比时（如 demo 合成数据留下的 3.9 分）跳过：真实数据永远够不着
        # 合成数据的量级，继续比较等于让这条管线永久只回退不前进。
        # Task 4：同符号/周期/配置才可比；旧锚点缺身份字段按 unknown 不压新模型。
        anchor = self.zoo.best_info("factor_miner")
        _factor_identity = _training_identity(
            "factor_miner", symbol, self._effective_timeframe(),
            state_window=1,
            factor_signature=str(result.get("selected_factors", [])),
            config={
                "episodes": int(getattr(settings, "evolve_factor_miner_episodes", 8)),
                "ic_window": 120, "corr_threshold": 0.85,
                "train_ratio": 0.6, "val_ratio": 0.2,
                # E-EXP：纳入身份——旧锚点（无掩码训练）自动判不可比，
                # 重建基线而非与不同训练口径比较回退
                "action_masking": True,
            })
        if best_agent is not None and anchor.get("comparable"):
            _factor_mismatch = _identity_mismatch(anchor, _factor_identity)
            if _factor_mismatch:
                self._mark_deployment_status("factor_miner", anchor_mismatch=_factor_mismatch)
                log.info("[evolve] 因子挖掘锚点身份不可比（%s），跳过回退比较", _factor_mismatch)
            else:
                old_fitness = anchor.get("fitness") or 0.0
                if _should_rollback(new_fitness, old_fitness, self._rollback_threshold):
                    log.warning("[evolve] 因子挖掘新模型退化 (new=%.4f < old=%.4f*%.2f)，回退到最佳版本",
                                new_fitness, old_fitness, self._rollback_threshold)
                    try:
                        from core.notify import notify
                        await notify(self.db, "进化引擎：模型回退",
                                     f"因子挖掘新模型退化（new={new_fitness:.4f} < "
                                     f"old={old_fitness:.4f}×{self._rollback_threshold}），已回退最佳版本")
                    except Exception:  # noqa: BLE001
                        pass
                    # 统一部署入口回退到当前锚点版本（候选 fitness 与部署分离）
                    _target_version = anchor.get("version") or self.zoo.best_version("factor_miner")
                    await self.deploy_model(
                        "factor_miner", _target_version,
                        outcome="rollback", symbol=symbol, timeframe=self._effective_timeframe(),
                        reason=f"新模型退化({new_fitness:.4f})")
                    self._mark_deployment_status(
                        "factor_miner", outcome="rollback",
                        candidate_fitness=new_fitness)
                    # P2-13：回退轮次也要落库（此前回退路径漏记，曲线因此长期空白）
                    await self._log_round(
                        "factor_miner", symbol, self._effective_timeframe(),
                        data_source_used, new_fitness, status="rollback")
                    # P4-E1：回退轮补发 evolve_train 事件——此前三条 rollback 分支
                    # 不发事件，前端若依赖事件流刷新进度，回退轮完全看不到更新
                    await self.bus.publish(Event(EventType.SYSTEM, {
                        "kind": "evolve_train",
                        "model": "factor_miner",
                        "fitness": float(new_fitness),
                        "status": "rollback",
                    }, source="evolve_engine"))
                    return
        elif best_agent is not None:
            log.warning("[evolve] 因子挖掘锚点不可比（%s），跳过回退保护，本轮模型重建基线",
                        anchor.get("reason") or "未知")

        # 7.0 OOS 安检门前置（Task 4）：未通过安检的因子不得成为 factor best
        # 基线，也不得发布级联因子——此前安检在 save_agent(is_best=True) 之后，
        # 被拦模型照样覆盖 best（研究最佳与部署基线混为一谈）。
        gate_ok, gate_reason = _oos_gate_passed(result.get("report"))
        if data_source_used == "demo":
            self._factor_miner_status["oos_rejected"] = "演示数据训练，禁止级联发布"
            log.warning("[evolve] 因子挖掘使用演示数据训练，跳过最佳保存与级联发布")
            try:
                await asyncio.to_thread(
                    self.zoo.save_agent_archive, result["agent"], "factor_miner",
                    meta={"fitness": new_fitness, "timestamp": time.time(),
                          "data_source": "demo", "demo_blocked": True})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 因子挖掘 demo 存档失败: %s", e)
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked")
            return "demo"
        if not gate_ok:
            self._factor_miner_status["oos_rejected"] = gate_reason
            log.warning("[evolve] 组合因子未通过 OOS 安检（%s），仅存档不设为 best", gate_reason)
            try:
                await asyncio.to_thread(
                    self.zoo.save_agent_archive, result["agent"], "factor_miner",
                    meta={"fitness": new_fitness, "timestamp": time.time(),
                          "data_source": data_source_used,
                          "oos_rejected": gate_reason})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 被拦因子模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：因子 OOS 安检拦截",
                             f"因子挖掘新模型未通过样本外安检（{gate_reason}），"
                             f"已保留原因子最佳基线")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round("factor_miner", symbol, self._effective_timeframe(),
                                  data_source_used, new_fitness, status="oos_rejected",
                                  selected_factors=result.get("selected_factors", []))
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "factor_miner",
                "fitness": float(new_fitness),
                "oos_rejected": gate_reason,
            }, source="evolve_engine"))
            return "oos_rejected"

        # 7. 保存最佳模型（OOS 安检通过的 exchange 模型才写 best）
        is_best = True
        await asyncio.to_thread(
            self.zoo.save_agent, result["agent"], "factor_miner",
            meta={"fitness": new_fitness, "timestamp": time.time(),
                  "data_source": data_source_used, **_factor_identity},
            is_best=is_best)
        self._mark_deployment_status("factor_miner", outcome="accepted",
                                     candidate_fitness=new_fitness)
        try:
            from core.notify import notify
            # P4-E1：old_fitness 只在上方 comparable 分支内绑定；锚点不可比路径
            # 直接引用会 NameError（被外层 except 吞掉，通知静默丢失）。
            _old_fit = anchor.get("fitness") or 0.0
            await notify(self.db, "进化引擎：因子挖掘模型更新",
                         f"fitness {_old_fit:.4f} → {new_fitness:.4f}"
                         f"（{data_source_used} 数据）")
        except Exception:  # noqa: BLE001
            pass
        log.info("[evolve] 因子挖掘训练完成，fitness=%.4f，已保存", new_fitness)

        # 7.1 保存组合因子供策略 DRL 级联使用（已过 7.0 安检）
        import numpy as np
        composite_series = result.get("composite")
        self._factor_miner_status["oos_rejected"] = ""
        # 组合因子权重（combo_spec）落盘不依赖 composite：手动重新部署只读权重
        # 文件，composite 为 None（跳过级联 npy 保存）时权重仍需持久化（M-5）。
        weights = result.get("weights", {})
        if weights:
            import json as _json
            cascade_weights_path = self.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
            with open(str(cascade_weights_path), "w", encoding="utf-8") as f:
                _json.dump(weights, f, ensure_ascii=False)
            log.info("[evolve] 组合因子权重已保存到 %s", cascade_weights_path)
        # 清理旧格式单文件（兼容升级前的遗留），防止被误加载
        legacy = self.zoo.models_dir / "_cascade_factor_values.npy"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:  # noqa: BLE001
                pass
        if composite_series is not None:
            import pandas as pd
            # 对齐到原始 df 的索引（factor_miner 内部可能截断或排序）
            factor_values = pd.Series(composite_series.to_numpy(), index=composite_series.index)
            aligned = factor_values.reindex(df.index).to_numpy(float)
            # 按标的分文件保存（P4-A1）：level-2 防御。尽管三管线现已同标训练，
            # 分文件 + 加载时按标的匹配仍防止任何异常路径下的跨标错配。
            cascade_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.npy"
            np.save(str(cascade_path), aligned)
            # P4-E1：与 npy 同存窗口尾时间戳 companion JSON——加载端只校验长度
            # 无法发现"窗口右移后的静默错位"（长度同为 rolling_window），
            # 存尾时间戳让加载端能按窗口身份校验，错位即拒绝注入。
            try:
                import json as _json
                tail_ts = None
                try:
                    tail_ts = int(df.index[-1].timestamp())
                except Exception:  # noqa: BLE001
                    tail_ts = int(df.index[-1]) if df.index[-1] is not None else None
                if tail_ts is not None:
                    meta_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.json"
                    meta_path.write_text(_json.dumps({"tail_ts": tail_ts, "n": int(len(aligned))}),
                                         encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 级联因子时间戳元数据写入失败（不影响 npy 保存）: %s", e)
            log.info("[evolve] 组合因子已保存到 %s，供策略 DRL 级联注入", cascade_path)

        # 8. 发布系统事件
        # P2-13：训练轮次落库（fitness/选中因子/数据源）
        await self._log_round(
            "factor_miner", symbol, self._effective_timeframe(),
            data_source_used, new_fitness,
            status="ok", selected_factors=result.get("selected_factors", []))
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "evolve_train",
            "model": "factor_miner",
            "fitness": new_fitness,
            "selected_factors": result.get("selected_factors", []),
        }, source="evolve_engine"))

        # 9. 自动部署：OOS 安检已过（gate_ok），把组合权重部署为按标的的组合因子策略。
        # 部署失败只记 combo_deploy_error，不阻断训练流程（训练主体已完成并落库）。
        try:
            _weights = result.get("weights") or {}
            if not _weights:
                # 权重缺失：显式记错并跳过部署，不更新已有部署状态
                # （注册表里已部署的策略仍有效，组合策略相关键保持上一轮值不动）
                self._factor_miner_status["combo_deploy_error"] = "权重缺失，跳过自动部署"
                log.warning("[evolve] 组合因子策略自动部署跳过(%s): 权重缺失", symbol)
            else:
                _meta = {
                    "fitness": float(new_fitness),
                    "selected_factors": result.get("selected_factors", []),
                    "round_no": self._factor_miner_status.get("episode", 0),
                }
                _dep = await self._deploy_factor_strategy(
                    symbol, _weights, result.get("report"), _meta)
                self._factor_miner_status["combo_strategy"] = _dep["name"]
                self._factor_miner_status["combo_version"] = _dep["version"]
                self._factor_miner_status["combo_deployed_at"] = time.time()
                self._factor_miner_status["combo_deploy_error"] = ""
                # 后台回测摘要（fire-and-forget，不阻塞训练循环；持有任务引用防 GC）
                _task = asyncio.create_task(
                    self._refresh_factor_strategy_backtest(_dep["name"], symbol, df, _dep["version"]))
                self._backtest_tasks[_dep["name"]] = _task
                _task.add_done_callback(
                    lambda t, n=_dep["name"]: self._backtest_tasks.pop(n, None))
        except Exception as e:  # noqa: BLE001
            self._factor_miner_status["combo_deploy_error"] = f"自动部署失败: {e}"
            log.warning("[evolve] 组合因子策略自动部署失败(%s): %s", symbol, e)

    # ---- 策略 DRL 持续训练循环 ----

    async def _train_strategy_drl_once(self, force: bool = False) -> Optional[bool]:
        """执行一次策略 DRL 训练，并注册为可用策略（轮换训练标的：BTC/ETH/SOL）。"""
        # 1. 获取最新数据（交易所不可达时按配置决定：跳过本轮 或 用演示数据兜底）
        data_source_used = "exchange"
        symbol = self._next_symbol()
        self._strategy_drl_status["symbol"] = symbol
        df = await self._fetch_latest_data(symbol=symbol)
        if df is None or len(df) < self._rolling_window * 0.1:
            if df is None:
                # P4-C1：递增连续 demo 计数（供 _run_loop 降频）
                self._strategy_drl_status["_demo_streak"] = int(self._strategy_drl_status.get("_demo_streak", 0)) + 1
                # P1：evolve_train_on_demo=False（默认）时故障期直接跳过——
                # demo 模型永远不会被部署，故障期全量重训纯烧 CPU（与因子挖掘同口径）。
                if not getattr(settings, "evolve_train_on_demo", False):
                    self._strategy_drl_status["last_error"] = f"交易所数据不可达，已暂停训练（{symbol}）"
                    log.warning("[evolve] 策略DRL（%s）：交易所数据不可达，跳过本轮（演示训练已禁用）", symbol)
                    return False
                data_source_used = "demo"
                self._strategy_drl_status["last_error"] = f"交易所数据不可达，已用合成数据兜底（{symbol}）"
                log.warning("[evolve] 策略DRL（%s）：交易所数据不可达，使用合成数据兜底", symbol)
                df = self._generate_symbol_demo(symbol, n=self._rolling_window, timeframe=self._effective_timeframe())
            else:
                self._strategy_drl_status["last_error"] = f"数据不足（{len(df)} < {int(self._rolling_window * 0.1)}），5分钟后重试"
                log.info("[evolve] 策略DRL（%s）：数据不足（%d < %d），5分钟后重试", symbol, len(df), int(self._rolling_window * 0.1))
                raise InsufficientDataError(f"数据不足: {len(df)} < {int(self._rolling_window * 0.1)}")
        else:
            # 真实交易所数据：清零连续 demo 计数（恢复正常训练节奏）
            self._strategy_drl_status["_demo_streak"] = 0

        # 2. 截取滚动窗口
        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]

        # P2-11 增量门：数据无新增 K 线则跳过本轮（仅连续训练生效，手动触发总执行）。
        # demo 轮直接放行且不写 prev（合成数据尾戳 ≈now 会污染真实数据增量判定）。
        if not force and not self._data_incremented("strategy_drl", df, symbol=symbol,
                                                    is_demo=(data_source_used == "demo")):
            return False

        # P0-族群：冠军/挑战者 K=2 交替。_pop_round 每完成一轮真实训练递增一次
        # （增量门跳过/故障跳过不计），偶数为冠军轮（走原路径，部署+注册），
        # 奇数为挑战者轮（独立谱系 strategy_drl_alt，须显著优于冠军才晋升）。
        # 每轮仍只训一个模型 → 墙钟成本不变；多样性来自两条谱系独立进化。
        _pop_on = bool(getattr(settings, "evolve_strategy_drl_population", True))
        # force（手动触发）恒走冠军谱系：手动训练语义是"更新部署模型"，
        # 挑战者轮只由后台连续训练节奏驱动。
        _round_no = int(self._strategy_drl_status.get("_pop_round", 0))
        is_challenger = bool(_pop_on and not force and (_round_no % 2 == 1))
        self._strategy_drl_status["_pop_round"] = _round_no + 1
        self._strategy_drl_status["lineage"] = "challenger" if is_challenger else "main"
        lineage_name = "strategy_drl_alt" if is_challenger else "strategy_drl"

        # 3. 加载本轮谱系的上次最佳模型（续训）
        agent = self.zoo.load_agent(lineage_name)
        best_agent = agent

        # 级联因子值：按当前训练标的加载，保证注入的因子与训练标的一致（P4-A1）
        cascade_factor_values = self._load_cascade_factor_values(df, symbol=symbol)
        # 级联因子复算配方（选中因子权重）：与 npy 同标保存，供部署端实盘复算
        cascade_recipe = (self._load_cascade_factor_recipe(symbol)
                          if cascade_factor_values is not None else None)

        # 训练配置提升到函数作用域：跨标传导 OOS（下方）需要复用 state_window，
        # 曾定义在 _train 闭包内导致成功路径必然 NameError
        train_cfg = _strategy_drl_train_cfg(
            getattr(settings, "evolve_strategy_drl_episodes", 8))

        # 4. 训练（P0：续训——best_agent 传入作为起点。此前 load_agent 只用于
        # 回退对照，训练每次随机初始化重训，8 轮 PPO 从零起步学不到东西 →
        # 每轮都被锚点回退，形成死循环。现在每轮在旧模型上小步增量，
        # 成本大幅下降、模型真正进化。续训只换起点：OOS 硬门、回退保护、
        # 训练身份隔离、demo 短路全部照常生效。）
        def _train():
            # 级联因子注入：加载 factor_miner 产出的组合因子值（若有），
            # 作为策略 DRL 的额外状态特征（与训练/部署口径一致）
            if cascade_factor_values is not None:
                train_cfg["factor_values"] = cascade_factor_values
                if cascade_recipe is not None:
                    train_cfg["factor_composite"] = cascade_recipe
                log.info("[evolve] 已注入级联组合因子到策略 DRL 训练（含复算配方 %s）",
                         "有" if cascade_recipe is not None else "无")
            train_cfg["base_agent"] = best_agent   # 续训起点（None=首轮从零）
            train_cfg["time_budget"] = float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0)
            return train_drl(df, train_cfg, on_progress=None)

        result = await self._run_train(_train)
        # 取最后一条历史记录的收益
        history = result.get("history", [])
        if not history:
            _err = "策略DRL训练无历史记录（训练未产生结果）"
            log.warning("[evolve] %s", _err)
            raise RuntimeError(_err)
        # fitness 口径与 /api/drl/train 一致：取 train_drl 汇总的 best_ret。
        # 旧写法从 history 末行现取，缺 best_ret 时落到单轮 total_ret，
        # 与跨版本回退比较不同量级。
        # P4-E1：不能用 `or` 链——best_ret=0.0 是 falsy，会被 history 末行
        # 的数值静默替换，锚点 fitness 记错并传导到回退比较。显式判 None。
        _br = result.get("best_ret")
        new_fitness = float(_br if _br is not None
                            else history[-1].get("best_ret") or 0.0)
        anchor = self.zoo.best_info("strategy_drl")
        prev_fitness = anchor.get("fitness") or 0.0
        self._strategy_drl_status["episode"] += 1
        self._strategy_drl_status["fitness"] = round(new_fitness, 5)

        # 训练身份：与锚点同口径才允许回退比较（不同符号/周期/状态维度/配置不可比）
        _identity = _training_identity(
            "strategy_drl", symbol, self._effective_timeframe(),
            state_window=int(train_cfg.get("state_window", 1)),
            factor_signature=str(result.get("factor_expression", "") or ""),
            config={
                "episodes": int(getattr(settings, "evolve_strategy_drl_episodes", 8)),
                "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
                "oos_min_bars": int(getattr(settings, "evolve_oos_min_bars", 250)),
            })

        # 5.0 P4-E1：demo 合成数据轮短路——不回退比较、不跑 OOS 部署门。
        # 交易所故障时合成数据的 OOS 几乎必为负，会触发无意义的
        # save_agent_best（覆盖 flat）+ 重新导出 Pine + _log_round("rollback")，
        # 且 _reject_streak 递增可能误触发 stalled 横幅。demo 模型只存档留证，
        # 记为 demo_blocked（与 factor_miner 口径一致），best/flat 保持真实模型。
        if data_source_used == "demo":
            log.warning("[evolve] 策略DRL（%s）demo 数据轮：跳过回退/OOS 门，存档留证", symbol)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": "demo", "demo_blocked": True,
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 策略DRL demo 存档失败: %s", e)
            await self._log_round("strategy_drl", symbol, self._effective_timeframe(),
                                  "demo", new_fitness, status="demo_blocked")
            return "demo"

        # 5. 最佳模型回退保护（P0-4 负 fitness 方向反转修复；
        # P1-6 回退判据统一为 OOS 收益——与部署门同口径，不再用训练段收益比较）
        # 锚点口径不可比时跳过比较：demo 合成数据留下的高分基线（实测 3.9167）真实
        # 行情一辈子也够不到，只会把每一轮都判成"退化"，进化循环事实上停摆。
        # P0-族群：冠军轮走"退化即回退"；挑战者轮走"显著优于冠军才晋升"——
        # 挑战者永不动部署端（best/flat/注册全保留），只有晋升那一轮才接管冠军槽位。
        # 挑战者轮的晋升只依赖冠军锚点（best_info("strategy_drl")），与挑战者自身
        # 起点无关——首轮 alt 槽位为空（best_agent=None）也应能与冠军比较。
        promote = False  # 挑战者轮晋升标志（冠军轮恒 False，不用）
        if is_challenger:
            if anchor.get("comparable") and not _identity_mismatch(anchor, _identity):
                oos_report = result.get("oos_report") or {}
                new_oos = float(oos_report.get("oos_ret") or 0.0) if oos_report.get("enabled") else None
                prev_oos = anchor.get("oos_ret")
                if new_oos is not None and prev_oos is not None:
                    # 挑战者须显著优于冠军才晋升（> 冠军×(1/threshold)，防噪音替换）
                    promote = new_oos > float(prev_oos) * (1.0 / max(self._rollback_threshold, 1e-6))
                    cmp_desc = f"挑战者OOS {new_oos:.4f} vs 冠军 {float(prev_oos):.4f}×(1/{self._rollback_threshold})"
                else:
                    # 旧模型缺 OOS 记录（升级前版本）：退化为训练段收益比较
                    promote = new_fitness > prev_fitness * (1.0 / max(self._rollback_threshold, 1e-6))
                    cmp_desc = f"挑战者fitness {new_fitness:.4f} vs 冠军 {prev_fitness:.4f}×(1/{self._rollback_threshold})"
                if promote:
                    log.info("[evolve] 策略DRL挑战者显著优于冠军（%s），晋升接管冠军槽位", cmp_desc)
                else:
                    log.info("[evolve] 策略DRL挑战者未显著优于冠军（%s），保留冠军，挑战者更新自身谱系", cmp_desc)
            else:
                log.info("[evolve] 策略DRL挑战者轮：冠军锚点不可比（%s），不晋升",
                         anchor.get("reason") or "身份不匹配")
        elif best_agent is not None and anchor.get("comparable"):
            identity_mismatch = _identity_mismatch(anchor, _identity)
            if identity_mismatch:
                self._mark_deployment_status("strategy_drl", anchor_mismatch=identity_mismatch)
                log.info("[evolve] 策略DRL 锚点身份不可比（%s），跳过回退比较", identity_mismatch)
            oos_report = result.get("oos_report") or {}
            new_oos = float(oos_report.get("oos_ret") or 0.0) if oos_report.get("enabled") else None
            prev_oos = anchor.get("oos_ret")
            if new_oos is not None and prev_oos is not None:
                # 新旧均有 OOS 收益：以 OOS 为准（与部署门一致）
                rollback_hit = _should_rollback(new_oos, float(prev_oos), self._rollback_threshold)
                cmp_desc = f"OOS {new_oos:.4f} < {float(prev_oos):.4f}×{self._rollback_threshold}"
            else:
                # 旧模型缺 OOS 记录（升级前版本）：退化为训练段收益比较
                rollback_hit = _should_rollback(new_fitness, prev_fitness, self._rollback_threshold)
                cmp_desc = f"fitness {new_fitness:.4f} < {prev_fitness:.4f}×{self._rollback_threshold}"
            if identity_mismatch:
                rollback_hit = False
            if rollback_hit:
                log.warning("[evolve] 策略DRL新模型退化 (%s)，回退到最佳版本", cmp_desc)
                try:
                    from core.notify import notify
                    await notify(self.db, "进化引擎：模型回退",
                                 f"策略DRL新模型退化（{cmp_desc}），已回退最佳版本")
                except Exception:  # noqa: BLE001
                    pass
                await self.deploy_model(
                    "strategy_drl", anchor.get("version") or self.zoo.best_version("strategy_drl"),
                    outcome="rollback", symbol=symbol, timeframe=self._effective_timeframe(),
                    reason=f"新模型退化({cmp_desc})")
                self._mark_deployment_status("strategy_drl", outcome="rollback",
                                             candidate_fitness=new_fitness)
                # P2-13：回退轮次也要落库（此前回退路径漏记，曲线因此长期空白）
                _oos_report_rb = result.get("oos_report") or {}
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness,
                    status="rollback",
                    oos_ret=_oos_report_rb.get("oos_ret", 0.0) or 0.0,
                    decay=_oos_report_rb.get("decay", 0.0) or 0.0,
                    position_ratio=_oos_report_rb.get("oos_position_ratio", 0.0) or 0.0)
                # P4-E1：回退轮补发 evolve_train 事件（与因子挖掘一致）
                await self.bus.publish(Event(EventType.SYSTEM, {
                    "kind": "evolve_train",
                    "model": "strategy_drl",
                    "fitness": float(new_fitness),
                    "status": "rollback",
                    "deployed_version": self.zoo.best_version("strategy_drl"),
                }, source="evolve_engine"))
                return
        elif best_agent is not None:
            log.warning("[evolve] 策略DRL 锚点不可比（%s），跳过回退保护，本轮模型重建基线",
                        anchor.get("reason") or "未知")

        # 5.5 OOS 硬门：未过样本外安检的模型不得接管实盘模型、不得注册策略
        gate_ok, gate_reason = _strategy_deploy_gate(result)
        self._strategy_drl_status["oos_rejected"] = "" if gate_ok else gate_reason
        if not gate_ok:
            # P4-C2：只存档不占版本名额——best/flat 文件保持原最佳模型，
            # 已注册的 rl_evolve 继续用旧权重，不会被未验证的模型替换。
            # 此前 is_best=False 的 save_agent 也会递增版本号占名额，被拦截
            # 模型多了会把真改进版挤出版本历史。
            log.warning("[evolve] 策略DRL 未通过 OOS 硬门，保留原最佳模型：%s", gate_reason)
            try:
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_rejected": gate_reason,
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 被拦截模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：OOS 硬门拦截",
                             f"策略DRL 新模型未通过样本外安检，已保留原最佳模型：{gate_reason}")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="oos_rejected",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0)
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "oos_rejected": gate_reason,
            }, source="evolve_engine"))
            return

        # 5.7 P1-8：跨标的传导 OOS（低权重附加门）——模型在另一训练标的上
        # 的样本外表现。跨标收益非正时：仅存档不设为最佳（保持旧 best 与 flat 文件），
        # 防止"单标的特有模式"被当成通用能力上线。
        # 级联因子注入时 agent 状态含因子列，跨标数据无法重建同维状态，跳过（不拦）。
        if cascade_factor_values is not None:
            self._strategy_drl_status["cross_oos"] = None
            log.info("[evolve] 级联因子注入中，跳过跨标传导 OOS")
            cross_oos = None
        else:
            cross_oos = await self._cross_symbol_oos(result["agent"], symbol,
                                                     int(train_cfg["state_window"]))
        self._strategy_drl_status["cross_oos"] = cross_oos
        if cross_oos is not None and cross_oos.get("ret", 0.0) <= 0:
            log.warning("[evolve] 策略DRL 跨标 OOS 未通过（%s: %.4f），仅存档不设为最佳",
                        cross_oos.get("symbol"), cross_oos.get("ret", 0.0))
            try:
                # P4-C2：仅存档不占版本名额（跨标被拦模型非有效改进）
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "cross_oos_rejected": True,
                                              "cross_symbol": cross_oos.get("symbol"),
                                              "cross_oos_ret": cross_oos.get("ret", 0.0),
                                              "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 跨标被拦模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：跨标安检拦截",
                             f"策略DRL 新模型跨标收益 {cross_oos.get('ret', 0.0):.4f} 非正"
                             f"（{cross_oos.get('symbol')}），已保留原最佳模型")
            except Exception:  # noqa: BLE001
                pass
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="cross_rejected",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0)
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "cross_oos_rejected": True,
            }, source="evolve_engine"))
            return

        # 6. 保存最佳模型（含训练元数据）
        # P0-2：demo 数据训练只存档不写 best（is_best=False），
        # 且跳过兼容格式 flat 写入（rl_adaptive 部署端读取的正是 flat 文件，
        # 若被合成数据模型覆盖，实盘会被未经验证的权重接管）。
        # P0-族群：挑战者轮未晋升 → 只更新挑战者谱系槽位（strategy_drl_alt，
        # 独立 best/flat，不触碰部署端），记 challenger_kept 后收尾；
        # 晋升 → 与冠军轮同样落盘部署（下方 else 分支）。
        if is_challenger and not promote:
            # 挑战者谱系自保护：新模型若相对挑战者自身锚点退化（< 自身×threshold），
            # 只存档不覆盖 alt 槽位——谱系质量优先，退化模型不进族群。
            alt_anchor = self.zoo.best_info("strategy_drl_alt")
            alt_prev = alt_anchor.get("fitness")
            if alt_prev is not None and _should_rollback(new_fitness, float(alt_prev), self._rollback_threshold):
                log.info("[evolve] 策略DRL挑战者相对自身退化（%.4f < %.4f×%.2f），只存档不更新谱系",
                         new_fitness, float(alt_prev), self._rollback_threshold)
                try:
                    await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], "strategy_drl_alt",
                                            meta={"fitness": new_fitness, "timestamp": time.time(),
                                                  "data_source": data_source_used,
                                                  "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                                  "lineage": "challenger", "challenger_degraded": True})
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 挑战者退化存档失败: %s", e)
                await self._log_round(
                    "strategy_drl", symbol, self._effective_timeframe(),
                    data_source_used, new_fitness,
                    status="challenger_degraded",
                    oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                    decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                    position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0)
                await self.bus.publish(Event(EventType.SYSTEM, {
                    "kind": "evolve_train",
                    "model": "strategy_drl",
                    "fitness": new_fitness,
                    "lineage": "challenger",
                    "status": "challenger_degraded",
                }, source="evolve_engine"))
                return
            log.info("[evolve] 策略DRL挑战者更新自身谱系（不晋升，冠军部署不动）")
            try:
                await asyncio.to_thread(self.zoo.save_agent, result["agent"], "strategy_drl_alt",
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "base_oos_ret": anchor.get("oos_ret"),
                                              "base_fitness": anchor.get("fitness"),
                                              "lineage": "challenger", **_identity})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 挑战者谱系保存失败: %s", e)
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                data_source_used, new_fitness,
                status="challenger_kept",
                oos_ret=(result.get("oos_report") or {}).get("oos_ret", 0.0) or 0.0,
                decay=(result.get("oos_report") or {}).get("decay", 0.0) or 0.0,
                position_ratio=(result.get("oos_report") or {}).get("oos_position_ratio", 0.0) or 0.0)
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "strategy_drl",
                "fitness": new_fitness,
                "lineage": "challenger",
                "status": "challenger_kept",
            }, source="evolve_engine"))
            return

        if data_source_used == "demo":
            try:
                # P4-C2：demo 模型仅存档不占版本名额（合成数据模型非真实改进）
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], lineage_name,
                                        meta={"fitness": new_fitness, "timestamp": time.time(),
                                              "data_source": data_source_used,
                                              "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                              "demo_blocked": True, "lineage": ("challenger" if is_challenger else "main")})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] demo 模型存档失败: %s", e)
            log.warning("[evolve] 策略DRL使用演示数据训练，仅存档不部署（防实盘被未验证模型接管）")
        else:
            # P0-族群加强：挑战者晋升时角色互换——旧冠军不丢弃，转入挑战者
            # 槽位继续进化（PBT top-k 保留思想）。A/B 实验（scripts/ab_population3.py）
            # 显示互换比"晋升即丢弃旧冠军" final 提升约 +92%（0.0161→0.0310）：
            # 两条谱系都保持"曾当过冠军"的强度，多样性不减、信息不丢。
            # 必须在本轮覆盖冠军槽位**之前**先读旧冠军权重，否则读到新模型。
            if is_challenger:
                try:
                    old_champ = self.zoo.load_agent("strategy_drl")
                    if old_champ is not None:
                        await asyncio.to_thread(self.zoo.save_agent, old_champ, "strategy_drl_alt",
                                                meta={"fitness": anchor.get("fitness"),
                                                      "timestamp": time.time(),
                                                      "data_source": anchor.get("data_source") or data_source_used,
                                                      "oos_ret": anchor.get("oos_ret"),
                                                      "lineage": "challenger",
                                                      "swapped_from_champion": True,
                                                      **_identity})
                        log.info("[evolve] 策略DRL挑战者晋升：旧冠军已转入挑战者槽位（角色互换）")
                except Exception as e:  # noqa: BLE001
                    log.warning("[evolve] 旧冠军转入挑战者槽位失败: %s", e)
            await asyncio.to_thread(self.zoo.save_agent, result["agent"], "strategy_drl",
                                    meta={"fitness": new_fitness, "timestamp": time.time(),
                                          "data_source": data_source_used,
                                          "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                                          # P2 续训护栏审计：记录起点（旧模型）的 OOS/收益，
                                          # 让"续训后是否真进步"可追溯。接受路径已保证
                                          # new_oos ≥ prev_oos×threshold（rollback 守卫），
                                          # 这里把对比写进 meta 供前端/审计展示。
                                          "base_oos_ret": anchor.get("oos_ret"),
                                          "base_fitness": anchor.get("fitness"),
                                          "base_version": anchor.get("version"),
                                          "lineage": ("challenger_promoted" if is_challenger else "main"),
                                          **_identity})
            self._mark_deployment_status("strategy_drl", outcome="accepted",
                                         candidate_fitness=new_fitness)
        try:
            from core.notify import notify
            _oos = (result.get("oos_report") or {}).get("oos_ret")
            await notify(self.db, "进化引擎：策略DRL模型更新",
                         f"fitness {prev_fitness:.4f} → {new_fitness:.4f}"
                         f"，OOS 收益 {_oos if _oos is not None else '无'}（{data_source_used} 数据）")
        except Exception:  # noqa: BLE001
            pass
        if data_source_used == "demo":
            log.warning("[evolve] 策略DRL使用演示数据训练完成（仅供测试，勿用于实盘），收益=%.4f", new_fitness)
            # P0-2：demo 模型不写兼容格式、不注册，直接结束本轮
            log.warning("[evolve] 策略DRL使用演示数据训练，跳过元数据写入与策略注册（防止误用实盘）")
            # P2-13：demo 轮次落库（标记 demo_blocked）
            _oos_report_d = result.get("oos_report") or {}
            await self._log_round(
                "strategy_drl", symbol, self._effective_timeframe(),
                "demo", new_fitness,
                status="demo_blocked",
                oos_ret=_oos_report_d.get("oos_ret", 0.0) or 0.0,
                decay=_oos_report_d.get("decay", 0.0) or 0.0,
                position_ratio=_oos_report_d.get("oos_position_ratio", 0.0) or 0.0)
            return
        else:
            # 6.1 在兼容格式 JSON 中追加训练元数据（与 web/api/drl.py 同格式）
            try:
                flat_path = self.zoo._flat_path("strategy_drl")
                if flat_path.exists():
                    import json

                    def _write_flat_meta() -> None:
                        import json
                        with open(flat_path, "r", encoding="utf-8") as f:
                            _md = json.load(f)
                        _md["train_meta"] = {
                            "data_source": data_source_used,
                            "symbol": symbol,
                            "timeframe": self._effective_timeframe(),
                            "limit": self._rolling_window,
                            "episodes": getattr(settings, "evolve_strategy_drl_episodes", 8),
                            "best_ret": result.get("best_ret"),
                            "best_val_ret": result.get("best_val_ret"),
                            "oos_report": result.get("oos_report"),
                        }
                        _md["factor_expression"] = result.get("factor_expression", "")
                        _md["factor_mu"] = float(result.get("factor_mu", 0.0))
                        _md["factor_sd"] = float(result.get("factor_sd", 1.0))
                        if result.get("factor_composite"):
                            # 级联因子复算配方：实盘 rl_adaptive 据此在滚动缓冲
                            # 上复算组合因子（曾只读写死数组，实盘因子列恒 None）
                            _md["factor_composite"] = result["factor_composite"]
                        _md["min_trade_zone"] = 0.05
                        _md["pine_code"] = result.get("pine_code", "")
                        _md["pine_note"] = result.get("pine_note", "")
                        # run25 R5 修复：原 open("w")+json.dump 非原子重写，
                        # 写盘窗口内读取方（rl_adaptive/回测/load_agent flat 回退）
                        # 可能读到截断 JSON；改走 ModelZoo 原子写（tmp+fsync+replace）
                        ModelZoo._atomic_write_bytes(flat_path,
                                                     ModelZoo._json_bytes(_md))

                    await asyncio.to_thread(_write_flat_meta)
            except Exception as e:
                log.warning("[evolve] 模型元数据写入失败: %s", e)

            # 7. 注册为可用策略（演示数据训练的模型不注册，防止误用实盘）
            try:
                await self._register_rl_evolve()
            except Exception as e:
                log.warning("[evolve] 策略注册失败: %s", e)

        log.info("[evolve] 策略DRL训练完成，收益=%.4f，已保存并注册", new_fitness)

        # 8. 发布系统事件
        # P2-13：训练轮次落库
        _oos_report = result.get("oos_report") or {}
        await self._log_round(
            "strategy_drl", symbol, self._effective_timeframe(),
            data_source_used, new_fitness,
            status="ok",
            oos_ret=_oos_report.get("oos_ret", 0.0) or 0.0,
            decay=_oos_report.get("decay", 0.0) or 0.0,
            position_ratio=_oos_report.get("oos_position_ratio", 0.0) or 0.0)
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "evolve_train",
            "model": "strategy_drl",
            "fitness": new_fitness,
            "lineage": ("challenger_promoted" if is_challenger else "main"),
        }, source="evolve_engine"))

    # ---- 元策略控制器持续训练 ----

    async def _train_meta_controller_once(self, force: bool = False) -> Optional[bool]:
        """训练 DRL 元策略控制器（统一管理子策略）。

        子策略取用户在「统一策略」中为 meta_controller 选定的池（未选则回退默认内置池），
        元控制器在它们之上学习"何时信任哪个策略"。产出模型注册为
        meta_controller 策略，供用户启用。
        """
        symbol = self._next_symbol()
        self._meta_controller_status["symbol"] = symbol

        # P0-3：加载上次最佳模型（元控制器此前缺回退保护，与另两管线不一致）
        agent = self.zoo.load_agent("meta_controller")
        best_agent = agent
        prev_fitness = self.zoo.best_fitness("meta_controller") or 0.0

        # 1. 获取最新数据
        df = await self._fetch_latest_data(symbol=symbol)
        if df is None or len(df) < self._rolling_window * 0.1:
            if df is None:
                # P4-C1：交易所不可达 → 递增连续 demo 计数（供 _run_loop 降频）。
                # P4-E1：返回 False（与增量门跳过同语义）——否则 _run_loop 会把
                # 「未发生的训练」记为 last_success=True 并推进标的轮换。
                self._meta_controller_status["_demo_streak"] = int(self._meta_controller_status.get("_demo_streak", 0)) + 1
                self._meta_controller_status["last_error"] = "交易所数据不可达，跳过元策略训练"
                log.warning("[evolve] 元策略：交易所数据不可达，跳过本轮")
                return False
            self._meta_controller_status["last_error"] = f"数据不足（{len(df)} < {int(self._rolling_window * 0.1)}）"
            raise InsufficientDataError(f"数据不足: {len(df)} < {int(self._rolling_window * 0.1)}")
        else:
            # 真实交易所数据：清零连续 demo 计数（恢复正常训练节奏）
            self._meta_controller_status["_demo_streak"] = 0

        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]

        # P2-11 增量门：数据无新增 K 线则跳过本轮（仅连续训练生效，手动触发总执行）
        if not force and not self._data_incremented("meta_controller", df, symbol=symbol,
                                                    is_demo=False):
            return False

        # 2. 子策略池：用户在「统一策略」中为 meta_controller 选定的子策略优先
        from strategies import get_dynamic
        sub_strategies = resolve_meta_sub_strategies(
            (get_dynamic("meta_controller") or {}).get("params"))
        self._meta_controller_status["sub_strategies"] = sub_strategies

        # 3. 训练元控制器（P0：续训——best_agent 传入作为起点，与另两管线同口径。
        # 此前 load_agent 只用于回退对照，元策略每次随机初始化重训 → 每轮被锚点
        # 回退的死循环。现在每轮在旧模型上小步增量；OOS 硬门/回退保护/身份隔离
        # 全部照常生效。子策略池变更时本轮仍作废，续训起点不受影响。）
        def _train():
            return train_meta_controller(
                df, {
                    "episodes": getattr(settings, "evolve_meta_episodes", 8),
                    "n_episodes": 4,
                    "hidden": [64, 64],
                    "seed": None,
                    "sub_strategies": sub_strategies,
                    "fee_rate": 0.001,
                    "meta_window": 20,
                    "base_agent": best_agent,   # 续训起点（None=首轮从零）
                    "time_budget": float(getattr(settings, "evolve_train_time_budget", 0.0) or 0.0),
                    # 信号对齐塑形（P4-E2）：让元控制器学会跟随高置信子策略信号，
                    # 逃出"空仓=0"陷阱——此前 289 轮全部 fitness=0.0 被回退/拦截。
                    # 塑形只进训练梯度，best_ret/OOS 仍按真实收益，验证口径不变。
                    "reward_signal_align": float(getattr(settings, "evolve_meta_signal_align", 6.0) or 0.0),
                },
                on_progress=None,
            )

        result = await self._run_train(_train)
        history = result.get("history", [])
        if not history:
            raise RuntimeError("元策略训练无历史记录")

        # 训练耗时以十秒计，期间用户可能在「统一策略」里改了子策略池。本轮模型是按旧池
        # 训练的，覆盖模型文件 + 回写 params 会静默打回用户的选择，故直接作废等下轮重训
        cur_pool = resolve_meta_sub_strategies(
            (get_dynamic("meta_controller") or {}).get("params"))
        if cur_pool != sub_strategies:
            log.info("[evolve] 训练期间子策略池已被用户改为 %s，本轮结果作废（下轮按新池重训）", cur_pool)
            # P4-E1：返回 False（本轮未训练，不推进 last_success/标的轮换）
            # 作废轮在上面的增量门已消费了"新数据额度"，这里回退该记录，
            # 下轮仍按真实数据增量判定（否则作废轮会白吃掉一批新 K 线）
            self._last_tail_ts.pop(f"meta_controller:{symbol}:{self._effective_timeframe()}", None)
            return False

        # P4-E1：不能用 `or` 链——best_ret=0.0 是 falsy，会被 history 末行数值
        # 静默替换（与 strategy_drl 同款修复）。显式判 None。
        _mbr = result.get("best_ret")
        best_ret = float(_mbr if _mbr is not None
                         else history[-1].get("best_ret") or 0.0)
        self._meta_controller_status["episode"] += 1
        self._meta_controller_status["fitness"] = round(best_ret, 5)

        # 训练身份：同级策略仅同口径可比（元控制器版本同样按符号/周期/配置分桶）
        _meta_identity = _training_identity(
            "meta_controller", symbol, self._effective_timeframe(),
            state_window=1,
            factor_signature=f"subs:{sub_strategies}",
            config={
                "episodes": int(getattr(settings, "evolve_meta_episodes", 8)),
                "sub_strategies": sub_strategies,
            })

        # 3.4 最佳模型回退保护（P0-3 补齐：元控制器此前无回退保护；
        # P0-4 负 fitness 方向反转修复，统一走 _should_rollback；
        # 锚点口径不可比时跳过，同策略DRL/因子挖掘）
        # P4-E1：判据统一为 OOS 收益（与 strategy_drl L1055-1069 同口径）——
        # 旧实现用训练段 best_ret 与锚点比较：新模型 OOS 优于锚点但训练段略低
        # 会被误回退（丢真改进）；反过来 OOS 相对锚点明显回撤却不触发回退
        # （meta 进化可在 OOS 口径上持续退步）。新旧均有 oos_ret 时以 OOS 为准，
        # 缺失时回落训练段比较（旧版本锚点无 OOS 记录）。
        anchor = self.zoo.best_info("meta_controller")
        meta_rollback_hit = False
        meta_cmp_desc = ""
        if best_agent is not None and anchor.get("comparable"):
            _meta_mismatch = _identity_mismatch(anchor, _meta_identity)
            if _meta_mismatch:
                self._mark_deployment_status("meta_controller", anchor_mismatch=_meta_mismatch)
                log.info("[evolve] 元控制器锚点身份不可比（%s），跳过回退比较", _meta_mismatch)
            _moos = result.get("oos_report") or {}
            new_moos = float(_moos.get("oos_ret") or 0.0) if _moos.get("enabled") else None
            prev_moos = anchor.get("oos_ret")
            if new_moos is not None and prev_moos is not None:
                meta_rollback_hit = _should_rollback(new_moos, float(prev_moos), self._rollback_threshold)
                meta_cmp_desc = f"OOS {new_moos:.4f} < {float(prev_moos):.4f}×{self._rollback_threshold}"
            else:
                meta_rollback_hit = _should_rollback(best_ret, prev_fitness, self._rollback_threshold)
                meta_cmp_desc = f"fitness {best_ret:.4f} < {prev_fitness:.4f}×{self._rollback_threshold}"
            if _meta_mismatch:
                meta_rollback_hit = False
        if meta_rollback_hit:
            log.warning("[evolve] 元控制器新模型退化 (%s)，回退到最佳版本", meta_cmp_desc)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：模型回退",
                             f"元控制器新模型退化（{meta_cmp_desc}），已回退最佳版本")
            except Exception:  # noqa: BLE001
                pass
            await self.deploy_model(
                "meta_controller", anchor.get("version") or self.zoo.best_version("meta_controller"),
                outcome="rollback", symbol=symbol, timeframe=self._effective_timeframe(),
                reason=f"新模型退化({meta_cmp_desc})")
            self._mark_deployment_status("meta_controller", outcome="rollback",
                                         candidate_fitness=best_ret)
            # P2-13：回退轮次也要落库（此前回退路径漏记，曲线因此长期空白）
            await self._log_round(
                "meta_controller", symbol, self._effective_timeframe(),
                "exchange", best_ret, status="rollback")
            # P4-E1：回退轮补发 evolve_train 事件（与另两条管线一致）
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "meta_controller",
                "fitness": float(best_ret),
                "status": "rollback",
            }, source="evolve_engine"))
            return

        # 3.5 OOS 硬门：元模型同样不得零样本外证据接管实盘
        # （save_agent(is_best=True) 会覆盖 rl/meta 加载的 meta_controller.json）
        gate_ok, gate_reason = _strategy_deploy_gate(result)
        self._meta_controller_status["oos_rejected"] = "" if gate_ok else gate_reason
        if not gate_ok:
            log.warning("[evolve] 元控制器未通过 OOS 硬门，保留原最佳模型：%s", gate_reason)
            try:
                # P4-C2：仅存档不占版本名额（被 OOS 拦截的元模型非有效改进）
                await asyncio.to_thread(self.zoo.save_agent_archive, result["agent"], "meta_controller",
                                        meta={"fitness": best_ret, "timestamp": time.time(),
                                              "data_source": "exchange",
                                              "oos_rejected": gate_reason})
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 被拦截元模型存档失败: %s", e)
            try:
                from core.notify import notify
                await notify(self.db, "进化引擎：OOS 硬门拦截",
                             f"元控制器新模型未通过样本外安检，已保留原最佳模型：{gate_reason}")
            except Exception:  # noqa: BLE001
                pass
            await self.bus.publish(Event(EventType.SYSTEM, {
                "kind": "evolve_train",
                "model": "meta_controller",
                "fitness": best_ret,
                "oos_rejected": gate_reason,
            }, source="evolve_engine"))
            # P2-13：OOS 拦截轮次落库
            _oos_report_m = result.get("oos_report") or {}
            await self._log_round(
                "meta_controller", symbol, self._effective_timeframe(),
                "exchange", best_ret,
                status="oos_rejected",
                oos_ret=_oos_report_m.get("oos_ret", 0.0) or 0.0,
                decay=_oos_report_m.get("decay", 0.0) or 0.0,
                position_ratio=_oos_report_m.get("oos_position_ratio", 0.0) or 0.0)
            return

        # 4. 保存元模型
        import json as _json
        await asyncio.to_thread(
            self.zoo.save_agent, result["agent"], "meta_controller",
            meta={"fitness": best_ret, "timestamp": time.time(),
                  "data_source": "exchange",
                  "oos_ret": (result.get("oos_report") or {}).get("oos_ret"),
                  **_meta_identity})
        self._mark_deployment_status("meta_controller", outcome="accepted",
                                     candidate_fitness=best_ret)

        # 5. 注册为可用策略
        try:
            flat_path = self.zoo._flat_path("meta_controller")
            # 只合并训练负责的键：曾整体替换 params，训练一次就把用户设的
            # meta_window/止损止盈/下单比例等参数打回默认值
            _params = dict((get_dynamic("meta_controller") or {}).get("params") or {})
            # mode 由用户决定：曾无条件写 "drl"，把「统一策略」选的集成模式打回 DRL
            _mode = _params.get("mode") or "drl"
            _params.update({"sub_strategies": sub_strategies,
                            "mode": _mode,
                            "model_path": str(flat_path)})
            _pool_txt = sub_strategies.replace(',', ' / ')
            spec = {
                "name": "meta_controller",
                "title": f"元策略控制器·{symbol}",
                "description": (f"DRL 元控制器：在 {_pool_txt} 子策略间动态选择" if _mode == "drl"
                                else f"集成元控制器：按置信度加权 {_pool_txt}"),
                "logic": "元级 PPO，状态含子策略信号+置信度+历史胜率",
                "executor": "meta_controller",
                "params": _params,
                "risk_tips": ["元策略依赖子策略质量，建议定期重训"],
                "created_by": "evolve_engine",
                "version": f"v{self._meta_controller_status['episode']}",
                "base_symbol": symbol,
                "base_timeframe": self._effective_timeframe(),
                # 关键要求「图上给出买卖点」：元控制器在多个子策略间动态切换，
                # Pine 端无等价实现（既非单一逻辑也无可内联的持仓规则）。
                # 明确留空 + 说明原因，由前端提示，绝不套别的策略模板冒充。
                "pine_code": "",
                "pine_note": ("元策略控制器在子策略间动态切换，无等价 Pine 模板，"
                              "不支持导出图表自动交易代码；如需在图上看买卖点，"
                              "请对单个子策略（如 rl_evolve）导出 Pine"),
            }
            register_dynamic("meta_controller", spec)
            # 持久化
            from core.database import AiStrategy
            from sqlalchemy import select
            async with self.db.session() as s:
                row = (await s.execute(select(AiStrategy).where(AiStrategy.name == "meta_controller"))).scalar_one_or_none()
                if row:
                    row.spec_json = _json.dumps(spec, ensure_ascii=False)
                else:
                    s.add(AiStrategy(name="meta_controller", spec_json=_json.dumps(spec, ensure_ascii=False)))
                await s.commit()
            log.info("[evolve] 元策略控制器已注册（best_ret=%.4f）", best_ret)
        except Exception as e:
            log.warning("[evolve] 元策略注册失败: %s", e)

        # 6. 发布事件
        # P2-13：训练轮次落库
        _oos_report_ok = result.get("oos_report") or {}
        await self._log_round(
            "meta_controller", symbol, self._effective_timeframe(),
            "exchange", float(best_ret),
            status="ok",
            oos_ret=_oos_report_ok.get("oos_ret", 0.0) or 0.0,
            decay=_oos_report_ok.get("decay", 0.0) or 0.0,
            position_ratio=_oos_report_ok.get("oos_position_ratio", 0.0) or 0.0)
        await self.bus.publish(Event(EventType.SYSTEM, {
            "kind": "evolve_train",
            "model": "meta_controller",
            "fitness": float(best_ret),
        }, source="evolve_engine"))

    # ---- 数据获取 ----

    def _data_incremented(self, name: str, df, symbol: str = "",
                          is_demo: bool = False) -> bool:
        """P2-11 增量门：比较数据末尾时间戳是否较上次训练有新变化。

        返回 False 表示"数据无足够新增 K 线，应跳过本轮训练"（连续训练场景下
        60s 间隔 + 5 分钟数据缓存时数据往往没更新，空转重训纯属浪费）。
        - 首次训练（无记录）恒为 True。
        - 有新增 K 线但不足 evolve_min_new_bars 根时仍跳过（等待积累，
          1h 周期默认 2 根 ≈ 2 小时；面板可调大以降低训练频率）。
        - P4-E1：键含 symbol——此前只按管线名键控，标的轮换后拿上一标的的
          tail 比较，min_new_bars 阈值被跨标时间戳绕开（阈值形同虚设）。
        - P4-E1：尾时间戳倒退（数据源清空/截断/跨源切换）视为数据源重置——
          更新记录并放行本轮，避免管线因 prev 永不回落而无限跳过。
        - is_demo：演示数据轮直接放行且**不更新** prev（合成数据的尾时间戳
          ≈now 会污染 prev，导致故障恢复后的首个真实训练被推迟最多
          min_new_bars 根；同周期连续 demo 轮尾戳相同还会触发自身限频）。
        """
        if is_demo:
            return True  # demo 轮不参与增量门、不写 prev
        try:
            idx = df.index
            # 按索引自身单位换算成 epoch 秒：pandas 3 起 DatetimeIndex 可以是
            # s/ms/us/ns 精度（to_datetime(unit="ms") 不再升频到 ns），
            # 用 asi8 // 10**9 假定纳秒会让 ms 索引下的秒数缩水 1000 倍。
            if idx.dtype.kind != "M":
                return True  # 非时间戳索引（demo 兜底数据等），不拦截
            sec = idx.values.astype("datetime64[s]").astype("int64")
            tail_ts = float(sec[-1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return True  # 无法取时间戳，不拦截
        tf = self._effective_timeframe()
        key = f"{name}:{symbol}:{tf}" if symbol else name
        prev = self._last_tail_ts.get(key)
        if prev is None:
            self._last_tail_ts[key] = tail_ts
            return True
        if tail_ts == prev:
            return False  # 末尾时间戳无变化，无新根
        if tail_ts < prev:
            # 数据源已重置/缩短（本地K线库清空后回填更短历史、跨源切换）：
            # 旧 prev 永不回落会导致无限跳过，更新记录并放行本轮恢复训练
            log.info("[evolve] %s 数据尾时间戳倒退（%.0f < 上次 %.0f），"
                     "视为数据源重置，放行本轮", name, tail_ts, prev)
            self._last_tail_ts[key] = tail_ts
            return True
        try:
            # 统计自上次训练以来的新增根数（与 tail_ts 同为 epoch 秒）
            n_new = int((sec > prev).sum())
            # 增量门槛 = max(手工 min_new_bars, 按周期自动换算)。
            # P1：evolve_min_train_gap_sec>0 时按训练周期换算所需新增根数——
            # 训练间隔至少覆盖该秒数（5m 周期下 2 根=10 分钟一轮仍偏密，
            # 换算后与训练成本/行情节奏更贴合）。0 则仅用 min_new_bars。
            min_bars = max(1, int(getattr(settings, "evolve_min_new_bars", 2)))
            gap_sec = int(getattr(settings, "evolve_min_train_gap_sec", 0) or 0)
            if gap_sec > 0:
                try:
                    from backtest.data_loader import _TF_SECONDS
                    tf_sec = int(_TF_SECONDS.get(tf, 1))
                    need_by_gap = max(1, int(math.ceil(gap_sec / max(tf_sec, 1))))
                    min_bars = max(min_bars, need_by_gap)
                except Exception:  # noqa: BLE001
                    pass  # 换算失败回落 min_bars
            if n_new < min_bars:
                return False  # 有变化但根数不足，等待积累
        except Exception:  # noqa: BLE001
            return True
        self._last_tail_ts[key] = tail_ts
        return True

    async def _log_round(self, model: str, symbol: str, timeframe: str,
                         data_source: str, fitness: float, status: str = "ok",
                         oos_ret: float = 0.0, decay: float = 0.0,
                         position_ratio: float = 0.0,
                         selected_factors: Optional[list] = None,
                         round_no: Optional[int] = None,
                         audit: Optional[dict] = None) -> None:
        """P2-13：训练轮次落库（evolve_rounds 表）。失败仅记日志，不影响训练流程。

        Task 5：audit 为人工操作的审计载荷（manual_rollback 等），与
        selected_factors 一样以 JSON 文本落库（老行 read 端兼容两种格式）。
        """
        # 连续拒绝计数先于落库：锁死判定要在落库失败时依然有效（前端提示依赖它）
        self._rounds_total[model] = int(self._rounds_total.get(model, 0)) + 1
        if status == "ok":
            self._reject_streak[model] = 0
        elif status == "demo_blocked":
            # 交易所故障的演示数据轮是"降级训练"不是"模型被拒绝"：
            # 计入 streak 会让故障期误报锁死、诱导用户重置锚点（P1-3）
            pass
        else:
            self._reject_streak[model] = int(self._reject_streak.get(model, 0)) + 1
        try:
            from core.database import EvolveRound
            from sqlalchemy import select, func, delete
            if round_no is None:
                async with self.db.session() as s:
                    cur_max = (await s.execute(
                        select(func.max(EvolveRound.round_no))
                        .where(EvolveRound.model == model))).scalar_one()
                    round_no = int(cur_max or 0) + 1
            async with self.db.session() as s:
                s.add(EvolveRound(
                    model=model, symbol=symbol, timeframe=timeframe,
                    data_source=data_source, round_no=round_no,
                    fitness=float(fitness), oos_ret=float(oos_ret),
                    decay=float(decay), position_ratio=float(position_ratio),
                    selected_factors=json.dumps(list(selected_factors or []),
                                                ensure_ascii=False),
                    status=status,
                    audit_json=json.dumps(audit or {}, ensure_ascii=False),
                ))
                # P2-9：每模型保留最近 N 轮（含 demo/oos_rejected/rollback），
                # 防止 evolve_rounds 表长期运行无界增长、重启全表扫描变慢
                keep = max(100, int(getattr(settings, "evolve_rounds_keep", 500)))
                # run12 E1：cutoff 查询 + delete 合并为单条 DELETE（标量子查询），
                # 省一次 aiosqlite 往返（约 -17% 全序列）；无 cutoff 时（表未超
                # keep 行）子查询返回 NULL，`id < NULL` 恒假 → 不删行，语义等价
                # （.optim/probe_evolve_merge.py：超阈/未超阈边界逐行对比 PASS）
                cutoff_subq = (select(EvolveRound.id)
                               .where(EvolveRound.model == model)
                               .order_by(EvolveRound.id.desc())
                               .offset(keep).limit(1).scalar_subquery())
                await s.execute(delete(EvolveRound)
                                .where(EvolveRound.model == model,
                                       EvolveRound.id < cutoff_subq))
                await s.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 训练轮次落库失败(%s): %s", model, e)

    async def _cross_symbol_oos(self, agent, train_symbol: str,
                                state_window: int) -> Optional[dict]:
        """P1-8：跨标的传导 OOS（低权重附加门）。

        模型在另一个训练标的（非当前标的）上跑一次 greedy 样本外评估，
        检验是否学到了"币种无关"的交易规律而非单标的特有过拟合。
        仅当训练未注入级联因子（state_dim 无因子列）时可执行；
        注入因子时无法在跨标数据上重建同维状态，跳过并返回 None。
        """
        if not getattr(settings, "evolve_cross_symbol_oos", True):
            return None
        others = [s for s in self._symbols if s != train_symbol]
        if not others:
            return None
        sym = others[0]
        try:
            df = await self._fetch_latest_data(symbol=sym)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 跨标评估数据获取失败(%s): %s", sym, e)
            return None
        if df is None or len(df) < max(300, int(self._rolling_window * 0.1)):
            return None
        if len(df) > self._rolling_window:
            df = df.iloc[-self._rolling_window:]
        from drl.env import TradingEnv, run_episode

        def _run_cross_oos() -> Optional[dict]:
            """在工作线程中构造环境并跑 greedy rollout：
            TradingEnv 构造会做滚动特征预计算、run_episode 是纯同步 CPU 前向，
            留在主协程会阻塞事件循环（WebSocket/下单/其他管线全卡顿）。
            与 train_drl 的 asyncio.to_thread 隔离口径一致。"""
            env = TradingEnv(
                df, start_cash=float(getattr(settings, "default_start_cash", 10000.0)),
                fee_rate=0.001,
                vol_penalty=float(getattr(settings, "evolve_vol_penalty", 20.0)),
                slippage=0.0005,
                min_trade_zone=0.05,
                extra_factors=None, state_window=max(1, int(state_window)))
            traj = run_episode(env, lambda s: agent.greedy_action(s))
            return {"symbol": sym, "ret": float(traj["total_ret"]),
                    "position_ratio": float(traj["final_position_ratio"])}

        try:
            return await asyncio.to_thread(_run_cross_oos)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 跨标评估失败(%s): %s", sym, e)
            return None

    def _load_cascade_factor_values(self, df, symbol: str = "") -> Optional[Any]:
        """加载 factor_miner 产出的级联组合因子值，对齐到 df 索引。

        P4-A1：按标的分文件读取（_cascade_factor_values_{symbol}.npy），
        只注入与当前训练标的匹配的因子——彻底杜绝跨标错配（此前单一文件 +
        仅长度校验：BTC 因子被误当 ETH 因子注入，长度一致时静默通过）。
        P4-E1：同标但跨窗口的错位同样危险——因子值与 df 长度一致但窗口已右移
        N 根（数据 TTL 过期后取到更新的 K 线），仅长度校验会静默注入错位数据。
        保存端写了 companion JSON 记录窗口尾时间戳；加载端校验一致才注入。
        返回 numpy 数组（长度 == len(df)），无匹配级联因子时返回 None。
        """
        try:
            import numpy as np

            def _tail_ts():
                """当前 df 窗口尾时间戳（epoch 秒，兼容 DatetimeIndex/int index）。"""
                idx = df.index[-1]
                try:
                    return int(idx.timestamp())
                except Exception:  # noqa: BLE001
                    return int(idx) if idx is not None else None

            def _match_window(values_len: int) -> bool:
                if values_len != len(df):
                    log.warning("[evolve] 级联因子长度 %d 与当前数据 %d 不一致，跳过注入",
                                values_len, len(df))
                    return False
                # P4-E1：有 companion JSON 时校验窗口尾时间戳；无 companion（旧版本
                # 遗留）退化为仅长度校验（无法验证，保守放行以兼容升级前文件）。
                meta_path = cascade_path.with_suffix(".json") if symbol else None
                if meta_path is not None and meta_path.exists():
                    try:
                        import json as _json
                        rec = _json.loads(meta_path.read_text(encoding="utf-8"))
                        if rec.get("tail_ts") != _tail_ts():
                            log.warning(
                                "[evolve] 级联因子窗口尾时间戳 %s 与当前数据 %s 不一致"
                                "（窗口已右移？），拒绝注入", rec.get("tail_ts"), _tail_ts())
                            return False
                    except Exception as e:  # noqa: BLE001
                        log.warning("[evolve] 级联因子时间戳校验失败（放行）: %s", e)
                return True

            if symbol:
                cascade_path = self.zoo.models_dir / f"_cascade_factor_values_{symbol.replace('/', '_')}.npy"
                if not cascade_path.exists():
                    return None
                values = np.asarray(np.load(str(cascade_path)), dtype=float).reshape(-1)
                if not _match_window(len(values)):
                    return None
                return values
            # 无标的参数（旧调用路径兼容）：先试当前轮换标的，再回退旧单文件
            cur = self._next_symbol()
            cur_path = self.zoo.models_dir / f"_cascade_factor_values_{cur.replace('/', '_')}.npy"
            if cur_path.exists():
                cascade_path = cur_path
                values = np.asarray(np.load(str(cur_path)), dtype=float).reshape(-1)
                if _match_window(len(values)):
                    return values
            legacy = self.zoo.models_dir / "_cascade_factor_values.npy"
            if legacy.exists():
                cascade_path = legacy
                values = np.asarray(np.load(str(legacy)), dtype=float).reshape(-1)
                if len(values) == len(df):
                    return values
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 加载级联因子失败: %s", e)
            return None

    def _load_cascade_factor_recipe(self, symbol: str = "") -> Optional[dict]:
        """加载级联组合因子的复算配方（选中因子权重映射）。

        与 _cascade_factor_values_{symbol}.npy 同标同轮保存
        （_cascade_weights_{symbol}.json），部署端把它写进模型文件，
        实盘 rl_adaptive/meta_controller 按配方在滚动缓冲上复算组合因子。
        返回 {"weights": {因子key: 权重}}，找不到/解析失败返回 None（不阻断）。
        """
        try:
            if not symbol:
                return None
            import json as _json
            weights_path = (self.zoo.models_dir
                            / f"_cascade_weights_{symbol.replace('/', '_')}.json")
            if not weights_path.exists():
                return None
            weights = _json.loads(weights_path.read_text(encoding="utf-8"))
            if not isinstance(weights, dict) or not weights:
                return None
            clean = {str(k): float(v) for k, v in weights.items()
                     if isinstance(v, (int, float)) and v}
            if not clean:
                return None
            return {"weights": clean}
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 加载级联因子配方失败: %s", e)
            return None

    def _next_symbol(self) -> str:
        """返回当前训练标的（三管线共享，不递增）。

        P4-A1：标的轮换改由 _mark_pipeline_done 在「三条管线均完成当前标的
        训练后」统一推进，保证 factor_miner 产出的级联因子与 strategy_drl
        训练标的始终一致（此前三管线各自递增轮换 → 级联因子跨标错配）。
        """
        return self._symbols[self._symbol_idx % len(self._symbols)]

    def _mark_pipeline_done(self, pipeline: str) -> None:
        """标记一条管线已在当前标的完成训练；三管线齐全后切换到下一标的。

        仅在真实执行了一次训练（含失败/跳过后的完成）后调用，挂在各
        _train_*_once 的 finally 里。因子挖掘优先级最低（它是链路源头，
        需要先给下游产出因子），因此只有三管线都完成后才推进标的。
        """
        self._cycle_pipelines_done.add(pipeline)
        if len(self._cycle_pipelines_done) >= 3:
            self._cycle_pipelines_done.clear()
            self._symbol_idx += 1
            nxt = self._symbols[self._symbol_idx % len(self._symbols)]
            log.info("[evolve] 三管线均完成 %s 训练，切换至 %s",
                     self._symbols[(self._symbol_idx - 1) % len(self._symbols)], nxt)

    def _generate_symbol_demo(self, symbol: str, n: int, timeframe: str) -> Optional[Any]:
        """生成指定标的的合成K线数据（交易所不可达时兜底），
        使用更接近真实价格水平的参数，让训练更有意义。"""
        # 各标的参数：起始价、波动率
        params = {
            "BTC/USDT": {"start_price": 60000.0, "vol": 0.015},
            "ETH/USDT": {"start_price": 3000.0, "vol": 0.02},
            "SOL/USDT": {"start_price": 150.0, "vol": 0.03},
        }
        p = params.get(symbol, {"start_price": 100.0, "vol": 0.01})
        # 每次生成不同种子，保证训练多样性
        seed = random.randint(0, 99999)
        return generate_demo(n=n, start_price=p["start_price"], vol=p["vol"],
                             timeframe=timeframe, seed=seed)

    async def _fetch_latest_data(self, symbol: str = "") -> Optional[Any]:
        """获取最新行情数据（用于训练），带 5 分钟内存缓存避免频繁拉取。
        本地K线库优先（binance 公开数据域直连，增量拉取缺失段）；
        失败返回 None（由调用方决定是否使用演示数据兜底）。

        Args:
            symbol: 标的（如 "BTC/USDT"），留空使用 settings.default_symbol。
        """
        from backtest.data_loader import load_klines_cached

        symbol = symbol or settings.default_symbol
        # 训练时间周期：面板独立配置（evolve_timeframe）优先，缺省回落 default_timeframe
        timeframe = self._effective_timeframe()
        cache_key = f"{settings.default_exchange}:{symbol}:{timeframe}"
        now = time.time()
        # 缓存命中且在 TTL 内
        if cache_key in self._data_cache:
            cached_ts, cached_df = self._data_cache[cache_key]
            if now - cached_ts < self._data_cache_ttl:
                log.debug("[evolve] 使用缓存数据（%ds 前获取）", now - cached_ts)
                return cached_df

        # P0-1：修复窗口上限 bug——原先 min(max(rolling_window,1000),3000) 把配置的
        # 5000 根永远压到最多 3000 根。现在拉满 rolling_window，且 load_klines_cached
        # 支持 max_candles 历史回填（本地不足时自动翻页补足）。
        limit = max(int(self._rolling_window), 1000)
        max_candles = max(int(self._rolling_window), 1000)

        # P4-E1：拉数串行锁——三管线 + 跨标 OOS 并发打交易所会放大 429；
        # 统一串行化后同一时刻只有一条 fetch 在飞（429 重试逻辑见 data_loader）。
        try:
            async with self._fetch_lock:
                df = await load_klines_cached(settings.default_exchange, symbol, timeframe,
                                              limit=limit, max_candles=max_candles)
            if df is not None and not df.empty:
                log.info("[evolve] K线数据获取成功：%d 根K线 (%s %s)",
                         len(df), symbol, timeframe)
                # 写入缓存
                # P4-E1：容量上限——symbol×timeframe 组合变更后旧键永不清理，
                # 长时间运行+频繁改配置下无界增长；超过上限时淘汰最旧键。
                self._data_cache[cache_key] = (now, df)
                if len(self._data_cache) > _DATA_CACHE_MAX_KEYS:
                    oldest_key = min(self._data_cache, key=lambda k: self._data_cache[k][0])
                    self._data_cache.pop(oldest_key, None)
                return df
        except Exception as e:
            log.warning("[evolve] 拉取K线数据失败: %s", e)
            # 缓存中有旧数据则返回（即使过期，也比完全没数据好）
            if cache_key in self._data_cache:
                _, cached_df = self._data_cache[cache_key]
                log.info("[evolve] 拉取失败，返回缓存数据")
                return cached_df

        return None