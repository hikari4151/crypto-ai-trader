"""观测方法 mixin（2026-09-22，P2-④ 第二阶段拆分）。

从 drl/evolve_engine.py 抽出的观测/候选门方法（walkforward/gap01/longwindow/
候选长窗门），作为 mixin 供 EvolveEngine 继承——self 仍为 EvolveEngine 实例，
依赖其 zoo/_effective_timeframe/_strategy_drl_status/_fetch_latest_data 等。
"""
import asyncio
import time
from typing import Optional

import logging

log = logging.getLogger("drl.evolve_observations")

from config.settings import settings
from drl.model_zoo import ModelZoo
from drl.evolve_rules import _longwindow_verdict


class EvolveObservationsMixin:
    async def _observe_strategy_drl_walkforward(self, df=None) -> Optional[dict]:
        """每轮 strategy DRL 训练后的 walk-forward 多折观测（同品种，不改接受逻辑）。

        用部署中的最佳模型（zoo.load_agent("strategy_drl")）在最近
        evolve_strategy_drl_walkforward_bars 根K线上切 folds 段做 greedy 评估，
        结果写 models/walkforward_obs.json + 日志摘要，作为"换一段历史也赚"的
        常规可见指标。无模型 / 开关关闭 / 数据不足 → 静默跳过（返回 None）。
        """
        if not getattr(settings, "evolve_strategy_drl_walkforward_obs", True):
            return None
        folds = max(1, int(getattr(settings, "evolve_strategy_drl_walkforward_folds", 2) or 0))
        bars = max(2000, int(getattr(settings, "evolve_strategy_drl_walkforward_bars", 6000) or 0))
        min_trades = max(1, int(getattr(settings, "evolve_walkforward_min_trades", 8) or 0))
        agent = self.zoo.load_agent("strategy_drl")
        if agent is None:
            log.info("[evolve] walk-forward观测跳过：无 strategy_drl 最佳模型")
            return None
        symbol = self._strategy_drl_status.get("symbol") or self._next_symbol()
        if df is None:
            df = await self._fetch_latest_data(symbol=symbol)
        # 2026-09-21：观测口径与训练窗口解耦——考试要覆盖更长历史（bars），
        # 训练 rolling_window 保持较小（成本可控）；df 不足 bars 时按 bars 独立重拉。
        if df is None or len(df) < bars:
            try:
                from backtest.data_loader import load_klines_cached
                df2 = await load_klines_cached(
                    getattr(settings, "default_exchange", "binance"),
                    symbol, self._effective_timeframe(),
                    limit=bars, max_candles=bars)
                if df2 is not None and len(df2) >= (len(df) if df is not None else 0):
                    df = df2
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] walk-forward观测按 %d 根重拉失败（沿用现有数据）: %s", bars, e)
        if df is None or len(df) < 500:
            log.info("[evolve] walk-forward观测跳过：数据不足")
            return None
        if len(df) > bars:
            df = df.iloc[-bars:]
        state_window = int(getattr(settings, "evolve_strategy_drl_state_window", 4))
        warmup = max(300, state_window * 50)

        # 因子列口径：flat 元数据优先；缺失时用已部署权重 + 折内局部标准化近似
        import json as _json
        factor_composite: Optional[dict] = None
        factor_mu, factor_sd = 0.0, 1.0
        flat_path = self.zoo._flat_path("strategy_drl")
        if flat_path.exists():
            try:
                md = _json.loads(flat_path.read_text(encoding="utf-8"))
                fc = md.get("factor_composite")
                if isinstance(fc, dict) and fc.get("weights"):
                    factor_composite = fc
                factor_mu = float(md.get("factor_mu", 0.0) or 0.0)
                factor_sd = float(md.get("factor_sd", 1.0) or 1.0)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] walk-forward读取flat元数据失败（按无因子列）: %s", e)

        from drl.env import TradingEnv, run_episode
        from drl.walkforward import (build_factor_column, expanding_fold_bounds,
                                     metrics_from_traj, summarize_folds)
        probe = TradingEnv(df.iloc[:min(200, len(df))], start_cash=10000.0, fee_rate=0.001,
                           vol_penalty=float(getattr(settings, "evolve_vol_penalty", 40.0)),
                           slippage=0.0005, min_trade_zone=0.05,
                           extra_factors=None, state_window=state_window)
        base_dim = probe.state_dim
        factor_mode = "model_metadata" if factor_composite is not None else (
            "deployed_weights_fold_local_approx" if agent.state_dim > base_dim else "none")

        def _run() -> Optional[dict]:
            folds_out = []
            try:
                for i, (s, e) in enumerate(expanding_fold_bounds(len(df), warmup, folds), 1):
                    seg = df.iloc[s:e]
                    if len(seg) < state_window + 10:
                        continue
                    extra = None
                    if factor_composite is not None:
                        extra = build_factor_column(seg, factor_composite.get("weights") or {},
                                                    mu=factor_mu, sd=factor_sd)
                    elif agent.state_dim > base_dim:
                        wpath = self.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
                        if wpath.exists():
                            w = _json.loads(wpath.read_text(encoding="utf-8"))
                            raw = build_factor_column(seg, w, mu=0.0, sd=1.0)
                            if raw is not None:
                                n_train = max(60, int(len(seg) * 0.6))
                                mu = float(raw[:n_train].mean())
                                sd = float(raw[:n_train].std()) + 1e-8
                                extra = (raw - mu) / sd
                    env = TradingEnv(seg, start_cash=10000.0, fee_rate=0.001,
                                     vol_penalty=float(getattr(settings, "evolve_vol_penalty", 40.0)),
                                     slippage=0.0005, min_trade_zone=0.05,
                                     extra_factors=extra, state_window=state_window)
                    if env.state_dim != agent.state_dim:
                        log.warning("[evolve] walk-forward跳过：模型state_dim=%d vs 环境%d",
                                    agent.state_dim, env.state_dim)
                        continue
                    traj = run_episode(env, lambda st: agent.greedy_action(st))
                    m = metrics_from_traj(traj, self._effective_timeframe())
                    m.update({"fold": i, "start": str(seg.index[0]), "end": str(seg.index[-1]),
                              "bars": len(seg), "symbol": symbol,
                              "timeframe": self._effective_timeframe()})
                    folds_out.append(m)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] walk-forward观测执行失败: %s", e)
                return None
            if not folds_out:
                return None
            summary = summarize_folds(folds_out, min_trades=min_trades)
            return {"symbol": symbol, "timeframe": self._effective_timeframe(),
                    "model": "strategy_drl(best)", "state_window": state_window,
                    "bars": len(df), "warmup": warmup, "min_trades": min_trades,
                    "factor_mode": factor_mode, "folds": folds_out,
                    **summary, "generated_at": time.time()}

        report = await asyncio.to_thread(_run)
        if report is None:
            return None
        try:
            obs_path = self.zoo.models_dir / "walkforward_obs.json"
            ModelZoo._atomic_write_bytes(obs_path, ModelZoo._json_bytes(report))
            log.info("[evolve] walk-forward观测 %s：%s（%s）",
                     symbol, report["verdict"], report["reason"])
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] walk-forward观测落盘失败: %s", e)
        return report

    async def _observe_gap01(self, df=None) -> Optional[dict]:
        """GAP-01 实盘验收观测（同品种单标的口径，诊断报告建议第6条(b)）。

        对部署中的最佳模型在最近 N 根K线上跑同品种多折 walk-forward，
        分两档成本（基准 fee/slippage 与 ×evolve_gap01_cost_multiplier），
        合并为 PASS/INCONCLUSIVE/FAIL（drl.gap01.summarize_gap01）。
        单标的部署豁免跨品种门禁（LIVE_READINESS E7）。结果写
        models/gap01_obs.json + 日志；不改接受逻辑。
        """
        if not getattr(settings, "evolve_gap01_obs", True):
            return None
        folds = max(1, int(getattr(settings, "evolve_gap01_folds", 2) or 0))
        bars = max(2000, int(getattr(settings, "evolve_gap01_bars", 6000) or 0))
        min_trades = max(1, int(getattr(settings, "evolve_gap01_min_trades", 8) or 0))
        mult = max(1.0, float(getattr(settings, "evolve_gap01_cost_multiplier", 2.0) or 0.0))
        agent = self.zoo.load_agent("strategy_drl")
        if agent is None:
            log.info("[evolve] GAP-01观测跳过：无 strategy_drl 最佳模型")
            return None
        symbol = self._strategy_drl_status.get("symbol") or self._next_symbol()
        if df is None:
            df = await self._fetch_latest_data(symbol=symbol)
        # 2026-09-21：GAP-01 观测与训练窗口解耦——考试按 bars（15000）覆盖更长
        # 历史，训练 rolling_window 保持较小；df 不足 bars 时按 bars 独立重拉。
        if df is None or len(df) < bars:
            try:
                from backtest.data_loader import load_klines_cached
                df2 = await load_klines_cached(
                    getattr(settings, "default_exchange", "binance"),
                    symbol, self._effective_timeframe(),
                    limit=bars, max_candles=bars)
                if df2 is not None and len(df2) >= (len(df) if df is not None else 0):
                    df = df2
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] GAP-01观测按 %d 根重拉失败（沿用现有数据）: %s", bars, e)
        if df is None or len(df) < 500:
            log.info("[evolve] GAP-01观测跳过：数据不足")
            return None
        if len(df) > bars:
            df = df.iloc[-bars:]
        state_window = int(getattr(settings, "evolve_strategy_drl_state_window", 4))
        warmup = max(300, state_window * 50)

        import json as _json
        factor_composite: Optional[dict] = None
        factor_mu, factor_sd = 0.0, 1.0
        flat_path = self.zoo._flat_path("strategy_drl")
        if flat_path.exists():
            try:
                md = _json.loads(flat_path.read_text(encoding="utf-8"))
                fc = md.get("factor_composite")
                if isinstance(fc, dict) and fc.get("weights"):
                    factor_composite = fc
                factor_mu = float(md.get("factor_mu", 0.0) or 0.0)
                factor_sd = float(md.get("factor_sd", 1.0) or 1.0)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] GAP-01读取flat元数据失败（按无因子列）: %s", e)

        from drl.env import TradingEnv, run_episode
        from drl.walkforward import (build_factor_column, expanding_fold_bounds,
                                     metrics_from_traj, summarize_folds)
        from drl.gap01 import summarize_gap01
        _vol = float(getattr(settings, "evolve_vol_penalty", 40.0))
        probe = TradingEnv(df.iloc[:min(200, len(df))], start_cash=10000.0, fee_rate=0.001,
                           vol_penalty=_vol, slippage=0.0005, min_trade_zone=0.05,
                           extra_factors=None, state_window=state_window)
        base_dim = probe.state_dim
        factor_mode = ("model_metadata" if factor_composite is not None else
                       ("deployed_weights_fold_local_approx" if agent.state_dim > base_dim else "none"))

        def _run_tier(fee_rate: float, slippage: float) -> Optional[dict]:
            folds_out = []
            try:
                for i, (s, e) in enumerate(expanding_fold_bounds(len(df), warmup, folds), 1):
                    seg = df.iloc[s:e]
                    if len(seg) < state_window + 10:
                        continue
                    extra = None
                    if factor_composite is not None:
                        extra = build_factor_column(seg, factor_composite.get("weights") or {},
                                                    mu=factor_mu, sd=factor_sd)
                    elif agent.state_dim > base_dim:
                        wpath = self.zoo.models_dir / f"_cascade_weights_{symbol.replace('/', '_')}.json"
                        if wpath.exists():
                            w = _json.loads(wpath.read_text(encoding="utf-8"))
                            raw = build_factor_column(seg, w, mu=0.0, sd=1.0)
                            if raw is not None:
                                n_train = max(60, int(len(seg) * 0.6))
                                mu = float(raw[:n_train].mean())
                                sd = float(raw[:n_train].std()) + 1e-8
                                extra = (raw - mu) / sd
                    env = TradingEnv(seg, start_cash=10000.0, fee_rate=fee_rate,
                                     vol_penalty=_vol, slippage=slippage,
                                     min_trade_zone=0.05, extra_factors=extra,
                                     state_window=state_window)
                    if env.state_dim != agent.state_dim:
                        continue
                    traj = run_episode(env, lambda st: agent.greedy_action(st))
                    m = metrics_from_traj(traj, self._effective_timeframe())
                    m.update({"fold": i, "start": str(seg.index[0]), "end": str(seg.index[-1]),
                              "bars": len(seg), "symbol": symbol,
                              "timeframe": self._effective_timeframe(),
                              "fee_rate": fee_rate, "slippage": slippage})
                    folds_out.append(m)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] GAP-01观测档执行失败: %s", e)
                return None
            if not folds_out:
                return None
            return {"summary": summarize_folds(folds_out, min_trades=min_trades),
                    "folds": folds_out}

        base = await asyncio.to_thread(_run_tier, 0.001, 0.0005)
        cost2 = await asyncio.to_thread(_run_tier, 0.001 * mult, 0.0005 * mult)
        if base is None and cost2 is None:
            return None
        base_sum = (base or {}).get("summary") or {"verdict": "INCONCLUSIVE",
                                                   "n_folds": 0, "positive_folds": 0,
                                                   "win_rate": 0.0, "median_ret": 0.0,
                                                   "mean_ret": 0.0, "reason": "无评估折"}
        cost2_sum = (cost2 or {}).get("summary") or dict(base_sum)
        merged = summarize_gap01(base_sum, cost2_sum)
        report = {
            "symbol": symbol, "timeframe": self._effective_timeframe(),
            "model": "strategy_drl(best)", "state_window": state_window,
            "bars": len(df), "warmup": warmup, "folds": folds,
            "min_trades": min_trades, "cost_multiplier": mult,
            "factor_mode": factor_mode,
            "base_cost": {"fee_rate": 0.001, "slippage": 0.0005,
                          "summary": base_sum, "folds": (base or {}).get("folds", [])},
            "cost2": {"fee_rate": round(0.001 * mult, 5), "slippage": round(0.0005 * mult, 5),
                      "summary": cost2_sum, "folds": (cost2 or {}).get("folds", [])},
            "verdict": merged["verdict"], "reason": merged["reason"],
            "generated_at": time.time(),
        }
        try:
            obs_path = self.zoo.models_dir / "gap01_obs.json"
            ModelZoo._atomic_write_bytes(obs_path, ModelZoo._json_bytes(report))
            log.info("[evolve] GAP-01观测 %s：%s（%s）",
                     symbol, report["verdict"], report["reason"])
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] GAP-01观测落盘失败: %s", e)
        return report

    # ---- 长窗回放观测（2026-09-21，§16.5 升级：实盘候选须 30/90 天长窗盈利） ----

    async def _observe_strategy_drl_longwindow(self) -> None:
        """每轮训练后对部署模型跑 30/90 天真实数据引擎撮合回放，写
        models/longwindow_obs.json + status 字段；不改接受逻辑（防呆观测）。

        背景：v44 短窗 stable PASS 但 30 天 -26.6% / 90 天 -67.8%——"近期
        窗口效应"曾冒充实盘候选；本观测把长窗回放纳入常规可见指标。
        """
        if not getattr(settings, "evolve_strategy_drl_longwindow_obs", True):
            return
        if not self.zoo._flat_path("strategy_drl").exists():
            return
        bars30 = max(1000, int(getattr(settings, "evolve_strategy_drl_longwindow_30d_bars", 8640)))
        bars90 = max(1000, int(getattr(settings, "evolve_strategy_drl_longwindow_90d_bars", 25920)))
        symbol = self._strategy_drl_status.get("symbol") or self._next_symbol()
        try:
            from backtest.data_loader import load_klines_cached
            df = await load_klines_cached(
                getattr(settings, "default_exchange", "binance"),
                symbol, self._effective_timeframe(),
                limit=bars90, max_candles=bars90)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 长窗回放观测拉取失败: %s", e)
            self._strategy_drl_status["longwindow_verdict"] = "INCONCLUSIVE"
            return
        if df is None or len(df) < 1000:
            self._strategy_drl_status["longwindow_verdict"] = "INCONCLUSIVE"
            return
        df30 = df.iloc[-bars30:]
        df90 = df.iloc[-bars90:]
        data_ok = len(df30) >= 5000 and len(df90) >= 5000
        model_path = str(self.zoo._flat_path("strategy_drl"))
        timeframe = self._effective_timeframe()

        def _run(df_part):
            from backtest.engine import BacktestConfig, run_backtest
            cfg = BacktestConfig(symbol=symbol, timeframe=timeframe,
                                 strategy_name="rl_adaptive",
                                 strategy_params={"model_path": model_path},
                                 start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
            return run_backtest(df_part, cfg)

        async def _run_win(bars: int) -> Optional[float]:
            dfp = df.iloc[-bars:]
            try:
                r = await asyncio.to_thread(_run, dfp)
                return float((r.get("metrics") or {}).get("total_return") or 0.0)
            except Exception as e:  # noqa: BLE001
                log.warning("[evolve] 长窗回放失败(%d根): %s", bars, e)
                return None

        # P2-③（2026-09-22）：30d/90d 并行回放——共用同一数据源 df
        #（90d 数据切出 30d 段，避免重复拉取/预处理），两档 asyncio.gather 同时跑。
        ret30, ret90 = await asyncio.gather(_run_win(bars30), _run_win(bars90))
        verdict = _longwindow_verdict(ret30, ret90, data_ok)
        self._strategy_drl_status["longwindow_verdict"] = verdict["verdict"]
        self._strategy_drl_status["longwindow_30d_ret"] = ret30
        self._strategy_drl_status["longwindow_90d_ret"] = ret90
        report = {
            "verdict": verdict["verdict"], "reason": verdict["reason"],
            "30d_ret": ret30, "90d_ret": ret90,
            "bars_30d": len(df30), "bars_90d": len(df90),
            "generated_at": time.time(),
        }
        try:
            obs_path = self.zoo.models_dir / "longwindow_obs.json"
            ModelZoo._atomic_write_bytes(obs_path, ModelZoo._json_bytes(report))
            log.info("[evolve] 长窗回放观测：%s（30d=%s 90d=%s）",
                     verdict["verdict"], ret30, ret90)
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 长窗回放观测落盘失败: %s", e)

    async def _candidate_longwindow_gate(self, agent, symbol: str, train_cfg: dict,
                                         factor_composite: Optional[dict] = None,
                                         factor_mu: float = 0.0,
                                         factor_sd: float = 1.0) -> tuple[bool, str]:
        """部署候选长窗门（2026-09-22，§20 落地）：候选模型部署前跑 30 天长窗回放，
        30d>0 才允许部署（PASS）；FAIL 不部署（仅存档），防高换手短窗模型
        （v50/52/53/55 类长窗 -10%~-56%）顶替长窗稳健模型。数据不足/异常/
        门槛关闭 → 放行（保守，不阻断进化）。
        """
        if not getattr(settings, "evolve_strategy_drl_deploy_requires_longwindow", False):
            return True, ""
        try:
            from backtest.data_loader import load_klines_cached
            # 2026-09-22 §20.3：部署门口径与观测对齐——可配置加 90 天档
            #（默认仅 30d 控成本；开启后 30d 与 90d 均正才部署）。
            _90d_on = bool(getattr(settings, "evolve_strategy_drl_deploy_longwindow_90d", False))
            _windows = [(8640, "30d")] + ([(25920, "90d")] if _90d_on else [])
            max_bars = 25920 if _90d_on else 8640
            df = await load_klines_cached(
                getattr(settings, "default_exchange", "binance"),
                symbol, self._effective_timeframe(),
                limit=max_bars, max_candles=max_bars)
            if df is None or len(df) < 5000:
                return True, "长窗数据不足（放行）"
            import json as _json
            import tempfile as _tmp
            d = agent.to_dict()
            d["factor_composite"] = factor_composite or {}
            d["factor_mu"] = float(factor_mu or 0.0)
            d["factor_sd"] = float(factor_sd or 1.0)
            d["factor_expression"] = ""
            d["state_window"] = int(train_cfg.get("state_window", 4))
            with _tmp.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
                _json.dump(d, f)
                fp = f.name
            from backtest.engine import BacktestConfig, run_backtest

            async def _run_win(bars: int) -> Optional[float]:
                dfp = df.iloc[-bars:]
                cfg = BacktestConfig(symbol=symbol, timeframe=self._effective_timeframe(),
                                     strategy_name="rl_adaptive",
                                     strategy_params={"model_path": fp, "buy_zone": 0.02},
                                     start_cash=10000.0, fee_rate=0.001, slippage=0.0005)
                r = await asyncio.to_thread(run_backtest, dfp, cfg)
                return float((r.get("metrics") or {}).get("total_return") or 0.0)

            # P2-③（2026-09-22）：多窗口并行回放（共用数据源 df）
            _results = await asyncio.gather(*[_run_win(bars) for bars, _ in _windows])
            for (bars, tag), ret in zip(_windows, _results):
                if ret is None:
                    return True, f"长窗{tag}回放失败（放行）"
                if ret <= 0:
                    return False, f"长窗{tag}={ret:+.4f}（FAIL，不部署）"
            _tags = "/".join(t for _, t in _windows)
            return True, f"长窗{_tags}均正（PASS）"
        except Exception as e:  # noqa: BLE001
            log.warning("[evolve] 部署长窗门计算失败（放行）: %s", e)
            return True, f"长窗门计算失败（放行）: {e}"

    # ---- 元策略控制器持续训练 ----
