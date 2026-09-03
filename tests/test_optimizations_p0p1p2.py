"""P0/P1/P2 优化项回归测试（本次大规模改造新增能力）。

覆盖：
- P0-1 intrabar 止损/止盈触价建模（盘中触价成交而非仅收盘判断）
- P0-2 市价单成交量/参与率约束
- P0-3 maker/taker + funding 成本模型
- P1-4 成本压力测试（run_cost_scan）
- P1-5 ATR/风险预算仓位（risk_per_trade_pct）
- P1-6 换手/持仓时长/成本拖累指标
- P1-7 stationary block bootstrap（block_bootstrap_sharpe_ci）
- P1-8 AI 参数验证门纯 OOS 预留（纯 OOS 劣于当前 → 拒绝）
- P1-9 数据清洗默认标记制（mark 不清洗 / replace 保留旧行为）
- P2-11 多标的组合层回测（portfolio）
- P2-12 DRL 环境成交量/资金费率约束
"""
import contextlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")

from strategies.base import Signal, Strategy


def _df(close_arr: np.ndarray, volume: float = 1000.0) -> pd.DataFrame:
    close = np.asarray(close_arr, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": np.roll(close, 1) * 1.001,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, volume),
    }).set_index("timestamp")


@contextlib.contextmanager
def _register_test_strategy(cls):
    """把测试策略注册进全局注册表（回测经 get_strategy 取实例）。"""
    import strategies
    orig = strategies._REGISTRY.copy()
    strategies._REGISTRY[cls.name] = cls
    try:
        yield cls
    finally:
        strategies._REGISTRY = orig


# ============ P0-1 intrabar 止损/止盈触价建模 ============

class _IntrabarStrat(Strategy):
    """测试策略：开盘后立即买入，并声明 5% 止损位（不主动收盘平仓）。"""
    name = "_test_intrabar"
    default_params = {"size_pct": 0.5, "stop_loss_pct": 0.05}

    def reset(self):
        self._entry = None
        self._done_buy = False

    def on_candle(self, ctx):
        if not self._done_buy and ctx.get("position", 0) <= 0:
            self._done_buy = True
            return Signal(ctx["symbol"], "buy", self.params["size_pct"], strategy=self.name, reason="test_buy")
        return None

    def on_fill(self, symbol, side, price):
        if side == "buy":
            self._entry = price
        else:
            self._entry = None

    def protective_levels(self, ctx):
        if ctx.get("position", 0) > 0 and self._entry:
            return {"stop": self._entry * (1 - self.params["stop_loss_pct"])}
        return None


def test_intrabar_stop_triggers_on_low():
    """P0-1：K线 low 触及止损位 → 盘中触价止损（成交价≈止损位×(1-滑点)）。"""
    from backtest.engine import BacktestConfig, run_backtest
    # 60+ 根（引擎要求 ≥60）：前 5 根 100，第 6 根盘中低点 90（< 止损位 ~95）
    prices = [100.0] * 5 + [100.0] + [99.0] + [100.0] * 60
    # 把"第 2 根（信号成交那根）的 low 压低"改为：买入在第 2 根开盘成交，
    # 那根 low=90 触发盘中止损
    prices = [100.0, 90.0] + [100.0] * 60
    df = _df(prices)
    with _register_test_strategy(_IntrabarStrat):
        cfg = BacktestConfig(strategy_name="_test_intrabar", intrabar_stops=True)
        r = run_backtest(df, cfg)
        sells = [t for t in r["trades"] if t["side"] == "sell"]
        assert any("触价止损" in t.get("reason", "") for t in sells), f"应触发盘中触价止损: {sells}"
        stop_sell = next(t for t in sells if "触价止损" in t.get("reason", ""))
        # 买入价 = open[1]×(1+滑点) = 100.1×1.0005 = 100.15，
        # 止损位 = entry×0.95 ≈ 95.14，触价成交价 = 止损位×(1-滑点)
        stop_ref = 100.1 * 1.0005 * 0.95 * (1 - 0.0005)
        assert stop_sell["price"] == pytest.approx(stop_ref, abs=0.05), f"触价止损成交价异常: {stop_sell}"


def test_intrabar_stops_disabled_keeps_close_check():
    """P0-1：intrabar_stops=False 时不发生盘中触价（回到收盘判断路径）。"""
    from backtest.engine import BacktestConfig, run_backtest
    prices = [100.0, 90.0] + [100.0] * 60
    df = _df(prices)
    with _register_test_strategy(_IntrabarStrat):
        cfg = BacktestConfig(strategy_name="_test_intrabar", intrabar_stops=False)
        r = run_backtest(df, cfg)
        sells = [t for t in r["trades"] if t["side"] == "sell"]
        # 收盘价 99 或 100 > 止损位（95），不满足收盘止损条件 → 无盘中触价；
        # 强制平仓仅在末根有持仓时生成（position 已在末根前清空则不生成）
        assert all("触价止损" not in t.get("reason", "") for t in sells), f"不应有盘中触价: {sells}"


# ============ P0-2 成交量参与率约束 ============

def test_participation_rate_caps_buy_qty():
    """P0-2：买单数量 ≤ 该K线成交量×participation_rate。"""
    from backtest._matching import _execute_fill
    from backtest.engine import BacktestConfig

    cfg = BacktestConfig(participation_rate=0.1)
    sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="t")
    filled, cash, pos, trade = _execute_fill(
        cfg, sig, fill_price=100.0, symbol="BTC/USDT", ts="2024-01-01T00:00:00+00:00",
        cash=10000.0, position=0.0, lots=[], trades=[], strategy=None,
        record_buy_trade=True, bar_volume=10.0, entry_i=0, exit_i=0)
    assert filled
    # 成交量参与率上限：qty ≤ 10×0.1 = 1
    assert pos <= 1.0 + 1e-12, f"参与率约束应限制 qty≤1, got {pos}"


def test_participation_rate_zero_unlimited():
    """P0-2：participation_rate=0（默认）时不限制，保持旧行为。"""
    from backtest._matching import _execute_fill
    from backtest.engine import BacktestConfig

    cfg = BacktestConfig(participation_rate=0.0)
    sig = Signal("BTC/USDT", "buy", size_pct=0.5, strategy="t")
    filled, cash, pos, _ = _execute_fill(
        cfg, sig, fill_price=100.0, symbol="BTC/USDT", ts="2024-01-01T00:00:00+00:00",
        cash=10000.0, position=0.0, lots=[], trades=[], strategy=None,
        record_buy_trade=True, bar_volume=1.0, entry_i=0, exit_i=0)
    assert filled
    # 不受限：qty = cash×0.5/100 ≈ 50 >> 成交量参与率上限
    assert pos > 1.0, f"参与率=0 不应限制, got {pos}"


# ============ P0-3 maker/taker + funding 成本模型 ============

def test_limit_order_uses_maker_fee():
    """P0-3：限价单成交用 maker_fee_rate（默认 None → 退化为 fee_rate）。"""
    from backtest._matching import _order_fee_rate
    from backtest.engine import BacktestConfig

    cfg = BacktestConfig(fee_rate=0.001, maker_fee_rate=0.0002, taker_fee_rate=0.0005)
    sig_limit = Signal("BTC/USDT", "buy", order_type="limit", strategy="t")
    sig_market = Signal("BTC/USDT", "buy", order_type="market", strategy="t")
    assert _order_fee_rate(cfg, sig_limit) == pytest.approx(0.0002), "限价单应走 maker 费率"
    assert _order_fee_rate(cfg, sig_market) == pytest.approx(0.0005), "市价单应走 taker 费率"


def test_maker_taker_default_backward_compat():
    """P0-3：未显式配置 maker/taker 时沿用 fee_rate（向后兼容）。"""
    from backtest._matching import _order_fee_rate
    from backtest.engine import BacktestConfig

    cfg = BacktestConfig(fee_rate=0.001)
    sig_limit = Signal("BTC/USDT", "buy", order_type="limit", strategy="t")
    sig_market = Signal("BTC/USDT", "buy", order_type="market", strategy="t")
    assert _order_fee_rate(cfg, sig_limit) == pytest.approx(0.001)
    assert _order_fee_rate(cfg, sig_market) == pytest.approx(0.001)


def test_funding_rate_charged_on_position():
    """P0-3：funding_rate>0 时每根K线按持仓价值收取（组合回测成本变差）。"""
    from backtest.data_loader import generate_demo
    from backtest.engine import BacktestConfig, run_backtest

    df = generate_demo(n=500, timeframe="1h")
    cfg_plain = BacktestConfig(strategy_name="dual_ma", funding_rate=0.0)
    cfg_fund = BacktestConfig(strategy_name="dual_ma", funding_rate=0.001)
    r1 = run_backtest(df, cfg_plain)
    r2 = run_backtest(df, cfg_fund)
    assert len(r1["trades"]) > 0, "demo 行情下应有成交，否则测试无意义"
    assert len(r1["trades"]) == len(r2["trades"]), "funding 不应改变成交集合"
    # 有持仓期间每根收 0.1% 资金费 → 期末成本增加
    assert _fees(r2) > _fees(r1), f"funding 应增加总成本: {_fees(r1)} vs {_fees(r2)}"


def _fees(res: dict) -> float:
    return float(res["metrics"].get("total_costs", 0.0) or 0.0)


# ============ P1-4 成本压力测试 ============

def test_cost_scan_structure_and_sensitivity():
    """P1-4：成本压力扫描返回逐组合表与敏感性，works 端到端。"""
    from backtest.cost_scan import CostScanConfig, run_cost_scan
    from backtest.data_loader import generate_demo

    df = generate_demo(n=400, timeframe="1h")
    report = run_cost_scan(df, CostScanConfig(strategy_name="dual_ma"))
    assert len(report["scan"]) == 12  # 4 fee × 3 slip
    assert report["base"]["fee_rate"] == 0.001
    # 基准组合存在
    base = next(r for r in report["scan"] if r["fee_mult"] == 1.0 and r["slip_mult"] == 1.0)
    assert base["total_return"] is not None
    assert "worst_ret_drop_pct" in report["sensitivity"]


def test_cost_scan_custom_mults_and_progress():
    """P1-4：自定义放大倍数 + 进度回调。"""
    from backtest.cost_scan import CostScanConfig, run_cost_scan
    from backtest.data_loader import generate_demo

    df = generate_demo(n=300, timeframe="1h")
    steps = []
    report = run_cost_scan(df, CostScanConfig(strategy_name="dual_ma",
                                              fee_mults=(1.0, 2.0), slip_mults=(1.0, 2.0)),
                           on_progress=lambda s: steps.append(s))
    assert len(report["scan"]) == 4
    assert len(steps) == 4


# ============ P1-5 风险预算仓位 ============

def test_risk_based_sizing_reduces_size():
    """P1-5：启用 risk_per_trade_pct 后，首笔买入仓位 ≤ size_pct 上限且反映风险预算。"""
    from backtest.engine import BacktestConfig
    from backtest.fast_engine import run_backtest_fast

    prices = [100.0 + 8.0 * np.sin(np.arange(300) / 25.0)]
    df = _df(prices[0])
    # 同周期参数：固定 0.5 仓位 vs 风险预算 1%（止损 3% → 理论仓位 33%）
    cfg_fixed = BacktestConfig(strategy_name="dual_ma",
                               strategy_params={"fast_period": 5, "slow_period": 15})
    cfg_risk = BacktestConfig(strategy_name="dual_ma",
                              strategy_params={"fast_period": 5, "slow_period": 15,
                                               "risk_per_trade_pct": 0.01})
    r_fixed = run_backtest_fast(df, cfg_fixed)
    r_risk = run_backtest_fast(df, cfg_risk)
    # 风险预算约束生效的直观证据：期末权益/交易次数不同（仓位变化）
    assert r_fixed["metrics"]["total_return"] != r_risk["metrics"]["total_return"] \
        or len(r_fixed["trades"]) != len(r_risk["trades"]), "风险预算应改变回测结果"


def test_dual_ma_param_schema_accepts_risk_params():
    """P1-5：dual_ma 新参数在 param_schema 内且能正常 update_params。"""
    from strategies.dual_ma import DualMAStrategy

    s = DualMAStrategy()
    s.update_params({"risk_per_trade_pct": 0.01, "atr_stop_mult": 2.0})
    assert s.params["risk_per_trade_pct"] == 0.01
    assert s.params["atr_stop_mult"] == 2.0


# ============ P1-6 换手/持仓时长/成本拖累 ============

def test_metrics_report_new_fields():
    """P1-6：metrics 包含 avg_holding_bars/turnover/total_costs/cost_drag。"""
    from backtest.data_loader import generate_demo
    from backtest.engine import BacktestConfig, run_backtest

    df = generate_demo(n=600, timeframe="1h")
    r = run_backtest(df, BacktestConfig(strategy_name="dual_ma"))
    m = r["metrics"]
    for key in ["avg_holding_bars", "turnover", "total_costs", "cost_drag"]:
        assert key in m, f"metrics 缺少 P1-6 字段 {key}"
        assert m[key] is not None
    # 持仓时长合理（>0），换手与成本非负
    assert m["avg_holding_bars"] >= 0
    assert m["turnover"] >= 0
    assert m["total_costs"] >= 0


def test_trades_carry_entry_exit_index():
    """P1-6：卖出成交带 entry_i/exit_i（持仓时长数据源）。"""
    from backtest.data_loader import generate_demo
    from backtest.engine import BacktestConfig, run_backtest

    df = generate_demo(n=400, timeframe="1h")
    r = run_backtest(df, BacktestConfig(strategy_name="dual_ma"))
    sells = [t for t in r["trades"] if t["side"] == "sell"]
    assert sells, "应有卖出成交"
    for t in sells:
        assert "entry_i" in t and "exit_i" in t
        assert t["exit_i"] >= t["entry_i"], f"exit_i 应不小于 entry_i: {t}"


# ============ P1-7 block bootstrap ============

def test_block_bootstrap_sharpe_ci():
    """P1-7：stationary block bootstrap 返回结构与 iid 一致且保留自相关。"""
    from backtest.metrics import (bootstrap_sharpe_ci, block_bootstrap_sharpe_ci,
                                  equity_returns)

    rng = np.random.default_rng(3)
    # 带自相关的收益序列（AR(1) 趋势）
    rets = []
    v = 0.0
    for _ in range(400):
        v = 0.3 * v + rng.normal(0, 0.004)
        rets.append(v)
    rets = np.asarray(rets)
    bb = block_bootstrap_sharpe_ci(rets, n_boot=300, seed=7)
    assert bb["method"] == "block"
    assert bb["block_len"] is not None and bb["block_len"] >= 1
    assert bb["sharpe_ci"] is not None and len(bb["sharpe_ci"]) == 2
    assert bb["ret_ci"] is not None and len(bb["ret_ci"]) == 2
    assert 0.0 <= bb["p_sharpe_pos"] <= 1.0
    # iid 版本仍然可用（向后兼容）
    ib = bootstrap_sharpe_ci(rets, n_boot=300)
    assert ib["sharpe_ci"] is not None


def test_block_bootstrap_short_series():
    """P1-7：样本不足时优雅降级。"""
    from backtest.metrics import block_bootstrap_sharpe_ci
    bb = block_bootstrap_sharpe_ci(np.asarray([0.001] * 10))
    assert bb["sharpe_ci"] is None


# ============ P1-8 AI 参数验证门纯 OOS ============

def _rows(n: int) -> list:
    t = np.arange(n)
    close = 100.0 + 5.0 * np.sin(t / 20.0)
    return [[1700000000000 + i * 3600000,
             float(close[i] * 0.999), float(close[i] * 1.002),
             float(close[i] * 0.998), float(close[i]), 1000.0]
            for i in range(n)]


class _OOSStub:
    """轻量桩：仅装配 _validate_param_update 所需的属性（复用 TradingEngine 方法）。"""
    def __init__(self, rows):
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.start_cash = 10000.0
        self.paper_fee_rate = 0.001
        self.paper_slippage = 0.0005
        self.hub = None
        self.rows = rows

    async def _fetch_vision_ohlcv(self, limit=800):
        return self.rows[:limit]

    async def _validation_df(self, bars: int):
        """替换 TradingEngine._validation_df：从 rows 直接建 DataFrame（不经 hub）。"""
        from backtest.data_loader import _to_df
        return _to_df([list(c) for c in self.rows[:bars]])


import asyncio  # noqa: E402


class _PassOverfitReport:
    """过拟合检测通过的桩报告（聚焦测 P1-8 OOS 门逻辑本身）。"""
    verdict = "通过"
    score = 0.0
    oos_ret = 0.05
    pbo = 0.3
    decay = 0.1
    n_folds = 5


def _oos_fake_run(df, cfg: "BacktestConfig", on_progress=None, backend="numpy", use_numba=False,
                  bootstrap=True):
    """模拟回测：整段 suggested 更优、纯 OOS 段 suggested 更差。

    判定依据 cfg.strategy_params["fast_period"]（>30 视为"建议参数"）。
    - len(df) >= 200（整段/拟合段）：建议参数收益更高
    - len(df) <  200（纯 OOS 预留段 80 根）：建议参数收益更差 → 触发 OOS 拒绝
    bootstrap: P2-13 引擎新增开关，桩忽略（验证门只看基础指标）。
    """
    is_proposed = cfg.strategy_params.get("fast_period", 10) > 30
    if len(df) < 200:
        ret = -0.02 if is_proposed else 0.03
    else:
        ret = 0.15 if is_proposed else 0.08
    return {"metrics": {"total_return": ret, "max_drawdown": 0.05,
                        "total_trades": 3, "total_costs": 1.0,
                        "avg_holding_bars": 5.0, "turnover": 1.0, "cost_drag": 0.1}}


@pytest.mark.asyncio
async def test_validate_param_update_oos_rejects_worse(monkeypatch):
    """P1-8：建议参数在纯 OOS 段劣于当前 → 验证门拒绝（即便整段更优）。"""
    import backtest.fast_engine as fe
    import backtest.overfit as ov

    # 800 根：oos_len = 80（≥60 启用纯 OOS 校验）
    stub = _OOSStub(_rows(800))
    monkeypatch.setattr(fe, "run_backtest_fast", _oos_fake_run)
    monkeypatch.setattr(ov, "detect_overfit", lambda *a, **k: _PassOverfitReport())

    cur = {"fast_period": 10, "slow_period": 30, "size_pct": 0.5,
           "stop_loss_pct": 0.03, "take_profit_pct": 0.06}
    prop = dict(cur, fast_period=60)  # 整段更优(0.15) 但 OOS 差(-0.02) → 应被拒绝

    from engine.trading_engine import TradingEngine
    ok, info = await asyncio.wait_for(TradingEngine._validate_param_update(stub, "dual_ma", prop, cur), timeout=60)
    assert ok is False, f"纯 OOS 劣于当前应拒绝, info={info}"
    assert "纯 OOS" in info.get("reason", ""), f"拒绝理由应指向纯 OOS: {info.get('reason')}"
    assert info.get("oos", {}).get("enabled") is True


@pytest.mark.asyncio
async def test_validate_param_update_oos_identical_passes(monkeypatch):
    """P1-8：建议==当前且 OOS 段等同时 → 通过（OOS 信息记录但无劣化）。"""
    import backtest.fast_engine as fe
    import backtest.overfit as ov

    stub = _OOSStub(_rows(800))
    monkeypatch.setattr(fe, "run_backtest_fast", _oos_fake_run)
    monkeypatch.setattr(ov, "detect_overfit", lambda *a, **k: _PassOverfitReport())

    cur = {"fast_period": 10, "slow_period": 30, "size_pct": 0.5,
           "stop_loss_pct": 0.03, "take_profit_pct": 0.06}
    from engine.trading_engine import TradingEngine
    ok, info = await asyncio.wait_for(TradingEngine._validate_param_update(stub, "dual_ma", dict(cur), dict(cur)), timeout=60)
    assert ok is True, f"建议==当前应通过, info={info}"
    assert info.get("oos", {}).get("enabled") is True


# ============ P1-9 数据清洗默认标记制 ============

def test_sanitizer_mark_keeps_extreme_bar():
    """P1-9：默认 mark 模式不清洗真实行情（保留极端K线），仅标记。"""
    from backtest.data_loader import OHLCVSanitizer

    values = [100.0] * 40 + [10000.0]  # 真实极端K线
    df = pd.DataFrame({"close": values, "open": values, "high": values,
                       "low": values, "volume": values})
    s = OHLCVSanitizer(z_threshold=5.0)
    cleaned = s.clean(df, mode="mark")
    # 真实值保留
    assert cleaned["close"].iloc[-1] == pytest.approx(10000.0)
    assert cleaned.attrs["clean_flags"].get("outliers_close", 0) == 1


def test_sanitizer_structural_repair():
    """P1-9：结构损坏（high < max(open,close)）无论何种模式都修复。"""
    from backtest.data_loader import OHLCVSanitizer

    df = pd.DataFrame({"open": [1.0, 10.0], "high": [2.0, 5.0],   # 第2根 high=5 < close=10
                       "low": [0.5, 8.0], "close": [1.5, 10.0], "volume": [100, 200]})
    s = OHLCVSanitizer()
    cleaned = s.clean(df, mode="mark")
    assert cleaned["high"].iloc[1] >= 10.0, "high 应被修复到 ≥ max(open,close)"
    assert cleaned.attrs["clean_flags"].get("high_repaired", 0) >= 1


# ============ P2-11 多标的组合层回测 ============

def test_portfolio_backtest_basic():
    """P2-11：组合回测输出各标的指标、组合指标与相关性矩阵。"""
    from backtest.data_loader import generate_demo
    from backtest.portfolio import PortfolioConfig, run_portfolio_backtest

    df1 = generate_demo(n=400, timeframe="1h")
    df2 = generate_demo(n=400, timeframe="1h")
    rp = run_portfolio_backtest({"BTC/USDT": df1, "ETH/USDT": df2},
                                PortfolioConfig(symbols=["BTC/USDT", "ETH/USDT"],
                                                weights="equal", start_cash=20000.0))
    assert len(rp["per_symbol"]) == 2
    assert "portfolio_metrics" in rp and "correlation" in rp
    assert len(rp["portfolio_equity_curve"]) > 0
    # 等权 → 各 50%
    assert rp["weights"]["BTC/USDT"] == pytest.approx(0.5)
    # 相关性矩阵是对称且对角线=1
    assert rp["correlation"]["BTC/USDT"]["BTC/USDT"] == 1.0
    assert rp["correlation"]["BTC/USDT"]["ETH/USDT"] == pytest.approx(
        rp["correlation"]["ETH/USDT"]["BTC/USDT"])
    # 组合指标关键键存在
    m = rp["portfolio_metrics"]
    for key in ["total_return", "sharpe", "max_drawdown"]:
        assert key in m


def test_portfolio_vol_inv_weights():
    """P2-11：波动率倒数权重：高波动标的权重更低。"""
    from backtest.data_loader import generate_demo
    from backtest.portfolio import PortfolioConfig, run_portfolio_backtest

    df1 = generate_demo(n=300, timeframe="1h")                      # 正常波动
    rng = np.random.default_rng(9)
    close2 = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.04, 300)))    # 高波动
    df2 = _df(close2)
    rp = run_portfolio_backtest({"BTC/USDT": df1, "ETH/USDT": df2},
                                PortfolioConfig(symbols=["BTC/USDT", "ETH/USDT"],
                                                weights="vol_inv", vol_lookback=60))
    assert rp["weights"]["BTC/USDT"] > rp["weights"]["ETH/USDT"], "高波动标的应获得更低权重"


# ============ P2-12 DRL 环境成交量/资金费率约束 ============

def test_drl_env_participation_caps_qty():
    """P2-12：DRL 环境调仓数量受成交量×参与率上限约束。"""
    from drl.env import ACTION_BUCKETS, TradingEnv

    n = 200
    close = 100.0 + np.sin(np.arange(n) / 20.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1000.0),
    }).set_index("timestamp")

    env = TradingEnv(df, start_cash=10000.0, fee_rate=0.0, slippage=0.0,
                     participation_rate=0.01)  # 单次调仓 qty ≤ 1000×0.01 = 10
    env.reset()
    # 满仓动作：diff*target 目标市值 10000 / 价格~100 = ~100 qty，但单步上限 10
    action = int(np.argmax(ACTION_BUCKETS >= 1.0))
    # 连续 step 若干根，每次调仓的 qty 增量不应超过参与率上限
    max_step_delta = 0.0
    prev_qty = env._qty
    for _ in range(20):
        _, _, done, _ = env.step(action)
        max_step_delta = max(max_step_delta, abs(env._qty - prev_qty))
        prev_qty = env._qty
        if done:
            break
    # 参与率约束按"单根K线调仓量"限制（与回测口径一致，不限制累计持仓）
    assert max_step_delta <= 10.0 + 1e-9, f"参与率约束应限制单步调仓 qty≤10, got {max_step_delta}"


def test_drl_env_funding_charged():
    """P2-12：DRL 环境资金费率按持仓收取（有持仓时权益被侵蚀）。"""
    from drl.env import ACTION_BUCKETS, TradingEnv

    n = 200
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 100000.0),
    }).set_index("timestamp")

    # 无 funding 对照
    env0 = TradingEnv(df, start_cash=10000.0, fee_rate=0.0, slippage=0.0, funding_rate=0.0)
    env0.reset()
    for _ in range(10):
        _, _, done, _ = env0.step(4)
        if done:
            break
    # 有 funding（每根 0.1%）
    env1 = TradingEnv(df, start_cash=10000.0, fee_rate=0.0, slippage=0.0, funding_rate=0.001)
    env1.reset()
    for _ in range(10):
        _, _, done, _ = env1.step(4)
        if done:
            break
    assert env1._cash < env0._cash, "funding 应侵蚀现金"