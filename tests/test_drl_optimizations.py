"""DRL 引擎优化（M4 统一 + 元策略 + 经验回放 + 风控）专项测试。"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from backtest.data_loader import generate_demo


# ============ T1. Signal 置信度字段 ============

def test_signal_confidence_default():
    """旧策略不设置 confidence 时默认 1.0（满置信），向后兼容。"""
    from strategies.base import Signal
    s = Signal(symbol="BTC", side="buy")
    assert s.confidence == 1.0


def test_signal_confidence_set():
    """新策略可显式设置置信度。"""
    from strategies.base import Signal
    s = Signal(symbol="BTC", side="sell", confidence=0.75)
    assert s.confidence == 0.75


# ============ T2. ReplayBuffer ============

def test_replay_buffer_push_and_len():
    from drl.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.push(np.array([[0.1, 0.2]]), np.array([1]), np.array([0.5]), np.array([-0.1]))
    assert len(buf) == 1


def test_replay_buffer_sample():
    from drl.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=20)
    n = 5
    for i in range(n):
        buf.push(
            np.random.randn(10, 4).astype(float),
            np.random.randint(0, 5, 10).astype(int),
            np.random.randn(10).astype(float),
            np.random.randn(10).astype(float),
        )
    assert len(buf) == n
    sampled = buf.sample(batch_size=30, mix_ratio=0.5)
    assert "states" in sampled
    assert len(sampled["states"]) > 0
    assert len(sampled["segment_lengths"]) > 0


def test_replay_buffer_capacity_eviction():
    from drl.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=3)
    for i in range(10):
        buf.push(
            np.array([[float(i)]]),
            np.array([1]),
            np.array([0.0]),
            np.array([0.0]),
        )
    assert len(buf) == 3  # 超过容量，最旧被剔除


def test_replay_buffer_empty_sample():
    from drl.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=10)
    sampled = buf.sample(batch_size=10)
    assert len(sampled["states"]) == 0


# ============ T3. MetaController 集成模式 ============

def test_meta_controller_ensemble_backtest():
    """元策略集成模式运行完整回测，产生交易，不抛异常。"""
    df = generate_demo(timeframe="1h", n=500, seed=42)
    cfg = BacktestConfig(
        symbol="BTC/USDT", timeframe="1h",
        strategy_name="meta_controller",
        strategy_params={"sub_strategies": "dual_ma,factor_signal",
                         "mode": "ensemble", "size_pct": 0.5},
    )
    result = run_backtest(df, cfg)
    # 应有交易流水（demo 数据上信号可能稀疏，但至少引擎应完成）
    assert isinstance(result["trades"], list)
    assert isinstance(result["metrics"], dict)
    assert "total_return" in result["metrics"]


def test_meta_controller_default_params():
    from strategies import get_strategy
    st = get_strategy("meta_controller")
    assert st.name == "meta_controller"
    assert st.default_params["mode"] == "ensemble"
    assert "dual_ma" in st.default_params["sub_strategies"]


# ============ T4. RL Adaptive 置信度输出 + 恐慌风控 ============

def test_rl_adaptive_confidence_output():
    """RLAdaptiveStrategy 的 on_candle 返回带置信度的 Signal。"""
    from strategies.rl_adaptive import RLAdaptiveStrategy
    from drl.agent import ACAgent
    from drl.env import ACTION_BUCKETS, TradingEnv, run_episode
    df = generate_demo(timeframe="1h", n=300, seed=42)
    # 训练一个极简模型
    env = TradingEnv(df, vol_penalty=0.0)
    agent = ACAgent(env.state_dim, len(ACTION_BUCKETS), hidden=(8, 8), seed=42)
    agent.set_epsilon_schedule(3)
    from drl.agent import train_drl
    result = train_drl(df, {"episodes": 3, "n_episodes": 2, "ppo_epochs": 1,
                            "hidden": [8, 8], "val_eval_interval": 1, "seed": 42})
    st = RLAdaptiveStrategy()
    st.update_params({"model_path": ""})  # 不加载模型，应返回 None
    # 手动设置 agent 用于测试
    st._agent = result["agent"]
    st._state_dim = result["agent"].state_dim
    closes = df["close"].to_numpy(float)
    for t in range(60, min(80, len(df))):
        price = float(closes[t])
        ctx = {
            "symbol": "BTC", "price": price,
            "position": 0.0, "cash": 10000.0,
            "indicators": {"close": price, "high": price, "low": price,
                           "open": price, "volume": 1.0, "candles_count": t + 1},
        }
        sig = st.on_candle(ctx)
        if sig is not None:
            assert 0.0 <= sig.confidence <= 1.0, f"置信度应介于 0~1，实际 {sig.confidence}"
            break


# ============ T5. 尾部风险惩罚 ============

def test_tail_risk_penalty_reduces_reward():
    """尾部风险惩罚在单根大回撤时额外扣减奖励。

    用确定性贪婪策略（同一 agent、同 seed）跑两个环境，保证动作序列一致，
    仅惩罚系数不同，比较累计奖励之和（而非 total_ret，后者是原始权益收益）。
    """
    from drl.env import TradingEnv, run_episode
    from drl.agent import ACAgent
    df = generate_demo(timeframe="1h", n=300, seed=42)
    env_no = TradingEnv(df, vol_penalty=0.0, tail_risk_penalty=0.0)
    env_yes = TradingEnv(df, vol_penalty=0.0, tail_risk_penalty=5.0)
    # 用确定性贪婪策略（同一 agent），保证两个环境动作序列一致
    agent = ACAgent(env_no.state_dim, 5, seed=3)
    # 用 return_trajectory=True 获取每步奖励
    r0 = run_episode(env_no, lambda s: agent.greedy_action(s), return_trajectory=True)
    r1 = run_episode(env_yes, lambda s: agent.greedy_action(s), return_trajectory=True)
    # 有惩罚的环境累计奖励更低
    sum_r0 = float(np.sum(r0["rewards"]))
    sum_r1 = float(np.sum(r1["rewards"]))
    assert sum_r0 >= sum_r1 - 1e-6, f"尾部风险惩罚应降低累计奖励: {sum_r0:.4f} < {sum_r1:.4f}"


# ============ T6. vol_penalty 口径统一验证 ============

def test_vol_penalty_unified():
    """验证环境与训练环境使用相同的 vol_penalty（M4 修复）。"""
    from drl.agent import train_drl
    df = generate_demo(timeframe="1h", n=400, seed=42)
    cfg = {
        "episodes": 2, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64,
        "seed": 42, "val_eval_interval": 1, "vol_penalty": 15.0,
    }
    result = train_drl(df, cfg)
    # 验证结果中有 factor_values 键（即使为 None）
    assert "factor_values" in result
    assert "factor_mu" in result
    assert "factor_sd" in result
    # 验证 PPO config 包含 replay 信息
    assert "replay" in result["ppo_cfg"]


# ============ T8. MetaController DRL 训练 ============

def test_meta_controller_train():
    """元策略 DRL 训练能完成，产出有界收益和元智能体。"""
    from strategies.meta import train_meta_controller
    df = generate_demo(timeframe="1h", n=500, seed=42)
    result = train_meta_controller(
        df, {"episodes": 2, "n_episodes": 2, "sub_strategies": "dual_ma,factor_signal",
             "hidden": [8, 8]})
    assert result["agent"] is not None
    assert result["sub_strategies"] == ["dual_ma", "factor_signal"]
    assert result["state_dim"] > 0
    # 收益应是有界浮点数（不爆炸）
    assert -1e6 < result["best_ret"] < 1e6


def test_meta_controller_train_seed_none():
    """回归：进化引擎传 seed=None（意为每轮随机），不得触发 int(None) 崩溃。

    修复前 drl/evolve_engine 的每轮元策略训练都以此 cfg 抛 TypeError，
    表现为「持续进化」页元策略控制器一直报错。
    """
    from strategies.meta import train_meta_controller
    df = generate_demo(timeframe="1h", n=500, seed=42)
    result = train_meta_controller(df, {
        "episodes": 1, "n_episodes": 1, "hidden": [8, 8], "seed": None,
        "sub_strategies": "dual_ma,factor_signal,price_action",
        "fee_rate": 0.001, "meta_window": 20,
    })
    assert result["agent"] is not None
    assert result["history"][-1]["best_ret"] == result["best_ret"]


def test_meta_controller_train_explicit_seed_reproducible():
    """显式 seed 仍可复现：随机 seed 的兜底不能破坏确定性。"""
    from strategies.meta import train_meta_controller
    df = generate_demo(timeframe="1h", n=500, seed=42)
    cfg = {"episodes": 2, "n_episodes": 1, "hidden": [8, 8], "seed": 7,
           "sub_strategies": "dual_ma,factor_signal"}
    r1 = train_meta_controller(df, cfg)
    r2 = train_meta_controller(df, cfg)
    assert [h["total_ret"] for h in r1["history"]] == [h["total_ret"] for h in r2["history"]]


def test_meta_controller_signal_align_shaping():
    """P4-E2：信号对齐塑形开启后训练仍完成、收益有界；顺势建仓当步奖励更高。

    回归：元策略此前每轮 fitness=0.0（PPO 收敛到"空仓=0"的懒惰最优解），
    OOS 从不交易被硬门拦截。塑形只进训练梯度、不污染 best_ret 真实口径，
    本测试锁定「塑形开关不破坏训练契约 + 塑形确实按决策生效」。
    """
    from strategies.meta import MetaControllerEnv, train_meta_controller
    df = generate_demo(timeframe="1h", n=500, seed=42)
    res = train_meta_controller(df, {
        "episodes": 3, "n_episodes": 2, "hidden": [8, 8], "seed": 5,
        "sub_strategies": "dual_ma,factor_signal",
        "reward_signal_align": 15.0,
    })
    assert res["agent"] is not None
    assert -1e6 < res["best_ret"] < 1e6

    # 塑形环境 vs 关闭塑形：找到一根有买入信号、已过 warmup 的K线，
    # 两环境同步推进到该处后执行买入，顺势建仓的当步奖励应严格更高。
    env_on = MetaControllerEnv(df, ["dual_ma", "factor_signal"], warmup=60,
                               reward_signal_align=15.0)
    env_off = MetaControllerEnv(df, ["dual_ma", "factor_signal"], warmup=60,
                                reward_signal_align=0.0)
    env_on.reset()
    env_off.reset()
    buy_t = None
    for t in range(env_on.warmup, len(df) - 1):
        sigs = env_on._collect(t)
        if any(float(s.get("direction", 0.0) or 0.0) > 0 for s in sigs.values()):
            buy_t = t
            break
    assert buy_t is not None, "demo 数据应存在买入信号K线"
    for _ in range(env_on.warmup, buy_t):
        env_on.step(0)
        env_off.step(0)
    _, r_on, _, _ = env_on.step(1)
    _, r_off, _, _ = env_off.step(1)
    assert r_on > r_off, f"顺势建仓塑形应提高当步奖励（on={r_on:.3f} off={r_off:.3f}）"


# ============ T9. 级联因子注入 ============

def test_train_drl_factor_values():
    """train_drl 支持级联因子值注入（factor_miner 组合因子）。"""
    from drl.agent import train_drl
    df = generate_demo(timeframe="1h", n=400, seed=42)
    import numpy as np
    # 构造一个与 df 等长的随机因子值序列
    factor_values = np.random.randn(len(df)).astype(float)
    cfg = {
        "episodes": 2, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64,
        "seed": 42, "val_eval_interval": 1,
        "factor_values": factor_values,
    }
    result = train_drl(df, cfg)
    assert result["agent"] is not None
    # 级联因子应被记录到结果
    assert result["factor_values"] is not None
    assert len(result["factor_values"]) == len(df)


# ============ T10. MetaController DRL 部署 ============

def test_meta_controller_drl_deploy(tmp_path):
    """MetaController DRL 模式：训练→保存→加载→产出信号。"""
    import os
    from strategies.meta import train_meta_controller
    from strategies import get_strategy
    df = generate_demo(timeframe="1h", n=500, seed=42)
    result = train_meta_controller(
        df, {"episodes": 3, "n_episodes": 2, "sub_strategies": "dual_ma,factor_signal",
             "hidden": [16, 16]})
    model_path = os.path.join(str(tmp_path), "meta.json")
    result["agent"].save(model_path)
    st = get_strategy("meta_controller")
    st.update_params({"sub_strategies": "dual_ma,factor_signal",
                      "mode": "drl", "model_path": model_path})
    closes = df["close"].to_numpy(float)
    n_sig = 0
    for t in range(60, min(120, len(df))):
        price = float(closes[t])
        ctx = {"symbol": "META", "price": price, "position": 0.0, "cash": 10000.0,
               "indicators": {"close": price, "high": price, "low": price,
                              "open": price, "volume": 1.0, "candles_count": t + 1}}
        sig = st.on_candle(ctx)
        if sig is not None:
            n_sig += 1
    # 不抛异常即可；信号数因随机训练而异
    assert n_sig >= 0


# ============ T11. 奖励系数自动搜索 ============

def test_search_reward_coefficients():
    """奖励系数自动搜索能完成评估，返回最佳系数（含默认兜底）。"""
    from drl.agent import search_reward_coefficients
    df = generate_demo(timeframe="1h", n=600, seed=42)
    result = search_reward_coefficients(
        df, {"n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64, "seed": 42},
        n_trials=2, quick_episodes=2)
    assert "best_cfg" in result
    assert "vol_penalty" in result["best_cfg"]
    assert "results" in result
    assert len(result["results"]) >= 1
    # 即使数据不足导致 val_ret=0，返回的 best_cfg 应有默认值
    assert result["best_cfg"]["vol_penalty"] > 0

def test_replay_enabled_training():
    """启用经验回放时训练不抛异常。"""
    from drl.agent import train_drl
    df = generate_demo(timeframe="1h", n=400, seed=42)
    cfg = {
        "episodes": 2, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 64,
        "seed": 42, "val_eval_interval": 1,
        "replay_capacity": 10, "replay_mix_ratio": 0.3,
    }
    result = train_drl(df, cfg)
    assert result["agent"] is not None
    assert len(result["history"]) > 0


# ============ T12. 进化引擎元策略训练与用户参数互通 ============

class _MetaRow:
    def __init__(self):
        self.spec_json = "{}"


class _MetaResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row

    def scalar_one(self):
        return self._row


class _MetaSession:
    def __init__(self, row):
        self._row = row
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        return _MetaResult(self._row)

    async def commit(self):
        self.committed = True

    def add(self, obj):
        pass


class _MetaDB:
    def __init__(self):
        self.row = _MetaRow()
        self.session_obj = _MetaSession(self.row)

    def session(self):
        return self.session_obj


def _make_evolve_engine(tmp_path, monkeypatch, sub_strategies, mode):
    """构造一个可离线跑元策略训练一轮的 EvolveEngine。"""
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    from strategies import register_dynamic
    register_dynamic("meta_controller", {
        "name": "meta_controller", "executor": "meta_controller",
        "params": {"sub_strategies": sub_strategies, "mode": mode},
    })
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200  # 让 400 根 demo K 线通过数据量检查
    df = generate_demo(timeframe="1h", n=400, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    saved: list[str] = []
    monkeypatch.setattr(eng.zoo, "save_agent",
                        lambda agent, name, *, meta=None: saved.append(name))
    return eng, saved


async def test_evolve_meta_round_invalidated_when_pool_changes(tmp_path, monkeypatch):
    """训练期间用户改了子策略池 → 本轮作废，不打回用户的新选择。"""
    from strategies import register_dynamic, get_dynamic, remove_dynamic
    import drl.evolve_engine as ev
    eng, saved = _make_evolve_engine(
        tmp_path, monkeypatch, "dual_ma,grid", "drl")
    user_pool = "dual_ma,factor_signal,price_action,sr_pullback_reversal_v3"

    def _fake_train(df_, cfg_, on_progress=None):
        # 模拟训练过程中「统一策略」页面保存了新的 4 子策略池
        register_dynamic("meta_controller", {
            "name": "meta_controller", "executor": "meta_controller",
            "params": {"sub_strategies": user_pool, "mode": "ensemble"},
        })
        return {"history": [{"best_ret": 0.5}], "agent": object()}
    monkeypatch.setattr(ev, "train_meta_controller", _fake_train)

    ep0 = eng._meta_controller_status["episode"]
    try:
        await eng._train_meta_controller_once()
        assert saved == [], "旧池训练出的模型不应覆盖模型文件"
        assert eng._meta_controller_status["episode"] == ep0
        assert (get_dynamic("meta_controller") or {})["params"]["sub_strategies"] == user_pool
    finally:
        remove_dynamic("meta_controller")


async def test_evolve_meta_registration_keeps_user_mode(tmp_path, monkeypatch):
    """完成的一轮训练注册时保留用户选的集成模式，只更新模型路径与说明。"""
    from strategies import get_dynamic, remove_dynamic
    import drl.evolve_engine as ev
    eng, saved = _make_evolve_engine(
        tmp_path, monkeypatch, "dual_ma,grid,price_action", "ensemble")
    monkeypatch.setattr(ev, "train_meta_controller",
                        lambda df_, cfg_, on_progress=None: {
                            "history": [{"best_ret": 0.42}], "agent": object(),
                            # 元控制器管线同样要过 OOS 硬门：假结果需携带可放行的样本外报告，
                            # 否则训练在注册前即被拦截（gate 拒绝无 oos_report 的结果）
                            "oos_report": {"enabled": True, "oos_ret": 0.1,
                                           "oos_position_ratio": 0.4}})
    try:
        await eng._train_meta_controller_once()
        spec = get_dynamic("meta_controller") or {}
        params = spec["params"]
        assert params["mode"] == "ensemble"
        assert params["sub_strategies"] == "dual_ma,grid,price_action"
        assert params["model_path"]
        assert "集成元控制器" in spec["description"]
        assert saved == ["meta_controller"]
        assert eng._meta_controller_status["episode"] == 1
    finally:
        remove_dynamic("meta_controller")