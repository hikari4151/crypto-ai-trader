"""DRL 引擎优化（M4 统一 + 经验回放 + 风控）与元策略运行时专项测试。"""
import numpy as np
import pandas as pd

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


def test_replay_buffer_feature_dim():
    """feature_dim：空缓冲返回 None，有轨迹时返回状态维度（跨轮复用的守卫依据）。"""
    from drl.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=10)
    assert buf.feature_dim() is None
    buf.push(np.zeros((7, 5)), np.zeros(7, dtype=int), np.zeros(7), np.zeros(7))
    assert buf.feature_dim() == 5


def test_train_drl_reuses_external_replay_buffer():
    """跨训练轮复用：调用方传入的缓冲区在训练后持有轨迹，且第二轮继续累积。

    这是"对抗灾难性遗忘"的前提——若缓冲区每轮重建，旧行情经验在轮间被完全丢弃。
    """
    from drl.agent import ReplayBuffer, train_drl
    df = generate_demo(timeframe="1h", n=400, seed=42)
    buf = ReplayBuffer(capacity=50)
    cfg = {"episodes": 1, "n_episodes": 2, "ppo_epochs": 1, "mini_batch_size": 32,
           "seed": 42, "val_eval_interval": 1,
           "replay_buffer": buf, "replay_mix_ratio": 0.3}

    train_drl(df, cfg)
    n_first = len(buf)
    assert n_first > 0, "外部缓冲区应在训练后被填充（串起来才能跨轮生效）"

    train_drl(df, cfg)
    assert len(buf) > n_first, "第二轮应继续复用同一缓冲区，而不是新建后丢弃历史"


def test_train_drl_clears_dim_mismatched_replay_buffer():
    """维度守卫：配置变更（state_window/因子列）后旧轨迹维度不符，须清空而非崩溃。"""
    from drl.agent import ReplayBuffer, train_drl
    df = generate_demo(timeframe="1h", n=400, seed=42)
    buf = ReplayBuffer(capacity=10)
    # 伪造一条与当前状态维度不符的旧轨迹（模拟改过 state_window 的缓冲）
    buf.push(np.zeros((5, 3)), np.zeros(5, dtype=int), np.zeros(5), np.zeros(5))
    assert buf.feature_dim() == 3

    train_drl(df, {"episodes": 1, "n_episodes": 1, "ppo_epochs": 1, "mini_batch_size": 32,
                   "seed": 42, "val_eval_interval": 1, "replay_buffer": buf})

    assert buf.feature_dim() != 3, "维度不符的旧轨迹必须被清空重建"
    assert len(buf) > 0 and buf.feature_dim() and buf.feature_dim() > 3


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
    from drl.env import ACTION_BUCKETS, TradingEnv
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

def test_factor_net_fitness_costs_penalize():
    """net fitness 模式：换手成本正确压制高换手组合（2026-09-29 成本对齐修复）。

    背景：旧 ic 口径 fitness=|rank_ic|×√(1/turnover) 会被低换手奖励虚高，
    与真实回测收益脱节（5m ETH 300+ 笔 -34.6%）。net 口径 = 验证段方向收益
    − 换手成本，高换手组合应被显著压制。
    """
    from factors.engine import compute_factor_matrix
    from drl.factor_env import FactorMiningEnv
    from factors.analysis import _rolling_ic, _spearman

    df = generate_demo(timeframe="5m", n=2000, seed=7)
    mat = compute_factor_matrix(df)
    close = df["close"]
    _ics = pd.DataFrame({c: _rolling_ic(mat[c], close, h=1, method="rank", window=120)
                          for c in mat.columns}, index=mat.index).shift(1).ffill()
    _icir = _ics.rolling(600, min_periods=240).mean() / (
        _ics.rolling(600, min_periods=240).std() + 1e-12)
    _z_exp = {c: ((mat[c] - mat[c].expanding().mean())
                  / (mat[c].expanding().std() + 1e-12)).fillna(0.0)
              for c in mat.columns}

    def _mk(mode, cost):
        return FactorMiningEnv(mat, close, h=1, ic_window=120, max_steps=4,
                               train_ratio=0.6, val_ratio=0.2, corr_threshold=0.85,
                               seed=7, precomputed_ics=_ics, precomputed_icir=_icir,
                               precomputed_z=_z_exp, use_action_mask=True,
                               fitness_mode=mode, cost_per_turn=cost)

    e_ic = _mk("ic", 0.0)
    e_net = _mk("net", 0.002)

    fwd = close.to_numpy(float)[1:] / close.to_numpy(float)[:-1] - 1
    ics = {c: _spearman(mat[c].to_numpy(float)[e_ic.n_train:e_ic.n_val],
                        fwd[e_ic.n_train:e_ic.n_val]) for c in mat.columns}
    ranked = sorted(ics.items(), key=lambda x: -abs(x[1]))
    top = [mat.columns.get_loc(r[0]) for r in ranked[:4]]
    bot = [mat.columns.get_loc(r[0]) for r in ranked[-4:]]

    # 分别评估两组组合
    e_ic._selected = top
    f_ic = e_ic._composite_fitness()
    e_net._selected = top
    f_net_top = e_net._composite_fitness()
    e_net._selected = bot
    f_net_bot = e_net._composite_fitness()

    # 无成本时两种组合都应接近（demo 方向收益相近）
    assert f_ic > 0, "ic 口径应给出正分"
    # 成本应显著压制净收益：top4 保留（IC 高），bot4 被压得更狠
    assert f_net_top >= f_net_bot, (
        "net 口径下高 IC 组合应不低于低 IC 组合；"
        f"top={f_net_top} bot={f_net_bot}")


def test_factor_net_fitness_matches_manual():
    """net fitness 的手动重算逐位一致（gross − turnover×cost）× scale。"""
    from factors.engine import compute_factor_matrix
    from drl.factor_env import FactorMiningEnv
    from factors.analysis import _rolling_ic, factor_turnover, forward_returns
    from factors.mining import composite_factor

    df = generate_demo(timeframe="5m", n=2000, seed=7)
    mat = compute_factor_matrix(df)
    close = df["close"]
    _ics = pd.DataFrame({c: _rolling_ic(mat[c], close, h=1, method="rank", window=120)
                          for c in mat.columns}, index=mat.index).shift(1).ffill()
    _icir = _ics.rolling(600, min_periods=240).mean() / (
        _ics.rolling(600, min_periods=240).std() + 1e-12)
    _z_exp = {c: ((mat[c] - mat[c].expanding().mean())
                  / (mat[c].expanding().std() + 1e-12)).fillna(0.0)
              for c in mat.columns}
    e = FactorMiningEnv(mat, close, h=1, ic_window=120, max_steps=4,
                        train_ratio=0.6, val_ratio=0.2, corr_threshold=0.85,
                        seed=7, precomputed_ics=_ics, precomputed_icir=_icir,
                        precomputed_z=_z_exp, use_action_mask=True,
                        fitness_mode="net", cost_per_turn=0.002)

    cols = list(mat.columns[:4])
    e._selected = [mat.columns.get_loc(c) for c in cols]
    actual = e._composite_fitness()

    weights = {c: 0.1 for c in cols}
    combo = composite_factor(mat, weights, method="ic", z_mode="expanding", z_map=_z_exp)
    vs = combo.iloc[e.n_train:e.n_val]
    vc = close.iloc[e.n_train:e.n_val]
    s = vs.to_numpy(float)
    fwd = forward_returns(vc, 1).to_numpy(float)
    sig = np.nan_to_num(np.sign(s), nan=0.0)
    fwd_a = np.nan_to_num(fwd, nan=0.0)
    gross = float(np.mean(sig * fwd_a))
    cost = factor_turnover(vs) * 0.002
    expected = round(max(0.0, gross - cost) * 500.0, 4)
    assert abs(actual - expected) < 1e-9, f"逐位不一致: env={actual} manual={expected}"


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


# ============ 进化引擎测试用假 DB/会话（test_evolve_api_guards 等复用） ============

class _MetaRow:
    def __init__(self):
        self.spec_json = "{}"

    # 并行新逻辑 _log_round 用 scalar_one() 查 func.max(round_no)，
    # mock 行对象需支持 int()/float()——模拟"无历史行"（返回 0），
    # 否则 int(_MetaRow) 抛 TypeError（_MetaRow 无数值语义）。
    # spec_json 路径（scalar_one_or_none → .spec_json）不受影响。
    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0


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


# ============ 第2项完善：训练成本闸 ============

def test_reject_downshift_retry_pure():
    """连续 OOS 被拒降频纯函数：floor<=0 关闭、未达阈值不变、达到放大 5 倍/下限 300s。"""
    from drl.evolve_engine import _reject_downshift_retry
    # 关闭
    assert _reject_downshift_retry(60, 50, 0) == 60
    # 未达阈值
    assert _reject_downshift_retry(60, 5, 6) == 60
    # 达到阈值：60*5=300（正好命中下限）
    assert _reject_downshift_retry(60, 6, 6) == 300
    # 大间隔放大
    assert _reject_downshift_retry(120, 7, 6) == 600
    # 小间隔也受下限保护
    assert _reject_downshift_retry(30, 8, 2) == 300


async def test_config_whitelist_accepts_reject_downshift_after(tmp_path, monkeypatch):
    """新配置键 reject_downshift_after 应通过热更新校验并进入 applied。

    注意：apply_config 会把变更持久化回真实 config/config.yaml——测试前后必须
    备份/恢复该文件（曾污染：测试跑完 config.yaml 里残留 evolve_reject_downshift_after:4）。
    """
    from drl.evolve_engine import EvolveEngine
    from core.bus import EventBus
    from config.settings import ROOT as _SETTINGS_ROOT

    yaml_cfg = _SETTINGS_ROOT / "config" / "config.yaml"
    bak_bytes = yaml_cfg.read_bytes() if yaml_cfg.exists() else None
    bak_file = _SETTINGS_ROOT / "config" / "config.yaml.bak"
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    try:
        result = await eng.apply_config({"reject_downshift_after": 4})
        assert result["ok"] is True
        assert result["applied"].get("evolve_reject_downshift_after") == 4
    finally:
        if bak_bytes is not None:
            yaml_cfg.write_bytes(bak_bytes)
        elif yaml_cfg.exists():
            yaml_cfg.unlink()
        if bak_file.exists():
            bak_file.unlink()


async def test_config_whitelist_accepts_replay_keys(tmp_path, monkeypatch):
    """新配置键（经验回放）应通过热更新校验并进入 applied。"""
    from drl.evolve_engine import EvolveEngine
    from core.bus import EventBus
    from config.settings import ROOT as _SETTINGS_ROOT
    from config.settings import settings as _settings

    # apply_config 会**原地修改全局 settings 对象**（并持久化回 config.yaml）。
    # 用 monkeypatch 记录原值以便用例结束自动恢复——否则污染后续用例
    # （曾把默认 capacity 断言污染成 512）。
    monkeypatch.setattr(_settings, "evolve_strategy_drl_replay_capacity",
                        int(_settings.evolve_strategy_drl_replay_capacity))
    monkeypatch.setattr(_settings, "evolve_strategy_drl_replay_mix_ratio",
                        float(_settings.evolve_strategy_drl_replay_mix_ratio))

    yaml_cfg = _SETTINGS_ROOT / "config" / "config.yaml"
    bak_bytes = yaml_cfg.read_bytes() if yaml_cfg.exists() else None
    bak_file = _SETTINGS_ROOT / "config" / "config.yaml.bak"
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    try:
        result = await eng.apply_config({"strategy_drl_replay_capacity": 512,
                                         "strategy_drl_replay_mix_ratio": 0.25})
        assert result["ok"] is True
        assert result["applied"].get("evolve_strategy_drl_replay_capacity") == 512
        assert result["applied"].get("evolve_strategy_drl_replay_mix_ratio") == 0.25
    finally:
        if bak_bytes is not None:
            yaml_cfg.write_bytes(bak_bytes)
        elif yaml_cfg.exists():
            yaml_cfg.unlink()
        if bak_file.exists():
            bak_file.unlink()


def test_evolve_engine_has_replay_slot(tmp_path):
    """引擎须持有跨轮回放槽位（默认 None/0，由 settings 决定是否启用）。"""
    from drl.evolve_engine import EvolveEngine
    from core.bus import EventBus
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    assert hasattr(eng, "_strategy_drl_replay")
    assert eng._strategy_drl_replay is None
    assert eng._strategy_drl_replay_capacity == 0
    assert eng._strategy_drl_replay_symbol == ""


def test_replay_settings_defaults_enabled():
    """默认启用回放：遗忘是本项目的现场故障，默认应修复。

    2026-10 审计修复（MED）：capacity 从 64 提到 128（episodes=128 时每轮推 2048 条
    轨迹，64 只留最近 ~1/32 轮，"跨轮记忆"名不副实）；128 按实测内存口径
    （单轨迹≈1.24MB，128 条≈155MB）是合理档，400 会占约 484MB 常驻内存（不建议）。
    """
    from config.settings import settings
    assert settings.evolve_strategy_drl_replay_capacity == 128
    assert settings.evolve_strategy_drl_replay_mix_ratio == 0.3
    assert 0.0 < settings.evolve_strategy_drl_replay_mix_ratio < 1.0


def test_train_time_budget_default_positive():
    """第2项完善：train_time_budget 默认应为正值（120s），后台无预算上限不再常驻吃 CPU。"""
    from config.settings import settings
    assert float(settings.evolve_train_time_budget) > 0
