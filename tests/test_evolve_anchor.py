"""进化锚点（best_fitness）可比性与锁死自检。

持续进化的回退保护拿"历史最佳 fitness"当锚点。锚点一旦是 demo（合成数据）训练
出来的高分模型，真实交易所数据就永远够不着：实测 72/72 轮全部 rollback，进化循环
事实上停摆，而前端只显示"一直在训练"，看不出任何异常。

这里锁三件事：
1. 锚点必须自带口径信息（data_source / oos_ret / 建立时间），否则无法判断能不能比；
2. 不可比（demo 锚点 vs 真实数据）时不得回退——让真实模型重建锚点；
3. 锁死要能看见（status 暴露）并能一键解开（reset_anchor，归档式、不删权重）。
"""
import json

import pytest

import drl.evolve_engine as ev
from backtest.data_loader import generate_demo
from config.settings import settings
from drl.agent import ACAgent
from drl.evolve_engine import EvolveEngine
from drl.model_zoo import ModelZoo
from tests.test_drl_optimizations import _MetaDB


# ---------------- ModelZoo：锚点元数据 ----------------

@pytest.fixture
def zoo(tmp_path):
    return ModelZoo(tmp_path / "models")


def _agent(state_dim=2, n_actions=2):
    return ACAgent(state_dim=state_dim, n_actions=n_actions, hidden=(2, 2), seed=7)


def test_best_info_reads_the_deployed_anchor_not_the_version_list(zoo):
    """save_agent_best（回退路径）不递增版本，锚点信息必须从 best 文件本身读，
    否则会读到"最近一轮被拦截模型"的 meta（实测 meta_controller 就是这样错配的：
    best_fitness=0.016223 却指向 fitness=0.0 的 v289）。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 9.0, "data_source": "exchange",
                                        "oos_ret": 0.12})
    zoo.save_agent(_agent(), "m", meta={"fitness": 1.0, "data_source": "exchange"},
                   is_best=False)
    zoo.save_agent_best(_agent(), "m", meta={"fitness": 9.0, "rollback": True})

    info = zoo.best_info("m")
    assert info["fitness"] == 9.0
    assert info["data_source"] == "exchange"
    assert info["oos_ret"] == 0.12
    assert info["comparable"] is True
    # 回退只是重新部署同一份权重：版本号必须还指向锚点本身（v1），
    # 不能跳到"最近存档的那一轮"（v2，fitness 完全不同）
    assert zoo.best_version("m") == 1
    assert info["rolled_back_at"]


def test_best_info_marks_demo_anchor_as_incomparable(zoo):
    """demo 高分锚点与真实数据不同口径，必须标记为不可比并给出原因。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 3.916665, "data_source": "demo"})
    info = zoo.best_info("m")
    assert info["comparable"] is False
    assert "demo" in info["reason"] or "合成" in info["reason"]


def test_best_info_recovers_provenance_stripped_by_legacy_rollback(zoo):
    """现场 data/models/strategy_drl 就是这个形状：老代码的回退写盘只带
    {fitness, timestamp, rollback}，把 best 文件的 data_source 抹掉了，但 versions
    里还记着 demo。若因此判成"来源未知→按可比处理"，新代码在真实仓库上依旧锁死。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 3.916665, "data_source": "demo"})
    data = zoo._read_json_gz(zoo._best_path("m"))
    data["_meta"] = {"fitness": 3.916665, "timestamp": 123.0, "rollback": True}
    zoo._write_json_gz(zoo._best_path("m"), data)

    info = zoo.best_info("m")
    assert info["data_source"] == "demo"
    assert info["comparable"] is False
    assert info["source_unknown"] is False
    assert "demo" in info["reason"] or "合成" in info["reason"]
    # 只补缺失的来源字段：fitness/timestamp 仍以 best 文件自身为准
    assert info["fitness"] == 3.916665
    assert info["timestamp"] == 123.0


def test_reset_keeps_deployed_weights_provenance_visible(zoo):
    """重置只解除回退基线，在线权重仍是那份 demo 模型：横幅必须还能说出 "demo"。
    老仓库的 best 文件 _meta 里没有 data_source 且重置后 best_version 归零，
    只能靠 best 文件自带的 _version 反查版本存档。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 3.916665, "data_source": "demo"})
    data = zoo._read_json_gz(zoo._best_path("m"))
    data["_meta"] = {"fitness": 3.916665, "timestamp": 123.0, "rollback": True}
    zoo._write_json_gz(zoo._best_path("m"), data)

    zoo.reset_anchor("m")

    info = zoo.best_info("m")
    assert info["fitness"] is None
    assert info["data_source"] == "demo"


def test_best_info_on_empty_zoo(zoo):
    info = zoo.best_info("nothing_here")
    assert info["fitness"] is None
    assert info["comparable"] is False
    assert info["data_source"] == ""


def test_restore_deployment_preserves_all_deployment_metadata(zoo):
    agent = _agent()
    zoo.save_agent(
        agent, "strategy_drl",
        meta={
            "fitness": 0.42, "data_source": "exchange", "oos_ret": 0.08,
            "symbol": "BTC/USDT", "timeframe": "1h", "state_window": 3,
            "factor_expression": "rsi(close, 14)", "factor_mu": 1.25,
            "factor_sd": 0.5, "min_trade_zone": 0.05,
            "pine_code": "//@version=5\\nstrategy('x')",
        },
    )
    zoo.save_agent(agent, "strategy_drl", meta={"fitness": 0.10}, is_best=False)

    result = zoo.restore_deployment("strategy_drl", 1, reason="test")

    assert result["version"] == 1
    data = zoo._read_json_gz(zoo._best_path("strategy_drl"))
    assert data["_meta"]["factor_expression"] == "rsi(close, 14)"
    assert data["_meta"]["factor_mu"] == 1.25
    assert data["_meta"]["pine_code"].startswith("//@version")
    flat = json.loads(zoo._flat_path("strategy_drl").read_text(encoding="utf-8"))
    assert flat["factor_expression"] == "rsi(close, 14)"
    assert flat["min_trade_zone"] == 0.05


def test_restore_deployment_failure_keeps_previous_deployment(zoo, monkeypatch):
    agent = _agent()
    zoo.save_agent(agent, "m", meta={"fitness": 0.8, "data_source": "exchange"})
    zoo.save_agent(agent, "m", meta={"fitness": 0.9, "data_source": "exchange"})
    best_before = zoo._best_path("m").read_bytes()
    flat_before = zoo._flat_path("m").read_bytes()
    meta_before = zoo._meta_path("m").read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("injected flat write failure")

    monkeypatch.setattr(zoo, "_write_flat_json", fail_write)
    with pytest.raises(Exception):
        zoo.restore_deployment("m", 1, reason="test")

    assert zoo._best_path("m").read_bytes() == best_before
    assert zoo._flat_path("m").read_bytes() == flat_before
    assert zoo._meta_path("m").read_bytes() == meta_before


def test_max_versions_one_keeps_best_version_file(zoo):
    limited = ModelZoo(zoo.models_dir, max_versions=1)
    for fitness in (1.0, 2.0, 3.0):
        limited.save_agent(_agent(), "m", meta={"fitness": fitness})
        best = limited.best_version("m")
        assert best > 0
        assert limited._version_path("m", best).exists()


def test_reset_anchor_archives_and_clears_without_touching_weights(zoo):
    """解锁必须可回滚：meta.json 归档一份带时间戳的副本，权重文件原样保留
    （删权重会让实盘/纸面在下一轮接受前无模型可用）。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 3.916665, "data_source": "demo"})
    best_before = zoo._best_path("m").read_bytes()
    flat_before = zoo._flat_path("m").read_bytes()

    result = zoo.reset_anchor("m", reason="demo 锚点锁死迭代")

    assert result["cleared"]["fitness"] == 3.916665
    assert zoo.best_fitness("m") is None
    assert zoo.best_version("m") == 0
    assert zoo.best_info("m")["comparable"] is False
    # 锚点清了，但 best 文件里的权重仍在线部署着：来源要照实报，前端得知道"现在跑的是 demo 权重"
    assert zoo.best_info("m")["data_source"] == "demo"
    assert zoo._best_path("m").read_bytes() == best_before, "权重文件不得被动"
    assert zoo._flat_path("m").read_bytes() == flat_before
    meta = json.loads(zoo._meta_path("m").read_text(encoding="utf-8"))
    assert meta["anchor_reset"]["reason"] == "demo 锚点锁死迭代"
    assert meta["versions"], "版本历史保留（重置锚点不等于清空研究记录）"
    archives = list((zoo._model_dir("m") / "anchor_reset").glob("*_meta.json"))
    assert archives, "重置前的 meta.json 必须留档"
    assert json.loads(archives[0].read_text(encoding="utf-8"))["best_fitness"] == 3.916665


def test_reset_anchor_loads_next_round_from_scratch(zoo):
    """重置后 load_agent 仍返回旧权重（用于热启动），但不再有回退锚点。"""
    zoo.save_agent(_agent(), "m", meta={"fitness": 3.916665, "data_source": "demo"})
    zoo.reset_anchor("m")
    assert zoo.load_agent("m") is not None
    assert zoo.best_info("m")["fitness"] is None


# ---------------- 进化循环：不可比锚点不得锁死迭代 ----------------

def _result(best_ret=0.30, oos_ret=0.05):
    return {
        "agent": _agent(),
        "history": [{"total_ret": -3.0, "best_ret": best_ret}],
        "best_ret": best_ret,
        "deployment_blocked": False,
        "oos_report": {"enabled": True, "oos_ret": oos_ret, "train_ret": best_ret,
                       "hard_rejected": False, "reason": "",
                       "oos_position_ratio": 0.6},
        "pine_code": "", "factor_expression": "", "factor_mu": 0.0, "factor_sd": 1.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from core.bus import EventBus
    eng = EvolveEngine(_MetaDB(), EventBus(), tmp_path / "models")
    eng._rolling_window = 200
    df = generate_demo(timeframe="1h", n=400, seed=42)

    async def _fake_fetch(symbol: str = ""):
        return df
    monkeypatch.setattr(eng, "_fetch_latest_data", _fake_fetch)
    holder = {"result": None}
    monkeypatch.setattr(ev, "train_drl",
                        lambda df_, cfg_, on_progress=None: holder["result"])
    registered: list = []
    monkeypatch.setattr(ev, "register_dynamic", lambda name, spec: registered.append(name))
    monkeypatch.setattr(eng, "_register_rl_evolve", lambda: _noop_register(registered))

    async def _no_cross(agent, train_symbol, state_window):
        return None
    # 跨标门与本题无关，桩掉以免结论受"维度不匹配恰好报错"这种偶然性影响
    monkeypatch.setattr(eng, "_cross_symbol_oos", _no_cross)
    return eng, registered, holder


async def _noop_register(registered):
    registered.append("rl_evolve")


def _seed_anchor(eng, fitness, data_source, oos_ret=None, identity=None):
    meta = {"fitness": fitness, "data_source": data_source}
    if oos_ret is not None:
        meta["oos_ret"] = oos_ret
    if identity:
        meta.update(identity)
    eng.zoo.save_agent(_agent(), "strategy_drl", meta=meta)


@pytest.mark.asyncio
async def test_demo_anchor_does_not_block_a_profitable_exchange_model(engine):
    """锁死现场复现：demo 锚点 fitness=3.9167，真实交易所一轮 best_ret=0.30
    （且样本外赚钱、过了部署硬门）。旧逻辑判 rollback，新逻辑判不可比并放行。"""
    eng, registered, holder = engine
    _seed_anchor(eng, 3.916665, "demo")
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)

    await eng._train_strategy_drl_once()

    assert eng.zoo.best_fitness("strategy_drl") == 0.30, "真实模型应重建锚点"
    assert eng.zoo.best_info("strategy_drl")["data_source"] == "exchange"
    assert registered == ["rl_evolve"]


@pytest.mark.asyncio
async def test_comparable_exchange_anchor_still_protects(engine):
    """守卫不能顺手关掉回退保护：同口径（交易所+同训练身份）时劣质模型仍要被旧锚点压住。"""
    eng, registered, holder = engine
    # 锚点必须与当前训练轮同身份（symbol/timeframe/state_window/config_fingerprint）
    # 才会进入回退比较——缺身份字段的旧锚点按 unknown 处理，不压新模型。
    from drl.evolve_engine import _training_identity, _strategy_drl_train_cfg
    tf = eng._effective_timeframe()
    sym = eng._symbols[eng._symbol_idx % len(eng._symbols)]  # 不推进轮换索引
    cfg = _strategy_drl_train_cfg(8)
    identity = _training_identity(
        "strategy_drl", sym, tf,
        state_window=int(cfg["state_window"]),
        config={
            "episodes": int(getattr(settings, "evolve_strategy_drl_episodes", 8)),
            "vol_penalty": float(getattr(settings, "evolve_vol_penalty", 20.0)),
            "oos_min_bars": int(getattr(settings, "evolve_oos_min_bars", 250)),
        })
    _seed_anchor(eng, 9.0, "exchange", oos_ret=0.12, identity=identity)
    holder["result"] = _result(best_ret=0.30, oos_ret=0.01)

    await eng._train_strategy_drl_once()

    assert eng.zoo.best_fitness("strategy_drl") == 9.0, "同口径退化模型不得覆盖锚点"
    assert eng.zoo.best_info("strategy_drl")["fitness"] == 9.0


@pytest.mark.asyncio
async def test_legacy_anchor_without_identity_does_not_block_new_model(engine):
    """旧版本锚点缺身份字段应按 unknown 处理：不回退比较，候选凭自身 OOS 建立新基线。"""
    eng, registered, holder = engine
    _seed_anchor(eng, 9.0, "exchange", oos_ret=0.12)
    holder["result"] = _result(best_ret=0.30, oos_ret=0.05)

    await eng._train_strategy_drl_once()

    assert eng.zoo.best_fitness("strategy_drl") == 0.30
    assert eng._deploy_status["strategy_drl"]["anchor_mismatch"]


@pytest.mark.asyncio
async def test_status_exposes_anchor_and_stall(engine):
    eng, registered, holder = engine
    _seed_anchor(eng, 3.916665, "demo")
    eng._reject_streak["strategy_drl"] = 72
    eng._rounds_total["strategy_drl"] = 72

    st = eng.status()
    anchor = st["anchor"]["strategy_drl"]
    assert anchor["fitness"] == 3.916665
    assert anchor["data_source"] == "demo"
    assert anchor["comparable"] is False
    assert st["strategy_drl"]["reject_streak"] == 72
    assert st["strategy_drl"]["stalled"] is True
    assert st["strategy_drl"]["stall_reason"]
    # 真锁死：不可比锚点 + 连续拒绝（前端只在这种情况给"重置锚点"按钮）
    assert st["strategy_drl"]["anchor_locked"] is True

    # 锚点同口径时的连续拒绝是"正常保护"：模型确实不如基线，重置锚点于事无补
    _seed_anchor(eng, 9.0, "exchange")
    assert eng.status()["strategy_drl"]["anchor_locked"] is False

    # 重置后（或重启从库里恢复出历史 streak 时）没有基线就谈不上"被基线锁死"
    eng.zoo.reset_anchor("strategy_drl")
    view = eng.status()["strategy_drl"]
    assert view["stalled"] is True and view["anchor_locked"] is False

    eng._reject_streak["strategy_drl"] = 1
    st2 = eng.status()
    assert st2["strategy_drl"]["stalled"] is False


@pytest.mark.asyncio
async def test_reset_anchor_via_engine_unlocks_and_reports(engine):
    eng, registered, holder = engine
    _seed_anchor(eng, 3.916665, "demo")

    out = await eng.reset_anchor("strategy_drl", reason="UI 一键重置")

    assert out["ok"] is True
    assert eng.zoo.best_fitness("strategy_drl") is None
    assert eng.status()["anchor"]["strategy_drl"]["fitness"] is None
    assert eng._reject_streak["strategy_drl"] == 0

    holder["result"] = _result(best_ret=0.28, oos_ret=0.03)
    await eng._train_strategy_drl_once()
    assert eng.zoo.best_fitness("strategy_drl") == 0.28


@pytest.mark.asyncio
async def test_reset_anchor_rejects_unknown_model(engine):
    out = await engine[0].reset_anchor("nope")
    assert out["ok"] is False


@pytest.mark.asyncio
async def test_log_round_counts_reject_streak(engine):
    """streak 是纯内存计数，落库失败也不能丢（前端锁死提示依赖它）。"""
    eng, _, _ = engine
    await eng._log_round("strategy_drl", "BTC/USDT", "1h", "exchange", 0.1, status="rollback")
    await eng._log_round("strategy_drl", "BTC/USDT", "1h", "exchange", 0.1, status="oos_rejected")
    assert eng._reject_streak["strategy_drl"] == 2
    await eng._log_round("strategy_drl", "BTC/USDT", "1h", "exchange", 0.1, status="ok")
    assert eng._reject_streak["strategy_drl"] == 0


@pytest.mark.asyncio
async def test_restore_streak_stops_at_anchor_reset(tmp_path):
    """锚点重置是一条账界线。重置前的连续拒绝针对的是已作废的基线，重启后再把它们
    算回来，横幅就会报一个"没人能追平"的 streak（现场实测：重置后重启显示 35/10）。"""
    from datetime import datetime, timedelta, timezone
    from core.bus import EventBus
    from core.database import Database, EvolveRound

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    eng = EvolveEngine(db, EventBus(), tmp_path / "models")
    eng.zoo.save_agent(_agent(), "strategy_drl",
                       meta={"fitness": 3.916665, "data_source": "demo"})
    eng.zoo.reset_anchor("strategy_drl")
    meta = json.loads(eng.zoo._meta_path("strategy_drl").read_text(encoding="utf-8"))
    cut = datetime.fromtimestamp(meta["anchor_reset"]["timestamp"], tz=timezone.utc).replace(tzinfo=None)

    async with db.session() as s:
        # 重置前 6 轮：针对的是那份已作废的 demo 锚点
        for i in range(1, 7):
            s.add(EvolveRound(model="strategy_drl", round_no=i, status="rollback",
                              ts=cut - timedelta(minutes=7 - i)))
        # 重置后 2 轮：才是当前该被数进 streak 的历史
        for i in range(7, 9):
            s.add(EvolveRound(model="strategy_drl", round_no=i, status="rollback",
                              ts=cut + timedelta(minutes=i - 6)))
        # 没有重置记录的管线：全部历史都算数
        for i in range(1, 4):
            s.add(EvolveRound(model="factor_miner", round_no=i, status="rollback",
                              ts=cut + timedelta(minutes=i)))
        await s.commit()
    await eng._restore_episode_counts()

    assert eng._reject_streak["strategy_drl"] == 2
    assert eng._reject_streak["factor_miner"] == 3
    # 轮次是累计事实，与账界线无关
    assert eng._strategy_drl_status["episode"] == 8
