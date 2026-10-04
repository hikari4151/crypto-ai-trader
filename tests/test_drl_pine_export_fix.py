"""修复"DRL 训练失去 Pine 导出"专项测试。

背景（2026-10）：Pine 导出链路本身未损坏（try_build_drl_pine / 模型文件导出 /
evolve 注册三入口均正常），但**手动训练 API 默认 hidden=[64,64] 产生的脚本
必然超过 Pine 体积门（64KB）被拦截**——用户用默认配置训练完永远看不到
"查看 Pine 交易代码"，表现为"失去了 pine 导出"。

修复：
- TrainIn.hidden 默认 [64,64] → [16,16]（与 evolve 引擎对齐，默认训练可导出）
- 前端训练面板暴露 hidden 配置，让用户可改且可见原因
- 大网络模型训练结果明确给 pine_note（不静默）

覆盖：
- API 默认配置可导出 Pine
- 用户调大 hidden 时给出明确 pine_note 而非静默
- 训练流程中 pine_code/pine_note 正确透出（end-to-end）
"""
import json

import pytest


def test_train_drl_default_hidden_exports_pine():
    """train_drl 用 API 默认 hidden=[16,16] 训练：必须产出可导出的 Pine 代码。

    本用例只验证 **Pine 体积门/导出链路**，故关闭 OOS 硬门（`oos_hard_gate: False`）：
    2026-10 起过拟合衰减改走**超额**口径（docs/DRL成本标定-决策频率与滑点复核.md §4.1），
    在 `generate_demo` 的随机游走数据上"空仓在下跌段白拿超额"会被判为不可复现 →
    deployment_blocked=True → 没有 Pine 代码。那是判据的正确行为（由
    tests/test_deploy_gate_benchmark.py 覆盖），与"网络装不装得进 Pine"无关。
    """
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="5m", n=700, seed=7)
    cfg = {"episodes": 2, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 1, "oos_min_bars": 80, "time_budget": 0,
           "oos_hard_gate": False}
    res = train_drl(df, cfg)
    assert res.get("pine_code"), "默认 [16,16] 训练必须产出 Pine 代码"
    assert not res.get("pine_note")
    assert "plotshape" in res["pine_code"], "Pine 代码必须含买卖点标记"


def test_train_drl_large_hidden_gives_explicit_note():
    """用户调大 hidden=[64,64] 且关闭蒸馏：明确给出"装不进 Pine"的原因，不静默。

    （蒸馏开启时大网络会自动蒸馏导出 Pine——见 test_drl_distill.py；
    本测试覆盖"关闭蒸馏"的旧行为：仍要显式原因而非静默。
    同上级：关闭 OOS 硬门，隔离"体积门"这一个被测点。）
    """
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    df = generate_demo(timeframe="5m", n=700, seed=7)
    cfg = {"episodes": 1, "hidden": [64, 64], "n_episodes": 1, "ppo_epochs": 1,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 1, "oos_min_bars": 80, "time_budget": 0,
           "distill": False, "oos_hard_gate": False}
    res = train_drl(df, cfg)
    assert not res.get("pine_code"), "关闭蒸馏时大网络模型应被体积门拦截"
    note = res.get("pine_note") or ""
    assert "Pine" in note or "装不进" in note or "KB" in note, \
        f"应给出明确的 Pine 不可导出原因，实际 note={note!r}"


def test_train_drl_multi_symbol_still_exports_pine():
    """多标的混合训练（经验广度）不影响 Pine 导出——默认网络仍可导出。"""
    from backtest.data_loader import generate_demo
    from drl.agent import train_drl
    main_df = generate_demo(timeframe="5m", n=700, seed=1)
    eth_df = generate_demo(timeframe="5m", n=700, seed=2)
    cfg = {"episodes": 2, "hidden": [16, 16], "n_episodes": 2, "ppo_epochs": 2,
           "mini_batch_size": 32, "vol_penalty": 20, "seed": 1,
           "start_cash": 10000, "fee_rate": 0.001, "slippage": 0.0005,
           "state_window": 1, "oos_min_bars": 80, "time_budget": 0,
           "extra_mix_ratio": 0.5, "oos_hard_gate": False}
    res = train_drl(main_df, cfg, extra_dfs={"ETH/USDT": eth_df})
    assert res.get("multi_symbol") is True
    assert res.get("pine_code"), "多标训练 + 默认网络仍应可导出 Pine"
    assert "plotshape" in res["pine_code"]


def test_train_in_default_hidden_is_exportable():
    """API TrainIn 默认 hidden=[16,16]（不再 [64,64]——后者必然被 Pine 体积门拦截）。"""
    from web.api.drl import TrainIn
    body = TrainIn()
    assert body.hidden == [16, 16], \
        f"默认 hidden 应为可导出的 [16,16]，实际 {body.hidden}"


def test_train_in_large_hidden_still_valid():
    """用户显式调大 hidden 仍被接受（软件内运行场景合法），校验不误拦。"""
    from web.api.drl import TrainIn
    body = TrainIn(hidden=[64, 64])
    assert body.hidden == [64, 64]


# ============ 模型库"使用"注册的 internal_only 标记 ============

def _write_model(tmp_path, hidden, state_window):
    """写一个可直接被 export_pine_from_model 读取的模型文件。"""
    from drl.agent import ACAgent
    sd = 13 * state_window + 2
    agent = ACAgent(sd, 5, hidden=tuple(hidden), state_window=state_window, seed=1)
    data = agent.to_dict()
    data["state_window"] = state_window
    path = tmp_path / f"model_{'_'.join(map(str, hidden))}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    """把 MODEL_DIR 指向临时目录（避免测试污染真实模型库）。"""
    fake = tmp_path / "models"
    fake.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("web.api.drl.MODEL_DIR", fake)
    return fake


@pytest.fixture
def use_api(tmp_path, monkeypatch, isolated_model_dir):
    """挂载 drl 路由的 TestClient + 假 engine（use_model 端点专用）。"""
    from web.deps import get_db, get_engine
    import strategies
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.api.drl import router

    class _StubEngine:
        async def select_strategy(self, name):
            self.selected = name
            return True

    class _StubDB:
        pass

    app = FastAPI()
    app.include_router(router)
    engine = _StubEngine()
    app.state.db = _StubDB()
    app.state.engine = engine
    app.dependency_overrides[get_db] = lambda: _StubDB()
    app.dependency_overrides[get_engine] = lambda: engine
    captured = {}
    monkeypatch.setattr(strategies, "register_dynamic",
                        lambda name, spec: captured.update(spec=spec))
    return TestClient(app), captured


def test_use_model_marks_internal_only_for_large_net(use_api, isolated_model_dir, tmp_path):
    """模型库"使用"大网络模型（装不进 Pine）→ 策略 spec.internal_only=True。"""
    client, captured = use_api
    model_path = _write_model(tmp_path, [64, 64], 1)  # 198KB > 64KB 体积门
    (isolated_model_dir / "big_net_test.json").write_bytes(model_path.read_bytes())

    r = client.post("/api/drl/models/big_net_test/use")
    assert r.status_code == 200, r.text
    assert captured["spec"].get("internal_only") is True, \
        "大网络模型注册时应标记'仅软件内运行'"
    assert not captured["spec"].get("pine_code")
    assert captured["spec"].get("pine_note")


def test_use_model_no_internal_only_when_pine_ok(use_api, isolated_model_dir, tmp_path):
    """模型库"使用"可导出 Pine 的模型 → 不标 internal_only（有真实代码）。"""
    client, captured = use_api
    model_path = _write_model(tmp_path, [16, 16], 1)  # 23KB，可导出
    (isolated_model_dir / "small_net_test.json").write_bytes(model_path.read_bytes())

    r = client.post("/api/drl/models/small_net_test/use")
    assert r.status_code == 200, r.text
    assert captured["spec"].get("internal_only") is not True
    assert captured["spec"].get("pine_code")
