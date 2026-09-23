"""TDD：训练能力参数可配置化（诊断报告 §15 剩余杠杆 1）。

train_drl 批量/更新参数（n_episodes、ppo_epochs）当前在
_strategy_drl_train_cfg 硬编码 4/4。模型训练段能交易但 val/OOS 不交易，
下一步实验把批量加大（4→8，降方差）与 PPO 更新更充分（4→8），
观察模型是否开始交易。两参数不进入 _training_identity（episodes/vol_penalty
才在身份里），改它们不触发锚点身份不可比，可平滑生效。
"""

from drl.evolve_engine import _strategy_drl_train_cfg


def test_train_cfg_default_batch_params_upgraded(monkeypatch):
    """默认 n_episodes=8、ppo_epochs=8（settings 可覆盖）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_n_episodes", 8,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_ppo_epochs", 8,
                        raising=False)
    cfg = _strategy_drl_train_cfg(32)
    assert cfg["n_episodes"] == 8
    assert cfg["ppo_epochs"] == 8
    assert cfg["episodes"] == 32


def test_train_cfg_respects_settings_override(monkeypatch):
    """settings 覆盖生效（可热调回 4 保持旧行为）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_n_episodes", 4,
                        raising=False)
    monkeypatch.setattr(settings, "evolve_strategy_drl_ppo_epochs", 4,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["n_episodes"] == 4
    assert cfg["ppo_epochs"] == 4


def test_train_cfg_default_min_trade_zone_upgraded(monkeypatch):
    """默认 min_trade_zone=0.10（训练环境最小交易区加大 → 模型学会少动，降换手）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_min_trade_zone", 0.10,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["min_trade_zone"] == 0.10


def test_train_cfg_min_trade_zone_override(monkeypatch):
    """settings 覆盖回 0.05（旧行为）生效。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "evolve_strategy_drl_min_trade_zone", 0.05,
                        raising=False)
    cfg = _strategy_drl_train_cfg(8)
    assert cfg["min_trade_zone"] == 0.05