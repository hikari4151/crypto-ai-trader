"""TDD：ModelZoo 部署元数据持久化——factor_composite 必须随 flat 文件保存。

背景（2026-09-19 walk-forward 评估中发现）：strategy_drl 最佳模型 state_dim=55
（54 基础维 + 1 级联因子列），但 flat/best 文件里 factor_composite/factor_mu/factor_sd
全部丢失 → rl_adaptive 与评估脚本都无法重建模型状态（此前 _write_flat_meta 只在
接受轮把因子元数据写进 flat，下次 restore_deployment 从 gz 重建 flat 时被抹掉）。

修复：_DEPLOYMENT_META_KEYS 补上 factor_composite（已有 factor_mu/factor_sd/
factor_expression），使 gz 的 _meta 能随 flat 一起持久化。
"""
from drl.model_zoo import ModelZoo


def test_flat_payload_promotes_factor_composite():
    """factor_composite 属于部署元数据，必须从 _meta 提升到 flat 顶层。"""
    data = {
        "state_dim": 55,
        "_meta": {
            "fitness": 0.01,
            "factor_composite": {"weights": {"mom_7": 1.0}},
            "factor_mu": 0.2,
            "factor_sd": 1.5,
        },
        "_version": 3,
    }
    flat = ModelZoo._flat_payload(data)
    assert flat["factor_composite"] == {"weights": {"mom_7": 1.0}}
    assert flat["factor_mu"] == 0.2
    assert flat["factor_sd"] == 1.5
