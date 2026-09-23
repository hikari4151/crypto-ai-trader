"""TDD：GAP-01 验收口径固定为 4 折 × 15000 根（诊断建议③）。

背景（2026-09-21）：引擎 gap01 观测原为 2 折 × 6000 根，v36(round978) 在
该口径 PASS 但 4 折 × 15000 根 FAIL——2 折是"最近窗口红利"，让过拟合模型
假过考试。固定 4 折 × 15000 根与 scripts/walkforward_rl_evolve.py 口径一致，
消除幸存者错觉。
"""
from config.settings import settings


def test_gap01_default_folds_fixed_to_4():
    assert settings.evolve_gap01_folds == 4, \
        f"gap01 折数应固定为 4（当前 {settings.evolve_gap01_folds}）"


def test_gap01_default_bars_fixed_to_15000():
    assert settings.evolve_gap01_bars == 15000, \
        f"gap01 观测窗口应固定为 15000 根（当前 {settings.evolve_gap01_bars}）"