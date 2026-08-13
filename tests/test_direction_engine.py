"""验证方向复算引擎：趋势/震荡数据上都能给出合理方向。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import numpy as np
from ai.direction_engine import compute_direction


def make_candles(closes, seed=1):
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, len(closes))))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, len(closes))))
    vols = np.abs(rng.normal(100, 20, len(closes)))
    return [[i, o, h, l, c, v] for i, (o, h, l, c, v) in enumerate(zip(opens, highs, lows, closes, vols))]


# 场景1：强上升趋势
rng = np.random.default_rng(7)
up_closes = 100 * np.exp(np.cumsum(rng.normal(0.004, 0.008, 120)))
# 场景2：强下降趋势
down_closes = 100 * np.exp(np.cumsum(rng.normal(-0.004, 0.008, 120)))
# 场景3：震荡（无趋势）
flat_closes = 100 + rng.normal(0, 0.6, 120).cumsum()
# 场景4：先跌后涨（拐点）
v_closes = 100 * np.exp(np.concatenate([np.cumsum(rng.normal(-0.005, 0.009, 60)),
                                        np.cumsum(rng.normal(0.006, 0.009, 60))]))

print("=== 方向复算引擎测试 ===")
for name, closes in [("上升趋势", up_closes), ("下降趋势", down_closes),
                     ("震荡", flat_closes), ("先跌后涨拐点", v_closes)]:
    res = compute_direction(make_candles(closes))
    print(f"\n{name}: direction={res['direction']} score={res['score']:+d} confidence={res['confidence']}")
    for s in res["signals"]:
        print(f"   {s['name']:<16} {s['sign']:+d}  {s['reason']}")
    assert res["direction"] in ("long", "short", "neutral")
    assert 0 <= res["confidence"] <= 1

# 断言：上升趋势应判 long，下降趋势应判 short
assert compute_direction(make_candles(up_closes))["direction"] == "long", "上升趋势应判 long"
assert compute_direction(make_candles(down_closes))["direction"] == "short", "下降趋势应判 short"
print("\nDIRECTION ENGINE OK")
