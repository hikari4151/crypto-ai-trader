"""验证早停修复：patience=0 跑满，patience>0 提前停。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import numpy as np
import pandas as pd


def make_data(n=800, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.003, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    ohlc = np.column_stack([np.roll(close, 1), close * 1.002, close * 0.998, close])
    rows = [[i * 3600000, ohlc[i, 0], ohlc[i, 1], ohlc[i, 2], ohlc[i, 3],
             float(abs(rng.normal(100, 20)))] for i in range(n)]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


df = make_data()
from drl import train_drl

base = {"hidden": [32, 32], "lr_actor": 2e-3, "lr_critic": 5e-3, "vol_penalty": 20.0,
        "entropy_coef": 0.05, "val_eval_interval": 8, "lr_final_ratio": 0.3, "seed": 42}

print("=== 早停修复验证（patience=0 应跑满 30 轮，patience=2 应提前停） ===")
for patience, episodes in [(0, 30), (2, 30)]:
    res = train_drl(df, dict(base, episodes=episodes, early_stop_patience=patience), on_progress=None)
    hist = res["history"]
    stopped = len(hist)
    early = any(h.get("early_stopped") for h in hist)
    exp = "提前停" if patience > 0 else "跑满"
    status = "OK" if ((early and patience > 0) or (not early and patience == 0)) else "FAIL"
    print(f"patience={patience}: 轮数={stopped}/{episodes} 早停触发={early} 预期[{exp}] -> {status}")
