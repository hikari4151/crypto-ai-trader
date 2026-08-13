"""早停 patience 调参实测：对比不同 patience 的停止时机与最终质量。"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import numpy as np
import pandas as pd

def make_data(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.012)
    drift = np.zeros(n)
    drift[0:500] = 0.004
    drift[500:900] = 0.0
    drift[900:1400] = 0.003
    rets = rng.normal(drift, vol)
    close = 100.0 * np.exp(np.cumsum(rets))
    ohlc = np.column_stack([np.roll(close, 1), close * 1.001, close * 0.999, close])
    rows = [[i * 3600000, ohlc[i, 0], ohlc[i, 1], ohlc[i, 2], ohlc[i, 3],
             float(abs(rng.normal(100, 20)))] for i in range(n)]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)

df = make_data()
from drl import train_drl

base = {"hidden": [32, 32], "lr_actor": 2e-3, "lr_critic": 5e-3,
        "vol_penalty": 20.0, "entropy_coef": 0.05, "val_eval_interval": 10,
        "lr_final_ratio": 0.3, "episodes": 300, "seed": 42}

print("=== 早停 patience 对比（每10轮验证一次，目标300轮） ===")
print(f"{'patience':<10}{'停止轮数':<10}{'节省轮数':<10}{'best_val_ret':<14}{'达到最优轮':<10}")
for patience in [2, 3, 5, 8, 0]:  # 0=不早停，训练满300轮
    cfg = dict(base, early_stop_patience=patience)
    res = train_drl(df, cfg, on_progress=None)
    hist = res["history"]
    stopped = len(hist)
    saved = 300 - stopped
    # 最优轮：找到 best_val_ret 对应的 episode
    best_ep = 0
    bv = -1e9
    for h in hist:
        if h.get("val_ret") is not None and h["val_ret"] > bv:
            bv = h["val_ret"]
            best_ep = h["episode"]
    label = "满300轮" if patience == 0 else f"p={patience}"
    print(f"{label:<10}{stopped:<10}{saved:<10}{res['best_val_ret']:<14.4f}{best_ep:<10}")
print("\n注: best_val_ret 是验证段最优收益（真实质量），达到最优轮=训练中第几轮达到最优")
