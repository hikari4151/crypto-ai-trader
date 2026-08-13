"""监督学习因子（model-as-factor，借鉴 Qlib 范式）。

用轻量 MLP 在价量特征上预测未来 h 期收益，把模型输出作为"因子"接入现有因子体系：
    特征(_precompute_features 13维, trailing 无未来信息) → MLP 回归预测未来收益
    → 预测序列 = 因子 → 因子安检门（IC/ICIR/换手/分层）→ 可选注册到 factor_custom

防过拟合（核心设计，严格时间切分）：
- train/val/OOS 三段按时间顺序切分，互不重叠；val 只用于早停，OOS 只用于最终报告
- 标准化 scaler 只在 train 段拟合，val/OOS 复用（杜绝统计泄漏）
- 早停：val 段 MSE 连续 patience 轮无改善则停止，保留最佳权重
- L2 正则（weight decay）抑制权重膨胀
- 安检判定段 = val 段 IC（模型从未在 val 上训练；OOS 不参与任何选择，只报告衰减）
- 门槛不过 → valid=False 并给出原因（与 AI 挖掘因子同一套 DEFAULT_GATES）

效率：
- 纯 numpy 矩阵运算（复用 drl.nnet.MLP），特征一次性向量化预计算
- 2000 根K线、200 轮训练秒级完成（web 层用 asyncio.to_thread 防阻塞事件循环）
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .analysis import DEFAULT_GATES, factor_quality_gate

log = logging.getLogger(__name__)

_FEAT_DIM = 13  # drl.env._precompute_features 的输出维度


def _forward_returns_label(close: np.ndarray, h: int) -> np.ndarray:
    """未来 h 期收益率标签（%）：末尾 h 期为 NaN（无未来数据，避免前视）。"""
    n = len(close)
    y = np.full(n, np.nan)
    y[: n - h] = close[h:] / close[: n - h] - 1.0
    return y * 100.0  # 百分比标度，与特征量级一致


def _fit_mlp_regressor(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       hidden: tuple, lr: float, l2: float,
                       epochs: int, patience: int, seed: int):
    """训练 MLP 回归（MSE + L2 + val 早停），返回 (model, best_val_loss, best_epoch)。

    X_tr/y_tr 已剔除 NaN 标签；val 段标签尾部可能仍有 NaN，用 mask 处理。
    """
    from drl.nnet import MLP

    model = MLP([X_tr.shape[1], *hidden, 1], seed=seed, lr=lr)
    best_loss, best_epoch = float("inf"), 0
    best_W = best_b = None
    stall = 0
    mask_val = np.isfinite(y_val)
    for ep in range(1, epochs + 1):
        # 前向 + MSE 梯度（均值归一，数值稳定）
        acts = model.forward(X_tr)
        pred = acts[-1]
        dout = 2.0 * (pred - y_tr.reshape(-1, 1)) / max(1, len(y_tr))
        gW, gb = model.backward(acts, dout)
        # L2 正则（weight decay，不加 bias）
        for i in range(len(gW)):
            gW[i] = gW[i] + l2 * model.W[i]
        model.apply_grad(gW, gb)

        # 每 5 轮在 val 段评估一次（早停判定）
        if ep % 5 == 0 and mask_val.sum() > 0:
            vpred = model.forward(X_val)[-1].reshape(-1)
            vloss = float(np.mean((vpred[mask_val] - y_val[mask_val]) ** 2))
            if vloss < best_loss - 1e-6:
                best_loss, best_epoch = vloss, ep
                best_W = [w.copy() for w in model.W]
                best_b = [bb.copy() for bb in model.b]
                stall = 0
            else:
                stall += 5
                if stall >= patience:
                    break
    if best_W is not None:
        model.W, model.b = best_W, best_b  # 恢复 val 最优权重
    return model, best_loss, best_epoch


def _segment_report(factor: pd.Series, close: pd.Series, h: int,
                    n_train: int, n_val: int, gates: Optional[dict]) -> dict:
    """分段 IC 报告：train / val / OOS 各段跑安检门，判定段 = val。"""
    g = {**DEFAULT_GATES, **(gates or {})}
    n = len(factor)
    segments: dict[str, dict] = {}
    for name, a, b in (("train", 0, n_train),
                       ("val", n_train, n_val),
                       ("oos", n_val, n)):
        if b - a >= 30:
            segments[name] = factor_quality_gate(
                factor.iloc[a:b], close.iloc[a:b], h=h, gates=g)
    val = segments.get("val")
    if val is None:
        val = {"valid": False, "reason": "val 段过短，无法独立验证（请增加数据量）"}
        segments["val"] = val
    report = {
        "segments": segments,
        "valid": bool(val.get("valid")),
        "reason": val.get("reason", ""),
        "gate": g,
        "decay": _ic_decay(segments),
    }
    return report


def _ic_decay(segments: dict) -> Optional[float]:
    """OOS 相对 val 的 |rank_ic| 衰减（>0.5 视为疑似过拟合，与 drl OOS 报告同思路）。

    val 段 IC 弱于门槛时衰减无统计意义，返回 None（避免无意义的大负值）。
    """
    val_ic = abs((segments.get("val") or {}).get("rank_ic", 0.0))
    oos_ic = abs((segments.get("oos") or {}).get("rank_ic", 0.0))
    if val_ic < DEFAULT_GATES["min_abs_ic"]:
        return None
    return round(1.0 - oos_ic / val_ic, 4)


def fit_model_factor(df: pd.DataFrame, h: int = 1,
                     hidden: tuple = (32, 32), epochs: int = 200,
                     lr: float = 2e-3, l2: float = 1e-4, patience: int = 25,
                     train_ratio: float = 0.7, val_ratio: float = 0.15,
                     seed: int = 42, gates: Optional[dict] = None) -> dict:
    """训练监督学习因子并做三段式质检。

    df: 标准 OHLCV（index=时间戳，列 open/high/low/close/volume）
    返回 {factor: Series(预测值=因子), report: 分段 IC/安检, model: MLP, meta}
    """
    from drl.env import _precompute_features

    n = len(df)
    if n < 300:
        raise ValueError(f"模型因子需要至少 300 根K线（当前 {n}）")
    h = max(1, int(h))
    close = df["close"]

    # 1) 特征（trailing 窗口，无未来信息）
    feats = np.asarray(_precompute_features(df), dtype=float)
    # 2) 标签：未来 h 期收益（%）
    y = _forward_returns_label(close.to_numpy(float), h)

    # 3) 严格时间切分（train → val → OOS，互不重叠）
    n_train = max(int(n * train_ratio), 60)
    n_val = max(int(n * (train_ratio + val_ratio)), n_train + 60)
    if n_val >= n - 30:
        raise ValueError("数据量不足：val/OOS 段过短，请增加数据或调低 train_ratio")

    # 4) 标准化：scaler 只在 train 段拟合（val/OOS 复用，杜绝统计泄漏）
    mu = feats[:n_train].mean(axis=0)
    sd = feats[:n_train].std(axis=0) + 1e-8
    X = (feats - mu) / sd

    # 5) 训练（val 早停；训练样本剔除标签 NaN）
    mask_tr = np.isfinite(y[:n_train])
    X_tr, y_tr = X[:n_train][mask_tr], y[:n_train][mask_tr]
    model, best_val_loss, best_epoch = _fit_mlp_regressor(
        X_tr, y_tr, X[n_train:n_val], y[n_train:n_val],
        tuple(hidden), lr, l2, epochs, patience, seed)

    # 6) 全量预测 = 因子序列；尾部 h 期无标签可验证，预测值也置 NaN（严谨性）
    pred = model.forward(X)[-1].reshape(-1)
    factor = pd.Series(pred, index=df.index, name="model_factor")
    factor.iloc[n - h:] = np.nan

    # 7) 三段 IC 报告 + 安检（判定段 = val）
    report = _segment_report(factor, close, h, n_train, n_val, gates)

    meta = {
        "h": h, "hidden": list(hidden), "epochs": best_epoch or epochs,
        "lr": lr, "l2": l2, "patience": patience, "seed": seed,
        "trained_bars": int(mask_tr.sum()),
        "best_val_loss": round(best_val_loss, 6),
    }
    return {"factor": factor, "report": report, "model": model, "meta": meta,
            "feature_dim": _FEAT_DIM,
            "feature_desc": "drl.env._precompute_features（13 维价量特征，trailing 无未来信息）"}


def walk_forward_model_factor(df: pd.DataFrame, h: int = 1,
                              train_ratio: float = 0.5, step_ratio: float = 0.1,
                              oos_ratio: float = 0.15,
                              hidden: tuple = (32, 32), epochs: int = 200,
                              lr: float = 2e-3, l2: float = 1e-4,
                              patience: int = 25, seed: int = 42) -> dict:
    """walk-forward 滚动重训评估（Qlib 滚动再训练范式）。

    与单次 train/val/OOS（fit_model_factor）不同，本函数模拟部署现实：
    - 每窗：训练段内切 train/val（早停），在紧邻的 OOS 段评估——只报告从未见过的数据
    - 窗口滚动前移，输出整条 OOS IC 序列：均值/正率/稳定性，泛化由滚动序列说话
      （单次切分只有一段 OOS，容易被一段行情误导）
    - 返回最后一窗模型（部署用）及其全量预测因子

    防前视：每窗的标准化只在该窗训练段拟合；OOS 段不参与任何拟合/选择。
    """
    from drl.env import _precompute_features

    n = len(df)
    if n < 500:
        raise ValueError(f"walk-forward 需要至少 500 根K线（当前 {n}）")
    h = max(1, int(h))
    close = df["close"]
    feats = np.asarray(_precompute_features(df), dtype=float)
    y = _forward_returns_label(close.to_numpy(float), h)

    train_len = max(200, int(n * train_ratio))
    oos_len = max(60, int(n * oos_ratio))
    step = max(60, int(n * step_ratio))

    windows: list[dict] = []
    model = None
    mu = sd = None
    start = 0
    while start + train_len + oos_len <= n:
        end_tr = start + train_len
        end_oos = end_tr + oos_len
        # 标准化只在训练段拟合（防统计泄漏）
        mu = feats[start:end_tr].mean(axis=0)
        sd = feats[start:end_tr].std(axis=0) + 1e-8
        X = (feats - mu) / sd
        # 训练段内部切 val（早停用）：末尾 15% 为 val，训练段其余为 train
        n_val = max(60, int(train_len * 0.15))
        tr_end = end_tr - n_val
        mask = np.isfinite(y[start:tr_end])
        model, _, best_ep = _fit_mlp_regressor(
            X[start:tr_end][mask], y[start:tr_end][mask],
            X[tr_end:end_tr], y[tr_end:end_tr],
            tuple(hidden), lr, l2, epochs, patience, seed + start)
        # OOS 段评估（只报告，不参与选择）
        pred_oos = model.forward(X[end_tr:end_oos])[-1].reshape(-1)
        factor_oos = pd.Series(pred_oos, index=df.index[end_tr:end_oos])
        gate = factor_quality_gate(factor_oos, close.iloc[end_tr:end_oos], h=h)
        windows.append({
            "start": str(df.index[start]), "end": str(df.index[end_oos]),
            "train_bars": int(end_tr - start), "oos_bars": int(end_oos - end_tr),
            "oos_rank_ic": gate["rank_ic"], "oos_icir": gate["icir"],
            "oos_turnover": gate["turnover"], "oos_fitness": gate["fitness"],
            "best_epoch": best_ep,
        })
        start += step

    if not windows:
        raise ValueError("窗口数量不足，请增大数据量或调小 train_ratio/step_ratio")

    oos_ics = np.asarray([w["oos_rank_ic"] for w in windows])
    summary = {
        "n_windows": len(windows),
        "oos_ic_mean": round(float(oos_ics.mean()), 4),
        "oos_ic_std": round(float(oos_ics.std()), 4),
        "oos_ic_positive_rate": round(float((oos_ics > 0).mean()), 4),
        "oos_icir_mean": round(float(np.mean([w["oos_icir"] for w in windows])), 4),
        # 稳定：多数窗口 OOS IC 为正（方向一致）
        "stable": bool((oos_ics > 0).mean() >= 0.6),
    }

    # 最后一窗模型 → 全量预测（部署用；尾部 h 期置 NaN 无标签区）
    pred_all = model.forward((feats - mu) / sd)[-1].reshape(-1)
    factor = pd.Series(pred_all, index=df.index, name="model_factor_wf")
    factor.iloc[n - h:] = np.nan

    return {"factor": factor, "windows": windows, "summary": summary,
            "model": model, "meta": {"h": h, "hidden": list(hidden), "seed": seed,
                                     "train_len": train_len, "oos_len": oos_len,
                                     "step": step, "n_windows": len(windows)}}
