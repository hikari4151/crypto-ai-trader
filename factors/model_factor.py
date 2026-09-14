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
                     seed: int = 42, gates: Optional[dict] = None,
                     cross_validate_data: Optional[dict[str, pd.DataFrame]] = None) -> dict:
    """训练监督学习因子并做三段式质检。

    df: 标准 OHLCV（index=时间戳，列 open/high/low/close/volume）
    cross_validate_data: 可选跨品种验证数据 {symbol: OHLCV DataFrame}（≥200 根）。
        用训练段 mu/sd 标准化后经最后一窗模型预测，对各品种算 rank_ic/icir，
        写入 meta.cross_val_ic（默认 None → 空 dict，行为与旧版一致）。
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

    # 8) 跨品种验证：用训练段 mu/sd 标准化后经模型预测，算 rank_ic
    cross_val_ic = {}
    if cross_validate_data:
        for sym, cdf in cross_validate_data.items():
            if len(cdf) < 200:
                continue
            try:
                from drl.env import _precompute_features
                feats_c = np.asarray(_precompute_features(cdf), dtype=float)
                X_c = (feats_c - mu) / sd
                pred_c = model.forward(X_c)[-1].reshape(-1)
                s_c = pd.Series(pred_c, index=cdf.index)
                g_c = factor_quality_gate(s_c, cdf["close"], h=h)
                cross_val_ic[sym] = {"rank_ic": g_c["rank_ic"], "icir": g_c["icir"],
                                     "samples": g_c["samples"]}
            except Exception as e:
                log.warning("[factor] 模型因子跨品种 %s 验证失败: %s", sym, e)
    meta["cross_val_ic"] = cross_val_ic

    return {"factor": factor, "report": report, "model": model, "meta": meta,
            "feature_dim": _FEAT_DIM,
            "feature_desc": "drl.env._precompute_features（13 维价量特征，trailing 无未来信息）"}

