"""时序骨干网络：GRU 序列编码 + MLP head（可选 actor 架构，P1 能力天花板）。

背景（2026-10）：Pine 体积门曾把 actor 压到 16×16 MLP（能力上限低），
蒸馏导出已解决「大网络 → 可导出小网络」。本模块提供带时序记忆的 actor
骨干——GRU 把 state_window 内的 K 线序列建模为时序（而非简单展平堆叠），
突破 MLP 的「只看最近 N 根各自孤立」的局限，让策略能学到形态/惯性等
跨帧模式。

设计要点：
- **输入仍是展平窗口向量**（13×window + 账户 + 可选因子），外部接口与
  nnet.MLP 完全一致（forward / backward / apply_grad / predict_proba /
  predict_log_proba / to_dict / from_dict），因此：
    - ACAgent / train_batch / env / rl_adaptive 无需感知 backbone 差异；
    - D5 梯度范数裁剪（train_batch 内的 _clip_grad_norm）兼容——backward
      返回与 MLP 同风格的 (gW_list, gb_list)，GRU 参数按固定顺序并入。
- 内部把前 13×window 维 reshape 成 (window, 13) 序列喂 GRU；账户/因子
  作静态特征 concat 到 GRU 末态，再接 MLP head。
- 纯 numpy：GRUCell 前向 + BPTT 反向，Adam 更新。
- 默认不启用：ACAgent 用 backbone="mlp" 时完全走旧路径（向后兼容）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# 价量特征数（与 env._precompute_features / pine_export.N_PRICE_FEATURES 一致）
N_PRICE_FEATURES = 13

# GRU 参数组顺序：apply_grad 按此顺序消费 backward 输出的 gW/gb 前缀
_GRU_W_NAMES = ("Wz", "Wr", "Wn", "Uz", "Ur", "Un")
_GRU_B_NAMES = ("bz", "br", "bn")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(x).sum(axis=-1, keepdims=True) + 1e-12)
    return x - logsumexp


class GRUCell:
    """单层 GRU 单元（批量、纯 numpy、支持 BPTT 反向）。

    公式（标准 GRU）：
        z = σ(Wz·x + Uz·h_prev + bz)
        r = σ(Wr·x + Ur·h_prev + br)
        n = tanh(Wn·x + Un·(r ⊙ h_prev) + bn)
        h = (1 - z) ⊙ h_prev + z ⊙ n
    注：候选隐藏用「重置门缩放旧隐藏」标准形式（n 依赖 r⊙h_prev）。
    """

    def __init__(self, in_dim: int, hidden: int, seed: Optional[int] = None,
                 lr: float = 1e-3) -> None:
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.lr = lr
        lim = 1.0 / np.sqrt(max(1, in_dim + hidden))
        self.Wz = rng.uniform(-lim, lim, (in_dim, hidden))
        self.Uz = rng.uniform(-lim, lim, (hidden, hidden))
        self.bz = np.zeros(hidden)
        self.Wr = rng.uniform(-lim, lim, (in_dim, hidden))
        self.Ur = rng.uniform(-lim, lim, (hidden, hidden))
        self.br = np.zeros(hidden)
        self.Wn = rng.uniform(-lim, lim, (in_dim, hidden))
        self.Un = rng.uniform(-lim, lim, (hidden, hidden))
        self.bn = np.zeros(hidden)
        # Adam 状态
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._step_count = 0
        for name in _GRU_W_NAMES + _GRU_B_NAMES:
            self._m[name] = np.zeros_like(getattr(self, name))
            self._v[name] = np.zeros_like(getattr(self, name))

    # ---- 前向 ----
    def forward(self, x_seq: np.ndarray, h0: Optional[np.ndarray] = None
                ) -> tuple[list[dict], np.ndarray]:
        """前向：x_seq (batch, T, in_dim) → (每步缓存, 末态 (batch, hidden))。

        cache 每步含 (x, h_prev, z, r, n, h)，供 backward（BPTT）反传。
        """
        batch, T, _ = x_seq.shape
        h = (np.zeros((batch, self.hidden))
             if h0 is None else np.asarray(h0, dtype=float))
        cache: list[dict] = []
        for t in range(T):
            x = x_seq[:, t, :]
            z = _sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
            r = _sigmoid(x @ self.Wr + h @ self.Ur + self.br)
            n = np.tanh(x @ self.Wn + (r * h) @ self.Un + self.bn)
            h_new = (1.0 - z) * h + z * n
            cache.append({"x": x, "h_prev": h, "z": z, "r": r, "n": n, "h": h_new})
            h = h_new
        return cache, h

    # ---- 反向（BPTT） ----
    def backward(self, cache: list[dict], dh: np.ndarray
                 ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """反向：dh 为末态梯度 (batch, hidden)。

        返回与 nnet.MLP.backward 同风格的 (gW_list, gb_list)，顺序见
        _GRU_W_NAMES / _GRU_B_NAMES（apply_grad 按此顺序消费）。
        """
        grads = {name: np.zeros_like(getattr(self, name))
                 for name in _GRU_W_NAMES + _GRU_B_NAMES}
        T = len(cache)
        dx_seq = np.zeros((dh.shape[0], T, self.in_dim))
        h_next_grad = dh
        for t in range(T - 1, -1, -1):
            c = cache[t]
            x, h_prev, z, r, n = (c["x"], c["h_prev"], c["z"], c["r"],
                                  c["n"])
            d_h = h_next_grad
            # h = (1-z)*h_prev + z*n
            dz = d_h * (n - h_prev)
            d_h_prev = d_h * (1.0 - z)
            dn = d_h * z
            # n = tanh(x@Wn + (r*h_prev)@Un + bn)
            dn_pre = dn * (1.0 - n * n)
            grads["Wn"] += x.T @ dn_pre
            d_rh = dn_pre @ self.Un.T
            grads["Un"] += (r * h_prev).T @ dn_pre
            grads["bn"] += dn_pre.sum(axis=0)
            dr = d_rh * h_prev
            d_h_prev_r = d_rh * r
            # r = sigmoid(x@Wr + h_prev@Ur + br)
            dr_pre = dr * r * (1.0 - r)
            grads["Wr"] += x.T @ dr_pre
            grads["Ur"] += h_prev.T @ dr_pre
            grads["br"] += dr_pre.sum(axis=0)
            # z = sigmoid(x@Wz + h_prev@Uz + bz)
            dz_pre = dz * z * (1.0 - z)
            grads["Wz"] += x.T @ dz_pre
            grads["Uz"] += h_prev.T @ dz_pre
            grads["bz"] += dz_pre.sum(axis=0)
            # 输入梯度（本步 x）
            dx = (dz_pre @ self.Wz.T + dr_pre @ self.Wr.T
                  + dn_pre @ self.Wn.T)
            dx_seq[:, t, :] += dx
            # 上一步隐藏梯度（汇聚所有路径）
            h_next_grad = (d_h_prev + d_h_prev_r
                           + dz_pre @ self.Uz.T + dr_pre @ self.Ur.T)
        # 按固定顺序组列表（与 _GRU_W_NAMES/_GRU_B_NAMES 一致）
        gW = [grads[n] for n in _GRU_W_NAMES]
        gb = [grads[n] for n in _GRU_B_NAMES]
        return gW, gb

    def apply_grad(self, gW: list[np.ndarray], gb: list[np.ndarray]) -> None:
        """Adam 更新（与 nnet.MLP.apply_grad 同风格）。"""
        self._step_count += 1
        t = max(1, self._step_count)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1_t, b2_t = b1 ** t, b2 ** t
        for i, name in enumerate(_GRU_W_NAMES):
            g = gW[i]
            self._m[name] = b1 * self._m[name] + (1 - b1) * g
            self._v[name] = b2 * self._v[name] + (1 - b2) * (g * g)
            mh = self._m[name] / (1 - b1_t)
            vh = self._v[name] / (1 - b2_t)
            setattr(self, name,
                    getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))
        for i, name in enumerate(_GRU_B_NAMES):
            g = gb[i]
            self._m[name] = b1 * self._m[name] + (1 - b1) * g
            self._v[name] = b2 * self._v[name] + (1 - b2) * (g * g)
            mh = self._m[name] / (1 - b1_t)
            vh = self._v[name] / (1 - b2_t)
            setattr(self, name,
                    getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "in_dim": self.in_dim, "hidden": self.hidden, "lr": self.lr,
            "Wz": self.Wz.tolist(), "Uz": self.Uz.tolist(), "bz": self.bz.tolist(),
            "Wr": self.Wr.tolist(), "Ur": self.Ur.tolist(), "br": self.br.tolist(),
            "Wn": self.Wn.tolist(), "Un": self.Un.tolist(), "bn": self.bn.tolist(),
            "step_count": int(self._step_count),
        }

    @classmethod
    def from_dict(cls, data: dict, lr: Optional[float] = None) -> "GRUCell":
        cell = cls(int(data["in_dim"]), int(data["hidden"]),
                   lr=lr if lr is not None else float(data.get("lr", 1e-3)))
        for name in _GRU_W_NAMES + _GRU_B_NAMES:
            setattr(cell, name,
                    np.asarray(data[name], dtype=float).reshape(
                        getattr(cell, name).shape))
        cell._step_count = int(data.get("step_count", 0))
        return cell


class GRUActor:
    """时序 actor 骨干：GRU(窗口序列) → concat 静态 → MLP head → 动作 logits。

    外部接口与 nnet.MLP 对齐（forward / backward / apply_grad / predict_proba /
    predict_log_proba / to_dict / from_dict），从而：
    - ACAgent / train_batch / env / rl_adaptive 无感知；
    - train_batch 的 D5 梯度裁剪直接可用——backward 返回
      (gW = [GRU W...] + [head W...], gb = [GRU b...] + [head b...])，
      apply_grad 按前缀分派给 GRU 与 head。
    - 蒸馏导出直接作用于「GRU teacher → MLP student」。

    输入约定（与 env._build_state 一致）：
        前 13×window 维 = 窗口序列（旧→新堆叠），reshape (window, 13)；
        其后 n_static 维 = 账户状态（+ 可选因子），concat 到 GRU 末态。
    """

    def __init__(self, in_dim: int, n_actions: int, state_window: int = 4,
                 gru_hidden: int = 32, head_hidden: tuple[int, ...] = (32, 32),
                 seed: Optional[int] = None, lr: float = 1e-3,
                 n_price_features: int = N_PRICE_FEATURES) -> None:
        self.in_dim = int(in_dim)
        self.n_actions = int(n_actions)
        self.state_window = max(1, int(state_window))
        self.n_price_features = int(n_price_features)
        self.lr = float(lr)
        seq_len = self.n_price_features * self.state_window
        self.n_static = self.in_dim - seq_len
        if self.n_static < 0:
            raise ValueError(
                f"[gru] in_dim {in_dim} < 13×window {seq_len}，无法解析序列")
        self.gru_hidden = int(gru_hidden)
        self.gru = GRUCell(self.n_price_features, self.gru_hidden,
                           seed=seed, lr=lr)
        head_in = self.gru_hidden + self.n_static
        from .nnet import MLP
        self.head = MLP([head_in, *[int(h) for h in head_hidden],
                         self.n_actions], seed=(seed or 0) + 100, lr=lr)
        # 训练缓存（forward 时填充，backward/apply_grad 消费）
        self._last_cache: Optional[list[dict]] = None
        self._last_h: Optional[np.ndarray] = None
        self._last_seq: Optional[np.ndarray] = None
        self._last_static: Optional[np.ndarray] = None
        self._head_acts: Optional[list[np.ndarray]] = None

    # ---- 前向 ----
    def _split_seq(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """展平窗口向量 → (序列 (batch,T,13), 静态 (batch,n_static))。

        支持 1D（单样本）与 2D（批量）。
        """
        x2 = np.asarray(x, dtype=float)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        seq = x2[:, :self.n_price_features * self.state_window]
        seq = seq.reshape(x2.shape[0], self.state_window,
                          self.n_price_features)
        static = x2[:, self.n_price_features * self.state_window:]
        return seq, static

    def forward(self, x: np.ndarray,
                cache_for_backward: bool = True) -> list[np.ndarray]:
        """返回激活列表，末元素 = 输出 logits（与 MLP.forward 兼容）。

        cache_for_backward=True 时缓存 GRU BPTT 缓存与 head 激活（train_batch
        用）；False 时只推理（采样/部署/蒸馏收集路径，不缓存省内存）。
        """
        seq, static = self._split_seq(x)
        cache, h = self.gru.forward(seq)
        x_in = np.concatenate([h, static], axis=1)
        head_acts = self.head.forward(x_in, cache_for_backward=cache_for_backward)
        if cache_for_backward:
            self._last_cache = cache
            self._last_h = h
            self._last_seq = seq
            self._last_static = static
            self._head_acts = head_acts
        return [np.asarray(x, dtype=float), *head_acts[1:]]

    def predict_proba(self, x: np.ndarray, cache_for_backward: bool = True) -> np.ndarray:
        logits = self.forward(x, cache_for_backward=cache_for_backward)[-1]
        if logits.ndim == 2 and logits.shape[0] == 1 and logits.shape[1] <= 16:
            from .nnet import _softmax_1row
            return _softmax_1row(logits)
        return _softmax(logits)

    def predict_log_proba(self, x: np.ndarray, cache_for_backward: bool = True) -> np.ndarray:
        logits = self.forward(x, cache_for_backward=cache_for_backward)[-1]
        if logits.ndim == 2 and logits.shape[0] == 1 and logits.shape[1] <= 16:
            from .nnet import _log_softmax_1row
            return _log_softmax_1row(logits)
        return _log_softmax(logits)

    # ---- 反向 ----
    def backward(self, acts: list[np.ndarray], dout: np.ndarray
                 ) -> tuple[list, list]:
        """反向：head 反传 + GRU BPTT。返回 (gW, gb) 兼容 MLP 风格。

        - head 反传得输入梯度 d_in（concat 向量 [h, static] 的梯度）
        - 拆分 d_in → d_h（GRU 末态梯度）+ d_static
        - GRU BPTT（用 forward 缓存的 cache）
        - gW = [GRU W 梯度×6] + [head W 梯度]；gb = [GRU b 梯度×3] + [head b 梯度]

        前置：forward(cache_for_backward=True) 已调用（train_batch 保证）。
        """
        if self._head_acts is None or self._last_cache is None:
            raise RuntimeError("[gru] backward 前必须先 forward(cache_for_backward=True)")
        gW_h, gb_h = self.head.backward(self._head_acts, dout)
        # head 输入梯度：用 nnet.MLP.backward_input_grad（与参数反传同路径，
        # 含 ReLU/LN 反传），把 dout 反传到 concat 输入 [h, static]
        d_in = self.head.backward_input_grad(self._head_acts, dout)
        if self.n_static > 0:
            d_h = d_in[:, :self.gru_hidden]
        else:
            d_h = d_in
        gW_gru, gb_gru = self.gru.backward(self._last_cache, d_h)
        gW = [*gW_gru, *gW_h]
        gb = [*gb_gru, *gb_h]
        return gW, gb

    def apply_grad(self, gW, gb) -> None:
        """按前缀分派：前 6 个 W / 前 3 个 b 给 GRU，其余给 head。"""
        n_w = len(_GRU_W_NAMES)
        n_b = len(_GRU_B_NAMES)
        self.gru.apply_grad(list(gW[:n_w]), list(gb[:n_b]))
        self.head.apply_grad(list(gW[n_w:]), list(gb[n_b:]))

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "type": "gru",
            "in_dim": self.in_dim, "n_actions": self.n_actions,
            "state_window": self.state_window,
            "n_price_features": self.n_price_features,
            "gru_hidden": self.gru_hidden, "lr": self.lr,
            "gru": self.gru.to_dict(), "head": self.head.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, lr: Optional[float] = None) -> "GRUActor":
        from .nnet import MLP
        eff_lr = lr if lr is not None else float(data.get("lr", 1e-3))
        head_hidden: tuple[int, ...] = ()
        if "head" in data:
            dims = data["head"].get("dims") or []
            if len(dims) > 2:
                head_hidden = tuple(int(h) for h in dims[1:-1])
        actor = cls(int(data["in_dim"]), int(data["n_actions"]),
                    state_window=int(data.get("state_window", 4)),
                    gru_hidden=int(data.get("gru_hidden", 32)),
                    head_hidden=head_hidden or (32, 32),
                    lr=eff_lr,
                    n_price_features=int(data.get("n_price_features",
                                                  N_PRICE_FEATURES)))
        actor.gru = GRUCell.from_dict(data["gru"], lr=eff_lr)
        actor.head = MLP.from_dict(data["head"], lr=eff_lr)
        return actor
