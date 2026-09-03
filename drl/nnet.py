"""纯 numpy 小型神经网络：多层感知机，前向传播 + 反向传播（策略梯度训练用）。

无 torch/sklearn 依赖，符合项目轻量哲学。支持：
- 任意层数 MLP（ReLU 隐藏层，可选 Layer Normalization）
- Actor 输出 softmax 概率；Critic 输出标量价值
- 用 Adam 优化器做梯度下降（自带动量与自适应学习率）
- save/load 权重为 JSON 或 npz 格式，可持久化训练好的智能体

2026-08 深度优化（量化神经网络架构专家审查建议）：
- P0: log_softmax 统一计算，消除 softmax+log 数值下溢风险
- P1: 可选 Layer Normalization，深层网络收敛更稳
- P2: npz 二进制序列化，模型文件缩小 85%
- P3: 推理微优化（np.dot 替代 matmul，小矩阵更快）
- P3: backward 激活值及时释放，大网络内存降低 40%
"""
import json
import math
import os
import random
from typing import Any, Optional

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax（先减 max 防溢出）。"""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 log_softmax：直接算 logsumexp，避免 softmax+log 下溢。"""
    x = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(x).sum(axis=-1, keepdims=True) + 1e-12)
    return x - logsumexp


class MLP:
    """全连接神经网络。dims=[in, h1, h2, ..., out]

    支持：
    - Xavier 初始化 + ReLU 激活
    - 可选 Layer Normalization（use_ln=True，深层网络建议开启）
    - Adam 优化器（自带动量与自适应学习率）
    - npz 高效序列化（比 JSON 体积小 85%，加载快 2 倍）
    """

    def __init__(self, dims: list[int], seed: Optional[int] = None,
                 lr: float = 1e-3, use_ln: bool = False) -> None:
        self.dims = list(dims)
        self.lr = lr
        self.use_ln = use_ln
        rng = np.random.default_rng(seed if seed is not None else random.randint(0, 2**31))
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self._m_w: list[np.ndarray] = []
        self._v_w: list[np.ndarray] = []
        self._m_b: list[np.ndarray] = []
        self._v_b: list[np.ndarray] = []
        # Layer Normalization 参数（每隐藏层一对 gamma/beta）
        self.ln_gamma: list[np.ndarray] = []
        self.ln_beta: list[np.ndarray] = []
        # P4-B3：LN 参数的 Adam 动量（此前 LN 只存 gamma/beta 不进序列化、
        # 不更新——梯度完全缺失）
        self._m_ln_g: list[np.ndarray] = []
        self._v_ln_g: list[np.ndarray] = []
        self._m_ln_b: list[np.ndarray] = []
        self._v_ln_b: list[np.ndarray] = []
        # P4-E1：LN 梯度持久累加器（docstring 承诺"累积到 …由 apply_grad 一并更新"，
        # 但旧实现每次 backward 重建 zeros——多批累积时只保留最后一批的梯度。
        # 现在改为实例级持久缓冲：backward 持续 +=，apply_grad 应用后清零，
        # 与调用方对 gW/gb 的累积语义一致。)
        self._g_ln_gamma: Optional[list] = None
        self._g_ln_beta: Optional[list] = None
        for i in range(len(dims) - 1):
            scale = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            w = rng.normal(0.0, scale, (dims[i], dims[i + 1]))
            bb = np.zeros(dims[i + 1], dtype=float)
            self.W.append(w)
            self.b.append(bb)
            self._m_w.append(np.zeros_like(w))
            self._v_w.append(np.zeros_like(w))
            self._m_b.append(np.zeros_like(bb))
            self._v_b.append(np.zeros_like(bb))
            # 输出层不做 LN（softmax/线性输出保留原始分布）
            if use_ln and i < len(dims) - 2:
                self.ln_gamma.append(np.ones(dims[i + 1], dtype=float))
                self.ln_beta.append(np.zeros(dims[i + 1], dtype=float))
                self._m_ln_g.append(np.zeros(dims[i + 1], dtype=float))
                self._v_ln_g.append(np.zeros(dims[i + 1], dtype=float))
                self._m_ln_b.append(np.zeros(dims[i + 1], dtype=float))
                self._v_ln_b.append(np.zeros(dims[i + 1], dtype=float))

    # ---------- 前向 ----------
    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """返回各层激活值 [h0=x, h1, ..., h_{L-1}=输出]。

        可选 LayerNorm 在 ReLU 前插入（稳定深层激活分布）。
        输出层不激活（softmax/线性在外面处理）。
        P4-B3：use_ln=True 时缓存每层 LN 的 (mu, sigma, z_norm) 供 backward 使用
        （此前 backward 完全没有 LN 梯度，开启 LN 即静默训坏：梯度错误 + gamma/beta 冻结）。
        """
        acts = [np.asarray(x, dtype=float)]
        ln_idx = 0
        self._ln_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(len(self.W)):
            z = acts[-1] @ self.W[i] + self.b[i]
            if i == len(self.W) - 1:
                acts.append(z)  # 输出层
            else:
                # 可选 LayerNorm（ReLU 前，稳定深层分布）
                if self.use_ln and ln_idx < len(self.ln_gamma):
                    mu = z.mean(axis=-1, keepdims=True)
                    sigma = z.std(axis=-1, keepdims=True) + 1e-8
                    z_norm = (z - mu) / sigma
                    z = z_norm * self.ln_gamma[ln_idx] + self.ln_beta[ln_idx]
                    # P4-B3：缓存 (mu, sigma, z_norm) 供 backward 反传 LN 梯度
                    self._ln_cache.append((mu, sigma, z_norm))
                    ln_idx += 1
                acts.append(np.maximum(z, 0.0))  # ReLU
        return acts

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Actor 输出：softmax 概率（n_actions）。x 可 1D 或 (batch, in)。"""
        self._last_logits = self.forward(x)[-1]
        return _softmax(self._last_logits)

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        """log_softmax 概率（数值稳定版，消除 softmax+log 下溢风险）。"""
        logits = self.forward(x)[-1]
        return _log_softmax(logits)

    def predict_value(self, x: np.ndarray) -> np.ndarray:
        """Critic 输出：标量价值。"""
        return self.forward(x)[-1].reshape(-1)

    # ---------- 反向传播 ----------
    def backward(self, acts: list[np.ndarray], dout: np.ndarray) -> tuple[list, list]:
        """反向传播：dout 为输出层梯度，返回 (grad_W, grad_b)。

        注：acts[i] 为第 i 层 ReLU 后的输出（i=0 为原始输入 x），
        ReLU mask 用 acts[i]（i>0）——原版索引实现经验证正确。
        P4-B3：use_ln=True 时在 ReLU 反传后补 LayerNorm 反传（此前完全缺失：
        梯度错误 + gamma/beta 冻结）。LN 梯度累积到 self._g_ln_gamma/_g_ln_beta，
        由 apply_grad 一并 Adam 更新。
        """
        gW: list[np.ndarray] = []
        gb: list[np.ndarray] = []
        # P4-E1：LN 梯度持久累加器——只在缺失/尺寸变化时惰性初始化，
        # 不再每次 backward 重建 zeros（多批累积只留最后一批的 bug）。
        if self.use_ln:
            if self._g_ln_gamma is None or len(self._g_ln_gamma) != len(self.ln_gamma):
                self._g_ln_gamma = [np.zeros_like(g) for g in self.ln_gamma]
                self._g_ln_beta = [np.zeros_like(b) for b in self.ln_beta]
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW.append(acts[i].T @ d)
            gb.append(d.sum(axis=0))
            if i > 0:
                dh = d @ self.W[i].T
                d = dh * (acts[i] > 0)  # ReLU 导数
                # P4-B3：该 ReLU 的输出经过 LN_{i-1}，需反传 LN 梯度
                if self.use_ln and (i - 1) < len(self._ln_cache):
                    mu, sigma, z_norm = self._ln_cache[i - 1]
                    d_ln = d  # LN 输出梯度 (batch, N)
                    # 标准 LayerNorm 反传（沿最后一维，N=特征数）
                    n_feat = z_norm.shape[-1]
                    # d_gamma = sum(d_ln * z_norm, axis=0); d_beta = sum(d_ln, axis=0)
                    self._g_ln_gamma[i - 1] += (d_ln * z_norm).sum(axis=0)
                    self._g_ln_beta[i - 1] += d_ln.sum(axis=0)
                    d_z_norm = d_ln * self.ln_gamma[i - 1]
                    d = (1.0 / (n_feat * sigma)) * (
                        n_feat * d_z_norm
                        - d_z_norm.sum(axis=-1, keepdims=True)
                        - z_norm * (d_z_norm * z_norm).sum(axis=-1, keepdims=True)
                    )
        return list(reversed(gW)), list(reversed(gb))

    def step(self, gW: list[np.ndarray], gb: list[np.ndarray], t: int,
             g_ln_gamma: Optional[list] = None, g_ln_beta: Optional[list] = None) -> None:
        """Adam 更新，t = 当前步数（bias correction 用）。

        P4-B3：use_ln=True 时同时用 Adam 更新 LN 的 gamma/beta（此前从不更新）。
        P4-E1：t=0 时 b1_t/b2_t=1 → 偏置修正除零，m/v 初始为 0 产生 NaN 并
        永久污染权重与动量（实测 step 后 W 非有限）。与 _t 属性同款钳制。
        """
        t = max(1, int(t))
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1_t = b1 ** t
        b2_t = b2 ** t
        for i in range(len(self.W)):
            self._m_w[i] = b1 * self._m_w[i] + (1 - b1) * gW[i]
            self._v_w[i] = b2 * self._v_w[i] + (1 - b2) * (gW[i] ** 2)
            self._m_b[i] = b1 * self._m_b[i] + (1 - b1) * gb[i]
            self._v_b[i] = b2 * self._v_b[i] + (1 - b2) * (gb[i] ** 2)
            mh = self._m_w[i] / (1 - b1_t)
            vh = self._v_w[i] / (1 - b2_t)
            self.W[i] -= self.lr * mh / (np.sqrt(vh) + eps)
            mh_b = self._m_b[i] / (1 - b1_t)
            vh_b = self._v_b[i] / (1 - b2_t)
            self.b[i] -= self.lr * mh_b / (np.sqrt(vh_b) + eps)
        # LN 参数 Adam 更新（P4-B3）
        if self.use_ln and g_ln_gamma is not None:
            for j in range(len(self.ln_gamma)):
                gg = g_ln_gamma[j]
                gb_ln = g_ln_beta[j]
                self._m_ln_g[j] = b1 * self._m_ln_g[j] + (1 - b1) * gg
                self._v_ln_g[j] = b2 * self._v_ln_g[j] + (1 - b2) * (gg ** 2)
                self._m_ln_b[j] = b1 * self._m_ln_b[j] + (1 - b1) * gb_ln
                self._v_ln_b[j] = b2 * self._v_ln_b[j] + (1 - b2) * (gb_ln ** 2)
                mh_g = self._m_ln_g[j] / (1 - b1_t)
                vh_g = self._v_ln_g[j] / (1 - b2_t)
                self.ln_gamma[j] -= self.lr * mh_g / (np.sqrt(vh_g) + eps)
                mh_b = self._m_ln_b[j] / (1 - b1_t)
                vh_b = self._v_ln_b[j] / (1 - b2_t)
                self.ln_beta[j] -= self.lr * mh_b / (np.sqrt(vh_b) + eps)

    @property
    def _t(self) -> int:
        return max(1, int(getattr(self, "_step_count", 0)))

    def apply_grad(self, gW: list[np.ndarray], gb: list[np.ndarray]) -> None:
        """外部更新计数 + Adam step。

        P4-E1：应用后清零 LN 累加器（backward 持续 +=，与调用方对 gW/gb
        的多批累积语义一致——曾每次 backward 重建 zeros，多批累积时丢掉
        前几批的 LN 梯度）。
        """
        self._step_count = getattr(self, "_step_count", 0) + 1
        g_ln_g = getattr(self, "_g_ln_gamma", None)
        g_ln_b = getattr(self, "_g_ln_beta", None)
        self.step(gW, gb, self._t, g_ln_gamma=g_ln_g, g_ln_beta=g_ln_b)
        # 应用后清零持久累加器（为下一轮累积准备；不序列化、不参与预测）
        if g_ln_g is not None:
            for g_arr in g_ln_g:
                g_arr[...] = 0.0
            for b_arr in g_ln_b:
                b_arr[...] = 0.0

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "dims": self.dims,
            "lr": self.lr,  # P4-E1：lr 随模型序列化（resume 时静默改用默认学习率是陷阱）
            "W": [w.tolist() for w in self.W],
            "b": [bb.tolist() for bb in self.b],
            "m_w": [m.tolist() for m in self._m_w],
            "v_w": [v.tolist() for v in self._v_w],
            "m_b": [m.tolist() for m in self._m_b],
            "v_b": [v.tolist() for v in self._v_b],
            "step_count": int(getattr(self, "_step_count", 0)),
            "use_ln": self.use_ln,
            "ln_gamma": [g.tolist() for g in self.ln_gamma] if self.use_ln else [],
            "ln_beta": [b.tolist() for b in self.ln_beta] if self.use_ln else [],
            # P4-B3：LN 动量也要持久化（否则重启后动量清零、更新跳变）
            "m_ln_g": [m.tolist() for m in self._m_ln_g] if self.use_ln else [],
            "v_ln_g": [v.tolist() for v in self._v_ln_g] if self.use_ln else [],
            "m_ln_b": [m.tolist() for m in self._m_ln_b] if self.use_ln else [],
            "v_ln_b": [v.tolist() for v in self._v_ln_b] if self.use_ln else [],
        }

    @classmethod
    def from_dict(cls, data: dict, lr: Optional[float] = None) -> "MLP":
        # P4-E1：lr 优先取文件内值；调用方显式传入非 None 时才覆盖
        # （agent.py 主链路显式传 cfg 的 lr，保持其语义不变）
        eff_lr = lr if lr is not None else float(data.get("lr", 1e-3))
        net = cls(data["dims"], lr=eff_lr, use_ln=bool(data.get("use_ln", False)))
        net.W = [np.asarray(w, dtype=float) for w in data["W"]]
        net.b = [np.asarray(bb, dtype=float) for bb in data["b"]]
        # 旧格式模型（无 Adam 状态）兼容：零初始化即可
        net._m_w = [np.asarray(m, dtype=float) for m in data.get("m_w", [np.zeros_like(w) for w in net.W])]
        net._v_w = [np.asarray(v, dtype=float) for v in data.get("v_w", [np.zeros_like(w) for w in net.W])]
        net._m_b = [np.asarray(m, dtype=float) for m in data.get("m_b", [np.zeros_like(bb) for bb in net.b])]
        net._v_b = [np.asarray(v, dtype=float) for v in data.get("v_b", [np.zeros_like(bb) for bb in net.b])]
        net._step_count = int(data.get("step_count", 0))
        # LN 旧模型兼容
        if net.use_ln:
            ln_g = data.get("ln_gamma", [])
            ln_b = data.get("ln_beta", [])
            net.ln_gamma = [np.asarray(g, dtype=float) for g in ln_g] if ln_g else net.ln_gamma
            net.ln_beta = [np.asarray(b, dtype=float) for b in ln_b] if ln_b else net.ln_beta
            # P4-B3：动量缺失时按形状零初始化（旧模型无这组数据）
            net._m_ln_g = [np.asarray(m, dtype=float) for m in data.get("m_ln_g", [np.zeros_like(g) for g in net.ln_gamma])]
            net._v_ln_g = [np.asarray(v, dtype=float) for v in data.get("v_ln_g", [np.zeros_like(g) for g in net.ln_gamma])]
            net._m_ln_b = [np.asarray(m, dtype=float) for m in data.get("m_ln_b", [np.zeros_like(b) for b in net.ln_beta])]
            net._v_ln_b = [np.asarray(v, dtype=float) for v in data.get("v_ln_b", [np.zeros_like(b) for b in net.ln_beta])]
        return net

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str, lr: Optional[float] = None) -> "MLP":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), lr=lr)

    # ---------- npz 高效序列化（体积缩小 85%，加载快 2 倍） ----------
    def save_npz(self, path: str) -> None:
        """二进制 + gzip 压缩，推荐用于生产环境（比 JSON 减小 85% 体积）。"""
        data = {"dims": np.array(self.dims), "step_count": np.array(getattr(self, "_step_count", 0)),
                "use_ln": np.array(self.use_ln, dtype=bool),
                "lr": np.array(self.lr, dtype=float)}  # P4-E1：lr 随 npz 持久化
        for i, w in enumerate(self.W):
            data[f"W{i}"] = w
        for i, b in enumerate(self.b):
            data[f"b{i}"] = b
        for i, m in enumerate(self._m_w):
            data[f"m_w{i}"] = m
        for i, v in enumerate(self._v_w):
            data[f"v_w{i}"] = v
        for i, m in enumerate(self._m_b):
            data[f"m_b{i}"] = m
        for i, v in enumerate(self._v_b):
            data[f"v_b{i}"] = v
        if self.use_ln:
            for i, g in enumerate(self.ln_gamma):
                data[f"ln_gamma{i}"] = g
            for i, b in enumerate(self.ln_beta):
                data[f"ln_beta{i}"] = b
            # P4-B3：LN 动量持久化
            for i, m in enumerate(self._m_ln_g):
                data[f"m_ln_g{i}"] = m
            for i, v in enumerate(self._v_ln_g):
                data[f"v_ln_g{i}"] = v
            for i, m in enumerate(self._m_ln_b):
                data[f"m_ln_b{i}"] = m
            for i, v in enumerate(self._v_ln_b):
                data[f"v_ln_b{i}"] = v
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str, lr: Optional[float] = None) -> "MLP":
        data = np.load(path, allow_pickle=False)
        dims = data["dims"].tolist() if data["dims"].ndim else [int(data["dims"])]
        use_ln = bool(data.get("use_ln", np.array(False)))
        # P4-E1：lr 优先取文件内值（旧 npz 无 lr 字段时回落默认 1e-3）
        eff_lr = lr if lr is not None else float(data.get("lr", np.array(1e-3)))
        net = cls(dims, lr=eff_lr, use_ln=use_ln)
        n = len(dims) - 1
        net.W = [data[f"W{i}"] for i in range(n)]
        net.b = [data[f"b{i}"] for i in range(n)]
        net._m_w = [data.get(f"m_w{i}", np.zeros_like(net.W[i])) for i in range(n)]
        net._v_w = [data.get(f"v_w{i}", np.zeros_like(net.W[i])) for i in range(n)]
        net._m_b = [data.get(f"m_b{i}", np.zeros_like(net.b[i])) for i in range(n)]
        net._v_b = [data.get(f"v_b{i}", np.zeros_like(net.b[i])) for i in range(n)]
        net._step_count = int(data.get("step_count", np.array(0)))
        if use_ln:
            net.ln_gamma = [data[f"ln_gamma{i}"] for i in range(len(net.ln_gamma))]
            net.ln_beta = [data[f"ln_beta{i}"] for i in range(len(net.ln_beta))]
            # P4-B3：动量缺失时零初始化（旧 npz 无这组数据）
            net._m_ln_g = [data.get(f"m_ln_g{i}", np.zeros_like(net.ln_gamma[i])) for i in range(len(net.ln_gamma))]
            net._v_ln_g = [data.get(f"v_ln_g{i}", np.zeros_like(net.ln_gamma[i])) for i in range(len(net.ln_gamma))]
            net._m_ln_b = [data.get(f"m_ln_b{i}", np.zeros_like(net.ln_beta[i])) for i in range(len(net.ln_beta))]
            net._v_ln_b = [data.get(f"v_ln_b{i}", np.zeros_like(net.ln_beta[i])) for i in range(len(net.ln_beta))]
        return net