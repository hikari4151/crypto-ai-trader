"""纯 numpy 小型神经网络：多层感知机，前向传播 + 反向传播（策略梯度训练用）。

无 torch/sklearn 依赖，符合项目轻量哲学。支持：
- 任意层数 MLP（ReLU 隐藏层）
- Actor 输出 softmax 概率；Critic 输出标量价值
- 用 Adam 优化器做梯度下降（自带动量与自适应学习率）
- save/load 权重为 JSON（与 numpy 数组互转），可持久化训练好的智能体
"""
import json
import math
import random
from typing import Any, Optional

import numpy as np


class MLP:
    """全连接神经网络。dims=[in, h1, h2, ..., out]"""

    def __init__(self, dims: list[int], seed: Optional[int] = None,
                 lr: float = 1e-3) -> None:
        self.dims = list(dims)
        self.lr = lr
        rng = np.random.default_rng(seed if seed is not None else random.randint(0, 2**31))
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self._m_w: list[np.ndarray] = []
        self._v_w: list[np.ndarray] = []
        self._m_b: list[np.ndarray] = []
        self._v_b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            # Xavier 初始化
            scale = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            w = rng.normal(0.0, scale, (dims[i], dims[i + 1]))
            bb = np.zeros(dims[i + 1], dtype=float)
            self.W.append(w)
            self.b.append(bb)
            self._m_w.append(np.zeros_like(w))
            self._v_w.append(np.zeros_like(w))
            self._m_b.append(np.zeros_like(bb))
            self._v_b.append(np.zeros_like(bb))

    # ---------- 前向 ----------
    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """返回各层激活值 [h0=x, h1, ..., h_{L-1}=输出]。"""
        acts = [np.asarray(x, dtype=float)]
        for i in range(len(self.W)):
            z = acts[-1] @ self.W[i] + self.b[i]
            if i == len(self.W) - 1:
                acts.append(z)          # 输出层不激活（softmax/线性在外面处理）
            else:
                acts.append(np.maximum(z, 0.0))  # ReLU
        return acts

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Actor 输出：softmax 概率（n_actions）。x 可 1D 或 (batch, in)。"""
        logits = self.forward(x)[-1]
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def predict_value(self, x: np.ndarray) -> np.ndarray:
        """Critic 输出：标量价值。"""
        return self.forward(x)[-1].reshape(-1)

    # ---------- 反向传播 ----------
    def backward(self, acts: list[np.ndarray], dout: np.ndarray) -> tuple[list, list]:
        """反向传播：dout 为输出层梯度，返回 (grad_W, grad_b)。"""
        gW: list[np.ndarray] = []
        gb: list[np.ndarray] = []
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW.append(acts[i].T @ d)
            gb.append(d.sum(axis=0))
            if i > 0:
                dh = d @ self.W[i].T
                d = dh * (acts[i] > 0)  # ReLU 导数
        return list(reversed(gW)), list(reversed(gb))

    def step(self, gW: list[np.ndarray], gb: list[np.ndarray]) -> None:
        """Adam 更新。"""
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self._m_w[i] = b1 * self._m_w[i] + (1 - b1) * gW[i]
            self._v_w[i] = b2 * self._v_w[i] + (1 - b2) * (gW[i] ** 2)
            self._m_b[i] = b1 * self._m_b[i] + (1 - b1) * gb[i]
            self._v_b[i] = b2 * self._v_b[i] + (1 - b2) * (gb[i] ** 2)
            mh = self._m_w[i] / (1 - b1 ** self._t)
            vh = self._v_w[i] / (1 - b2 ** self._t)
            self.W[i] -= self.lr * mh / (np.sqrt(vh) + eps)
            mh_b = self._m_b[i] / (1 - b1 ** self._t)
            vh_b = self._v_b[i] / (1 - b2 ** self._t)
            self.b[i] -= self.lr * mh_b / (np.sqrt(vh_b) + eps)

    @property
    def _t(self) -> int:
        return max(1, int(getattr(self, "_step_count", 0)))

    def apply_grad(self, gW: list[np.ndarray], gb: list[np.ndarray]) -> None:
        """外部更新计数 + Adam step。"""
        self._step_count = getattr(self, "_step_count", 0) + 1
        self.step(gW, gb)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "dims": self.dims,
            "W": [w.tolist() for w in self.W],
            "b": [bb.tolist() for bb in self.b],
            # Adam 状态持久化：续训时保留动量/二阶矩，避免首轮更新失真
            "m_w": [m.tolist() for m in self._m_w],
            "v_w": [v.tolist() for v in self._v_w],
            "m_b": [m.tolist() for m in self._m_b],
            "v_b": [v.tolist() for v in self._v_b],
            "step_count": int(getattr(self, "_step_count", 0)),
        }

    @classmethod
    def from_dict(cls, data: dict, lr: float = 1e-3) -> "MLP":
        net = cls(data["dims"], lr=lr)
        net.W = [np.asarray(w, dtype=float) for w in data["W"]]
        net.b = [np.asarray(bb, dtype=float) for bb in data["b"]]
        # 旧格式模型（无 Adam 状态）兼容：零初始化即可
        net._m_w = [np.asarray(m, dtype=float) for m in data.get("m_w", [np.zeros_like(w) for w in net.W])]
        net._v_w = [np.asarray(v, dtype=float) for v in data.get("v_w", [np.zeros_like(w) for w in net.W])]
        net._m_b = [np.asarray(m, dtype=float) for m in data.get("m_b", [np.zeros_like(bb) for bb in net.b])]
        net._v_b = [np.asarray(v, dtype=float) for v in data.get("v_b", [np.zeros_like(bb) for bb in net.b])]
        net._step_count = int(data.get("step_count", 0))
        return net

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str, lr: float = 1e-3) -> "MLP":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), lr=lr)
