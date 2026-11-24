"""The stroke classifier: 128 points in, class scores and an arrowhead mask out.

A small conv stem followed by a few self-attention blocks. Everything the
network needs beyond raw x, y (directions, turning, distance to the stroke's
ends) is computed inside the graph, so the exported model takes the plain
[B, 128, 2] tensor described in preprocess.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CLASSES, N_POINTS


def point_features(p: torch.Tensor) -> torch.Tensor:
    """[B, N, 2] -> [B, N, 13] hand-made per-point features."""
    d = p[:, 1:] - p[:, :-1]
    step = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    u = d / step
    nxt = torch.cat([u, u[:, -1:]], dim=1)   # direction to the next point
    prv = torch.cat([u[:, :1], u], dim=1)    # direction from the previous point
    cos = (nxt * prv).sum(-1, keepdim=True)
    sin = (prv[..., :1] * nxt[..., 1:] - prv[..., 1:] * nxt[..., :1])
    # points are evenly spaced, so the mean step is the path length / (N - 1)
    length = torch.log(step.mean(dim=1, keepdim=True) * (p.shape[1] - 1)).expand(-1, p.shape[1], -1)
    return torch.cat([p, nxt, prv, cos, sin, length, p - p[:, :1], p - p[:, -1:]], dim=-1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if torch.onnx.is_in_onnx_export():  # spelled out, so the graph is plain MatMul + Softmax
            o = torch.softmax(q @ k.transpose(-1, -2) * (D // self.heads) ** -0.5, dim=-1) @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class Net(nn.Module):
    def __init__(self, d: int = 96, depth: int = 4, heads: int = 4, drop: float = 0.1,
                 n_classes: int = len(CLASSES)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(13, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, N_POINTS, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.cls = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_classes))
        self.head = nn.Linear(d, 1)

    def forward(self, points: torch.Tensor):
        """Returns class logits [B, n_classes] and arrowhead logits [B, 128]."""
        x = self.stem(point_features(points).transpose(1, 2)).transpose(1, 2) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return self.cls(pooled), self.head(x).squeeze(-1)


class Exported(nn.Module):
    """What ships: probabilities instead of logits."""

    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def forward(self, points: torch.Tensor):
        logits, head = self.net(points)
        return F.softmax(logits, dim=-1), torch.sigmoid(head)
