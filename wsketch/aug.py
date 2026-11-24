"""Batched GPU augmentation. Works on already-normalized 128-point strokes and
re-applies the input contract (resample + normalize) afterwards."""

from __future__ import annotations

import math

import torch

from . import CLASS_INDEX, CLASSES

# For real (Quick, Draw!) strokes: how far a class may be rotated / stretched
# before it stops being that class. Synthetic strokes already carry their own
# rotation and aspect, so they only get a light touch.
_ROT = {"rectangle": 20, "diamond": 12, "cloud": 12, "heart": 15}          # default: any angle
_STRETCH = {"ellipse": 3.0, "rectangle": 3.5, "triangle": 1.8, "diamond": 1.7, "hexagon": 1.25,
            "star": 1.2, "cloud": 1.5, "scribble": 2.0, "none": 1.5, "line": 1.0, "arrow": 1.0}


def _table(d: dict, default: float) -> torch.Tensor:
    return torch.tensor([d.get(c, default) for c in CLASSES])


ROT = _table(_ROT, 180.0)
STRETCH = _table(_STRETCH, 1.0)


def resample(P: torch.Tensor, M: torch.Tensor):
    """Arc-length resample a batch [B, N, 2] (and a per-point channel [B, N])."""
    B, N, _ = P.shape
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(B, 1, device=P.device, dtype=P.dtype), seg.cumsum(dim=1)], dim=1)
    t = torch.linspace(0, 1, N, device=P.device, dtype=P.dtype)[None] * s[:, -1:]
    i = torch.searchsorted(s.contiguous(), t.contiguous(), right=True).clamp(1, N - 1)
    s0, s1 = s.gather(1, i - 1), s.gather(1, i)
    w = ((t - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0, 1)
    g = lambda a, j: a.gather(1, j[..., None].expand(-1, -1, 2))
    out = g(P, i - 1) + (g(P, i) - g(P, i - 1)) * w[..., None]
    return out, M.gather(1, i - 1) + (M.gather(1, i) - M.gather(1, i - 1)) * w


def normalize(P: torch.Tensor) -> torch.Tensor:
    lo, hi = P.amin(dim=1, keepdim=True), P.amax(dim=1, keepdim=True)
    return (P - (lo + hi) / 2) / (hi - lo).amax(dim=-1, keepdim=True).clamp_min(1e-6)


def augment(P: torch.Tensor, M: torch.Tensor, y: torch.Tensor, real: torch.Tensor):
    """P [B, N, 2] float, M [B, N] float (arrowhead), y [B] class, real [B] bool."""
    B, dev = P.shape[0], P.device
    rnd = lambda *shape: torch.rand(*shape, device=dev)

    # stretch along x or y
    k = torch.where(real, STRETCH.to(dev)[y], torch.full_like(y, 1.1, dtype=torch.float))
    ratio = torch.exp(rnd(B) * torch.log(k))
    sx = torch.where(rnd(B) < 0.5, ratio, torch.ones_like(ratio))
    sy = torch.where(sx == 1, ratio, torch.ones_like(ratio))
    P = P / torch.stack([sx, sy], dim=-1)[:, None]

    # rotate
    mx = torch.where(real, ROT.to(dev)[y], torch.full_like(y, 5.0, dtype=torch.float))
    a = (rnd(B) * 2 - 1) * mx * math.pi / 180
    c, s = torch.cos(a), torch.sin(a)
    P = torch.stack([P[..., 0] * c[:, None] - P[..., 1] * s[:, None],
                     P[..., 0] * s[:, None] + P[..., 1] * c[:, None]], dim=-1)

    # mirror left-right; draw in the opposite direction
    P = torch.where((rnd(B) < 0.5)[:, None, None], P * torch.tensor([-1.0, 1.0], device=dev), P)
    rev = rnd(B) < 0.5
    P = torch.where(rev[:, None, None], P.flip(1), P)
    M = torch.where(rev[:, None], M.flip(1), M)

    # a little jitter; and a small shape on a real screen snaps to whole pixels
    P = P + torch.randn_like(P) * (rnd(B, 1, 1) * 0.004)
    q = 0.003 + rnd(B, 1, 1) * 0.017
    P = torch.where(rnd(B, 1, 1) < 0.3, torch.round(P / q) * q, P)

    P, M = resample(P, M)
    return normalize(P), M
