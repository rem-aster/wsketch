"""The input contract: raw stroke -> the 128x2 tensor the model sees.

1. Resample to N points evenly spaced along the stroke (first and last kept).
2. Move the bounding-box centre to the origin.
3. Scale so the longer bounding-box side is 1.

Coordinates are canvas coordinates (y grows downward). The same three steps
must be reproduced by any client (JS, Swift, ...) before running the model.
"""

from __future__ import annotations

import numpy as np

from . import N_POINTS


def resample(pts: np.ndarray, n: int = N_POINTS, extra: np.ndarray | None = None):
    """Arc-length resample (m, 2) -> (n, 2). `extra` (m,) is interpolated alongside."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 1e-9:  # a dot
        out = np.repeat(pts[:1], n, axis=0)
        return (out, np.zeros(n)) if extra is not None else out
    t = np.linspace(0.0, total, n)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)
    if extra is not None:
        return out, np.interp(t, s, np.asarray(extra, dtype=np.float64))
    return out


def normalize(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = max((hi - lo).max(), 1e-9)
    return (pts - (lo + hi) / 2) / size


def preprocess(pts: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Raw stroke points (m, 2) -> model input (n, 2) float32."""
    return normalize(resample(np.asarray(pts)[:, :2], n)).astype(np.float32)
