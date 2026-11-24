"""Turn the downloaded Quick, Draw! strokes into labelled training data.

Quick, Draw! labels say what the person was asked to draw, not what the stroke
looks like, so every category goes through simple geometric checks (is it
closed, does it go around once, how much of its bounding box does it fill)
and strokes that fail are dropped rather than trusted.

    python -m wsketch.real            # writes data/real.npz
    python -m wsketch.real sheets     # also renders kept / dropped contact sheets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import CLASS_INDEX, CLASSES, N_POINTS
from . import quickdraw as qd
from .preprocess import preprocess


def smooth(X: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(X, ((0, 0), (r, r), (0, 0)), mode="edge")
    return sum(k[i] * pad[:, i:i + X.shape[1]] for i in range(2 * r + 1))


def features(X: np.ndarray) -> dict[str, np.ndarray]:
    """Geometric measurements of normalized strokes X [B, 128, 2]."""
    X = X.astype(np.float64)
    seg = np.diff(X, axis=1)
    length = np.hypot(seg[..., 0], seg[..., 1]).sum(axis=1)
    chord = np.hypot(*(X[:, -1] - X[:, 0]).T)

    # how many times the stroke goes around its own centre
    c = X - X.mean(axis=1, keepdims=True)
    ang = np.unwrap(np.arctan2(c[..., 1], c[..., 0]), axis=1)
    wind = np.abs(ang[:, -1] - ang[:, 0]) / (2 * np.pi)

    # enclosed area as a fraction of the bounding box
    x, y = X[..., 0], X[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1))
    box = np.ptp(x, axis=1) * np.ptp(y, axis=1)

    # total turning of the smoothed stroke, in degrees
    S = smooth(X)
    d = np.diff(S, axis=1)
    th = np.unwrap(np.arctan2(d[..., 1], d[..., 0]), axis=1)
    turning = np.degrees(np.abs(np.diff(th, axis=1)).sum(axis=1))

    # sharp reversals: the direction 3 points back vs 3 points ahead differs by > 100 degrees
    w = 3
    a, b = X[:, w:-w] - X[:, :-2 * w], X[:, 2 * w:] - X[:, w:-w]
    cosang = (a * b).sum(-1) / np.maximum(np.hypot(a[..., 0], a[..., 1]) * np.hypot(b[..., 0], b[..., 1]), 1e-9)
    sharp = cosang < np.cos(np.radians(100))
    reversals = (sharp[:, 1:] & ~sharp[:, :-1]).sum(axis=1) + sharp[:, 0]
    # share of the stroke that lies between its first and last reversal
    first, last = sharp.argmax(axis=1), sharp.shape[1] - 1 - sharp[:, ::-1].argmax(axis=1)
    cover = np.where(sharp.any(axis=1), (last - first) / sharp.shape[1], 0.0)

    # a diamond has its four extreme points in the middle of the box sides
    idx = np.arange(len(X))
    ext = np.stack([np.abs(x[idx, y.argmin(1)]), np.abs(x[idx, y.argmax(1)]),
                    np.abs(y[idx, x.argmin(1)]), np.abs(y[idx, x.argmax(1)])], axis=1).max(axis=1)

    # how far the ends are from closing the loop; passing the start point still counts as closed
    q = X.shape[1] // 4
    d_end = np.hypot(*np.moveaxis(X[:, :q] - X[:, -1:], -1, 0)).min(axis=1)
    d_start = np.hypot(*np.moveaxis(X[:, -q:] - X[:, :1], -1, 0)).min(axis=1)
    gap = np.minimum(d_end, d_start)

    return dict(closure=chord, gap=gap, straight=chord / np.maximum(length, 1e-9), wind=wind,
                fill=area / np.maximum(box, 1e-9), turning=turning, reversals=reversals, cover=cover,
                extremes=ext, length=length)


def sides(P: np.ndarray, eps: float = 0.05) -> int:
    """Number of sides of a closed stroke, by Douglas-Peucker simplification."""
    n = len(P)
    keep, stack = {0, n - 1}, [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg, rel = P[b] - P[a], P[a + 1:b] - P[a]
        L = np.hypot(*seg)
        d = np.hypot(rel[:, 0], rel[:, 1]) if L < 1e-9 else np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        i = int(d.argmax())
        if d[i] > eps:
            keep.add(a + 1 + i)
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    V = P[sorted(keep)]
    k = len(V) - 1
    if k >= 3:  # started mid-edge or overshot: first and last segment are one side
        u, v = V[1] - V[0], V[-1] - V[-2]
        if np.dot(u, v) / max(np.hypot(*u) * np.hypot(*v), 1e-9) > np.cos(np.radians(30)):
            k -= 1
    return k


def once(f):  # closed, and goes around exactly once
    return (f["gap"] < 0.2) & (f["wind"] > 0.85) & (f["wind"] < 1.4)


def partial(f):  # an unfinished outline: Quick, Draw! stops the drawing once it is recognized
    return (f["gap"] > 0.35) & (f["wind"] > 0.35) & (f["wind"] < 0.8) & (f["turning"] > 200)


def polygon(f):
    return once(f) & (f["fill"] > 0.6) & (f["fill"] < 0.9)


def zigzag(f):  # back and forth, sharply, along most of the stroke: a scratch-out
    return (f["reversals"] >= 5) & (f["cover"] >= 0.6)


def unclear(f):  # too few reversals to call it a scratch-out, too many to call it anything else
    return (f["reversals"] >= 4) & ~zigzag(f)


# Categories whose strokes are often plain zigzags whatever the player was asked to draw.
# The same geometry must get the same label, so they all share one rule.
SCRATCHY = [("scribble", zigzag), ("drop", unclear), ("none", lambda f: f["turning"] > 220)]

# category -> list of (class name, rule). First matching rule wins; no match (or "drop") = dropped.
RULES = {
    "line": [("line", lambda f: f["straight"] > 0.85)],
    "circle": [("ellipse", lambda f: once(f) & (f["fill"] > 0.6) & (f["fill"] < 0.92))],
    "square": [("rectangle", lambda f: once(f) & (f["fill"] > 0.78)), ("none", partial)],
    "triangle": [("triangle", lambda f: once(f) & (f["fill"] > 0.35) & (f["fill"] < 0.65)), ("none", partial)],
    "diamond": [("diamond", lambda f: once(f) & (f["fill"] > 0.38) & (f["fill"] < 0.63) & (f["extremes"] < 0.2)),
                ("none", partial)],
    # people asked for a hexagon or an octagon draw anything from 4 to 9 sides: count them
    "hexagon": [("hexagon", lambda f: polygon(f) & (f["sides"] == 6)),
                ("pentagon", lambda f: polygon(f) & (f["sides"] == 5)), ("none", partial)],
    "octagon": [("hexagon", lambda f: polygon(f) & (f["sides"] == 6)),
                ("pentagon", lambda f: polygon(f) & (f["sides"] == 5)), ("none", partial)],
    "star": [("star", lambda f: (f["gap"] < 0.25) & (f["reversals"] >= 4)
              & (((f["wind"] > 0.85) & (f["wind"] < 1.4)) | ((f["wind"] > 1.75) & (f["wind"] < 2.4)))),
             ("none", partial)],
    "cloud": [("cloud", lambda f: once(f) & (f["turning"] > 500))],
    "zigzag": [("scribble", zigzag), ("drop", unclear), ("none", lambda f: f["reversals"] <= 3)],
    "squiggle": [("scribble", zigzag), ("drop", unclear), ("none", lambda f: f["turning"] > 540)],
    "ocean": SCRATCHY, "lightning": SCRATCHY, "hurricane": SCRATCHY,
}
NEG_RULE = [("none", lambda f: (f["turning"] > 220))]


def load_category(cat: str):
    strokes = qd.load(cat)
    X = np.stack([preprocess(s) for s in strokes])
    f = features(X)
    if cat in ("hexagon", "octagon"):
        f["sides"] = np.array([sides(p) for p in X.astype(np.float64)])
    label = np.full(len(X), -1, np.int16)
    for name, rule in RULES.get(cat, NEG_RULE):
        label[(label == -1) & rule(f)] = CLASS_INDEX.get(name, -2)  # -2: deliberately dropped
    return X, label, f


def build(sheets: bool = False, out: str = "data/real.npz"):
    X_all, Y_all, S_all = [], [], []
    cats = qd.SHAPES + qd.NEGATIVES
    for si, cat in enumerate(cats):
        try:
            X, label, f = load_category(cat)
        except FileNotFoundError:
            print(f"{cat:14s} missing, skipped")
            continue
        keep = label >= 0
        print(f"{cat:14s} {len(X):7d} strokes, kept {keep.mean() * 100:5.1f}%  "
              + " ".join(f"{CLASSES[c]}={int((label == c).sum())}" for c in np.unique(label[keep])), flush=True)
        X_all.append(X[keep].astype(np.float16)), Y_all.append(label[keep].astype(np.uint8))
        S_all.append(np.full(keep.sum(), si, np.uint8))
        if sheets:
            from .viz import sheet

            rng = np.random.default_rng(0)
            pick = lambda m: X[rng.choice(np.flatnonzero(m), min(128, m.sum()), replace=False)] if m.any() else []
            name = cat.replace(" ", "_")
            for c in np.unique(label):  # one sheet per assigned class, "dropped" for the rest
                tag = CLASSES[c] if c >= 0 else "dropped"
                sheet(list(pick(label == c)), f"runs/viz/real_{name}_{tag}.png")
    # Corrections collected in the web demo ("Скачать примеры"): drop the JSON files into data/user/.
    user = [c for f in sorted(Path("data/user").glob("*.json")) for c in json.loads(f.read_text(encoding="utf-8"))]
    user = [c for c in user if c["label"] in CLASS_INDEX and len(c["points"]) >= 2]
    if user:
        X_all.append(np.stack([preprocess(np.array(c["points"], dtype=np.float64)) for c in user]).astype(np.float16))
        Y_all.append(np.array([CLASS_INDEX[c["label"]] for c in user], np.uint8))
        S_all.append(np.full(len(user), len(cats), np.uint8))
        print(f"{'your examples':14s} {len(user):7d} strokes")
    cats = cats + ["user"]
    np.savez(out, X=np.concatenate(X_all), Y=np.concatenate(Y_all), S=np.concatenate(S_all), cats=np.array(cats))
    Y = np.concatenate(Y_all)
    print("wrote", out, {k: int((Y == i).sum()) for k, i in CLASS_INDEX.items() if (Y == i).any()})


if __name__ == "__main__":
    build(sheets="sheets" in sys.argv)
