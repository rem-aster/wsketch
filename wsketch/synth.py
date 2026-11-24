"""Synthetic single-stroke shapes with a hand-drawing noise model.

Every generator returns a dense path in "math" coordinates (y up, size ~1)
plus an optional per-point arrowhead mask. `sample()` then applies the hand
model, flips to canvas coordinates and runs the input contract.

    python -m wsketch.synth            # writes data/synth.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import CLASS_INDEX, CLASSES, N_POINTS
from .preprocess import normalize, resample

TAU = 2 * np.pi


# ----------------------------------------------------------------- helpers

def rot(P: np.ndarray, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return P @ np.array([[c, s], [-s, c]])


def rot_any(rng, p_canon: float = 0.5, sd: float = 6.0) -> float:
    """Any angle, but often close to the canonical orientation."""
    return rng.normal(0, sd) if rng.random() < p_canon else rng.uniform(0, 360)


def rot_small(rng, mx: float) -> float:
    return rng.normal(0, mx / 5) if rng.random() < 0.7 else rng.uniform(-mx, mx)


def logu(rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def size_of(P: np.ndarray) -> float:
    return float(max(np.ptp(P, axis=0).max(), 1e-6))


def length_of(P: np.ndarray) -> float:
    return float(np.hypot(*np.diff(P, axis=0).T).sum())


def seg(a, b, k: int = 24) -> np.ndarray:
    t = np.linspace(0, 1, k)[:, None]
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def polyline(V, k: int = 24) -> np.ndarray:
    V = np.asarray(V, float)
    return np.concatenate([seg(V[i], V[i + 1], k)[:-1] for i in range(len(V) - 1)] + [V[-1:]])


def poly_loop(V, rng, bow: float = 0.02, m: int = 512):
    """Closed polygon -> uniformly sampled loop (m, 2) and corner positions in [0, 1)."""
    V = np.asarray(V, float)
    n, k = len(V), 24
    t = np.linspace(0, 1, k, endpoint=False)[:, None]
    parts = []
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        d = b - a
        nrm = np.array([-d[1], d[0]])
        parts.append(a + d * t + nrm * rng.normal(0, bow) * 4 * t * (1 - t))
    loop = np.concatenate(parts)
    closed = np.vstack([loop, loop[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    corners = s[np.arange(n) * k] / s[-1]
    return resample(closed, m + 1)[:-1], corners


def curve_loop(P: np.ndarray, m: int = 512) -> np.ndarray:
    """Closed dense curve (no repeated end point) -> uniformly sampled loop."""
    return resample(np.vstack([P, P[:1]]), m + 1)[:-1]


def trace(loop, rng, corners=None, extent=None, start=None, p_corner: float = 0.6):
    """Walk around a closed loop the way a hand does: random start, random
    direction, stopping a bit short of or a bit past the starting point."""
    m = len(loop)
    if start is None:
        if corners is not None and rng.random() < p_corner:
            start = rng.choice(corners) + rng.normal(0, 0.012)
        else:
            start = rng.random()
    if extent is None:
        extent = float(np.clip(1 + rng.normal(0, 0.035), 0.93, 1.12))
    sign = 1 if rng.random() < 0.5 else -1
    idx = (int(round(start * m)) + sign * np.arange(int(round(extent * m)) + 1)) % m
    return loop[idx]


def gauss_smooth(P: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.35:
        return P
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(P, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(P.shape[1])], axis=1)


def hand(P, mask, rng, smooth=(0.002, 0.02), wobble=(0.004, 0.022), drift=0.0, m: int = 384):
    """Turn an ideal path into something a hand could have drawn."""
    if mask is None:
        mask = np.zeros(len(P))
    P, mask = resample(P, m, mask)
    size, L = size_of(P), max(length_of(P), 1e-6)
    step = L / (m - 1)

    # fast drawing rounds the corners
    P = gauss_smooth(P, size * logu(rng, *smooth) / step)

    # slow wobble: a few sinusoids along the arc length
    s = np.arange(m) * step
    w = np.zeros((m, 2))
    for _ in range(5):
        lam = size * logu(rng, 0.3, 2.5)
        w += lam * rng.normal(0, 1, 2) * np.sin(TAU * s[:, None] / lam + rng.uniform(0, TAU, 2))
    w *= size * logu(rng, *wobble) / max(w.std(), 1e-9)
    P = P + w

    # the end does not come back exactly where the start was
    if drift > 0:
        a = rng.uniform(0, TAU)
        v = np.array([np.cos(a), np.sin(a)]) * rng.uniform(0, drift) * size
        P = P + np.linspace(0, 1, m)[:, None] * v

    # sensor jitter
    P = P + rng.normal(0, rng.uniform(0, 0.003) * size, P.shape)

    # pen-down / pen-up hooks
    for end in (0, -1):
        if rng.random() < 0.1:
            tan = P[1] - P[0] if end == 0 else P[-1] - P[-2]
            tan = tan / max(np.hypot(*tan), 1e-9)
            h = rot(tan[None], rng.choice([-1, 1]) * rng.uniform(60, 150))[0] * rng.uniform(0.01, 0.06) * size
            hook = np.stack([P[end] + h * 0.5, P[end] + h])
            if end == 0:
                P = np.vstack([hook[::-1], P])
                mask = np.concatenate([[mask[0]] * 2, mask])
            else:
                P = np.vstack([P, hook])
                mask = np.concatenate([mask, [mask[-1]] * 2])
    return P, mask


def regular(n: int, rng, rj: float = 0.06, aj: float = 4.0, phase: float = 90.0) -> np.ndarray:
    a = np.deg2rad(phase + np.arange(n) * 360 / n + rng.normal(0, aj, n))
    r = 0.5 * (1 + rng.normal(0, rj, n))
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


# ----------------------------------------------------------- closed shapes
# Each returns (loop, corners, start) with rotation and aspect already applied.

def loop_rectangle(rng):
    h = 1 / logu(rng, 1, 4)
    w, h = (1, h) if rng.random() < 0.6 else (h, 1)
    V = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    V = V + rng.normal(0, 0.035 * min(w, h), V.shape)
    V[:, 0] += V[:, 1] * rng.normal(0, 0.05)  # slight parallelogram
    loop, c = poly_loop(V, rng, bow=0.02)
    return rot(loop, rot_small(rng, 22)), c, None


def loop_ellipse(rng):
    t = np.linspace(0, TAU, 400, endpoint=False)
    r = 1 + sum(rng.normal(0, 0.03) * np.cos(k * t + rng.uniform(0, TAU)) for k in (2, 3, 4))
    P = np.stack([0.5 * r * np.cos(t), 0.5 / logu(rng, 1, 3.5) * r * np.sin(t)], axis=1)
    return rot(curve_loop(P), rng.uniform(0, 360)), None, None


def loop_triangle(rng):
    u = rng.random()
    if u < 0.4:  # isosceles-ish, upright
        hgt = rng.uniform(0.5, 1.4)
        V = np.array([[-0.5, 0], [0.5, 0], [rng.normal(0, 0.12), hgt]])
    elif u < 0.55:  # right-angled
        V = np.array([[0, 0], [1, 0], [0, rng.uniform(0.5, 1.4)]])
    else:  # anything not too flat
        while True:
            a = np.sort(rng.uniform(0, TAU, 3))
            V = np.stack([np.cos(a), np.sin(a)], axis=1) * 0.5
            e = np.roll(V, -1, 0) - V
            ang = [np.degrees(np.arccos(np.clip(np.dot(-e[i - 1], e[i]) /
                   (np.hypot(*e[i - 1]) * np.hypot(*e[i])), -1, 1))) for i in range(3)]
            if min(ang) > 25:
                break
    V = V + rng.normal(0, 0.03, V.shape)
    loop, c = poly_loop(V, rng, bow=0.02)
    deg = rng.choice([0, 180], p=[0.8, 0.2]) + rng.normal(0, 6) if rng.random() < 0.5 else rng.uniform(0, 360)
    return rot(loop, deg), c, None


def loop_diamond(rng):
    b = 0.5 * logu(rng, 0.55, 1.9)
    V = np.array([[0.5, 0], [0, b], [-0.5, 0], [0, -b]])
    V = V + rng.normal(0, 0.03 * min(1, 2 * b), V.shape)
    loop, c = poly_loop(V, rng, bow=0.02)
    return rot(loop, rot_small(rng, 13)), c, None


def loop_pentagon(rng):
    if rng.random() < 0.15:  # "house" pentagon
        V = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, rng.uniform(0, 0.2)],
                      [0, 0.5], [-0.5, rng.uniform(0, 0.2)]]) + rng.normal(0, 0.03, (5, 2))
        deg = rot_small(rng, 12)
    else:
        V = regular(5, rng, 0.06, 5)
        V[:, 0] *= rng.uniform(0.75, 1.3)
        deg = rng.normal(0, 8) if rng.random() < 0.6 else rng.uniform(0, 360)
    loop, c = poly_loop(V, rng, bow=0.015)
    return rot(loop, deg), c, None


def loop_hexagon(rng):
    V = regular(6, rng, 0.06, 4)
    V[:, 0] *= rng.uniform(0.75, 1.3)
    loop, c = poly_loop(V, rng, bow=0.015)
    deg = rng.choice([0, 30]) + rng.normal(0, 8) if rng.random() < 0.6 else rng.uniform(0, 360)
    return rot(loop, deg), c, None


def loop_star(rng):
    n = 5 if rng.random() < 0.75 else 6
    if n == 5 and rng.random() < 0.4:  # pentagram, drawn through the middle
        V = regular(5, rng, 0.06, 5)[[0, 2, 4, 1, 3]]
        loop, c = poly_loop(V, rng, bow=0.012)
    else:
        a = np.deg2rad(90 + np.arange(2 * n) * 180 / n + rng.normal(0, 4, 2 * n))
        inner = rng.uniform(0.36, 0.6) if n == 5 else rng.uniform(0.45, 0.65)
        r = 0.5 * np.where(np.arange(2 * n) % 2 == 0, 1, inner) * (1 + rng.normal(0, 0.06, 2 * n))
        loop, c = poly_loop(np.stack([r * np.cos(a), r * np.sin(a)], axis=1), rng, bow=0.012)
        c = c[::2] if rng.random() < 0.7 else c  # people start at a tip
    loop[:, 0] *= rng.uniform(0.85, 1.2)
    deg = rng.normal(0, 10) if rng.random() < 0.65 else rng.uniform(0, 360)
    return rot(loop, deg), c, None


def loop_heart(rng):
    t = np.linspace(0, TAU, 400, endpoint=False)
    s = np.sin(t)
    x = 16 * np.sign(s) * np.abs(s) ** rng.uniform(2.2, 3.4)
    y = (13 * rng.uniform(0.9, 1.1) * np.cos(t) - 5 * rng.uniform(0.7, 1.3) * np.cos(2 * t)
         - 2 * rng.uniform(0.5, 1.5) * np.cos(3 * t) - rng.uniform(0, 1.6) * np.cos(4 * t))
    P = np.stack([x * rng.uniform(0.8, 1.25), y], axis=1) / 32
    u = rng.random()  # start at the notch, at the bottom tip, or anywhere
    start = rng.normal(0, 0.015) if u < 0.5 else 0.5 + rng.normal(0, 0.015) if u < 0.8 else rng.random()
    return rot(curve_loop(P), rot_small(rng, 22)), None, start % 1


def loop_cloud(rng):
    a, b = 0.5, 0.5 / rng.uniform(1.2, 2.4)
    flat = rng.random() < 0.35
    n = int(rng.integers(3, 6)) if flat else int(rng.integers(5, 10))
    lo, hi = (np.deg2rad(-25), np.deg2rad(205)) if flat else (0.0, TAU)
    cuts = lo + (hi - lo) * (np.arange(n + 1) + np.r_[0, rng.normal(0, 0.15, n - 1), 0]) / n
    if not flat:
        cuts = cuts + rng.uniform(0, TAU)
    knots = np.stack([a * np.cos(cuts), b * np.sin(cuts)], axis=1)
    t = np.linspace(0, 1, 30, endpoint=False)[:, None]
    parts = []
    for i in range(n):
        p, q = knots[i], knots[i + 1]
        d = q - p
        c = np.hypot(*d)
        along, out = d / c, np.array([d[1], -d[0]]) / c  # outward for counter-clockwise knots
        phi = np.deg2rad(rng.uniform(140, 235))  # arc sweep of this bump
        R = c / (2 * np.sin(phi / 2))
        centre = (p + q) / 2 - out * R * np.cos(phi / 2)
        al = (t - 0.5) * phi
        parts.append(centre + R * (np.sin(al) * along + np.cos(al) * out))
    if flat:  # close along the bottom
        p, q = knots[-1], knots[0]
        d = q - p
        parts.append(p + d * t + np.array([d[1], -d[0]]) * rng.normal(0.03, 0.03) * 4 * t * (1 - t))
    P = np.concatenate(parts)
    return rot(curve_loop(P), rot_small(rng, 12)), None, None


LOOPS = {
    "rectangle": loop_rectangle, "ellipse": loop_ellipse, "triangle": loop_triangle,
    "diamond": loop_diamond, "pentagon": loop_pentagon, "hexagon": loop_hexagon,
    "star": loop_star, "heart": loop_heart, "cloud": loop_cloud,
}


def gen_closed(name: str, rng):
    loop, corners, start = LOOPS[name](rng)
    extent = None
    if name == "ellipse" and rng.random() < 0.05:
        extent = rng.uniform(1.12, 2.05)  # traced over more than once
    return trace(loop, rng, corners, extent, start), None


# ------------------------------------------------------------ open shapes

def open_curve(rng) -> np.ndarray:
    """A connector from (0,0) to (1,0): straight, or a gentle C / S curve."""
    u = rng.random()
    if u < 0.6:
        return seg([0, 0], [1, 0], 64)
    if u < 0.75:  # circular arc
        sweep = np.deg2rad(rng.uniform(30, 150))  # more than that is no longer a connector
        t = np.linspace(-sweep / 2, sweep / 2, 96)
        P = np.stack([np.sin(t), np.cos(t)], axis=1)
        P = (P - P[0]) / max(np.hypot(*(P[-1] - P[0])), 1e-6)
        return P * [1, rng.choice([-1, 1])]
    h1 = rng.uniform(0.05, 0.6) * rng.choice([-1, 1])
    h2 = h1 * rng.uniform(0.3, 1.2) if rng.random() < 0.6 else -h1 * rng.uniform(0.3, 0.8)
    c = np.array([[0, 0], [rng.uniform(0.15, 0.45), h1], [rng.uniform(0.55, 0.85), h2], [1, 0]])
    t = np.linspace(0, 1, 96)[:, None]
    return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3])


def line_angle(rng) -> float:
    if rng.random() < 0.4:
        return rng.choice([0, 45, 90, 135, 180, 225, 270, 315]) + rng.normal(0, 4)
    return rng.uniform(0, 360)


def gen_line(rng):
    return rot(open_curve(rng), line_angle(rng)), None


def arrow_head(tip, tan, L, rng) -> np.ndarray:
    """Points drawn after the pen reaches the tip, without lifting."""
    b = L * logu(rng, 0.1, 0.45)
    a1 = rng.uniform(18, 50)
    a2 = a1 * rng.uniform(0.7, 1.3)
    back = -tan
    B1 = tip + rot(back[None], a1)[0] * b * rng.uniform(0.75, 1.25)
    B2 = tip + rot(back[None], -a2)[0] * b * rng.uniform(0.75, 1.25)
    if rng.random() < 0.5:
        B1, B2 = B2, B1
    u = rng.random()
    if u < 0.55:  # out, back to the tip, out the other side
        ret = B1 + (tip - B1) * rng.uniform(0.75, 1.0) + rng.normal(0, 0.01 * L, 2)
        V = [tip, B1, ret, B2]
    elif u < 0.7:  # the same, finishing back at the tip
        V = [tip, B1, tip + rng.normal(0, 0.01 * L, 2), B2, tip + rng.normal(0, 0.015 * L, 2)]
    elif u < 0.85:  # closed triangle
        V = [tip, B1, B2, tip + rng.normal(0, 0.015 * L, 2)]
    else:  # across, left open
        V = [tip, B1, B2]
    return polyline(V, 12)[1:]


def gen_arrow(rng):
    shaft = open_curve(rng)
    L = length_of(shaft)
    tan = shaft[-1] - shaft[-4]
    head = arrow_head(shaft[-1], tan / np.hypot(*tan), L, rng)
    P = np.vstack([shaft, head])
    mask = np.r_[np.zeros(len(shaft)), np.ones(len(head))]
    u = rng.random()
    if u < 0.08:  # a head at both ends
        tan0 = shaft[0] - shaft[3]
        head0 = arrow_head(shaft[0], tan0 / np.hypot(*tan0), L, rng)[::-1]
        P = np.vstack([head0, P])
        mask = np.r_[np.ones(len(head0)), mask]
    elif u < 0.2:  # head drawn first
        P, mask = P[::-1], mask[::-1]
    return rot(P, line_angle(rng)), mask


def gen_scribble(rng):
    """A scratch-out: sharp strokes back and forth. Loops and coils are not scribbles."""
    n = int(rng.integers(7, 24))
    amp = rng.uniform(0.3, 1.5)
    x = (np.arange(n) + rng.normal(0, 0.25, n)) / n
    if rng.random() < 0.2:  # sweeps forward and back again
        x = 1 - np.abs(2 * x - 1)
    y = amp / 2 * np.where(np.arange(n) % 2 == 0, 1, -1) * rng.uniform(0.6, 1.1, n)
    P = polyline(np.stack([x + y * rng.uniform(-0.6, 0.6), y + rng.normal(0, 0.04, n)], axis=1), 16)
    return rot(P, rng.uniform(0, 360)), None


# -------------------------------------------------------------- not shapes

def turns_polyline(turns, lens) -> np.ndarray:
    pts, ang = [np.zeros(2)], 0.0
    for i, ln in enumerate(lens):
        if i:
            ang += turns[i - 1]
        pts.append(pts[-1] + ln * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))]))
    return polyline(pts, 20)


def none_polyline(rng):
    u, sgn = rng.random(), rng.choice([-1, 1])
    if u < 0.25:  # V, check mark
        P = turns_polyline([sgn * rng.uniform(105, 160)], [rng.uniform(0.3, 1), 1])
    elif u < 0.45:  # Z, N
        t = rng.uniform(100, 155, 2)
        P = turns_polyline([sgn * t[0], -sgn * t[1]], rng.uniform(0.6, 1.2, 3))
    elif u < 0.6:  # U with corners
        P = turns_polyline(sgn * rng.uniform(70, 110, 2), [rng.uniform(0.5, 1.2), rng.uniform(0.4, 1.2), rng.uniform(0.5, 1.2)])
    elif u < 0.75:  # M, W
        t = rng.uniform(110, 155, 3)
        P = turns_polyline([sgn * t[0], -sgn * t[1], sgn * t[2]], rng.uniform(0.6, 1.2, 4))
    elif u < 0.87:  # stairs
        k = int(rng.integers(4, 8))
        P = turns_polyline([sgn * 90 * (-1) ** i + rng.normal(0, 8) for i in range(k - 1)], rng.uniform(0.5, 1.2, k))
    else:  # random
        k = int(rng.integers(3, 6))
        P = turns_polyline(rng.choice([-1, 1], k - 1) * rng.uniform(40, 150, k - 1), rng.uniform(0.4, 1.2, k))
    return P


def none_partial(rng):
    name = rng.choice(list(LOOPS))
    loop, corners, _ = LOOPS[name](rng)
    extent = rng.uniform(0.6, 0.8) if name == "ellipse" else rng.uniform(0.4, 0.75)
    return trace(loop, rng, corners, extent)


def none_spiral(rng):
    n = rng.uniform(1.6, 4)
    t = np.linspace(0, 1, 500)
    r = rng.uniform(0, 0.2) + (1 - rng.uniform(0, 0.2)) * t ** rng.uniform(0.7, 1.3)
    P = np.stack([r * np.cos(TAU * n * t), r * rng.uniform(0.6, 1) * np.sin(TAU * n * t)], axis=1)
    return P[::-1] if rng.random() < 0.5 else P


def none_eight(rng):
    t = np.linspace(0, TAU * rng.uniform(0.92, 1.1), 400) + rng.uniform(0, TAU)
    return np.stack([np.sin(t), np.sin(t) * np.cos(t) * rng.uniform(0.6, 1.4)], axis=1)


def none_wave(rng):
    x = np.linspace(0, 1, 400)
    return np.stack([x, rng.uniform(0.08, 0.4) * np.sin(TAU * rng.uniform(1.75, 5) * x + rng.uniform(0, TAU))], axis=1)


def none_tail(rng):
    name = rng.choice(list(LOOPS))
    loop, corners, start = LOOPS[name](rng)
    P = trace(loop, rng, corners, None, start)
    out = P[-1] - P.mean(axis=0)
    out = out / max(np.hypot(*out), 1e-6)
    tail = open_curve(rng) * rng.uniform(0.4, 1.2) * size_of(P)
    tail = rot(tail, np.degrees(np.arctan2(out[1], out[0])) + rng.uniform(-60, 60))
    P = np.vstack([P, P[-1] + tail[1:]])
    return P[::-1] if rng.random() < 0.4 else P


def none_loops(rng):
    n = int(rng.integers(2, 4))  # a few spaced loops, like handwriting
    t = np.linspace(-0.25, n - 0.25, 500)
    r = rng.uniform(0.25, 0.45)
    return np.stack([t * rng.uniform(1.0, 1.6) - r * np.sin(TAU * t), -r * rng.uniform(0.8, 1.6) * np.cos(TAU * t)], axis=1)


def none_two(rng):
    parts, x = [], 0.0
    for _ in range(2):
        loop, corners, start = LOOPS[rng.choice(list(LOOPS))](rng)
        P = trace(loop, rng, corners, None, start) * rng.uniform(0.6, 1)
        P = P - P[0] + [x, rng.normal(0, 0.1)]
        parts.append(P)
        x = P[:, 0].max() + rng.uniform(-0.1, 0.3) + 0.5
    return np.vstack(parts)


def none_dot(rng):
    return np.cumsum(rng.normal(0, 1, (int(rng.integers(4, 30)), 2)), axis=0)


def none_walk(rng):
    while True:
        n = 500
        k = gauss_smooth(rng.normal(0, 1, (n, 1)), rng.uniform(8, 30))[:, 0]
        k = k / max(np.abs(k).std(), 1e-6) * rng.uniform(1.5, 5) * TAU / n
        th = np.cumsum(k) + rng.uniform(0, TAU)
        if np.abs(np.diff(th)).sum() > np.deg2rad(330):
            return np.cumsum(np.stack([np.cos(th), np.sin(th)], axis=1), axis=0)


def none_cross(rng):
    a, b = rng.uniform(0.3, 0.6, 2)
    V = [[-0.5, 0], [0.5, 0], [rng.normal(0, 0.05), rng.normal(0, 0.03)], [0, a], [0, -b]]
    return polyline(V, 20)


NONE = [  # (generator, weight)
    (none_polyline, 20), (none_partial, 22), (none_spiral, 7), (none_eight, 6),
    (none_wave, 9), (none_tail, 11), (none_loops, 7), (none_two, 6),
    (none_dot, 2), (none_walk, 8), (none_cross, 2),
]
_NONE_P = np.array([w for _, w in NONE], float) / sum(w for _, w in NONE)


def gen_none(rng):
    P = NONE[rng.choice(len(NONE), p=_NONE_P)][0](rng)
    P = P * [1, rng.uniform(0.6, 1.6)]
    return rot(P, rot_any(rng, 0.4, 8)), None


# ------------------------------------------------------------------- driver

HAND = {
    "none": dict(smooth=(0.002, 0.03), wobble=(0.004, 0.025), drift=0.05),
    "line": dict(smooth=(0.002, 0.01), wobble=(0.003, 0.02)),
    "arrow": dict(smooth=(0.002, 0.012), wobble=(0.003, 0.015)),
    "rectangle": dict(smooth=(0.002, 0.04), wobble=(0.004, 0.022), drift=0.07),
    "ellipse": dict(smooth=(0.002, 0.02), wobble=(0.004, 0.03), drift=0.09),
    "triangle": dict(smooth=(0.002, 0.03), wobble=(0.004, 0.022), drift=0.07),
    "diamond": dict(smooth=(0.002, 0.025), wobble=(0.004, 0.022), drift=0.07),
    "pentagon": dict(smooth=(0.002, 0.018), wobble=(0.004, 0.02), drift=0.07),
    "hexagon": dict(smooth=(0.002, 0.014), wobble=(0.004, 0.018), drift=0.07),
    "star": dict(smooth=(0.002, 0.014), wobble=(0.004, 0.02), drift=0.07),
    "heart": dict(smooth=(0.002, 0.015), wobble=(0.004, 0.02), drift=0.08),
    "cloud": dict(smooth=(0.002, 0.008), wobble=(0.004, 0.015), drift=0.07),
    "scribble": dict(smooth=(0.003, 0.03), wobble=(0.004, 0.03)),
}


def raw(name: str, rng):
    if name in LOOPS:
        return gen_closed(name, rng)
    return {"none": gen_none, "line": gen_line, "arrow": gen_arrow, "scribble": gen_scribble}[name](rng)


def sample(name: str, rng):
    """One synthetic stroke: (128, 2) float32 model input and (128,) arrowhead mask."""
    P, mask = raw(name, rng)
    P, mask = hand(P, mask, rng, **HAND[name])
    if rng.random() < 0.5:
        P = P * [-1, 1]
    P = P * [1, -1]  # canvas coordinates: y down
    pts, mask = resample(P, N_POINTS, mask)
    return normalize(pts).astype(np.float32), (mask > 0.5).astype(np.uint8)


def _job(args):
    name, n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, N_POINTS, 2), np.float16)
    M = np.empty((n, N_POINTS), np.uint8)
    for i in range(n):
        X[i], M[i] = sample(name, rng)
    return name, X, M


def generate(per_class: int = 120_000, none: int = 300_000, seed: int = 0, out: str = "data/synth.npz"):
    chunk = 10_000
    jobs, k = [], 0
    for name in CLASSES:
        total = none if name == "none" else per_class
        for start in range(0, total, chunk):
            jobs.append((name, min(chunk, total - start), seed * 100_000 + k))
            k += 1
    X, M, Y = [], [], []
    with Pool() as pool:
        for i, (name, x, m) in enumerate(pool.imap_unordered(_job, jobs)):
            X.append(x), M.append(m), Y.append(np.full(len(x), CLASS_INDEX[name], np.uint8))
            print(f"\r{i + 1}/{len(jobs)}", end="", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.concatenate(X), M=np.concatenate(M), Y=np.concatenate(Y))
    print(f"\nwrote {out}: {sum(len(x) for x in X)} strokes")


if __name__ == "__main__":
    generate(*(int(a) for a in sys.argv[1:3]))
