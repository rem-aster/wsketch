// Hand-drawn-looking strokes with known geometry, for the tests.
import { bboxSize, resample, rotate } from "../src/geom.js"
import type { Pt } from "../src/types.js"

/** Deterministic random numbers in [0, 1). */
export function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const deg = (d: number) => (d * Math.PI) / 180

/** Slow wobble along the path plus sensor jitter, both relative to the shape's size. */
export function hand(path: Pt[], seed: number, wobble = 0.012, jitter = 0.002): Pt[] {
  const r = rng(seed)
  const P = resample(path, 300).pts
  const size = bboxSize(P)
  const waves = Array.from({ length: 4 }, () => ({
    lam: size * (0.4 + 2 * r()),
    px: r() * 2 * Math.PI,
    py: r() * 2 * Math.PI,
    a: (r() - 0.5) * 2,
  }))
  let s = 0
  return P.map((p, i) => {
    if (i) s += Math.hypot(p.x - P[i - 1].x, p.y - P[i - 1].y)
    let dx = 0, dy = 0
    for (const w of waves) {
      dx += w.a * Math.sin((2 * Math.PI * s) / w.lam + w.px)
      dy += w.a * Math.sin((2 * Math.PI * s) / w.lam + w.py)
    }
    const k = (wobble * size) / 2
    return { x: p.x + dx * k + (r() - 0.5) * 2 * jitter * size, y: p.y + dy * k + (r() - 0.5) * 2 * jitter * size }
  })
}

/** Around a closed polygon, starting `start` of the way round, for `extent` turns (1.1 overshoots by 10%). */
export function aroundPolygon(V: Pt[], start = 0, extent = 1.05): Pt[] {
  const loop = resample([...V, V[0]], 401).pts.slice(0, 400)
  return around(loop, start, extent)
}

export function aroundEllipse(c: Pt, rx: number, ry: number, rot: number, start = 0, extent = 1.05): Pt[] {
  const loop = Array.from({ length: 400 }, (_, i) => {
    const a = (2 * Math.PI * i) / 400
    return rotate({ x: c.x + rx * Math.cos(a), y: c.y + ry * Math.sin(a) }, c, rot)
  })
  return around(loop, start, extent)
}

function around(loop: Pt[], start: number, extent: number): Pt[] {
  const m = loop.length
  const i0 = Math.round(start * m)
  return Array.from({ length: Math.round(extent * m) + 1 }, (_, i) => loop[(i0 + i) % m])
}

export function rectCorners(c: Pt, w: number, h: number, rot = 0): Pt[] {
  return [
    { x: c.x - w / 2, y: c.y - h / 2 }, { x: c.x + w / 2, y: c.y - h / 2 },
    { x: c.x + w / 2, y: c.y + h / 2 }, { x: c.x - w / 2, y: c.y + h / 2 },
  ].map((p) => rotate(p, c, rot))
}

/** Corners of a regular n-gon; `phase` is the angle of the first corner (-90° puts it on top). */
export function ngon(c: Pt, r: number, n: number, phase = -90, sx = 1): Pt[] {
  return Array.from({ length: n }, (_, k) => {
    const a = deg(phase + (360 * k) / n)
    return { x: c.x + sx * r * Math.cos(a), y: c.y + r * Math.sin(a) }
  })
}

/** A straight arrow from `a` to `b`, the head drawn without lifting the pen: out to one barb, back, out to the other. */
export function arrowStroke(a: Pt, b: Pt, head = 0.25): { pts: Pt[]; head: number[] } {
  const L = Math.hypot(b.x - a.x, b.y - a.y)
  const ang = Math.atan2(a.y - b.y, a.x - b.x)
  const barb = (s: number) => ({ x: b.x + head * L * Math.cos(ang + s * deg(30)), y: b.y + head * L * Math.sin(ang + s * deg(30)) })
  const shaft = resample([a, b], 60).pts
  const tail = resample([b, barb(1), b, barb(-1)], 30).pts.slice(1)
  return { pts: [...shaft, ...tail], head: [...shaft.map(() => 0), ...tail.map(() => 1)] }
}
