// Lines and arrows: a straight segment when the stroke barely bows, otherwise a cubic Bézier.

import {
  add, angleOf, dist, distToSegment, dot, fitLine, mul, norm, pathLength, resample, rotate, snap, sub,
} from "./geom.js"
import type { Arrow, FitOptions, Line, Pt } from "./types.js"

type Connector = { from: Pt; to: Pt; control?: [Pt, Pt] }

/** How many points at the start of `P` form a pen-down hook: a short flick at a sharp angle to the rest. */
function hookLength(P: Pt[]): number {
  const n = P.length
  const lim = Math.floor(n * 0.1), w = Math.max(2, Math.floor(n * 0.08))
  let best = 0, at = 0
  for (let k = 1; k <= lim && k + w < n; k++) {
    const a = sub(P[k], P[0]), b = sub(P[k + w], P[k])
    const la = norm(a), lb = norm(b)
    if (la < 1e-9 || lb < 1e-9) continue
    const ang = Math.acos(Math.max(-1, Math.min(1, dot(a, b) / (la * lb))))
    if (ang > best) {
      best = ang
      at = k
    }
  }
  return best > (55 * Math.PI) / 180 ? at : 0
}

function bezier(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: number): Pt {
  const u = 1 - t
  const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t
  return { x: a * p0.x + b * p1.x + c * p2.x + d * p3.x, y: a * p0.y + b * p1.y + c * p2.y + d * p3.y }
}

const unit = (v: Pt): Pt => {
  const l = norm(v)
  return l > 1e-12 ? mul(v, 1 / l) : { x: 0, y: 0 }
}

/**
 * Control points of the cubic from P[0] to P[n-1] closest to `P`, its end tangents following the
 * drawn ends (Schneider's fit: least squares along fixed tangents, with Newton reparametrization).
 */
function fitCubic(P: Pt[]): [Pt, Pt] {
  const n = P.length
  const p0 = P[0], p3 = P[n - 1]
  const k = Math.max(2, Math.round(n * 0.08))
  const t1 = unit(sub(P[k], p0)), t2 = unit(sub(P[n - 1 - k], p3))
  const chord = dist(p0, p3)
  const t = new Float64Array(n)
  for (let i = 1; i < n; i++) t[i] = t[i - 1] + dist(P[i - 1], P[i])
  for (let i = 1; i < n; i++) t[i] /= t[n - 1] || 1
  let c1 = add(p0, mul(t1, chord / 3))
  let c2 = add(p3, mul(t2, chord / 3))
  for (let iter = 0; iter < 4; iter++) {
    // P_i ≈ (b0 + b1)·p0 + (b2 + b3)·p3 + α1·b1·t1 + α2·b2·t2
    let c11 = 0, c12 = 0, c22 = 0, x1 = 0, x2 = 0
    for (let i = 0; i < n; i++) {
      const s = t[i], u = 1 - s
      const b0 = u * u * u, b1 = 3 * u * u * s, b2 = 3 * u * s * s, b3 = s * s * s
      const a1 = mul(t1, b1), a2 = mul(t2, b2)
      const r = { x: P[i].x - (b0 + b1) * p0.x - (b2 + b3) * p3.x, y: P[i].y - (b0 + b1) * p0.y - (b2 + b3) * p3.y }
      c11 += dot(a1, a1)
      c12 += dot(a1, a2)
      c22 += dot(a2, a2)
      x1 += dot(a1, r)
      x2 += dot(a2, r)
    }
    const det = c11 * c22 - c12 * c12
    let al1 = Math.abs(det) > 1e-12 ? (x1 * c22 - x2 * c12) / det : chord / 3
    let al2 = Math.abs(det) > 1e-12 ? (c11 * x2 - c12 * x1) / det : chord / 3
    // a negative or huge handle would make a loop: fall back to the usual third of the chord
    if (!(al1 > 1e-6 * chord && al1 < 2 * chord)) al1 = chord / 3
    if (!(al2 > 1e-6 * chord && al2 < 2 * chord)) al2 = chord / 3
    c1 = add(p0, mul(t1, al1))
    c2 = add(p3, mul(t2, al2))
    for (let i = 1; i < n - 1; i++) {
      const s = t[i], u = 1 - s
      const e = sub(bezier(p0, c1, c2, p3, s), P[i])
      const d1 = {
        x: 3 * u * u * (c1.x - p0.x) + 6 * u * s * (c2.x - c1.x) + 3 * s * s * (p3.x - c2.x),
        y: 3 * u * u * (c1.y - p0.y) + 6 * u * s * (c2.y - c1.y) + 3 * s * s * (p3.y - c2.y),
      }
      const d2 = {
        x: 6 * u * (c2.x - 2 * c1.x + p0.x) + 6 * s * (p3.x - 2 * c2.x + c1.x),
        y: 6 * u * (c2.y - 2 * c1.y + p0.y) + 6 * s * (p3.y - 2 * c2.y + c1.y),
      }
      const den = dot(d1, d1) + dot(e, d2)
      if (den > 1e-12) t[i] = Math.min(1, Math.max(0, s - dot(e, d1) / den))
    }
  }
  return [c1, c2]
}

/** A connector through the points; pen hooks are dropped at the ends listed in `trim`. */
function connector(points: ArrayLike<Pt>, o: Required<FitOptions>, trim: { start: boolean; end: boolean }): Connector {
  let P = resample(points, 96).pts
  if (trim.start) P = P.slice(hookLength(P))
  if (trim.end) P = P.slice(0, P.length - hookLength([...P].reverse()))
  const a = P[0], b = P[P.length - 1]
  const chord = dist(a, b)
  let bow = 0
  for (const p of P) bow = Math.max(bow, distToSegment(p, a, b))
  if (bow > o.straight * Math.max(chord, 1e-9)) return { from: a, to: b, control: fitCubic(P) }

  // straight: the total least squares line, between the projections of the ends
  const { c, d } = fitLine(P)
  let from = add(c, mul(d, dot(sub(a, c), d)))
  let to = add(c, mul(d, dot(sub(b, c), d)))
  const ang = angleOf(sub(to, from))
  const snapped = snap(ang, Math.PI / 2, (o.snapAngle * Math.PI) / 180)
  if (snapped !== ang) {
    const mid = mul(add(from, to), 0.5)
    from = rotate(from, mid, snapped - ang)
    to = rotate(to, mid, snapped - ang)
  }
  return { from, to }
}

export function fitLineShape(points: ArrayLike<Pt>, o: Required<FitOptions>): Line {
  return { kind: "line", ...connector(points, o, { start: true, end: true }) }
}

/**
 * `head` holds, for every point, the probability that it belongs to an arrowhead
 * (the model's `head` output, aligned with the points it was computed for).
 * Without it the head is taken to start at the point farthest from the start of the stroke.
 */
export function fitArrowShape(points: ArrayLike<Pt>, head: ArrayLike<number> | undefined, o: Required<FitOptions>): Arrow {
  const P = Array.from(points)
  const n = P.length
  let isHead: boolean[]
  if (head && head.length === n) {
    isHead = Array.from(head, (h) => h > 0.5)
  } else {
    let tip = 0
    for (let i = 1; i < n; i++) if (dist(P[0], P[i]) > dist(P[0], P[tip])) tip = i
    isHead = P.map((_, i) => i > tip)
  }

  // the shaft is the longest run of points outside any head
  let a = 0, b = -1
  for (let i = 0; i < n; ) {
    if (isHead[i]) {
      i++
      continue
    }
    let j = i
    while (j + 1 < n && !isHead[j + 1]) j++
    if (j - i > b - a) [a, b] = [i, j]
    i = j + 1
  }
  if (b - a + 1 < 4) [a, b] = [0, n - 1]

  let shaft = P.slice(a, b + 1)
  let tipHead = P.slice(b + 1)
  let startHead = a >= 2
  if (tipHead.length < 2 && startHead) {
    // the head was drawn first: turn the arrow around so the head is at `to`
    shaft = shaft.reverse()
    tipHead = P.slice(0, a)
    startHead = false
  }
  const c = connector(shaft, o, { start: true, end: false })
  let headSize = 0
  for (const p of tipHead) headSize = Math.max(headSize, dist(p, c.to))
  if (headSize === 0) headSize = 0.2 * pathLength(shaft)
  return { kind: "arrow", ...c, startHead, headSize }
}
