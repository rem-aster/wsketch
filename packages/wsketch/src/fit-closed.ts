// Closed shapes. The stroke is cut to one turn around, then:
// - polygons: the best n-gon through the stroke (dynamic programming), each side refitted
//   as a straight line and the corners put where neighbouring sides cross;
// - ellipses: a least squares conic.

import {
  angleOf, bboxSize, centroid, cross, dir, dist, fitLine, intersect, mul, add, pathLength, resample,
  resampleClosed, rotate, snap, solve, sub,
} from "./geom.js"
import type { Box, FitOptions, Polygon, Pt } from "./types.js"

const QUARTER = Math.PI / 2
const rad = (deg: number) => (deg * Math.PI) / 180

/** The part of a closed stroke that goes around once: overshoot past the start and pen hooks cut off. */
export function closeLoop(points: ArrayLike<Pt>): Pt[] {
  const M = 200
  const P = resample(points, M).pts
  const step = pathLength(P) / (M - 1)
  let best = Infinity, bi = 0, bj = M - 1
  // where the end comes back to the start; cutting the stroke short costs half its length
  for (let i = 0; i < M * 0.2; i++) {
    for (let j = Math.floor(M * 0.5); j < M; j++) {
      const cost = dist(P[i], P[j]) + 0.5 * step * (i + M - 1 - j)
      if (cost < best) {
        best = cost
        bi = i
        bj = j
      }
    }
  }
  return P.slice(bi, bj + 1)
}

/** Indices into the closed loop `Q` of the n-gon whose sides pass closest to its points. */
function bestPolygon(Q: Pt[], n: number): number[] {
  const K = Q.length
  // C[i*K + l]: squared distances from the points strictly between i and i+l to the chord i -> i+l
  const C = new Float64Array(K * K)
  for (let i = 0; i < K; i++) {
    const a = Q[i]
    for (let l = 2; l < K; l++) {
      const b = Q[(i + l) % K]
      const abx = b.x - a.x, aby = b.y - a.y
      const l2 = abx * abx + aby * aby
      let s = 0
      for (let k = 1; k < l; k++) {
        const p = Q[(i + k) % K]
        const px = p.x - a.x, py = p.y - a.y
        const t = l2 > 0 ? Math.min(1, Math.max(0, (px * abx + py * aby) / l2)) : 0
        const dx = px - t * abx, dy = py - t * aby
        s += dx * dx + dy * dy
      }
      C[i * K + l] = s
    }
  }
  const f = new Float64Array(n * K)
  const from = new Int32Array(n * K)
  let best = Infinity
  let corners: number[] = []
  for (let s = 0; s < K; s++) {
    // f[m*K + j]: cost of the first m sides when corner m is j points after corner 0, which sits at s
    f.fill(Infinity)
    f[0] = 0
    for (let m = 1; m < n; m++) {
      for (let j = m; j <= K - (n - m); j++) {
        let b = Infinity, bi = -1
        for (let i = m - 1; i < j; i++) {
          const v = f[(m - 1) * K + i] + C[((s + i) % K) * K + (j - i)]
          if (v < b) {
            b = v
            bi = i
          }
        }
        f[m * K + j] = b
        from[m * K + j] = bi
      }
    }
    for (let j = n - 1; j < K; j++) {
      const v = f[(n - 1) * K + j] + C[((s + j) % K) * K + (K - j)]
      if (v < best) {
        best = v
        const idx = [j]
        for (let m = n - 1; m > 0; m--) idx.push(from[m * K + idx[idx.length - 1]])
        corners = idx.reverse().map((i) => (s + i) % K)
      }
    }
  }
  return corners
}

/** Corners of the n-gon traced by the loop, each side straightened by a line fit. */
function polygonCorners(loop: Pt[], n: number): Pt[] {
  const K = 96
  const Q = resampleClosed(loop, K)
  const idx = bestPolygon(Q, n)
  const size = bboxSize(Q)
  const sides = idx.map((a, k) => {
    const b = idx[(k + 1) % n]
    const l = (b - a + K) % K || K
    const trim = Math.floor(l * 0.15) // corners are rounded: fit the middle of each side
    const pts: Pt[] = []
    for (let t = trim; t <= l - trim; t++) pts.push(Q[(a + t) % K])
    return pts.length >= 3 ? fitLine(pts) : { c: Q[a], d: sub(Q[b], Q[a]) }
  })
  return idx.map((v, k) => {
    const p = intersect(sides[(k - 1 + n) % n], sides[k])
    return p && dist(p, Q[v]) < 0.25 * size ? p : Q[v]
  })
}

/** A box with rotation brought into (-π/4, π/4], snapped to the axes and to a square when close. */
function makeBox(kind: Box["kind"], center: Pt, width: number, height: number, rotation: number, o: Required<FitOptions>): Box {
  let r = rotation - Math.round(rotation / QUARTER) * QUARTER
  let w = width, h = height
  if (Math.round((rotation - r) / QUARTER) % 2 !== 0) [w, h] = [h, w]
  if (r <= -Math.PI / 4) {
    r += QUARTER
    ;[w, h] = [h, w]
  }
  r = snap(r, QUARTER, rad(o.snapAngle))
  if (Math.min(w, h) / Math.max(w, h) >= o.snapRatio) {
    w = h = (w + h) / 2
    if (kind === "ellipse") r = 0
  }
  return { kind, center, width: w, height: h, rotation: r === 0 ? 0 : r }
}

export function fitRectangle(points: ArrayLike<Pt>, o: Required<FitOptions>): Box {
  const V = polygonCorners(closeLoop(points), 4)
  // side directions averaged modulo 90 degrees, longer sides weigh more
  let sc = 0, ss = 0
  for (let k = 0; k < 4; k++) {
    const e = sub(V[(k + 1) % 4], V[k])
    const len = Math.hypot(e.x, e.y), a = 4 * angleOf(e)
    sc += len * Math.cos(a)
    ss += len * Math.sin(a)
  }
  const theta = Math.atan2(ss, sc) / 4
  // in the rectangle's own frame: two corners on each side
  const c = centroid(V)
  const local = V.map((p) => rotate(p, c, -theta))
  const xs = local.map((p) => p.x).sort((a, b) => a - b)
  const ys = local.map((p) => p.y).sort((a, b) => a - b)
  const left = (xs[0] + xs[1]) / 2, right = (xs[2] + xs[3]) / 2
  const top = (ys[0] + ys[1]) / 2, bottom = (ys[2] + ys[3]) / 2
  const center = rotate({ x: (left + right) / 2, y: (top + bottom) / 2 }, c, theta)
  return makeBox("rectangle", center, right - left, bottom - top, theta, o)
}

export function fitDiamond(points: ArrayLike<Pt>, o: Required<FitOptions>): Box {
  const V = polygonCorners(closeLoop(points), 4)
  const d1 = sub(V[2], V[0]), d2 = sub(V[3], V[1])
  // the diagonals of a rhombus are perpendicular: average their directions modulo 180 degrees
  const a1 = 2 * angleOf(d1), a2 = 2 * (angleOf(d2) - QUARTER)
  const phi = Math.atan2(Math.sin(a1) + Math.sin(a2), Math.cos(a1) + Math.cos(a2)) / 2
  const u = dir(phi), v = dir(phi + QUARTER)
  const c = centroid(V)
  const m1 = mul(add(V[0], V[2]), 0.5), m2 = mul(add(V[1], V[3]), 0.5)
  const center = add(c, add(mul(u, (m1.x - c.x) * u.x + (m1.y - c.y) * u.y), mul(v, (m2.x - c.x) * v.x + (m2.y - c.y) * v.y)))
  const width = Math.abs(d1.x * u.x + d1.y * u.y)
  const height = Math.abs(d2.x * v.x + d2.y * v.y)
  return makeBox("diamond", center, width, height, phi, o)
}

export function fitEllipse(points: ArrayLike<Pt>, o: Required<FitOptions>): Box {
  const P = closeLoop(points)
  const c = centroid(P)
  let s = 0
  for (const p of P) s += (p.x - c.x) ** 2 + (p.y - c.y) ** 2
  s = Math.sqrt(s / P.length) || 1
  // A x² + B xy + C y² + D x + E y = 1, in coordinates centred on the centroid and scaled by s
  const M = Array.from({ length: 5 }, () => new Array<number>(5).fill(0))
  const r = new Array<number>(5).fill(0)
  for (const p of P) {
    const x = (p.x - c.x) / s, y = (p.y - c.y) / s
    const row = [x * x, x * y, y * y, x, y]
    for (let i = 0; i < 5; i++) {
      r[i] += row[i]
      for (let j = 0; j < 5; j++) M[i][j] += row[i] * row[j]
    }
  }
  const w = solve(M, r)
  if (w) {
    const [A, B, C, D, E] = w
    const det = 4 * A * C - B * B
    if (det > 1e-9) {
      const x0 = (B * E - 2 * C * D) / det, y0 = (B * D - 2 * A * E) / det
      const k = 1 - (A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0)
      const th = 0.5 * Math.atan2(B, A - C)
      const cs = Math.cos(th), sn = Math.sin(th)
      const a = A * cs * cs + B * cs * sn + C * sn * sn
      const b = A * sn * sn - B * cs * sn + C * cs * cs
      const size = bboxSize(P)
      if (k / a > 0 && k / b > 0) {
        const width = 2 * s * Math.sqrt(k / a), height = 2 * s * Math.sqrt(k / b)
        if (Math.max(width, height) < 2 * size) {
          return makeBox("ellipse", { x: c.x + s * x0, y: c.y + s * y0 }, width, height, th, o)
        }
      }
    }
  }
  // not an ellipse after all: the box of the points along their principal axes
  const { d } = fitLine(P)
  const th = angleOf(d)
  const local = P.map((p) => rotate(p, c, -th))
  const xs = local.map((p) => p.x), ys = local.map((p) => p.y)
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  return makeBox("ellipse", rotate({ x: (x0 + x1) / 2, y: (y0 + y1) / 2 }, c, th), x1 - x0, y1 - y0, th, o)
}

/** The regular n-gon closest to the corners `V` (taken in order), or null when it is not close enough. */
function regularPolygon(V: Pt[], o: Required<FitOptions>): Pt[] | null {
  const n = V.length
  const c = centroid(V)
  let best: { rel: number; zr: number; zi: number; sign: number } | null = null
  for (const sign of [1, -1]) {
    // similarity fit in complex numbers: V_k - c ≈ z · e^{i·sign·2πk/n}
    let zr = 0, zi = 0
    for (let k = 0; k < n; k++) {
      const a = (sign * 2 * Math.PI * k) / n
      const qx = V[k].x - c.x, qy = V[k].y - c.y
      zr += (qx * Math.cos(a) + qy * Math.sin(a)) / n
      zi += (qy * Math.cos(a) - qx * Math.sin(a)) / n
    }
    let res = 0
    for (let k = 0; k < n; k++) {
      const a = (sign * 2 * Math.PI * k) / n
      const px = zr * Math.cos(a) - zi * Math.sin(a), py = zr * Math.sin(a) + zi * Math.cos(a)
      res += (V[k].x - c.x - px) ** 2 + (V[k].y - c.y - py) ** 2
    }
    const rel = Math.sqrt(res / n) / (Math.hypot(zr, zi) || 1)
    if (!best || rel < best.rel) best = { rel, zr, zi, sign }
  }
  if (!best || best.rel > o.regular) return null
  const { zr, zi, sign } = best
  // upright when close: a vertex or the middle of a side straight above the centre
  const alpha = snap(Math.atan2(zi, zr) + QUARTER, Math.PI / n, rad(o.snapAngle)) - QUARTER
  const R = Math.hypot(zr, zi)
  return V.map((_, k) => add(c, mul(dir(alpha + (sign * 2 * Math.PI * k) / n), R)))
}

/** Sides within the snap angle of horizontal or vertical are made exactly so by moving their two ends; other corners stay. */
function straighten(V: Pt[], o: Required<FitOptions>): Pt[] {
  const W = V.map((p) => ({ ...p }))
  const n = W.length
  for (let k = 0; k < n; k++) {
    const p = W[k], q = W[(k + 1) % n]
    const a = Math.abs(angleOf(sub(q, p)))
    const fromHorizontal = Math.min(a, Math.PI - a)
    if (fromHorizontal <= rad(o.snapAngle)) p.y = q.y = (p.y + q.y) / 2
    else if (Math.abs(a - QUARTER) <= rad(o.snapAngle)) p.x = q.x = (p.x + q.x) / 2
  }
  return W
}

/** Clockwise on screen, starting from the top corner (the leftmost of equals). */
export function orderPolygon(V: Pt[]): Pt[] {
  let area = 0
  for (let k = 0; k < V.length; k++) area += cross(V[k], V[(k + 1) % V.length])
  const W = area < 0 ? [...V].reverse() : [...V]
  const eps = 1e-6 * (bboxSize(W) || 1)
  let top = 0
  for (let k = 1; k < W.length; k++) {
    if (W[k].y < W[top].y - eps || (Math.abs(W[k].y - W[top].y) <= eps && W[k].x < W[top].x)) top = k
  }
  return [...W.slice(top), ...W.slice(0, top)]
}

export function fitPolygon(kind: Polygon["kind"], points: ArrayLike<Pt>, o: Required<FitOptions>): Polygon {
  const n = { triangle: 3, pentagon: 5, hexagon: 6 }[kind]
  const V = polygonCorners(closeLoop(points), n)
  const R = (o.regular > 0 && regularPolygon(V, o)) || straighten(V, o)
  return { kind, points: orderPolygon(R) }
}
