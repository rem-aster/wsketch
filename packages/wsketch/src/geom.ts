import type { Pt } from "./types.js"

export const pt = (x: number, y: number): Pt => ({ x, y })
export const add = (a: Pt, b: Pt): Pt => ({ x: a.x + b.x, y: a.y + b.y })
export const sub = (a: Pt, b: Pt): Pt => ({ x: a.x - b.x, y: a.y - b.y })
export const mul = (a: Pt, k: number): Pt => ({ x: a.x * k, y: a.y * k })
export const dot = (a: Pt, b: Pt) => a.x * b.x + a.y * b.y
export const cross = (a: Pt, b: Pt) => a.x * b.y - a.y * b.x
export const norm = (a: Pt) => Math.hypot(a.x, a.y)
export const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y)
export const dir = (a: number): Pt => ({ x: Math.cos(a), y: Math.sin(a) })
export const angleOf = (v: Pt) => Math.atan2(v.y, v.x)

/** `p` rotated by `a` radians around `c` (clockwise on screen for a > 0). */
export function rotate(p: Pt, c: Pt, a: number): Pt {
  const cos = Math.cos(a), sin = Math.sin(a)
  const dx = p.x - c.x, dy = p.y - c.y
  return { x: c.x + dx * cos - dy * sin, y: c.y + dx * sin + dy * cos }
}

export function pathLength(pts: ArrayLike<Pt>): number {
  let s = 0
  for (let i = 1; i < pts.length; i++) s += dist(pts[i - 1], pts[i])
  return s
}

export function centroid(pts: ArrayLike<Pt>): Pt {
  let x = 0, y = 0
  for (let i = 0; i < pts.length; i++) {
    x += pts[i].x
    y += pts[i].y
  }
  return { x: x / pts.length, y: y / pts.length }
}

export function bboxSize(pts: ArrayLike<Pt>): number {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  return Math.max(maxX - minX, maxY - minY)
}

/** `n` points evenly spaced along the path, first and last kept; `extra` values are interpolated alongside. */
export function resample(pts: ArrayLike<Pt>, n: number, extra?: ArrayLike<number>): { pts: Pt[]; extra: number[] } {
  const m = pts.length
  const s = new Float64Array(m)
  for (let i = 1; i < m; i++) s[i] = s[i - 1] + dist(pts[i - 1], pts[i])
  const total = s[m - 1]
  const out: Pt[] = []
  const ex: number[] = []
  let j = 0
  for (let i = 0; i < n; i++) {
    if (m < 2) {
      out.push({ x: pts[0].x, y: pts[0].y })
      if (extra) ex.push(extra[0])
      continue
    }
    const t = n > 1 ? (total * i) / (n - 1) : 0
    while (j < m - 2 && s[j + 1] < t) j++
    const span = s[j + 1] - s[j]
    const w = span > 0 ? Math.min(1, Math.max(0, (t - s[j]) / span)) : 0
    out.push({ x: pts[j].x + (pts[j + 1].x - pts[j].x) * w, y: pts[j].y + (pts[j + 1].y - pts[j].y) * w })
    if (extra) ex.push(extra[j] + (extra[j + 1] - extra[j]) * w)
  }
  return { pts: out, extra: ex }
}

/** `n` points evenly spaced around the closed path (the last point joins the first). */
export function resampleClosed(pts: ArrayLike<Pt>, n: number): Pt[] {
  const closed = Array.from(pts)
  closed.push(closed[0])
  return resample(closed, n + 1).pts.slice(0, n)
}

/** Total least squares line: through the centroid, along the direction of greatest spread. */
export function fitLine(pts: ArrayLike<Pt>): { c: Pt; d: Pt } {
  const c = centroid(pts)
  let sxx = 0, sxy = 0, syy = 0
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - c.x, dy = pts[i].y - c.y
    sxx += dx * dx
    sxy += dx * dy
    syy += dy * dy
  }
  return { c, d: dir(0.5 * Math.atan2(2 * sxy, sxx - syy)) }
}

/** Where the lines `a.c + t·a.d` and `b.c + u·b.d` cross, or null when they are (nearly) parallel. */
export function intersect(a: { c: Pt; d: Pt }, b: { c: Pt; d: Pt }): Pt | null {
  const den = cross(a.d, b.d)
  if (Math.abs(den) < 1e-6 * norm(a.d) * norm(b.d)) return null
  const t = cross(sub(b.c, a.c), b.d) / den
  return add(a.c, mul(a.d, t))
}

export function distToSegment(p: Pt, a: Pt, b: Pt): number {
  const ab = sub(b, a)
  const l2 = dot(ab, ab)
  const t = l2 > 0 ? Math.min(1, Math.max(0, dot(sub(p, a), ab) / l2)) : 0
  return dist(p, add(a, mul(ab, t)))
}

/** `a` moved to the nearest multiple of `period` when it is within `tol` of it. */
export function snap(a: number, period: number, tol: number): number {
  const k = Math.round(a / period) * period
  return Math.abs(a - k) <= tol ? k : a
}

/** Solves A·x = b (A is n×n, row-major) by Gaussian elimination with partial pivoting; null if singular. */
export function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    if (Math.abs(M[p][c]) < 1e-12) return null
    ;[M[c], M[p]] = [M[p], M[c]]
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const x = new Array<number>(n)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]
    x[r] = s / M[r][r]
  }
  return x
}
