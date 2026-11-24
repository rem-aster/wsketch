// The model's input contract, mirrored from wsketch/preprocess.py:
// 1. resample to N points evenly spaced along the stroke (first and last kept)
// 2. move the bounding-box centre to the origin
// 3. scale so the longer bounding-box side is 1

import type { Pt } from "./types.js"

export const N_POINTS = 128

export type Prepared = {
  /** The model input, [n * 2]: x0, y0, x1, y1, ... */
  input: Float32Array
  /** The same n points in canvas coordinates. */
  samples: Pt[]
  /** Bounding-box centre and longer side of the samples: canvas = centre + input * size. */
  cx: number
  cy: number
  size: number
  /** Path length in canvas units. */
  length: number
}

export function preprocess(pts: ArrayLike<Pt>, n = N_POINTS): Prepared {
  const m = pts.length
  if (m === 0) throw new Error("preprocess: empty stroke")
  const s = new Float64Array(m)
  for (let i = 1; i < m; i++) {
    s[i] = s[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
  }
  const total = s[m - 1]
  const input = new Float32Array(n * 2)
  const samples: Pt[] = []
  let j = 0
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (let i = 0; i < n; i++) {
    let x = pts[0].x, y = pts[0].y
    if (m > 1) {
      const t = (total * i) / (n - 1)
      while (j < m - 2 && s[j + 1] < t) j++
      const span = s[j + 1] - s[j]
      const w = span > 0 ? Math.min(1, Math.max(0, (t - s[j]) / span)) : 0
      x = pts[j].x + (pts[j + 1].x - pts[j].x) * w
      y = pts[j].y + (pts[j + 1].y - pts[j].y) * w
    }
    samples.push({ x, y })
    minX = Math.min(minX, x); maxX = Math.max(maxX, x)
    minY = Math.min(minY, y); maxY = Math.max(maxY, y)
  }
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const size = Math.max(maxX - minX, maxY - minY, 1e-9)
  for (let i = 0; i < n; i++) {
    input[2 * i] = (samples[i].x - cx) / size
    input[2 * i + 1] = (samples[i].y - cy) / size
  }
  return { input, samples, cx, cy, size, length: total }
}
