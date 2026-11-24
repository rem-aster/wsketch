// Turning shapes back into points and paths, for drawing them or for hit testing.

import { add, dir, mul, norm, rotate, sub } from "./geom.js"
import type { Arrow, Box, Pt, Shape } from "./types.js"

export const isClosed = (s: Shape) => s.kind !== "line" && s.kind !== "arrow"

/** Corners of a rectangle (top-left first, clockwise) or a diamond (top first, clockwise), rotation applied. */
export function boxCorners(b: Box): Pt[] {
  const w = b.width / 2, h = b.height / 2
  const local =
    b.kind === "diamond"
      ? [{ x: 0, y: -h }, { x: w, y: 0 }, { x: 0, y: h }, { x: -w, y: 0 }]
      : [{ x: -w, y: -h }, { x: w, y: -h }, { x: w, y: h }, { x: -w, y: h }]
  return local.map((p) => rotate(add(b.center, p), b.center, b.rotation))
}

function cubicAt(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: number): Pt {
  const u = 1 - t
  const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t
  return { x: a * p0.x + b * p1.x + c * p2.x + d * p3.x, y: a * p0.y + b * p1.y + c * p2.y + d * p3.y }
}

/**
 * Points along the shape: the corners of polygons, rectangles and diamonds, `segments` points
 * around an ellipse or along a curved connector. Closed outlines do not repeat their first point.
 * Arrowheads are not included, see `arrowHeads`.
 */
export function outline(shape: Shape, segments = 64): Pt[] {
  switch (shape.kind) {
    case "line":
    case "arrow": {
      if (!shape.control) return [shape.from, shape.to]
      const [c1, c2] = shape.control
      return Array.from({ length: segments + 1 }, (_, i) => cubicAt(shape.from, c1, c2, shape.to, i / segments))
    }
    case "rectangle":
    case "diamond":
      return boxCorners(shape)
    case "ellipse":
      return Array.from({ length: segments }, (_, i) => {
        const a = (2 * Math.PI * i) / segments
        const p = { x: shape.center.x + (shape.width / 2) * Math.cos(a), y: shape.center.y + (shape.height / 2) * Math.sin(a) }
        return rotate(p, shape.center, shape.rotation)
      })
    default:
      return shape.points
  }
}

/**
 * Each arrowhead as three points: one barb, the tip, the other barb.
 * The head at `to` comes first; a second one follows when the arrow has a head at `from`.
 */
export function arrowHeads(arrow: Arrow, opts: { size?: number; angle?: number } = {}): [Pt, Pt, Pt][] {
  const size = opts.size ?? arrow.headSize
  const angle = opts.angle ?? (28 * Math.PI) / 180
  const head = (tip: Pt, from: Pt): [Pt, Pt, Pt] => {
    let back = sub(from, tip)
    if (norm(back) < 1e-9) back = { x: -1, y: 0 }
    const a = Math.atan2(back.y, back.x)
    return [add(tip, mul(dir(a + angle), size)), tip, add(tip, mul(dir(a - angle), size))]
  }
  const [c1, c2] = arrow.control ?? [arrow.from, arrow.to]
  const heads = [head(arrow.to, norm(sub(c2, arrow.to)) > 1e-9 ? c2 : arrow.from)]
  if (arrow.startHead) heads.push(head(arrow.from, norm(sub(c1, arrow.from)) > 1e-9 ? c1 : arrow.to))
  return heads
}

const num = (v: number, digits: number) => {
  const s = v.toFixed(digits)
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s
}

/** SVG path data (`<path d="...">`, or `new Path2D(d)` on a canvas). Arrows include their heads. */
export function toSvgPath(shape: Shape, digits = 2): string {
  const p = (q: Pt) => `${num(q.x, digits)} ${num(q.y, digits)}`
  switch (shape.kind) {
    case "line":
    case "arrow": {
      let d = `M${p(shape.from)}`
      d += shape.control ? `C${p(shape.control[0])} ${p(shape.control[1])} ${p(shape.to)}` : `L${p(shape.to)}`
      if (shape.kind === "arrow") for (const [a, tip, b] of arrowHeads(shape)) d += `M${p(a)}L${p(tip)}L${p(b)}`
      return d
    }
    case "ellipse": {
      const rx = num(shape.width / 2, digits), ry = num(shape.height / 2, digits)
      const deg = num((shape.rotation * 180) / Math.PI, digits)
      const a = rotate({ x: shape.center.x + shape.width / 2, y: shape.center.y }, shape.center, shape.rotation)
      const b = rotate({ x: shape.center.x - shape.width / 2, y: shape.center.y }, shape.center, shape.rotation)
      return `M${p(a)}A${rx} ${ry} ${deg} 0 1 ${p(b)}A${rx} ${ry} ${deg} 0 1 ${p(a)}Z`
    }
    default: {
      const pts = outline(shape)
      return `M${pts.map(p).join("L")}Z`
    }
  }
}
