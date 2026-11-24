import { fitDiamond, fitEllipse, fitPolygon, fitRectangle } from "./fit-closed.js"
import { fitArrowShape, fitLineShape } from "./fit-open.js"
import { bboxSize } from "./geom.js"
import type { FitOptions, Pt, Shape, ShapeKind } from "./types.js"

export const SHAPE_KINDS: readonly ShapeKind[] = [
  "line", "arrow", "rectangle", "ellipse", "diamond", "triangle", "pentagon", "hexagon",
]

export const FIT_DEFAULTS: Readonly<Required<FitOptions>> = {
  snapAngle: 8,
  snapRatio: 0.9,
  regular: 0.12,
  straight: 0.06,
}

export const isShapeKind = (kind: string): kind is ShapeKind => (SHAPE_KINDS as readonly string[]).includes(kind)

export type FitInput = FitOptions & {
  /** Arrows only: per point, the probability that it belongs to an arrowhead (the model's `head` output). */
  head?: ArrayLike<number>
}

/**
 * The clean shape of kind `kind` closest to the stroke, in the stroke's own coordinates.
 * Null for kinds the library cannot rebuild (`none`, `star`, ...) and for strokes without extent.
 */
export function fitShape(kind: string, points: ArrayLike<Pt>, options: FitInput = {}): Shape | null {
  if (!isShapeKind(kind) || points.length < 2 || bboxSize(points) <= 0) return null
  const o = { ...FIT_DEFAULTS }
  for (const key of Object.keys(o) as (keyof FitOptions)[]) {
    const v = options[key]
    if (typeof v === "number" && !Number.isNaN(v)) o[key] = v
  }
  switch (kind) {
    case "line":
      return fitLineShape(points, o)
    case "arrow":
      return fitArrowShape(points, options.head, o)
    case "rectangle":
      return fitRectangle(points, o)
    case "ellipse":
      return fitEllipse(points, o)
    case "diamond":
      return fitDiamond(points, o)
    case "triangle":
    case "pentagon":
    case "hexagon":
      return fitPolygon(kind, points, o)
  }
}
