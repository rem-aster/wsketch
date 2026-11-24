/** A point in canvas coordinates: x to the right, y down. Extra fields (pressure, time) are ignored. */
export type Pt = { x: number; y: number }

/** Shapes the library can rebuild from a stroke. */
export type ShapeKind =
  | "line"
  | "arrow"
  | "rectangle"
  | "ellipse"
  | "diamond"
  | "triangle"
  | "pentagon"
  | "hexagon"

/** A connector. With `control` it is a cubic Bézier curve from `from` to `to`, otherwise a straight segment. */
export type Line = {
  kind: "line"
  from: Pt
  to: Pt
  control?: [Pt, Pt]
}

/** A connector with a head at `to`, and also at `from` when `startHead` is set. */
export type Arrow = {
  kind: "arrow"
  from: Pt
  to: Pt
  control?: [Pt, Pt]
  startHead: boolean
  /** How far the drawn head reached back from the tip, in canvas units. */
  headSize: number
}

/**
 * A shape defined by its box: the rectangle itself, the ellipse inscribed in it,
 * or the diamond whose corners are the midpoints of its sides.
 * `rotation` is in radians, in the direction of `CanvasRenderingContext2D.rotate`
 * and SVG `rotate()` (clockwise on screen), always in (-π/4, π/4].
 */
export type Box = {
  kind: "rectangle" | "ellipse" | "diamond"
  center: Pt
  width: number
  height: number
  rotation: number
}

/** Corners in clockwise order on screen, starting from the top one (the leftmost of equals). */
export type Polygon = {
  kind: "triangle" | "pentagon" | "hexagon"
  points: Pt[]
}

export type Shape = Line | Arrow | Box | Polygon

/** The model description exported next to the .onnx file (shapes.json). */
export type ModelMeta = {
  /** Class names in the order of the `probs` output. */
  classes: string[]
  /** Classes that mean "not a shape": the stroke stays ink. */
  reject_classes: string[]
  /** Minimum probability for a class to be accepted (0.5 when missing). */
  thresholds?: Record<string, number>
  input?: { name?: string; n_points?: number }
}

export type FitOptions = {
  /** Degrees. Straight lines and arrows, boxes and polygons this close to horizontal/vertical are made exactly so. 0 disables. Default 8. */
  snapAngle?: number
  /** Width/height ratio above which an ellipse becomes a circle and a rectangle or diamond a square one. 1 disables. Default 0.9. */
  snapRatio?: number
  /** Triangles, pentagons and hexagons within this relative deviation of a regular polygon become regular. 0 disables. Default 0.12. */
  regular?: number
  /** A line or arrow that bows less than this fraction of its length becomes straight, otherwise a smooth curve. Infinity always straightens. Default 0.06. */
  straight?: number
}
