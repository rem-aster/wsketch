export type { Arrow, Box, FitOptions, Line, ModelMeta, Polygon, Pt, Shape, ShapeKind } from "./types.js"
export { FIT_DEFAULTS, SHAPE_KINDS, fitShape, isShapeKind, type FitInput } from "./fit.js"
export { arrowHeads, boxCorners, isClosed, outline, toSvgPath } from "./render.js"
export { N_POINTS, preprocess, type Prepared } from "./preprocess.js"
export { MODEL_META } from "./meta.js"
export {
  Recognizer,
  fitClassification,
  type Classification,
  type ModelSource,
  type OrtLike,
  type OrtSession,
  type Recognition,
  type RecognizerOptions,
} from "./recognizer.js"
