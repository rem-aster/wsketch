import { fitShape } from "./fit.js"
import { MODEL_META } from "./meta.js"
import { N_POINTS, preprocess } from "./preprocess.js"
import type { FitOptions, ModelMeta, Pt, Shape } from "./types.js"

/**
 * The parts of an ONNX Runtime module the recognizer uses. `onnxruntime-web`,
 * `onnxruntime-node` and `onnxruntime-react-native` all fit: pass the module itself.
 */
export interface OrtLike {
  InferenceSession: { create(model: Uint8Array, options?: unknown): Promise<OrtSession> }
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown
}

export interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown }>>
}

/** A URL to fetch the .onnx file from, or its bytes. */
export type ModelSource = string | URL | ArrayBuffer | Uint8Array

export type RecognizerOptions = {
  ort: OrtLike
  /** The model, e.g. the URL of `wsketch/model.onnx`. */
  model: ModelSource
  /** The model's description; the bundled model's when omitted. */
  meta?: ModelMeta
  /** Passed to `InferenceSession.create`, e.g. `{ executionProviders: ["wasm"] }`. */
  sessionOptions?: unknown
}

export type Classification = {
  /** The accepted class, or null when the stroke should stay ink. */
  kind: string | null
  /** The most likely class, the "not a shape" class included. */
  best: string
  /** Probability of `best`. */
  confidence: number
  /** Probability of every class. */
  probs: Record<string, number>
  /** The points the model looked at (the stroke resampled), in canvas coordinates. */
  samples: Pt[]
  /** Per sample: the probability that it belongs to an arrowhead. */
  head: Float32Array
  /** Inference time in milliseconds, preprocessing excluded. */
  ms: number
}

export type Recognition = Classification & {
  /** The clean shape to put in place of the stroke, or null to keep the stroke as it is. */
  shape: Shape | null
}

async function readModel(model: ModelSource): Promise<Uint8Array> {
  if (model instanceof Uint8Array) return model
  if (model instanceof ArrayBuffer) return new Uint8Array(model)
  const res = await fetch(model)
  if (!res.ok) throw new Error(`wsketch: cannot load the model from ${model}: HTTP ${res.status}`)
  return new Uint8Array(await res.arrayBuffer())
}

/**
 * The shape for a stroke the model has already classified, or null when it stays ink.
 * Lets an app refit with other options without running the model again.
 */
export function fitClassification(c: Classification, points: ArrayLike<Pt>, fit: FitOptions = {}): Shape | null {
  if (!c.kind) return null
  // the arrowhead output is aligned with the samples, not with the raw points
  return c.kind === "arrow" ? fitShape("arrow", c.samples, { ...fit, head: c.head }) : fitShape(c.kind, points, fit)
}

export class Recognizer {
  readonly meta: ModelMeta
  /** Size of the model file. */
  readonly bytes: number
  #ort: OrtLike
  #session: OrtSession
  #queue: Promise<unknown> = Promise.resolve()

  private constructor(ort: OrtLike, session: OrtSession, meta: ModelMeta, bytes: number) {
    this.#ort = ort
    this.#session = session
    this.meta = meta
    this.bytes = bytes
  }

  static async load({ ort, model, meta = MODEL_META, sessionOptions }: RecognizerOptions): Promise<Recognizer> {
    if (meta.input?.n_points && meta.input.n_points !== N_POINTS) {
      throw new Error(`wsketch: the model expects ${meta.input.n_points} points, this library prepares ${N_POINTS}`)
    }
    const bytes = await readModel(model)
    const session = await ort.InferenceSession.create(bytes, sessionOptions)
    return new Recognizer(ort, session, meta, bytes.byteLength)
  }

  /** Runs the model only: which shape the stroke is, if any. */
  classify(points: ArrayLike<Pt>): Promise<Classification> {
    // one inference at a time: some runtimes refuse concurrent runs of a session
    const run = this.#queue.then(() => this.#classify(points))
    this.#queue = run.catch(() => undefined)
    return run
  }

  /** Runs the model and, when it accepts a shape, fits that shape to the stroke. */
  async recognize(points: ArrayLike<Pt>, fit: FitOptions = {}): Promise<Recognition> {
    const c = await this.classify(points)
    return { ...c, shape: fitClassification(c, points, fit) }
  }

  async #classify(points: ArrayLike<Pt>): Promise<Classification> {
    const { classes, reject_classes, thresholds } = this.meta
    if (points.length < 2) {
      const reject = reject_classes[0] ?? classes[0]
      const probs = Object.fromEntries(classes.map((c) => [c, c === reject ? 1 : 0]))
      const samples = Array.from(points, (p) => ({ x: p.x, y: p.y }))
      return { kind: null, best: reject, confidence: 1, probs, samples, head: new Float32Array(samples.length), ms: 0 }
    }
    const prepared = preprocess(points)
    const input = new this.#ort.Tensor("float32", prepared.input, [1, N_POINTS, 2])
    const t0 = performance.now()
    const out = await this.#session.run({ [this.meta.input?.name ?? "points"]: input })
    const ms = performance.now() - t0
    const p = out.probs.data as Float32Array
    const head = out.head.data as Float32Array
    let top = 0
    for (let i = 1; i < p.length; i++) if (p[i] > p[top]) top = i
    const best = classes[top]
    const confidence = p[top]
    const accepted = !reject_classes.includes(best) && confidence >= (thresholds?.[best] ?? 0.5)
    const probs = Object.fromEntries(classes.map((c, i) => [c, p[i]]))
    return { kind: accepted ? best : null, best, confidence, probs, samples: prepared.samples, head, ms }
  }
}
