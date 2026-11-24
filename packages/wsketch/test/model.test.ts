// The input contract and the bundled model, end to end.
import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import * as ort from "onnxruntime-web"
import { MODEL_META, N_POINTS, Recognizer, preprocess, type Arrow, type Box, type Pt } from "../src/index.js"
import { aroundEllipse, aroundPolygon, arrowStroke, hand, ngon, rectCorners } from "./strokes.js"

ort.env.wasm.numThreads = 1
const models = join(import.meta.dir, "..", "..", "..", "models")

test("preprocessing matches wsketch/preprocess.py", () => {
  const cases: { points: number[][]; input: number[][] }[] = JSON.parse(
    readFileSync(join(import.meta.dir, "fixtures", "preprocess.json"), "utf8"),
  )
  for (const c of cases) {
    const { input } = preprocess(c.points.map(([x, y]) => ({ x, y })))
    expect(input).toHaveLength(N_POINTS * 2)
    let worst = 0
    for (let i = 0; i < N_POINTS; i++) {
      worst = Math.max(worst, Math.abs(input[2 * i] - c.input[i][0]), Math.abs(input[2 * i + 1] - c.input[i][1]))
    }
    expect(worst).toBeLessThan(1e-5)
  }
})

test("bundled meta is the current models/shapes.json (run `bun run build` after exporting a model)", () => {
  const json = JSON.parse(readFileSync(join(models, "shapes.json"), "utf8"))
  expect(MODEL_META.classes).toEqual(json.classes)
  expect(MODEL_META.reject_classes).toEqual(json.reject_classes)
  expect(MODEL_META.thresholds).toEqual(json.thresholds)
})

describe("Recognizer with the light model", async () => {
  const rec = await Recognizer.load({ ort, model: readFileSync(join(models, "shapes.int8.onnx")) })
  const C = { x: 400, y: 300 }

  test("recognizes and fits closed shapes", async () => {
    const cases: [string, Pt[]][] = [
      ["rectangle", hand(aroundPolygon(rectCorners(C, 260, 150), 0.1, 1.06), 1)],
      ["ellipse", hand(aroundEllipse(C, 140, 80, 0.3, 0.6, 1.04), 2)],
      ["triangle", hand(aroundPolygon([{ x: 300, y: 420 }, { x: 540, y: 410 }, { x: 380, y: 200 }], 0, 1.03), 3)],
      ["hexagon", hand(aroundPolygon(ngon(C, 120, 6, 0), 0.05, 1.04), 4)],
      ["pentagon", hand(aroundPolygon(ngon(C, 120, 5, -90), 0.05, 1.04), 5)],
    ]
    for (const [kind, pts] of cases) {
      const r = await rec.recognize(pts)
      expect(r.best).toBe(kind)
      expect(r.kind).toBe(kind)
      expect(r.shape?.kind).toBe(kind as never)
      expect(r.samples).toHaveLength(N_POINTS)
      expect(Object.keys(r.probs)).toEqual(MODEL_META.classes)
    }
    const r = await rec.recognize(hand(aroundPolygon(rectCorners(C, 260, 150), 0.1, 1.06), 1))
    const box = r.shape as Box
    expect(Math.abs(box.width - 260)).toBeLessThan(10)
    expect(Math.abs(box.height - 150)).toBeLessThan(10)
  })

  test("arrow: the head found by the model puts the tip at `to`", async () => {
    const { pts } = arrowStroke({ x: 100, y: 400 }, { x: 450, y: 180 }, 0.2)
    const r = await rec.recognize(hand(pts, 6, 0.004, 0.001))
    expect(r.kind).toBe("arrow")
    const a = r.shape as Arrow
    expect(Math.hypot(a.to.x - 450, a.to.y - 180)).toBeLessThan(12)
    expect(Math.hypot(a.from.x - 100, a.from.y - 400)).toBeLessThan(12)
    expect(a.startHead).toBe(false)
  })

  test("a squiggle stays ink", async () => {
    const pts = Array.from({ length: 200 }, (_, i) => ({ x: 100 + i * 2, y: 300 + 40 * Math.sin(i / 6) }))
    const r = await rec.recognize(pts)
    expect(r.kind).toBeNull()
    expect(r.shape).toBeNull()
  })

  test("concurrent calls are queued, a single point is not run", async () => {
    const strokes = [1, 2, 3, 4].map((s) => hand(aroundPolygon(rectCorners(C, 200, 120), 0, 1.04), s))
    const rs = await Promise.all(strokes.map((p) => rec.recognize(p)))
    expect(rs.map((r) => r.kind)).toEqual(["rectangle", "rectangle", "rectangle", "rectangle"])
    const dot = await rec.recognize([{ x: 1, y: 1 }])
    expect(dot.kind).toBeNull()
    expect(dot.best).toBe("none")
  })
})
