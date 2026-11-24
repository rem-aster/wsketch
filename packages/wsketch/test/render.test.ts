import { expect, test } from "bun:test"
import { arrowHeads, boxCorners, isClosed, outline, toSvgPath, type Arrow, type Box, type Line } from "../src/index.js"

test("box corners: rectangle from top-left, diamond from top, both clockwise", () => {
  const r: Box = { kind: "rectangle", center: { x: 100, y: 50 }, width: 40, height: 20, rotation: 0 }
  expect(boxCorners(r)).toEqual([{ x: 80, y: 40 }, { x: 120, y: 40 }, { x: 120, y: 60 }, { x: 80, y: 60 }])
  const d: Box = { ...r, kind: "diamond" }
  expect(boxCorners(d)).toEqual([{ x: 100, y: 40 }, { x: 120, y: 50 }, { x: 100, y: 60 }, { x: 80, y: 50 }])
  const turned = boxCorners({ ...r, rotation: Math.PI / 2 })
  expect(turned[0].x).toBeCloseTo(110)
  expect(turned[0].y).toBeCloseTo(30)
})

test("svg paths", () => {
  const line: Line = { kind: "line", from: { x: 0, y: 0 }, to: { x: 10.5, y: 3 } }
  expect(toSvgPath(line)).toBe("M0 0L10.5 3")
  expect(toSvgPath({ ...line, control: [{ x: 1, y: 2 }, { x: 3, y: 4 }] })).toBe("M0 0C1 2 3 4 10.5 3")
  const e: Box = { kind: "ellipse", center: { x: 50, y: 50 }, width: 40, height: 20, rotation: 0 }
  expect(toSvgPath(e)).toBe("M70 50A20 10 0 0 1 30 50A20 10 0 0 1 70 50Z")
  expect(toSvgPath({ kind: "triangle", points: [{ x: 0, y: 0 }, { x: 4, y: 4 }, { x: 0, y: 4 }] })).toBe("M0 0L4 4L0 4Z")
})

test("arrowheads point back along the shaft", () => {
  const a: Arrow = { kind: "arrow", from: { x: 0, y: 0 }, to: { x: 100, y: 0 }, startHead: true, headSize: 10 }
  const [end, start] = arrowHeads(a)
  expect(end[1]).toEqual(a.to)
  expect(end[0].x).toBeCloseTo(100 - 10 * Math.cos((28 * Math.PI) / 180))
  expect(end[0].y).toBeCloseTo(-end[2].y)
  expect(start[1]).toEqual(a.from)
  expect(start[0].x).toBeGreaterThan(0)
  expect(toSvgPath(a).match(/M/g)).toHaveLength(3)
})

test("outlines", () => {
  const e: Box = { kind: "ellipse", center: { x: 0, y: 0 }, width: 20, height: 10, rotation: 0 }
  expect(outline(e, 32)).toHaveLength(32)
  expect(isClosed(e)).toBe(true)
  const curve: Line = { kind: "line", from: { x: 0, y: 0 }, to: { x: 9, y: 0 }, control: [{ x: 3, y: 5 }, { x: 6, y: 5 }] }
  const pts = outline(curve, 10)
  expect(pts).toHaveLength(11)
  expect(pts[10]).toEqual(curve.to)
  expect(isClosed(curve)).toBe(false)
})
