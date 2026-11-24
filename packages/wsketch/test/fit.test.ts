import { describe, expect, test } from "bun:test"
import { fitShape, type Arrow, type Box, type Line, type Polygon, type Pt } from "../src/index.js"
import { aroundEllipse, aroundPolygon, arrowStroke, deg, hand, ngon, rectCorners } from "./strokes.js"

const near = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol)
const nearPt = (a: Pt, b: Pt, tol: number) => expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeLessThanOrEqual(tol)
const C = { x: 400, y: 300 }

/** Every true corner has a fitted corner within `tol`. */
function sameCorners(got: Pt[], want: Pt[], tol: number) {
  expect(got).toHaveLength(want.length)
  for (const w of want) expect(Math.min(...got.map((g) => Math.hypot(g.x - w.x, g.y - w.y)))).toBeLessThanOrEqual(tol)
}

describe("rectangle", () => {
  test("axis-aligned, started mid-side with overshoot", () => {
    const s = fitShape("rectangle", hand(aroundPolygon(rectCorners(C, 240, 140), 0.13, 1.1), 1)) as Box
    expect(s.kind).toBe("rectangle")
    expect(s.rotation).toBe(0)
    near(s.width, 240, 8)
    near(s.height, 140, 8)
    nearPt(s.center, C, 6)
  })

  test("tilted 20 degrees stays tilted", () => {
    const s = fitShape("rectangle", hand(aroundPolygon(rectCorners(C, 200, 120, deg(20)), 0, 1.04), 2)) as Box
    near(s.rotation, deg(20), deg(2))
    near(s.width, 200, 8)
    near(s.height, 120, 8)
  })

  test("tilted 5 degrees snaps to the axes; a near-square becomes a square", () => {
    const s = fitShape("rectangle", hand(aroundPolygon(rectCorners(C, 150, 143, deg(5)), 0.5, 0.97), 3)) as Box
    expect(s.rotation).toBe(0)
    expect(s.width).toBe(s.height)
    near(s.width, 146, 8)
  })

  test("standing rectangle keeps width < height", () => {
    const s = fitShape("rectangle", hand(aroundPolygon(rectCorners(C, 90, 220, deg(-4)), 0.3, 1.02), 4)) as Box
    expect(s.rotation).toBe(0)
    near(s.width, 90, 8)
    near(s.height, 220, 8)
  })
})

describe("ellipse", () => {
  test("rotated ellipse", () => {
    const s = fitShape("ellipse", hand(aroundEllipse(C, 150, 70, deg(30), 0.2, 1.06), 5)) as Box
    near(s.rotation, deg(30), deg(3))
    near(s.width, 300, 10)
    near(s.height, 140, 10)
    nearPt(s.center, C, 6)
  })

  test("a nearly round one becomes a circle", () => {
    const s = fitShape("ellipse", hand(aroundEllipse(C, 100, 94, deg(50), 0.7, 0.96), 6)) as Box
    expect(s.width).toBe(s.height)
    expect(s.rotation).toBe(0)
    near(s.width, 194, 10)
  })

  test("traced twice", () => {
    const s = fitShape("ellipse", hand(aroundEllipse(C, 80, 140, 0, 0, 2.02), 7)) as Box
    expect(s.rotation).toBe(0)
    near(s.width, 160, 10)
    near(s.height, 280, 10)
  })
})

describe("diamond", () => {
  test("upright diamond", () => {
    const V = [{ x: C.x, y: C.y - 120 }, { x: C.x + 80, y: C.y }, { x: C.x, y: C.y + 120 }, { x: C.x - 80, y: C.y }]
    const s = fitShape("diamond", hand(aroundPolygon(V, 0.05, 1.05), 8)) as Box
    expect(s.kind).toBe("diamond")
    expect(s.rotation).toBe(0)
    near(s.width, 160, 8)
    near(s.height, 240, 8)
    nearPt(s.center, C, 6)
  })
})

describe("polygons", () => {
  test("irregular triangle keeps its corners", () => {
    const V = [{ x: 300, y: 400 }, { x: 560, y: 375 }, { x: 330, y: 250 }]
    const s = fitShape("triangle", hand(aroundPolygon(V, 0.4, 1.03), 9), { snapAngle: 0 }) as Polygon
    sameCorners(s.points, V, 8)
  })

  test("nearly equilateral triangle becomes equilateral", () => {
    const V = [{ x: 300, y: 400 }, { x: 520, y: 380 }, { x: 350, y: 200 }]
    const s = fitShape("triangle", hand(aroundPolygon(V, 0.4, 1.03), 9)) as Polygon
    const sides = s.points.map((p, k) => Math.hypot(p.x - s.points[(k + 1) % 3].x, p.y - s.points[(k + 1) % 3].y))
    near(Math.max(...sides), Math.min(...sides), 1e-6)
  })

  test("triangle with a nearly flat base gets a flat base, the apex stays", () => {
    const V = [{ x: 300, y: 400 }, { x: 520, y: 390 }, { x: 330, y: 230 }]
    const s = fitShape("triangle", hand(aroundPolygon(V, 0, 1.04), 10)) as Polygon
    const flat = s.points.some((p, k) => {
      const q = s.points[(k + 1) % 3]
      return Math.abs(p.y - q.y) < 1e-6
    })
    expect(flat).toBe(true)
    nearPt(s.points[0], V[2], 8)
  })

  test("house pentagon: walls vertical, floor horizontal", () => {
    const V = [{ x: 300, y: 450 }, { x: 497, y: 445 }, { x: 503, y: 300 }, { x: 400, y: 200 }, { x: 298, y: 305 }]
    const s = fitShape("pentagon", hand(aroundPolygon(V, 0.3, 1.03), 19)) as Polygon
    sameCorners(s.points, V, 10)
    const [roof, right, br, bl, left] = s.points
    expect(right.x).toBe(br.x)
    expect(left.x).toBe(bl.x)
    expect(br.y).toBe(bl.y)
    expect(roof.y).toBeLessThan(right.y)
  })

  test("nearly regular pentagon becomes regular and upright", () => {
    const V = ngon(C, 120, 5, -86).map((p, k) => ({ x: p.x + [4, -6, 3, 5, -2][k], y: p.y + [-3, 4, 6, -4, 2][k] }))
    const s = fitShape("pentagon", hand(aroundPolygon(V, 0.1, 1.04), 11)) as Polygon
    expect(s.points).toHaveLength(5)
    const top = s.points[0]
    const cx = s.points.reduce((a, p) => a + p.x, 0) / 5
    near(top.x, cx, 1e-6) // a corner exactly above the centre
    const r = s.points.map((p) => Math.hypot(p.x - cx, p.y - s.points.reduce((a, q) => a + q.y, 0) / 5))
    near(Math.max(...r), Math.min(...r), 1e-6)
  })

  test("flat-topped hexagon", () => {
    const V = ngon(C, 110, 6, 3)
    const s = fitShape("hexagon", hand(aroundPolygon(V, 0.02, 1.05), 12)) as Polygon
    sameCorners(s.points, ngon(C, 110, 6, 0), 10)
    near(s.points[0].y, s.points[1].y, 1e-6) // the top side is horizontal
  })

  test("wide hexagon is not forced to be regular", () => {
    const V = ngon(C, 100, 6, 0, 1.6)
    const s = fitShape("hexagon", hand(aroundPolygon(V, 0.5, 1.02), 13)) as Polygon
    sameCorners(s.points, V, 12)
  })

  test("corners come clockwise on screen from the top", () => {
    const V = [{ x: 300, y: 400 }, { x: 350, y: 200 }, { x: 520, y: 380 }] // counter-clockwise
    const s = fitShape("triangle", hand(aroundPolygon(V, 0, 1.02), 14), { snapAngle: 0, regular: 0 }) as Polygon
    nearPt(s.points[0], { x: 350, y: 200 }, 10)
    nearPt(s.points[1], { x: 520, y: 380 }, 10)
  })
})

describe("line", () => {
  test("almost horizontal becomes horizontal", () => {
    const s = fitShape("line", hand([{ x: 100, y: 300 }, { x: 500, y: 330 }], 15, 0.01)) as Line
    expect(s.control).toBeUndefined()
    near(s.from.y, s.to.y, 1e-9)
    near(s.to.x - s.from.x, 401, 8)
  })

  test("a slanted line keeps its angle", () => {
    const s = fitShape("line", hand([{ x: 100, y: 100 }, { x: 400, y: 300 }], 16, 0.01)) as Line
    nearPt(s.from, { x: 100, y: 100 }, 8)
    nearPt(s.to, { x: 400, y: 300 }, 8)
  })

  test("a clearly bent stroke becomes a curve", () => {
    const arc = Array.from({ length: 50 }, (_, i) => {
      const a = deg(200 + (140 * i) / 49)
      return { x: C.x + 200 * Math.cos(a), y: C.y + 200 * Math.sin(a) }
    })
    const s = fitShape("line", hand(arc, 17, 0.008)) as Line
    expect(s.control).toBeDefined()
    nearPt(s.from, arc[0], 8)
    nearPt(s.to, arc[49], 8)
  })
})

describe("arrow", () => {
  const a = { x: 100, y: 200 }, b = { x: 500, y: 215 }

  test("head at the end", () => {
    const { pts, head } = arrowStroke(a, b)
    const s = fitShape("arrow", pts, { head }) as Arrow
    expect(s.startHead).toBe(false)
    nearPt(s.to, { x: b.x, y: 207.5 }, 4) // snapped to horizontal around the middle
    nearPt(s.from, { x: a.x, y: 207.5 }, 4)
    near(s.headSize, 100, 8)
  })

  test("head drawn first is turned around", () => {
    const { pts, head } = arrowStroke(a, b)
    const s = fitShape("arrow", [...pts].reverse(), { head: [...head].reverse() }) as Arrow
    expect(s.startHead).toBe(false)
    near(s.to.x, b.x, 4)
    near(s.from.x, a.x, 4)
  })

  test("heads at both ends", () => {
    const one = arrowStroke(b, a)
    const two = arrowStroke(a, b)
    const pts = [...[...one.pts].reverse(), ...two.pts.slice(60)]
    const head = [...[...one.head].reverse(), ...two.head.slice(60)]
    const s = fitShape("arrow", pts, { head }) as Arrow
    expect(s.startHead).toBe(true)
    near(s.from.x, a.x, 4)
    near(s.to.x, b.x, 4)
  })

  test("without the model's head output the farthest point is the tip", () => {
    const { pts } = arrowStroke({ x: 100, y: 100 }, { x: 300, y: 400 })
    const s = fitShape("arrow", pts) as Arrow
    nearPt(s.to, { x: 300, y: 400 }, 4)
    nearPt(s.from, { x: 100, y: 100 }, 4)
  })
})

describe("edge cases", () => {
  test("kinds without a shape and strokes without extent give null", () => {
    expect(fitShape("none", [{ x: 0, y: 0 }, { x: 10, y: 10 }])).toBeNull()
    expect(fitShape("star", [{ x: 0, y: 0 }, { x: 10, y: 10 }])).toBeNull()
    expect(fitShape("rectangle", [{ x: 5, y: 5 }, { x: 5, y: 5 }])).toBeNull()
    expect(fitShape("line", [{ x: 5, y: 5 }])).toBeNull()
  })

  test("options can switch snapping off", () => {
    const pts = hand(aroundPolygon(rectCorners(C, 200, 120, deg(5)), 0, 1.03), 18)
    const s = fitShape("rectangle", pts, { snapAngle: 0 }) as Box
    near(s.rotation, deg(5), deg(1.5))
  })
})
