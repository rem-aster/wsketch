<script lang="ts">
  import { Button } from "@ui-sdk/svelte/button"
  import { Meter, MeterIndicator, MeterTrack, MeterValue } from "@ui-sdk/svelte/meter"
  import { Switch, SwitchThumb } from "@ui-sdk/svelte/switch"
  import * as ort from "onnxruntime-web/wasm"
  import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"
  import { onMount } from "svelte"
  import {
    Recognizer,
    fitClassification,
    toSvgPath,
    type Classification,
    type FitOptions,
    type Pt,
    type Shape,
  } from "wsketch"
  import modelUrl from "wsketch/model.onnx?url"

  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = { wasm: wasmUrl }

  const RU: Record<string, string> = {
    none: "не фигура", line: "линия", arrow: "стрелка", rectangle: "прямоугольник",
    ellipse: "эллипс", triangle: "треугольник", diamond: "ромб", pentagon: "пятиугольник",
    hexagon: "шестиугольник", star: "звезда", heart: "сердце", cloud: "облако", scribble: "каракули",
  }
  const STORE = "wsketch-corrections"
  // the shape as drawn: no snapping to the axes, to circles or to regular polygons
  const AS_DRAWN: FitOptions = { snapAngle: 0, snapRatio: 1, regular: 0 }

  type Stroke = { pts: Pt[]; result?: Classification }
  type Correction = { label: string; predicted: Record<string, string>; points: number[][] }

  let rec = $state.raw<Recognizer | null>(null)
  let error = $state("")
  let strokes = $state.raw<Stroke[]>([])
  let replace = $state(true)
  let tidy = $state(true)
  let showInk = $state(true)
  let showPoints = $state(false)
  let corrections = $state.raw<Correction[]>(load())
  let savedAs = $state("")

  // refitting is cheap, so the switches apply to strokes already drawn without running the model again
  let shapes = $derived(strokes.map((s) => (s.result ? fitClassification(s.result, s.pts, tidy ? {} : AS_DRAWN) : null)))
  let last = $derived(strokes.at(-1)?.result)
  let lastShape = $derived(shapes.at(-1) ?? null)
  let classes = $derived(rec?.meta.classes ?? [])

  let canvas: HTMLCanvasElement
  let current: Pt[] | null = null

  const pct = (x: number) => Math.round(x * 100)

  function load(): Correction[] {
    try {
      return JSON.parse(localStorage.getItem(STORE) ?? "[]")
    } catch {
      return []
    }
  }

  onMount(() => {
    Recognizer.load({ ort, model: modelUrl, sessionOptions: { executionProviders: ["wasm"] } }).then(
      (r) => (rec = r),
      (e) => (error = String(e)),
    )
    const ro = new ResizeObserver(draw)
    ro.observe(canvas)
    const scheme = matchMedia("(prefers-color-scheme: dark)")
    scheme.addEventListener("change", draw)
    return () => {
      ro.disconnect()
      scheme.removeEventListener("change", draw)
    }
  })

  $effect(() => {
    void shapes, replace, showInk, showPoints
    draw()
  })

  function point(e: PointerEvent): Pt {
    const r = canvas.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  function down(e: PointerEvent) {
    if (e.button !== 0 || !rec) return
    canvas.setPointerCapture(e.pointerId)
    current = [point(e)]
    savedAs = ""
  }

  function move(e: PointerEvent) {
    if (!current) return
    const fine = e.getCoalescedEvents?.() ?? []
    for (const c of fine.length ? fine : [e]) current.push(point(c))
    draw()
  }

  async function up() {
    const pts = current
    current = null
    if (!pts || !rec) return
    let length = 0
    for (let i = 1; i < pts.length; i++) length += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
    if (length < 12) return draw() // a tap, not a stroke
    const stroke: Stroke = { pts }
    strokes = [...strokes, stroke]
    stroke.result = await rec.classify(pts)
    strokes = [...strokes]
  }

  function undo() {
    strokes = strokes.slice(0, -1)
    savedAs = ""
  }

  function clear() {
    strokes = []
    savedAs = ""
  }

  function correct(label: string) {
    const s = strokes.at(-1)
    if (!s?.result) return
    corrections = [...corrections, {
      label,
      predicted: { light: s.result.best },
      points: s.pts.map((p) => [+p.x.toFixed(1), +p.y.toFixed(1)]),
    }]
    localStorage.setItem(STORE, JSON.stringify(corrections))
    savedAs = label
  }

  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(corrections)], { type: "application/json" }))
    const a = Object.assign(document.createElement("a"), { href: url, download: "wsketch-corrections.json" })
    a.click()
    URL.revokeObjectURL(url)
  }

  /** What the shape turned out to be, in a few words. */
  function describe(s: Shape): string {
    const px = (v: number) => Math.round(v)
    const turn = (r: number) => (r ? `, поворот ${Math.round((r * 180) / Math.PI)}°` : "")
    switch (s.kind) {
      case "line":
      case "arrow": {
        const form = s.control ? "кривая" : `прямая, ${px(Math.hypot(s.to.x - s.from.x, s.to.y - s.from.y))} px`
        return s.kind === "line" ? form : `${form}, ${s.startHead ? "наконечники с двух сторон" : "наконечник в конце"}`
      }
      case "ellipse":
        return s.width === s.height ? `круг ⌀ ${px(s.width)}` : `${px(s.width)} × ${px(s.height)}${turn(s.rotation)}`
      case "rectangle":
      case "diamond":
        return `${px(s.width)} × ${px(s.height)}${s.width === s.height ? ", равные стороны" : ""}${turn(s.rotation)}`
      default: {
        const n = s.points.length
        const sides = s.points.map((p, k) => Math.hypot(p.x - s.points[(k + 1) % n].x, p.y - s.points[(k + 1) % n].y))
        const regular = Math.max(...sides) - Math.min(...sides) < 1e-6
        return `${n} ${n < 5 ? "вершины" : "вершин"}${regular ? ", правильный" : ""}`
      }
    }
  }

  function draw() {
    if (!canvas) return
    const dpr = devicePixelRatio || 1
    const { width, height } = canvas.getBoundingClientRect()
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
    }
    const dark = matchMedia("(prefers-color-scheme: dark)").matches
    const ink = dark ? "#f2f2f2" : "#1c1c1c"
    const faint = dark ? "#8a8a8a" : "#767676"
    const ghost = dark ? "rgba(242,242,242,0.22)" : "rgba(28,28,28,0.18)"
    const accent = dark ? "#6aa5ff" : "#1f6feb"
    const alert = dark ? "#ff7b72" : "#d1242f"
    const g = canvas.getContext("2d")!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, width, height)
    g.lineCap = g.lineJoin = "round"
    g.font = "12px ui-sans-serif, system-ui, sans-serif"

    const all = current ? [...strokes, { pts: current } as Stroke] : strokes
    all.forEach((s, i) => {
      const isLast = i === all.length - 1
      const shape = replace ? shapes[i] : null
      if (!shape || showInk) {
        g.strokeStyle = shape ? ghost : isLast ? ink : faint
        g.lineWidth = shape ? 5 : isLast ? 2.5 : 1.5
        g.beginPath()
        s.pts.forEach((p, k) => (k ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y)))
        g.stroke()
      }
      if (shape) {
        g.strokeStyle = isLast ? accent : ink
        g.lineWidth = 2.5
        g.stroke(new Path2D(toSvgPath(shape)))
      }
      const r = s.result
      if (!r) return
      if (isLast && showPoints) {
        r.samples.forEach((p, k) => {
          g.fillStyle = r.head[k] > 0.5 ? alert : accent
          g.beginPath()
          g.arc(p.x, p.y, r.head[k] > 0.5 ? 2.6 : 1.6, 0, 7)
          g.fill()
        })
      }
      let top = Infinity, left = Infinity
      for (const p of s.pts) { top = Math.min(top, p.y); left = Math.min(left, p.x) }
      g.fillStyle = isLast ? ink : faint
      g.fillText(`${RU[r.kind ?? "none"]} · ${pct(r.confidence)}%`, left, Math.max(12, top - 8))
    })
  }
</script>

<main>
  <section class="board">
    <canvas
      bind:this={canvas}
      onpointerdown={down}
      onpointermove={move}
      onpointerup={up}
      onpointercancel={up}
      aria-label="Холст для рисования"
    ></canvas>
    {#if error}
      <p class="hint error">Модель не загрузилась: {error}</p>
    {:else if !rec}
      <p class="hint">Загружаю модель…</p>
    {:else if strokes.length === 0}
      <p class="hint">Нарисуйте фигуру одним штрихом</p>
    {/if}
  </section>

  <aside>
    <header>
      <h1>wsketch</h1>
      <p class="muted">
        Штрих превращается в фигуру. Модель работает в браузере{rec ? `, ${(rec.bytes / 1024).toFixed(0)} КБ` : ""}.
      </p>
    </header>

    <div class="verdict" data-accepted={last?.kind ? "" : undefined}>
      <span class="tag">Последний штрих</span>
      {#if last}
        <strong>{RU[last.kind ?? "none"]}</strong>
        <span class="muted">
          {#if !last.kind && last.best !== "none"}
            похоже на «{RU[last.best]}», {pct(last.confidence)}% ниже порога
          {:else}
            уверенность {pct(last.confidence)}%
          {/if}
          · {last.ms.toFixed(1)} мс
        </span>
        {#if lastShape}
          <span class="shape">{describe(lastShape)}</span>
        {:else}
          <span class="muted">остаётся штрихом</span>
        {/if}
      {:else}
        <strong class="muted">—</strong>
        <span class="muted">ждёт первого штриха</span>
      {/if}
    </div>

    <div class="settings">
      <label data-slot="label" class="setting">
        Заменять штрих фигурой
        <Switch bind:checked={replace}><SwitchThumb /></Switch>
      </label>
      <label data-slot="label" class="setting">
        Выравнивать: по осям, в круг, в правильный многоугольник
        <Switch bind:checked={tidy} disabled={!replace}><SwitchThumb /></Switch>
      </label>
      <label data-slot="label" class="setting">
        Показывать исходный штрих под фигурой
        <Switch bind:checked={showInk} disabled={!replace}><SwitchThumb /></Switch>
      </label>
      <label data-slot="label" class="setting">
        Показывать 128 точек модели
        <Switch bind:checked={showPoints}><SwitchThumb /></Switch>
      </label>
      {#if showPoints}
        <p class="muted small">Красным отмечены точки, которые модель считает наконечником стрелки.</p>
      {/if}
    </div>

    <div class="meters">
      {#each classes as name (name)}
        <div class="row" data-top={last?.best === name ? "" : undefined}>
          <span class="name">{RU[name] ?? name}</span>
          <Meter value={(last?.probs[name] ?? 0) * 100} aria-label={RU[name] ?? name}>
            <MeterTrack><MeterIndicator /></MeterTrack>
            <MeterValue />
          </Meter>
        </div>
      {/each}
    </div>

    <div class="fix">
      <p>Модель ошиблась? Отметьте, что это было на самом деле:</p>
      <div class="chips">
        {#each classes as name (name)}
          <Button disabled={!last} onclick={() => correct(name)} data-chosen={savedAs === name ? "" : undefined}>
            {RU[name] ?? name}
          </Button>
        {/each}
      </div>
      <p class="muted small">
        {#if savedAs}Сохранено как «{RU[savedAs]}». {/if}Собрано примеров: {corrections.length}. Их можно добавить в обучение.
      </p>
      <Button disabled={corrections.length === 0} onclick={download}>Скачать примеры</Button>
    </div>

    <footer>
      <Button disabled={strokes.length === 0} onclick={undo}>Отменить штрих</Button>
      <Button data-variant="solid" onclick={clear}>Очистить холст</Button>
    </footer>
  </aside>
</main>
