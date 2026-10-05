# Chromium inward feedback in 3D

A self-contained, offline browser demonstration inspired by the supplied Chromium
stack diagram. It executes a bounded rendering pipeline, projects its stages and
live framebuffer in 3D, and automatically tests rendering policies against an
independent pixel reference before adopting them.

## Run

Open `index.html` directly. No build, CDN, network, or npm dependency is required
by the app. For an HTTP origin with persistent local policy storage:

```sh
cd apps/chromium-inward-3d
node serve.cjs
```

Open `http://127.0.0.1:8000`. `PORT` selects another local port. The server binds
to loopback, serves only `/`, `/index.html`, and `/health.json`, and blocks external
runtime assets with its content security policy. The health endpoint checks the
static app service; browser renderer state belongs to each tab.

- **Auto feedback** starts one finite optimization pass after a verified frame.
- **Inward optimise / Stop optimisation** starts or cancels a pass.
- **Run / Pause**, **Step**, and **Inspect stage** expose the actual transformations.
- **Change heading** invalidates affected rendering work; **Scroll +24 px** reuses tiles.
- Drag to orbit the stack; tap a numbered stage or use the stage selector.
- **Exploded view** separates the browser, renderer, and Viz slabs.
- The disclosures expose the framebuffer, stage receipts, and optimization decisions.

## Executed pipeline

| Stage | Transformation | Result |
|---|---|---|
| Navigation | Route the built-in document | HTML/CSS source |
| Parse & style | DOMParser and supported rule matching | Styled node tree |
| Layout | Block placement and text wrapping | Boxes |
| Pre-paint & paint | Record draw and clip/scroll properties | Display list |
| Commit | Copy the display list | Immutable compositor input |
| Layerize & tile | Partition the document surface | Adaptive tile jobs |
| Raster | Execute intersecting commands, reuse matching cached tiles | RGBA canvases |
| Activate | Reference the raster tiles | Draw quads |
| Aggregate & draw | Compose page and browser chrome | Back buffer |
| Present | Compare every RGBA pixel to the full-list reference | Verified front-buffer swap |

The framebuffer is 320 × 226, including 26px of browser chrome. The document is a
small built-in HTML/CSS subset; this app does not navigate arbitrary websites or
implement all browser layout, isolation, GPU, or scheduling behavior. The three
process slabs represent software stages in one JavaScript execution context.
Canvas2D projects real 3D coordinates; there is no WebGL dependency or Chromium binary.
The logical 1024³ coordinate map is sparse and does not allocate a billion cells.

## Inward optimization contract

The mutable policy is `{tile, tightBounds, visibleOnly, reuseDocument}`. Tile edges
are limited to 32, 64, or 128 pixels. A pass probes neighboring policies on six
requests: both built-in headings, two scroll offsets, unchanged repetition, and
return to the initial heading. The objective is lexicographic: fewer counted
drawing calls, then fewer stage executions, then fewer rasterized pixels.

Each proposal runs in isolated buffers and must have zero differing pixels across
all 433,920 checked pixels, no more than 512 tiles, and no more than 2,000,000
counted probe pixel bytes. A strictly better proposal is then applied to the live
renderer and checked again. A failed proposal retains the previous policy. A bad
live policy retries the default 64px policy; any remaining mismatch holds the
previous verified front buffer. Source/layout correctness is outside the pixel
reference: both paths share the document parser, layout, and display list.

If native Canvas-to-Canvas tile transport changes pixel values, the compositor
retries using the raster tiles' own RGBA bytes and explicit viewport clipping.
This preserves the independent full-list reference and the zero-difference gate;
reference pixels are never substituted into candidate output. The snapshot names
the selected `compositionBackend`; the Present receipt records any fallback.

Each pass allows at most eight candidate probes plus its baseline. The 2,500ms
compute budget is checked between probes and is a soft bound, so one probe can
overshoot it. Probes are synchronous bounded work; timers yield between them.
The scene stops work when hidden or outside view. There is no endless idle search.
A tested neighborhood fixed policy is not a global optimum.

Feedback packets rotate and contract in three spatial coordinates by `0.68^depth`
toward the controller core. Ω stores a four-value moving average of observed work,
tile size, stage work, and pixel residual. This is a descriptive controller memory,
not trained model weights. Optimization changes bounded policies, not source code.
Counted calls include drawing and tile-copy submissions (`drawImage` or
`putImageData`) and exclude the independent reference render, verification, 3D view, and
tuning overhead. These counts are not FPS, latency, GPU throughput, or a speedup;
probe elapsed time is reported separately.

Policy and Ω use the localStorage key `chromium-3d-inward-policy-v1`. Unsupported
tile sizes, invalid JSON, and unavailable storage fall back safely. Loaded policies
are verified before a new frame is presented. Storage is local to the browser
origin; clearing it restores the default. No telemetry is sent.

## Diagnostics

```js
const api = document.getElementById('chromium-rendering-3d').crEmulator;
api.snapshot();             // copied frame, policy, work, and geometry state
api.feedback();             // copied policy trials, decisions, Ω, and contraction
api.inspect(6);             // Raster stage receipt
api.trace();                // bounded stage trace
api.pixels();               // copy of the current front-buffer RGBA bytes
api.probePolicy({tile: 64}); // isolated measurement; does not mutate the live frame
```

The snapshot distinguishes core pixel storage, view pixel storage, and logical
cells. These pixel-byte counters omit DOM, JavaScript objects, driver overhead,
and temporary comparison arrays; they are not total process RSS.

## Validation and packaging

Node 22+ and Python 3 are required for the developer suites. Dependencies are
exactly pinned in `package-lock.json`; they are used only for validation.

```sh
cd apps/chromium-inward-3d
npm ci
npm test
npx playwright install chromium webkit --with-deps
npm run test:browser
BROWSER_NAME=webkit npm run test:browser
npm run package
```

The Canvas suite uses real software rasterization with a small Python HTMLParser
DOM adapter. It checks pixel identity, bounded convergence, independent probes,
document/tile reuse, mutation, scroll, 3D contraction, corrupted/blocked storage,
candidate rejection, live rollback, byte-copy fallback/clipping, and cancellation. Native browser tests run
seven profiles per engine: desktop, mobile, injected bad tile rendering, poisoned
stored policy, corrupt storage, blocked storage, and direct-file offline use.
They also check responsive layout at 320px, controls, persistence, and pointer
orbit. Chromium mobile orbit uses native CDP touch delivery; WebKit mobile controls
use taps and orbit uses mouse delivery. Frame hashes are compared only within one
engine/environment, since fonts and rasterization vary across platforms.

`test-results/` contains machine-readable evidence and screenshots. The scoped
GitHub Actions workflow runs both suites on changes and manual dispatch, and
uploads browser evidence even on failure. The offline bundle is published as a
workflow artifact only after Canvas, Chromium, and WebKit checks pass. It contains
the HTML, optional local server, this README, a versioned manifest, and SHA-256 sums.
`dist/chromium-inward-3d/` can also be generated locally without npm dependencies.

This isolated demonstration has no authority over the canonical Jarvis-X runtime.
Its finite policy search does not implement unrestricted self-modification,
general intelligence, or the full self-hosting runtime specification.

Architecture reference: [Chromium RenderingNG](https://developer.chrome.com/docs/chromium/renderingng-architecture).
