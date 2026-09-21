# Dr Moagi Hyperscale Geometric VM

A bounded browser demonstration derived from Matladi Maxwell Moagi's supplied
64,000-particle inward-burst concept. It preserves the nested torus rings,
wireframe core, particle shell and 5% contraction target.

Originator of the Dr Moagi 3D Ephemeral-Notion Intelligence Framework:
Matladi Maxwell Moagi (Lord-Xido). [Canonical provenance](../../docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md).

## Run

Open `index.html` directly, or serve this directory:

```sh
python -m http.server 8000 --directory apps/hyperscale-geometric-vm
```

Then open `http://localhost:8000`. No build step is needed. Styles, model,
controls and a Canvas renderer are embedded. Three.js r128 is loaded
asynchronously from the supplied CDN for the WebGL renderer. If the CDN fails,
WebGL is unavailable or its context is lost, the Canvas renderer remains usable.

- **Engage inward burst:** contracts the shell over 1.2 seconds of active animation time.
- **Pause / Resume:** freezes / resumes geometry, rotation and animation scheduling.
- **Reset:** restores the original seeded geometry, camera and counters; preserves pause state.
- Drag or use arrow keys to orbit; wheel, pinch or `+` / `-` to zoom.
- Hidden tabs suspend animation; returning does not catch up hidden time.
- Reduced-motion preference starts paused; the burst button explicitly starts motion.

## What executes

For progress `p = min(elapsed / 1.2, 1)` and `e = sin(pi p / 2)`:

```text
x_i(p) = (1 - 0.95 e) x_i(0) + 0.1 (1 - e) w_i(elapsed)
```

`w_i` is a deterministic bounded trigonometric vector. At completion the jitter
is zero and `x_i = 0.05 x_i(0)` up to Float32 rounding. This is a finite
interpolation, not a learned optimizer or proof of fixed-point convergence.
Positions are generated with seed 7, uniform direction and uniform volume
within a shell of radius 9–15 scene units. Work and storage are `O(N)` with
`N = 64,000`. Only an active burst updates particle positions; idle motion
rotates the rendered cloud.

| Readout | Definition |
|---|---|
| Symbolic scale | `a^(a^a)`, `a = 1,000,000`; text only, never evaluated or executed |
| Simulated particles | 64,000 positions in a Float32 buffer |
| Rendered particles | 64,000 submitted to WebGL, or a 4,000-particle sample in Canvas; camera clipping can hide some |
| Render rate | completed animation-frame intervals divided by their elapsed wall time |
| Position updates / second | actual particle update count divided by elapsed sampled wall time; zero while idle |
| Target RMSE | square root of mean squared coordinate error relative to `0.05 x(0)` |
| 64-bin radial summary | mean Euclidean radius of particles in each `index modulo 64` group |

Readouts sample at about 4 Hz. `window.getHyperscaleTelemetry()` returns a
copied snapshot; RMSE and radial summary describe the latest measurement.
The first five summary values appear numerically, with all 64 plotted as bars.
The summary is a lossy descriptive statistic, not trained latent weights.

This app has no bytecode interpreter, quantum execution, trained autoencoder,
backend integration or authority over the canonical Jarvis-X runtime. The
symbolic exponent tower is not a measured throughput. Rendering contraction
does not establish computational acceleration, intelligence or information
compression. No telemetry is transmitted.

## Validation

Run the dependency-free numerical and script-parsing tests with Node:

```sh
node --test apps/hyperscale-geometric-vm/test_model.cjs
```

Tests cover seed replay, shell bounds, frame-rate-independent positions,
analytic contraction endpoints, continuous rotation, measured summary/RMSE,
in-place reset, counters and invalid input rejection.

The optional Playwright smoke script exercises controls, offline fallback,
desktop/mobile layouts, reduced motion and WebGL context loss:

```sh
npm install --no-save playwright@1.62.1 three@0.128.0
npx playwright install chromium --only-shell
node apps/hyperscale-geometric-vm/test_browser.cjs
```

`PLAYWRIGHT_MODULE` may point to an existing Playwright installation, and
`THREE_JS_PATH` to `three.min.js` from r128. The browser
smoke script serves the app on loopback and closes the server on completion.
The focused GitHub Actions workflow runs both suites and saves screenshots.
