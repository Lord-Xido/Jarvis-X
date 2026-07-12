# Neural Core GPU Particle Field

`examples/neural_core_gpu.html` is a standalone WebGL2/GLSL3 visualization of a procedurally generated mermaid-to-vortex particle field with a holographic neural-grid overlay.

It is a rendering and adaptive-control demonstration. It does not claim to simulate biological neurons or train a neural network.

## Run locally

The ES-module imports require an HTTP origin rather than opening the file directly:

```bash
cd sdk/vann_rom_sdk
python -m http.server 8000
```

Open:

```text
http://localhost:8000/examples/neural_core_gpu.html
```

The browser must support WebGL2. The example imports Three.js `0.160.0` from jsDelivr.

## Procedural particle model

For particle identifier `i`, normalized longitudinal coordinate

\[
s_i=\frac{i+\tfrac12}{N}
\]

selects a point on a parametric mermaid surface `p_mermaid(i)`. A time-dependent vortex target `p_vortex(i,t)` is generated independently in the vertex shader.

The displayed position is

\[
p_i(t)=
(1-m_t)p_{\text{mermaid}}(i)
+m_t p_{\text{vortex}}(i,t)
+\tau n_i(t),
\]

where `m_t` is the adapted morph coefficient, `τ` is turbulence, and `n_i(t)` is bounded procedural value noise.

The morph control follows the stable first-order update

\[
m_{k+1}=m_k+(m^*-m_k)\left(1-e^{-\eta\Delta t}\right),
\]

with target `m*`, adaptation rate `η > 0`, and clamped frame interval `Δt`.

Because

\[
0\le 1-e^{-\eta\Delta t}<1,
\]

the update cannot overshoot a fixed target and converges monotonically:

\[
\lim_{k\to\infty}m_k=m^*.
\]

This is a bounded visual adaptation law, not gradient descent.

## GPU execution contract

The particle shader explicitly uses:

- WebGL2;
- GLSL ES 3.00 through `THREE.GLSL3`;
- `gl_VertexID` for deterministic procedural geometry;
- GLSL3 `out`/`in` varyings;
- bounded perspective denominators;
- point-size clamping to `[1, 22]` pixels;
- additive blending with depth writes disabled.

A zero-valued position buffer remains necessary to establish the vertex draw count. Particle positions and sizes are otherwise generated in the vertex shader.

The procedural point cloud disables object-level frustum culling because CPU-side geometry does not describe the shader-generated locations.

## Adaptive particle budget

The initial particle count is selected once from browser capability hints:

| Profile | Particles |
|---|---:|
| Reduced motion | 100,000 |
| Mobile / constrained | 250,000 |
| Mid-range desktop | 600,000 |
| Higher-capability desktop | 1,000,000 |

These are conservative heuristics, not hardware guarantees.

## Runtime quality governor

The quality controller observes an exponentially smoothed frame interval and adjusts:

- renderer/composer pixel ratio;
- bloom strength.

For target frame interval `T`, quality decreases when

\[
\bar{F}>1.28T
\]

and may increase when

\[
\bar{F}<0.78T.
\]

Pixel ratio is bounded below by `0.65` and above by the device-specific cap. A 120-sample cooldown prevents rapid oscillation.

## Telemetry

The display reports:

- active particle count;
- visual pulse frequency derived from the actual sinusoid angular frequency;
- mean frame interval;
- GPU elapsed time when `EXT_disjoint_timer_query_webgl2` is available;
- active pixel ratio and bloom strength.

GPU time is shown as `n/a` when the timer extension is unavailable. Frame time is not presented as isolated neural or inference latency.

## Lifecycle handling

The example:

- pauses requestAnimationFrame work when the document is hidden;
- resumes when visible;
- handles WebGL context loss and restoration;
- updates renderer, composer, and camera state on resize;
- reduces particle count and animation speed under reduced-motion preferences.

## Validation

`tests/test_neural_core_html.py` checks:

- the WebGL2/GLSL3 contract;
- shader interface compatibility;
- bounded point size and perspective division;
- stable morph adaptation;
- device-adaptive particle tiers;
- deterministic particle sizing;
- frustum-culling safety;
- quality-governor bounds;
- telemetry naming and timer-query usage;
- lifecycle handlers;
- absence of per-frame bounding-sphere reconstruction;
- JavaScript module syntax through `node --check` when Node.js is installed.

The tests validate source-level rendering invariants. They do not replace browser/GPU compatibility testing on real devices.
