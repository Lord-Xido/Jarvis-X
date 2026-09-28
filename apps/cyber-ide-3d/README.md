# 3D Cyber-IDE Pixel Engine

A bounded browser emulator for the pixel-wise 3D Cyber-IDE architecture.

It maps a deterministic 1,000,000-line virtual C++ codebase into 2,500 source-file towers, morphs the same indexed objects between a 50 × 50 Euclidean grid and a toroidal manifold, overlays synthetic execution heat, projects the 3D scene into a real framebuffer, and supports pixel-ID picking back into source metadata.

Canonical provenance for named Dr Moagi research artifacts remains governed by the repository attribution documents.

## Run

Open `index.html` directly, or serve the directory:

~~~sh
python -m http.server 8000 --directory apps/cyber-ide-3d
~~~

Then open `http://localhost:8000`.

The app is dependency-free. It uses Canvas 2D as the framebuffer but performs the world/camera/perspective calculations explicitly in 3D.

## Arithmetic contract

The emulator keeps these invariants exact:

- `L = 1,000,000` source lines.
- `N = 2,500` file towers.
- `50 × 50 = 2,500` grid indexing.
- Mean occupancy is `400 LOC/tower`.
- Per-file variation is pair-balanced so `sum(L_i) = 1,000,000`.
- Design target `Φ = 10^12 LOC/s`.
- Arithmetic target pass latency `L / Φ = 10^-6 s = 1 μs`.
- Tokenized state assumption `32 B/LOC` gives `32 MB/pass`.
- The same design target therefore implies `32 TB/s = 256 Tb/s`.
- At 40 characters/line and an 8 × 16 logical glyph field, the codebase represents `5.12 × 10^9` logical glyph pixels.

Those throughput and bandwidth quantities are declared design arithmetic. They are not measured browser, CPU, GPU, HBM, compiler, or AVX-512 performance.

## What executes

The browser actually executes a bounded deterministic model:

1. partition one million LOC across 2,500 towers;
2. compute both grid and toroidal coordinates for every tower;
3. relax the topology state with `λ(t+dt)=λ*+(λ(t)-λ*)exp(-k dt)`;
4. generate bounded synthetic execution events and exponentially decaying heat;
5. optionally apply a bounded harmonic tower-height pulse and synthetic objective decay;
6. transform each tower from world coordinates into camera coordinates;
7. perform perspective projection into the current viewport;
8. depth-sort visible tower projections;
9. rasterize their screen-space quads;
10. render a parallel pixel-ID buffer for click selection;
11. decode a selected tower into its LOC range and representative source operation;
12. rasterize the currently selected character into an 8 × 16 alpha field.

The pixel-ID buffer makes the visual mapping bidirectional:

`source tower → 3D primitive → framebuffer pixel → source tower`.

## Controls

- **Pause / Resume execution**: controls bounded synthetic execution events.
- **Morph to torus / grid**: changes the topology target.
- **Optimizer**: enables a visual harmonic pulse plus a labeled synthetic objective.
- **Reset**: restores deterministic initial state.
- **Simulation rate**: scales emulator time from 0.25× to 4×.
- Drag the scene to orbit, use the mouse wheel to zoom, and click a tower to decode its pixel ID back into source metadata.

## Telemetry

`window.getCyberIDE3DTelemetry()` returns a copied snapshot containing architectural constants, morph state, synthetic objective, measured render rate, bounded emulator activity, and selected tower.

The measured render rate and emulator activity are observational browser measurements. They must not be conflated with the design target `10^12 LOC/s`.

## Validation

Run:

~~~sh
node --test apps/cyber-ide-3d/test_model.cjs
~~~

The tests verify the exact line partition, topology geometry, torus radii, target arithmetic, logical glyph-pixel count, exponential morph law, deterministic bounded runtime state, and inline script syntax.

## Capability boundary

This is a visualization / emulation surface, not an authoritative compiler, JIT, AST optimizer, GPU driver, HBM controller, autonomous source-code modifier, or benchmark proving physical 1 μs whole-codebase optimization. It does not mutate the canonical Jarvis-X VM state.
