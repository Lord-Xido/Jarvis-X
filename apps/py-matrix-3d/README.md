# PY-MATRIX 3D — 1M LOC Mega-Code Engine

This self-contained WebGL2 application is a bounded research surface for mapping a **logical 1,000,000-line Python code space** onto the DM-vΩΞ⁺ / Ψ-Φ-Λ-Ω-Θ visualization stack.

It does not ingest, execute, or retain a literal million-line Python program. Instead it defines a deterministic address transform:

```text
line -> cluster -> local 20 x 20 x 10 cell -> procedural 3D position
```

The browser renders only 250 cluster instances with `drawArraysInstanced`. Line-level positions are derived on demand in O(1) arithmetic.

## Stack mapping

- **Ψ** — 1.0 Hz logical travelling pulse over cluster index space.
- **Φ** — deterministic AST/LOC spatial index; 4,000 LOC per cluster.
- **Λ** — explicit semantic-coherence target (`0.9998`) and measured frame-budget telemetry.
- **Ω** — 250-cluster procedural spatial memory matrix; not a one-million-object VRAM allocation.
- **Θ** — interactive line selection and camera/focus vector.

## Performance semantics

The renderer is designed to avoid per-frame cluster allocation and uses WebGL2 instanced drawing. `60 FPS` is a target that the HUD measures; it is **not** a browser real-time guarantee. The logical pulse remains exactly 1 Hz independent of observed frame cadence.

## Run

Open `index.html` in a WebGL2-capable browser. No CDN, package manager, or network connection is required.

## Trust boundary

This is a research/visualization surface. It has no network, shell, filesystem, market, medical, infrastructure, or device authority. It does not mutate authoritative Jarvis-X state and does not replace `jarvisx.system_runtime`.
