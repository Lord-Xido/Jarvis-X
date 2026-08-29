# DM–vΩΞ⁺ Total Permeation 3D

This browser app is a bounded visualization surface for the current Jarvis-X Phase3D architecture.

It preserves the supplied **Total Permeation Singularity** visual language while aligning the HUD with the executable runtime that is now on `main`:

```text
K_t = (X_t, P_t, M)
-> relativistic phase update
-> implicit field
-> measured benchmark
-> runtime candidate
-> KineticTransactionEngine
-> verify
-> commit | rollback
-> receipt
```

## What is visualized

- a deformable 3D manifold that can cycle between torus-knot, icosahedral and octahedral forms;
- a 20,000-particle shockwave field;
- the Phase3D equilibrium-shell equation `F_r = A/r^2 - k r`;
- the derived equilibrium `r* = (A/k)^(1/3) = cubert(3)` for the default Phase3D coefficients;
- local browser frame rate and frame latency;
- optional externally supplied Phase3D telemetry.

## Run

Open `index.html` in a modern browser with network access. The app loads Three.js r128 from cdnjs.

Controls:

- **Morph Manifold** cycles the rendered geometry.
- **Wireframe Matrix** toggles wireframe rendering.
- **Permeate Totality** injects a bounded visual shockwave.
- Drag the scene to rotate the manifold.
- Use the mouse wheel or trackpad to zoom.

## Telemetry bridge

The app measures its own local render loop. It does not invent hardware throughput.

A host page or future Phase3D adapter can replace the main telemetry readout with measured runtime data:

```js
window.setPhase3DTelemetry({
  source: "MEASURED PHASE3D CUDA",
  queries_per_second: 1250000,
  node_updates_per_second: 820000,
  latency_ms: 1.73,
  peak_memory_mb: 742,
  semantic_error: 2.1e-5,
  energy_drift: 8.0e-6
});
```

Set `null` to return to local browser telemetry:

```js
window.setPhase3DTelemetry(null);
```

## Measurement boundary

The original mockup displayed quantities such as effectively infinite throughput, zero register latency and infinite reality-gap gamma. Those are not measurable properties of this browser runtime and are therefore not presented as telemetry here.

The display distinguishes:

- **measured local** — browser render-loop timing;
- **measured external** — values explicitly injected by a runtime adapter;
- **derived/model** — equilibrium radius, model node count and visualization coherence.

No visual effect is evidence of a new physical law, quantum execution, consciousness, or external state-of-the-art performance.

## Authority boundary

This browser app is visualization-only. It does not mutate authoritative Phase3D state, model weights, runtime configuration, files, devices or external systems.

Authoritative runtime configuration changes remain governed by the canonical Jarvis-X transaction law:

```text
snapshot -> observe -> encode -> propose -> shadow -> verify
         -> commit | rollback -> journal -> re-enter
```
