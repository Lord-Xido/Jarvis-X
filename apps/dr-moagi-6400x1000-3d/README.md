# Dr Moagi 6400×1000 Three.js Visualizer

Interactive Three.js reference visualization for the finite Dr Moagi 3D geometric coordinating autoencoding/decoding topology.

## Topology

- Spatial field: `20 × 20 × 16 = 6400` logical cells.
- Cluster hierarchy: `100 × 64 = 6400`.
- Programmable depth: `1000` logical iteration slots per panel.
- Inward path: `6400 → 800 → 100 → 18 → 4 → 1`.
- Outward path: `1 → 4 → 18 → 100 → 800 → 6400`.

The scene renders all 6400 logical cells with a Three.js `InstancedMesh`, plus deterministic particle flow, a fixed-point/core-solve visualization, and interactive playback controls.

## Run locally

Serve the repository through any local HTTP server and open:

```text
/apps/dr-moagi-6400x1000-3d/
```

For example:

```bash
python -m http.server 8080
```

Then visit:

```text
http://localhost:8080/apps/dr-moagi-6400x1000-3d/
```

## Controls

- **Pause / Play** — stop or resume the timeline.
- **Reset** — restart at ingestion.
- **Core solve** — jump to the central fixed-point phase.
- **Speed** — adjust animation speed.
- **Timeline** — scrub directly through contraction, solve, and expansion.
- **Pointer drag / wheel** — move the camera and adjust distance.

## Operational boundary

This app is a deterministic visualization of the runtime topology and its programmed state transitions. It does not assert that 6400 logical cells execute as 6400 physical CPU cores, and it does not treat the 1000-slot program depth as a measured 1000× wall-clock acceleration.

The repository's C++ runtime remains the executable reference for measured compute behavior.
