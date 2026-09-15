# Dr Moagi 3D Operational Console

## Purpose

The operational console turns the Three.js visualization into a live renderer for the real sparse 3D runtime.

The authoritative loop is:

```text
Python engine
  -> encode
  -> latent state
  -> Omega memory
  -> decode
  -> residual
  -> correction
  -> residual-priority scheduling
  -> telemetry
  -> WebSocket
  -> Three.js renderer
```

The browser does not generate authoritative MSE, latency, active-tile positions, or Cube64 words. Those values are emitted by `jarvisx.dr_moagi_operational_console` from `DrMoagi3D1000xEngine` state.

## Canonical sparse contract

The reference configuration uses:

```text
virtual cube       = 1000 x 1000 x 1000 = 1,000,000,000 logical voxels
tile edge          = 10
voxels per tile    = 1,000
active tiles       = 1,000
active voxels      = 1,000,000
work reduction     = 1,000,000,000 / 1,000,000 = 1000x
```

`1000x` is therefore an algorithmic active-work reduction under the canonical sparse configuration. It is not a claim of guaranteed 1000x end-to-end wall-clock acceleration.

## Runtime state

Each WebSocket telemetry frame contains:

- authoritative cycle telemetry;
- active 3D tile IDs and `(x, y, z)` tile coordinates;
- per-tile residual and normalized residual;
- a sample of packed 64-bit Cube64 control words;
- an explicit claims/role contract separating computation from visualization.

Residual magnitude controls rendered voxel prominence. The scheduler ordering determines which active tiles are surfaced first.

## Running

Install the acceleration backend and launch the package console:

```bash
python -m pip install -e ".[accel]"
jarvisx-3d-console
```

Then open:

```text
http://127.0.0.1:8765
```

The console exposes:

```text
GET /          Three.js operational UI
GET /api/state current authoritative snapshot
WS  /ws        live telemetry and control channel
```

WebSocket control messages are:

```json
{"type":"pause"}
{"type":"resume"}
{"type":"step"}
{"type":"cadence","ms":100}
```

Cadence is bounded to 16-1000 ms.

## Browser dependency boundary

Three.js, OrbitControls, and Tailwind are currently loaded from public CDNs. The Python runtime and telemetry path remain local. A future offline deployment can vendor those browser assets without changing the runtime protocol.

## Engineering boundary

This console is an operational visualization and control surface over a single-process reference engine. It does not itself provide multi-host consensus, GPU residency management, distributed storage, or authenticated cloud-worker transport.

The architectural invariant is:

```text
computation generates the animation
```

not:

```text
animation pretends to be computation
```
