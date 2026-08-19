# Operational 3D Auto-Encoding / Decoding System

## Status

Integration candidate on `agent/operational-auto-codec-loop` / PR #120.

This subsystem makes 3D space part of the executable data contract rather than a presentation-only metaphor. The authoritative state remains a bounded sparse scalar field on integer `(x, y, z)` coordinates. Each cycle encodes those coordinates into a reversible 63-bit Morton address, quantizes the scalar value, decodes the requested support, computes the existing Dr Moagi residual-field update, validates the candidate transaction, commits or rolls it back, records the state in the Omega hash chain, and emits a 3D runtime frame from the resulting authoritative state.

## End-to-end kinetic path

```text
3D INPUT FIELD
    ↓
VALIDATE (x,y,z,value)
    ↓
MORTON 3D ADDRESS ENCODE
    ↓
QUANTIZE VALUE
    ↓
SPARSE 3D LATENT
    ↓
DECODE REQUESTED 3D SUPPORT
    ↓
RECONSTRUCTION
    ↓
RESIDUAL R = Psi - D(E(Psi))
    ↓
3D SIX-FACE LAPLACIAN + GLYPH COUPLING
    ↓
PROJECT INTO NUMERICAL / RESOURCE BOUNDS
    ↓
VERIFY
   ↙  ↘
ROLLBACK  COMMIT
            ↓
        OMEGA JOURNAL
            ↓
     SPATIAL METRICS
            ↓
        3D FRAME
            ↓
   CONVERGENCE / FIXED POINT / BUDGET
            ↓
           LOOP
```

## Spatial latent

`MortonQuantizedFieldCodec3D` maps a coordinate

`(x, y, z)`

to a Morton/Z-order key by bit-interleaving 21 bits from each axis:

`m = interleave_3(x, y, z)`.

The resulting key is reversible for coordinates in `[0, 2^21 - 1]` on each axis. Values are independently quantized as

`q = round(value / step)`

and reconstructed as

`value_hat = q * step`.

The reference latent therefore consists of sparse records

`(morton_63, signed_quantized_value)`.

For a portable packed representation the runtime reports an estimate based on an unsigned 64-bit Morton key plus a signed 32-bit value code, i.e. 12 bytes per active latent entry. This is a wire-format estimate, not Python heap usage.

## 3D field mechanics

The existing field runtime evaluates the same-space equation

```text
dPsi/dt =
  -alpha (I - D o E)[Psi]
  + lambda * Laplacian((I - D o E)[Psi])
  + eta * G_moagi(Psi)
```

on the sparse logical 3D lattice.

The Laplacian uses six face-neighbours (`±x`, `±y`, `±z`). Candidate state is projected into configured value and active-cell bounds before publication. A validator may reject the candidate without mutating authoritative state.

## 3D telemetry

Every captured frame is derived from `runtime.snapshot()` after a completed cycle. Frames therefore represent execution state, not a separately simulated animation.

The runtime reports:

- active sparse cell count;
- axis-aligned 3D bounds;
- geometric centroid;
- absolute-value-weighted centroid;
- RMS spatial radius;
- L1 and L2 field energy;
- number of occupied six-face neighbour links;
- sparse occupancy ratio;
- state SHA-256 digest;
- bounded point-cloud sample for visualization.

Point samples are deterministically ordered by Morton key and downsampled when the render budget is exceeded.

## Execution surfaces

### Python library

Use `MortonQuantizedFieldCodec3D`, `DrMoagiFieldRuntime`, `AutoCodecLoop`, and `SpatialAutoCodec3DSystem` directly.

### CLI

```bash
jarvisx codec3d examples/auto_codec_3d_run.json
```

The command returns a JSON receipt with the loop result, 3D frames, spatial metrics, latent digest, state digest, and Omega journal verification.

### Cloud API

```text
POST /codec/3d/run
```

accepts sparse 3D cells and bounded runtime parameters.

`GET /health` advertises the active 3D codec endpoint and spatial codec mode.

### Browser

`GET /` serves a dependency-free WebGL control surface. The viewer consumes only frame data returned by `/codec/3d/run`.

Controls:

- drag: orbit;
- wheel/pinch: zoom;
- frame slider: inspect a specific runtime cycle;
- Play/Pause: traverse captured execution frames.

Positive and negative scalar values are rendered differently, while point size scales with absolute field magnitude.

## Verification and termination

The system is bounded by:

- field-side and coordinate validation;
- maximum active cells;
- conservative explicit-step guard;
- value projection;
- decoder support confinement;
- optional external transaction validator;
- maximum cycles;
- reconstruction MSE target;
- fixed-point state digest detection;
- rejection circuit breaker;
- frame count and render-point budgets;
- hash-chained Omega journal verification.

## Operational boundary

This is a fully executable 3D **reference runtime**: the spatial codec, field operators, transaction gates, receipts, CLI, API, container entry point, and WebGL viewer are implemented as software paths.

It is not a claim that:

- the Morton codec is a trained or optimal neural latent representation;
- arbitrary 3D assets are losslessly compressed;
- the sparse scalar field is a general mesh, CAD, physics, or volumetric rendering engine;
- the runtime provides hostile-code isolation;
- visualization proves physical hardware, electromagnetic, or AGI properties;
- the current reference performance is production-scale.

Those require separate adapters, benchmarks, security controls, and empirical evidence.
