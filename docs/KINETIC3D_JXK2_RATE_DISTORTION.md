# Kinetic 3D JXK2 Rate-Distortion Capsule

## Status

This document specifies the next bounded execution layer above the native kinetic 3D backend.

It adds two capabilities that were previously missing:

1. a self-describing, integrity-sealed delta capsule that can be decoded independently from the live runtime when supplied with the exact predictor it is bound to;
2. deterministic rate-distortion planning that searches a bounded set of spatial codec parameters and selects the smallest valid capsule under an explicit maximum-error budget.

This is a measurable codec/runtime optimization. It is **not** a claim of universal state-of-the-art compression, learned neural coding, GPU leadership, or information-theoretic optimality.

## Runtime path

```text
OBSERVE W_t
  -> obtain committed predictor W_hat_t
  -> deterministic candidate search
       active threshold tau_a
       coarse factor k
       refinement threshold tau_r
  -> reject candidates with max reconstruction error > Lambda
  -> serialize every admissible candidate as JXK2
  -> select minimum transport bytes with deterministic tie breaks
  -> execute selected plan on requested backend
  -> VERIFY
  -> COMMIT
  -> build JXK2 capsule from authoritative reconstruction path
  -> independent capsule decode against predictor digest
  -> EMIT
```

The planning oracle uses the pure-Python reference backend so plan selection is deterministic and independent of wall-clock timing. The selected plan may then execute on `cpu-reference`, `native-cpu`, or a future backend that satisfies the same semantic contract.

## Rate-distortion objective

For each candidate `m`, Jarvis-X measures:

- maximum absolute reconstruction error `E_max,m`;
- MSE;
- active cell count;
- coarse latent count;
- fine correction count;
- exact serialized JXK2 byte length `B_m`.

Only candidates satisfying

```text
E_max,m <= Lambda
```

are admissible.

The primary selection rule is

```text
m* = argmin B_m
```

with deterministic tie breaks over error, active work, correction count and spatial parameters.

This makes the optimization falsifiable: the selected candidate must actually serialize to the reported byte count and independently reconstruct within the requested error budget.

## JXK2 transport format

The capsule is a predictor-bound delta object.

```text
fixed header
  magic = JXK2
  version
  shape X/Y/Z
  active threshold
  coarse factor
  refinement threshold
  tolerance
  active/coarse/fine counts
  SHA-256(predictor float64 image)

active-set payload
  run-length encoded active indices

coarse payload
  block coordinate varints
  float64 coarse residual

fine payload
  delta-coded active index
  float64 correction

SHA-256(all preceding capsule bytes)
```

The active set is encoded as runs rather than as a dense mask or one fixed-width index per voxel. Spatially coherent change can therefore remain compact while sparse irregular updates remain explicit.

## Reconstruction

Given a verified capsule and the exact predictor:

```text
reconstructed = predictor
for each active index i:
    reconstructed[i] += coarse[block(i)]
for each fine correction (i, delta_i):
    reconstructed[i] += delta_i
```

The predictor must hash to the digest embedded in the capsule. A capsule cannot silently apply to the wrong committed world.

## Integrity and failure semantics

The parser fails closed on:

- bad magic or unsupported version;
- checksum mismatch;
- truncated varints or float payloads;
- non-finite numerical metadata;
- out-of-bounds or overlapping active runs;
- duplicate/out-of-range coarse blocks;
- fine corrections for inactive cells;
- coarse-block coverage that does not exactly match the active set;
- trailing unparsed payload bytes;
- predictor digest mismatch during decode.

The adaptive API additionally decodes the newly generated capsule independently and requires it to reproduce the committed reconstruction before emitting the result.

## API

Adaptive execution:

```text
POST /v2/kinetic3d/execute-adaptive
```

Input:

```json
{
  "session_id": "demo",
  "shape": [8, 8, 8],
  "values": ["512 numeric values"],
  "tolerance": 0.0,
  "backend": "auto"
}
```

The response includes:

- ordinary kinetic verification and transaction state;
- selected rate-distortion plan and all admissible candidates;
- JXK2 byte count and actual wire compression ratio;
- transport SHA-256;
- Base64 capsule bytes.

Independent decode:

```text
POST /v2/kinetic3d/decode-capsule
```

The caller supplies the Base64 capsule and predictor vector. Decode fails if the predictor digest differs from the one bound into the capsule.

## Why this is deeper than the previous layer

The prior kinetic runtime exposed latent values and a value-count compression proxy, but the latent was not a complete transport artifact. Index and metadata overhead were not represented in that proxy.

JXK2 changes the measurement boundary:

```text
before: scalar latent count / source scalar count
now:    exact serialized capsule bytes / raw float64 bytes
```

A ratio greater than one is reported only when the actual JXK2 transport is smaller than the raw float64 source image. Small or irregular volumes are allowed to report expansion rather than a fabricated compression win.

## Capability boundary

This implementation establishes a deterministic, reversible, predictor-bound sparse 3D transport and bounded rate-distortion search. It does not establish external SOTA performance by itself.

A defensible SOTA claim requires a benchmark suite against relevant current codecs/runtimes on shared datasets and hardware, including at minimum:

- rate-distortion curves;
- encode/decode throughput;
- end-to-end latency;
- resident and transferred bytes;
- energy where measurable;
- corruption/recovery behavior;
- exact-versus-lossy operating points.

Until those comparative experiments are green, the correct claim is: **Jarvis-X now has the machinery required to optimize and measure this axis rigorously.**
