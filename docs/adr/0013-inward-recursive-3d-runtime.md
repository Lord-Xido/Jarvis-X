# ADR-013: Bound the inward-recursive 3D runtime as a reference laboratory

**Status:** Proposed  
**Date:** 2026-09-05  
**Extends:** ADR-002, ADR-010

## Context

Jarvis-X has several spatial and inward-processing references, but the current design work requires a narrower mechanism that can be falsified end to end:

1. map bytes into explicit 3D bit planes;
2. contract those planes into a learned latent state;
3. update that state recursively using the previous 3D reconstruction error;
4. decode back into a bit volume;
5. measure differentiable reconstruction quality separately from exact Hamming/SHA identity; and
6. reject training mutations that worsen an immutable validation score.

Earlier prototypes re-encoded the same input at each recursion depth. That is error-conditioned repetition, not true latent-state inheritance. This ADR requires the recurrent state itself to advance:

\[
z_{r+1}=z_r+g_r\,S_\psi(z_r,e_r)
\]

with

\[
\hat x_r=D_\phi(z_{r+1}),\qquad e_r=x-\hat x_r.
\]

## Decision

Adopt `src/jarvisx/dr_moagi_inward3d_runtime.py` as a bounded **reference laboratory**, subject to the following contract.

### Binary boundary

A byte payload is expanded into eight least-significant-bit-first planes:

\[
B\in\{0,1\}^{8\times N\times N\times N}.
\]

The reference exposes and tests the binary dot-product identity

\[
q(a)^Tq(b)=2\,\operatorname{popcount}(\operatorname{XNOR}(a,b))-n.
\]

The learned convolutional path is still ordinary floating-point PyTorch. XNOR/POPCOUNT is therefore a verified primitive, **not** a claim that the neural kernel itself is binary.

### Inward recurrence

The encoder is evaluated once for the initial state. Each recursive fold inherits the previous latent state and applies an error-conditioned correction. Telemetry records gate strength, correction RMS, reconstruction-error RMS and the actual latent-state displacement.

### Objective and guarded adaptation

Training may optimize reconstruction, cycle, spatial-gradient, self-consistency and latent-energy terms. Acceptance is evaluated with a separate immutable composite score. A candidate update is committed only when that score improves; otherwise model and optimizer state are rolled back.

### Exact reconstruction boundary

Approximate neural loss must never be reported as lossless identity. Exact output is measured independently using:

- byte accuracy;
- bit Hamming distance;
- bit accuracy; and
- SHA-256 equality.

The terminal lossless condition is:

\[
H(B,\hat B)=0
\]

and matching SHA-256 digests.

### Virtual scale boundary

Large volumetric labels describe a virtual streamed address space. This implementation materializes finite tiles only. It does not establish a literal `1000 TB^3` allocation, distributed shard topology, hardware throughput or global physical storage geometry.

### Tile boundary

The present three-stage stride-2 encoder requires `tile_edge >= 16` and divisibility by eight. This explicitly closes the `8^3 -> 1^3` normalization/degeneracy boundary encountered in earlier prototypes.

## Verification

Focused tests cover:

1. byte -> 3D bit-plane -> byte exact round trip;
2. XNOR/POPCOUNT equivalence to the bipolar dot product;
3. exact Hamming/SHA identity telemetry;
4. the minimum tile-size boundary;
5. non-zero inherited latent displacement across recursive folds; and
6. guarded optimization: an accepted update cannot have a worse validation score.

## Consequences

The runtime may be described as a self-correcting 3D neural codec laboratory with explicit bit boundaries and true latent recurrence.

It must **not** be described as evidence of:

- lossless learned compression before exact identity is achieved;
- a production binary neural accelerator;
- arbitrary self-improvement or unrestricted source mutation;
- physical allocation of the virtual large-volume label; or
- consciousness, phenomenology or other non-computational properties.

Promotion beyond reference-laboratory status requires held-out benchmarks, rollback replay tests, persistent checkpoint/version binding, a binary-kernel implementation if XNOR/POPCOUNT execution is claimed, and inclusion in the consolidated empirical-validation artifact.
