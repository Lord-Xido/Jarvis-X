# ADR-009: Photonic pixel-field runtime boundary

Status: proposed

## Context

Jarvis-X already distinguishes deterministic VM authority, transactional state transitions, sparse spatial domains and non-authoritative visualization. The graphics discussion introduces a stronger interpretation: a pixel is a measurement function over space, wavelength, direction and time, while graphics rendering is a physical-to-digital transduction pipeline.

Without an explicit boundary, three different claims can be conflated:

1. a browser animation of light;
2. a radiometric numerical model;
3. a full electromagnetic field solver.

Only the second is introduced by this decision.

## Decision

Add a dependency-free electromagnetic-photonic rendering reference above the canonical VM and transaction layers.

The subsystem shall:

- define pixels as bounded spectral detector measurements;
- preserve scene, camera, tile, frame and receipt data contracts;
- separate deterministic pixel semantics from CPU/GPU/cloud execution strategy;
- commit a frame only after complete verification;
- project optional pixel/depth samples into the canonical sparse `1000^3` address domain;
- remain non-authoritative with respect to canonical VM register and memory state unless an explicit adapter submits a separately validated transaction.

The subsystem shall not claim to solve Maxwell's equations or provide production GPU rendering.

## Consequences

Positive consequences:

- the visual layer gains a falsifiable numerical substrate;
- pixel, voxel and sparse-address concepts share one coordinate contract;
- GPU acceleration can be evaluated against a stable reference;
- rendering errors and state updates become transactionally auditable.

Costs and constraints:

- physically richer behavior requires independent validation data;
- floating-point cross-platform identity is not assumed;
- full-wave phenomena remain outside scope;
- large images remain bounded by explicit `max_pixels` and sampling limits.

## Validation

The integration candidate must provide tests for:

- wavelength and sensor-response bounds;
- deterministic tile partitioning;
- deterministic frame digest and replay;
- inverse-square attenuation behavior;
- quantization bounds;
- full-cycle rollback on invalid resource requests;
- bounded pixel-to-lattice mapping.
