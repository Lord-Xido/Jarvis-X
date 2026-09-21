# ADR-027: Volumetric bytecode symbolic traversal reference

- **Status:** Proposed
- **Date:** 2026-09-21
- **Scope:** 3D sparse bytecode / symbolic iteration research layer

## Context

A supplied Jarvis-X pseudocode program defines multimodal packing, sparse octree
selection, SIMD encode/decode, exact 3D mirror folding, XOR delta measurement,
fixed-point recursion and a symbolic `1,000,000^1,000,000` target.

A literal implementation would create several correctness problems:

1. the symbolic iteration count cannot be physically enumerated;
2. 512 AVX bits do not mean 512 voxels when voxels are multi-bit;
3. a flattened block reversal is not identical to XYZ central reflection;
4. a global logical changed-bit fraction becomes misleadingly tiny for sparse
   astronomical domains;
5. generic repeated squaring does not automatically make arbitrary stateful
   recurrence executable in `O(log T)` time;
6. convergence must remain bounded by a finite physical iteration ceiling.

## Decision

Add `jarvisx.volumetric_bytecode` as a bounded software reference. It does not
replace `JX3DVM1` or the canonical transactional control plane.

The reference:

- stores the astronomical cycle target as an unevaluated descriptor;
- defaults to a `10^24`-per-axis logical coordinate domain without dense
  allocation;
- materializes only non-zero active voxels;
- uses typed 8-bit lanes;
- groups active voxels into sparse octree bins;
- uses an invertible byte rotate as a testable encode/decode reference;
- implements the exact coordinate mirror
  `(x,y,z)->(N-1-x,N-1-y,N-1-z)`;
- specializes mirror injection as bitwise union with the mirrored latent voxel;
- computes XOR/Hamming delta telemetry;
- reports both active-support and full-logical changed-bit fractions;
- gates fixed-point convergence on the active-support fraction;
- verifies exact codec round-trip before candidate promotion;
- caps physical recursion with `max_iterations`;
- publishes a non-authoritative permeation snapshot after verified state.

## SIMD boundary

The reference lane is 8 bits. Therefore an AVX-512 lowering may process 64
lanes/vector. A one-bit mask backend may process 512 bit voxels/vector. Backends
must report which representation they use.

## Symbolic iteration boundary

For the default target:

```text
(10^6)^(10^6) = 10^6,000,000
```

The descriptor exposes logarithmic metadata without constructing the integer.
No performance claim follows from the symbolic depth.

Repeated-squaring/operator-power acceleration is permitted only when a backend
proves that its operator representation is compositionally closed and that
composition itself avoids replaying the represented work.

## Convergence boundary

A sparse global fraction

```text
changed_bits / logical_bits
```

is telemetry only. With a `10^72` voxel universe it may be effectively zero even
when the active working set changed substantially. The reference therefore uses

```text
changed_bits / active_union_bits
```

for the fixed-point gate.

## Trust boundary

`PERMEATE` means publish verified state metadata to observation/GUI surfaces. It
does not mean unrestricted mutation, external actuation, source-code rewrite or
bypass of CTR / Pi_Lambda authority.

## Validation

Tests cover:

- symbolic target metadata without target materialization;
- sparse residency in a `10^24`-per-axis logical domain;
- exact involutive XYZ mirror geometry;
- exact byte encode/decode round-trip;
- mirror-union convergence;
- active-only octree bin accounting;
- separation of active and logical changed-bit fractions;
- finite physical recursion budget;
- fail-closed lane/active-set validation;
- verification/permeation trace presence.
