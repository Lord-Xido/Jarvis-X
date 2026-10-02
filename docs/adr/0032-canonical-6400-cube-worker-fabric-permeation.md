# ADR-032: Canonical 6400³ logical worker-fabric permeation

**Status:** Accepted  
**Date:** 2026-10-02  
**Extends:** ADR-016, ADR-017, ADR-022, ADR-026, ADR-031  
**Implementation:** `src/jarvisx/dr_moagi_worker_fabric.py`  
**Specification:** `docs/DR_MOAGI_6400_CUBE_WORKER_FABRIC.md`

## Context

Jarvis-X already separates logical geometry from physical allocation and uses bounded
worker pools in several sparse runtimes. The Dr Moagi ANN IDE now exposes an exact
`6400 x 6400 x 6400` logical worker lattice, equal to
`262,144,000,000` addressable workers.

Without one repository-wide contract, downstream documentation and runtime adapters
could accidentally treat that logical cardinality as resident state or physical
thread parallelism.

## Decision

Adopt the 6400³ worker fabric as a canonical Layer-4 scheduling/addressing profile.

The canonical geometry is:

```text
logical lattice      = 6400 x 6400 x 6400
logical workers      = 262,144,000,000
brick side           = 64
brick lattice        = 100 x 100 x 100
logical bricks       = 1,000,000
workers per brick    = 262,144
```

The invariant is:

```text
logical address space != active work != resident state != physical parallelism
```

The fabric SHALL:

- preserve exact 3D coordinate <-> worker-ID addressing;
- partition the logical domain into 64³-worker bricks;
- materialize and execute only a bounded active set;
- report active work, resident bricks, physical workers and measured wall time
  independently;
- remain subordinate to ADR-016 candidate/verification/commit semantics;
- expose topology through the machine-readable runtime-fabric manifest;
- permit Python, C++, CUDA/WebGPU, distributed and future hardware adapters to
  implement the same contract without claiming dense 262.144-billion-way execution.

## Runtime-fabric permeation

`apps/dr-moagi-platform-java/runtime-fabric.json` records the exact geometry,
implementation, specification and authority boundary. The manifest verifier checks
the arithmetic and required source/specification paths.

The browser ANN IDE is registered as a specialist runtime because it exposes
`/v1/fabric` and `/v1/fabric/step` and visualizes measured worker receipts.

## Transaction boundary

A worker-fabric step is provisional computation. It does not gain authority merely
because a logical worker coordinate exists or because a worker was active.

```text
logical coordinate
  -> active-set admission
  -> bounded physical execution
  -> receipt / residual / telemetry
  -> policy + validation
  -> COMMIT or ROLLBACK
```

Distributed adapters may move a brick or active shard to another process, GPU,
machine or accelerator, but promotion authority remains outside the worker itself.

## Consequences

- 6400³ becomes a reusable scheduling/addressing profile rather than an IDE-only
  number.
- Existing 6400-panel and other sparse runtimes remain valid; they are separate
  geometry profiles and are not silently reinterpreted as 6400³.
- Performance claims must continue to report actually executed work and measured
  hardware behavior.
- Dense memory allocation or one-native-thread-per-logical-worker implementations
  are explicitly non-canonical.

## Validation

Acceptance requires:

- exact topology and partition arithmetic tests;
- deterministic sparse-step tests;
- manifest-verifier coverage;
- ANN IDE API/container smoke tests;
- repository static analysis and cross-version Python tests.

The first implementation was merged through PR #350 and is further permeated by
this ADR.
