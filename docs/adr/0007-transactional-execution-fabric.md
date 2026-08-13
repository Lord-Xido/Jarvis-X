# ADR-007: Adopt the Jarvis-X transactional execution fabric

**Status:** Accepted
**Date:** 2026-08-13

## Context

Jarvis-X now contains a deterministic VM, provenance/journaling, sparse spatial runtimes, Dr Moagi field dynamics, codec/quantization research components, Moagi-Helmholtz orchestration, and the M³-ACME data-facing adapter. The remaining systems-level gap is a single authority-preserving transaction boundary that composes those subsystems without allowing a research transform to mutate authoritative state directly.

## Decision

Jarvis-X adopts a universal transaction fabric with the operational sequence:

```text
OBSERVE
-> NORMALIZE
-> PROVENANCE
-> AUTHORIZE
-> ENCODE
-> TRANSFORM
-> RECONSTRUCT
-> MEASURE
-> PROPOSE
-> VERIFY
-> PI_LAMBDA
-> COMMIT / ROLLBACK
-> OMEGA RECEIPT
-> RE-ENTER
```

The executable reference is `src/jarvisx/transaction_fabric.py`.

The authoritative transition is:

```text
Xi_(t+1) = Commit(Pi_Lambda(T_nu(Xi_t, I_t)))
```

where `T_nu` is a version-bound research transform. The fabric never grants authority to a candidate solely because it was generated, decoded, rendered, predicted, or optimized.

## Required invariants

1. Input admission occurs before research execution.
2. Every optional transform is adapter-driven and version identified.
3. Candidate state is isolated from authoritative state until validation succeeds.
4. Pi_Lambda enforces resource and numerical bounds and adapter-specific validation.
5. Failed candidates roll back atomically to the prior authoritative state.
6. Input/output state receives deterministic canonical digests.
7. The transaction emits explicit commit/rollback decisions and metrics.
8. Omega receipts form an append-only hash chain with tamper detection.
9. Virtual scale is not reported as measured throughput or resident allocation.
10. The fabric is an orchestration boundary, not an OS security sandbox.

## Consequences

Jarvis-X gains one executable end-to-end control plane that can host existing and future research adapters while retaining the deterministic-core authority model. Specialized adapters remain responsible for their own numerical, model, codec, geometry, or domain-specific correctness. The transaction fabric provides composition, rollback, provenance, and resource gating rather than pretending to prove those external properties.

## Validation

Promotion requires focused tests for deterministic execution, successful adapter commit, failed adapter rollback, resource rejection, input admission behavior, Omega-chain verification, and stable receipt serialization. CI must pass before merge.
