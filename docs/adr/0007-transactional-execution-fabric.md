# ADR-007: Adopt the Jarvis-X transactional execution fabric

**Status:** Accepted  
**Date:** 2026-08-13

## Context

Jarvis-X now contains a deterministic VM, provenance/journaling, sparse spatial runtimes, Dr Moagi field dynamics, codec/quantization research components, Moagi-Helmholtz orchestration, and the M³-ACME data-facing adapter. The remaining systems-level gap is a single authority-preserving transaction boundary that composes those subsystems without allowing a research transform to mutate authoritative state directly.

## Decision

Jarvis-X adopts a universal transaction fabric:

```text
OBSERVE -> NORMALIZE -> PROVENANCE -> AUTHORIZE -> ENCODE -> TRANSFORM
-> RECONSTRUCT -> MEASURE -> PROPOSE -> VERIFY -> PI_LAMBDA
-> COMMIT / ROLLBACK -> OMEGA RECEIPT -> RE-ENTER
```

The executable reference is `src/jarvisx/transaction_fabric.py`.

The authoritative transition is:

```text
Xi_(t+1) = Commit(Pi_Lambda(T_nu(Xi_t, I_t)))
```

where `T_nu` is a version-bound research transform.

## Required invariants

1. Input admission precedes research execution.
2. Every optional transform is adapter-driven and version identified.
3. Candidate state is isolated from authoritative state until validation succeeds.
4. Pi_Lambda enforces resource, numerical and adapter-specific validation.
5. Failed candidates roll back to the prior authoritative state.
6. Input/output state receives deterministic canonical digests.
7. Transactions emit explicit decisions and metrics.
8. Omega receipts form an append-only hash chain with verification.
9. Virtual scale is separated from measured throughput and resident allocation.
10. The fabric is an orchestration boundary, not an OS security sandbox.

## Consequences

Jarvis-X gains one executable control plane that can host existing and future research adapters while retaining deterministic-core authority. Specialized adapters remain responsible for their own mathematical and domain-specific correctness; the fabric supplies composition, rollback, provenance and resource gating.

## Validation

Promotion requires focused tests for deterministic execution, adapter commit, adapter rollback, resource rejection, admission behavior, Omega-chain verification and stable receipt serialization. CI must pass before merge.
