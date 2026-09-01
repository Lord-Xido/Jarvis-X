# ADR-002: Treat Moagi OmegaFold as a bounded verified optimizer

**Status:** proposed  
**Date:** 2026-07-31  
**Tracking:** #62

## Context

The Moagi OmegaFold cosmogram describes a hyper-recursive optimizer using terms such as algorithmic singularity, `O(0)`, temporal collapse, Hilbert-space folding, zero-time execution and a `10^3000x` speedup. These terms express the intended direction—remove sequential work by identifying invariant structure—but do not define executable semantics or measurable performance.

Jarvis-X requires deterministic behavior, bounded execution, explicit capability boundaries, tests and evidence for performance claims. Broad adaptive systems must remain isolated from the authoritative VM until transaction and rollback semantics are complete.

## Decision

Jarvis-X will represent OmegaFold as an experimental bounded optimizer with these rules:

1. Every problem supplies an initial state, transition and independently measured residual.
2. Every run has an explicit finite iteration bound.
3. Closed-form candidates are hints, not authority; they must pass shape, finiteness and residual checks.
4. Repeated canonical states terminate as cycles.
5. Results include a hash-bound certificate recording method, iterations, residual and terminal reason.
6. Non-finite values, malformed dimensions and invalid residuals fail closed.
7. Performance claims require a reproducible benchmark contract.
8. The subsystem has no authority to mutate canonical VM state.
9. `O(0)`, zero-time, tachyonic and perfect-loss language remains symbolic and may not appear as an implemented capability.

## Consequences

### Positive

- the cosmogram gains a testable engineering interpretation;
- closed-form acceleration can be measured without overstating it;
- non-convergence and cycles become explicit outcomes;
- deterministic certificates support audit and replay;
- later spectral, graph, CUDA or RTL work has a stable contract.

### Negative

- the reference implementation cannot promise arbitrary acceleration;
- Python callables remain outside a complete security boundary;
- decimal canonicalization is a reference compromise, not a universal floating-point identity guarantee;
- memoization and advanced reductions remain future work.

## Validation

The initial reference layer must pass tests for convergence, rejected closed forms, cycle detection, iteration limits, dimensional changes, non-finite states, tampered certificates and deterministic replay. Its benchmark must emit workload, precision, environment and measured timings.
