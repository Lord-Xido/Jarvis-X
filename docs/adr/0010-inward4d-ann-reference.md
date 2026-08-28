# ADR-010: Adopt a bounded inward-4D graph autoencoder reference

**Status:** Proposed
**Date:** 2026-08-28
**Extends:** ADR-003, ADR-006, ADR-007

## Context

The proposed `10 x 10 x 10` engine combines four separate ideas: discrete
lattice addressing, a toroidal `R^4` coordinate map, an autoencoder objective,
and online structural adaptation. Without an explicit boundary, the geometry
can be mistaken for the neural computation, a radius graph can be assigned an
unsupported edge count, and energy minimization can trivially delete every
weight instead of learning a reconstruction.

The repository needs one executable interpretation whose topology, objective,
gradient, update authority, and claims can be tested independently.

## Decision

Jarvis-X will treat this subsystem as a bounded reference laboratory with the
following contracts.

1. Integer coordinates and the `100x + 10y + z` map are authoritative.
2. The `R^4` fold is immutable feature geometry, not a physical-dimension
   claim.
3. Graph membership is positive-axis six-neighbour support plus a folded
   distance gate. A pure all-pairs radius graph is not assigned a predetermined
   edge count.
4. The exact open and full-fold edge counts are 2,700 and 3,000.
5. Encoder and decoder use the same symmetric, non-negative edge weights.
6. The reference is same-width and therefore does not claim compression.
7. The objective contains reconstruction, folded edge energy, weight
   homeostasis, and bias regularization. Edge energy alone is rejected because
   it has a trivial all-zero-conductance direction.
8. Gradients are analytic and checked against finite differences.
9. Pruning must preserve minimum degree and whole-graph connectivity.
10. A candidate update becomes authoritative only when the complete objective
    does not regress and any supplied validator accepts it.

## Consequences

- The fourth coordinate changes coupling strengths but never changes node
  identity.
- The runtime has explicit `O(N+M)` forward and gradient passes for fixed-degree
  topology.
- A residual target below `1e-6` is a measured stopping condition for a supplied
  input, not a universal convergence guarantee.
- "Auto-optimizing" means bounded parameter and topology updates inside the
  declared objective; it does not mean arbitrary code rewriting.
- Performance and superiority claims require separate benchmarks and baselines.

## Evidence required for acceptance

- exact address and topology invariant tests;
- closed-form coordinate fixture;
- central finite-difference gradient checks;
- deterministic replay;
- objective non-regression on commit;
- validator rollback;
- connectivity-preserving pruning;
- documentation of resource and inference boundaries.

The proposed implementation is `src/jarvisx/inward4d_ann.py`, with focused tests
in `tests/test_inward4d_ann.py` and the full arithmetic in
`docs/DR_MOAGI_10X10X10_INWARD_4D_ANN.md`.
