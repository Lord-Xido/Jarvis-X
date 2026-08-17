# ADR-0010: Dr Moagi geometric state equation as a bounded Layer-5 operator

**Status:** Proposed  
**Date:** 2026-08-17

## Context

ADR-002 accepts the Dr Moagi 3D adaptive codec-runtime as a bounded research architecture, while `docs/ARCHITECTURE.md` requires same-space evolution, candidate-first adaptation, fail-closed validation, honest scale reporting, and separation between research layers and the deterministic VM core.

The canonical research specification already contains codec, latent refinement, predictive branching, memory, inward recurrence and `Pi_Lambda` concepts. What was missing was one explicit executable contract for the compact geometric recurrence:

\[
\Xi_{t+1}^{3D}
=
\Pi_{\Lambda_t}\left[
\Xi_t^{3D}
+P_{1:M}^{\circlearrowleft}(\Xi_t^{3D})
-E_t^{3D}
+\Omega_t^{3D}
+\kappa_tR_t^{\circlearrowleft}
-\eta_t\nabla_\Theta L_t
-\zeta_t\nabla_HC_t
\right].
\]

Without a typed boundary, the notation can be interpreted inconsistently across visualization, codec, swarm, field and optimization layers.

## Decision

Jarvis-X will treat the equation above as a **Layer-5 candidate-state operator**, not as a replacement for the canonical VM instruction loop.

A conforming implementation must satisfy the following execution order:

```text
snapshot Xi_t
  -> evaluate P_1:M on the frozen snapshot
  -> merge predictive branches deterministically
  -> acquire same-space E_t, Omega_t, R_t, grad_Theta L_t, grad_H C_t
  -> form raw candidate
  -> apply Pi_Lambda
  -> validate candidate
  -> commit or rollback
  -> emit telemetry/provenance
```

The initial reference implementation lives at:

```text
src/jarvisx/dr_moagi_state_equation.py
```

with conformance tests at:

```text
tests/test_dr_moagi_state_equation.py
```

and the detailed systems mapping at:

```text
docs/research/DR_MOAGI_3D_GEOMETRIC_STATE_EQUATION.md
```

## Semantic contract

### Authoritative state space

For a sparse volumetric realization:

```text
Xi_t : Coordinate3D -> scalar
Coordinate3D = (x, y, z)
```

Vector, tensor or structured voxel values may be introduced later, but all additive terms in one authoritative update must share a declared compatible state type and units.

### Predictive branching

`P_1:M` may evaluate several candidate futures, but the additive equation consumes one merged predictive field. The reference implementation uses a convex weighted merge:

\[
\bar P_t = \sum_{m=1}^{M} a_m P_m(\Xi_t),
\qquad a_m\ge0,\quad \sum_m a_m=1.
\]

This prevents prediction magnitude from increasing merely because branch count `M` increases. Other deterministic aggregators are permitted if they document their scaling law and preserve the declared state type.

### Projection

`Pi_Lambda` is a real admission boundary. It may clip, normalize, project, reject, enforce resource ceilings, check version compatibility, or apply a richer policy manifold. Projection cannot silently escape the authoritative support or introduce non-finite state.

### Candidate-first authority

The raw and projected candidates are non-authoritative until validation succeeds. Rejection returns the frozen input state unchanged. Research code may not bypass the core architectural rule that adaptive or predictive outputs do not become canonical VM state by naming or visualization.

## Required invariants

1. **Same-space evolution:** every additive term has compatible support, type and units.
2. **Frozen-snapshot evaluation:** terms for one step are evaluated against one logical `Xi_t` snapshot unless an implementation explicitly defines a staged operator composition.
3. **Finite arithmetic:** non-finite inputs or candidates fail closed.
4. **Bounded support:** materialized active state has an explicit cell/resource ceiling.
5. **Deterministic branch merge:** fixed branches and weights produce the same merged field.
6. **Projection before authority:** `Pi_Lambda` runs before commit.
7. **Rollback:** validator rejection preserves `Xi_t`.
8. **Support integrity:** projection cannot add or remove authoritative cells silently when same-support mode is selected.
9. **Separation of authority:** this Layer-5 operator cannot silently mutate Layer-0/1 VM state.
10. **Measured claims:** branch count, virtual depth and visual frame rate are not physical throughput metrics.

## Relationship to existing runtimes

- **ADR-002 / codec runtime:** supplies encoded/decoded representations, residuals, memory, rate-distortion objectives and inward latent refinement.
- **ADR-003 / field runtime:** supplies bounded sparse same-space field semantics and candidate-first commit behavior.
- **ADR-006 / geometric diffusion:** may provide one implementation of a refinement or propagation field.
- **ADR-007 / control plane:** may supply policy/resource constraints consumed by `Pi_Lambda`.
- **Layer 6 visualization:** may render the equation geometrically but remains observational unless bound to the authoritative Layer-5 transition and validated.

## Consequences

### Positive

- the geometric equation now has executable, testable semantics;
- notation is aligned with same-space field invariants;
- predictive branching has an explicit bounded aggregation rule;
- projection and rollback are structural rather than decorative concepts;
- the deterministic VM boundary is preserved.

### Negative

- the compact equation does not by itself define how each field is learned or generated;
- richer tensor-valued states will require explicit algebra and unit contracts;
- a convex branch merge is deliberately conservative and may not match every research predictor;
- stability still depends on the operators, coefficients and projection manifold.

## Validation

Acceptance requires at minimum:

- exact arithmetic tests for the recurrence;
- predictive-branch scaling tests;
- same-support rejection tests;
- non-finite rejection tests;
- projection-boundary tests;
- rollback tests;
- explicit documentation that the implementation remains Layer 5 and candidate-first.
