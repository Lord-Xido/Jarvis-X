# ADR-0010: Dr Moagi geometric state equation as a bounded Layer-5 operator

**Status:** Proposed  
**Date:** 2026-08-17

## Context

ADR-002 accepts the Dr Moagi 3D adaptive codec-runtime as a bounded research architecture, while `docs/ARCHITECTURE.md` requires same-space evolution, candidate-first adaptation, fail-closed validation, honest scale reporting, and separation between research layers and the deterministic VM core.

The compact geometric recurrence is a state-space law. Every additive term must therefore inhabit the declared `Xi/Z` state space. Earlier notation placed `grad_Theta L` directly inside that recurrence even though a parameter gradient inhabits parameter space. The hardened Codex already separates parameter learning from latent-state evolution, so the Layer-5 recurrence must use the same type discipline.

The canonical same-space form is:

\[
\Xi_{t+1}^{3D}
=
\Pi_{\Lambda_t}\left[
\Xi_t^{3D}
+P_{1:M}^{\circlearrowleft}(\Xi_t^{3D})
-E_t^{3D}
+\Omega_t^{3D}
+\kappa_tR_t^{\circlearrowleft}
-\eta_{Z,t}\nabla_Z L_t
-\zeta_t\nabla_HC_t
\right].
\]

Parameter learning remains a separate transition:

\[
\Theta_{t+1}^{cand}
=
\Theta_t-\eta_{\Theta,t}\nabla_\Theta L_t.
\]

A parameter gradient may participate in the state equation only through an explicit transport map

\[
T_{\Theta\rightarrow Z}(\nabla_\Theta L)\in Z,
\]

whose output type, units and support are declared and validated.

## Decision

Jarvis-X treats the geometric equation as a **Layer-5 candidate-state operator**, not as a replacement for the canonical VM instruction loop.

A conforming implementation must execute:

```text
snapshot Xi_t
  -> evaluate P_1:M on the frozen snapshot
  -> merge predictive branches deterministically
  -> acquire same-space E_t, Omega_t, R_t, grad_Z L_t, grad_H C_t
  -> form raw candidate
  -> apply Pi_Lambda
  -> validate candidate
  -> commit or rollback
  -> emit telemetry/provenance
```

Parameter-space learning occurs outside this additive state update and remains subject to the higher-level epistemic and authority commit gates.

The reference implementation lives at `src/jarvisx/dr_moagi_state_equation.py` with conformance tests in `tests/test_dr_moagi_state_equation.py`.

## Semantic contract

### Authoritative state space

For a sparse volumetric realization:

```text
Xi_t : Coordinate3D -> scalar
Coordinate3D = (x, y, z)
```

All additive terms in one authoritative update must share a declared compatible state type and units.

### Gradient-space separation

`grad_Z L` is a state-space correction and may be additive only when it shares the `Xi/Z` support and units.

`grad_Theta L` is a parameter-space object and updates `Theta`; it is not implicitly reinterpreted as a spatial field. Any cross-space contribution requires an explicit typed transport/Jacobian map.

### Predictive branching

`P_1:M` may evaluate several candidate futures, but the additive equation consumes one merged predictive field. The reference implementation uses a convex weighted merge:

\[
\bar P_t = \sum_{m=1}^{M} a_m P_m(\Xi_t),
\qquad a_m\ge0,\quad \sum_m a_m=1.
\]

This prevents prediction magnitude from increasing merely because branch count `M` increases.

### Projection and authority

`Pi_Lambda` is a real admission boundary. It may project, reject, enforce resource ceilings, check version compatibility, or apply a richer policy manifold. The raw and projected candidates are non-authoritative until validation succeeds. Rejection returns the frozen input state unchanged.

## Required invariants

1. **Same-space evolution:** every additive term has compatible support, type and units.
2. **Gradient-space separation:** `grad_Theta L` cannot be added to `Xi/Z` without an explicit transport map; the direct state correction is `grad_Z L`.
3. **Frozen-snapshot evaluation:** terms for one step are evaluated against one logical `Xi_t` snapshot unless explicitly staged.
4. **Finite arithmetic:** non-finite inputs or candidates fail closed.
5. **Bounded support:** materialized active state has an explicit cell/resource ceiling.
6. **Deterministic branch merge:** fixed branches and weights produce the same merged field.
7. **Projection before authority:** `Pi_Lambda` runs before commit.
8. **Rollback:** validator rejection preserves `Xi_t`.
9. **Support integrity:** projection cannot silently add or remove authoritative cells in same-support mode.
10. **Separation of authority:** this Layer-5 operator cannot silently mutate Layer-0/1 VM state.
11. **Measured claims:** branch count, virtual depth and visual frame rate are not physical throughput metrics.

## Relationship to existing runtimes

- **ADR-002 / codec runtime:** encoded/decoded representations, residuals, memory, rate-distortion objectives and inward latent refinement.
- **ADR-003 / field runtime:** bounded sparse same-space field semantics and candidate-first commit behavior.
- **ADR-006 / geometric diffusion:** refinement or propagation field semantics.
- **ADR-007 / control plane:** policy/resource constraints for authoritative execution.
- **ADR-0011 / epistemic gate:** promotion from decoded hypothesis to verified research state and parameter-learning eligibility.
- **Layer 6 visualization:** observational unless bound to an admitted and committed transition.

## Consequences

The geometric equation now has executable, testable and type-coherent semantics. Older callers using `eta` / `loss_gradient` must migrate to `eta_z` / `latent_gradient`. Cross-space optimization requires explicit transport operators rather than shorthand notation.

## Validation

Acceptance requires exact arithmetic tests, explicit `grad_Z` terminology in the executable API, predictive-branch scaling tests, same-support and non-finite rejection tests, projection-boundary and rollback tests, and documentation that `grad_Theta` remains outside the additive state equation.
