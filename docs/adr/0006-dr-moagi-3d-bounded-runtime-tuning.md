# ADR-006: Dr Moagi 3D Bounded Runtime Tuning

- **Status:** Proposed
- **Date:** 2026-08-13
- **Scope:** Research/runtime layer only

## Context

Jarvis-X already defines a Dr Moagi 3D field runtime and a canonical inward-turned mechanics specification. The missing executable bridge is a bounded mechanism that can evaluate alternative numerical mechanics against the same 3D state without allowing arbitrary source rewriting or unverified deployment.

The field runtime is transactional: it computes a candidate from a frozen snapshot, projects it into the admissible set, and commits only after validation. The inward mechanics layer should preserve that authority boundary.

## Decision

Add `src/jarvisx/dr_moagi_runtime_tuner.py` as a finite, shadow-evaluated mechanics tuner around `DrMoagiFieldRuntime`.

For active field state `X_t`, active mechanics `M_t`, source anchor `X_0`, and bounded transformation set `A_opt`, the tuner computes

```text
M_t^(k) = A_k(M_t)
X_(t+1)^(k) = F_{M_t^(k)}(X_t)
```

in isolated shadow runtimes.

The baseline is evaluated from the same snapshot:

```text
X_(t+1)^base = F_{M_t}(X_t)
```

A candidate is admissible only if all declared gates pass, including reconstruction error, source-anchor drift, semantic distance from the baseline transition, numerical stability, resource limits, and the existing runtime validator.

The selected mechanics are

```text
k* = argmin_k J(X_(t+1)^(k), T_k, M_t^(k))
```

and are committed only when

```text
J_base - J_k* >= epsilon_accept
```

The active field state is not advanced during mechanics search. After a mechanics commit, the next authoritative `runtime.step()` executes under the selected configuration.

## Allowed search dimensions

The initial implementation permits bounded candidate factors for:

- explicit step size `dt`;
- reconstruction closure gain `alpha`;
- residual Laplacian gain `lambda_residual`;
- glyph/permeation gain `eta`;
- sparse pruning threshold `prune_epsilon`.

The tuner cannot change:

- logical lattice dimensions;
- active-cell authority budget;
- projection value bounds;
- source code;
- privileges;
- external systems;
- validation policy.

## Default objective

The reference objective is

```text
J =
    w_rec    * reconstruction_mse
  + w_anchor * source_anchor_mse
  + w_sparse * support_fraction
  + w_rhs    * max_abs_rhs
  + w_stab   * stability_load
```

Callers may provide a workload-specific finite scalar objective, but it does not bypass the admissibility gates.

## Inward 3D loop

The coupled mechanics/state cycle is:

```text
observe active 3D state
-> encode runtime telemetry
-> generate bounded mechanics candidates
-> fork shadow 3D transitions
-> score and compare
-> semantic/resource/stability gate
-> commit better mechanics or retain baseline
-> execute one authoritative field transition
-> record improvement memory
-> repeat
```

In compact form:

```text
M_(t+1) = Pi_Lambda^M[argmin_{M in N(M_t)} J(F_M(X_t))]
X_(t+1) = Pi_Lambda^X[F_{M_(t+1)}(X_t)]
Omega_(t+1) = rho Omega_t + (1-rho) DeltaJ_t
```

This is the operational meaning of turning the Dr Moagi loop inward: the 3D field evolves under `F`, while the bounded mechanics controlling `F` are themselves observed, compared, refined, and conditionally updated.

## Safety and determinism

- Every candidate starts from the same frozen state.
- Shadow execution cannot mutate active state.
- The existing field validator is reused for baseline and candidates.
- Conservative stability checks in `DrMoagiFieldConfig` remain authoritative.
- Candidate search is finite and declared by policy.
- No arbitrary source-code mutation is provided.
- Rejection retains the current mechanics unchanged.

## Consequences

### Positive

- Makes the inward mechanics concept executable.
- Preserves the deterministic core/research boundary.
- Supports workload-specific tuning without unbounded search.
- Makes source-anchor drift and semantic divergence explicit acceptance criteria.
- Separates mechanics adaptation from field-state evolution.

### Trade-offs

- Shadow evaluation adds compute cost.
- One-step shadow scoring may not predict long-horizon behavior.
- A workload-specific objective can bias tuning, so semantic and anchor gates remain mandatory.
- Persistent cross-process tuning memory is not yet implemented; the reference tuner uses bounded in-process EWMA memory.

## Follow-up

Before promotion from proposed to accepted, CI should execute focused tests for shadow-state isolation, candidate rejection, semantic-distance gating, mechanics commit behavior, and improvement-memory accounting.
