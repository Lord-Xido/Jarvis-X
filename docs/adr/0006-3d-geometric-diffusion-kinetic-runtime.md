# ADR-006: Adopt the 3D geometric diffusion kinetic runtime boundary

**Status:** Accepted  
**Date:** 2026-08-15  
**Extends:** ADR-004, ADR-005

## Context

Jarvis-X already defines a bounded same-space field runtime, the Moagi-Helmholtz generative/orchestration contract, and an orthogonal transform precision boundary. The next research step requires one explicit contract for the recurring system pattern discussed across the 3D auto-encoding/decoding work:

```text
observe
-> encode
-> relational geometry
-> bounded branching
-> graph diffusion
-> inward refinement
-> decode / manifest
-> reverse verification
-> memory
-> bounded mechanics evolution
-> re-enter
```

Without a precise boundary, phrases such as "3D LLM", "graphical diffusion", "exact image generation" and "self-evolution" can be mistaken for claims about physical model internals, arbitrary inversion, or unrestricted source-code mutation.

Jarvis-X therefore needs a deterministic reference semantics that keeps the useful geometry while preserving the repository's existing authority, evidence, rollback and scale rules.

## Decision

Jarvis-X adopts a **virtual 3D geometric diffusion kinetic runtime** as a Layer 5 research contract.

The 3D coordinates are a computational representation. They do not imply that a transformer or language model physically stores its hidden state in Euclidean 3D space.

A relational state is represented by an undirected finite graph

```text
G = (V, E)
```

where each node contains

```text
(position in R^3, feature in R^d).
```

The graph is topology-validated before diffusion or publication.

### Forward graphical diffusion

For a state component `z` and declared `beta in [0,1]`, the reference corruption law is

```text
z_tau = sqrt(1-beta) * z_(tau-1) + sqrt(beta) * epsilon
```

where the reference implementation uses a seeded deterministic pseudo-random stream for reproducible conformance fixtures.

This is a research corruption operator. It is not evidence that a production diffusion model, image generator, or LLM uses this exact implementation.

### Geometry-conditioned reverse step

The reference reverse operator contracts a candidate toward an immutable per-cycle observation/anchor while smoothing feature state over graph adjacency:

```text
p_i' = p_i + g (p_i^anchor - p_i)

h_i' = h_i
       + g (h_i^anchor - h_i)
       + gamma (mean_{j in N(i)} h_j - h_i)
       + mu Omega_i.
```

The implementation then applies an explicit projection to bound per-component position and feature displacement.

### Candidate-first transaction

One runtime cycle is

```text
observation
-> validate topology/resources
-> forward graphical diffusion
-> geometry-conditioned reverse denoising
-> bounded memory injection
-> projection Pi_Lambda
-> reconstruction metrics
-> verification threshold
-> optional external validator
-> COMMIT or ROLLBACK
-> provenance / telemetry
```

No candidate becomes authoritative before the complete gate passes.

### Bounded branching

Exploration may construct multiple seeded diffusion candidates, but branch width is an explicit positive integer resource bound. Branch generation does not itself authorize execution or publication.

### Exactness semantics

Jarvis-X does not define "exact image generation" as a promise that an underspecified prompt uniquely determines an arbitrary target image.

Exactness is meaningful only relative to an explicit target, representation, metric and tolerance, for example

```text
d(target, reconstruction) <= epsilon.
```

Geometric, perceptual, semantic and pixel metrics are distinct and must not be conflated.

### Memory

Working memory stores bounded residual feature information. Memory is state, not authority: it is an input to the next candidate and remains subordinate to projection and verification.

### System auto-evolution

The reference evolution layer is restricted to **versioned runtime-configuration candidates**. It does not rewrite arbitrary source code.

A candidate mutation `mu` is promoted only when

```text
Fitness(candidate) > Fitness(current)
AND verification(candidate) >= threshold
AND all existing policy/resource/invariant gates pass.
```

Otherwise the current configuration remains authoritative.

The reference fitness score combines normalized quality, reliability, efficiency and coherence, with an explicit fault penalty. Production systems may use other declared objectives, but they must preserve candidate-first promotion and rollback semantics.

## Relationship to the Dr Moagi recurrence

The architecture-level recurrence remains a compositional research notation:

```text
Xi_(t+1)^3D = Pi_Lambda[
    Xi_t^3D
    + A_3D(Xi_t)
    + MLP(Xi_t)
    + P_(1:M)^<-(Xi_t)
    + lambda D_tau^graph(Xi_t)
    + Omega_t
    - E_t
    + kappa R_t^<-
    - eta grad L_t
    - zeta grad C_t
    + U_tool
].
```

Terms combined in an executable implementation must still satisfy the existing same-space/type/units rule. This ADR does not permit arbitrary addition of incompatible latent, field, token, image or tool-state values.

## Architectural consequences

1. The canonical 64-bit VM is unchanged.
2. The new runtime is a Layer 5 research primitive and may depend on Layer 4 sparse/graph representations, never the reverse.
3. Visualization remains Layer 6 and is non-authoritative unless separately promoted.
4. Graph diffusion is explicitly separated from pixel diffusion and from ordinary transformer attention.
5. Virtual 3D coordinates are representation metadata, not a hardware-performance claim.
6. Reverse decoding is inference unless side information and a restricted domain establish exact replay.
7. Self-evolution is bounded configuration adaptation, not unrestricted self-modification.
8. Tool actions remain behind the existing policy/transaction boundary.
9. Numerical or perceptual tolerances may not hide malformed topology, non-finite state or a failed lower-level precision gate.

## Validation

Acceptance requires executable tests for:

- graph topology normalization and invalid-edge rejection;
- seeded deterministic forward diffusion;
- reverse denoising reducing distance to an immutable anchor in the declared fixture;
- explicit graph smoothness/dispersion telemetry;
- candidate rollback on numerical or validator failure;
- bounded deterministic branch generation;
- finite, shape-correct working memory;
- mutation promotion only when both fitness and verification gates pass.

The dependency-free reference implementation is `src/jarvisx/geometric_diffusion_runtime.py` with focused tests in `tests/test_geometric_diffusion_runtime.py`.
