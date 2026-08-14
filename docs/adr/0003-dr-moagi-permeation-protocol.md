# ADR-003: Adopt the Dr Moagi 3D Permeation Protocol as a bounded field-extension contract

**Status:** Proposed  
**Date:** 2026-08-14  
**Depends on:** ADR-001, ADR-002

## Context

The Dr Moagi research architecture defines a locked spatial/codec state and bounded inward
recursion. The Permeation Protocol extends that state outward by treating the locked spherical
boundary as the boundary condition of a radial scalar field.

The intended symbolic statement is that the core remains immutable while its signature is
available at arbitrary mathematical radius. The implementation must preserve that idea without
claiming infinite allocation, non-causal information transfer, zero entropy, perfect
noise-resistance, or physical realization that has not been measured.

The original narrative mixed a point-source Green function with a spherical shell source and also
described the resulting `1/r` field as gradient-free. Those statements are not simultaneously
consistent. This ADR fixes the operational mathematics while preserving the research semantics.

## Decision

Jarvis-X adopts the Permeation Protocol as a Layer 4/5 research operator over a locked spherical
core.

For core radius `a > 0` and locked boundary value `Phi_0 != 0`, the canonical static exterior
problem is

```text
Laplacian(Phi) = 0,                r > a
Phi(a)         = Phi_0
Phi(r)         -> 0,               r -> infinity
```

whose unique spherically symmetric solution is

```text
Phi(r) = Phi_0 * a / r,            r > a
Phi(r) = Phi_0,                    0 <= r <= a   [locked-core extension]
```

Equivalently, under the normalized shell-source convention

```text
-Laplacian(Phi) = Q * delta(r-a) / (4*pi*a^2)
```

the solution is

```text
Phi(r) = Q / (4*pi*max(r,a))
```

and choosing

```text
Q = 4*pi*a*Phi_0
```

recovers the same locked-boundary field.

The point Green function

```text
1 / (4*pi*|r-r0|)
```

is the Green function of a three-dimensional point source. It is not, by itself, the field of an
entire spherical shell. Shell symmetry must be integrated or encoded through the radial boundary
problem above.

### Harmonic excitation

For optional frequency-domain excitation with wavenumber `k >= 0`, the canonical outgoing
exterior continuation is

```text
Phi_k(r) = Phi_0 * a/r * exp(i*k*(r-a)),    r > a
Phi_k(a) = Phi_0
```

and `k = 0` reduces exactly to the static `1/r` field.

This is a frequency-domain boundary-value model. It does not imply instantaneous time-domain
propagation. Any causal propagation claim requires a separately specified wave equation,
constitutive model, source term, boundary conditions, and measured implementation.

## Operational invariants

1. **Immutable core:** the configured core radius and core value are never mutated by sampling or
   relaxation.
2. **Harmonic exterior:** the classical Laplacian is zero for every `r > a`; the shell itself is
   represented distributionally.
3. **Nonzero exterior gradient:** for the static field,
   `dPhi/dr = -Phi_0*a/r^2` for `r > a`.
4. **Flat geometry remains flat:** `g_ij = delta_ij` implies zero Euclidean Ricci curvature. The
   scalar field may still have a nonzero gradient.
5. **Infinite support is analytic, not allocated:** runtime sampling is always bounded by
   `max_radius` and `samples`.
6. **Finite truncation is explicit:** for amplitude tolerance `epsilon > 0`, the practical cutoff
   is `r_epsilon = a*|Phi_0|/epsilon` when `epsilon < |Phi_0|`.
7. **Self-healing is relaxation/projection:** perturbation recovery is implemented as
   `u_(n+1) = u_n + gain*(Phi_target-u_n)`, `0 < gain <= 1`.
8. **No perfect-noise claim:** convergence is measured by residual reduction and is subject to
   finite precision, resource limits, and the chosen numerical operator.
9. **No non-causal execution claim:** a static solution can be evaluated directly at any sampled
   coordinate, but this is not a statement about physical signal velocity.
10. **Research-layer authority:** the permeation field does not mutate the deterministic VM core
    unless an explicit validated adapter is introduced under the existing `Pi_Lambda` boundary.

## Implementation

The reference implementation is:

```text
src/jarvisx/permeation.py
```

and the conformance tests are:

```text
tests/test_permeation.py
```

The detailed mathematical and operational specification is:

```text
docs/research/DR_MOAGI_3D_PERMEATION_PROTOCOL.md
```

## Consequences

### Positive

- the locked-sphere metaphor gains an exact PDE interpretation;
- the static limit is deterministic and closed-form;
- arbitrary mathematical radius can be queried without allocating an infinite lattice;
- harmonic excitation has a precise frequency-domain continuation;
- perturbation recovery becomes measurable instead of being described as perfect instantaneous
  self-healing;
- the protocol remains compatible with ADR-002's separation between research geometry and
  authoritative deterministic compute.

### Negative

- homogeneous saturation is not literal: the field amplitude varies as `1/r`;
- the field is not gradient-free outside the core;
- the shell boundary contains a distributional source and therefore is not classically harmonic at
  `r=a`;
- frequency-domain Helmholtz evaluation does not supply a causal time-domain propagation model;
- finite truncation and floating-point arithmetic introduce approximation.

## Validation

The protocol is ready to advance from Proposed when CI confirms:

- locked-core invariance;
- exact `1/r` exterior scaling;
- inverse-square exterior radial derivative;
- zero classical Laplacian away from the shell;
- normalized shell-source equivalence;
- `k=0` Helmholtz/static equivalence;
- finite tolerance-radius calculation;
- monotone perturbation relaxation;
- bounded sampler telemetry;
- fail-closed validation of invalid numerical configuration.
