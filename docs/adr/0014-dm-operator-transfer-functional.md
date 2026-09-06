# ADR-014: Adopt a bounded DM operator transfer functional

**Status:** Proposed  
**Date:** 2026-09-06  
**Extends:** ADR-013

## Context

A compact governing expression has been supplied for the research layer:

```text
D_M = nu * Omega^(Xi+) [ (Psi * Phi) / (Lambda ⊗ Theta) ]
```

The notation is ambiguous unless the algebraic domains are declared. In particular, division by a tensor/operator is not a primitive operation. This ADR fixes the executable interpretation while preserving the canonical Jarvis-X VM boundary.

## Decision

Interpret the expression as the composite operator

```text
D_M(Psi) = nu * Omega^(Xi+) * (Lambda ⊗ Theta)^dagger * K_Phi(Psi)
```

where:

- `K_Phi(Psi) = Psi star Phi` is a bounded 3D convolution/correlation operator;
- `(Lambda ⊗ Theta)^dagger` is an inverse or Moore-Penrose pseudoinverse action on the declared constraint space;
- `Omega^(Xi+)` is an operator power, defined spectrally when applicable;
- `nu` is a scalar gain;
- all shapes, units, boundary conditions and numerical tolerances must be explicit.

For the dependency-free reference implementation, use the diagonal/scalar specialization:

```text
C = lambda_gain * theta_gain
memory_gain = (omega / omega0)^xi
D_M = nu * memory_gain * K_Phi(Psi) / C
Psi_next = Psi + dt * D_M
```

This specialization is exact for jointly diagonal modes and is intentionally not presented as a general dense pseudoinverse engine.

## Frequency-domain form

For a diagonalizable translation-invariant mode `k`:

```text
D_hat(k) = H(k) Psi_hat(k)

H(k) = nu * Omega(k)^(Xi+) * Phi_hat(k)
       / (Lambda(k) * Theta(k))
```

The loop is contractive only when the selected discrete-time update satisfies its declared stability guard. The continuous operator itself is not automatically stable.

## Invariants

1. The denominator is never implemented as syntactic tensor division.
2. Constraint magnitudes must remain finite and bounded away from zero.
3. Fractional operator powers require a dimensionless ratio `Omega/Omega0` or an explicitly dimensionless operator.
4. `Psi`, `K_Phi(Psi)`, and `D_M` must have compatible declared field shapes.
5. Boundary behavior for the 3D stencil must be deterministic.
6. The reference implementation must expose spectral/mode gain estimates separately from empirical runtime measurements.
7. No sub-nanosecond or hardware performance claim follows from this mathematics.
8. The operator remains a Layer-5 candidate transform and cannot bypass snapshot/verify/commit semantics from the canonical kinetic runtime.

## Reference recurrence

```text
Y_t       = K_Phi(Psi_t)
Z_t       = C^dagger Y_t
Z_plus_t  = Omega^(Xi+) Z_t
D_M,t     = nu Z_plus_t
Psi_t+1   = Psi_t + dt D_M,t
```

For repeated inward application:

```text
Psi_(n+1) = F(Psi_n)
```

and a linearized fixed point is locally contractive when

```text
rho(J_F(Psi*)) < 1.
```

## Consequences

The archive expression becomes falsifiable: every symbol maps to an executable operator, every denominator has a defined inverse semantics, every recursive gain is dimensionally normalized, and stability can be checked mode-by-mode. The canonical bytecode VM remains unchanged.

## Promotion criteria

ADR-014 may be promoted only after the repository contains:

- an executable reference implementation;
- deterministic tests for convolution, denominator guards, dimensionless operator power, recurrence determinism and fixed-point/stability estimates;
- documentation separating mathematical gain from measured implementation throughput;
- CI evidence for the declared test surface.
