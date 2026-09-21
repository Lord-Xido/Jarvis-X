# ADR-025: Finite-N Dr Moagi Code Print-Rate Continuum

## Status

Accepted.

## Context

The DM-vOmegaXi+ code-generation plane already measures physical source-line emission and keeps throughput claims empirical. A recursive formulation introduces an inward geometric rate continuum with geometric contraction, adaptive validation tolerance, recurrent memory, and operator optimization.

A direct transcription creates four problems:

1. `K_in(r)=r^-2 exp(-r/r0)` is pointwise singular at the origin.
2. Multiplying projected state by a failed validation indicator destroys the last valid state.
3. `1/Lambda` is singular at zero.
4. Measured wall-clock throughput is not differentiable with respect to source code or runtime scheduling.

## Decision

Implement a finite-N reference runtime in `src/jarvisx/dr_moagi_code_rate_continuum.py`.

The runtime:

- contracts a real 3D radius vector using `r_(n+1)=gamma R_z(omega) r_n`;
- integrates the inward kernel against exact spherical shell volume;
- gates rate contribution on validation while preserving the last committed operator on rejection;
- floors inverse-rate denominators and epsilon;
- uses an exponentially weighted measured-rate memory;
- replaces fake throughput gradients with measured derivative-free candidate selection;
- reports `Lambda_N` and an assumption-conditioned empirical geometric tail bound instead of claiming to execute `N=infinity`;
- reuses the existing deterministic code emitter as the physical measurement backend.

## Mathematical consequence

For `r_(n+1)=gamma r_n` in norm,

```text
K_in(r_n) Delta V_n
  = 4*pi/3 (1-gamma^3) r_n exp(-r_n/r0),
```

so the shell weighting is geometrically summable for bounded local rates.

## Consequences

The architecture gains a falsifiable operational bridge from the asymptotic equation to measured code emission while preserving the repository boundary:

```text
logical/asymptotic model != finite implementation != measured performance.
```

No infinite-throughput or hardware-independent performance claim follows from the continuum notation.
