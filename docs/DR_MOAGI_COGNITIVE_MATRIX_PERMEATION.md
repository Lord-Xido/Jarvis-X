# Dr Moagi Cognitive Matrix Permeation Reference

**Status:** Bounded numerical reference  
**Date:** 2026-09-29  
**Extends:** ADR-016, ADR-017, ADR-028, ADR-031

## Purpose

This document binds the existing 3D Cognitive Matrix visualization to a finite, testable numerical substrate without treating its virtual extent as resident memory or its rendered telemetry as measurement.

The reference implementation is:

- `src/jarvisx/cognitive_matrix_permeation.py`
- `tests/test_cognitive_matrix_permeation.py`

The browser visualization remains:

- `examples/3D_Cognitive_Matrix_Operational_Mechanics.html`
- `docs/3D_COGNITIVE_MATRIX_OPERATIONAL_MECHANICS.md`

The numerical reference does not make the browser authoritative. It emits bounded evidence receipts that may be supplied to CTR / Pi_Lambda.

## Virtual domain and active support

For logical axis size N, the virtual domain contains N^3 sites. At the short-scale septillion example N = 10^24, the logical extent is 10^72 sites.

This is metadata describing the address universe. The executable reference materializes only M particles, with M bounded by configuration. No array proportional to 10^72 is allocated.

## Sampled input, encoding and reconstruction

The bounded input ensemble is a seeded 3D Gaussian cloud. The reference encoder is the componentwise bounded map:

`z = tanh(s * x)`

with inverse:

`x_hat = atanh(z) / s`.

This same-width reference exists to verify state flow and reconstruction accounting. It is not a trained compressor and does not claim dimensionality reduction.

Reconstruction evidence is the mean squared coordinate error across the sampled ensemble.

## Three geometric phases

The visualization geometry is separate from the encoded numerical value:

1. phase 0: sampled Gaussian input cloud;
2. phase 1: toroidal latent visualization;
3. phase 2: ordered cubic output lattice.

Transitions use cubic smoothstep interpolation. The rendered phase is a projection of state, not an alternate authority namespace.

## Radial permeation PDE

The reference solves a bounded radially symmetric reaction-diffusion equation:

`dPsi/dt = D * (d2Psi/dr2 + (2/r) * dPsi/dr) + S(r) - gamma * Psi`

with Gaussian source:

`S(r) = S0 * exp(-r^2 / (2 * sigma_S^2))`.

The origin uses the symmetric radial limit. The outer boundary is Dirichlet, Psi(R) = 0. Explicit integration is internally sub-stepped against a conservative diffusion stability bound.

This is a computational scalar field. It is not a claim of physical energy, electromagnetism, biology or consciousness.

## Particle response

Particles respond to the radial field gradient using an overdamped Langevin update:

`x(t+dt) = x(t) + mu * grad(Psi) * dt + sqrt(2 * Dp * dt) * eta`

where eta is standard Gaussian noise. The stochastic term scales with sqrt(dt), and positions are bounded by the configured simulation radius.

## Entropy receipt

Spatial entropy is finite histogram Shannon entropy over the sampled 3D particle positions:

`H = -sum_k p_k log2(p_k)`.

This is not thermodynamic entropy and is not assumed to be monotonic across visual phases.

## Permeation current and flux

For the diffusive field, the radial current is:

`J_r = -D * dPsi/dr`.

The spherical-boundary flux is:

`Phi = 4 * pi * R^2 * J_r(R)`.

A HUD percentage is normalized against an explicit configured reference magnitude and clamped to 100 percent. The raw flux proxy and the normalized display value are both reported.

## Receipt and authority

One step emits a `CognitiveMatrixReceipt` containing:

- virtual axis, virtual site count and virtual bit count;
- bounded active particle count;
- current geometric phase;
- reconstruction MSE;
- spatial histogram entropy;
- permeation flux and normalized flux percentage;
- maximum field value;
- finite-state status;
- acceptance status;
- fixed claim status `bounded_stochastic_projection`.

The receipt is non-authoritative:

```text
candidate state
  -> Cognitive Matrix receipt
  -> optional cognitive geometry receipt
  -> CTR
  -> Pi_Lambda
  -> COMMIT or ROLLBACK
```

## Relation to ADR-031

The Cognitive Matrix is a bounded spatial/numerical adapter of the self-hosting runtime, not a competing authority model.

```text
virtual field metadata
  -> bounded stochastic active support
  -> encode
  -> latent geometric projection
  -> permeation PDE
  -> Langevin response
  -> reconstruction / entropy / flux receipts
  -> CTR contrast
  -> Pi_Lambda verification
  -> COMMIT or ROLLBACK
  -> re-encode
```

The locked ADR-031 invariant remains:

`SELF-HOSTING != SELF-AUTHORIZING`

## Claim boundary

This reference does not establish physical residency of the septillion-scale field, physical energy transport, a trained variational autoencoder, lossless compression into a smaller latent state, consciousness, AGI, external truth from low reconstruction error, or hardware throughput from logical scale.

Browser HUD values remain synthetic unless explicitly wired to numerical receipts.

Originator of the Dr Moagi 3D Ephemeral-Notion Intelligence Framework: Matladi Maxwell Moagi (Lord-Xido). Canonical provenance: `docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`.
