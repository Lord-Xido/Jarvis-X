# ADR-032: Couple 3D SDF geometry, differentiable 2D projection, and latent fixed-point refinement

**Status:** Proposed  
**Date:** 2026-09-30  
**Extends:** ADR-002, ADR-013, ADR-031

## Context

Jarvis-X already separates authoritative sparse geometry, rendering, latent adaptation, and candidate-first commit boundaries. A recurring research path now requires one auditable operator that evaluates a geometric candidate in 3D, projects it to 2D, embeds the projection into a latent attention representation, and applies a bounded second-order update.

The motivating reference geometry is a 1000 mm planar copper hairpin with 1.0 mm diameter, 2.0 mm inner gap, 3.0 mm centerline separation, and 1.5 mm U-bend centerline radius.

Several mathematical distinctions must remain explicit:

- a hard inside/outside material field is discontinuous at the SDF zero set;
- an exact Euclidean SDF satisfies the Eikonal equation only where differentiable;
- ordinary raster/volume rendering is not globally invertible to unique geometry;
- Newton updates require regularization or trust control when the Hessian is ill-conditioned;
- a small recursive residual is evidence of local update stability, not proof of a unique global fixed point.

## Decision

Adopt `docs/DR_MOAGI_COUPLED_3D_2D_LATENT_FIXED_POINT.md` as the bounded research specification and `src/jarvisx/dr_moagi_coupled_fixed_point.py` as its dependency-free executable reference.

The coupled operator is defined over typed spaces:

```text
Z_geom
  -> SDF geometry
  -> smooth render surrogate
  -> I in R^(H x W)
  -> patch tokens
  -> attention features
  -> constrained loss
  -> damped second-order update
  -> Z_geom'
```

The authoritative solid remains the hard SDF. Only the rendering/optimization path may use smooth occupancy.

The recursive stability certificate is

```math
r_\Phi(z)=\|\Phi(\Phi(z))-\Phi(z)\|_2.
```

Promotion into any authoritative runtime state remains subject to the normal Jarvis-X candidate-first policy, validation, resource bounds, provenance, commit, and rollback rules.

## Consequences

Positive consequences:

- the geometry, projection, and latent spaces are connected by explicit typed operators;
- the requested hairpin dimensions are executable and testable;
- differentiability claims no longer rely on a contradictory hard density step;
- Eikonal behavior is measurable without pretending the SDF is smooth on medial loci;
- second-order optimization is bounded by damping and a trust radius;
- the fixed-point claim has an operational residual.

Costs and limitations:

- the reference renderer is an opacity/silhouette integrator, not a full copper Fresnel/microfacet renderer;
- attention uses a small dependency-free Q=K=V reference rather than learned Transformer projections;
- derivatives and Hessians are finite-difference approximations in the reference backend;
- production neural-SDF, autodiff, and BRDF backends remain optional Layer 5 implementations.

## Verification

`tests/test_dr_moagi_coupled_fixed_point.py` verifies:

- requested hairpin dimensions and arc-length accounting;
- SDF sign and zero-set behavior;
- sampled Eikonal residual away from non-smooth loci;
- hard material occupancy;
- 2D renderer separation of solid and empty rays;
- patch-token and attention dimensionality;
- loss reduction under the damped Newton reference step;
- zero recursive residual for an idempotent projection;
- loss-reducing execution of one complete 3D -> 2D -> latent update.