# ADR-032: Tri-Space Fixed-Point SDF and Differentiable Vision Closure

**Status:** Proposed for integration  
**Date:** 2026-09-30  
**Applies to:** Dr Moagi geometric autoencoders, differentiable rendering, multimodal vision loops, latent fixed-point optimization  
**Extends:** ADR-016, ADR-017, ADR-028 and ADR-030  
**Canonical specification:** `docs/DR_MOAGI_TRISPACE_FIXED_POINT_SDF.md`

## Context

Jarvis-X already defines candidate-first autoencoding, bounded fixed-point refinement,
geometry receipts and explicit verification boundaries. A new formulation couples three
spaces in one recursive loop:

1. a continuous 3D signed-distance geometry;
2. a differentiable 2D rendered image;
3. a Transformer/vision latent space used to refine the geometry.

Without a repository-level contract, several mathematically important distinctions are
easy to blur: a fixed point versus idempotence, hard occupancy versus differentiable
occupancy, electrical conductivity versus optical extinction, discrete autodiff
Jacobians versus exact physical-continuum derivatives, and raw Newton steps versus
bounded second-order optimization.

## Decision

Jarvis-X adopts
`docs/DR_MOAGI_TRISPACE_FIXED_POINT_SDF.md`
as the research specification for this coupled geometry-rendering-vision loop.

The canonical fixed-point law is

[
mathbf z^*=Phi(mathbf z^*).
]

Therefore

[
Phi(Phi(mathbf z^*))=Phi(mathbf z^*)
]

is treated as a consequence of fixed-point closure, not as the primary definition.

The composite operator is modeled as

[
Phi
=
mathcal U
circ
mathcal F
circ
mathcal V
circ
mathcal R
circ
mathcal G,
]

with geometry decode, differentiable rendering, vision encoding, evidence fusion and a
bounded latent update as explicit semantic stages.

## Required semantics

Implementations claiming conformance SHALL preserve the following distinctions.

1. A neural SDF defines geometry through its zero level set and approximately satisfies
   the Eikonal property through an explicit residual or loss.
2. Surface normals are derived from normalized spatial SDF gradients.
3. Training/inverse rendering uses a smooth SDF-to-occupancy or SDF-to-opacity map;
   a hard interior/exterior step is only a limiting or inference-time representation.
4. Electrical conductivity and optical extinction are separate fields.
5. A 1.0 mm diameter wire with a 2.0 mm inner gap has 3.0 mm centerline spacing.
   A semicircular planar hairpin therefore has centerline bend radius 1.5 mm and
   curvature (2/3 {m mm}^{-1}).
6. A differentiable renderer may expose autodiff Jacobians for its implemented
   computational graph, but this is not described as an exact physical-continuum
   Jacobian without a separate proof.
7. Material reflection is represented explicitly; a microfacet BRDF should retain
   Fresnel, distribution and geometry/masking terms.
8. Full Hessian inversion is not required and is not the default operational update.
   Damped Newton, Gauss-Newton/Levenberg-Marquardt, Hessian-vector methods or L-BFGS
   are admissible bounded approximations.
9. A fixed-point claim must record a residual such as
   (|Phi(z)-z|), iteration/resource bounds and a termination reason.
10. Contractivity claims must identify the actual map, norm/metric, domain and measured
    or bounded Jacobian/Lipschitz condition.
11. Visual/perceptual agreement does not replace explicit dimensional geometry checks.
12. Fixed-point convergence is internal numerical evidence and not proof of physical
    correctness or external-world correspondence.

## Hairpin dimensional contract

For wire radius

[
a=0.5 {m mm}
]

and inner gap

[
g=2.0 {m mm},
]

the parallel-leg centerline separation is

[
D_{m cc}=g+2a=3.0 {m mm}.
]

For a semicircular U-bend,

[
R=rac{D_{m cc}}{2}=1.5 {m mm},
qquad
kappa=rac1R=rac23 {m mm}^{-1},
qquad
	au=0.
]

For total centerline length (L=1000 {m mm}), equal straight legs have length

[
L_{m leg}
=
rac{1000-pi R}{2}
approx
497.6438055 {m mm}.
]

These values form the nominal analytical geometry receipt for the reference hairpin.

## Differentiable rendering boundary

The renderer SHALL distinguish

[
sigma_e(mathbf x)
]

for electrical conductivity from

[
alpha(mathbf x)
]

for optical extinction/opacity.

A smooth training map may use

[
o_eta(mathbf x)
=
operatorname{sigmoid}(-eta f_	heta(mathbf x)),
]

with the binary solid recovered only as the sharpness limit.

If volumetric rendering is used, transmittance is

[
T(t)
=
expleft(
-int_{t_n}^{t}alpha(mathbf r(s)),ds
ight),
]

and image formation is a differentiable integral or a documented discrete
approximation thereof.

## Optimization boundary

The preferred second-order form is

[
(H_k+lambda_k I)Delta z_k=-
ablamathcal L(z_k),
]

followed by a bounded candidate update

[
z_{k+1}
=
Pi_{mathcal Z}(z_k+eta_kDelta z_k).
]

The damping, step scale, projection domain and stopping criteria must be explicit.

## Transaction boundary

This ADR does not create a new authority path.

A converged tri-space state remains a candidate under ADR-016/017 semantics:

```text
latent geometry
 -> SDF
 -> dimensional geometry checks
 -> differentiable render
 -> vision tokens
 -> evidence fusion
 -> bounded latent update
 -> fixed-point / contractivity receipt
 -> CTR / external verification
 -> atomic commit OR rollback
```

No internal optimizer or renderer may directly promote authoritative state.

## Consequences

### Positive

- gives the geometry-rendering-vision loop one precise fixed-point definition;
- removes a non-differentiable hard-density assumption from training semantics;
- makes the hairpin dimensional geometry exactly reproducible;
- separates optical and electrical material quantities;
- makes Jacobian claims numerically precise;
- replaces unsafe/unscalable raw Hessian inversion with bounded second-order methods;
- keeps multimodal perceptual evidence subordinate to physical dimensional constraints;
- preserves the repository's existing candidate/verify/commit law.

### Costs

- implementations need explicit geometry and optimizer receipts;
- a differentiable renderer must document its discretization and opacity model;
- contractivity cannot be asserted from symbolic form alone;
- exact copper optical parameters require a declared wavelength/channel model or
  calibrated approximation;
- production claims still require independent benchmarks and external verification.

## Validation

A conforming implementation should demonstrate:

1. an SDF zero-set geometry;
2. Eikonal residual measurement;
3. normal computation from SDF gradients;
4. 1.0 mm diameter and 2.0 mm inner-gap dimensional checks;
5. the 1.5 mm bend-radius / (2/3 {m mm}^{-1}) curvature relation;
6. a smooth differentiable opacity path;
7. a documented copper BRDF/material model;
8. image-to-vision-token encoding;
9. a fixed-point residual and bounded optimizer receipt;
10. commit/rollback through the enclosing Jarvis-X verification boundary.

## Non-goals

This ADR does not claim that the current repository already contains a production-grade
neural SDF renderer, a calibrated copper optics model, a globally contractive learned
map, an exact continuum Jacobian, or independently validated physical reconstruction.
