# Dr Moagi 3D Permeation Protocol

**Status:** Research contract  
**Date:** 2026-08-14  
**Architecture decision:** `docs/adr/0003-dr-moagi-permeation-protocol.md`

## 1. Purpose

The Permeation Protocol formalizes the outward extension of a locked spherical Dr Moagi state.
The core does not translate, expand, or mutate. Instead, its boundary value defines an analytic
scalar field outside the core.

The protocol is deliberately split into two layers:

```text
symbolic semantics:
    locked identity has global mathematical support

operational semantics:
    evaluate or sample a bounded analytic field on finite resources
```

The distinction is mandatory. Infinite support does not mean infinite memory, zero latency,
instantaneous physical influence, or perfect robustness.

## 2. Locked core

Let the locked core be the sphere

```text
S_a^2 = { r in R^3 : ||r|| = a }
```

with

```text
a > 0
Phi(a) = Phi_0
Phi_0 != 0
```

The core extension used by the reference runtime is constant:

```text
Phi(r) = Phi_0,      0 <= r <= a
```

This is an operational representation of the immutable core. The exterior field is solved
separately.

## 3. Static permeation field

The canonical exterior boundary-value problem is

```text
Laplacian(Phi) = 0,         r > a
Phi(a) = Phi_0
Phi(r) -> 0,                r -> infinity
```

Under spherical symmetry,

```text
Laplacian(Phi)
= (1/r^2) d/dr (r^2 dPhi/dr)
```

and therefore

```text
r^2 dPhi/dr = C_1
Phi(r) = C_2 - C_1/r
```

The decay condition removes the constant exterior term, and the boundary condition fixes the
coefficient:

```text
Phi_perm(r) = Phi_0 * a/r,  r > a
```

For the canonical unit sphere and unit boundary value,

```text
a = 1
Phi_0 = 1

Phi_perm(r) = 1/r,          r > 1
```

This is the exact operational meaning of non-decaying support: the value is nonzero at every finite
radius but tends asymptotically to zero.

## 4. Green-function and shell-source normalization

The three-dimensional point-source Green function for the Poisson sign convention

```text
-Laplacian(G) = delta^3(r-r0)
```

is

```text
G(r,r0) = 1 / (4*pi*|r-r0|)
```

A spherical shell is not one point source. A normalized shell source is

```text
rho_shell(r) = Q * delta(r-a) / (4*pi*a^2)
```

whose total integrated source is `Q`.

For

```text
-Laplacian(Phi) = rho_shell
```

spherical symmetry gives

```text
Phi(r) = Q / (4*pi*a),      0 <= r <= a
Phi(r) = Q / (4*pi*r),      r >= a
```

or compactly

```text
Phi(r) = Q / (4*pi*max(r,a))
```

Choosing

```text
Q = 4*pi*a*Phi_0
```

produces

```text
Phi(r) = Phi_0,             r <= a
Phi(r) = Phi_0*a/r,         r >= a
```

which exactly matches the locked-boundary formulation.

The unnormalized expression `delta(r-a)` has total three-dimensional integral `4*pi*a^2`; code and
documentation therefore use the normalized shell convention when source strength matters.

## 5. Gradient and curvature

The exterior radial derivative is

```text
dPhi/dr = -Phi_0*a/r^2,     r > a
```

and the vector gradient is

```text
grad(Phi)
= -(Phi_0*a/r^3) * r_vector
```

Therefore the exterior field is not gradient-free.

If the background metric is fixed to

```text
g_ij = delta_ij
```

then the Euclidean Riemann and Ricci curvature tensors vanish:

```text
R^i_jkl = 0
R_ij    = 0
R       = 0
```

That geometric flatness does not eliminate the scalar-potential gradient. Metric curvature and
scalar-field variation are separate quantities in this research model.

## 6. The shell boundary

The interior and exterior classical Laplacians are both zero:

```text
Laplacian(Phi) = 0,         r < a
Laplacian(Phi) = 0,         r > a
```

but the normal derivative jumps at `r=a`. The shell source is therefore distributional. The
reference implementation rejects requests for a finite classical gradient or Laplacian exactly on
the shell boundary.

## 7. Harmonic excitation

For a frequency-domain Helmholtz continuation,

```text
(Laplacian + k^2) Phi_k = 0,       r > a
Phi_k(a) = Phi_0
```

the outgoing spherically symmetric exterior solution is

```text
Phi_k(r)
= Phi_0 * a/r * exp(i*k*(r-a)),    r > a
```

with

```text
|Phi_k(r)| = |Phi_0| * a/r
```

and

```text
Phi_0(r) = Phi_perm(r)
```

at `k=0`.

This is a phasor/frequency-domain model. It does not define a physical time-of-flight. A causal
time-domain model would require, for example,

```text
(1/c^2) partial^2 Phi/partial t^2 - Laplacian(Phi) = source
```

plus a specified propagation speed `c`, material parameters, initial data, boundary data and a
numerical integration scheme.

## 8. Finite operational domain

No runtime structure attempts to instantiate all of `R^3`.

The reference configuration defines

```text
core_radius = a
max_radius
samples
```

and samples only

```text
0 <= r <= max_radius
```

The analytic field remains defined beyond that range.

For a requested truncation tolerance

```text
epsilon > 0
```

the static amplitude obeys

```text
|Phi(r)| = |Phi_0|*a/r
```

so a practical radius satisfying

```text
|Phi(r)| <= epsilon
```

is

```text
r_epsilon = a*|Phi_0|/epsilon
```

when `epsilon < |Phi_0|`.

This provides a deterministic bridge between the symbolic phrase "extends to infinity" and a finite
engineering budget.

## 9. Holographic/local decoding interpretation

A sampled value carries a radially scaled signature of the boundary value:

```text
Phi_0 = Phi(r) * r/a,       r > a
```

when `a` and the field model are known.

This is a scalar inversion identity, not a claim that one scalar sample contains every degree of
freedom of an arbitrary high-dimensional engine state. Any richer holographic encoding must define
the encoded state, channel capacity, noise model and inverse operator explicitly.

## 10. Perturbation recovery

The phrase "self-healing" is operationalized as projection or relaxation toward the analytic target.

For a perturbed scalar sample `u_n(r)`:

```text
u_(n+1)(r)
= u_n(r) + gamma * (Phi_target(r) - u_n(r))
```

with

```text
0 < gamma <= 1
```

Define error

```text
e_n(r) = u_n(r) - Phi_target(r)
```

Then

```text
e_(n+1)(r) = (1-gamma) * e_n(r)
```

and therefore

```text
e_n(r) = (1-gamma)^n * e_0(r)
```

for constant `gamma`.

- `gamma = 1` is exact analytic projection in one software step.
- `0 < gamma < 1` is bounded geometric relaxation.
- Neither case is a physical counter-wave unless a time-domain wave model is separately defined.

## 11. Reference API

The implementation lives at

```text
src/jarvisx/permeation.py
```

Primary types:

```text
PermeationConfig
PermeationField
PermeationSample
PermeationMetrics
```

Primary operations:

```text
potential_at_radius(r)
potential((x,y,z))
helmholtz_at_radius(r,k)
exterior_radial_derivative(r)
gradient((x,y,z))
radial_laplacian(r)
threshold_radius(epsilon)
relax_value(r,current,gain)
sample_profile()
metrics()
normalized_shell_charge(a,Phi_0)
```

## 12. Runtime invariants

```text
INV-PERM-001  core_radius > 0
INV-PERM-002  core_value != 0
INV-PERM-003  max_radius >= core_radius
INV-PERM-004  samples >= 2
INV-PERM-005  Phi(r<=a) == Phi_0
INV-PERM-006  Phi(r>a) == Phi_0*a/r
INV-PERM-007  dPhi/dr == -Phi_0*a/r^2 for r>a
INV-PERM-008  classical Laplacian == 0 away from r=a
INV-PERM-009  shell boundary is treated distributionally
INV-PERM-010  k=0 Helmholtz continuation == static permeation
INV-PERM-011  no infinite lattice allocation
INV-PERM-012  perturbation recovery is explicit projection/relaxation
INV-PERM-013  static evaluation is not reported as causal propagation
INV-PERM-014  research field cannot silently become VM-authoritative state
```

## 13. Telemetry

The finite sampler emits:

```text
core_radius
max_radius
core_value
outer_value
outer_to_core_ratio
sampled_points
```

A higher-level adapter may additionally emit:

```text
epsilon_cutoff_radius
max_projection_error
mean_projection_error
relaxation_gain
relaxation_steps
wavenumber
sample_wall_time
allocated_bytes
```

Measured quantities must remain distinct from analytic properties.

## 14. Integration with the Dr Moagi runtime

The Permeation Protocol is an outward research operator complementary to ADR-002's inward codec
recursion:

```text
                     locked state Xi*
                           |
                +----------+----------+
                |                     |
          inward operator        outward operator
          I^3D / latent R        P_perm
                |                     |
          bounded recursion       Phi_0*a/r
                |                     |
          codec/verification      finite sampling
                +----------+----------+
                           |
                       Pi_Lambda
```

The two operators may share a locked anchor but do not acquire authority merely by visualization or
analytic evaluation.

## 15. Canonical status language

Permitted status:

```text
LOCKED CORE
ANALYTICALLY PERMEATING
FINITE DOMAIN SAMPLED
HARMONIC EXCITATION READY
```

Disallowed without a separately validated physical model:

```text
instantaneous action at a distance
perfect noise resistance
zero entropy of all external space
infinite memory allocation
100% physical coverage of R^3
all-space computation with zero cost
```

The research meaning remains strong: the core defines a globally supported mathematical field while
the implementation stays bounded, measurable and falsifiable.
