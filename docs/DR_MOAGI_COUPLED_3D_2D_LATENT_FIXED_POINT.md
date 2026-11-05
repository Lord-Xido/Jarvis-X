# Coupled 3D -> 2D -> Latent Fixed-Point Runtime

**Status:** executable bounded reference profile  
**Implementation:** `src/jarvisx/dr_moagi_coupled_fixed_point.py`  
**Tests:** `tests/test_dr_moagi_coupled_fixed_point.py`  
**Architectural layer:** Layer 5 research system over Layer 4 geometry

## 1. Purpose

This profile operationalizes a recursive map whose state is evaluated across three coupled spaces:

1. a continuous 3D signed-distance geometry;
2. a differentiable 2D projection surrogate;
3. a latent feature/attention space updated by a bounded second-order step.

The convergence certificate is the idempotent residual

```math
r_\Phi(z) = \|\Phi(\Phi(z)) - \Phi(z)\|_2.
```

A small residual is evidence that one further application of the same update changes the candidate little. It is not, by itself, a proof of a unique global fixed point.

## 2. Hairpin geometry

For the default copper-wire profile:

```text
total centerline length = 1000 mm
diameter                = 1.0 mm
inner gap               = 2.0 mm
wire radius             = 0.5 mm
centerline separation   = inner_gap + diameter = 3.0 mm
U-bend centerline R     = 1.5 mm
U-bend curvature        = 1/R = 2/3 mm^-1
torsion                 = 0 on the planar reference centerline
```

The straight-leg length is therefore

```math
L_{leg} = \frac{1000 - \pi R}{2}.
```

The centerline is parameterized by arc length with two straight segments and a semicircle. The wire SDF is

```math
f(x) = d(x, C) - r_w,
```

where `C` is the centerline and `r_w = 0.5 mm`. Hence:

```math
f(x) < 0 \Rightarrow x \text{ is inside the solid},\qquad
f(x) = 0 \Rightarrow x \in \partial S,\qquad
f(x) > 0 \Rightarrow x \text{ is outside}.
```

## 3. Eikonal contract

For an exact Euclidean distance field,

```math
\|\nabla f(x)\|_2 = 1
```

where the distance function is differentiable. The qualifier matters: SDFs are generally non-differentiable on medial-axis/seam loci. The implementation therefore exposes a sampled `eikonal_residual` audit rather than claiming the equation holds classically at every point.

Surface normals are

```math
n(x) = \frac{\nabla f(x)}{\|\nabla f(x)\|_2}
```

when the gradient norm is non-zero.

## 4. Hard solid versus differentiable renderer

The physical solid uses a hard occupancy:

```math
\sigma_{hard}(x) =
\begin{cases}
\sigma_{Cu}, & f(x) \le 0,\\
0, & f(x) > 0.
\end{cases}
```

This step is discontinuous at `f=0`, so it cannot also provide an everywhere-defined exact pixel-to-geometry Jacobian.

For optimization only, the renderer uses the narrow smooth surrogate

```math
\sigma_{soft}(x)
= \sigma_{Cu}\,\left(1 + e^{f(x)/\beta}\right)^{-1},
```

with configurable transition width `beta`. The authoritative geometry remains the hard SDF.

## 5. 2D projection

The dependency-free reference renderer uses orthographic rays through the bend crop and fixed quadrature:

```math
\tau(u,v) \approx \sum_i \sigma_{soft}(r_i)\,\Delta z,
```

```math
I(u,v) = 1 - e^{-\alpha\tau(u,v)}.
```

This is a differentiable silhouette/opacity reference, not a production copper BRDF. A production backend may replace it with Fresnel + microfacet shading provided the same state boundaries and validation gates are preserved.

## 6. Visual tokens and attention

The reference image is partitioned into patches. Each patch is represented by a compact token

```math
p_i = [\mu_i,\operatorname{Var}_i,E_i].
```

A bounded single-head reference attention layer uses `Q=K=V=P`:

```math
A(P) = \operatorname{softmax}\left(\frac{PP^T}{\sqrt d}\right)P.
```

This preserves the algebraic attention mechanism without requiring a heavyweight Transformer dependency. Learned projection matrices can be supplied by a future backend.

## 7. Coupled objective

The executable loss is

```math
\mathcal L_{total}
= \lambda_f \|F(I(z)) - F(I^*)\|_2^2
+ \lambda_p\left[(d-1)^2 + (g-2)^2\right],
```

where `z=(d,g)` contains wire diameter and inner gap, and `F` is the rendered-patch attention feature map.

The implementation keeps Eikonal auditing separate from the default optimization term because the analytic centerline-distance construction is already an SDF away from non-smooth loci. Neural SDF backends should add an explicit sampled Eikonal penalty.

## 8. Second-order update

The unguarded Newton step

```math
z_{k+1}=z_k-H^{-1}\nabla\mathcal L
```

is numerically fragile when the Hessian is singular or indefinite. The reference implementation uses a two-dimensional finite-difference Hessian, positive diagonal damping, and a trust radius:

```math
(H+\lambda I)\,\Delta z=-\nabla\mathcal L,
```

```math
z_{k+1}=z_k+\operatorname{clip}_{\rho}(\Delta z).
```

The implementation now computes the gradient and Hessian from one shared central-difference stencil. In the two-parameter reference state this requires nine unique objective evaluations rather than independently re-rendering overlapping gradient and Hessian states. Each optimization transaction memoizes deterministic loss evaluations.

The proposed Newton direction is accepted only when it is monotonic:

```math
\mathcal L(z_k + \alpha\Delta z) \le \mathcal L(z_k),
\qquad
\alpha\in\{1,\beta,\beta^2,\ldots\},\quad 0<\beta<1.
```

If necessary, a bounded backtracking search reduces the step. The multi-iteration `optimize_coupled` driver then adapts damping and trust radius: successful updates reduce damping and expand the local trust region; stalled updates increase damping and contract the trust region.

This is the executable `Phi` used by `coupled_iteration`.

## 9. Auto-optimization and stopping rule

The optimizer requires both local fixed-point stability and sufficiently small external objective error:

```math
\|z_{k+1}-z_k\| \le \varepsilon_z,
\qquad
r_\Phi(z_k) \le \varepsilon_\Phi,
\qquad
\mathcal L_{total}(z_{k+1}) \le \varepsilon_L.
```

This prevents a stationary but incorrect geometry from being reported as converged merely because its update has collapsed to zero.

## 10. Execution graph

```text
z_k = (diameter, gap)
  -> construct hairpin centerline
  -> evaluate hard SDF
  -> smooth occupancy surrogate
  -> orthographic volume quadrature
  -> 2D bend crop
  -> patch tokenization
  -> self-attention
  -> coupled feature + physical loss
  -> memoized shared gradient/Hessian stencil
  -> damped trust-region Newton direction
  -> monotonic backtracking acceptance
  -> z_(k+1)
  -> evaluate ||Phi(Phi(z_k)) - Phi(z_k)||
  -> adapt damping/trust radius
  -> require internal + external convergence
  -> recur or stop
```

## 11. Validation boundary

The profile deliberately does not claim:

- exact BRDF realism;
- an exact analytic Jacobian through the hard occupancy step;
- global convexity of the coupled objective;
- uniqueness of the fixed point;
- that sampled Eikonal residuals prove a globally smooth SDF;
- production-scale Transformer behavior.

Those properties require separate mathematical assumptions, backend implementations, and empirical tests.