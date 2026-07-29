# Hierarchical 3D Fractional Geometric Smoothing

## Scope

This module operationalises a deterministic reference model for three-dimensional
fractional diffusion, multiscale gradient smoothing, and equilibrium search.
It works on a periodic scalar lattice and is designed as a correctness model for
future FFT, sparse-octree, GPU, and latent-diffusion implementations.

The word **fractional** refers to a non-integer power of the discrete Laplacian.
It does not mean that voxel values are merely divided by 1000.

## 1. Three-dimensional field memory

A field is stored as

\[
u[x,y,z]\in\mathbb R,
\qquad
0\le x<N_x,\;0\le y<N_y,\;0\le z<N_z.
\]

The flat cell address is

\[
\boxed{a=x+N_x(y+N_yz)}.
\]

Every cell is therefore an explicit 3D state location. The current reference
implementation stores one scalar per voxel, but the same mechanics can be
applied independently to channels such as density, temperature, RGB, latent
features, error, or an Ω correction field.

## 2. Local 3D Laplacian

For a six-neighbour periodic lattice, the positive discrete Laplacian is

\[
(Lu)_{x,y,z}
=
6u_{x,y,z}
-u_{x+1,y,z}-u_{x-1,y,z}
-u_{x,y+1,z}-u_{x,y-1,z}
-u_{x,y,z+1}-u_{x,y,z-1}.
\]

Classical diffusion is

\[
\frac{\partial u}{\partial t}=-D Lu.
\]

Only the immediate six neighbours appear in the stencil, but repeated updates
propagate information through the full volume.

## 3. Fractional 3D operator

The fractional model replaces `L` with

\[
L^\alpha,
\qquad 0<\alpha\le 1.
\]

The evolution equation is

\[
\boxed{
\frac{\partial u}{\partial t}
=-D L^\alpha u+\Omega.
}
\]

- `alpha = 1` gives classical lattice diffusion.
- `alpha < 1` changes the spectral attenuation law and produces a nonlocal
  operator in physical space.
- `D` controls diffusion strength.
- `Omega` is an optional persistent correction or forcing field.

## 4. Arithmetic spectral execution

The periodic discrete Fourier transform converts each spatial mode into an
independent scalar update.

### Forward transform

\[
\hat u_{p,q,r}
=
\sum_{z=0}^{N_z-1}
\sum_{y=0}^{N_y-1}
\sum_{x=0}^{N_x-1}
u_{x,y,z}
\exp\left[-2\pi i\left(
\frac{px}{N_x}+\frac{qy}{N_y}+\frac{rz}{N_z}
\right)\right].
\]

### Laplacian eigenvalue

For mode `(p,q,r)`,

\[
\boxed{
\lambda_{p,q,r}
=
6
-2\cos\frac{2\pi p}{N_x}
-2\cos\frac{2\pi q}{N_y}
-2\cos\frac{2\pi r}{N_z}.
}
\]

Therefore

\[
\widehat{L^\alpha u}_{p,q,r}
=
\lambda_{p,q,r}^{\alpha}\hat u_{p,q,r}.
\]

### Exact unforced diffusion step

For a timestep `tau`,

\[
\boxed{
\hat u'_{p,q,r}
=
\exp\left(-\tau D\lambda_{p,q,r}^{\alpha}\right)
\hat u_{p,q,r}.
}
\]

The zero-frequency eigenvalue is zero, so

\[
\hat u'_{0,0,0}=\hat u_{0,0,0}.
\]

Consequently the total mass is preserved in the unforced system.

### Exact step with constant Ω forcing

Let

\[
a=D\lambda^\alpha.
\]

For `a > 0`,

\[
\boxed{
\hat u'
=e^{-a\tau}\hat u
+\frac{1-e^{-a\tau}}{a}\hat\Omega.
}
\]

For the constant mode `a = 0`,

\[
\hat u'_{0}=\hat u_0+\tau\hat\Omega_0.
\]

By default the implementation removes the mean of Ω, so Ω reshapes the field
without changing its total mass.

## 5. Hierarchical 3D geometry

The fine field is level zero:

\[
G_0\in\mathbb R^{N_x\times N_y\times N_z}.
\]

A 2×2×2 restriction operator creates the next level:

\[
\boxed{
G_{k+1}[i,j,l]
=
\frac{1}{8}
\sum_{d_x,d_y,d_z\in\{0,1\}}
G_k[2i+d_x,2j+d_y,2l+d_z].
}
\]

Each restriction divides the voxel count by eight. A hierarchy with three
levels has the form

```text
fine:   Nx × Ny × Nz
middle: Nx/2 × Ny/2 × Nz/2
coarse: Nx/4 × Ny/4 × Nz/4
```

Each level receives its own fractional order and smoothing time:

\[
G'_k
=
\exp(-\tau_kD L_k^{\alpha_k})G_k.
\]

A typical fine-to-coarse schedule is

```text
level 0: alpha = 1.00, tau = 0.05   local detail
level 1: alpha = 0.75, tau = 0.10   regional structure
level 2: alpha = 0.50, tau = 0.20   global structure
```

## 6. Coarse-to-fine reconstruction

Constant prolongation copies each coarse voxel into a 2×2×2 block:

\[
(PG)[2i+d_x,2j+d_y,2l+d_z]=G[i,j,l].
\]

It satisfies

\[
\boxed{R(P(G))=G}
\]

up to floating-point roundoff.

This is the operational meaning of the earlier reciprocal loop:

```text
1 coarse state -> 8 fine cells -> 1 reconstructed coarse state
```

or, at a larger symbolic scale,

```text
1/1000 compression <-> 1000/1 expansion.
```

The absolute scale changes, but normalising the complete round trip by itself
returns the identity on the coarse subspace.

At each fine level, the locally smoothed state is fused with the prolonged
coarse state:

\[
\boxed{
V_k=(1-\gamma_k)G'_k+\gamma_kP(V_{k+1}),
\qquad 0\le\gamma_k\le 1.
}
\]

`gamma` determines how strongly the global coarse equilibrium influences the
local fine geometry.

## 7. Complete mechanistic loop

One transaction executes:

```text
LOAD_FINE_3D
-> RESTRICT_2X2X2 until the coarsest level
-> DFT3 each level
-> compute lambda(p,q,r)
-> multiply each mode by exp(-tau * D * lambda^alpha)
-> inject exact Ω forcing when present
-> inverse DFT3
-> calculate variance, gradient energy, and equilibrium residual
-> PROLONG_2X2X2 from coarse to fine
-> FUSE_COARSE_FINE using gamma
-> VERIFY mass drift and update magnitude
-> COMMIT result
```

The Python result includes an explicit instruction trace with opcodes:

```text
RESTRICT_2X2X2
FRACTIONAL_HEAT_3D
PROLONG_2X2X2
FUSE_COARSE_FINE
VERIFY_MASS_AND_UPDATE
```

## 8. Gradient smoothing

For a scalar energy functional `E(u)`, ordinary gradient flow is

\[
\frac{\partial u}{\partial t}
=-\nabla_uE.
\]

A fractional preconditioned gradient flow can be written as

\[
\boxed{
\frac{\partial u}{\partial t}
=-L^\alpha\nabla_uE.
}
\]

Alternatively, when `u` itself is the quantity being regularised,

\[
E_\alpha(u)=\frac{1}{2}\langle u,L^\alpha u\rangle
\]

has gradient

\[
\nabla_uE_\alpha=L^\alpha u,
\]

and its gradient flow is exactly fractional diffusion:

\[
\frac{\partial u}{\partial t}=-L^\alpha u.
\]

The hierarchy acts as a multiscale preconditioner:

- the coarse levels remove global low-frequency imbalance;
- the fine levels remove local high-frequency irregularity;
- the fusion step coordinates the scales.

## 9. Equilibrium arithmetic

For the unforced system, equilibrium satisfies

\[
\boxed{L^\alpha u^*=0.}
\]

On a connected periodic lattice, the only scalar equilibria are spatially
constant fields.

For a zero-mean forcing field,

\[
\boxed{DL^\alpha u^*=\Omega.}
\]

The implementation measures the root-mean-square residual

\[
r(u)
=
\sqrt{
\frac{1}{N}
\sum_a
\left[-D(L^\alpha u)_a+\Omega_a\right]^2
}.
\]

Repeated cycles terminate when

\[
\sqrt{
\frac{1}{N}
\sum_a(u^{(t+1)}_a-u^{(t)}_a)^2
}
\le\varepsilon.
\]

This is numerical convergence, not a universal proof that every nonlinear
extension will have a unique equilibrium.

## 10. Applications

### Latent diffusion and score fields

A denoising score tensor can be smoothed spatially before the reverse update:

\[
\tilde s_\theta
=
\sum_k w_kP_kL_k^{\alpha_k}R_k s_\theta.
\]

This can suppress local score noise while retaining long-range structure.
The method must be validated carefully because excessive smoothing can bias the
learned score and reduce detail.

### Geometry and signed-distance fields

Fractional smoothing can regularise voxel occupancy, density, SDFs, point-cloud
fields, or mesh-derived scalar quantities while allowing nonlocal propagation.

### Physical equilibrium systems

The solver can model anomalous or nonlocal transport, temperature-like fields,
chemical concentration, or correction-driven steady states where distant
regions interact through a fractional operator.

### Optimization

The hierarchy can be used as a gradient preconditioner. Coarse levels correct
large-scale error modes that local descent removes slowly, while fine levels
retain local resolution.

## 11. Reference API

```python
from jarvisx.fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    hierarchical_fractional_smooth,
    run_to_equilibrium,
)

field = Grid3D.impulse((8, 8, 8), (4, 4, 4), amplitude=1.0)
config = FractionalHierarchyConfig(
    alphas=(1.0, 0.75, 0.50),
    taus=(0.05, 0.10, 0.20),
    coarse_blends=(0.20, 0.35),
)

single_cycle = hierarchical_fractional_smooth(field, config)
equilibrium = run_to_equilibrium(
    field,
    config,
    tolerance=1.0e-6,
    max_cycles=100,
)
```

## 12. Engineering boundaries

The current implementation intentionally uses a direct separable DFT and dense
Python arrays. It is transparent and deterministic but not intended for large
volumes. Production evolution should proceed in this order:

1. validate the equations and invariants on small dense fields;
2. replace the direct DFT with an FFT backend;
3. introduce sparse bricks or an octree hierarchy;
4. add vector-valued and batched tensors;
5. port the spectral kernels to GPU;
6. couple Ω to measured residuals or learned correction fields;
7. benchmark convergence, energy, memory, and runtime against classical
   diffusion, multigrid, and established fractional PDE solvers.
