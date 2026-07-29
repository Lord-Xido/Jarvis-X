# Hierarchical 3D Fractional Smoothing

## Classification

This subsystem is a deterministic, dependency-free **numerical reference kernel** for small periodic scalar grids. It is intended for equation-level verification, fixtures and auditable experiments.

It is not a production FFT, sparse-volume solver, calibrated physical model or performance claim for large three-dimensional fields.

## 1. State and addressing

A scalar grid has shape

```text
(Nx, Ny, Nz)
```

and immutable values stored with `x` as the fastest-moving coordinate:

```text
a(x,y,z) = x + Nx (y + Ny z)
```

All dimensions are positive. Every stored value must be finite.

The current reference implementation materializes the complete grid and complete complex spectrum. Therefore resident storage is proportional to

```text
O(Nx Ny Nz)
```

for each field or spectrum, despite any broader virtual-space research context in Jarvis-X.

## 2. Periodic six-neighbour operator

For a periodic lattice, define

```text
(-Delta u)[x,y,z]
  = 6 u[x,y,z]
    - u[x-1,y,z] - u[x+1,y,z]
    - u[x,y-1,z] - u[x,y+1,z]
    - u[x,y,z-1] - u[x,y,z+1]
```

with every coordinate wrapped modulo its axis length.

The corresponding Fourier eigenvalue is

```text
lambda(kx,ky,kz)
  = 6
    - 2 cos(2 pi kx / Nx)
    - 2 cos(2 pi ky / Ny)
    - 2 cos(2 pi kz / Nz)
```

where

```text
0 <= lambda <= 12.
```

The zero mode has `lambda = 0`, so unforced diffusion preserves the field mean and total mass.

## 3. Fractional operator

For

```text
0 < alpha <= 1,
```

the fractional operator is defined spectrally:

```text
F[(-Delta)^alpha u](k)
  = lambda(k)^alpha F[u](k).
```

At `alpha = 1`, this is exactly the periodic six-neighbour stencil above, up to floating-point roundoff.

The implementation uses a separable direct DFT rather than an FFT dependency:

```text
DFT_x -> DFT_y -> DFT_z
```

and the inverse applies one `1/N` normalization factor per axis.

For a cubic `N × N × N` grid, the separable arithmetic cost is

```text
O(N^4),
```

not the `O(N^3 log N)` cost expected from a production 3D FFT.

## 4. Exact mode update

The governing equation is

```text
du/dt = -D (-Delta)^alpha u + Omega,
```

where:

- `D >= 0` is diffusivity;
- `Omega` is constant over one timestep;
- `tau >= 0` is the timestep.

For one mode with

```text
r(k) = D lambda(k)^alpha,
```

the exact update is

```text
u_hat(t + tau)
  = exp(-tau r) u_hat(t)
    + (1 - exp(-tau r)) Omega_hat / r,  r > 0
```

and for the zero-rate mode:

```text
u_hat(t + tau)
  = u_hat(t) + tau Omega_hat.
```

When `zero_mean_omega=True`, the forcing mean is removed before transformation. This prevents forcing from changing total mass, up to floating-point roundoff.

Special cases are explicit:

- `tau = 0` returns the authoritative input field unchanged;
- `D = 0` performs the exact physical-space update `u + tau Omega`;
- no forcing and a constant field produce equilibrium.

## 5. Multiresolution hierarchy

Restriction averages every `2 × 2 × 2` block:

```text
R(u)[i,j,k]
  = (1/8) sum u[2i+dx, 2j+dy, 2k+dz].
```

Constant prolongation replicates one coarse value into eight fine cells:

```text
P(v)[2i+dx, 2j+dy, 2k+dz] = v[i,j,k].
```

Therefore

```text
R(P(v)) = v
```

on the coarse subspace, up to floating-point roundoff.

A hierarchy with `L` levels requires every original dimension to be divisible by

```text
2^(L-1).
```

Each level receives its own fractional order `alpha_l` and smoothing time `tau_l`.

After local smoothing, coarse information returns to the next finer level through

```text
u_fused
  = (1 - beta_l) u_local + beta_l P(u_coarse),
```

where

```text
0 <= beta_l <= 1.
```

Because local unforced steps preserve mean, restriction preserves mean, prolongation restores the corresponding fine-grid mass and blending is affine, the complete unforced hierarchy preserves total mass up to numerical roundoff.

## 6. Mechanistic trace

One hierarchy transaction emits an ordered trace containing:

```text
RESTRICT_2X2X2
FRACTIONAL_HEAT_3D
PROLONG_2X2X2
FUSE_COARSE_FINE
VERIFY_MASS_AND_UPDATE
```

Each instruction records its sequence number, hierarchy level and numerical detail. The trace is explanatory telemetry, not executable Jarvis-X bytecode.

## 7. Metrics

The solver exposes:

- field mean and mass;
- variance;
- classical periodic gradient energy;
- fractional equilibrium residual;
- hierarchy mass drift;
- update RMS;
- per-level before/after traces;
- repeated-cycle update and residual histories.

Classical gradient energy is

```text
G(u) = mean((dx u)^2 + (dy u)^2 + (dz u)^2) / 3.
```

The equilibrium residual is

```text
R(u) = RMS(-D (-Delta)^alpha u + Omega).
```

`run_to_equilibrium` stops when update RMS is less than or equal to the requested positive tolerance, or reports non-convergence after `max_cycles`.

## 8. Independent validation

Tests do not rely only on internal consistency. They compare:

1. the separable transform against the literal direct 3D DFT definition;
2. `alpha = 1` against the spatial periodic stencil;
3. split timesteps against the exact semigroup identity;
4. coarse prolongation/restriction against the exact coarse identity;
5. mass and gradient behavior across hierarchy transactions;
6. malformed geometry, spectra, parameters and convergence bounds.

## 9. Production path

A production solver would require separate engineering decisions:

- NumPy, FFTW, MKL, cuFFT or another verified FFT backend;
- real-to-complex symmetry and plan reuse;
- sparse bricks, octrees or domain decomposition;
- boundary conditions beyond periodicity;
- vector and tensor fields;
- calibrated units and physical constitutive models;
- precision and stability studies;
- benchmark comparison with established fractional PDE and multigrid solvers;
- CPU/GPU memory, throughput and scaling measurements.

Those capabilities are not implied by the reference implementation.
