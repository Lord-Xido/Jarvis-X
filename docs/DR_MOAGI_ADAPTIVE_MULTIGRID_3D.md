# Dr Moagi Adaptive Multigrid 3D Runtime

## Purpose

This runtime operationalises the submitted self-looping 3D mechanics as a mathematically consistent geometric multigrid solver with bounded operator adaptation and candidate-first verification.

It is not a neural autoencoder. The inward/outward loop is a multiresolution PDE correction process:

```text
fine solution
 -> pre-smooth
 -> residual
 -> restrict residual
 -> recursive coarse solve
 -> prolong correction
 -> post-smooth
 -> verify
 -> optional permeability proposal
 -> COMMIT / ROLLBACK
 -> recur
```

## Geometry

The default root is a periodic `128^3` lattice with the hierarchy

```text
128^3 -> 64^3 -> 32^3 -> 16^3 -> 8^3 -> 4^3.
```

Power-of-two periodic geometry is used deliberately: every level has an exact factor-two spatial relationship and the final odd-index prolongation samples halfway between adjacent coarse nodes, including across the periodic boundary.

## Operator

For voxel `i` and its six face-neighbours `j`, define face permeability

\[
w_{ij}=\frac{k_i+k_j}{2}.
\]

The discrete operator is

\[
(A_k u)_i
=
\alpha u_i
+
\sum_{j\in N(i)} w_{ij}(u_i-u_j),
\qquad \alpha=0.1.
\]

The positive reaction term removes the constant-field nullspace of a purely periodic diffusion operator.

Writing

\[
d_i=\alpha+\sum_j w_{ij},
\]

the weighted-Jacobi smoother solves the same equation measured by the residual:

\[
u_i^{J}
=
\frac{b_i+\sum_j w_{ij}u_j}{d_i},
\]

\[
u_i^{new}
=
(1-\omega)u_i+\omega u_i^J,
\qquad \omega=0.72.
\]

The residual is

\[
r=b-A_k u.
\]

This is the central correction relative to the submitted sketch: the smoother consumes the right-hand side that the residual and restriction stages produce.

## Manufactured truth and forcing

The implementation keeps the reference solution `truth` separate from the PDE right-hand side `rhs`.

A smooth periodic manufactured truth is generated from several Fourier modes. The authoritative fine-grid forcing is then constructed as

\[
b=A_k u^\star.
\]

Therefore `u*` is an exact discrete solution for the current permeability field. Verification can independently measure both

\[
\rho=\frac{\|b-A_k u\|_2}{\max(\|b\|_2,\epsilon)}
\]

and

\[
\mathrm{MSE}(u,u^\star).
\]

## Restriction

The coarse right-hand side is the restricted fine-grid residual, not a pre-seeded target.

Full weighting uses

\[
R=\frac{1}{64}[1\;2\;1]^{\otimes3}.
\]

So

\[
b_{\ell+1}=R\,r_\ell,
\qquad
u_{\ell+1}=0.
\]

The recursive coarse level then actually solves this correction equation.

## Prolongation

The coarse correction is returned with true factor-two trilinear interpolation.

For odd fine coordinates the interpolation coordinate is `0.5`, not `1`. Thus the eight-corner weights are products of

\[
(1-t_d),t_d,\quad (1-t_h),t_h,\quad (1-t_w),t_w
\]

with each `t` equal to either `0` or `0.5`.

The correction is accumulated:

\[
u_\ell \leftarrow u_\ell + P u_{\ell+1}.
\]

## V-cycle

One recursive V-cycle is

\[
\mathcal V_\ell
=
S_\ell^{post}
\circ
P_\ell
\circ
\mathcal V_{\ell+1}
\circ
R_\ell
\circ
\mathcal R_\ell
\circ
S_\ell^{pre}.
\]

At the coarsest level the system performs additional Jacobi sweeps instead of recursing further.

This is the operational meaning of the inward loop: low-frequency fine-grid error is mapped into a smaller spatial state where it is cheaper to reduce, then projected outward as a correction.

## Permeability adaptation

The outer loop proposes a new bounded permeability field from the current spatial error. Regions with larger normalised error are proposed higher permeability, while all values remain in

\[
0.05\le k_i\le1.
\]

The proposal is not authoritative immediately.

For each candidate permeability field:

1. propagate `k` down the hierarchy;
2. rebuild the fine-grid forcing so the immutable manufactured truth remains the exact target;
3. solve with V-cycles;
4. measure relative residual and truth MSE;
5. commit only if both remain within the non-regression gate;
6. otherwise restore the previous `u`, `k`, and `rhs` and reduce the adaptation rate.

This implements the repository transaction grammar:

```text
GENERATE -> CONTRAST -> RECKON -> VERIFY -> COMMIT / ROLLBACK -> RECUR
```

## Validation

The executable contains a deterministic self-test that verifies:

- exact operator/forcing consistency for the manufactured truth;
- constant preservation by full-weighting restriction;
- constant preservation by trilinear prolongation;
- one V-cycle reduces the relative residual from the initial state;
- the adaptive outer loop returns finite metrics without regressing the accepted residual.

Run:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-adaptive-multigrid-3d --parallel
./build/cpp-runtime/DrMoagi-Adaptive-Multigrid-3D --self-test
```

A bounded operational run:

```bash
./build/cpp-runtime/DrMoagi-Adaptive-Multigrid-3D \
  --dim 64 \
  --outer 2 \
  --cycles 12 \
  --tol 1e-4
```

Local strict GCC validation before repository integration used C++17, OpenMP, `-O3`, `-Wall`, `-Wextra`, `-Wpedantic`, `-Wconversion`, and `-Wshadow`. A `64^3` bounded run reached a relative residual of approximately `4.6e-5` and truth MSE approximately `3.5e-10`. These are validation receipts for that run, not portable performance guarantees.

## Parallelism

OpenMP is optional. CMake links it when available and defines `JARVISX_HAVE_OPENMP`; otherwise the same source remains a serial C++17 implementation.

Parallelism changes wall-clock execution only. The numerical acceptance gate depends on residual/MSE evidence, not elapsed time.

## FLOP telemetry

The runtime tracks approximate arithmetic work for the core stencil/restriction/prolongation kernels. It deliberately labels this as an approximate kernel count.

It does not represent memory traffic, OpenMP overhead, norm calculations, adaptation work, or a hardware-independent benchmark. No ZFLOP/TFLOP claim is made without measured evidence.

## Capability boundary

This is a bounded numerical reference solver. It does not establish that all recursive systems converge, does not prove optimal multigrid complexity for arbitrary variable-coefficient PDEs, does not train neural weights, and does not turn geometric recursion into intelligence by itself.

The important invariant is narrower and testable: every V-cycle solves the same discrete `A_k u=b` equation that its residual measures, and every operator adaptation remains provisional until verification.
