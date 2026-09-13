# 1 MiB × 1 MiB × 1 MiB Volumetric ROM ANN

This runtime operationalizes the sparse virtual 3D ANN model as a C++17 executable in `cpp_runtime`.

## Geometry

Each axis is a 20-bit coordinate:

\[
N_x=N_y=N_z=2^{20}=1,048,576.
\]

The logical volume therefore contains

\[
2^{20}\cdot2^{20}\cdot2^{20}=2^{60}
\]

voxel addresses. At one byte per logical voxel this is **1 EiB of virtual capacity**. The runtime never allocates 1 EiB; it demand-materializes only active `32^3` tiles.

A voxel address is packed as

\[
A=(x\ll40)\;|\;(y\ll20)\;|\;z.
\]

## Geometric inward encoder

Every active tile is contracted through a true dyadic 3D pyramid:

```text
32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1
```

Each parent voxel is the mean of its eight children. Four adjacent-level residual energies, plus global mean/RMS/min/max, form the 8-D latent state.

## Recursive neural core

The fixed-point update is

\[
z_i^{(k+1)}=\tanh\left(0.55z_i^{(k)}+0.20z_{i+1}^{(k)}+0.15\omega_i+0.10\theta_i\right).
\]

Iteration stops at the configured relative tolerance or when the runtime step budget is exhausted.

## Decode, compare, learn, recur

The decoder expands the `4^3` coarse field over the active `32^3` tile and modulates it with the converged latent state. The residual field

\[
e=X-\hat X
\]

produces an MSE and a sparse high-error active set. Then:

\[
\Omega_{t+1}=\rho\Omega_t+(1-\rho)(Z_t^*+d(E_t)),
\]

\[
\Theta_{t+1}=\Theta_t-\eta E_tZ_t^*,
\]

while the runtime policy \(\Pi\) raises or lowers the fixed-point iteration budget according to reconstruction error.

The immutable ROM sequence is:

```text
RESOLVE
FETCH_ALLOC
ENCODE_PYRAMID
CONTRACT
FIXPOINT
DECODE
COMPARE
UPDATE_OMEGA
UPDATE_THETA
OPTIMIZE_RUNTIME
STORE
RECUR
```

## Build and run

From the repository root:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
ctest --test-dir build/cpp-runtime -C Release --output-on-failure
./build/cpp-runtime/DrMoagi-Volumetric-ROM-ANN --cycles 32 --active-tiles 64
```

On multi-config generators such as Visual Studio, the executable will be inside the `Release` configuration directory.

## Operational invariant

The architecture is deliberately sparse:

\[
|\mathcal A_t|\ll2^{60}.
\]

Logical capacity and physical memory are therefore separate. The machine executes only the active tiles, contracts them inward, stabilizes a latent representation, reconstructs outward, measures error, adapts \(\Omega\), \(\Theta\), and \(\Pi\), then recurs to the next 60-bit coordinate.
