# 3D Dr Moagi Autoencoding & Decoding ANN Engine

## Status

Operational C++17 research runtime for the canonical Moagi-3D architecture.

The implementation emulates the hardware/software contract in portable C++; it does **not** claim that the present host CPU physically implements the future HBM/SRAM/tensor-core/crossbar silicon datapath.

## Closed operator

For input state `P^(n)` and K independent encoder/decoder channels,

```text
Z_k^(n) = E_k(P^(n))
R_k^(n) = D_k(Z_k^(n))
alpha^(n) = softmax(-tau * MSE(P^(n), R_k^(n)))
F^(n) = sum_k alpha_k^(n) R_k^(n)
P^(n+1) = (1-g) P^(n) + g F^(n)
```

where `g in (0,1]` is the recursive feedback gain and `tau > 0` is the FRM softmax temperature.

The Recursive Attractor Controller evaluates

```text
Delta_n = ||P^(n+1) - P^(n)||_2
```

and terminates on

```text
Delta_n <= epsilon
```

or on the hard safety bound

```text
n == N_max.
```

The practical fixed point is therefore

```text
P* ~= T(P*)
```

with `T` equal to the encode/decode/fuse/feedback operator above.

## Subsystem mapping

| Blueprint subsystem | Portable runtime realization |
| --- | --- |
| VIF — Volumetric Ingestion Fabric | `Tensor4D` input validation and contiguous voxel storage |
| K-channel Moagi Manifold | K deterministic `Autoencoder3D` instances with distinct seeds |
| FRM — Fusion & Routing Matrix | numerically stable softmax over per-channel reconstruction MSE, followed by weighted voxel fusion |
| RAC — Recursive Attractor Controller | L2 delta, immutable-anchor drift telemetry, epsilon halt, finite-state checks, and hard iteration ceiling |

The current K channels are independent deterministic feature transforms. Distinct seeds provide kernel diversity, but **strict latent orthogonality is not asserted by seed diversity alone**. A future training path should enforce the architectural orthogonality contract explicitly, for example with

```text
L_orth = sum_(i != j) |<Z_i,Z_j>|^2 / (||Z_i||^2 ||Z_j||^2 + delta).
```

## Runtime invariants

1. Same configuration, seed and input produce deterministic output.
2. Every FRM alpha is finite and non-negative.
3. `sum(alpha_k) = 1` up to floating-point tolerance.
4. Non-finite input, latent, reconstruction or recursive state fails closed.
5. RAC never executes more than `N_max` iterations.
6. The immutable input anchor is retained for drift measurement.
7. `epsilon` convergence and `N_max` exhaustion are reported as distinct termination modes.
8. K (parallel channels) is distinct from `latent_channels` (per-channel latent width).

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-moagi3d-engine jarvisx-moagi3d-engine-tests
ctest --test-dir build/cpp-runtime -R moagi3d-engine --output-on-failure
```

The executable is emitted as:

```text
DrMoagi-3D-Engine
```

Example:

```bash
./build/cpp-runtime/DrMoagi-3D-Engine \
  --edge 8 \
  --channels 3 \
  --latent-channels 4 \
  --max-iterations 32 \
  --epsilon 0.00001 \
  --feedback-gain 0.5 \
  --temperature 8 \
  --pattern sphere
```

The CLI emits per-iteration RAC/FRM telemetry including `delta_l2`, `anchor_l2`, channel MSE values, fusion weights, convergence state and termination cause.

## Hardware correspondence

A silicon realization may map the same operator onto tiled HBM/SRAM transport, parallel tensor-core clusters, a hardware softmax/crossbar FRM and a dedicated convergence state machine. The portable runtime is the executable architectural reference against which such an implementation can be tested.

The phrase “Platonic ideal” remains a design metaphor. Mathematically, `P*` is a fixed point/self-consistent attractor of the learned operator. External measurements, task losses or proof gates are required before treating that attractor as ground truth.
