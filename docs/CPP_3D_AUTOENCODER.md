# Dependency-Free C++ 3D Autoencoder Runtime

## Classification

This subsystem is a bounded, deterministic C++17 reference implementation of a trainable three-dimensional convolutional autoencoder. It is intended for small cubic fields, correctness experiments and inspectable runtime behavior.

It is not a production deep-learning framework, GPU implementation, general compressor or claim of lossless reconstruction.

## Geometry

The input is a single-channel cubic tensor

```text
X ∈ [-1,1]^(1×N×N×N)
```

with an even edge `N` in `[4,64]`. The encoder produces

```text
Z ∈ [-1,1]^(C×N/2×N/2×N/2)
```

where `C` is the number of latent channels.

Both transforms use shared `3×3×3` kernels and periodic boundary conditions. The encoder has stride two. The decoder maps every output coordinate to its corresponding latent coordinate and applies a shared multichannel neighborhood kernel.

## Forward equations

For latent channel `c` and latent coordinate `p`:

```text
Z[c,p] = tanh(b_e[c] + Σ_k W_e[c,k] X[2p+k-1])
```

For output coordinate `q`:

```text
X_hat[q] = tanh(b_d + Σ_c Σ_k W_d[c,k] Z[floor(q/2)+k-1])
```

All coordinate arithmetic wraps periodically.

Optional final inference quantizes every latent activation into the signed three-bit domain

```text
Q3 = {-4,-3,-2,-1,0,1,2,3}
```

and immediately dequantizes it back to `[-1,1]` for decoding. Training uses continuous latent activations; no false differentiability claim is made for the discrete quantizer.

## Learning

The runtime minimizes mean squared reconstruction error with direct reverse-mode differentiation through both kernels and `tanh` activations. Updates use:

- deterministic Xavier-like initialization;
- single-sample stochastic gradient descent;
- elementwise gradient clipping;
- optional L2 weight penalty;
- finite and bounded input contracts.

The implementation records MSE, MAE, maximum absolute error, latent energy and gradient L2 norm for every step.

## Build and test

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel
ctest --test-dir build/cpp-runtime --output-on-failure
```

## Run

```bash
./build/cpp-runtime/jarvisx-autoencoder3d \
  --edge 8 \
  --channels 4 \
  --epochs 250 \
  --pattern sphere \
  --quantized \
  --export-dir .jarvisx-autoencoder3d
```

Supported synthetic fixtures are `sphere`, `shell`, `checker`, `wave` and deterministic `noise`.

The output directory contains:

```text
metrics.csv          per-step training telemetry
input.obj            input voxel point cloud
latent.obj           channel-stacked latent point cloud
reconstruction.obj   reconstructed voxel point cloud
model.jx3d            reloadable text model checkpoint
```

OBJ files contain point-cloud vertices rather than watertight meshes. They can be inspected in Blender, MeshLab or another 3D viewer.

## Persistence and replay

The checkpoint records dimensions, optimizer hyperparameters, seed, step count, encoder weights, decoder weights and biases using round-trip-safe decimal float precision. Loading validates the format and tensor sizes before accepting the model.

For a fixed compiler/platform, seed, input and training sequence, execution is deterministic. Cross-platform bit identity is not claimed because standard floating-point and transcendental implementations may differ slightly.

## Complexity boundary

For input edge `N`, latent channels `C` and kernel volume `K=27`:

```text
encoder forward:  O(C K N^3 / 8)
decoder forward:  O(C K N^3)
training step:     O(C K N^3)
model parameters: 2 C K + C + 1
```

The tensors are dense. The maximum accepted edge is deliberately bounded to prevent accidental unbounded allocation.
