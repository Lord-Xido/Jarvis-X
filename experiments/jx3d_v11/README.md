# Jarvis-X v11 — 3D Neural Encoding Reference Kernel

**Scope:** an isolated, readable, compilable C++20 integration slice from the locally verified `JarvisX_3D_Operational_v11` prototype. This pull request deliberately does **not** overwrite existing Jarvis-X native, cloud, or dashboard modules.

## Run the real CPU reference

From the repository root:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  experiments/jx3d_v11/tests/reference.cpp -o /tmp/jx3d_v11_reference
/tmp/jx3d_v11_reference
```

The test uses an actual 10×10×10 grid, true adjacent **2×2×2** spatial tiles, and a shared 76-parameter 8→4→8 trainable neural transform. Every test has a deterministic seed; the independent breadth-first-search implementation is the pathfinding oracle for 600 random cases.

### Mathematics

For voxel `i = x + D(y + Dz)` with healthy six-face-neighbors:

```text
psi_next[i] = clamp(0.78*psi[i] + 0.18*mean(face_neighbors) + 0.04*source[i], 0, 1)
z = tanh(W_enc*x_tile + b_enc)          # 8 -> 4
z_q = round(127.5 * (z + 1))            # four 8-bit latent symbols
x_hat = sigmoid(W_dec*(z_q/127.5 - 1) + b_dec)
residual = x_Q8 XOR predicted_Q8
decoded_Q8 = predicted_Q8 XOR residual  # exactly source Q8
```

For fixed topology and identical sources, the lattice map has an `L∞` Lipschitz bound `≤0.96`; injected faults or changing sources are outside that bound. Exact reconstruction is for **Q8 bytes**, not original continuous float values. Retaining explicit residual bytes can increase file size.

## Included in this PR

- `src/jx_ann.hpp`: the exact deterministic 76-parameter shared ANN header from the local v11 release.
- `tests/reference.cpp`: executable 3D tiling/inverse, training-loss decrease, latent-plus-XOR roundtrip, eight bitplanes, contractivity and 600 A*/BFS comparisons.
- `.github/workflows/jx3d-v11-native.yml`: isolated CPU compile and regression gate.

The checked-in ANN header has Git blob SHA `3899d6bae481810142111415988d4780b0a3e701`, matching the locally tested v11 source.

## Explicitly *not* integrated here

The full standalone v11 distribution also contains a larger C++ core, snapshots with CRC32, a local Python HTTP service, a browser UI and further integration tests. Those components remain in the separately supplied v11 ZIP; this PR does **not** claim that it deploys that complete application. Hardware electron transport, GPU/optical acceleration, distributed cloud execution, a trillion-parameter ANN, and production security are not implemented by this reference. Introduce the remaining subsystems only with source-visible commits and CI gates.
