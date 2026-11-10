# JARVIS-X: 3D Permeation Swarm (experimental CPU reference)

An isolated, runnable C++17/OpenMP adaptation of the user-supplied 3D MegaSwarm and PermeationVAE design. This is **research demonstration code**, not production AI, a trained VAE, a GPU implementation, a 3D codec, or an intellectual-property enforcement mechanism.

## Build and run

```sh
cmake -S experimental/permeation_swarm_v1 -B /tmp/jx_swarm_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx_swarm_build -j
ctest --test-dir /tmp/jx_swarm_build --output-on-failure
/tmp/jx_swarm_build/jx3d_permeation_swarm --swarm 2 --steps 5 --threads 4 --seed 42
```

The user's full **512 node × 150 step** configuration is opt-in via `--swarm 8 --steps 150`. It is compute-intensive and not a cloud-scale experiment. Default is eight nodes and five steps to support deterministic local checks.

## End-to-end stages

1. 16³ single-channel volume per node, with per-node random seeded input.
2. Individually initialized 3D 3×3×3 kernels: stride-two `16³→8³×16→4³×32`.
3. Independent linear maps from 2,048 features to 16 latent means and 16 log-variances; clamp logvar to `[-8,4]`; reparameterize with seeded Gaussian noise and optional drift.
4. Project the latent back to 2,048 spatial features, then **correct** gather-form transposed convolution to `8³×16→16³×1`, sigmoid to `[0,1]`.
5. Stable six-neighbour spatial diffusion (zero-flux boundaries; `0≤α≤1/6`).
6. Parallel per-node work, **synchronous real neighbour voxel exchange** from an immutable snapshot, then explicit `5%` blending toward a synthetic cross-lattice test target.
7. Measure post-step target MSE, voxel mean/min/max, deterministic FNV-1a hash of float32 bits, and observed wall-time in ms.

## Truth-in-performance and scientific boundaries

- **No training or SGD/backprop:** weights are reproducibly randomized, not learned. No evidence of compression fidelity or VAE likelihood optimization; outputs can converge to trivial patterns.
- **No latent exchange:** step six exchanges *decoded voxel fields*, not learned latent vectors. True latent coupling remains future work.
- **Not a 10^18-voxel emulator:** `1000000³` is a *symbolic virtual coordinate-domain annotation*. For `--swarm 8`, only `512 × 4096 = 2,097,152` explicit state voxels exist (plus temporary tensors and model weights). No claim that full trillion/trillion-scale domain is simulated.
- **The target is a synthetic shape, not a cryptographic watermark:** the injected `target_pull` trivially influences similarity. MSE improvements must not be presented as proof of IP ownership, origin, protection, or trained convergence.
- **No GPU:** uses OpenMP CPU only; do not interpret successful execution as GPU acceleration.
- **No universal bitwise portability:** hashes should match across repeated runs on the same toolchain/system with deterministic settings. Compiler math modes/hardware may differ; CI asserts within-build thread parity.
- **No autonomous optimization:** coefficients are fixed heuristics; no reward-based or measured self-tuning. Drift uses a sinusoidal schedule; no formal convergence guarantee.
- **Input shape fixed:** 16³ single-channel only, enforced at runtime. Outputs do not handle arbitrary dataset or multimodal input.

## Verification and follow-on engineering

`--self-test` checks mass conservation for diffusion, constant-field invariance, alpha rejection, correct transpose kernel indexing, output shape and bounds, same-seed replay, and deterministic 1-vs-4 OpenMP threads for a seeded two-side swarm. CI compiles with GCC/Clang and runs both CTest tests. To substantiate autoencoding/compression, implement supervised or self-supervised training, true latent packet serialization, reconstruction-to-input MSE, and external held-out volume benchmarks before reporting a compression ratio.
