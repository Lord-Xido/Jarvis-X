# G3M Ξ — Bounded LibTorch/C++17 Geometric Autoencoder

## Status and purpose

**Compiled and executed CPU reference**: arbitrary float32 XYZ input → point-cloud normalization → bounded support sample → kNN EdgeConv → distance-biased attention → 512-dimensional variational latent → deterministic folding decoder → generated XYZ output. An optional candidate-first latent recurrence applies `F(z) = E(D(z))` only when its consistency gap decreases and source-sample Chamfer error does not increase by more than 1%.

This is a **lossy ANN geometry generator**, not an exact lossless archive and not a self-replicating C++ quine. It does not demonstrate 1,000,000× acceleration or fidelity on real scans. A million-point input is **subsampled to S<=1024** for the expensive graph operations; output size is configurable and can be 1,000,000 points using chunked decoding. The test model is trained on one synthetic geometry and has limited reconstruction fidelity.

## Compile

Install CPU/GPU LibTorch appropriate to your platform, or use the CMake files of an installed Python torch package. This source requires the **C++17** front end.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j1
```

This environment used Clang 17, LibTorch/PyTorch 2.10.0+cpu, and `-O0 -g0` for reliable compilation in a constrained runner. Higher optimization can be tested separately. On another machine specify `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

## Demo: million-point synthetic input/output

```sh
./build/g3m --points 1000000 --support 128 --epochs 6 --decode 1000000 --refine 2 --output million_output.f32
```

Generated `.f32` output stores little-endian float32 interleaved `(x,y,z)` triples and has no header. A 1M-point output file occupies exactly 12,000,000 bytes. Its point count does **not** indicate reconstruction quality.

## Arbitrary input: train and export

Prepare a file of `N` float32 little-endian interleaved XYZ points, no header, length exactly `12*N` bytes:

```sh
./build/g3m --input input.f32 --support 128 --epochs 50 --decode 4096 --refine 3 \
  --save-weights trained_weights.pt --output output.f32
```

Output points are transformed back to the original coordinate frame. Shape and finiteness of input are validated. Maximum input and output are 1M points in this bounded reference.

## Reload a model (inference without retraining)

```sh
./build/g3m --input input.f32 --support 128 --weights trained_weights.pt \
  --decode 4096 --refine 3 --output output_reloaded.f32
```

Do not load untrusted model files; this is not a sandboxed weights format. Model and input data remain local.

## Verify

```sh
python3 tests/test_engine.py ./build/g3m
```

The test includes arbitrary input, save/load equality, rejection of non-finite and malformed input, and a million-point synthetic run with a bounded graph support.

## Detailed constraints

- `kNN`: exact pairwise kNN on **S support points** only; complexity `O(S²)` in time and memory, with self-exclusion.
- Sampling: deterministic pseudo-permutation indices, **not** learned importance or curvature-based sampling.
- EdgeConv: learned functions of central coordinates and neighbor displacement, max-aggregated per support point.
- Attention: `softmax(QKᵀ / sqrt(64) - 0.5*EuclideanDistance)`. The distances are a spatial **proxy**, not geodesic paths on the manifold.
- Bottleneck: Gaussian mu/logvar and normalized projection into an open unit ball. That projection alone does not establish a Poincaré manifold metric.
- Decoder: latent projection `512→64` and a folded normalized fixed 2D grid with output point count M (not a mesh with connectivity).
- Objective: squared Chamfer plus `1e-4 * KL`. Train/test are **the same single input** in the default demo. No independent validation dataset is implemented.
- Refinement: proposed latent `0.5*z + 0.5*E(D(z))`, accepted only if geometric and latent gates pass. This is a bounded empirical monotonicity guard, **not** a proven Banach contraction.
- Reconstruction is lossy. Exact reconstruction of general float32 input would require a residual channel or other sufficient side information.
- This implementation is CPU-only by default; GPU architecture, learned curvature scoring, topological losses, batch/multi-cloud dataset training, mesh reconstruction, cloud hosting, authenticated APIs and provenance ledger integration are *future work*.

## Measured CPU examples

- Synthetic torus: N=8,192, S=128, M=512, 12 epochs. Training loss 0.718902 → 0.510774, three refinement steps accepted, latent self-gap 0.182464 → 0.0227679.
- Synthetic torus: N=1,000,000, S=128, M=1,000,000, 6 epochs. Training loss 0.812659 → 0.686937; 2 refinement steps accepted; output 12,000,000 bytes. Observed runtime under one second on the benchmarked CPU instance, **not** an independent or production-grade speedup comparison.
- Arbitrary synthetic 4,096-point sphere: train/save → reload produced byte-identical 256-point outputs. Non-finite/truncated inputs rejected.

See `VALIDATION.md` for limitations and evidence. All tests use synthetic data.
