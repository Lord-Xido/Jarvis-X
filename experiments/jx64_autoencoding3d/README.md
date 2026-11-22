# JARVIS-X — AutoEncoding3DEngine 64³ correctness and separable-kernel reference

**Scope:** C++20 CPU implementation of the user-supplied 64 × 64 × 64 seeded nonlinear state transformer. This PR contains an unmodified-behaviour baseline file and a separate corrected implementation. It does **not** claim that the transformation is a trained ANN, lossless data codec, GPU kernel, or an 8→3→8 bottleneck.

## Audit

The original source:

- `R3` feedback persists across `execute()` calls because it is not cleared. A repeated invocation with the same seed is not generally a repeatable experiment.
- `matrixPropagate()` writes only the interior of `R2`, while `decode()` consumes edge voxels left over from the permutation; `decode()` also overwrites only the interior of `R0`, preserving the original seed on boundaries.
- `reconstructionLoss()` computes MSE between `R1` (nonlinear encoded activation) and `R0` (partially overwritten decoded field), **not** the actual original input (before encoding).
- The supplied `activate` function is an explicit rational approximation with clipping, not `std::tanh`, so its numerics should be tested as written.
- Multiplication by odd `stride` modulo `64³ = 2^18` is reversible as an **index permutation**, but the subsequent activation/feedback/stencil generally are not. There is no trained latent bottleneck or lossless residual in this implementation.

The corrected engine:

1. Retains the same 64³ geometry, seed PRNG, nonlinear activation, encoding parameters, modulo stride and feedback formula.
2. Resets every feedback run and treats `iterations=0` as the identity feedback path.
3. Implements **clamped boundaries for every output voxel**, and uses separate `R4` stencil storage so permutation, propagation and decoding cannot accidentally mix stale state.
4. Reports both MSE to true saved input and MSE to encoded activations.
5. Provides 27-neighbour reference decoder and 3-pass separable decoder, which implement the same linear 3×3×3 box filter **mathematically**. Their floating-point summation order differs: numerical tolerance is validated at `<1e-5`; hashes are intentionally distinct.
6. Uses a modular multiplicative inverse to test permutation roundtrip for any odd stride (zero-normalized as `stride|1`).

## Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/jx64_engine --seed 0xDEADBEEF12345678 --stride 73 --iterations 8 --decoder separable --repeat 10
./build/jx64_engine --seed 0xDEADBEEF12345678 --stride 73 --iterations 8 --decoder reference --repeat 10
```

`original_baseline.cpp` preserves the input implementation for review; the build defaults to the corrected code.

## Mathematical stages

For `i ∈ [0,262144)`:

```text
R0 = xorshift64(seed) bytes mapped to [-1,1)
R1[i] = f(2.5 R0[i] + 0.5)
R2[i] = R1[(i × (stride | 1)) mod 262144]
R3_t[i] = f(R2[i] + gain × R3_(t-1)[i]); initial R3 = 0
R4[x,y,z] = 0.4 R3[x,y,z] + 0.1 Σsix_clamped_neighbors R3
R0'[x,y,z] = f((1/27) Σ3x3x3_clamped_neighbors R4)
MSE = mean((R0' - original_input)^2)
```

The decoder's mean is separable because its uniform 3×3×3 convolution equals sequential 3-tap convolution in x/y/z. For bitwise-identical output, use `--decoder reference`; optimized mode is intentionally numerical, not bitwise, equivalent.

## Security and reproducibility

No credentials, external network dependency, or external libraries. CI builds with C++20, strict warnings and `-ffp-contract=off`. FNV-1a checksum is a noncryptographic diagnostic and does not authenticate an archive. Results may differ across FP architectures due to rounding or compiler behaviour.

## Limitations

This is a deterministic 3D field transformer with fixed coefficients; **it is not an autoencoder trained to preserve or compress inputs**. No original-byte reconstruction or GPU hardware speedup is claimed. GPU/Metal deployment needs a separate device build and parity tests.
