# JARVIS-X 3D Neural Autoencoder v4.0

Isolated C++17 **reference** runtime for trainable 3D U8 voxel autoencoding/decoding. This module is deliberately not wired into existing Jarvis-X targets or production services.

## Mathematical contract

For each non-overlapping `4×4×4` block `x ∈ [0,1]^64`:

```text
z = tanh(W2 tanh(W1 x + b1) + b2)       # 64 -> 32 -> 12
q = round(127 z)                         # 12 signed int8 latent values
z' = q / 127
xhat = sigmoid(W4 tanh(W3 z' + b3) + b4) # 12 -> 32 -> 64
prediction = round(255 xhat)
residual = original XOR prediction       # exact mode only
restored = prediction XOR residual
```

The shared model has **5,004 parameters**, not 10^36 physical weights. Each coordinate axis admits `0 <= coordinate < 10^12` for virtual placement; only bounded input volumes are allocated. Spatial blocks are processed independently: this is not a 3D convolutional or cross-block attention model.

## Build and verify

```bash
cmake -S cpp_runtime/ae3d_v4 -B build/ae3d_v4 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ae3d_v4 --parallel 2
ctest --test-dir build/ae3d_v4 --output-on-failure
```

Or run the CLI directly:

```bash
./build/ae3d_v4/jxae3d generate volume.jxvol 32 42
./build/ae3d_v4/jxae3d train volume.jxvol model.jxmodel 80 0.004
./build/ae3d_v4/jxae3d encode volume.jxvol model.jxmodel compressed.jxpkt lossy
./build/ae3d_v4/jxae3d decode compressed.jxpkt model.jxmodel decoded.jxvol
./build/ae3d_v4/jxae3d compare volume.jxvol decoded.jxvol
./build/ae3d_v4/jxae3d encode volume.jxvol model.jxmodel exact.jxpkt exact
./build/ae3d_v4/jxae3d decode exact.jxpkt model.jxmodel exact-restored.jxvol
./build/ae3d_v4/jxae3d --quine > reproduced.cpp
cmp reproduced.cpp cpp_runtime/ae3d_v4/src/jx_ae3d.cpp
```

The `generate` command produces deterministic synthetic data. `import-raw`, `export-raw`, `query`, `slice`, `inspect` and validation-gated `evolve` are also supported by the CLI.

## Reproducible local observations (2026-11-05)

- Native self-test and standalone GCC compile: **pass**.
- Clang C++17 compile and self-test: **pass with non-fatal warnings**. Strict Clang `-Werror` currently fails on misleading indentation and an unused variable; fix with a regenerated quine in a follow-up.
- Local 8³ independently created voxel volume: exact archive decoded to a byte-identical `.jxvol`; lossy archive decoded and reported distortion; truncated archive rejected; coordinate query passed.
- Earlier synthetic 32³ demonstration, **excluding the shared model checkpoint**: 32,768 raw voxel bytes, 6,218-byte lossy archive (5.27×), 39.15 dB PSNR; 38,944-byte exact archive (larger than raw). These numbers are dataset-specific and not a general-purpose compression guarantee.

## Safety and interoperability boundaries

- Input volumes are capped at 64 MiB; file reads at 128 MiB.
- On-disk CRC32 detects accidental corruption, **not** intentional forgery or tampering. The model fingerprint is non-cryptographic. Do not treat either as authentication or encryption.
- Preserve the matching model checkpoint to decode archived latents. Do not silently overwrite production checkpoints.
- No CUDA/GPU, distributed cluster, public cloud deployment, arbitrary-media-native codec or trillion-parameter training is claimed.
- No binary artifacts, private datasets, model checkpoints, credentials or original IP-locked v1 sources are committed to this public repository.
- The C++ program's `--quine` outputs its own exact source. It does not autonomously alter its source.
- CI is review-gated and is not evidence of cluster performance or cloud readiness.

This module complements existing `moagi3d_operational_v2` and ANN-related components without changing their APIs or semantics.
