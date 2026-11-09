# JARVIS-X Ω³ — 512-latent 3D autoencoder reference

Isolated C++17 research prototype. Existing 3D engines and locked baselines are not modified.

## Architecture

- Input: exactly 64×64×64 UINT8 voxels (262,144 bytes), x-fastest, then y, then z.
- Encoder: 8×8×8 box averaging into an 8×8×8 FP32 latent field = 512 values (2,048 bytes).
- Decoder: 27-tap local 3D *linear* shared stencil with trilinear spatial interpolation, 27 trainable FP64 coefficients and one FP64 bias.
- Training: at most six full-volume gradient steps with monotonic backtracking and rejection of non-improving candidate updates, measured with normalized unquantized MSE. The decoder uses only the stored latent and weights at inference.
- This is a trainable linear decoder with a **fixed encoder**, not a deep neural network, VQ-VAE, or lossless codec.
- Quantized output: 262,144 UINT8 voxels; PSNR is calculated against original UINT8 data using peak value 255.
- Training on the encoded input is **in-sample**. A second synthetic field is tested for basic out-of-sample behavior, but no real-world generalization is established.

## Binary archive v1: 2,304 bytes, little-endian

| Field | Bytes |
|---|---:|
| ASCII magic JXL51201 | 8 |
| UINT32 side=64, latent=512, taps=27, payload=2272 | 16 |
| FP32 latent: 512 × 4 | 2048 |
| FP64 decoder weights: 27 × 8 | 216 |
| FP64 decoder bias: 1 × 8 | 8 |
| FNV-1a-64 checksum of preceding bytes | 8 |
| **Total** | **2304** |

The decoder checks exact length, magic, dimension metadata, finite floats, latent range and checksum. FNV-1a is a non-cryptographic corruption check, not authentication or an adversarially robust MAC.

## Compression calculations

- **512×** = 262,144 / 512, an element count ratio **only**, not a byte ratio.
- **128×** = 262,144 / 2,048, excluding model weights, metadata and checksums.
- **113.7778×** = 262,144 / 2,304, including the per-volume model and archive overhead.
- This tool supports UINT8 source data only. Different source precision or external shared-model accounting changes these figures.

## Build and test from repository root

    cmake -S experimental/latent512_3d -B build/jxl512 -DCMAKE_BUILD_TYPE=Release
    cmake --build build/jxl512 --parallel
    ctest --test-dir build/jxl512 --output-on-failure
    build/jxl512/jxl512 demo
    build/jxl512/jxl512 encode input_64cube_uint8.raw latent.jxl512
    build/jxl512/jxl512 train input_64cube_uint8.raw trained.jxl512
    build/jxl512/jxl512 decode trained.jxl512 reconstructed_64cube_uint8.raw

The demo emits opcode-**tagged** execution trace records. They are not executable general-purpose bytecode or a full VM. Pipeline: LOAD(0x01) → ENCODE(0x10) → TRAIN(0x11) → PACK(0x20) → UNPACK(0x21) → DECODE(0x30) → EVAL(0x31) → HALT(0xff). Normal decode never performs training.

Selftest checks exact constant-field reconstruction, loss non-increase, deterministic same-build training, archive roundtrip, corruption and truncation rejection, a held-out synthetic volume, and finite MSE. Scoped GitHub Actions compiles with strict GCC and Clang, runs both CTests and archives the demo trace for 14 days.

## Local synthetic evidence (9 Nov 2026)

Strict GCC and Clang C++17 Release builds: **2/2 CTests passed**. Synthetic 64³ field generated from smooth sinusoidal and Gaussian structure with seed 11:

| Metric | Value |
|---|---:|
| Input UINT8 bytes | 262,144 |
| Archive bytes | 2,304 |
| Quantized MSE before training | 11.72634506 |
| Quantized MSE after six accepted updates | 10.91527557 |
| Final PSNR | 37.75045656 dB |
| Archive FNV-1a-64 | 8d5a6bd18ba0c8c7 |
| Reconstructed UINT8 FNV-1a-64 | bc20b1bc9fa11015 |

Only a synthetic CPU reference has been tested. **No** 1,000 GB/s streaming, 1,000× speedup, GPU/Metal execution, losslessness, photorealism, real clinical-image fidelity or production deployment is claimed. High-frequency information is discarded by the encoder. No autonomous source mutation or automatic merge.
