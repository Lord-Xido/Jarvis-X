# Dr Moagi Pixels-to-Bits Mapping Framework — End-to-End Reference

## Status

This document describes the executable software reference for the pixels-to-bits mapping layer governed by the canonical `DM-vOmegaXi+` 8192-byte container contract.

It implements a real round-trippable 10-bit RGB bitstream and uses the existing VCL-BVM-8 / DM-IMP execution core as the bounded inward control/adaptation layer.

It does **not** claim that an arbitrary 8K RGB10 frame fits losslessly inside 8KB. The 8KB container is the reusable execution, latent, residual, microcode and transactional workspace. Encoded frame information is carried by the external `DM-PXBT` bitstream.

It also does not assert measured 8K120 throughput, sub-nanosecond silicon timing, TSV/eSRAM realization, external SOTA or patentability.

## 1. Raw 8K120 arithmetic

For RGB10 7680 x 4320:

```text
pixels/frame       = 33,177,600
bits/pixel         = 30
raw bits/frame     = 995,328,000
packed bytes/frame = 124,416,000
120 Hz packed rate = 14.92992 GB/s
frame budget       = 8.333333... ms
```

These values are exposed by `uhd8k120_profile()` without allocating an 8K frame.

## 2. Executable pipeline

```text
RGB10 frame
   |
   v
8x8 pixel micro-tiles
   |
   +----> 512-byte VCL projection
   |          |
   |          v
   |     VCL-BVM-8 / Psi-Phi-Omega-Theta control path
   |          |
   |          v
   |     8192-byte DM container + commit receipt
   |
   v
Morton-ordered per-channel predictive residuals
   |
   v
candidate quantization shifts 0..N
   |
   v
Lambda quality gate (tile MSE + tile PSNR)
   |
   v
variable-bit residual packing
   |
   v
DM-PXBT v1 bitstream
   |
   v
bit fetch -> predictor reconstruction -> RGB10 frame
```

The actual frame codec is independently decodable. It does not require the original pixels or an unstored neural latent state to reconstruct the frame.

## 3. Lossless and near-lossless modes

### Lossless

`shift = 0` stores exact predictive differences. Reconstruction satisfies:

```text
P_hat == P
MSE   == 0
PSNR  == infinity
hbar_semantic_visual == 0
```

The bitstream may expand on high-entropy inputs; no universal compression claim is made.

### Near-lossless

For each tile the encoder evaluates quantization shifts from 0 through `max_shift`.

For step `s = 2^shift`:

```text
q       = round((pixel - reconstructed_predictor) / s)
P_hat   = clamp10(reconstructed_predictor + q*s)
```

A candidate is admissible only when:

```text
MSE_tile  <= max_tile_mse
PSNR_tile >= min_tile_psnr_db
```

Among admissible candidates the smallest encoded record wins. Exact shift-0 coding is always available as the fidelity fallback.

## 4. 3D / Morton mapping

Pixels inside each 8x8 tile are visited in 2D Morton/Z-order. The local DM container separately exposes a canonical toroidal `8x8x8` address mapping:

```text
x' = x mod 8
y' = y mod 8
z' = z mod 8
```

including correct wrapping for negative coordinates.

This is logical topology. Physical TSV toroidal routing remains a hardware workstream.

## 5. DM-PXBT v1 stream

The global stream contains:

- magic `DMPXBT1`;
- version;
- width / height;
- target FPS;
- RGB channel count;
- 10-bit source depth;
- 8x8 tile edge;
- mode and max-shift metadata;
- tile count;
- raw source bit count;
- length-prefixed tile records;
- final deterministic stream digest.

Each tile record contains:

- tile X/Y;
- valid edge dimensions;
- selected shift;
- three residual bit widths;
- three 10-bit predictor bases;
- packed per-channel residual payload;
- tile digest.

Malformed, truncated or corrupted streams fail closed.

## 6. Exact 8192-byte container integration

`dm8kb_container.hpp` implements the canonical state map exactly:

```text
CONTROL    128 B
VCL_STATE  512 B
THETA      512 B
MASKS      128 B
OMEGA      512 B
MICROCODE 1024 B
RESIDUAL  2048 B
FEATURES  2048 B
SHADOW    1024 B
INTEGRITY  256 B
----------------
TOTAL     8192 B
```

Every tile executes the canonical VCL program on an 8-bit projection of the RGB10 samples. The resulting state, adaptive weights, masks, Omega values, residual telemetry and candidate record are synchronized into the container.

The `SHADOW` region holds the candidate tile record before commit. The `INTEGRITY` region records:

- prior state digest;
- candidate digest;
- result digest;
- microcode digest;
- epoch;
- changed-region bitmap;
- semantic-gap telemetry;
- state-delta telemetry;
- accepted/rejected state.

The container digest deliberately excludes `INTEGRITY`, allowing receipts to describe authoritative state without self-referential digest recursion.

## 7. Semantic gap

Frame-level visual semantic residual is currently the normalized RMSE reference metric:

```text
hbar_semantic_visual = sqrt(MSE) / 1023
```

This is a measurable reference quantity, not a universal physical constant. Additional perceptual/task metrics belong in the benchmark workstream.

## 8. Executable target

CMake target:

```text
jarvisx-pixels-to-bits
```

Windows executable:

```text
DrMoagi-Pixels-to-Bits.exe
```

Generated deterministic gradient, near-lossless:

```powershell
.\DrMoagi-Pixels-to-Bits.exe `
  --width 1920 `
  --height 1080 `
  --fps 120 `
  --max-shift 4 `
  --max-mse 20 `
  --min-psnr 47 `
  --encoded frame.dmpb `
  --decoded frame.ppm
```

Exact lossless:

```powershell
.\DrMoagi-Pixels-to-Bits.exe `
  --input source.ppm `
  --lossless `
  --encoded source-lossless.dmpb `
  --decoded restored.ppm
```

Input PPM is binary P6 with maxval in `[1,1023]`. 8-bit PPM input is scaled into the canonical RGB10 domain; decoded PPM is emitted with maxval `1023`.

## 9. Verification

The regression suite checks:

1. exact 8192-byte memory layout;
2. hard container-local bounds;
3. toroidal address wrapping;
4. exact 8K120 raw arithmetic;
5. lossless round-trip including partial edge tiles;
6. quality-gated near-lossless reconstruction;
7. deterministic bitstream and final container state;
8. corruption detection / fail-closed decode;
9. bounded 1024-byte shadow staging;
10. executable CLI smoke path.

The existing C++ workflow compiles and runs the complete suite on GCC, Clang with sanitizers, and MSVC.

## 10. Evidence boundary

The software reference establishes functional semantics and deterministic bitstream behavior.

The following remain separate evidence gates:

- actual 8K120 wall-clock throughput and latency;
- GPU/NPU/SIMD comparison and external SOTA;
- FPGA/ASIC Fmax, timing closure and energy;
- TSV/eSRAM/3D-stacked implementation;
- codec perceptual benchmarks such as SSIM/MS-SSIM/VMAF;
- formal patentability/legal conclusions.

Those are tracked under the existing commercialization and hardware/benchmark workstreams.
