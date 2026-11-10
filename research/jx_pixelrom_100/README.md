# JARVIS-X Ω³ | JX-PROM-100 Pixel ROM (isolated research reference)

This C++17/Python module is a lossless RGB8 **100-bank Pixel ROM**: bytes inside
100 fixed 128 x 32 pixel patches in one 7856 x 4496 PNG reconstruct an exact
**1,024,000-byte JX-SUPERFRAME**. The recovered, validated JXEX program drives a
bounded multiparallel 3D DCT codec over a **virtual** 10^12 task address space.
The tested fixture processes 256 synthetic active 16^3 voxel tiles, not
one trillion physical tasks, nor a learned ANN. All generated output is ignored
by source control; no proprietary infographic is embedded in this PR.

## One-command reproducible build and verification

Requirements: CMake 3.16+, C++17 compiler, libpng, OpenSSL Crypto, zlib,
Python 3 with Pillow. On Ubuntu: `sudo apt-get install cmake g++ libpng-dev
libssl-dev zlib1g-dev python3-pil`.

```bash
./run_pipeline.sh
```

The script builds C++ binaries, **generates** the control frame via the C++
engine, generates a deterministic diagnostic illustration, encodes it as a
100-bank lossless RGB PNG, verifies that Python and C++ recover identical
bytes, executes the recovered VM, checks independent 3D reconstruction,
and runs CTest (including tamper checks and 1 vs 8 worker determinism).

To use the previously generated JARVIS-X 10x10 infographic as the visible
background rather than the diagnostic illustration, provide a **lossless
7856 x 4496 RGB PNG** to `bootstrap_pixelrom.py --background <source.png>`.
The designated pixel bank patches are overwritten. Do not JPEG-compress,
colour-grade, rescale or print-and-rescan the encoded PNG.

## Addressing and codec mechanics

- Task index: `i = 1,000,000*r+c`, with `0 <= r,c < 1,000,000`.
- 3D tiling: `x=i%10000`, `y=(i/10000)%10000`, `z=i/100000000`.
- 54-bit voxel address: 42-bit Morton tile address plus 12-bit local voxel.
- 100 banks, each 128x32x3=12,288 RGB8 raw bytes: 128-byte bank header,
  10,240-byte control payload, 1,920-byte padding.
- Bank coordinates: `(16+(i%10)*784+632, 16+(i//10)*448+392)`.
- 1,024,000-byte content: `JXSF1T12` header and JXEX interpreter at byte 4096.
- Decoder checks exact image dimensions, bank CRC32, per-bank SHA-256,
  full-frame SHA-256 and the original control frame CRC and opcode whitelist.
- Adaptive K=4/16/64 3D DCT with exact FP32 fallback if quality fails;
  outputs canonical ordered archive with reference verification.
- JXOP optimizer program is **not executable** in this reference.

## Security and performance boundaries

Checksums are **not keyed authentication**. This ROM executes a strict
non-general opcode whitelist, but must not be treated as a trusted remote-code
execution format. No physical ROM chip, GPU-scale acceleration or autonomous
neural model evolution is claimed. The 1000 KiB frame is mostly zero-reserved
space, and images/archives are generated on demand rather than checked in.

## Suggested acceptance gates

CI passes on GCC/Clang, Python and C++ byte-for-byte recovery, bit-tamper and
wrong-size rejection, and independent archive verification. Add signed
manifests, signed release artifacts and real data before production use.

The module is **path-scoped** and does not alter existing JARVIS-X runtimes.
