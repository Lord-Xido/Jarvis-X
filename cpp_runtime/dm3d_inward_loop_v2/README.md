# Dr Moagi 3D v2 — C++20 inward-loop codec + agent orchestration runtime

A compiled, tested, **lossless** 3D virtual-addressing and multimedia-byte processing engine. Its self-refining path is operational:

`INGEST → MAP_3D → PROPOSE_CODECS → DECODE_EACH → BYTEWISE_CTR → CHOOSE_VERIFIED_MINIMUM → INWARD_FOLD_3D → UPDATE_Ω → ARCHIVE → DECODE → AUDIT`

The runtime is a **reference engine**, not an LLM, AGI or an autonomous cloud service. Multimedia files are handled as opaque lossless bytes; source inspection recognizes PNG, JPEG, GIF, WAV, MP4/ISO-BMFF, PDF, and common file extensions, but there is no semantic model inference, frame/speech processing, or generative AI attached.

## Build and tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/dm3d selftest
```

Requires C++20, CMake, and GCC/Clang with `unsigned __int128` support. On Windows, use WSL or a compatible GCC/Clang toolchain; MSVC's native compiler does not support this index type as written.

## Full inward-loop command

```bash
./build/dm3d loop examples/generated/sample.mp4 example.dm3d recovered.mp4 --threads 4 --folds 12 --passes 5
cmp examples/generated/sample.mp4 recovered.mp4
```

`--threads` 1..128 limits the worker batch. `--folds` 1..1000 controls toroidal latent contraction iterations. `--passes` 1..5 controls independently verified reversible codec proposals per tile. The codec always has an exact raw fallback. `--passes 5` searches all five spatial predictors. The previous successful predictor is evaluated first in the next batch (rolling policy preference).

Equivalent operations:

```bash
./build/dm3d encode input.wav audio.dm3d --threads 4 --folds 12 --passes 5
./build/dm3d decode audio.dm3d restored.wav
./build/dm3d verify audio.dm3d input.wav
./build/dm3d inspect audio.dm3d
./build/dm3d inspect input.wav
```

## Virtual 3D memory

“1000GB × 1000GB × 1000GB” is interpreted as **10^12 coordinate positions per axis**, with one hypothetical byte per cell. The full logical cube contains 10^36 positions, **not** physically allocated storage. Indexing is exact over the entire cube using an unsigned 128-bit integer:

`index = x + 10^12 * (y + 10^12 * z)`.

A tile is `16 × 16 × 16 = 4096` bytes. An optional sparse append-only volume stores only explicitly written voxels, with unwritten cells implicitly zero:

```bash
./build/dm3d voxel-set world.vxl 999999999999 999999999999 999999999999 213
./build/dm3d voxel-get world.vxl 999999999999 999999999999 999999999999
./build/dm3d coord 999999999999 999999999999 999999999999
./build/dm3d uncoord 999999999999999999999999999999999999
```

The volume is an append log with last-write-wins semantics and O(number of records) lookup, not a production-scale spatial index.

## Actual recursive auto-encoding/decoding loop

For every tile the encoder starts with a lossless raw candidate. It then tries up to 5 causal spatial XOR predictors: preferred-axis priority; XOR of the three predecessor axes; X-only; Y-only; Z-only. Each prediction undergoes zero-run RLE. **Every proposal is decoded back to a 4096-byte tile and compared byte-for-byte with the original before adoption**. Only candidates that are exact and strictly smaller replace the incumbent. This is a bounded self-correction/optimization loop, not neural gradient learning.

The separate neural-inspired 3D latent field pools `16³ → 4³`, then iterates a periodic six-neighbor contraction. Its outputs update the Ω exponential-moving-average summary. The latent field **does not replace the original bytes** and therefore cannot cause lossy media reconstruction. See `SPEC.md`.

## Agent stages

- PLANNER: bounded parallel work batches, virtual tile coordinates.
- MULTIMODAL: signature detection and selected metadata for PNG/WAV/MP4/text.
- ENCODER: causal 3D spatial predictors, XOR residuals, RLE/raw fallback.
- INWARD: propose → decode → contrast → choose smaller verified output.
- FOLD: recursive 4³ periodic latent contraction.
- Ω MEMORY: rolling EMA and next-batch predictor preference.
- CTR/AUDITOR: per-tile decoded checks, exact original byte comparison, whole-stream checksum, atomic partial-file output.

These are deterministic C++ processing stages, not LLM-driven agents.

## Archive format: DM3DARC2 (version 2)

Little-endian header (48 bytes): magic `DM3DARC2` (8 bytes), version u32, modality u32, original size u64, tile count u64, FNV64 u64, Ω as IEEE-754 u64 bits. Per tile: 3×u64 virtual tile origin, u32 valid count, u8 mode (0 raw, 1–5 predictors), u32 payload length, u64 FNV64, encoded bytes. The archive is a project-specific format; version 1 archives are **not compatible** with v2.

`FNV64` is a noncryptographic accidental-corruption detector; there is no encryption, authentication, signatures, or adversarial protection. For production, use SHA-256/BLAKE3 and an authenticated manifest. Archive compression might **increase** size on small or incompressible files due to metadata and zero-padding.

## Engineering boundaries

Virtual addressability does not grant memory, FLOPs, bandwidth or the ability to inspect 10^36 independent data cells. Storage and runtime scale with **actual** bytes and chosen passes/folds. There are no performance claims beyond tested behavior. For full agentic multimedia AI, integrate media decoding (FFmpeg), model execution (ONNX Runtime or llama.cpp), tool permissions, content storage, queueing, and distributed worker RPC.
