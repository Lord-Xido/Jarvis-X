# DM3D Mandelbulb four-way lossless codec (MB4C v1)

This C++17 reference integrates a **power-8 Mandelbulb voxel sampler** with a **4-to-1 / 1-to-4 reversible spatial lifting hierarchy**. It is an independent component and does not change existing DM3D VM bytecode, CMX3, MP4 or other codec formats.

## Operational path

1. Generate 8-bit Mandelbulb escape-time voxels, or ingest an exact externally supplied byte volume.
2. Split each 3D volume in an XY, then YZ, then ZX repeating cycle. Each stage maps 4 children to 1 parent **plus 3 signed integer residuals**.
3. Repeat to the final attainable level (for side=16, six levels).
4. Encode the coarse values followed by deepest-to-shallowest residuals using zero-run-length tokens and zigzag unsigned LEB128.
5. Select a raw fallback when the transform stream is not smaller (unless forced).
6. Decode from archive alone, reconstruct every original byte, reject malformed/corrupt input, and require max absolute reconstruction error of zero.

At each four-child group, using child values x0, x1, x2, x3:

    parent = x0
    details = (x1-x0, x2-x0, x3-x0)
    inverse = (parent, parent+d1, parent+d2, parent+d3)

The hierarchy reduces coarse node count, not information entropy by fiat. Residuals are required for arbitrary lossless reconstruction. This is a reversible predictive codec, not a trained autoencoder or an adaptive spatial octree.

## Build and run

    cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release -DJARVISX_BUILD_GL_VISUALIZER=OFF
    cmake --build build/cpp-runtime --target jarvisx-mandelbulb4 --parallel 2
    ctest --test-dir build/cpp-runtime -R "^mandelbulb4-" --output-on-failure
    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 32 --iterations 18 --out mandelbulb.mb4 --decoded recovered.raw

Append --force-hierarchy to force the hierarchical stream.

Independent archive decoding, without regenerating the Mandelbulb or requiring the source field:

    ./build/cpp-runtime/jarvisx-mandelbulb4 --decode-archive mandelbulb.mb4 --decoded replay.raw

The CI performs a binary comparison of the internally reconstructed and independently decoded raw outputs.
To encode arbitrary XYZ-order uint8 sample bytes instead of generating a fractal:

    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 16 --input input.raw --out output.mb4 --decoded decoded.raw

An input volume must contain exactly side^3 bytes. The first axis (x) is the fastest-varying array index. The CLI currently supports power-of-two sides 4, 8, 16, 32, and 64; it intentionally materializes a bounded grid. Fractal iterations are restricted to 2..64.

## Archive contract

17-byte little-endian header, followed by either raw data or the hierarchical residual stream:

- ASCII magic MB4C (4 bytes)
- Version 1 (1 byte)
- Mode 0 raw or 1 hierarchical (1 byte)
- Side uint16 LE (2 bytes)
- Depth uint8 (1 byte)
- FNV-1a 64-bit checksum of uncompressed voxel bytes (8 bytes)

Hierarchical mode serializes the final coarse state and all detail levels in reverse order, reconstructible using the side and fixed XYZ stage schedule. Sequence records use token 0 + unsigned LEB128 positive zero count, or token 1 + zigzag unsigned LEB128 signed value. The decoder checks lengths, tokens, dimensions, overflow, trailing bytes, voxel range and checksum. FNV-1a is for accidental corruption detection, **not cryptographic authentication**. The archive is not yet a stable cross-project wire protocol.

## Verification and capability boundaries

CTest covers procedural volume generation, forced hierarchical round-trip, auto-mode round-trip, zero-field compression, raw fallback on low-redundancy data, integrity mismatch, truncation, trailing data, and an actual CLI archive/raw output cycle.

The Mandelbulb generator uses floating-point trigonometry; generated samples need not agree across different CPUs or math libraries. Archive decoding is exact independently of fractal evaluation, and verification is against the bytes actually sampled or supplied. The current geometric hierarchy is uniform in space; boundary-adaptive subdivision, certified distance estimates and learned neural predictors are future research, not implemented capabilities.

The baseline archive size is reported from actually serialized bytes. No fixed compression ratio, universal speedup or global convergence guarantee is claimed.


## Adaptive four-way geometry-guided codec (mode 2)

The updated codec supports an additional archive mode: mode 2, a geometry-guided *cost-gated 4-ary volume tree*. At each region, the encoder examines the exact sampled or ingested uint8 voxel field:

- If every voxel is identical, store 1-byte node tag 0 and one value (2 bytes total), without subdividing.
- Otherwise evaluate a literal leaf (1-byte tag 1 followed by all voxels in local XYZ traversal).
- When XY, YZ or ZX divisibility permits, recursively encode four spatial child regions; select a split node (1-byte tag 2 + four children) **only if its total serialized bytes are strictly fewer than a literal leaf**.

This computes a local minimum across constant, literal and split candidates with children optimized recursively. It is data-aware because Mandelbulb escape-time interiors and exterior regions tend to be locally uniform while intricate boundaries are mixed. It does **not** evaluate a certified surface-distance bound and does **not** guarantee that sampling accurately resolves features smaller than the chosen voxel pitch. It is also not an entropy coder or learned neural model.

The adaptive decoder uses the fixed XY/YZ/ZX subdivision schedule to parse the tree from the serialized archive, filling every output voxel exactly. It does not run the procedural Mandelbulb generator, making the archive self-contained. Malformed geometry, invalid node tags, truncated leaf data, trailing bytes, wrong depth and checksum mismatch are rejected.

Archive header remains 17 bytes. Mode 2 uses the same major archive version 1 as mode 0 and mode 1. **Legacy MB4C v1 readers that only implemented modes 0 and 1 reject mode 2**; the updated decoder supports all three. This is a research format, not a finalized backward-compatible ecosystem protocol.

Build instructions remain unchanged:

    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 32 --iterations 18 --adaptive --out adaptive.mb4 --decoded adaptive.raw
    ./build/cpp-runtime/jarvisx-mandelbulb4 --decode-archive adaptive.mb4 --decoded independently_replayed.raw

Compare modes with real serialized byte counts and median of three process-local encode/decode timings:

    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 32 --fixture mandelbulb --benchmark
    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 32 --fixture uniform --benchmark
    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 32 --fixture noise --benchmark

Use --auto-codec to select the smallest archive among raw, reversible hierarchical residual coding and adaptive four-way partitioning. Default operation still selects between raw and the original lossless hierarchy. Results depend on field geometry; mode 2 can be less efficient than the original hierarchy for some inputs, hence --auto-codec.

Reported encoding and decoding timings exclude Mandelbulb field generation and process startup, vary by hardware/compiler and should not be treated as statistically robust system-wide benchmarks. Compression ratios are measured using actual full archive byte sizes including header and tree topology.

## Integration points

The component is registered under cpp_runtime/CMakeLists.txt and uses jarvisx_harden. Its standalone MB4C stream can be treated as a byte-volume input/output adapter by DM3D processors or existing CTR evaluators without silently mutating their formats. A path-scoped GitHub Actions workflow tests GCC and Clang.
