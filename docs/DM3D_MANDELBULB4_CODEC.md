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

An input volume must contain exactly side^3 bytes. The first axis (x) is the fastest-varying array index. The CLI currently supports power-of-two sides 4, 8, 16, 32, 64, and 128; it intentionally materializes a bounded grid. Fractal iterations are restricted to 2..64.

## Archive contract

17-byte little-endian header, followed by either raw data or the hierarchical residual stream:

- ASCII magic MB4C (4 bytes)
- Version 1 (1 byte)
- Mode 0 raw or 1 hierarchical (1 byte)
- Side uint16 LE (2 bytes)
- Depth uint8 (1 byte)
- FNV-1a 64-bit checksum of uncompressed voxel bytes (8 bytes)

Hierarchical mode serializes the final coarse state and all detail levels in reverse order, reconstructible using the side and fixed XYZ stage schedule. Sequence records use token 0 + unsigned LEB128 positive zero count, or token 1 + zigzag unsigned LEB128 signed value. The decoder checks lengths, tokens, dimensions, overflow, trailing bytes, voxel range and checksum. See the separate mode-3 section for lossy envelope semantics. FNV-1a is for accidental corruption detection, **not cryptographic authentication**. The archive is not yet a stable cross-project wire protocol.

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

## Reproducible CI benchmark evidence (2026-10-18)

GitHub Actions run: https://github.com/Lord-Xido/Jarvis-X/actions/runs/39108158152

The dedicated workflow successfully compiled with GCC and Clang, passed all 2 CTests in each toolchain, decoded the original and adaptive archives in separate process invocations, and binary-compared decoded outputs. The example input was power-8 Mandelbulb escape-time data, 18 iterations, 8 bits per sample, and a 17-byte archive header.

| 3D side | Input bytes | Raw archive bytes | Legacy 4:1 bytes | Adaptive bytes | Adaptive / source ratio | Adaptive size improvement vs legacy |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32,768 | 32,785 | 12,717 | 9,748 | 3.3615:1 | 23.35% |
| 64 | 262,144 | 262,161 | 67,353 | 53,362 | 4.9126:1 | 20.77% |

Both compilers produced the same serialized sizes and byte-perfect reconstruction for both volumes. The adaptive 32^3 example used 1,813 tree nodes (453 splits, 766 constant leaves, 594 literal leaves). The 64^3 example used 14,557 nodes (3,639 splits, 7,172 constant leaves, 3,746 literal leaves). These are workload-specific, **not general compression guarantees**.

Control fixtures demonstrate the necessity of fallback:
- 32^3 uniform field: adaptive 19 bytes vs legacy hierarchical 42 bytes (high ratios arise because all voxels share one value).
- 32^3 pseudo-random field: 32,786 adaptive bytes vs 32,785 raw bytes, so --auto-codec selected raw storage.
- --benchmark gives single-process three-run median microsecond samples, not a rigorous independent machine-performance evaluation.


## Sampled-field error-bounded 3D sparse representation (mode 3)

An additional, opt-in MB4C v1 archive mode 3 stores a cost-selected four-way tree, with a user-declared **maximum absolute reconstruction error in 8-bit sample units**.

    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 128 --fixture mandelbulb --bounded-error 2 --out mb128-bounded.mb4 --decoded mb128-reconstruction.raw
    ./build/cpp-runtime/jarvisx-mandelbulb4 --decode-archive mb128-bounded.mb4 --decoded mb128-replay.raw
    ./build/cpp-runtime/jarvisx-mandelbulb4 --side 128 --fixture mandelbulb --bounded-error 2 --benchmark

For each nonconstant spatial region, the encoder **exhaustively scans each uint8 source sample in that region** and computes its exact minimum and maximum. It chooses the midpoint representative floor((min+max)/2) only when

    max(rep-min, max-rep) <= tolerance.

The tree evaluates constant leaves (2 bytes), literal leaves (1 + voxel_count bytes), bounded-envelope leaves (4 bytes) and optional recursive four-child subdivisions; only actual serialized size improves the selection. Encoded mode-3 leaves use tag 3, midpoint, original sampled minimum and sampled maximum. A node tag 0 remains losslessly constant; tag 1 remains a literal; tag 2 remains a four-child split. The decoder checks range ordering, representative containment and the declared envelope tolerance before filling approximated voxels.

The 17-byte common header remains unchanged in layout. Mode 3 appends to its payload a one-byte tolerance (0..64) and a little-endian uint64 source FNV-1a hash before the tree. For mode 3, the **header checksum covers the reconstructed data**; the extra source checksum is a provenance fingerprint only, because the original samples are not available to the decoder. FNV-1a is neither authentication nor collision-resistant cryptographic provenance.

After encoding, the source-aware CTR verifier performs an independent entire-volume comparison of the reconstructed voxels and the true sampled input. This establishes the finite-array guarantee:

    max_i |source[i] - decode(encode(source))[i]| <= tolerance

The decoder alone validates archive integrity and its asserted per-leaf envelope but **cannot independently prove the original input samples lay in those claimed envelopes** without possessing the original source. This is NOT a proof of Mandelbulb continuous geometry between sample locations, Hausdorff distance, certified raymarch step sizes or interval-arithmetic fractal set enclosure. Further certified continuous-field geometry requires interval methods or validated distance bounds and separate tests.

For exact mode 3, set \`--bounded-error 0\`, yielding maximum error zero. Existing lossless modes 0, 1 and 2 retain their original wire behavior; old MB4C readers that know only earlier modes will reject mode 3.

The runtime now permits bounded power-of-two sides 4, 8, 16, 32, 64 and **128** (128 cubed = 2,097,152 sampled bytes). The encoder still materializes the source volume in memory: **"sparse" describes the encoded representation and selective subdivision, not a low-memory streaming generator**. No multi-gigabyte streaming or general infinite-resolution claim is made.

## Integration points

The component is registered under cpp_runtime/CMakeLists.txt and uses jarvisx_harden. Its standalone MB4C stream can be treated as a byte-volume input/output adapter by DM3D processors or existing CTR evaluators without silently mutating their formats. A path-scoped GitHub Actions workflow tests GCC and Clang.
