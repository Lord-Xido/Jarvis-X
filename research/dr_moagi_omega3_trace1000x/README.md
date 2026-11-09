# Dr Moagi Ω³ — trace-governed 3D recursive auto-codec (research)

This module is an **isolated C++17 research implementation** based on the 1,000 MB trace engine. It does not modify the existing Jarvis-X application, canonical engine modules, or published mathematical baselines.

## What it executes

An input 8-bit volume X(x,y,z) is synchronously updated with periodic boundaries:

```text
S = floor((2*C + X- + X+ + Y- + Y+ + Z- + Z+ + 4)/8)
X_next = floor(((256-g)*S + g*C + 128)/256), with 0 <= g <= 256.
```

The bounded bytecode program executes **LOAD → M6 → ENCODE → DECODE → VERIFY → BITTRACE → COMMIT → HALT**. The archive encoder selects raw, RLE, or XOR-X+RLE; a candidate must reconstruct exactly before commit. Each execution record includes bit-plane flip counts, chosen kernel, codec, state hashes and a SHA-256-chained journal. Chaining is an unkeyed integrity mechanism, not a signature.

## Exact 3D periodic specialization

If a whole volume is provably a repetition of a bounded tile and each tile dimension divides the corresponding world dimension, a translation-invariant stencil has the same periods. The engine computes on the tile, scales bit-change counts by the exact replication factor, and uses exact byte equality to detect cycles/fixed points before fast-forwarding. Nonperiodic inputs use the included dense streaming fallback. These are algorithmic shortcuts, **not hardware-speedup guarantees**.

Safety limits in this implementation: 4,096 tile voxels, 1,000,000,000 logical voxel bytes, up to 1,000,000 symbolic iterations; dense fallback is separately bounded.

## Build & test

Requires CMake 3.16+, C++17 and OpenSSL libcrypto development headers.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
bash examples/run_demo.sh ./build/moagi_trace1000x
```

The demo does **not** allocate a 1 GB raw file. To materialize output explicitly:

```bash
./build/moagi_trace1000x materialize demo_output/final.tile demo_output/final.raw
```

To check real raw input for periodicity:

```bash
./build/moagi_trace1000x fromraw input.raw 1000 1000 1000 4 4 4 64 1000000 out
./build/moagi_trace1000x replay out.seed.tile out.trace.csv out.final.tile
```

## Verification and performance context

Local (pre-publication) Release CMake build and CTest selftest passed on 2026-11-09. The originating run reported 109,210 assertions across its own test harness. On an especially compressible 1,000 MB checkerboard, the original dense implementation measured 20.62 s for a one-step pass; periodic classification plus output materialization measured 3.88 s (~5.3x). This comparison used different trace granularity and is not a controlled performance result for arbitrary data. A million logical checkerboard iterations were represented as four computed updates plus 999,996 exact skips. CI validates code correctness but does not repeat the 1 GB benchmark.

**Not implemented/claimed:** 1,000x physical file-to-file acceleration, trained ANN, autonomous code evolution, arbitrary lossless compression, cryptographically authenticated execution, or distributed exabyte-scale compute.

Provenance source SHA-256 (pre-publication working copies):
- `moagi_trace1000x.cpp`: `9b7f2a2ce1a0113e108ec1699f3e4003ed056ca5a103b097d33c6d4c1464df50`
- `moagi_bittrace.cpp`: `e4a36d8862c10211c5caeb3d85ab5dd8c440b40fce72bd979488d8ddd7921e20`

Code review and repository-hosted CI are required before any merge.
