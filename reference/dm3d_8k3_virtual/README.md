# JARVIS-X DM3D 8K³ verified virtual tile reference

**Review-first, non-authoritative reference** for GitHub issue #390. No existing production VM, entrypoint, or control-plane module is modified.

## Build

From repository root, run the following commands independently:

    cmake -S reference/dm3d_8k3_virtual -B build/jx8k3 -DCMAKE_BUILD_TYPE=Release
    cmake --build build/jx8k3 --parallel
    ctest --test-dir build/jx8k3 --output-on-failure
    build/jx8k3/jarvisx_dm3d_8k3_virtual --rounds 3 --candidates 256 --budget 64 --seed 42 --report /tmp/report.json --tiles /tmp/raw.bin --encoded /tmp/tiles.jx8k
    python reference/dm3d_8k3_virtual/verify_packet.py --report /tmp/report.json --raw /tmp/raw.bin --encoded /tmp/tiles.jx8k --receipt /tmp/receipt.json
    python reference/dm3d_8k3_virtual/ctr_proposal.py --receipt /tmp/receipt.json --report /tmp/report.json --out /tmp/proposal.json

## Operational contract

The logical field is 8192³ = 2³⁹ addressable voxels with 2 bits/voxel. Global XYZ use 13 bits/axis interleaved into 39-bit Morton indices. Only selected 16³ tiles are materialized, each containing 4096 voxel values or 1024 bytes. The tile address grid is 512³. Octree path indexing is not the four-survivor fractal geometry of the existing Python module. Septillion/base-1024 exponent towers remain symbolic metadata.

Strict physical bounds: 1..16 rounds, 1..4096 candidates/round, 1..256 selections/round, selections <= candidates, and rounds×selections <= 1024.

## Versioned JX8KT1 binary packet

8-byte magic: JX8KT1 followed by two NUL bytes. Subsequent records are tile X, Y, Z (3 × little-endian unsigned 16-bit integers), payload length (little-endian unsigned 16-bit integer), and the payload.

Codec mode 0: mode byte plus exactly 1024 raw bytes. Modes 2/4/8/16: block width, unsigned 16-bit correction count, packed 2-bit majority predictors, followed by correction entries (local 16-bit XYZ-flattened index + unsigned 8-bit state). Value range 0..3. Four voxel values are packed per byte, least significant two bits first; X fastest. Independent decoder rejects duplicate/out-of-range corrections, malformed records, duplicate tile coordinates and trailing bytes.

## Interoperation and authority boundary

The existing 4000³ VME uses 12-bit/axis packed words and cannot represent an 8192³ address by reinterpretation. Existing 8³ RGB8 septillion tiles are not wire-compatible. No DM3D bytecode OPEN_TILE opcode or producer/consumer ABI is introduced.

The independent Python validator emits a JX8K-CTR-1 evidence receipt with SHA-256 of actual serialized files. The ADR-016-shaped proposal adapter marks authority PROVISIONAL and commit_eligible false, with unassessed external evidence, neural model generalization and latent convergence. It does not mutate production state or grant commit authority. SHA-256 is a digest, not a signature; FNV is not cryptographic.

## Sample-specific fixture, not hardware throughput

At seed 42 and 3×64 selected tiles: 196,608 raw bytes, 90,600 codec payload bytes, 92,144 bytes including stream/header overhead, 192/192 exact bitwise round trips. Raw SHA256: 96cd4f53f3e83a7e20ad2f1f57b09e3625fb6021bdeffb8313c42d9c39eb1c56. Packet SHA256: cb3a4cc9254bd9f93971007a5437629e2b4d580e027716094788fabe395d327b.

Compression ratio 2.1337× is valid only for this selected procedural dataset. No trained cognition, actual multimodal inference, 1000× speedup, dense 8K³ allocation, or astronomical computation is asserted.

For the production bridge: define versioned typed VM tile instructions, bounded scheduling and memory, verified atomic commit/rollback, cryptographic packet authentication when appropriate, multimodal fidelity tests, and actual end-to-end benchmarks.
