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

## Bounded JX8K tile-to-VM adapter (reference only)

The isolated `tile_vm_adapter.py` is the **first typed ingestion bridge**, not a production VM opcode extension. It reuses the existing 64-bit bytecode instruction *layout* (opcode byte in bits 63..56 and reserved fields), but defines a distinct **JX8K-TILEVM-1** experimental opcode profile that the existing `cpp_runtime/dm3d_bytecode/dm3d_vm.cpp` does **not** recognize. No existing VM or control plane is modified.

Its fixed seven-word bytecode is:

```
LOAD_TILE (0x70) -> DECODE_2BIT (0x71) ->
MAP_FLOAT32 (0x72) -> REPACK_2BIT (0x73) ->
VERIFY_CTR (0x74) -> EMIT_PROVISIONAL (0x75) -> HALT (0xFF)
```

The interpreter executes 6 operations/tile plus one HALT, no arbitrary branches. It requires independent raw-reference verification **before** materializing VM tensors; maps 4,096 2-bit values exactly to a 4,096-element float32 tensor for one tile at a time; reverses the conversion to check exact bytes; records per-tile Morton coordinates, label histograms and local occupied-cell centroids. It rejects unknown opcodes, reordered or incomplete programs, nonzero reserved operands, malformed tiles, or resource-limit violations.

The maximum of 1,024 selected tiles, 2 MiB serialized packet and 16 bytecode words bounds interpreter work. For each tile, the transient tensor occupies 16 KiB, separate from packet/reference storage. A successful run emits only a non-committable observational `JX8K-TILEVM-1` receipt; external-world verification, ANN training and production authorization are explicitly **not evaluated**. No executable action is permitted to write production state.

```bash
python reference/dm3d_8k3_virtual/tile_vm_adapter.py \
  --report /tmp/report.json --encoded /tmp/tiles.jx8k \
  --raw /tmp/raw.bin --receipt /tmp/tile_vm_receipt.json \
  --program-out /tmp/jx8k-reference.bc
```

This is an **adapter-level execution**, not yet an integration with the legacy production DM3D VM. A subsequent change must define a typed provider/consumer ABI and authoritative transaction policy before any real VM tile opcode can be introduced.

### CTR proof-chain check

The standalone workflow now runs independent packet validation, the fixed-word typed VM adapter, then the optional `ctr_proposal.py --vm-receipt /tmp/tile_vm_receipt.json` evidence cross-check. The latter verifies identical raw/packet SHA-256 digests, instruction program hash, tile coordinates, 39-bit Morton addresses, 4,096-value float32 histogram totals, bounded execution-step counts, and the explicitly non-committable receipt flags. Any mismatch fails closed.

The output is an **observational candidate transaction only**. It does not call the legacy DM3D bytecode VM, modify the canonical control plane, or claim authoritative commit capability. The codec and bridge remain finite procedural reference implementations, not learned models.

## Native C++ sandbox transaction (review-first implementation)

The native C++17 native_tile_vm.cpp executable consumes JX8KT1 packets directly and validates the exact seven-word JX8K experimental bytecode profile. It stages an entire finite selected tile batch in a private in-memory std::map<uint64_t, RawTile>, retaining the pre-existing sandbox state. For every tile, it executes the logical LOAD_TILE → DECODE_2BIT → MAP_FLOAT32 → REPACK_2BIT → VERIFY_CTR → EMIT_PROVISIONAL chain. Only after all records pass format, duplicate coordinate, bitwise reconstruction, independent raw-reference and resource checks does one state_.swap(staged) publish the batch **inside the sandbox**. Any exception or injected failure prevents this swap.

The C++ command emits JX8K-NATIVE-TILEVM-1 observational telemetry: sandbox_committed, rolled_back, state_unchanged_after_rollback, per-tile Morton 39-bit addresses, 2-bit histograms, non-cryptographic FNV fingerprints and production_state_modified=false / commit_eligible=false.

The separate native_receipt_link.py independently decodes and checks actual serialized packets against raw data, replays the experimental Python typed tile VM and cross-checks tile coordinates, Morton keys, histogram counts and report fingerprints. It emits a JX8K-NATIVE-LINK-1 provisional result containing independent raw/packet SHA-256 evidence.

After creating the original /tmp fixture and a 56-byte experimental tile_vm.bc via the Python adapter, run:

    build/jx8k3/jarvisx_dm3d_8k3_native_vm --encoded /tmp/tiles.jx8k --raw /tmp/raw.bin --program /tmp/tile_vm.bc --receipt /tmp/native_receipt.json
    python reference/dm3d_8k3_virtual/native_receipt_link.py --native-receipt /tmp/native_receipt.json --report /tmp/report.json --encoded /tmp/tiles.jx8k --raw /tmp/raw.bin --out /tmp/native_link.json

Rollback test with injected failure after staging the last tile:

    build/jx8k3/jarvisx_dm3d_8k3_native_vm --encoded /tmp/tiles.jx8k --raw /tmp/raw.bin --force-reject-at 191

Injection exits 2 and reports state_unchanged_after_rollback=true, sandbox_committed=false, and one unchanged sentinel record. Success for the same 192-tile input yields instruction_steps=1153 and sandbox_records_after=193 (one sentinel plus 192 verified tiles).

**Boundary:** The native C++ consumer owns a sandbox only. This change does not modify cpp_runtime/dm3d_bytecode/dm3d_vm.cpp, alter canonical runtime authority, authenticate packets, guarantee durable transactional storage, or establish trained intelligence. Production commits remain prohibited. CMake and GCC/Clang workflows exercise native self-test, cross-language comparison, and malicious/incomplete input rejection.
