# DM3D MP4 → 3D → Bytecode → CTR Reference (JXMP4T1)

**Scope:** Isolated C++17 research reference for Jarvis-X; **not** a production kernel, learned ANN, secure media parser or general video codec. This does **not** change any canonical VM opcode, state, binary format or deployment. Designed as a review-first interoperability milestone following ADR-016, ADR-017 and the October 2026 system-wide audit.

## Processing contract

```text
Arbitrary media file bytes (including MP4 with embedded UUID boxes)
  -> streaming tiles (up to 1024 bytes each; zero padded for VM)
  -> adaptive lossless byte codec (raw / RLE / cumulative-XOR / XOR+RLE)
  -> versioned JXMP4T1 transport archive
  -> independent decode and per-tile/global CRC32 verification
  -> 1 MiB bounded bytecode-VM arena, periodic 8×8×16 field
  -> OBSERVE_LAPLACE6 → XOR_A5 → REFLECT_X → REFLECT_X → XOR_A5 → ASSERT_IDENTICAL → HALT
  -> exact source-to-output byte comparison (in `roundtrip`)
  -> atomic rename of staged output, or rollback
  -> Python independent decoder and SHA-256 receipt
```

The VM's `OBSERVE_LAPLACE6` computes the absolute periodic 3D six-neighbor stencil sum without modifying authoritative input bytes. The other instructions are invertible demo transformations whose verified composition is identity. This is finite, explicitly bounded bytecode execution rather than autonomous neural inference.

### Logical geometry vs physical memory

- Geometry: `(10×1024×1024)^3` **logical byte-coordinate addresses**; no dense cube allocation.
- Virtual tile edge: `8×8×16` voxels (1024 bytes, one byte per voxel).
- Tile address mapping: `x=8*(tile_id mod (axis/8))`, `y=8*(floor(tile_id/(axis/8)) mod (axis/8))`, `z=16*floor(tile_id/((axis/8)^2))`.
- VM arena: **1 MiB resident**. It operates on one padded tile at a time; input files stream from disk.
- Max file per invocation: **128 MiB** by design (a resource safety budget, not a claim of geometrical capacity).

### Wire format v1 (little endian)

Header (28 bytes):

| Offset | Type | Field |
|---|---|---|
| 0 | 8 bytes | magic `JXMP4T1\0` |
| 8 | uint16 | version `1` |
| 10 | uint16 | flags `0` |
| 12 | uint64 | original byte count |
| 20 | uint32 | CRC32 of original file |
| 24 | uint32 | number of tiles |

Each tile: 16-byte record header `uint32 tile_index`, `uint32 tile_crc32`, `uint16 raw_len`, `uint16 encoded_len`, `uint8 mode`, `uint8[3] reserved`, followed by `encoded_len` bytes. Mode 0 raw, 1 RLE `(count:uint8,value:uint8)` pairs, 2 cumulative-XOR predictive, 3 predictive + RLE. Encoded mode selection minimizes payload length with raw as tie-break. Input data are processed in **byte order**; byte indices are geometrically addressable in the `8×8×16` coordinate field. This is not claimed to be Morton-order compatible with other Jarvis-X codecs.

**Versioning:** This distinct `JXMP4T1` profile does not reinterpret `JX3DVM1`, `JX8KT1`, or the earlier embedded-MP4 `uuid` payload contract. It processes the *entire* MP4 file as opaque bytes, preserving all standard and private boxes. It does not extract or execute the UUID payload; use the existing `dm3d_mp4` utility for that separately. It is not a video-frame decoder or an MP4 player.

## Build and run

```sh
cmake -S reference/dm3d_mp4_ctr -B build/dm3d-mp4-ctr -DCMAKE_BUILD_TYPE=Release
cmake --build build/dm3d-mp4-ctr --parallel
ctest --test-dir build/dm3d-mp4-ctr --output-on-failure

# Run end to end; source and reconstructed file MUST be byte-identical before output publish
./build/dm3d-mp4-ctr/dm3d-mp4-ctr roundtrip movie.mp4 movie.jxdm3 recovered.mp4
python reference/dm3d_mp4_ctr/independent_verify.py movie.jxdm3 --expected movie.mp4 --receipt movie.receipt.json
```

`encode INPUT ARCHIVE` and `decode ARCHIVE OUTPUT` can also be used separately, along with `selftest`. The `decode` mode returns `crc_verified_decode` rather than `verified_commit` because no external original is supplied. `roundtrip` emits `verified_commit` only after full source byte comparison while output remains staged.

**Integrity boundary:** CRC32 detects accidental corruption, not malicious forgery. SHA-256 from `independent_verify.py` is an unkeyed digest, not an identity signature or authentication. Do not execute an untrusted embedded payload. All outputs are written to a freshly created staging directory and renamed only after verified processing; existing output paths are never overwritten. This is a reference contract, not a hardened service for adversarial users.

## Validation and limitations

- Standalone native CTest and independent Python parser; zero, repeated, incremental, random >1 MiB and MP4-shaped fixtures.
- Corrupted header/payload, truncation and trailing bytes must fail, leaving no published output.
- Optional `DM3D_MP4_TEST_FILE=/path/to/real.mp4` adds a real-media byte-for-byte round trip test.
- No physical allocation for virtual `10 MiB × 10 MiB × 10 MiB` extent. No compression guarantee on incompressible media, multimedia understanding, FFmpeg decoding, trained ANN, privilege isolation or deployed GPU execution.
- New standalone wire format is **provisional** and **non-authoritative** under ADR-016 until interface review, security review and wider integration tests pass.
