# JARVIS-X 1 TiB Sparse 3D Bytecode Streaming Reference

C++17, standard library only, deterministic lossless tile codec, configurable virtual range, file-streaming adapter, disk-backed verification, restart checks, and cloud shard-plan generator.

## Size and exact geometry

- 1 TiB = 1,099,511,627,776 bytes = 8,796,093,022,208 bits.
- 3D virtual dimensions = 8192 x 8192 x 16384 one-byte voxels.
- Tile = 128 x 128 x 64 voxels = 1 MiB.
- Tile grid = 64 x 64 x 256 = 1,048,576 tiles.
- 1 TiB = 1024 x 1 GiB groups = 1,048,576 x 1 MiB tiles.
- For **decimal 1 TB** = 1,000,000,000,000 bytes, there are 953,674 full 1 MiB tiles and a final 331,776-byte tile (953,675 tiles total). The file adapter supports this partial final tile.

## System boundary

This is a verifiable *streaming-bytecode and addressing subsystem*, **not** a trained ANN, a multimedia generative model, or a deployed cloud service. It composes with the prior JARVIS-X v2 multimedia ANN generator, but that model is not embedded in this codec. The existing GitHub Actions cloud check (PR #457) validated a separate native codec, not this terabyte extension. No 1 TiB physical data set was actually processed in this reference test.

## Format and mathematical invariant

For each voxel byte in row-major local order, the causal spatial predictor uses already decoded left, upper and back neighbours:

```
p_i = left_i XOR up_i XOR back_i
r_i = x_i XOR p_i
xhat_i = r_i XOR predict(xhat_<i)
```

By induction, xhat equals x byte-for-byte when the residual and predictor are correctly preserved. Encoded records use a fixed 32-byte little-endian header (`JXTB`, version 2), tile coordinates, original valid length, payload length, original-block CRC32 and payload CRC32. Per tile the encoder chooses ZERO, CONSTANT, run-length encoded XOR residual, or DIRECT bytes, whichever of the supported candidates it selects. DIRECT prevents expansion of an incompressible or already compressed media chunk beyond the 32-byte record overhead. The RLE decoder bounds output to 1 MiB per record.

**CRC32 provides accidental-corruption detection, not cryptographic authentication or encryption.** Production requires authenticated encryption, durable manifests, rollback after partial writes and robust access controls.

## Build

```
make all
make test
```

Dependencies: a C++17 compiler, make, Python 3.8+ for tests and shard planning. No neural network libraries are required.

## Virtual 1 TiB simulation, in bounded physical slices

```
./jx3d_1tib --start 0 --tiles 64 --out slice-0000.jxtb --corrupt-test
./jx3d_1tib --verify slice-0000.jxtb
```

This generates 64 MiB of synthetic 3D tiles, encodes them into a real disk stream, decodes them again and confirms every tile matches the deterministic original. Verify re-reads and checks it from disk. The default `--tiles` is 64 and does not silently run 1 TiB.

Restart the synthetic stream with a larger *target total* count. The existing file is checked before appending:

```
./jx3d_1tib --start 64 --tiles 4 --out slice.jxtb
./jx3d_1tib --start 64 --tiles 12 --out slice.jxtb --resume
./jx3d_1tib --verify slice.jxtb
```

`--resume` scans all existing records before appending, so its restart cost grows with accumulated output. Unexpected truncation, invalid CRC, out-of-order coordinates and mismatched start tile fail closed. The stream may remain partially written after an interrupted process; production systems should use staged writes and atomic promotion to durable storage.

For a 1 GiB local benchmark without writing an encoded output file:

```
./jx3d_1tib --start 0 --tiles 1024 --corrupt-test
```

The full virtual extent has 1,048,576 one-MiB tiles. To avoid hours of compute and extensive disk use, do **not** request all those tiles by default.

## Real multimedia file transport

The `jx3d_media` adapter works with *any* binary file, including images, audio, video, and text. It does not *understand* their semantics or generate new media content; it preserves bytes exactly.

```
./jx3d_media encode original.mp4 encoded.jxtb
./jx3d_media verify encoded.jxtb original.mp4
./jx3d_media decode encoded.jxtb recovered.mp4
cmp original.mp4 recovered.mp4
```

Arbitrary file lengths (including partial final tile) are preserved through the `valid_length` field. Empty input is rejected. Inputs and outputs must be different paths; existing output files are not overwritten without explicit external removal. For large files, the encoder uses several MiB of buffers, not entire-file reads.

## Cloud scheduling plan (not deployment)

A natural batch partition is 1024 groups x 1 GiB, each containing 1024 contiguous tile IDs. To generate a JSON Lines manifest for two groups and assign them to 32 worker slots:

```
python3 plan_shards.py --workers 32 --groups 2 --output shards.jsonl
```

For a real cloud deployment, distribute the manifest to a durable queue, run each group in an isolated bounded-memory worker, store outputs in cloud object storage, persist validated manifests and SHA-256 digests, and attach authentication and encryption. The generator creates the plan but does **not** launch any remote cloud jobs.

## Tested performance and limitations

Local Linux reference run: 1024 one-MiB synthetic 3D tiles (1 GiB), 11.5910 s, 88.3444 MiB/s, 8,284 KiB maximum resident memory, 391,024,128 encoded bytes, 63.5830% encoded-size reduction, exact roundtrip, and injected corruption rejected. The generator's zero/constant/gradient/noise distribution is deliberately compressible; percentages do not generalize to arbitrary 1 TiB multimedia data.

At the same throughput, a hypothetical 1 TiB physical pass would take about 3.30 hours serially. This **is not a measured 1 TiB run**. I/O, network transfer, storage saturation, parallel overhead and content entropy affect actual time.

11 local regression checks passed: restart equivalence, high 2^40 boundary, corrupt and truncated data rejection, real multiblock file with partial final tile, exact output SHA, existing-file protection and more. The five v2 sample media files also decoded exactly; they were stored with 32-byte per-file overhead because they were already efficiently encoded or too small to benefit from compression.

## Connected cloud validation

This subsystem is staged alongside the earlier JX3D cloud codec on PR #457 in `cloud_runtime/tib/`. The `.github/workflows/jx3d-1tib-cloud.yml` workflow builds the two native executables, runs all 11 bounded regression tests, checks two shard plan records and actually processes a 1 GiB synthetic fixture on a GitHub-hosted CPU. A full terabyte, production service, ANN training and multimedia generation are **not** implied by this gate.
