# JX3D 1 TiB Reference — execution validation, 2026-11-04

## Executed locally

Build: `g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic` for both C++ executables. All 11 regression tests completed successfully (see `regression.log`). Real PNG, GIF, MP4, WAV and UTF-8 sample media were encoded, independently decoded and checked with SHA-256 against the original files (see `media_validation.json`).

### Full physically processed 1 GiB benchmark

Command: `./jx3d_1tib --start 0 --tiles 1024 --corrupt-test`

```
start_tile=0 corruption_test=PASS
tiles=1024 raw_bytes=1073741824 encoded_bytes=391024128
formats_zero=256 constant=256 rle=256 raw=0 direct=256
storage_reduction_percent=63.5830
elapsed_s=11.5910 input_MiB_per_s=88.3444
maxrss_kib=8284
verified_reconstruction=PASS
```

Raw bytes = 1,073,741,824. CRC verified for both payload and reconstructed original tile. Each tile was compared with its generated input bytes; no reconstruction errors. No full 1 TiB allocation and no full 1 TiB physical pass were made.

### 1 TiB high coordinate boundary

Tested logical tiles 1,048,574 and 1,048,575; decoding and re-read verification passed. A request starting at 1,048,575 for two tiles was rejected as out of bounds.

### File adapter roundtrip

Tested an arbitrary 3,145,765-byte input made from zero, constant and pseudorandom data, including a 37-byte final partial tile. Restored file SHA-256 matched. Deliberately altered encoded bytes failed validation.

### Cloud scope

No 1 TiB cloud worker ran. The `plan_shards.py` component only produces a deterministic allocation manifest; this new codec is not deployed to a cloud service. JARVIS-X GitHub PR #457 previously verified a *different, smaller* C++ cloud reference implementation.

### Security scope

CRC32 is accidental-integrity checking, not tamper-resistant proof or encryption. Untrusted production streams require cryptographic authenticated envelopes, strict permissions, quotas and storage lifecycle controls.