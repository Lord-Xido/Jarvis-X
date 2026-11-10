# Jarvis X — B3D1 cross-backend conformance contract

**Scope:** BitCube3D C++17 SDK ↔ Python standard-library reference ↔
independent Node.js golden oracle. Version: **B3D1 v1**.

This is a deliberately narrow interoperability boundary. It does **not**
claim that the separate `DM3DROM1` 64³-byte ROM format, neural latents,
PyTorch model weights, WebGPU textures or the full JavaScript application
are binary-equivalent to B3D1.

## Binary envelope (little endian)

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 4 bytes | ASCII magic `B3D1` |
| 4 | 4 bytes | Virtual cube side = `1,000,000,000` |
| 8 | 4 bytes | Outstanding reversible XOR rounds |
| 12 | 8 bytes | Number of active bricks (u64) |
| 20 | 524 × active bricks | Repeated sorted brick records |

Each 524-byte record stores:
- 12-byte `x,y,z` u32 little-endian **brick** coordinates,
- 512 bytes of packed `16×16×16` bits, x-fastest,
- bit index `i=(x%16)+16(y%16)+256(z%16)`, least-significant bit first.

Brick coordinates must be **strictly increasing in lexicographic (x,y,z)
order**. Repeated, out-of-range, unsorted or entirely zero bricks are invalid.
Unstored bricks represent all-zero data. The file is uncompressed and carries
**no cryptographic authentication or checksum**.

The total binary length is exactly:

```text
20 + active_brick_count * (12 + 512) bytes.
```

`10^27` is the **virtual bit address cardinality**; resident state and
executed work depend on the active brick set. Binary length may substantially
exceed the number of active bits, so B3D1 is not a compression claim.

## Reversible bit kernel

Forward passes (per brick): axes X→Y→Z, strides 1→2→4→8, apply:

```text
high ^= low  (GF(2))
```

Inverse passes: reverse axes and strides, applying the same XOR update.
The `rounds` field records how many forward passes remain to be undone.

All three implementations verify the same deterministic golden test vectors
in `native/bitcube3d/tests/fixtures/b3d1-v1.json`:
- empty snapshot,
- corner and brick-seam bits including 999,999,999³ corner,
- mixed coordinates with seven forward XOR rounds.

The fixture stores expected binary size, active brick count and SHA-256
digest. Digests are **test oracles**, not embedded authentication metadata.

## Operational acceptance

The Python implementation at `src/jarvisx/bitcube3d_b3d1.py`
has strict bounded parsing, writer canonicalization, binary round-trip and
explicit opt-in inverse GF(2) lifting. It limits transform generation to 32
rounds and limits active bricks to 100,000; parsing accepts metadata rounds up
to the native 10,000 but will not invert more than 32 in this reference.

The C++ reader enforces canonical coordinate ordering. Its CTest harness
performs **actual executable cross-language exchange**, not merely duplicate
in-language tests:

1. independently encode the fixture using Python and C++ CLI;
2. compare the *entire* B3D1 byte sequences and fixture SHA-256;
3. load the Python file via native C++, reserialize it, and compare bytes;
4. invert the native saved rounds and assert the source bits are recovered;
5. reject malformed bounds, ordering, duplicate coordinates and empty bricks
   in focused Python tests.

The Node.js script independently reproduces the bit layout, XOR transform,
binary envelope and reference digest. It runs in path-scoped CI without
importing either Python implementation or the native SDK.

### Commands

```bash
python -m pytest tests/test_bitcube3d_b3d1.py -q --no-cov
node tools/verify-b3d1-golden.mjs
cmake -S native/bitcube3d -B build/bitcube3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/bitcube3d -j
ctest --test-dir build/bitcube3d --output-on-failure
```

## Observational wall-clock benchmark (not a comparable kernel speedup)

`native/bitcube3d/tests/b3d1_benchmark.py` generates 512 occupied bit
positions in exactly eight `16^3` bricks, applies two rounds, independently
writes the B3D1 snapshot through Python and the native DSL/CLI, compares the
entire byte sequence, then inverts the file to verify that all source bits
are recovered exactly. It repeats execution after warm-up and emits a JSON
receipt recording full binary length, SHA-256, exactness, medians and observed
active input bits/second.

The two timing scopes are intentionally disclosed and **not equivalent**:
- Python time includes reference encoding and in-memory serialization.
- Native C++ CLI time includes process startup, DSL command interpretation,
  native encoding and file I/O.

CI publishes one observed JSON artifact per compiler. Reported numbers
depend on the GitHub runner and are not stable hardware benchmarks or evidence
of CPU/GPU speedup. No virtual-address cardinality is used as a count of
physically processed bits.

```bash
python native/bitcube3d/tests/b3d1_benchmark.py \
  build/bitcube3d/bitcube3d-cpp --output artifacts/b3d1-observed.json
```

## Explicit exclusions

The present work is **bitplane interoperability**, not a trained ANN,
cross-modal codec, production cloud distributed executor, performance win,
secure archive or physical `10^27`-bit machine.

Next engineering steps: define separately versioned `DM3DROM1`↔B3D1
conversion, add independent geometry/ANN quantization golden vectors,
measure hardware-specific native throughput and peak RSS, introduce
authenticated snapshot envelopes and enforce native allocation/time budgets
at the external execution boundary.

Reference: `docs/DM3D_SYSTEM_CONFORMANCE.md`.
