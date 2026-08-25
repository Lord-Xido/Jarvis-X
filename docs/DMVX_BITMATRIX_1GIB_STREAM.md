# DM-vOmegaXi+ 1 GiB Operational Bit-Wise 3D Stream

## Status

**Executable bounded streaming benchmark and lossless bit-transport reference.**

This subsystem operationalizes the 1 GiB kinetic processing proposal without treating illustrative performance numbers as facts. It measures the actual rate, compression ratio, reusable working set, chunk-mode decisions and exact reconstruction behavior on the machine that runs it.

It is a transport/benchmark layer beneath the existing DM-vOmegaXi+ Bit-Matrix autoencoder. It does not replace the learned ternary autoencoder and does not make a candidate state authoritative.

---

## 1. Exact 3D target volume

One GiB is

```text
1 GiB = 2^30 bytes = 2^33 bits = 8,589,934,592 bits
```

and

```text
2048^3 = 8,589,934,592.
```

Therefore the canonical full stream has an exact virtual geometry

```text
2048 x 2048 x 2048 bits.
```

The mapping is X-fastest:

```text
bit_index = z * 2048 * 2048 + y * 2048 + x
```

The runtime streams this virtual volume through reusable bounded chunks. It never needs to allocate a resident 1 GiB input buffer.

---

## 2. Five-phase operational interpretation

The kinetic radii remain a visualization mapping rather than physical distances inside CPU hardware:

```text
Ingestion      12.5 -> 8.0
Compression     8.0 -> 2.0
Core verify     2.0 -> 0.8
Buffer ring     0.8 -> 1.3
Reconstruction  1.3 -> 7.5
```

The executable phases are:

1. **Spatial ingestion** — deterministically materialize the current prefix/chunk of the 2048^3-bit field.
2. **Adaptive compression** — scan 64-bit words and encode zero/all-one runs; retain literal runs; fall back to raw mode when compression would expand the chunk.
3. **Fixed-point verification** — enforce exact `Decode(Encode(B)) == B` for every chunk.
4. **Reusable buffer ring** — reuse pre-reserved input, encoded and decoded buffers and report any hot-path capacity change as a reallocation.
5. **Outward reconstruction** — decode the chunk and compare every 64-bit word with the source before the stream advances.

This benchmark is deliberately lossless. The learned Bit-Matrix autoencoder remains a separate bounded-distortion subsystem.

---

## 3. Adaptive word codec

For each 64-bit word the codec classifies the stream into three token types:

```text
ZERO_RUN     repeated 0x0000000000000000
ONE_RUN      repeated 0xFFFFFFFFFFFFFFFF
LITERAL_RUN  all other words
```

Each run carries a 32-bit count. Literal runs carry their 64-bit words in little-endian order.

The encoder computes the candidate RLE form and compares its final byte count with a raw packet. If RLE is not smaller, the chunk is emitted in raw mode.

Therefore compression is data dependent:

```text
compression_ratio = raw_bytes / encoded_bytes
```

No fixed ratio such as 128x is assumed.

---

## 4. Codec fixed point

For each streamed chunk `B_k` the mandatory invariant is

```text
D(E(B_k)) = B_k.
```

The benchmark fails immediately on a mismatch. For the whole stream:

```text
codec_fixed_point = AND_k [D(E(B_k)) == B_k].
```

This is exact bit identity. It is not a claim that a lossy neural latent representation preserves all source bits.

---

## 5. 512-bit vector telemetry

A 512-bit vector contains 64 bytes. Therefore the full 1 GiB target contains exactly

```text
1,073,741,824 / 64 = 16,777,216
```

logical 512-bit vectors.

The reference backend currently executes portable 64-bit C++ operations. The number `16,777,216` is reported as a logical vector count only. It does not prove AVX-512 was selected by the compiler or executed by the CPU.

Architecture-specific AVX-512, AVX2, NEON and CUDA kernels remain separate optimization milestones.

---

## 6. 100 ms execution window

The runtime records the amount of raw source data completed by the first chunk boundary at or beyond the requested window:

```text
window_gbps = window_bytes * 8 / elapsed_window_seconds / 1e9.
```

The default requested window is 100 ms.

This is measured telemetry. The runtime does not hard-code `85.899 Gbps` or any other target throughput.

---

## 7. Reusable working set and allocation semantics

The hot loop owns three reusable buffers:

```text
input words
encoded bytes
decoded words
```

They are allocated/reserved before timed chunk processing. Capacity changes are counted as `hot_path_reallocations`.

A zero count means the benchmark's reusable vector capacities did not grow during the hot loop. It does **not** prove that the entire process, C++ runtime, operating system or allocator performed zero allocations.

The runtime reports

```text
reusable_working_set_bytes
```

from those reserved capacities.

---

## 8. L3 cache claim boundary

An optional `--l3-mib N` argument compares the reusable benchmark working set with a user-supplied L3 capacity:

```text
working_set_fits_configured_l3 = reusable_working_set_bytes <= configured_l3_bytes.
```

Even when true, the report does not claim zero DRAM traffic or zero memory latency. Cache placement, replacement, other processes, code/data footprints and hardware policy remain physical runtime effects.

---

## 9. Patterns

The deterministic generator supports:

- `sparse3d` — mostly zero words with coordinate-dependent sparse set bits;
- `checker3d` — structured zero/all-one regions;
- `zero` — all-zero upper-bound compression fixture;
- `random` — deterministic incompressible fixture expected to select raw passthrough.

This makes rate behavior falsifiable across very different source statistics.

---

## 10. Build and run

Build:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
```

Run the canonical full target:

```bash
./build/cpp-runtime/jarvisx-bitmatrix1gib \
  --bytes 1073741824 \
  --chunk-mib 8 \
  --pattern sparse3d \
  --window-ms 100 \
  --output-dir .jarvisx-bitmatrix1gib
```

Optionally compare the reusable working set with a known cache size:

```bash
./build/cpp-runtime/jarvisx-bitmatrix1gib \
  --bytes 1073741824 \
  --chunk-mib 8 \
  --pattern sparse3d \
  --window-ms 100 \
  --l3-mib 32
```

Generated artifacts:

- `metrics.csv` — per-chunk ingestion, compression, verification and decode timing;
- `report.txt` — aggregate rate, throughput, 100 ms window, working-set and invariant telemetry.

---

## 11. CI strategy

Normal C++ CI runs a bounded smoke workload so pull requests remain portable across GCC, Clang+ASan/UBSan and MSVC.

The dedicated `bitmatrix-1gib.yml` workflow runs a larger automated smoke benchmark on pushes/pull requests and exposes a manual `workflow_dispatch` path for the full 1 GiB target.

The full benchmark result is therefore an artifact to measure, not a number embedded in the framework definition.

---

## 12. Capability boundary

Implemented now:

- exact 2048^3-bit virtual addressing for a full 1 GiB stream;
- bounded chunked processing with no resident 1 GiB source allocation;
- deterministic structured/random 3D bit generation;
- adaptive zero/one-run + literal lossless encoding;
- raw fallback for incompressible data;
- exact codec fixed-point verification;
- logical 512-bit vector accounting;
- measured 100 ms-window telemetry;
- reusable working-set and hot-path reallocation telemetry;
- optional configured-L3 fit comparison;
- CLI, reports, CTest and cross-platform build integration.

Not claimed by this reference:

- actual AVX-512 execution;
- a fixed 8.362 MiB latent footprint;
- a fixed 85.899 Gbps transfer rate;
- a fixed 38.5x speedup;
- zero DRAM traffic;
- process-wide zero allocations;
- semantic equivalence inferred from bit identity;
- gamma=infinity as a measured hardware property;
- external state-of-the-art performance.

Those are either benchmark outcomes, future acceleration targets, or conceptual boundaries and must be reported separately.
