# Dr Moagi 1,000,000 Lines/Second Code Generation Plane

## Objective

The generation plane adds a measurable high-throughput source-emission runtime to Jarvis-X with a default target of **1,000,000 physical source lines per second**.

The target is empirical. It is not declared as a hardware-independent guarantee, and it is not used as a synonym for one million lines per second of novel model reasoning. The engine separates:

1. **semantic synthesis** — deciding what code should mean;
2. **IR/template compilation** — reducing that intent to deterministic source templates;
3. **mechanical emission** — materializing physical source lines into memory, files, or parallel shards;
4. **verification** — line counts, SHA-256 digests, syntax checks, provenance, and throughput telemetry.

The one-million-LPS target applies to stage 3 and is measured directly at runtime.

## Relationship to the 3D animation codec

The canonical 1,000-line Dr Moagi 3D animation autoencoder/decoder remains immutable. The codegen plane imports its expected SHA-256 and places that digest into every metrics record and shard manifest.

This keeps the architecture layered:

```text
canonical 3D AE/AD runtime
        |
        v
verified provenance SHA
        |
        v
semantic/IR template
        |
        v
adaptive codegen compiler
        |
        +--> repeat strategy --------> maximum emission ceiling
        |
        +--> indexed strategy -------> globally indexed expansion
        |
        v
bounded chunks / concurrent shards
        |
        v
file | memory | null benchmark sink
        |
        v
SHA-256 + LOC/s + MB/s + target ratio
```

## Two throughput classes

### Repeat mode

If the template does not contain `{index}`, one validated source line is compiled to UTF-8 once and expanded by byte multiplication. Full chunks are cached and reused. This isolates allocator, hashing, and I/O throughput and is the mode used for the 1,000,000-LPS performance gate.

Example:

```bash
jarvisx-dr-moagi-codegen benchmark \
  --lines 1000000 \
  --target-lps 1000000 \
  --template "pass  # DM-vOmegaXi scaffold" \
  --require-target
```

### Indexed mode

If the template contains `{index}`, every emitted line receives a global logical index. This is more representative of parameterized source generation but incurs formatting cost.

```bash
jarvisx-dr-moagi-codegen generate \
  --lines 1000000 \
  --template "dm_generated_{index} = {index}" \
  --output generated.py
```

Metrics identify the strategy explicitly so a fast repeated-line result cannot be misreported as one million unique semantic programs per second.

## Adaptive optimization

The autotuner measures candidate chunk sizes on the current machine and selects the highest-throughput configuration:

```bash
jarvisx-dr-moagi-codegen autotune --lines 1000000
```

Default candidate chunk sizes are 4,096, 16,384, 65,536, and 262,144 lines. The selected chunk size can then be passed to file or shard generation.

The optimization loop is bounded and measurable; it does not rewrite its own source code or silently alter correctness constraints.

## Parallel shard generation

For large generated systems, independent modules can be emitted concurrently:

```bash
jarvisx-dr-moagi-codegen shard \
  --lines 1000000 \
  --workers 8 \
  --chunk-lines 65536 \
  --template "dm_generated_{index} = {index}" \
  --directory build/generated
```

The output directory contains `shard-XXXX.py` files plus `manifest.json`. The manifest records total lines, bytes, elapsed time, aggregate LOC/s, target ratio, worker count, chunk size, strategy, canonical codec provenance, and a SHA-256 digest for every shard.

## Performance equation

For total generated physical lines `N`, bytes `B`, wall-clock generation interval `Delta t`, and target throughput `T = 10^6` lines/s:

```text
R_LOC = N / Delta t
R_MB  = B / (10^6 Delta t)
eta_T = R_LOC / T
PASS  = eta_T >= 1
```

No performance claim should be published without the measured `elapsed_seconds`, `lines_per_second`, strategy, sink type, hardware context, and whether hashing/file I/O were included.

## Design constraints

- Exact line count is part of correctness.
- Templates must describe exactly one physical source line.
- A syntax sample is compiled before emission.
- SHA-256 is enabled by default.
- File generation uses bounded chunks instead of constructing a million-line Python string at once.
- Single-file output is sequential to avoid synchronization corruption.
- Parallelism is applied to independent shards.
- The 1,000,000-LPS CI gate uses repeat-mode mechanical emission; indexed generation is tested for correctness without claiming the same rate.

## CLI

```text
jarvisx-dr-moagi-codegen benchmark
jarvisx-dr-moagi-codegen generate
jarvisx-dr-moagi-codegen autotune
jarvisx-dr-moagi-codegen shard
```

The engine is therefore a high-throughput deterministic compiler/emitter around the existing Dr Moagi runtime, with the one-million-lines-per-second figure promoted from a narrative claim to a benchmarkable systems target.
