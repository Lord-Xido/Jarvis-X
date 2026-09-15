# Dr Moagi 3D 1000x Sparse Multimodal Engine

## Status

This document specifies the NumPy-accelerated sparse 3D reference engine in
`src/jarvisx/dr_moagi_3d_1000x.py`.

The engine targets a **1000x reduction in logical voxel work** relative to a
fully materialized `1000 x 1000 x 1000` pass. It does **not** claim that every
end-to-end workload runs 1000x faster in wall-clock time.

## Virtual geometry

The canonical address space is

```text
1000 x 1000 x 1000 = 1,000,000,000 logical voxels
```

with a 64-bit Cube64 control/state descriptor.

The reference runtime partitions this space into `10 x 10 x 10` tiles:

```text
100 x 100 x 100 = 1,000,000 virtual tiles
1 tile = 1,000 logical voxels
```

At the canonical active fraction

```text
alpha = 0.001
```

the resident set is

```text
1,000 active tiles
= 1,000,000 active logical voxels
```

and therefore the logical work ratio is

```text
1,000,000,000 / 1,000,000 = 1000x
```

This is the exact meaning of the architecture's **1000x work-reduction target**.

## Execution loop

```text
multimodal bytes
  -> deterministic descriptor
  -> sparse active tiles
  -> vectorized encoder
  -> latent state Z
  -> Omega temporal memory
  -> vectorized decoder
  -> reconstruction
  -> residual field
  -> residual-priority scheduling
  -> bounded correction
  -> recur
```

The dense billion-voxel field is never allocated.

## Multimodal boundary

The dependency-light front-end accepts arbitrary byte payloads, so text, code,
images, audio, video, 3D assets and binary sensor payloads share one deterministic
entry contract. The current front-end is a byte descriptor, not a claim that it
replaces specialized CNN, audio, video or point-cloud encoders. Those encoders can
be attached later without changing the sparse runtime contract.

## 64-bit Cube64 control word

The vectorized control-word layout is

```text
payload[16] | feature[12] | memory[8] | activation[8] | residual[12] | opcode[8]
```

Total: 64 bits.

This is the sparse control/state plane. The existing Q16.16x3 QVector engine
remains the higher-precision numerical field plane.

## Cloud scheduling

`cloud_plan(workers=N)` partitions the active tile identifiers across logical
workers. The design rule is:

```text
move compute to resident tiles; synchronize summaries and halos
```

The reference implementation produces the scheduling plan but does not pretend
that local NumPy execution is authenticated production multi-host cloud compute.

## Performance contract

There are two distinct acceleration quantities and they must not be conflated.

### 1. Sparse work reduction

For the canonical geometry:

```text
S_sparse = 1 / alpha = 1 / 0.001 = 1000x
```

This is an algorithmic work ratio.

### 2. Vectorization microbenchmark

`benchmark()` compares the same small matrix round trip implemented as scalar
Python loops and NumPy matrix multiplication. This measures kernel vectorization
on the current machine. It is a microbenchmark, not an end-to-end cloud result.

The function also reports

```text
effective_upper_bound = vectorized_speedup * sparse_work_reduction
```

only as a theoretical composition of independent effects. It is explicitly **not**
a measured total speedup.

## Verification invariants

Tests lock the following properties:

- the canonical geometry produces exactly a 1000x logical work ratio;
- the dense `1000^3` cube is not resident in memory;
- multimodal bytes execute through encode/Omega/decode/residual;
- every active tile appears exactly once in a cloud dispatch plan;
- Cube64 packing matches the documented 64-bit layout;
- byte descriptors are deterministic;
- telemetry never labels the 1000x work ratio as a guaranteed wall-clock speedup;
- malformed configuration is rejected.

## Installation

The base Jarvis-X runtime does not require NumPy. Install the acceleration backend
explicitly:

```bash
python -m pip install -e ".[accel]"
```

Development and test extras include NumPy so CI can exercise the accelerated path.

## Next performance stage

The next backend should preserve this contract while replacing NumPy kernels with
GPU/CUDA/Triton execution and measuring real end-to-end throughput, latency and
memory movement. The acceptance criterion is empirical:

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```

A production claim of 1000x wall-clock acceleration requires reproducible benchmark
evidence against a defined dense baseline on specified hardware.
