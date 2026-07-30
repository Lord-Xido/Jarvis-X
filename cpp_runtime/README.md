# Jarvis-X Inward C++ Runtime

This dependency-free C++17 subsystem implements a bounded sparse auto-encoding processor and deterministic parameter/schedule search loop.

It exposes a virtual `8192 × 8192 × 8192` coordinate domain through lazily materialized `8 × 8 × 8` tiles. The virtual extent is an addressing contract, not a dense allocation.

## Operational cycle

1. ingest the executable image by default, or accept explicit text/binary input;
2. extract a fixed-width deterministic feature vector;
3. encode into the signed 3-bit set `{-4,-3,-2,-1,0,1,2,3}`;
4. scatter and diffuse the latent field through sparse coordinates;
5. decode and calculate reconstruction error;
6. generate bounded genome and bytecode-schedule candidates;
7. evaluate each candidate in a fresh processor instance;
8. apply the Lambda coherence and improvement gates;
9. commit the champion or retain the rollback anchor;
10. persist the genome, ROM and evolution journal.

The runtime mutates constrained parameters and synthesized bytecode schedules. It does **not** rewrite arbitrary native instructions, establish consciousness, provide hostile-code isolation or physically allocate the full virtual lattice.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
ctest --test-dir build/cpp-runtime -C Release --output-on-failure
```

Direct GCC/Clang build:

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic \
  -Icpp_runtime/include \
  cpp_runtime/src/main.cpp \
  -o jarvisx-runtime
```

## Run inward on the executable

```bash
./build/cpp-runtime/jarvisx-runtime \
  --generations 8 \
  --population 6
```

## Run on explicit input

```bash
./build/cpp-runtime/jarvisx-runtime \
  --file sample.bin \
  --generations 12 \
  --population 8
```

Text input is also supported:

```bash
./build/cpp-runtime/jarvisx-runtime \
  --text "deterministic replay fixture" \
  --generations 4 \
  --population 5
```

## DM-GEOM mechanical latent engine

`dm-geom` is an isolated mesh auto-encoding and decoding executable built on the same hardened C++17 profile. It:

1. extracts nine finite geometric features from a validated triangle mesh;
2. encodes mean radius and radial disorder into a bounded three-dimensional latent state;
3. decodes a fixed-topology sphere whose radius, deformation amplitude and deformation frequency vary continuously;
4. evaluates position, radius and surface-area reconstruction error;
5. applies central finite-difference gradient descent with gradient clipping and latent projection;
6. exports deterministic little-endian binary STL artifacts.

The fixed topology is intentional: it prevents the zero numerical gradient produced when loss depends only on an integer segment or vertex count.

Run the demonstration:

```bash
./build/cpp-runtime/dm-geom
```

The executable writes `recon.stl` and `optimized.stl` in the working directory and returns a non-zero status if evolution does not improve reconstruction loss.

## State artifacts

The default `.jarvisx-runtime/` directory contains:

- `genome.current` — atomically committed runtime genome;
- `runtime.rom` — big-endian 64-bit bytecode words;
- `evolution.csv` — accepted and rolled-back generations with telemetry.

Use `--reset` to discard an earlier checkpoint, `--state-dir PATH` to isolate an experiment and `--min-improvement X` to tighten the commit threshold.

## Determinism contract

Candidate generation, feature extraction, encoding, decoding and fitness selection are deterministic for the same input and genome. Wall-clock latency is recorded as telemetry but excluded from fitness. Platform floating-point implementations may still produce small cross-architecture differences; bit-exact portability is not claimed.

## Validation

The CTest suite includes:

- inward executable/text smoke execution;
- genome normalization before allocation;
- repeatable processor evaluation;
- proof that wall-clock latency does not alter deterministic fitness;
- DM-GEOM topology and non-degeneracy checks;
- nine-feature population and finite-value checks;
- continuous-loss evolution regression;
- decoder singularity regression at latent `x = -2`.

Optional sanitizer build:

```bash
cmake -S cpp_runtime -B build/cpp-runtime-san \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVISX_ENABLE_SANITIZERS=ON
cmake --build build/cpp-runtime-san --parallel
ctest --test-dir build/cpp-runtime-san --output-on-failure
```
