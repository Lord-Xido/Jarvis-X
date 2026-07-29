# Jarvis X Inward Autopoietic Runtime

This C++17 runtime turns the 3D 8K³ auto-encoding processor inward onto its own executable image.

It preserves a virtual `8192 × 8192 × 8192` coordinate space through sparse `8 × 8 × 8` tiles, then executes a bounded self-optimization cycle:

1. ingest its own executable by default;
2. extract fixed-width multimodal features;
3. encode them into the 3-bit set `{-4,-3,-2,-1,0,1,2,3}`;
4. scatter and diffuse the latent field through the sparse 3D lattice;
5. decode and calculate reconstruction error;
6. generate candidate runtime genomes and bytecode schedules;
7. evaluate every candidate in a fresh sandboxed processor instance;
8. apply the Lambda coherence gate;
9. commit an improvement or roll back;
10. checkpoint the genome, ROM, and CSV evolution journal.

The runtime evolves auditable parameters and bytecode schedules. It does **not** rewrite arbitrary native machine code or claim consciousness.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime
cmake --build build/cpp-runtime --config Release
```

Or compile directly:

```bash
g++ -std=c++17 -O3 -pthread \
  cpp_runtime/src/jarvis_x_autopoietic_runtime.cpp \
  -o jarvisx-runtime
```

## Run inward on the executable

```bash
./build/cpp-runtime/jarvisx-runtime \
  --generations 8 \
  --population 6
```

## Run on another multimodal or binary input

```bash
./build/cpp-runtime/jarvisx-runtime \
  --file sample.bin \
  --generations 12 \
  --population 8
```

## State artifacts

The default `.jarvisx-runtime/` directory contains:

- `genome.current` — atomically committed runtime genome;
- `runtime.rom` — big-endian 64-bit bytecode words;
- `evolution.csv` — accepted and rolled-back generations.

Use `--reset` to discard a previous checkpoint, `--state-dir PATH` to isolate experiments, and `--min-improvement X` to tighten the commit criterion.

## Validation

```bash
ctest --test-dir build/cpp-runtime --output-on-failure
```
