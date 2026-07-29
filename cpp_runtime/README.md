# Jarvis-X Inward C++ Runtime

This C++17 subsystem implements a bounded sparse auto-encoding processor, deterministic parameter/schedule search loop, a trainable dense 3D convolutional autoencoder reference and a transactional 4D multimodal composition runtime. The numerical components are dependency-free; optional OpenGL/GLUT targets render the actual ANN tensor state interactively.

The sparse processor exposes a virtual `8192 × 8192 × 8192` coordinate domain through lazily materialized `8 × 8 × 8` tiles. The virtual extent is an addressing contract, not a dense allocation.

## Sparse processor operational cycle

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

OpenGL and GLUT/freeglut are detected automatically. When either dependency is absent, CMake skips only the graphics targets; the headless runtimes and regression suite still build. Use `-DJARVISX_BUILD_GL_VISUALIZER=OFF` to disable detection explicitly.

Direct GCC/Clang build for the sparse processor:

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

## Train the 3D ANN autoencoder core

```bash
./build/cpp-runtime/jarvisx-autoencoder3d \
  --edge 8 \
  --channels 4 \
  --epochs 250 \
  --pattern sphere \
  --quantized \
  --export-dir .jarvisx-autoencoder3d
```

The 3D core implements a real trainable encode → latent → decode path with shared `3×3×3` kernels, stride-two spatial compression, direct reverse-mode gradients, gradient clipping, L2 regularization and optional signed 3-bit final inference.

Its output directory contains:

- `metrics.csv` — per-step MSE, MAE, maximum error, latent energy and gradient norm;
- `input.obj` — input voxel point cloud;
- `latent.obj` — channel-stacked latent point cloud;
- `reconstruction.obj` — reconstructed voxel point cloud;
- `model.jx3d` — reloadable model checkpoint.

## Run the interactive 3D engine

```bash
./build/cpp-runtime/jarvisx-autoencoder3d-gl \
  --edge 16 \
  --channels 4 \
  --pattern sphere \
  --learning-rate 0.015
```

This target evolves the original GLUT voxel demonstration into a live visualization of the actual model state:

- cyan voxels are the input tensor;
- blue/violet voxels are the multichannel latent field;
- pink voxels are the decoder reconstruction;
- gold voxels are the absolute reconstruction residual;
- cyan, pink and gold streams trace encode, decode and residual-learning flow;
- the HUD reports real MSE, MAE, latent energy, gradient norm and training step.

Core controls are `T`/Space for training, `Q` for Q3 inference, `P` for pattern changes, `0-4` for tensor views, `S`/`L` for checkpoints, mouse drag for rotation and mouse wheel for zoom.

Supported deterministic fixtures are `sphere`, `shell`, `checker`, `wave` and `noise`. See [`docs/CPP_3D_AUTOENCODER.md`](../docs/CPP_3D_AUTOENCODER.md) for equations, interactive controls, persistence and complexity boundaries.

## Run the self-optimizing 4D multimodal core

```bash
./build/cpp-runtime/jarvisx-multimedia4d \
  --edge 8 \
  --channels 4 \
  --temporal-depth 8 \
  --proposal-steps 2 \
  --cycles 120 \
  --learning-rate 0.03 \
  --temporal-decay 0.72 \
  --quantized \
  --output-dir .jarvisx-multimedia4d
```

The 4D runtime composes four independent 3D ANN models for deterministic visual, audio, text and generic fixtures. Each modality retains a bounded latent-history axis, receives an exponentially fused temporal latent and participates in a deterministic error-priority scheduler.

Every proposed update is evaluated in an isolated model copy. It commits only when reconstruction MSE remains within the configured acceptance gate; otherwise the authoritative model is retained and the transaction is recorded as a rollback.

The output directory contains multimodal CSV telemetry, input/latent/reconstruction OBJ point clouds for every modality and a replayable checkpoint containing all four models and their temporal histories.

## Run the interactive 4D multimedia engine

```bash
./build/cpp-runtime/jarvisx-multimedia4d-gl \
  --edge 16 \
  --channels 4 \
  --temporal-depth 8 \
  --proposal-steps 2 \
  --learning-rate 0.015
```

The visualizer renders the selected modality input, temporal latent stack, reconstruction and residual directly from the numerical runtime. Controls are `1-4` for modality selection, Space for training, `Q` for Q3 inference, `F` to follow the adaptive scheduler, `R` for camera rotation, `S`/`L` for checkpoint save/load and `+/-` for display density.

Its frame-budget controller adjusts rendered density, visible history depth and update cadence only. Wall-clock timing does not determine numerical candidate acceptance or rollback.

See [`docs/CPP_4D_MULTIMODAL_RUNTIME.md`](../docs/CPP_4D_MULTIMODAL_RUNTIME.md) for equations, transactional semantics, persistence and capability boundaries.

## Sparse runtime state artifacts

The default `.jarvisx-runtime/` directory contains:

- `genome.current` — atomically committed runtime genome;
- `runtime.rom` — big-endian 64-bit bytecode words;
- `evolution.csv` — accepted and rolled-back generations with telemetry.

Use `--reset` to discard an earlier checkpoint, `--state-dir PATH` to isolate an experiment and `--min-improvement X` to tighten the commit threshold.

## Determinism contract

Sparse candidate generation, feature extraction, encoding, decoding and fitness selection are deterministic for the same input and genome. The 3D ANN core uses deterministic initialization and update order for the same model configuration and training sequence. The 4D multimodal scheduler, candidate evaluation and temporal-history updates are deterministic for the same configuration and checkpoint.

Wall-clock latency and rendering cadence are telemetry only and excluded from sparse-processor fitness and multimodal commit decisions. Platform floating-point and transcendental implementations may produce small cross-architecture differences; bit-exact portability is not claimed.

## Validation

The CTest suite includes:

- inward executable/text smoke execution;
- genome normalization before allocation;
- repeatable sparse processor evaluation;
- proof that wall-clock latency does not alter deterministic fitness;
- deterministic 3D ANN initialization;
- measurable reconstruction-error reduction under training;
- signed 3-bit latent-level validation;
- exact 3D model save/load replay on the same platform;
- distinct deterministic visual/audio/text/generic fixtures;
- deterministic multimodal scheduling and transactional replay;
- bounded temporal histories and candidate MSE gates;
- exact multimodal checkpoint restoration on the same platform.

Optional sanitizer build:

```bash
cmake -S cpp_runtime -B build/cpp-runtime-san \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVISX_ENABLE_SANITIZERS=ON \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/cpp-runtime-san --parallel
ctest --test-dir build/cpp-runtime-san --output-on-failure
```
