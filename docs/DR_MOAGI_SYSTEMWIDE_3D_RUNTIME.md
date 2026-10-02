# Jarvis-X / Moagi system-wide 3D C++ map

This is a self-contained C++17 reference implementation of the systems-wide architecture.

It maps and executes:

`REALITY -> multimodal ingest -> sparse virtual 1000MB^3 substrate -> active tile fabric -> encoder -> distributed tile latents -> orchestrator/fusion -> Omega + possibility field -> Phi_in inward refinement -> decoder -> residual -> codec boundary -> CTR -> commit/serve -> REALITY`

It also exposes an arithmetic-planner integration point capable of selecting between `EXECUTE`, `REDUCE_3D`, `COMPOSE_OPERATOR`, and `SOLVE_FIXED_POINT`.

## Scale semantics

Each virtual axis is `1000 MB = 1,000,000,000` logical byte positions, so the logical coordinate universe is `(10^9)^3 = 10^27`. The program never allocates `10^27` cells; only a bounded active tile set is resident.

## Build

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic cpp_runtime/src/systemwide_3d_map_main.cpp -o moagi3d
```

## Run

```bash
./build/cpp-runtime/DrMoagi-Systemwide-3D-Map --tile-side 16 --block 4 --max-tiles 8 --iterations 10 --out moagi_system
```

## Outputs

- `moagi_system.obj` — 3D wireframe map for Blender/MeshLab, including active sparse tile points, nested latent shells, and a closed Reality feedback path.
- `moagi_system.dot` — Graphviz topology.
- `moagi_system.json` — execution telemetry and CTR receipts.

## Mathematical core

Tile encoding: `X_i -> E(X_i) -> Z_i`

Fusion: `Z_0 = sum_i w_i Z_i`

Residual memory: `Omega_(k+1) = beta Omega_k + (1-beta) E(X-Xhat)`

The geometric candidate applies a rotated/contracted RK2 backtrace through a torus + Lorenz-like + hyperfold field. CTR compares residual-guided and geometric candidates against the currently committed state and commits only a finite MSE-improving candidate.

The block codec is deliberately dependency-free. Replace `block_encode`, `block_decode`, `encode_tiles`, and `inward_fold` with libtorch/CUDA kernels to turn the reference geometry into a trained ANN runtime while preserving the systems contracts.

## Repository integration

The canonical executable target is `jarvisx-systemwide-3d-map` with output name
`DrMoagi-Systemwide-3D-Map`. It is built by the normal `cpp_runtime` CMake
project and exercised by CTest.

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-systemwide-3d-map --parallel
ctest --test-dir build/cpp-runtime -R systemwide-3d-map-runtime-smoke --output-on-failure
```

## Capability boundary

This executable is a bounded reference laboratory. The 1000 MB-per-axis cube is
virtual address geometry, not resident memory. The encoder/decoder are
deterministic block codecs rather than trained neural networks. The possibility
field and geometric inward branch are numerical reference mechanisms, and CTR
establishes candidate-first monotone reconstruction acceptance for the bounded
demo objective only. It does not establish universal convergence, semantic
truth, arbitrary lossless compression, or hardware throughput.
