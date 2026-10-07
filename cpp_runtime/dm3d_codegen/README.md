# DM3D 3D C++ Code-Generation Runtime

Closed-loop C++ autotuning/code-generation subsystem for Jarvis-X.

## Operational loop

`3D genome -> decode -> execute -> verify -> benchmark -> select -> inward fold + mutation -> emit C++ -> compile -> execute`

The program genome lives on `G = Z8 x Z8 x Z12`. Genome coordinates decode into 3D tile dimensions, X-unroll depth, traversal order, and toroidal boundary strategy. Only candidates within the numerical verification tolerance are eligible for performance selection.

## Build

```bash
cmake -S cpp_runtime/dm3d_codegen -B build/dm3d_codegen
cmake --build build/dm3d_codegen --config Release
```

## Run

```bash
./build/dm3d_codegen/dm3d_codegen 8 12 48 build/dm3d_codegen/generated/best_kernel.cpp
```

The host emits the winning standalone C++ kernel, compiles it with `g++`, runs it, and returns non-zero on compile/runtime failure.

## Verification invariant

`||K_theta(X) - K_ref(X)||_inf <= epsilon`

Performance is hardware/compiler dependent; semantic verification is the hard constraint.

## Next extension

Replace the compact kernel genome with a typed 3D IR supporting nodes such as `ENCODE`, `CONV3D`, `MATMUL`, `ATTENTION`, `FFT`, `DMA`, `REDUCE`, `DECODE`, and `FIX_POINT` while retaining the same verify-before-select loop.
