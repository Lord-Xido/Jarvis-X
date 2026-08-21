# DM–vΩΞ⁺ Infinity Turbo Core v7

## Status

Experimental CUDA acceleration track for the Jarvis-X C++ runtime.

The subsystem implements a deterministic Q16.16 data path:

```text
128-D state
   ↓
48-D latent encode
   ↓
128-D reconstruction
   ↓
128-D → 64³ deterministic geometric lift
   ↓
periodic 3D Q16.16 stencil recursion
   ↓
64³ → 128-D deterministic pooling
   ↓
measured reconstruction-health diagnostic
   ↓
bounded diagnostic feedback
   ↓
next 128-D state
```

The architecture is designed to test whether a volumetric latent-processing substrate can improve throughput or useful representation quality on supported CUDA hardware. It does **not** establish a fixed `10^9×` speedup, self-awareness, infinite computation, or a mathematical equivalence between physical dimensions and latent dimensions.

## Why v7.1 differs from the supplied v7.0 prototype

The original prototype contained several correctness hazards that prevented its advertised behavior from being measured safely:

- a `D_BATCH × 128` reconstruction allocation was passed to a kernel indexing `D_BATCH × 262,144` voxels;
- Q16.16 products were multiplied together without intermediate rescaling and could overflow before accumulation;
- decoder weights, decoder biases, volume buffers and diagnostic metrics were not fully initialized;
- the 3D → 128-D stage reused a matrix allocated for `48 × 128`, despite requesting `262,144 → 128` multiplication;
- the volumetric kernel updated its input buffer in place, creating data races between neighboring CUDA threads;
- throughput accounting mixed cumulative and per-cycle quantities, so the printed speed could grow without corresponding hardware performance;
- the code claimed tensor-core execution while using ordinary scalar integer CUDA kernels.

v7.1 replaces those paths with explicit, testable operators.

## Geometric lift and pooling

Let `s ∈ Q16.16^128`. The lift operator is

\[
L(s)_v = s_{v \bmod 128}, \qquad v=0,\ldots,64^3-1.
\]

Because `64^3 / 128 = 2048`, pooling is

\[
P(V)_i = \frac{1}{2048}\sum_{k=0}^{2047}V_{i+128k}.
\]

For an unmodified lifted volume,

\[
P(L(s)) = s
\]

exactly for integer Q16.16 values. The host-side contract test verifies this identity.

## 3D recursion

For radius `r`, each voxel is updated from a normalized periodic stencil:

\[
V^{(d+1)}_{x,y,z}
=
\phi_{Q16}
\left(
\sum_{\Delta x,\Delta y,\Delta z=-r}^{r}
K_{\Delta x,\Delta y,\Delta z}
V^{(d)}_{x+\Delta x,y+\Delta y,z+\Delta z}
\right),
\]

with periodic wrapping and a bounded Q16.16 tanh approximation. Two device buffers are ping-ponged between recursion depths so reads and writes do not race.

## Diagnostic feedback

The runtime measures a per-batch mean absolute state change and maps it into a bounded health score:

\[
h_b = 1 - \frac{1}{128}\sum_i\min(|s_{b,i}^{t+1}-s_{b,i}^{t}|,1).
\]

The diagnostic can weakly modulate the next state through a user-bounded feedback gain. This is operational self-monitoring; it is not evidence of consciousness or self-awareness.

## Performance measurement

GPU time is measured with CUDA events around one complete encode → decode → lift → recurse → pool → feedback cycle.

The executable reports an **estimated primitive operation rate** based on the known dense and stencil work counts divided by measured GPU time. This is an engineering throughput diagnostic, not a FLOP-equivalent standard benchmark.

A speedup factor must be computed only against a specified baseline implementation run on specified hardware with equivalent numerical work.

## Build

CUDA is opt-in so normal CPU CI remains portable:

```bash
cmake -S cpp_runtime -B build/cpp-runtime \
  -DJARVISX_BUILD_CUDA_TURBO=ON \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/cpp-runtime --config Release --parallel
```

Run:

```bash
./build/cpp-runtime/jarvisx-infinity-turbo \
  --batch 8 \
  --radius 2 \
  --depth 3 \
  --cycles 5
```

The implementation accepts batch sizes up to 256, but performs a CUDA free-memory check before allocating the two 64³ ping-pong volume buffers.

## Portable regression coverage

The CPU-only `jarvisx-infinity-turbo-contract-tests` target verifies:

1. saturating Q16.16 multiply and bounded activation behavior;
2. the exact 128-D → 64³ → 128-D lift/pool identity in the absence of processing;
3. the maximum-batch volume memory contract: 256 MiB per 64³ Q16.16 volume buffer and 512 MiB for ping-pong storage.

These tests run without a CUDA device and therefore protect the mathematical/layout contract in standard CI. GPU kernel correctness and performance still require CUDA-enabled CI or hardware benchmarking.
