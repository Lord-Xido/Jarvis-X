# Bounded 4D Clifford Torus -> 3D Projection Engine

This C++17 reference adapts the supplied `clifford_engine.cpp` into an isolated,
deterministic geometry/telemetry target. It is **not** an autoencoder, decoder,
cryptographic engine, GPU shader, physical mechanics model, quantum computer, or
connected neural inference pipeline.

## Mathematical model

The 4D Clifford torus is parametrized by

```text
p(u,v) = (R1 cos(u), R1 sin(u), R2 cos(v), R2 sin(v))
||p||_2^2 = R1^2 + R2^2
p' = R_ZW(a2) R_YW(a1) R_XW(a0) p
q = (p'_x,p'_y,p'_z) * P/(P-p'_w)
```

The ordered 4D rotations are orthogonal: their matrix satisfies
`R^T R = I` up to floating-point error, preserving `||p||_2^2`.
For all rotations, `|p'_w| <= sqrt(R1^2+R2^2)`.
The engine rejects `P <= sqrt(R1^2+R2^2)`, ensuring the projection
denominator is strictly positive. With the default `P=4`,
`R1=2`, and `R2=1`, this is a **pole-free perspective projection**,
not mathematically the standard stereographic projection of that
torus' containing 3-sphere.

## Operational contract

- A frame samples a `grid x grid` mesh, computes projected vertices,
  the 3D centroid, 4D mean absolute component magnitudes, maximum squared
  norm preservation residual, orthogonality residual and minimum projection
  denominator.
- A dedicated telemetry consumer receives a copy of each frame packet via
  a bounded mutex/condition-variable queue. On saturation, the oldest
  queued packet is discarded and counted. Shutdown drains the remaining queue.
- `flux_index = tanh(sum(mean_abs_4D_components)/4)` is a
  **heuristic geometric score**, not Shannon entropy, cognition, energy,
  a measured information flux or a physical quantity.
- `--time-scale` multiplies **simulated elapsed time only**. It does
  not accelerate actual execution. Console frame and vertex rates are measured
  using a monotonic wall clock; they depend on the CPU and workload.
- Runtime complexity is `O(frames * grid^2)`. Mesh storage is
  `O(grid^2)`, with at most `queue_capacity` telemetry packets.
  The default rotation rates are 0.012 (XW) and 0.008 (YW) radians per
  unit of simulated time, with ZW disabled as in the supplied defaults.

## Build and validate

```bash
cmake -S cpp_runtime -B build/clifford4d \
  -DJARVISX_BUILD_CLIFFORD_TORUS_4D=ON \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/clifford4d --target jarvisx-clifford-torus4d --parallel
ctest --test-dir build/clifford4d -R '^clifford-torus-4d-' --output-on-failure
./build/clifford4d/jarvisx-clifford-torus4d \
  --frames 128 --grid 40 --time-scale 1000000 --queue-capacity 64
```

The `--self-test` exercises 4D rotation, composed matrix, norm preservation,
orthogonality, projection identity and pole rejection, bounded queue accounting
and a finite large-time-scale run. This module does not alter codec or AEAD
implementations in the repository.
