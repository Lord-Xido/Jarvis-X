# Dr Moagi DM3D Operational Engine

This target operationalises the 3D auto-encoding/decoding matrix as a bounded-memory C++17 runtime.

## Execution contract

The default logical field is `1000 x 1000 x 1000` voxels:

- `1000^2 = 1,000,000` logical `(x,y)` lanes.
- `1000^3 = 1,000,000,000` addressable voxels.
- Default brick: `25^3 = 15,625` voxels.
- Default brick lattice: `40^3 = 64,000` schedulable 3D tasks.
- The full tensor is streamed through bricks; it is never allocated as one dense array.

## Operational mechanics

For each brick the engine constructs the eight-term geometric basis

```text
phi(u,v,w) = [1,u,v,w,uv,vw,wu,uvw]
```

after applying the current inward geometric warp.

Instead of repeatedly rereading a brick for line-search trials, one scan accumulates the sufficient statistics

```text
G = Phi^T Phi
b = Phi^T X
c = X^T X
```

and solves the exact eight-dimensional least-squares system

```text
(G + epsilon I) z = b
```

with pivoted Gauss-Jordan elimination. Reconstruction error is then evaluated from

```text
MSE(z) = (c - 2 z^T b + z^T G z) / M
```

without an additional source-volume pass.

## Geometry-coupled inward loop

The codec does not keep geometry as a disconnected visual state. Local coordinates are transformed before basis evaluation:

```text
r       = Rz Ry Rx p
h(r)    = [r_y r_z, r_z r_x, r_x r_y]
p_warp  = lambda * (r + kappa h(r))
X_hat   = z^T phi(p_warp)
```

Each iteration generates an inward rotated/hyperfolded candidate, solves its exact local latent, and commits only a finite non-worsening reconstruction candidate.

If the exploratory geometry is rejected, the runtime still advances a pure inward contraction analytically. For fixed rotation and curvature,

```text
phi' = diag(1,s,s,s,s^2,s^2,s^2,s^3) phi
```

so the runtime transforms `G`, `b`, and `z` directly rather than rereading source voxels.

This implements the invariant:

```text
GENERATE -> CONTRAST -> RECKON -> VERIFY -> COMMIT/CORRECT -> RECUR
```

with

```text
MSE_(k+1) <= MSE_k + tolerance
||q_(k+1)|| <= ||q_k||
```

## 3D spatial reconciliation

After local inward refinement, each active brick compares predictions at its six axial faces with adjacent brick predictions. A face-consistency correction is proposed through the scalar intercept and is committed only if that brick's local reconstruction MSE remains non-worsening.

Cube boundaries are non-periodic by default. `--toroidal` enables wrapped neighbours explicitly.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-dm3d-operational --parallel
./build/cpp-runtime/DrMoagi-DM3D-Operational --self-test
```

Bounded execution:

```bash
./build/cpp-runtime/DrMoagi-DM3D-Operational \
  --side 1000 --brick 25 --iterations 4 --bricks 256 --threads 16 \
  --out dm3d.json
```

Full logical sweep:

```bash
./build/cpp-runtime/DrMoagi-DM3D-Operational \
  --side 1000 --brick 25 --iterations 4 --full --threads 16 \
  --out dm3d-full.json
```

## Container

```bash
docker build -f deploy/dm3d-operational/Dockerfile -t jarvisx-dm3d-operational .
docker run --rm --cpus=16 jarvisx-dm3d-operational \
  --side 1000 --brick 25 --iterations 4 --bricks 256 --threads 16
```

The container health check uses the constant-time `--health` path; the computational self-test remains a build/CI gate rather than a periodic production workload.

## Capability boundary

This is a deterministic geometric reference codec, not a trained neural autoencoder. The default source is a synthetic continuous 3D field. The logical one-billion-voxel domain is real address geometry, but only configured bricks are materialized and processed. The verifier establishes monotone reconstruction and inward-radius invariants for the implemented objective; it does not establish universal convergence or lossless compression.
