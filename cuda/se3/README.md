# Jarvis-X analytic SE(3) CUDA runtime

This directory contains a standalone CUDA reference implementation of the batched rigid-body exponential map.

Each CUDA thread consumes one twist rate

\[
\xi=(\omega,v)\in\mathbb R^6
\]

and an integration interval `dt`, computes

\[
\phi=\omega\,dt,\qquad \rho=v\,dt,
\]

then emits the compact row-major pose `[R | t]` as three `float4` rows. The invariant homogeneous row `[0,0,0,1]` is not stored.

## Operational path

```text
aligned Twist8 input
→ one thread per pose
→ stable A/B/C coefficients
→ analytic Rodrigues rotation
→ two cross products for the left-Jacobian translation
→ aligned Pose3x4 output
→ FP64 CPU validation
```

The kernel deliberately does not pad 3×3 matrices into Tensor Core tiles. For this workload, direct scalar/vector arithmetic performs the required useful operations without a mostly empty 16×16 tile.

## Build

Direct NVCC build:

```bash
nvcc -O3 -std=c++17 -lineinfo \
  cuda/se3/jarvis_x_se3.cu \
  -o jarvis_x_se3
```

CMake build:

```bash
cmake -S cuda/se3 -B build/se3 -DCMAKE_BUILD_TYPE=Release
cmake --build build/se3 --parallel
```

Fast math is intentionally disabled by default. It can be enabled for a measured experiment:

```bash
cmake -S cuda/se3 -B build/se3-fast \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVIS_X_SE3_FAST_MATH=ON
```

## Run

```bash
./jarvis_x_se3 \
  --count 1048576 \
  --warmup 10 \
  --repeats 100 \
  --dt 0.01 \
  --device 0
```

The executable reports separately:

- pinned host-to-device time;
- mean GPU-resident kernel time;
- device-to-host time;
- end-to-end time for one batch;
- poses per second;
- effective input-plus-output bandwidth;
- rotational geodesic error against an FP64 CPU reference;
- translation error;
- rotation orthogonality and determinant error.

The process exits nonzero when the deterministic validation thresholds fail.

## Data contract

Input per pose: 32 bytes.

```cpp
struct Twist8 {
    float4 omega;     // rad/s in xyz; w reserved
    float4 velocity;  // m/s in xyz; w reserved
};
```

Output per pose: 48 bytes.

```cpp
struct Pose3x4 {
    float4 row0;  // R00 R01 R02 tx
    float4 row1;  // R10 R11 R12 ty
    float4 row2;  // R20 R21 R22 tz
};
```

Total compulsory kernel traffic is therefore 80 bytes per pose before cache effects.

## Numerical contract

For

\[
W=[\phi]_\times,\qquad \theta^2=\phi^T\phi,
\]

the kernel evaluates

\[
R=I+A W+B W^2,
\]

\[
t=\rho+B(\phi\times\rho)+C\bigl(\phi\times(\phi\times\rho)\bigr),
\]

where

\[
A=\frac{\sin\theta}{\theta},\quad
B=\frac{1-\cos\theta}{\theta^2},\quad
C=\frac{\theta-\sin\theta}{\theta^3}.
\]

A sixth-order Maclaurin expansion handles the removable singularity around `theta=0`. Exact zero angular rate therefore produces identity rotation without division by zero.

## Performance interpretation

The reported kernel interval measures repeated launches with data already resident on the selected GPU. Transfer and end-to-end numbers are separate. No fixed nanosecond, throughput, occupancy, or positional-accuracy claim is embedded in the implementation; those values must be obtained on a named GPU, clock/power policy, CUDA toolchain, batch size, and precision configuration.

Use Nsight Compute to inspect the actual limiting resource:

```bash
ncu --set full ./jarvis_x_se3 --count 1048576 --repeats 20
```
