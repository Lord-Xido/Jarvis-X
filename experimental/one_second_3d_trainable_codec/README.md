# JARVIS X one-shot 3D trainable decoder

Isolated C++17 scientific reference with a measured **1 second runtime budget**. No external dependencies, no GPU, no cloud service.

## Execute

```sh
cmake -S experimental/one_second_3d_trainable_codec -B /tmp/jx3d-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx3d-build -j2
ctest --test-dir /tmp/jx3d-build --output-on-failure
/tmp/jx3d-build/jx3d --budget-ms 1000 --epochs 120 --output /tmp/jx3d
```

## Mathematical field and learning

Spatial 3-torus `Omega=(Z/16Z)^3`, 4,096 quantized voxels. Deterministic inward encoder `e(c)=floor(sum(q[2c+delta])/8)` on `(Z/8Z)^3`, 512 latent bytes. Trainable decoder operates on a periodic `3x3x3` coarse neighborhood, separately for each of eight subvoxel phases:

`D_theta(e)[2c+delta]=clip(e(c)/255+0.32*tanh(b_delta+sum_k w_delta,k*(e(c+k)-e(c))/255),0,1)`.

Total learned parameters: 8 x (27+1) = 224. Optimize six independently seeded smooth, synthetic 3D training fields by full-batch Adam. Validate on another field, excluded from training. The encoder is fixed, so this is a *trainable decoder*, not a fully trainable encoder/decoder pair.

Controlled recurrence: `S_(t+1)=0.5*S_t+0.5*D_theta(e)`, initialized from nearest-neighbor decoding and iterated six times. Because the decoded target is fixed and values are bounded in [0,1], the recurrence is a 1/2 contraction in L-infinity norm (up to floating-point rounding).

Exact lossless residual path stores signed-16-bit `r(i)=q(i)-e(floor(i/2))` and reconstructs `q(i)=e+r` exactly. This is **not compression**: 512 latent bytes + 8,192 residual bytes > 4,096 source bytes (headers and model not counted).

## Local benchmark, 2026-11-20

GCC 14 C++17 strict warnings-as-errors build PASS; Clang 17 ASan/UBSan PASS; CTest 2/2 PASS; exact roundtrip 4,096/4,096. Default 120 epochs and CSV export finished in about 175 ms locally, excluding compilation. Smooth held-out RMSE 0.032119 (nearest baseline) to 0.0015781 (trained), 95.09% reduction. The same model did **not** improve independent random byte volumes: baseline RMSE 0.2749982 versus trained 0.2754519. This is family-specific generalization, not universal intelligence or universal compression.

The `--budget-ms` limit is measured and checked; it is **not** a hard-real-time guarantee across different CPUs. Training checks deadline between epochs and reserves 30% of the budget for verification and export. Process exits nonzero when full measured runtime exceeds the budget. No claim of a one-second compile or hard deadline.

`--self-test` verifies toroidal geometry, deterministic exact roundtrip, measured learning improvement, and controlled feedback. The outputs include 3D voxel CSV for inspection.
