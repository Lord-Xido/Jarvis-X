# C++ 3D Symmetry Performance Envelope

This benchmark turns the closed three-plane symmetry autoencoder into a measurable engineering target. It deliberately separates deterministic correctness metrics from wall-clock telemetry, because GitHub-hosted runner timing is useful for smoke/regression tracking but is not a substitute for controlled hardware benchmarking.

## Benchmark matrix

The full default sweep is:

- grid side `n`: `8,16,32,64,128,256`;
- flip probability `p`: `0,0.05,0.10,0.20,0.30,0.40`;
- repetitions per point: `5`;
- deterministic fixture and corruption seeds.

Run it with:

```bash
./build/self-editor3d/jarvisx-symmetry-benchmark3d \
  --output symmetry3d-benchmark.csv
```

A smaller CI-oriented sweep is available with `--quick`.

## Noise domains

Two failure models are measured independently.

### Common-mode input corruption

The source is corrupted before encoding:

```text
X -> corrupt(X) -> [X', H(X'), V(X')]
```

All three latent layers therefore inherit the same incorrect source bit. Symmetry redundancy cannot vote away a common-mode error. The `identity`, `symmetry-exact`, and `symmetry-closed-loop` rows make this boundary measurable rather than assuming denoising capability.

### Independent latent corruption

The clean source is encoded first and each latent layer is then corrupted independently:

```text
X -> [X,H(X),V(X)] -> independent layer noise -> consensus decode
```

For independent binary flips with probability `p`, exact three-copy majority decoding has theoretical bit-error probability

```text
P_error = 3 p^2 (1-p) + p^3 = 3 p^2 - 2 p^3.
```

The benchmark records both empirical bit accuracy and this theoretical repetition-code error for the exact majority model.

## Models currently measured

- `identity`: corrupted input returned directly;
- `symmetry-exact`: encode and aligned majority decode after common-mode corruption;
- `symmetry-closed-loop`: learned transport plus recurrent feedback after common-mode corruption;
- `single-copy`: one independently corrupted latent copy used as the baseline;
- `symmetry-majority`: exact aligned majority vote across three independently corrupted latent layers;
- `symmetry-learned-transport`: learned row-stochastic transport followed by aligned consensus decode.

Dense-AE, CNN-AE and other learned baselines are **not** labeled or simulated by proxy in this harness. They should be added only as real implementations with matched training data, parameter budgets and evaluation protocol.

## Recorded metrics

Each CSV row records:

- reconstruction MSE;
- binary bit accuracy;
- theoretical bit error where defined;
- feedback convergence steps and fixed-point residual;
- initial/final optimization objective and fractional reduction;
- optimization sweeps and accepted coordinate moves;
- optimization latency;
- inference/feedback latency;
- throughput in megapixels per second;
- estimated working-set bytes;
- latent scalar expansion ratio.

The current three-plane code has a latent scalar expansion ratio of `3.0`; this is redundancy, not compression. Reporting it explicitly prevents a robustness code from being mischaracterized as a storage codec.

## Interpretation rules

1. Timing is machine-dependent. Compare timing only within controlled hardware/compiler configurations or as a coarse CI regression signal.
2. Fidelity, deterministic replay, theoretical BER and invariant tests are platform-independent targets except for ordinary floating-point tolerance.
3. Fixed-point convergence is not equivalent to correct reconstruction. `reference_mse`/bit accuracy must be considered alongside fixed-point residual.
4. Common-mode and independent-layer corruption must never be pooled into one robustness number.
5. Performance claims should report the complete matrix or a clearly declared subset, not a single favorable fixture.

## CI

The `C++ Inward 3D Self Editor` workflow builds and tests the benchmark on GCC, Clang with ASan/UBSan and MSVC. The GCC Release leg also runs the quick envelope and uploads `symmetry3d-performance-envelope-gcc` as a CSV artifact. Sanitizer timing is not used as performance telemetry.
