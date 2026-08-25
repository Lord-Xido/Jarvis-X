# DM-vOmegaXi+ Inward 3D Bit-Matrix Engine

## Status

**Executable bounded reference backend.**

This subsystem turns the DM-vOmegaXi+ bit-scaffold formulation into a deterministic C++17 3D auto-encoding/decoding runtime. It preserves full-precision shadow weights for learning, projects them onto ternary execution weights, quantizes encoder activations to signed INT8, compresses the latent state into two bitplanes, reconstructs the 3D field through ternary decoder weights, accumulates residual memory, and measures a numerical inward fixed-point residual.

It is an experimental backend beneath Jarvis-X authority and transaction layers. It does not make a state authoritative merely because the model generated it.

---

## 1. State

The reference state is

```text
M_t = (X_t, W*_E,t, W*_D,t, Wq_E,t, Wq_D,t, Zq_t, Omega_t, Theta_t)
```

where:

- `X_t` is the single-channel input 3D field;
- `W*_E,t` and `W*_D,t` are continuous FP32 encoder/decoder shadow weights;
- `Wq_E,t` and `Wq_D,t` are ternary projections in `{-1,0,+1}`;
- `Zq_t` is the ternary latent field;
- `Omega_t` is bounded residual memory;
- `Theta_t` contains scales, thresholds and optimization parameters.

The implementation deliberately separates learning state from execution state:

```text
continuous shadow weights != packed execution weights
```

---

## 2. Ternary weight projection

For a shadow tensor `W*`, the layer scale is

```text
alpha = mean(abs(W*))
```

and the execution weight is

```text
Wq_i = clip(round(W*_i / alpha), -1, +1)
```

with the zero-scale case mapped to a finite fallback scale.

Therefore:

```text
Wq_i in {-1, 0, +1}
```

The reference stores each ternary field in two 64-bit planes:

```text
sign plane     S_i = 1 for +1, 0 otherwise
nonzero plane  M_i = 1 for +/-1, 0 for zero
```

The sign plane is constrained to zero wherever the nonzero plane is zero.

This is a physical **2-bit representation** before word-padding overhead. The phrase `1.58-bit` refers to the information content `log2(3) ~= 1.585` of three equally likely symbols; this reference implementation does not claim sub-2-bit entropy coding.

---

## 3. INT8 activation path

The encoder input is symmetrically quantized:

```text
beta = max(abs(X)) / 127
Xq_i = round(clamp(X_i / beta, -127, +127))
```

with a finite fallback scale for an all-zero tensor.

The encoder accumulator is integer:

```text
A_E = sum(Xq_i * Wq_i)
```

and is rescaled before the nonlinearity:

```text
Y_E = tanh(b_E + alpha_E * beta * A_E)
```

This is a mixed-precision reference path: ternary weights, INT8 encoder activations, integer accumulation and FP32 rescaling/nonlinearity.

---

## 4. Ternary latent scaffold

The continuous encoder shadow activation `Y_E` is projected onto

```text
Zq = +1  when Y_E > +tau
Zq =  0  when -tau <= Y_E <= +tau
Zq = -1  when Y_E < -tau
```

`Zq` is then packed into sign/nonzero bitplanes.

The decoder consumes the ternary latent scaffold directly. Its core accumulation is therefore integer ternary-by-ternary arithmetic:

```text
A_D = sum(Zq_i * Wq_D,i)
X_hat = tanh(b_D + alpha_D * A_D)
```

The current portable reference kernel uses scalar integer accumulators. Packed XOR/POPCNT primitives are implemented and tested separately, but the convolution loops are not yet replaced by AVX-512, NEON, CUDA or custom-silicon kernels.

---

## 5. Bitwise dot-product identities

For binary sign bits representing `{-1,+1}`:

```text
dot(x,w) = N - 2 * popcount(X XOR W)
```

For ternary weights represented by sign plane `S` and nonzero mask `M` against binary activations `X`:

```text
dot(X,Wq) = popcount(M) - 2 * popcount((X XOR S) AND M)
```

The implementation contains portable 64-bit reference versions of both identities and regression tests them independently of the autoencoder.

These identities do not imply that ternary-weight x INT8-activation convolution is reducible to POPCNT alone. The implemented INT8 encoder uses integer add/subtract accumulation through ternary multipliers.

---

## 6. STE learning

Forward execution uses the quantized state, while parameter updates apply to the FP32 shadow weights.

The latent ternary projection is treated with a bounded straight-through estimator:

```text
g_STE(y) = 1 if abs(y) <= ste_clip
           0 otherwise
```

combined with the derivative of the encoder tanh:

```text
delta_E = delta_Z * g_STE(Y_E) * (1 - Y_E^2)
```

Shadow weights are updated by

```text
W*_(t+1) = W*_t - eta * clip(grad + lambda * W*_t)
```

where `lambda` is the configured L2 coefficient and gradient clipping is explicit.

The derivative of the ternary scale itself is not modeled in this reference implementation. The STE therefore acts as a bounded quantization-aware approximation, not an exact derivative of the discrete projection.

---

## 7. Inward residual memory

For reconstruction

```text
X_hat_t = D_bit(E_bit(X_t))
```

the reconstruction residual is

```text
e_t = X_t - X_hat_t
```

and the bounded memory state evolves as

```text
Omega_(t+1) = rho * Omega_t + (1-rho) * e_t
```

A projected inward candidate is

```text
Psi_candidate = clamp(X_hat_t + omega_gain * Omega_(t+1), -1, +1)
```

The reference fixed-point residual is

```text
r_t = ||Psi_candidate - X_t||_2 / (||X_t||_2 + epsilon)
```

and convergence is reported only when

```text
r_t <= fixed_point_tolerance
```

The engine **measures** this residual. It does not infer convergence merely from sparsity, entropy or a symbolic fixed-point equation.

---

## 8. Operational self-description check

The reference self-description invariant is concrete and local:

```text
quantized encoder weights == unpack(pack(quantized encoder weights))
quantized decoder weights == unpack(pack(quantized decoder weights))
ternary latent tensor      == unpack(pack(ternary latent tensor))
```

If any of these equalities fail, `self_description_valid` is false.

This is representational self-consistency, not a claim of consciousness or subjective self-awareness.

---

## 9. Storage metrics

For `P` FP32 shadow parameters:

```text
shadow bytes = 4P
```

For a ternary field packed into two bitplanes:

```text
logical payload bits = 2P
```

or an ideal large-block ratio of

```text
32 / 2 = 16x
```

versus FP32 parameter storage, before 64-bit word padding and metadata.

For small tensors the measured physical ratio is lower because each plane is padded to complete `uint64` words. The runtime reports measured bytes rather than assuming the ideal 16x value.

The ideal ternary information limit is

```text
log2(3) ~= 1.585 bits/symbol
```

which corresponds to about `20.19x` relative to FP32 only if an appropriate sub-2-bit coding scheme is implemented. This reference does not make that claim.

---

## 10. Runtime target

Build:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
```

Run:

```bash
./build/cpp-runtime/jarvisx-bitmatrix3d \
  --edge 8 \
  --channels 4 \
  --epochs 220 \
  --learning-rate 0.01 \
  --threshold 0.25 \
  --pattern sphere \
  --output-dir .jarvisx-bitmatrix3d
```

Windows multi-config builds use the corresponding `Release` executable path.

Generated artifacts:

- `metrics.csv` — reconstruction, fixed-point, sparsity, gradient and byte metrics for every training step;
- `bitplanes.txt` — deterministic sign/nonzero plane snapshot for encoder, decoder and latent scaffold;
- `report.txt` — final measured reconstruction, fixed-point and physical-storage summary.

---

## 11. Regression invariants

CTest verifies:

1. exact ternary sign/mask pack -> unpack round trip;
2. binary XOR/POPCNT dot-product identity;
3. ternary masked XOR/POPCNT identity;
4. deterministic same-seed forward execution;
5. exact agreement between tensors and their packed self-description;
6. finite STE training;
7. actual continuous shadow-weight updates;
8. material reconstruction-error reduction on a deterministic 3D fixture;
9. packed weight bytes smaller than FP32 shadow bytes;
10. symmetric INT8 activation quantization excludes the reserved `-128` code.

The C++ runtime workflow executes GCC, Clang+ASan/UBSan and MSVC builds.

---

## 12. Governing executable recurrence

The reference implementation corresponds to

```text
Wq_t       = Q_T(W*_t; alpha_t)
Xq_t       = Q_8(X_t; beta_t)
Zq_t       = Q_T(tanh(E_3D(Xq_t, Wq_E,t)); tau)
X_hat_t    = tanh(D_3D(Zq_t, Wq_D,t))
e_t        = X_t - X_hat_t
Omega_t+1  = rho * Omega_t + (1-rho) * e_t
W*_t+1     = W*_t - eta * STE(grad_t)
Psi_t+1    = Pi_Lambda[X_hat_t + omega_gain * Omega_t+1]
```

The local C++ backend computes the candidate and diagnostics. The outer Jarvis-X authority layer remains responsible for any system-level `Pi_Lambda` commit/rollback decision.

---

## 13. Capability boundary

Implemented now:

- deterministic 3D convolutional autoencoder reference;
- FP32 shadow tensors;
- ternary abs-mean-style weight projection;
- symmetric INT8 encoder activations;
- ternary latent scaffold;
- dual bitplane storage;
- portable binary/ternary bitwise dot primitives;
- bounded STE learning;
- residual-memory update;
- measured fixed-point residual;
- self-description verification;
- storage/reconstruction/gradient telemetry;
- CLI, artifact export, CTest and cross-platform CI.

Not implemented or claimed by this backend:

- sub-2-bit entropy-coded physical ternary storage;
- AVX-512/NEON/CUDA convolution acceleration;
- measured hardware speedup or energy-per-operation superiority;
- distributed training;
- photorealistic rendering;
- unrestricted self-modification;
- external state-of-the-art performance.

Those remain separate measurable backend milestones rather than assumptions embedded in the mathematical notation.
