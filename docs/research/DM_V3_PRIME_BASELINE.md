# DM-V3-PRIME Baseline

**Status:** Executable research baseline  
**Repository:** `Lord-Xido/Jarvis-X`  
**Parent specification:** `docs/research/DR_MOAGI_3D_CODEC_RUNTIME.md`  
**Control plane:** `src/jarvisx/dm_v3_prime_control.py`  
**Reference experiment:** `experiments/dm_v3_prime_torch.py`

## 1. Purpose

DM-V3-PRIME binds the Dr Moagi 3D adaptive codec-runtime specification to a
concrete, measurable implementation without weakening the deterministic core
boundary.

The baseline implements four mechanisms that were previously only partially
represented in the prototype:

1. true 3D neighbourhood processing through `Conv3d` operators;
2. recursive spectral latent refinement through an FFT-domain residual cell;
3. explicit quantization plus a learned entropy-rate model and a real compressed
   latent bitstream;
4. fail-closed candidate verification with atomic commit-or-rollback semantics.

The implementation deliberately separates *incremental acceptance* from the
aspirational 1000x performance target.  A valid candidate may improve quality
or efficiency while still reporting that the 1000x target has **not** been met.

---

## 2. PRIME state transition

The executable baseline is summarized by

```text
X_t
  -> E_theta^3D
  -> Z_t^(0)
  -> [R_omega^spectral]^ell
  -> Z_t^*
  -> Q_Delta
  -> p_psi / entropy-rate model
  -> B_t
  -> Q_Delta^-1
  -> D_phi^3D
  -> X_hat_t
  -> Pi_(H,Lambda)
  -> COMMIT or ROLLBACK
```

For recursion depth `ell = 2`,

```text
Z^(k+1) = Z^(k) + alpha * R_omega^spectral(Z^(k)),  k in {0, 1}
```

where `R_omega^spectral` performs a real FFT over the latent axis, applies a
learned bounded spectral gain, transforms back to latent space, and applies a
learned residual MLP.

---

## 3. N^3 spatial encoder

For a scalar voxel field

```text
X in R^(1 x 1 x 32 x 32 x 32)
```

the encoder performs

```text
32^3
 -> Conv3d(1, 16, k=3, s=2)
 -> 16^3
 -> Conv3d(16, 32, k=3, s=2)
 -> 8^3
 -> Linear(32 * 8 * 8 * 8, L)
 -> Z
```

The convolutional receptive field makes each encoded feature a function of its
local 3D neighbourhood rather than an independent flattened voxel.

This is a topology-aware data-plane implementation of the neighbourhood rule;
it is **not** equivalent to physical hardware-placement optimisation.  The
Section 15 placement objective remains a separate scheduler/runtime problem.

---

## 4. Recursive spectral upwelling

The original stochastic perturbation

```text
Z' = Z + noise
```

is replaced by an actual spectral transform:

```text
U = FFT_r(LayerNorm(Z))
U' = sigmoid(g_omega) * U
R = MLP(IFFT_r(U'))
Z_(k+1) = Z_k + alpha * R
```

This makes the term "spectral" operational rather than metaphorical.

The residual step is bounded by an explicit step size and the training loop
clips gradient norm before updating parameters.

---

## 5. Quantization and rate model

Training uses straight-through scalar quantization:

```text
q = round(Z / Delta)
Z_q = Z + stop_gradient(q * Delta - Z)
```

The entropy model is a learned factorized Laplace distribution.  For each
quantization cell, the estimated discrete probability mass is

```text
p(q) = CDF((q + 1/2)Delta) - CDF((q - 1/2)Delta)
```

and the differentiable rate term is

```text
R_proxy = mean(-log2 p(q)).
```

The optimization objective used by the reference experiment is

```text
J = D_MSE(X, X_hat) + lambda_R * R_proxy.
```

For evaluation, quantized latent symbols are converted to signed 16-bit values,
protected with a CRC32 checksum, and compressed with zlib.  The measured
bitstream size is reported independently from the learned rate proxy.

---

## 6. Pi_(H,Lambda) verification gate

The dependency-free control module defines a transactional gate:

```text
Pi_(H,Lambda)(candidate, incumbent)
```

A candidate is accepted only when all configured constraints pass, including:

```text
finite telemetry
D_candidate <= D_max
memory_candidate <= memory_max
risk_candidate <= risk_max
Safe(candidate) == true
Stable(candidate) == true
latency_candidate > 0
T_incumbent / T_candidate >= min_speedup
J_incumbent - J_candidate >= min_objective_improvement
```

The deployment rule is

```text
if admissible(candidate):
    COMMIT candidate
else:
    ROLLBACK incumbent
```

The 1000x condition is separately recorded as

```text
speed_target_met = accepted and (T_incumbent / T_candidate >= 1000).
```

This prevents a convergence threshold or virtual recursion depth from being
misreported as measured physical acceleration.

---

## 7. Correct timing semantics

GPU kernels may execute asynchronously.  PRIME therefore synchronizes CUDA
immediately before and after a timed call and uses `time.perf_counter()`.

The experiment also constructs the 32^3 coordinate lattice once on the target
device, outside the optimization loop, so codec latency is not contaminated by
repeated mesh construction.

Evaluation latency is the median of repeated measurements after warmup.

---

## 8. Candidate transaction

The experiment treats training as candidate generation rather than immediate
constitutional deployment:

```text
snapshot incumbent state
        |
        v
train candidate
        |
        v
validation workload
  |       |       |
quality  rate   latency
  \       |      /
   Pi_(H,Lambda)
      /       \
   COMMIT    ROLLBACK
```

The initial state dict is preserved.  After training, the candidate and
incumbent are evaluated under the same validation workload.  The selected state
is loaded only after the gate returns its decision.

---

## 9. Operational limits

DM-V3-PRIME is an executable research baseline, not evidence of a demonstrated
1000x self-optimising computer.

In particular:

- weight training normally changes representation quality, not FLOP count;
- architecture-level acceleration requires kernel, graph, sparsity, placement,
  precision, compilation, scheduling or hardware changes;
- `1000x` is a benchmark target and must be measured against a declared
  baseline under the same workload and accuracy envelope;
- the learned entropy model is factorized rather than a full N^3 context model;
- zlib is a reference entropy backend, not a production learned arithmetic or
  range coder;
- physical thermal and energy telemetry are not yet implemented by the PyTorch
  experiment.

These boundaries preserve the distinction between an executable prototype and
an empirical hardware-performance claim.

---

## 10. Run contract

Install the optional research dependencies and run from the repository root:

```bash
python -m pip install -e ".[dm-v3-prime]"
python experiments/dm_v3_prime_torch.py --cycles 200
```

A minimal CPU smoke run is:

```bash
python experiments/dm_v3_prime_torch.py \
  --device cpu \
  --cycles 2 \
  --latent-dim 16 \
  --timing-repeats 1 \
  --log-every 1
```

Expected terminal output includes:

```text
BASELINE
training cycles
CANDIDATE
VERIFY Pi_(H,Lambda): COMMIT | ROLLBACK
Measured speedup=<value>x
1000x target=MET | NOT MET
```

---

## 11. Canonical PRIME signature

The baseline can be represented compactly as

```text
Sigma_DM-V3-PRIME =
Pi_(H,Lambda)
 o D_phi^3D
 o Q_Delta^-1
 o C^-1
 o C
 o Q_Delta
 o [R_omega^spectral]^ell
 o E_theta,N3^3D
```

with the explicit transactional invariant

```text
Reality of measurement > symbolic acceleration claim.
```

and the deployment invariant

```text
candidate mutation != active state until verification succeeds.
```
