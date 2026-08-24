# The Dr. Moagi Equation System · E₈

This document is the executable specification for the eight-stage geometry → latent → reconstruction → evaluation → heredity → control loop implemented in `src/jarvisx/dr_moagi_e8.py`.

## Canonical equations

### M₁ — Form Map

\[
r(\alpha)=\left(|\cos(m\alpha/4)|^{n_2}+|\sin(m\alpha/4)|^{n_3}\right)^{-1/n_1}
\]

\[
x(\theta,\phi)=S\left(r_1(\phi)\cos\phi\,r_2(\theta)\cos\theta,\;r_1(\phi)\sin\phi\,r_2(\theta)\cos\theta,\;r_2(\theta)\sin\theta\right)
\]

M₁ maps a bounded 12-parameter genome into a sampled 3D phenotype.

### M₂ — Encoder

\[
\hat p=p/\|p\|,\qquad k^*(p)=\arg\max_k\langle\hat p,\hat N_k\rangle
\]

\[
c_k=\sum_{i:k^*(p_i)=k}\|p_i\|
\]

A Fibonacci sphere provides the default fixed directional codebook `N_k`. The encoder stores the nearest directional index and accumulated radius for each active node.

### M₃ — Latent Code

\[
\bar r_k=c_k/n_k
\]

\[
H(c)=-\frac{\sum_k p_k\ln p_k}{\ln K}
\]

Operational convention: `p_k = c_k / Σ_j c_j`, so the entropy is a normalized radial-mass entropy. Empty nodes receive `r̄_k = 0`.

### M₄ — Decoder

\[
\hat x_i=\hat N_{k^*(i)}\bar r_{k^*(i)}+\epsilon_i,\qquad
\epsilon_i\sim U(-\sigma_\epsilon,\sigma_\epsilon)
\]

\[
\sigma_\epsilon=\operatorname{clamp}(0.5L,0.02,0.22)
\]

The runtime accepts an explicit `loss_signal`. If omitted, it uses same-cycle noiseless vector-quantization distortion as the signal that sets `σ_ε`, then computes the final M₅ loss after perturbation. This resolves execution ordering without redefining the equation.

### M₅ — Loss

\[
L=\frac1P\sum_{i=1}^{P}\|x_i-\hat x_i\|
\]

This is mean Euclidean reconstruction distortion, not MSE.

### M₆ — Value Functional

\[
F(g)=1.5e^{-(\mu-1.1)^2/0.22}+1.3e^{-(cv-0.3)^2/0.07}+0.6\min(r_{max}/2.6,1)+0.35e^{-(tw-1.1)^2/1.4}
\]

`F` scores phenotype characteristics and is intentionally separate from the codec loss `L`.

### M₇ — Heredity

\[
(A,B)=\operatorname{tournament}(pop,3)
\]

\[
g_{t+1}=\mathcal M_\sigma(mask\odot A\oplus\bar{mask}\odot B)
\]

\[
\mathcal M_\sigma:g_j\leftarrow g_j+\mathcal N(0,\sigma^2)\Delta_j\quad[p_m]
\]

The implementation bounds every mutated gene back to its canonical domain and preserves integer constraints on `m1` and `m2`.

### M₈ — Master Equation

\[
\tilde L=\min(L/0.35,1),\qquad \tilde\epsilon=\min(\epsilon/0.30,1)
\]

\[
\lambda=\operatorname{clamp}\left(0.16+0.5(1-C)+0.2\tilde L+0.25\tilde\epsilon,0.04,1\right)
\]

\[
v_{clock}=1.25-0.8\lambda
\]

The executable API keeps `C` explicit. It does **not** infer coherence from entropy, fitness, or reconstruction quality. If scalar `epsilon` is omitted, the pipeline uses the measured mean magnitude of the realized decoder perturbations.

## Closed operational loop

```text
g_t
  │
  ▼
M1 Form Map ──► X_t
  │
  ▼
M2 Encoder ───► (k*, c, n)
  │
  ▼
M3 Latent ────► (r̄, H)
  │
  ▼
M4 Decoder ───► X̂_t + ε
  │
  ▼
M5 Loss ──────► L_t
  │
  ├────────────► M8 Governor(C, L_t, ε_t) ─► λ_t ─► v_clock,t
  │
  ▼
M6 Value ─────► F(g_t)
  │
  ▼
M7 Heredity ──► g_{t+1}
  └──────────────────────────────────────────────► next cycle
```

## Runtime API

```python
import random
from jarvisx.dr_moagi_e8 import DrMoagiE8System, random_genome

rng = random.Random(42)
system = DrMoagiE8System(k=48, resolution=18)
genome = random_genome(rng)

result = system.forward(genome, coherence=0.90, rng=rng)
print(result.loss)
print(result.fitness)
print(result.governor.lambda_value)
print(result.governor.clock_velocity)
```

The governor is deliberately bounded:

- `0.04 <= λ <= 1.0`
- `0.45 <= v_clock <= 1.218`
- increasing control pressure lowers clock velocity.

## Scope

E₈ is a concrete reference runtime for the mathematical system above. It does not claim that the geometric latent code is a trained neural representation, nor that the evolutionary objective is a universal intelligence metric. Those are empirical questions that require benchmark evidence beyond the equations themselves.
