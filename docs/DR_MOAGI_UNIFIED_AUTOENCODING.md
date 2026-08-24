# Dr. Moagi Unified Autoencoding Reference

## Classification

This subsystem is a deterministic, dependency-free reference implementation of the proposed Dr. Moagi Unified Autoencoding (UEA) equations for the signal state

```text
s = [frequency, amplitude, phase].
```

It evaluates the objective, checks reconstruction fixed points, estimates the signal-space gradient and integrates the stated dynamical system. It is intended for equation-level validation and small auditable experiments.

It is not a production neural-network trainer, an automatic-differentiation framework, a calibrated physical signal model or evidence of convergence for arbitrary coefficients.

## 1. State geometry

The written state is

```text
s = [f, a, phi]^T.
```

Frequency and amplitude are ordinary real coordinates. Phase is periodic, so the operational state space is more accurately

```text
R × R × S1
```

than unconstrained Euclidean `R3`.

The reference therefore computes the phase residual with the shortest angular difference:

```text
d_phi = wrap(phi_hat - phi) in [-pi, pi).
```

The reconstruction metric is

```text
||s_hat - s||_G^2
  = w_f (f_hat - f)^2
  + w_a (a_hat - a)^2
  + w_phi wrap(phi_hat - phi)^2.
```

All metric weights must be finite and positive.

## 2. Probabilistic encoder

A KL divergence against a standard normal requires a probability distribution rather than a single deterministic code. The reference makes that distribution explicit:

```text
q_phi(z|s)
  = Normal(mu_phi(s), diag(exp(logvar_phi(s)))).
```

The current transparent model is linear:

```text
mu_phi(s)     = W_mu s + b_mu
logvar_phi(s) = W_v s + b_v
z             = mu_phi(s) + exp(0.5 logvar_phi(s)) elementwise* epsilon
s_hat         = W_D z + b_D.
```

Deterministic reconstruction uses the posterior mean, `z = mu_phi(s)`. A caller may supply an explicit epsilon vector to test a reproducible sampled path.

For a three-dimensional diagonal Gaussian, the implemented KL term is

```text
KL(q_phi(z|s) || Normal(0, I))
  = 0.5 sum_i [exp(logvar_i) + mu_i^2 - 1 - logvar_i].
```

## 3. Unified objective

For a batch `B`, the reference computes

```text
L_Moagi
  = mean_s in B ||s - D(E(s))||_G^2
  + sum_T mean_s in B ||T(s) - D(E(T(s)))||_G^2
  + beta mean_s in B KL(q_phi(z|s) || Normal(0, I)),
```

where

```text
T in {M, F, N}.
```

The returned `LossBreakdown` exposes:

- base reconstruction;
- one reconstruction value for each operation;
- mean KL regularization;
- total objective.

### Terminology boundary

The operation term requires the autoencoder to reconstruct each transformed input. This is operation-conditioned reconstruction or closure under the transformation family.

It is not strict latent invariance, which would require an additional term such as

```text
||E(T(s)) - E(s)||^2.
```

It is also not equivariance unless a latent operation `rho(T)` is defined and constrained through

```text
E(T(s)) approximately equals rho(T) E(s).
```

The implementation preserves the submitted equation and documents this distinction explicitly.

## 4. Operations M, F and N

The reference provides deterministic affine operations:

```text
T(s) = A_T s + b_T.
```

The default fixtures are interpretable rather than physically calibrated:

- `M`: amplitude scaling plus a phase offset;
- `F`: amplitude attenuation;
- `N`: a fixed additive perturbation.

A stochastic noise process cannot satisfy a pointwise deterministic fixed-point condition without conditioning on a realization. For reproducible tests, `N` is therefore one frozen deterministic realization. A production stochastic implementation should approximate the expectation with seeded Monte Carlo samples.

## 5. Reconstruction fixed point

The submitted fixed-point condition is evaluated as

```text
r_0(s) = s - D(E(s))
r_T(s) = T(s) - D(E(T(s))).
```

The subsystem reports the RMS magnitude of every residual and declares reconstruction fixed-point satisfaction when

```text
max(RMS(r_0), RMS(r_M), RMS(r_F), RMS(r_N)) <= tolerance.
```

This is a reconstruction fixed point. It is distinct from a full dynamical equilibrium when operation forcing is nonzero.

## 6. Signal-space gradient flow

The dynamic equation is

```text
ds/dt
  = -gamma grad_s L_Moagi
    + lambda_m M(s)
    + lambda_f F(s)
    + lambda_n N(s).
```

Because the operation functions may be arbitrary deterministic transforms, the reference estimates `grad_s L_Moagi` by a centered finite difference on each state coordinate:

```text
partial L / partial s_i
  approximately
  [L(s + h e_i) - L(s - h e_i)] / (2h).
```

This is slow but explicit and testable. Parameter gradients with respect to encoder and decoder weights are intentionally left to a future autograd-backed implementation.

### Transform-versus-vector-field interpretation

A state transform `T(s)` and a velocity field are not dimensionally identical concepts. The runtime therefore exposes two forcing modes:

```text
absolute: lambda_T T(s)
delta:    lambda_T [T(s) - s].
```

`absolute` reproduces the submitted equation literally.

`delta` is the default operational mode when `M`, `F` and `N` are supplied as state transformations. It converts each transform into the displacement it proposes from the current state. Identity operations then contribute zero forcing.

Phase displacement in delta mode uses the shortest circular residual.

## 7. Time integration

One explicit Euler step is

```text
s_(t+dt)
  = Pi_bounds[
      s_t
      + dt (
          -gamma grad_s L_Moagi
          + operation_forcing(s_t)
        )
    ].
```

`Pi_bounds` is optional. It can constrain frequency and amplitude and always wraps the resulting phase.

The equilibrium runner records:

- every accepted state;
- objective history;
- derivative-norm history;
- convergence status;
- executed step count.

It terminates when

```text
||ds/dt||_2 <= tolerance
```

or when the configured maximum number of steps is exhausted.

## 8. Complete mechanistic loop

```text
LOAD s = [f, a, phi]
  -> APPLY M, F, N
  -> ENCODE each base/transformed state
  -> FORM q(z|s) = Normal(mu, diag(exp(logvar)))
  -> DECODE posterior means
  -> COMPUTE circular reconstruction residuals
  -> ACCUMULATE base + operation + beta*KL objective
  -> CHECK reconstruction fixed point
  -> PERTURB each signal coordinate by +/-h
  -> ESTIMATE grad_s L_Moagi
  -> ADD weighted operation forcing
  -> EULER UPDATE
  -> PROJECT bounds and wrap phase
  -> RECORD telemetry
  -> REPEAT until derivative equilibrium or cycle limit
```

## 9. Public API

```python
from jarvisx.unified_autoencoding import (
    DrMoagiUEA,
    MoagiCoefficients,
    Signal3D,
)

engine = DrMoagiUEA(
    coefficients=MoagiCoefficients(beta=1.0e-3, gamma=0.25)
)
signal = Signal3D(440.0, 0.8, 0.25)

loss = engine.loss((signal,))
report = engine.fixed_point_report(signal)
trace = engine.run_to_equilibrium(signal, max_steps=10)
```

The runnable example is in

```text
examples/unified_autoencoding_reference.py
```

## 10. Relationship to hierarchical 3D fractional smoothing

The UEA state has three signal coordinates; it is not by itself a spatial `x-y-z` volume.

To couple it to the existing fractional 3D solver, define one spatial field for each component:

```text
f(x,y,z), a(x,y,z), phi(x,y,z).
```

Then compute the local UEA gradient at each voxel and fractionally smooth each component of that gradient before applying the state update:

```text
g_tilde_i
  = exp[-tau D (-Delta)^alpha] g_i,

partial s_i / partial t
  = -gamma g_tilde_i + operation_forcing_i.
```

A hierarchy can use separate `(alpha_l, tau_l)` schedules for coarse and fine gradient fields. This adapter is a follow-on integration, not part of the present pull request.

## 11. Correctness and convergence boundary

The implementation guarantees validation, deterministic arithmetic for identical inputs and explicit telemetry. It does not guarantee that the Euler trajectory converges.

Convergence depends on factors including:

- objective smoothness;
- finite-difference step;
- Euler time step;
- coefficient signs and magnitudes;
- operation stability;
- projection bounds;
- whether forcing admits an equilibrium.

A reconstruction fixed point satisfies the decoder residual equations. A full dynamic equilibrium instead satisfies

```text
-gamma grad_s L_Moagi
+ lambda_m V_M(s)
+ lambda_f V_F(s)
+ lambda_n V_N(s)
= 0,
```

where `V_T(s)` is either `T(s)` or `T(s)-s`, according to the selected forcing mode.

These two equilibrium concepts coincide only under additional assumptions.
