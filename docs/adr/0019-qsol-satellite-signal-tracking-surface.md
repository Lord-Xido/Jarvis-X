# ADR-019: QSOL Satellite Signal Tracking Research Surface with Dr Moagi ANN

- **Status:** Proposed
- **Date:** 2026-09-14
- **Decision scope:** bounded browser research visualization

## Context

Jarvis-X already contains a bounded QSOL kinetic 3D research surface and a governed production boundary. The satellite-tracking concept adds a distinct estimation problem: infer a moving physical state from radio measurements whose observable structure appears first in delay, Doppler, phase and angle space rather than directly in Cartesian position.

The research surface is now evolved with a bounded Dr Moagi ANN layer that learns a compact latent representation of synthetic kinetic/RF observables and recursively predicts, decodes, compares and adapts that representation.

The system therefore requires an explicit separation between:

1. physical Cartesian state space,
2. RF measurement space,
3. neural latent state,
4. probabilistic posterior state,
5. adaptive weights / temporal memory / runtime policy.

The research implementation must preserve the existing Jarvis-X trust boundary and must not introduce live radio, SDR, network, satellite-control or privileged device authority.

## Decision

Maintain `apps/qsol-satellite-3d/` as a self-contained browser research surface and extend its canonical loop to

```text
3D orbit
-> predicted RF geometry
-> synthetic I/Q observation
-> delay-Doppler pattern match
-> AUTO_ENCODE
-> LATENT_PREDICT
-> AUTO_DECODE
-> RESIDUAL_COMPARE
-> CLOUD_FUSE
-> POSTERIOR_CORRECT
-> ADAPT_WEIGHTS
-> MEMORY_POLICY_UPDATE
-> recur
```

The app uses deterministic synthetic observations and browser-local adaptive weights only. It does not connect to external RF or cloud infrastructure.

## State model

The bounded physical state is

```text
x_t = [r_t, v_t, b_t, bdot_t]
```

where `r_t` and `v_t` are 3D position and velocity and the remaining terms represent clock bias and drift in the estimator abstraction.

The predicted state is

```text
x_t^- = F_orbit(x_(t-1)^+) + w_t
```

and receiver `i` predicts

```text
z_hat_(t,i) = h_i(x_t^-).
```

The innovation is

```text
e_(t,i) = z_(t,i) - z_hat_(t,i).
```

## Measurement manifold

The signal-domain visualization is

```text
Xi(tau, f_D, theta, t)
```

where `tau` is delay, `f_D` is Doppler and `theta` is a compact arrival-direction coordinate for visualization. This is not identified with Cartesian physical space.

The bounded matched-filter abstraction is

```text
C_i(tau, nu) = |sum_k y_i[k] q*[k; tau] exp(-j 2 pi nu k Ts)|^2.
```

The browser app renders only a synthetic local peak representing the current hypothesis; it does not demodulate or decode real payloads.

## Dr Moagi ANN adaptive layer

Define the unified research state

```text
S_t = [X_t, Z_t, Xhat_t, E_t, Omega_t, Theta_t, Pi_t].
```

`X_t` is the bounded normalized observation vector, `Z_t` is the learned latent representation, `Xhat_t` is the reconstruction/prediction, `E_t` is the residual, `Omega_t` is temporal error/state memory, `Theta_t` is the ANN parameter state and `Pi_t` is the bounded runtime adaptation policy.

The encoder is

```text
Z_t = E_Theta(X_t, Omega_t).
```

The latent predictor is

```text
Z^-_(t+1) = F_Theta(Z_t, Omega_t).
```

The decoder produces

```text
Xhat_(t+1) = D_Theta(Z^-_(t+1)).
```

and the residual is

```text
E_(t+1) = X_(t+1) - Xhat_(t+1).
```

The browser reference minimizes a bounded reconstruction/prediction loss

```text
L_t = mean(E_t^2)
```

with clipped local parameter updates

```text
Theta_(t+1) = clip(Theta_t - eta_t * grad_Theta L_t).
```

The temporal memory and runtime policy update as

```text
Omega_(t+1) = rho*Omega_t + (1-rho)*G(E_t, Z_t)
Pi_(t+1)    = U_Pi(Pi_t, E_t, Sigma_t).
```

The reference implementation uses a small dense encoder, latent predictor and decoder. The update is intentionally illustrative and bounded; it is not represented as a production-grade training rule, autonomous code mutation, or hardware self-modification.

## Distributed fusion

Each simulated receptor contributes a local likelihood

```text
L_i(x_t) = p(z_(t,i) | x_t).
```

The cloud-style fusion state is

```text
p(x_t | z_1:t) proportional_to p^-(x_t) * product_i L_i(x_t).
```

The app visualizes the resulting uncertainty as a posterior ellipse/volume around the synthetic orbital state. The displayed inward contraction represents shrinking estimator/latent uncertainty, not spatial contraction.

## Core invariants

1. **Physical state, measurement manifold and latent state remain distinct.** `(x,y,z)` is not `Xi(tau,f_D,theta,t)` and neither is identical to `Z_t`.
2. **Visualization is not authority.** Browser state is tentative research state only.
3. **Synthetic input only.** The reference surface has no live radio, SDR, satellite-control or network source.
4. **No protected-payload recovery.** The reference model demonstrates tracking geometry, estimation and bounded neural adaptation only.
5. **Distributed fusion is explicit.** Receptor contributions remain individually modeled before global posterior fusion.
6. **Adaptive weights are bounded.** ANN parameter updates are clipped, local to browser memory and resettable.
7. **Memory and policy are explicit.** `Omega` and `Pi` remain observable state, not hidden claims of autonomous agency.
8. **Uncertainty contraction is explicit.** The inward loop is covariance/latent-error reduction around the state estimate.
9. **The canonical runtime is unchanged.** `jarvisx.system_runtime` remains the governed authoritative path.
10. **Authorized research only.** The reference surface uses synthetic or otherwise authorized/open signal abstractions and anonymous simulated tracks.

## Consequences

### Positive

- The orbit-to-RF-to-latent-to-posterior loop becomes directly inspectable in 3D.
- Delay-Doppler pattern matching stays separate from Cartesian state estimation and latent representation learning.
- ANN weights, reconstruction loss, memory and runtime policy become observable research state.
- Multi-receptor fusion and residual-driven adaptation can be explored without external hardware.
- The app remains portable and self-contained.

### Trade-offs

- The orbit model is illustrative rather than an astrodynamics-grade propagator.
- The signal field is synthetic and bounded rather than a real channel model.
- The ANN is deliberately small and its update rule is pedagogical rather than a production optimizer.
- Browser timing is not a real-time guarantee or RF/ML performance benchmark.
- The posterior contraction is a pedagogical estimator proxy, not a certified navigation solution.

## Validation

`tests/test_qsol_satellite_3d_app.py` verifies that the app remains self-contained, contains the required estimator and ANN stages, exposes the bounded adaptive-state markers and retains the research trust-boundary language.

## Promotion

This ADR remains **Proposed** until the evolved implementation passes repository CI and is merged through review.
