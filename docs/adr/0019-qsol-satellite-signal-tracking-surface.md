# ADR-019: QSOL Satellite Signal Tracking Research Surface

- **Status:** Proposed
- **Date:** 2026-09-14
- **Decision scope:** bounded browser research visualization

## Context

Jarvis-X already contains a bounded QSOL kinetic 3D research surface and a governed production boundary. The satellite-tracking concept adds a distinct estimation problem: infer a moving physical state from radio measurements whose observable structure appears first in delay, Doppler, phase and angle space rather than directly in Cartesian position.

The system therefore requires an explicit separation between:

1. physical state space,
2. RF measurement space,
3. probabilistic posterior state,
4. cloud-style distributed likelihood fusion.

The research implementation must preserve the existing Jarvis-X trust boundary and must not introduce live radio, SDR, network, satellite-control or privileged device authority.

## Decision

Add `apps/qsol-satellite-3d/` as a self-contained browser research surface.

The canonical research loop is

```text
3D orbit
-> predicted RF geometry
-> synthetic I/Q observation
-> delay-Doppler pattern match
-> local measurement track
-> cloud likelihood fusion
-> posterior correction
-> next acquisition-window prediction
-> recur
```

The app uses deterministic synthetic observations. It does not connect to external RF or cloud infrastructure.

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

## Distributed fusion

Each simulated receptor contributes a local likelihood

```text
L_i(x_t) = p(z_(t,i) | x_t).
```

The cloud-style fusion state is

```text
p(x_t | z_1:t) proportional_to p^-(x_t) * product_i L_i(x_t).
```

The app visualizes the resulting uncertainty as a posterior ellipse/volume around the synthetic orbital state. The displayed inward contraction represents shrinking estimator covariance, not spatial contraction.

## Core invariants

1. **Physical state and signal manifold remain distinct.** `(x,y,z)` is not `Xi(tau,f_D,theta,t)`.
2. **Visualization is not authority.** Browser state is tentative research state only.
3. **Synthetic input only.** The reference surface has no live radio, SDR, satellite-control or network source.
4. **No protected-payload recovery.** The reference model demonstrates tracking geometry and estimator mechanics only.
5. **Distributed fusion is explicit.** Receptor contributions remain individually modeled before global posterior fusion.
6. **Uncertainty contraction is explicit.** The inward loop is covariance reduction around the state estimate.
7. **The canonical runtime is unchanged.** `jarvisx.system_runtime` remains the governed authoritative path.

## Consequences

### Positive

- The orbit-to-RF-to-posterior loop becomes directly inspectable in 3D.
- Delay-Doppler pattern matching is separated from Cartesian state estimation.
- Multi-receptor fusion can be explored without external hardware.
- The app remains portable and self-contained.

### Trade-offs

- The orbit model is illustrative rather than an astrodynamics-grade propagator.
- The signal field is synthetic and bounded rather than a real channel model.
- Browser timing is not a real-time guarantee or RF performance benchmark.
- The posterior contraction is a pedagogical estimator proxy, not a certified navigation solution.

## Validation

`tests/test_qsol_satellite_3d_app.py` verifies that the app remains self-contained, contains the required 3D estimator stages and retains the research trust-boundary language.

## Promotion

This ADR remains **Proposed** until the implementation passes repository CI and is merged through review.
