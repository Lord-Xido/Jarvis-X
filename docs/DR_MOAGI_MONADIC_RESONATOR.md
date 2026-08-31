# Dr Moagi DM-vOmegaXi+ Monadic Resonator

The monadic resonator is the executable reference form of the recurrence

```text
X[t+1] = D_psi(E_phi(X[t]) + integral_t^{t+1} f_theta(Z(tau), tau) d tau)
```

with the latent state made explicit:

```text
Z_t   = E_phi(X_t)
Z_t+1 = Z_t + integral_t^{t+1} f_theta(Z(tau), tau) d tau
X_t+1 = D_psi(Z_t+1)
```

The runtime deliberately keeps the encoder `E_phi`, continuous latent dynamics
`f_theta`, and decoder `D_psi` as separate callables. This prevents a numerical
implementation from conflating representation, evolution, and reconstruction.

## Operational pipeline

1. Validate a finite observable state `X_t`.
2. Encode it to `Z_t = E_phi(X_t)`.
3. Integrate `dZ/dt = f_theta(Z,t)` over a finite interval.
4. Form the next latent state by adding the accumulated integral to `Z_t`.
5. Decode `X_t+1 = D_psi(Z_t+1)`.
6. Emit an audit report containing the latent start, latent integral, latent end,
   output state, solver, substep count, derivative-evaluation count, and latent
   displacement norm.
7. For a rollout, recursively feed each decoded state into the next transition.

## Numerical boundary

The recurrence is exact as a mathematical definition. A generic nonlinear latent
ODE normally does not have an analytic closed-form integral, so this reference
runtime approximates the integral with one of two bounded fixed-step methods:

- `euler`: one derivative evaluation per substep.
- `rk4`: classical fourth-order Runge-Kutta, four derivative evaluations per
  substep.

The implementation therefore does **not** claim exact continuous-time evolution
for arbitrary learned dynamics. Solver method and computational effort remain
visible in every report.

## Stability and validation

Each observable state, latent state, derivative, and decoded result must be
non-empty and finite. Latent dimensionality may not change inside one integration
interval. `max_abs_value` bounds runaway numerical states. The integration
interval and substep count must both be positive and finite.

These are execution invariants, not claims of semantic correctness. A production
learned model can add task-specific reconstruction loss, conservation laws,
spectral constraints, or a Pi_Lambda acceptance gate around the same core law.

## Reference dynamics

For deterministic validation the package includes

```text
dZ/dt = -rate * Z + forcing
```

For zero forcing the exact scalar solution is

```text
Z(t+Delta) = Z(t) * exp(-rate * Delta)
```

which supplies an independent check of the RK4 implementation.

## CLI

```bash
jarvisx-dr-moagi-resonator \
  --state 1,0.5,-0.25 \
  --steps 4 \
  --method rk4 \
  --substeps 32 \
  --rate 0.25
```

The command prints deterministic JSON telemetry for every transition.

## Architectural role

The resonator is complementary to the existing Deep Distiller. The Deep Distiller
implements bounded residual-memory and parameter-update dynamics. The resonator
implements the continuous latent-flow equation directly. They should remain
separate primitives until an explicit higher-level controller defines how latent
ODE evolution, residual memory, parameter learning, and transactional acceptance
compose.
