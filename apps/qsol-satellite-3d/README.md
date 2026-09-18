# QSOL Satellite 3D Tracking Surface — Dr Moagi ANN

This browser app is a bounded research visualization for the Jarvis-X satellite signal tracking loop evolved into a 3D auto-encoding, auto-decoding, auto-adapting ANN.

It models the closed kinetic chain

```text
3D orbit
-> predicted range / delay / Doppler / angle
-> synthetic complex I/Q observations
-> delay-Doppler pattern matching
-> AUTO_ENCODE
-> LATENT_PREDICT
-> AUTO_DECODE
-> RESIDUAL_COMPARE
-> distributed CLOUD_FUSE
-> POSTERIOR_CORRECT
-> ADAPT_WEIGHTS
-> MEMORY_POLICY_UPDATE
-> recur
```

The implementation keeps three spaces explicitly distinct:

```text
physical state:    x = [r, v, clock bias, clock drift]
measurement field: Xi(tau, f_D, theta, t)
neural latent:     Z_t = E_Theta(X_t, Omega_t)
```

The unified adaptive state is

```text
S_t = [X_t, Z_t, Xhat_t, E_t, Omega_t, Theta_t, Pi_t]
```

where `Theta` contains bounded adaptive ANN weights and `Pi` controls local research-policy variables such as recursion depth and sparsity.

The embedded simulation uses deterministic synthetic observations only. It does not connect to SDR hardware, radios, satellites, network streams, private telemetry, encrypted payloads, or external cloud services.

## Run

Open `index.html` in a modern browser. The app is self-contained and has no external JavaScript or CSS dependencies.

Controls:

- **Run / Pause** continuously advances the recursive loop.
- **Step** advances exactly one pipeline stage.
- **Reset** restores the deterministic initial state and ANN weights.
- **Synthetic noise** changes observation noise.
- **Receivers** changes the number of simulated edge receptors contributing to cloud fusion.
- **Learning rate** controls the bounded browser-local ANN weight update.

## Mathematical model

The orbital estimator predicts

```text
x_t^- = F_orbit(x_(t-1)^+) + w_t
```

and receptor `i` predicts

```text
z_hat_i = h_i(x_t^-).
```

The signal manifold remains

```text
Xi(tau, f_D, theta, t).
```

The Dr Moagi ANN layer is

```text
Z_t       = E_Theta(X_t, Omega_t)
Z^-_(t+1) = F_Theta(Z_t, Omega_t)
Xhat      = D_Theta(Z^-_(t+1))
E_(t+1)   = X_(t+1) - Xhat
```

with bounded adaptive weights

```text
Theta_(t+1) = clip(Theta_t - eta_t * grad_Theta L_t)
```

and temporal memory / runtime policy

```text
Omega_(t+1) = rho*Omega_t + (1-rho)*G(E_t, Z_t)
Pi_(t+1)    = U_Pi(Pi_t, E_t, Sigma_t).
```

The browser reference uses a small dense encoder, latent predictor and decoder. Its adaptive updates are intentionally bounded and illustrative; they are not claimed to be a production training algorithm.

Cloud fusion remains probabilistic:

```text
p(x_t | z_1:t) proportional_to p^-(x_t) * product_i p(z_(t,i) | x_t)
```

and the inward contraction shown by the app is uncertainty contraction in posterior/latent state, not literal physical contraction of space.

## Trust boundary

This app is a research and visualization surface. It does **not** replace `jarvisx.system_runtime`, does not mutate authoritative Jarvis-X state, and has no network, shell, filesystem, radio, SDR, satellite-control, infrastructure, or device authority.

Adaptive ANN weights exist only in browser memory. The surface uses anonymous simulated tracks and is intended only for lawful analysis of synthetic, public, open, consented, or otherwise authorized signals. It does not decode protected payloads, identify private persons, or bypass access controls.

The canonical production rule remains:

```text
prediction -> plan -> projection -> execution -> verification -> audit -> commit
```

The browser simulation is therefore an observable 3D estimator/ANN model, not a privileged execution path.
