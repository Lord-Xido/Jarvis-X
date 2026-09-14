# QSOL Satellite 3D Tracking Surface

This browser app is a bounded research visualization for the Jarvis-X satellite signal tracking loop.

It models the closed kinetic chain

```text
3D orbit
-> predicted range / delay / Doppler / angle
-> synthetic complex I/Q observations
-> delay-Doppler pattern matching
-> local tracking measurements
-> distributed cloud likelihood fusion
-> 3D posterior correction
-> next acquisition-window prediction
-> recur
```

The physical state and signal state are kept explicitly distinct:

```text
physical state:    x = [r, v, clock bias, clock drift]
measurement field: Xi(tau, f_D, theta, t)
posterior state:   p(x_t | z_1:t)
```

The embedded simulation uses deterministic synthetic observations only. It does not connect to SDR hardware, radios, satellites, network streams, private telemetry, encrypted payloads, or external cloud services.

## Run

Open `index.html` in a modern browser. The app is self-contained and has no external JavaScript or CSS dependencies.

Controls:

- **Run / Pause** continuously advances the recursive estimation loop.
- **Step** advances exactly one pipeline stage.
- **Reset** restores the initial state.
- **Noise** changes synthetic measurement noise.
- **Receivers** changes the number of simulated edge receptors contributing to cloud fusion.

## Mathematical model

The state transition is

```text
x_t^- = F(x_(t-1)^+) + w_t
```

Each receptor predicts a measurement

```text
z_hat_i = h_i(x_t^-)
```

and forms an innovation

```text
e_i = z_i - z_hat_i.
```

The cloud fuses independent local likelihoods as

```text
p(x_t | z_1:t) proportional_to p^-(x_t) * product_i p(z_(t,i) | x_t)
```

and the visualization represents the signal manifold as

```text
Xi(tau, f_D, theta, t).
```

The inward contraction shown by the app is uncertainty contraction in the posterior covariance, not literal physical contraction of space.

## Trust boundary

This app is a research and visualization surface. It does **not** replace `jarvisx.system_runtime`, does not mutate authoritative Jarvis-X state, and has no network, shell, filesystem, radio, SDR, satellite-control, infrastructure, or device authority.

It is intended only for lawful analysis of synthetic, public, open, or otherwise authorized signals. It does not decode protected payloads or bypass access controls.

The canonical production rule remains:

```text
prediction -> plan -> projection -> execution -> verification -> audit -> commit
```

The browser simulation is therefore an observable 3D estimator model, not a privileged execution path.
