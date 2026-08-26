# Moagi Physical-Latent Field Simulator

A bounded browser simulation of the 3-bit Moagi Physical-Latent Operator `M`.

The simulator models three binary predicates:

- `b2`: physical/link path is inside the configured latency/jitter/error envelope;
- `b1`: processing path is deterministic enough to meet the configured execution envelope;
- `b0`: the synthetic microstructure state is favourable rather than toxic.

The decoded state is

```text
state = (b2 << 2) | (b1 << 1) | b0
```

and maps all eight states to deterministic policy actions. Aggressive action is additionally protected by a generation/epoch commit barrier so a stale `111` decision cannot be transmitted after a newer synthetic snapshot changes the state.

## What this app verifies

- all 8 truth-table states map deterministically;
- the `111 -> 110` stale-decision race is rejected at the commit barrier;
- malformed/stale decisions fail closed;
- synthetic anomaly scenarios produce observable state transitions;
- the nominal `18.45 ns` path is treated as an architectural target budget, not as measured FPGA or exchange latency.

## Included anomaly presets

- **Burst Cancel**: favourable state collapses from `111` to `110` before commit;
- **Liquidity Vacuum**: repeated `b0` loss with defensive repricing;
- **Link Degrade**: `b2` drops while market structure remains favourable;
- **Compute Jitter**: `b1` drops while the physical path remains healthy;
- **Mixed Storm**: deterministic seeded multi-axis perturbation.

## Run

Open `index.html` in a modern browser. The app is self-contained and has no external JavaScript or CSS dependencies.

## Trust boundary

This is a synthetic research and visualization surface only. It has no exchange credentials, no broker connectivity, no live market-data dependency, no network transmission authority, and no production order-routing path.

`18.45 ns` is displayed only as the supplied target local decision-path budget:

```text
1.25 ns + 4.80 ns + 12.40 ns = 18.45 ns
```

Actual hardware determinism would require static timing closure, post-route timing, clock-domain/metastability analysis, PVT characterization, and physical measurements. Actual market latency additionally includes network, gateway, matching-engine, and return-path components.

The simulator therefore validates operator semantics and race-handling, not a claim of measured hardware or exchange performance.
