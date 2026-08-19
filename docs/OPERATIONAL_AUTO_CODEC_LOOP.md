# Operational Auto-Encoding / Decoding Loop

## Status

Integration candidate on `agent/operational-auto-codec-loop`.

This implementation turns the existing bounded Dr Moagi sparse-field codec step into an executable closed control loop. It is intentionally a deterministic reference runtime, not a claim of a trained neural autoencoder, unrestricted autonomy, or production-scale compression.

## Closed-loop execution

For field state `Psi_t`, codec encoder `E`, decoder `D`, and reconstruction residual

`R_t = Psi_t - D(E(Psi_t))`,

the existing field runtime evaluates

`dPsi/dt = -alpha R_t + lambda * Laplacian(R_t) + eta * G_moagi(Psi_t)`

and commits a projected candidate only after the runtime's validation and resource guards accept it.

The new controller repeatedly executes:

1. ingest a sparse 3D field;
2. encode it into a bounded latent representation;
3. decode exactly the requested sparse support;
4. calculate reconstruction error;
5. evaluate the Dr Moagi field update;
6. project and validate the candidate transaction;
7. commit or roll back;
8. hash the resulting state into the Omega journal;
9. test explicit stop criteria;
10. repeat until convergence, fixed point, rejection limit, or cycle budget.

Operationally:

`INPUT -> ENCODE -> LATENT -> DECODE -> RECONSTRUCT -> RESIDUAL -> UPDATE -> VERIFY -> COMMIT -> JOURNAL -> LOOP`

## Reference codec

`UniformQuantizedFieldCodec` provides a real deterministic encode/decode transform for the reference loop. Each active scalar value is quantized with an explicit step size and reconstructed from its integer code. Zero codes may be pruned from the latent map.

It is deliberately simple so reconstruction error and loop behavior remain inspectable. Learned codec backends can implement the existing `FieldCodec` protocol without changing the controller.

## Stop conditions

`AutoCodecLoopConfig` bounds every run with:

- `max_cycles` — hard upper execution bound;
- `min_cycles` — minimum work before convergence is accepted;
- `reconstruction_mse_target` — explicit reconstruction threshold;
- `max_consecutive_rejections` — circuit breaker for rejected candidates;
- `stop_on_fixed_point` — optional digest-based fixed-point termination.

Every run returns a machine-readable receipt containing cycle counts, convergence state, final reconstruction MSE, final sparse-state digest, journal verification status, journal head hash, and final sparse field.

## Cloud API

The FastAPI service exposes:

- `GET /` — executable browser dashboard;
- `GET /health` — health probe;
- `POST /run` — deterministic Jarvis-X VM execution;
- `POST /codec/run` — bounded auto-encoding/decoding loop.

Example request:

```json
{
  "cells": [
    {"x": 2, "y": 2, "z": 2, "value": 0.26},
    {"x": 3, "y": 2, "z": 2, "value": -0.37}
  ],
  "side": 64,
  "quantization_step": 0.1,
  "alpha": 1.0,
  "lambda_residual": 0.0,
  "eta": 0.0,
  "dt": 0.1,
  "expand_halo": false,
  "max_cycles": 64,
  "reconstruction_mse_target": 0.001
}
```

## CLI

Run the reference loop from a JSON file:

```bash
jarvisx codec examples/auto_codec_run.json
```

Start the browser/API service:

```bash
jarvisx api
```

or:

```bash
jarvisx web
```

## Container deployment

The container now starts the actual package application:

```bash
uvicorn jarvisx.api:app --host 0.0.0.0 --port "$PORT"
```

This replaces the previous invalid `src.main:app` entry point.

## Verification boundary

The runtime verifies software invariants only: bounded support, numeric projection, candidate commit/rollback, deterministic sparse serialization, cycle limits, reconstruction telemetry, and the hash-chained Omega journal. A passing run does not establish model quality, security isolation, physical-computation claims, or superiority to learned autoencoder baselines.
