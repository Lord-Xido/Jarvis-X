# Jarvis-X 3D Bit Code End-to-End Runtime

## Status

This is the runnable reference path that connects network ingress to a bounded 3D Q16.16 encode/decode execution loop, deterministic verification, and out-of-band telemetry.

It is deliberately narrower than the canonical 256-bit swarm ISA. The 32-bit words are the compact edge/lowering ABI used by this vertical slice; they do not replace the higher-level canonical instruction frame.

## Runtime path

```text
HTTP/2-capable edge (Envoy)
  -> FastAPI ingress
  -> bounded host staging
  -> FP input validation
  -> Q16.16 saturating conversion
  -> 3D spatial packing
  -> ENCODE3D contraction
  -> latent core
  -> DECODE3D expansion
  -> deterministic VERIFY
  -> response emission

                 +-> /metrics -> Prometheus
```

The 3D coordinate attached to every instruction is interpreted as:

- `x`: pipeline position;
- `y`: logical execution tick;
- `z`: execution depth, from ingress at `z=0` to latent core at `z=4` and back out.

## Compact 32-bit instruction

```text
31          24 23  20 19  16 15  12 11                 0
+-------------+------+------+------+---------------------+
| opcode 8 b  | dst  | src1 | src2 | signed immediate 12 |
+-------------+------+------+------+---------------------+
```

The current end-to-end program is:

```text
NET_RX
HOST_STAGE
Q16_CONVERT
PACK3D
ENCODE3D
LATENT_WRITE
DECODE3D
VERIFY
TELEMETRY
EMIT
HALT
```

`ENCODE3D` uses deterministic block-average contraction. `DECODE3D` expands each latent block back across its source region. This gives the runtime a concrete, testable codec today while keeping the backend contract stable for a future CUDA implementation.

## Numerical representation

Inputs are converted to Q16.16:

\[
q = \operatorname{clip}(\operatorname{round}(x 2^{16}), -2^{31}, 2^{31}-1)
\]

Non-finite values are rejected. Values outside the fixed-point range saturate and increment the telemetry clipping counter.

The verifier computes reconstruction MSE, maximum absolute error, and SHA-256 over the reconstructed signed 32-bit Q16.16 cells. A request is marked `passed` when maximum absolute error is within the caller's tolerance.

## Resource bounds

The reference runtime defaults to a maximum of 1,048,576 active input voxels per request. Shape and payload length must agree exactly. These bounds are part of the fail-closed execution contract rather than advisory metadata.

## Run locally

```bash
docker compose -f deploy/bitcode3d/docker-compose.yml up --build
```

Ingress is exposed on port `8080`, Envoy admin on `9901`, and Prometheus on `9090`.

Example request:

```bash
curl -sS \
  -H 'content-type: application/json' \
  -d '{
    "shape": [2, 2, 2],
    "values": [0, 1, 2, 3, 4, 5, 6, 7],
    "pool": 2,
    "tolerance": 4
  }' \
  http://127.0.0.1:8080/v1/bitcode3d/execute
```

Health:

```bash
curl -sS http://127.0.0.1:8080/healthz
```

Metrics:

```bash
curl -sS http://127.0.0.1:8080/metrics
```

## What is operational now

- networked request/response service;
- 32-bit compact Bit Code compilation and decode;
- 3D spatial instruction coordinates;
- bounded Q16.16 conversion with saturation accounting;
- deterministic 3D encode/latent/decode execution;
- closed-loop verification and checksum;
- Prometheus telemetry endpoint;
- Envoy ingress tier;
- Docker deployment;
- unit and container smoke CI.

## Backend boundary

The current `ENCODE3D` and `DECODE3D` operators are CPU reference kernels. Their observable contract is intentionally separated from the backend. A CUDA implementation can replace those operators while preserving:

1. the 32-bit lowering words;
2. Q16.16 semantics;
3. shape and resource checks;
4. deterministic verification;
5. API and telemetry schemas.

That boundary prevents the architecture from claiming GPU execution before GPU kernels are actually compiled and validated on supported hardware.
