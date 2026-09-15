# Dr Moagi Cloud OS

`jarvisx.cloud_os` operationalizes the inward 3D auto-encoding/decoding loop as a bounded **user-space cloud control plane**.

It is intentionally not presented as a bare-metal operating-system kernel, hypervisor, or electromagnetic hardware implementation. The runtime provides the OS-like control-plane functions that can be made real in the current Jarvis-X Python stack: resource registration, deterministic scheduling, auto-encoding execution, self-scoring candidate selection, journaling, verification, and HTTP control.

## Operational equation

The software realizes the constrained cycle

```text
Sigma(t+1) = Pi_Lambda[ Select_m( Decode(Encode(Sigma(t); m)) - Error + Omega - grad(J) ) ]
```

with the concrete mapping:

| Symbol | Runtime mechanism |
|---|---|
| Sigma^3D | `Field3D` plus cloud job/runtime state |
| Encode | deterministic 3D region-mean encoder |
| Decode | deterministic 3D broadcast decoder |
| P(1:M) | candidate latent-shape exploration |
| Error | reconstruction mean-squared error |
| Omega | hash-chain event ledger |
| J | reconstruction error + compression complexity |
| Pi_Lambda | shape, finiteness, capacity, concurrency and API validation |
| commit | verified job result journaled as `job.committed` |

The reference auto-optimizer evaluates candidate latent geometries and minimizes:

```text
J(m) = MSE(X, X_hat_m) + lambda * (latent_cells / source_cells)
```

This produces an explicit quality-versus-compression trade-off rather than an unbounded claim of autonomous optimization.

## Architecture

```text
                  +----------------------------+
                  |       FastAPI control      |
                  | /nodes /roundtrip /optimize|
                  +--------------+-------------+
                                 |
                                 v
+---------------+       +----------------------+
| virtual nodes |------>|  DrMoagiCloudOS      |
| capacity/load |       | scheduler + jobs     |
+---------------+       +----------+-----------+
                                   |
                          +--------+---------+
                          v                  v
                 3D autoencoder      candidate search
                 encode/decode       score + select
                          |                  |
                          +--------+---------+
                                   v
                         HashChainLedger
                      verify-before-commit
```

The reference scheduler is synchronous. A request is never hidden in an unbounded background queue: it either receives an eligible node, runs to a terminal state, and is committed, or fails explicitly.

## Run locally

```bash
python -m pip install -e .
JARVISX_CLOUD_LEDGER=state/cloud-os-ledger.jsonl \
  uvicorn jarvisx.cloud_service:app --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:8080/docs` for the generated API documentation.

## Run as a container

```bash
docker compose -f docker-compose.cloud.yml up --build
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

## Execute a 3D round trip

```bash
curl -X POST http://127.0.0.1:8080/v1/roundtrip \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "demo-roundtrip-001",
    "field": {
      "shape": [2, 2, 2],
      "values": [0,1,2,3,4,5,6,7]
    },
    "latent_shape": [1,1,1]
  }'
```

## Execute inward auto-optimization

```bash
curl -X POST http://127.0.0.1:8080/v1/auto-optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "demo-optimize-001",
    "field": {
      "shape": [2, 2, 2],
      "values": [0,1,2,3,4,5,6,7]
    },
    "complexity_weight": 0.1,
    "candidates": [[1,1,1], [2,2,2]]
  }'
```

The response includes the selected latent geometry, encoded state, reconstruction, MSE, compression ratio, objective value, deterministic result digest, selected node, and committed job state.

## Authentication

Authentication is disabled by default for local development. To require an API key:

```bash
export JARVISX_CLOUD_API_KEY='replace-me'
```

Then protected `/v1/*` requests must include:

```text
X-API-Key: replace-me
```

`/health` remains unauthenticated for container/orchestrator probes.

## State and replay

The ledger is newline-delimited canonical JSON. Each record contains the previous record digest and its own SHA-256 digest. The service refuses to load a corrupt chain.

Verify the live chain:

```bash
curl http://127.0.0.1:8080/v1/ledger/verify
```

## Cloud boundary

This reference runtime is deployable on a single Docker host and exposes the control-plane abstractions needed to evolve toward a distributed system. Production multi-host cloud operation still requires external infrastructure that this change intentionally does not pretend to implement: durable distributed storage, authenticated node-to-node transport, workload isolation, consensus or queueing, service discovery, observability, secrets management, TLS, autoscaling, and provider-specific compute adapters.

That boundary is deliberate: **Working -> Robust -> Portable -> Elegant -> Advanced.**
