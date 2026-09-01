# Jarvis-X HyperCloud Operations

**Status:** operational permeated single-node/stateful reference deployment  
**Scope:** durable sparse state, content-addressed multimedia, deterministic 3D routing, renewable leased multiparallel workers, persistent jobs, codec execution, pluggable chat inference

## 1. Start the complete stack

```bash
cp .env.hypercloud.example .env
# edit .env if authentication or a model backend is required
docker compose --env-file .env -f docker-compose.hypercloud.yml up -d --build
```

The stack contains:

- `api`: FastAPI control plane on port 8080 by default;
- `worker-a`: leased execution cell at `(4,4,4)` by default;
- `worker-b`: leased execution cell at `(12,12,12)` by default;
- `hypercloud-state`: shared durable volume containing SQLite/WAL state and content-addressed objects.

Verify:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/readyz
curl http://127.0.0.1:8080/metrics
curl http://127.0.0.1:8080/v1/workers
```

If `JARVISX_API_KEY` is set, add either `Authorization: Bearer <key>` or `X-API-Key: <key>` to `/v1/*` calls.

## 2. Permeated 3D execution loop

Every executable job is assigned a deterministic target coordinate in the finite deployment lattice:

```text
request / media digest
        |
        v
hierarchical sparse address
        |
        v
3D target (x,y,z)
        |
        v
live worker registry
        |
        v
nearest capable worker
        |
        v
renewable ownership lease
        |
        v
execute -> verify -> commit
```

The worker registry records worker identity, 3D coordinate, capability set, backend identity, current load and heartbeat timestamp.

Placement is intentionally transparent. The scheduler scores compatible workers using spatial distance plus a bounded load penalty. A queued worker claim independently prefers spatially nearest jobs, so geometry participates in execution rather than remaining decorative metadata.

Inspect a job's target and current placement recommendation:

```bash
curl http://127.0.0.1:8080/v1/jobs/<id>/placement
```

## 3. Execute a chat job

With the default configuration workers use the deterministic local reference backend. This validates the complete persistent execution path but is **not** represented as a neural LLM.

```bash
python examples/hypercloud_client.py "Explain the 3D sparse routing model"
```

Direct API sequence:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"namespace":"demo","prompt":"Explain sparse virtual parameters"}' \
  http://127.0.0.1:8080/v1/chat
```

The response contains a durable `id` and a non-null 3D `target`. Poll `/v1/jobs/<id>` until `status` becomes `succeeded` or `failed`.

## 4. Worker leases and failure recovery

The durable job state machine is:

```text
queued
  |
  | claim + lease
  v
running ------------------> succeeded
  |   ^
  |   | active owner renews lease
  |---+
  |
  +-----------------------> failed
  |
  | lease expires, attempts remain
  v
queued
```

Each claim records `lease_owner`, `lease_expires_at`, `attempts`, and `max_attempts`. While execution is active, the worker periodically renews both its job lease and its registry heartbeat. Renewal succeeds only for the current `lease_owner`.

If a worker disappears, renewal stops. Expired running work is automatically returned to `queued` while attempts remain. When the maximum attempt count is exhausted, the job becomes `failed` with an explicit lease-expiry error.

Completion, failure, and renewal writes are fenced by `lease_owner`; a stale worker cannot overwrite or extend a job after ownership has moved to another worker.

Reference settings:

```text
JARVISX_JOB_LEASE_SECONDS=300
JARVISX_WORKER_TTL_SECONDS=30
JARVISX_WORKER_POLL_SECONDS=0.25
```

Workers renew an active lease at approximately one-third of the configured lease interval, capped to keep heartbeats responsive.

## 5. Attach a real model server

HyperCloud can execute against an OpenAI-compatible chat-completions endpoint. This may be a self-hosted inference server or a managed provider.

Configure:

```bash
export JARVISX_MODEL_BASE_URL=http://model-server:8000
export JARVISX_MODEL_NAME=my-model
export JARVISX_MODEL_API_KEY='optional-secret'
docker compose -f docker-compose.hypercloud.yml up -d
```

Workers send `POST ${JARVISX_MODEL_BASE_URL}/v1/chat/completions` using `JARVISX_MODEL_NAME`. API keys are read from the environment and are never persisted in the HyperCloud state database.

A configured external model endpoint proves only that a model backend is attached. It does not by itself prove GPU acceleration, model size, throughput, or quality; those require backend telemetry and benchmarks.

## 6. Persist sparse parameters

Write an addressed value:

```bash
curl -sS -X PUT \
  -H 'Content-Type: application/json' \
  -d '{
        "namespace":"demo/model-a",
        "modality":"text",
        "digits":[17,42,999999],
        "value":0.875
      }' \
  http://127.0.0.1:8080/v1/parameters
```

Read it by POSTing the same namespace, modality and digits to `/v1/parameters/read`.

Physical allocation remains proportional to touched addresses; the symbolic astronomical upper bound is never expanded densely.

## 7. Ingest and auto-encode/decode multimedia

Media ingestion accepts base64 bytes and an explicit modality at `POST /v1/media`. The returned SHA-256 digest is scoped by namespace. Submit an executable codec round-trip by POSTing the namespace and digest to `/v1/jobs/codec`.

A codec-capable worker performs:

```text
media bytes
  -> zlib-deflate latent packet
  -> verified decode
  -> byte-for-byte reconstruction check
```

This is a real deterministic auto-encoding/decoding transport primitive, explicitly a lossless codec rather than a learned neural autoencoder. Learned image/audio/video codecs can attach later without changing the job/state contract.

## 8. Data layout

Default container paths:

```text
/var/lib/jarvisx/state.db
/var/lib/jarvisx/state.db-wal
/var/lib/jarvisx/state.db-shm
/var/lib/jarvisx/objects/<sha-prefix>/<sha-prefix>/<sha256>
```

SQLite stores sparse parameter values, namespace-scoped media metadata, durable job inputs/results/errors, 3D job targets and lease state, and worker coordinates/capabilities/load/heartbeats.

Object bytes are content-addressed and deduplicated globally by SHA-256 while metadata ownership remains namespace-scoped.

## 9. Observability

Prometheus text metrics are available at `GET /metrics`. Current gauges include materialized sparse parameters, namespace-scoped media objects, active workers inside the heartbeat TTL, and queued/running/succeeded/failed job counts.

Worker topology is available at `GET /v1/workers`. Readiness checks SQLite availability at `/readyz`; liveness is exposed at `/healthz`.

## 10. Kubernetes

Apply:

```bash
kubectl apply -f deploy/k8s/hypercloud.yaml
```

The manifest intentionally runs **one pod** containing API plus two worker sidecars sharing a `ReadWriteOnce` persistent volume. This exercises concurrent leased workers while respecting SQLite/WAL's single-node state boundary.

Do not increase pod replicas while using this SQLite state plane. Cross-node horizontal scale requires a distributed transactional job/metadata store and distributed object store first.

## 11. Operational capability boundary

Implemented and runnable:

- symbolic astronomical virtual parameter namespace;
- finite deterministic 3D shard mapping;
- durable sparse parameters;
- namespace-scoped multimedia metadata;
- content-addressed object persistence;
- lossless multimedia encode/decode execution;
- durable asynchronous jobs;
- deterministic 3D job targets;
- multiworker registration and heartbeat state;
- capability-aware spatial placement;
- expiring and renewable worker leases;
- abandoned-job recovery and bounded retries;
- lease-owner fencing for renewal/completion/failure;
- offline local response backend;
- OpenAI-compatible neural model adapter;
- optional API-key enforcement;
- readiness/liveness/metrics;
- Docker Compose two-cell worker deployment;
- Kubernetes single-pod/two-worker stateful deployment;
- CI unit and full-stack multiprocess execution tests.

Not yet established by this reference deployment:

- distributed database/object state across nodes;
- GPU-aware scheduler or verified accelerator execution;
- tensor/expert/pipeline parallel training;
- learned image/audio/video generation adapters;
- autoscaling, ingress/TLS, network policy and secret-manager integration;
- production tenant IAM, quotas and billing;
- measured SLOs or astronomical physical model capacity.

Those are infrastructure and model-data-plane increments, not properties that can be inferred from the symbolic virtual parameter extent.
