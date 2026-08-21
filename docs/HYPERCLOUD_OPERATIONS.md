# Jarvis-X HyperCloud Operations

**Status:** operational single-node/stateful reference deployment  
**Scope:** durable sparse state, content-addressed multimedia, deterministic 3D routing, persistent jobs, codec execution, pluggable chat inference

## 1. Start the complete stack

```bash
cp .env.hypercloud.example .env
# edit .env if authentication or a model backend is required
docker compose --env-file .env -f docker-compose.hypercloud.yml up -d --build
```

The stack contains:

- `api`: FastAPI control plane on port 8080 by default;
- `worker`: persistent job executor;
- `hypercloud-state`: shared durable volume containing SQLite/WAL state and content-addressed objects.

Verify:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/readyz
curl http://127.0.0.1:8080/metrics
```

If `JARVISX_API_KEY` is set, add either:

```text
Authorization: Bearer <key>
```

or:

```text
X-API-Key: <key>
```

for `/v1/*` calls.

## 2. Execute a chat job

With the default configuration the worker uses the deterministic local reference backend. This validates the entire persistent execution path but is **not** represented as a neural LLM.

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

The response contains a durable `id`. Poll:

```bash
curl http://127.0.0.1:8080/v1/jobs/<id>
```

until `status` becomes `succeeded` or `failed`.

## 3. Attach a real model server

HyperCloud can execute against an OpenAI-compatible chat-completions endpoint. This may be a self-hosted inference server or a managed provider.

Configure:

```bash
export JARVISX_MODEL_BASE_URL=http://model-server:8000
export JARVISX_MODEL_NAME=my-model
export JARVISX_MODEL_API_KEY='optional-secret'
docker compose -f docker-compose.hypercloud.yml up -d
```

The worker sends:

```text
POST ${JARVISX_MODEL_BASE_URL}/v1/chat/completions
```

using `JARVISX_MODEL_NAME`. API keys are read from the environment and are never persisted in the HyperCloud state database.

A configured external model endpoint proves only that a model backend is attached. It does not by itself prove GPU acceleration, model size, throughput, or quality; those require backend telemetry and benchmarks.

## 4. Persist sparse parameters

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

Read it:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{
        "namespace":"demo/model-a",
        "modality":"text",
        "digits":[17,42,999999]
      }' \
  http://127.0.0.1:8080/v1/parameters/read
```

Physical allocation remains proportional to touched addresses; the symbolic astronomical upper bound is never expanded densely.

## 5. Ingest and auto-encode/decode multimedia

Media ingestion accepts base64 bytes and an explicit modality:

```json
{
  "namespace": "demo",
  "kind": "image",
  "content_type": "image/png",
  "payload_base64": "..."
}
```

POST this payload to:

```text
POST /v1/media
```

The returned SHA-256 digest is scoped by namespace. Submit an executable codec round-trip:

```json
{
  "namespace": "demo",
  "media_sha256": "<digest>"
}
```

POST to:

```text
POST /v1/jobs/codec
```

The worker performs:

```text
media bytes
  -> zlib-deflate latent packet
  -> verified decode
  -> byte-for-byte reconstruction check
```

This is a real deterministic auto-encoding/decoding transport primitive, but it is explicitly a lossless codec rather than a learned neural autoencoder. Learned image/audio/video codecs can replace or supplement this adapter later without changing the job/state contract.

## 6. Data layout

Default container paths:

```text
/var/lib/jarvisx/state.db
/var/lib/jarvisx/state.db-wal
/var/lib/jarvisx/state.db-shm
/var/lib/jarvisx/objects/<sha-prefix>/<sha-prefix>/<sha256>
```

SQLite stores:

- sparse parameter values;
- namespace-scoped media metadata;
- durable job inputs/results/errors.

Object bytes are content-addressed and deduplicated globally by SHA-256 while metadata ownership remains namespace-scoped.

## 7. Failure semantics

Worker execution is transactional at the job-state boundary:

```text
queued -> running -> succeeded
                  -> failed
```

An exception is stored in the job's `error` field. The API process and worker process are isolated so model/backend failures do not crash the control plane.

The current worker intentionally does not auto-retry failed jobs. Retry policy should be added only with operation idempotency rules, attempt counters, backoff and dead-letter semantics.

## 8. Observability

Prometheus text metrics are available at:

```text
GET /metrics
```

Current gauges include:

- materialized sparse parameters;
- namespace-scoped media metadata objects;
- queued/running/succeeded/failed job counts.

Readiness checks SQLite availability at `/readyz`; liveness is exposed at `/healthz`.

## 9. Kubernetes

Apply:

```bash
kubectl apply -f deploy/k8s/hypercloud.yaml
```

The manifest intentionally runs **one pod** containing both API and worker with a `ReadWriteOnce` persistent volume. This is the correct topology for the SQLite/WAL reference backend.

Do not increase replicas while using this manifest's local SQLite state plane. Horizontal production scale requires replacing the persistence adapter with distributed state/object services and then introducing topology-aware workers.

## 10. Operational capability boundary

Implemented and runnable:

- symbolic astronomical virtual parameter namespace;
- finite deterministic 3D shard mapping;
- durable sparse parameters;
- namespace-scoped multimedia metadata;
- content-addressed object persistence;
- lossless multimedia encode/decode execution;
- durable asynchronous jobs;
- worker isolation and persisted failures;
- offline local response backend;
- OpenAI-compatible neural model adapter;
- optional API-key enforcement;
- readiness/liveness/metrics;
- Docker Compose deployment;
- Kubernetes single-pod stateful deployment;
- CI unit and full-stack execution tests.

Not yet established by this reference deployment:

- distributed database/object state;
- multi-node worker leasing/heartbeats;
- GPU-aware scheduler or verified accelerator execution;
- tensor/expert/pipeline parallel training;
- learned image/audio/video generation adapters;
- autoscaling, ingress/TLS, network policy and secret-manager integration;
- production tenant IAM, quotas and billing;
- measured SLOs or astronomical physical model capacity.

Those are infrastructure and model-data-plane increments, not properties that can be truthfully inferred from the symbolic virtual parameter extent.
