# Hyperion Operational Service

## Status

Hyperion is operationalised as a deterministic audit service and command-line tool. The runtime accepts already-extracted observations, executes the frozen Hyperion arithmetic, persists a replayable evidence bundle, and verifies stored results by deterministic re-execution.

It does **not** acquire microphone, camera, database, or eBPF evidence inside the service process. Acquisition remains a separate, source-specific boundary with its own authorization, privacy, clock provenance, and chain-of-custody controls.

## Runtime architecture

```text
signed/source-controlled extractors
             |
             v
     bounded JSON observations
             |
             v
  Hyperion service (frozen config + model)
             |
     deterministic audit kernel
             |
             v
 atomic replay bundle on protected volume
             |
      +------+------+
      |             |
      v             v
 REST retrieval   deterministic replay verification
```

One process owns one immutable `HyperionConfig` and one immutable `ScoreModel`. Their hashes are returned in every response and committed into every report.

## Installation

```bash
python -m pip install -e ".[dev]"
```

The package exposes:

```text
jarvisx-hyperion audit
jarvisx-hyperion verify
jarvisx-hyperion serve
```

## CLI audit

Input may be a JSON array or an object containing `observations`:

```json
{
  "observations": [
    {
      "source": "csv",
      "timestamp_ms": 1700000000000,
      "value": 149000.0,
      "quantity": "amount",
      "unit": "ZAR",
      "correlation_id": "transaction-42",
      "confidence": 1.0,
      "label": "verified beneficiary"
    }
  ]
}
```

Create a replay bundle:

```bash
jarvisx-hyperion audit observations.json \
  --config configs/hyperion.production.json \
  --model configs/hyperion.model.v1.json \
  --output evidence/report.json
```

Replay and verify it:

```bash
jarvisx-hyperion verify evidence/report.json \
  --config configs/hyperion.production.json \
  --model configs/hyperion.model.v1.json
```

Verification fails when the bundle, observations, report, model, or configuration differs from the committed values.

## Service startup

Local development:

```bash
jarvisx-hyperion serve \
  --host 127.0.0.1 \
  --port 8080 \
  --data-dir state/hyperion
```

Authenticated operation:

```bash
jarvisx-hyperion serve \
  --host 0.0.0.0 \
  --port 8080 \
  --data-dir /var/lib/hyperion \
  --config configs/hyperion.production.json \
  --model configs/hyperion.model.v1.json \
  --require-api-key \
  --api-key "$HYPERION_API_KEY"
```

The ASGI factory can also be run directly:

```bash
uvicorn jarvisx.hyperion_service:app_factory \
  --factory \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1
```

Configuration through environment variables:

| Variable | Purpose |
|---|---|
| `HYPERION_DATA_DIR` | Durable evidence directory |
| `HYPERION_MAX_OBSERVATIONS` | Maximum observations in one request |
| `HYPERION_CONFIG_FILE` | Frozen `HyperionConfig` JSON |
| `HYPERION_MODEL_FILE` | Frozen `ScoreModel` JSON |
| `HYPERION_REQUIRE_API_KEY` | Require `X-Hyperion-Key` when true |
| `HYPERION_API_KEY` | Shared API key; inject from a secret manager |

Do not commit real API keys to the repository.

## HTTP contract

Unauthenticated liveness and readiness endpoints:

```text
GET /healthz
GET /readyz
```

Operational endpoints:

```text
GET  /v1/hyperion/manifest
POST /v1/hyperion/audits
GET  /v1/hyperion/audits/{report_digest}
POST /v1/hyperion/audits/{report_digest}/verify
GET  /metrics
```

When authentication is enabled, supply:

```text
X-Hyperion-Key: <secret>
```

Create an audit:

```bash
curl -sS \
  -H "Content-Type: application/json" \
  -H "X-Hyperion-Key: $HYPERION_API_KEY" \
  --data @observations.json \
  http://127.0.0.1:8080/v1/hyperion/audits
```

The response contains the report and bundle digests, GHS, event count, critical count, model hash, and configuration hash.

## Persistence semantics

Evidence is written as canonical JSON under:

```text
<DATA_DIR>/<report_digest>.json
```

The write protocol is:

```text
serialize canonically
→ write temporary file
→ flush
→ fsync
→ atomic replace
```

A report digest is immutable. Repeating an identical audit is idempotent. Different bytes for an existing report digest fail closed as a collision.

The stored bundle contains the original normalized observations because deterministic replay requires them. The evidence directory therefore contains potentially sensitive information and must use encrypted storage, restricted permissions, retention policy, backup policy, and access logging appropriate to the deployment.

## Authentication and network boundary

The built-in API key is a minimal service boundary, not a full identity platform. Production deployments should additionally use:

- TLS termination;
- workload or service identity;
- network policy or private ingress;
- request-size limits at the proxy;
- rate limiting;
- secret-manager injection;
- audit logging without raw sensitive values;
- encrypted durable storage.

The service intentionally runs with one Uvicorn worker. Local file locks and in-process metrics do not provide distributed coordination. Horizontal scaling requires an external immutable object store, distributed idempotency control, and centralized metrics.

## Container operation

Build:

```bash
docker build -f deploy/hyperion.Dockerfile -t jarvisx-hyperion:local .
```

Run:

```bash
docker run --rm \
  -p 8080:8080 \
  -e HYPERION_API_KEY="$HYPERION_API_KEY" \
  -v hyperion-data:/var/lib/hyperion \
  jarvisx-hyperion:local
```

Or use:

```bash
HYPERION_API_KEY="$HYPERION_API_KEY" \
  docker compose -f deploy/hyperion-compose.yml up --build
```

## Monitoring

`/metrics` exposes Prometheus text metrics:

```text
hyperion_audits_total
hyperion_failures_total
hyperion_observations_total
hyperion_critical_events_total
hyperion_reports_stored
hyperion_last_success_unixtime_seconds
```

Recommended alerts:

- readiness failure;
- any persistence failure;
- sustained audit failure rate;
- no successful audit within the expected ingestion interval;
- unexpected increase in critical-event rate;
- evidence volume approaching capacity.

## Operational invariants

1. A service process never mutates its model or audit configuration.
2. Requests above the configured observation limit fail before arithmetic execution.
3. Every successful response corresponds to a report that passed `AuditReport.verify()`.
4. Stored evidence is immutable by report digest.
5. Replay uses the exact committed observations, model hash, and configuration hash.
6. A different model or configuration cannot verify an old bundle silently.
7. The service proves deterministic computation over supplied evidence, not source truth.

## Deployment gate

Repository compatibility is tested by:

- focused core and operational tests;
- CLI audit/verify round trip;
- authenticated network smoke test against Uvicorn;
- container image build;
- the existing Jarvis-X quality, type, package, dependency, empirical, and CodeQL gates.

A live release still requires a chosen infrastructure target, secrets, TLS, protected persistent storage, retention policy, and source-acquisition integrations.
