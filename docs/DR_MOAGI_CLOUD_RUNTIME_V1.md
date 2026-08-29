# Dr Moagi Cloud Runtime v1

This document describes the first production-oriented vertical slice of the Dr Moagi Cloud control plane. It is deliberately small: a request becomes a durable, verifiable job before larger agent, queue, model, or market subsystems are added.

## Runtime law

```text
X -> Validate -> Plan -> Dispatch -> Execute -> Verify -> Pi_Lambda -> Commit
```

A result is only authoritative when both verification and policy gates pass.

## Run locally

```bash
python -m pip install -e .
uvicorn jarvisx.dr_moagi_cloud_service:app --host 0.0.0.0 --port 8080
```

For production-like bootstrap authentication:

```bash
export DR_MOAGI_CLOUD_REQUIRE_API_KEY=true
export DR_MOAGI_CLOUD_API_KEY='replace-with-secret'
export DR_MOAGI_CLOUD_DATA_DIR='/var/lib/dr-moagi-cloud'
uvicorn jarvisx.dr_moagi_cloud_service:app --host 0.0.0.0 --port 8080
```

## Create a deterministic conformance job

```bash
curl -sS -X POST http://127.0.0.1:8080/api/v1/jobs \
  -H 'content-type: application/json' \
  -H 'X-Dr-Moagi-Key: replace-with-secret' \
  -H 'X-Dr-Moagi-Principal: user:operator' \
  -d '{
    "operation":"echo.v1",
    "request_id":"smoke-001",
    "input":{"message":"Dr Moagi Cloud"}
  }'
```

The response includes the canonical `job_id`, full transition journal, verification outcome, result digest, and envelope digest.

## Execute one bounded Dr Moagi field step

```bash
curl -sS -X POST http://127.0.0.1:8080/api/v1/jobs \
  -H 'content-type: application/json' \
  -H 'X-Dr-Moagi-Key: replace-with-secret' \
  -H 'X-Dr-Moagi-Principal: user:operator' \
  -d '{
    "operation":"dr-moagi-field-step.v1",
    "input":{
      "config":{
        "side":5,
        "alpha":0.0,
        "lambda_residual":0.0,
        "eta":0.0,
        "dt":0.1,
        "expand_halo":false
      },
      "field":[{"x":2,"y":2,"z":2,"value":0.5}]
    }
  }'
```

This adapter invokes the canonical bounded field runtime with an identity codec. The field runtime still owns coordinate bounds, sparse support, stability guards, projection, anchor semantics, and candidate commit.

## Replay and evidence

Given a `job_id`:

```text
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/events
POST /api/v1/jobs/{job_id}/verify
```

`verify` recomputes:

1. protocol identity;
2. envelope SHA-256;
3. every event SHA-256;
4. previous-event links;
5. event sequence numbers;
6. terminal state/event agreement;
7. result SHA-256 when a result exists.

A modified stored record therefore fails verification unless every dependent digest is recomputed. This is tamper-evidence, not cryptographic non-repudiation; signatures or external transparency logs can be layered on later.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `DR_MOAGI_CLOUD_DATA_DIR` | `state/dr-moagi-cloud` | Durable reference job store |
| `DR_MOAGI_CLOUD_REQUIRE_API_KEY` | `false` | Enforce bootstrap API-key auth |
| `DR_MOAGI_CLOUD_API_KEY` | unset | Shared bootstrap API key |
| `DR_MOAGI_CLOUD_MAX_INPUT_BYTES` | `1000000` | Canonical request-input ceiling |
| `DR_MOAGI_CLOUD_MAX_OUTPUT_BYTES` | `2000000` | Canonical result ceiling |
| `DR_MOAGI_CLOUD_MAX_RUNTIME_MS` | `5000` | Executor elapsed-time ceiling |

## Production hardening path

The v1 state machine is intentionally designed so these components can be replaced without changing the authority contract:

```text
bootstrap API key   -> OIDC/JWT/mTLS identity gateway
filesystem store    -> transactional database + immutable object evidence
in-process dispatch -> durable queue
local executor      -> isolated worker/sandbox
basic metrics       -> OpenTelemetry + Prometheus + traces
SHA-256 journal     -> signed evidence / transparency anchoring
single verifier     -> policy engine + schema tests + deterministic replay + shadow validation
```

The key invariant is unchanged throughout:

```text
candidate execution != authoritative state
candidate + verification + policy pass = eligible commit
```
