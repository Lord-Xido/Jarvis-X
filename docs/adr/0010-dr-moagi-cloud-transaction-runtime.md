# ADR-010: Adopt the Dr Moagi Cloud transactional runtime boundary

**Status:** Proposed  
**Date:** 2026-08-18  
**Extends:** ADR-002, ADR-003

## Context

Jarvis-X already separates deterministic authority from adaptive and spatial research layers. The Dr Moagi field runtime is candidate-first and fail-closed, but a cloud deployment also needs an execution boundary above individual numerical kernels: identity, job lifecycle, resource ceilings, verification, durable evidence, replay, and explicit promotion to authoritative state.

A web interface, agent plan, or executor result is not authoritative merely because it was produced successfully. Cloud execution must preserve the same rule already used by the field runtime: compute a candidate, verify it, then commit or reject it atomically.

## Decision

Jarvis-X adopts a Layer 6/7 reference control plane with the canonical transaction path:

```text
RECEIVED
  -> VALIDATED
  -> PLANNED
  -> DISPATCHED
  -> RUNNING
  -> VERIFIED
  -> COMMITTED
```

Any policy, resource, execution, or verification failure ends in `REJECTED` or `FAILED` without promotion.

The transaction envelope is identified by a durable `job_id` and records:

```text
T_i = (
  protocol,
  job_id,
  request_id,
  principal,
  operation,
  input,
  input_digest,
  plan,
  result,
  result_digest,
  verification,
  resource_usage,
  events,
  envelope_digest
)
```

The cloud promotion law is:

```text
Omega_(t+1) = Commit(Omega_t, DeltaOmega_t)
              iff verifier == PASS and policy == PASS
otherwise Omega_(t+1) = Omega_t
```

This is the operational cloud interpretation of `Pi_Lambda`: an explicit policy and verification gate, not an implied trust boundary.

## Durable evidence

Every transition appends a canonical JSON event containing:

```text
sequence
state
timestamp_ms
details
previous_digest
event_digest
```

Events form a SHA-256 hash chain rooted at a fixed genesis digest. The complete job envelope is separately SHA-256 sealed. Replay verification recomputes the envelope digest, the event chain, terminal-state coherence, and result digest.

The reference filesystem store uses atomic replacement and canonical JSON. This is a reference durability mechanism, not a substitute for a replicated production database or object store.

## Identity and authorization

The reference FastAPI surface supports bootstrap API-key authentication and an authenticated principal field. Production deployments should terminate stronger identity at a trusted gateway or identity provider and map it to the coordinator principal. Operation authorization is fail-closed through an explicit allowlist.

## Resource contract

Every job is bounded by:

- maximum canonical input bytes;
- maximum canonical output bytes;
- maximum executor runtime;
- operation-specific limits inside the selected executor.

Exceeding a post-execution runtime or output ceiling rejects the candidate result rather than committing it.

## Executor boundary

Executors are adapters, not authorities. They receive an immutable JSON-compatible input plus resource limits and return a candidate result. They cannot mark a job committed.

The initial reference operations are:

- `echo.v1` for deterministic transaction-spine conformance;
- `dr-moagi-field-step.v1` for one bounded same-space Dr Moagi field step using the existing `DrMoagiFieldRuntime`.

## HTTP contract

The reference service exposes:

```text
GET  /health/live
GET  /health/ready
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/events
POST /api/v1/jobs/{job_id}/verify
GET  /metrics
```

The first implementation is intentionally synchronous. The state machine includes `DISPATCHED` so a durable external queue can replace the in-process dispatch path later without changing the transaction semantics.

## Architectural placement

```text
Layer 0-3  canonical deterministic VM, policy, transactions, provenance
Layer 4    sparse geometric state and spatial operators
Layer 5    codec/residual/adaptive research runtimes
Layer 6    Dr Moagi Cloud transaction coordinator and operation adapters
Layer 7    HTTP/UI/agent interfaces and deployment integrations
```

The cloud layer does not modify the canonical VM instruction format and does not let visualization or agent output bypass lower authority boundaries.

## Required invariants

1. Every accepted request receives exactly one canonical UUID `job_id`.
2. No executor may directly set `COMMITTED`.
3. Verification and policy are fail-closed.
4. Job events are ordered and hash chained.
5. The envelope is independently integrity sealed.
6. Result digests bind committed output to the job record.
7. Input, output, and runtime ceilings are explicit.
8. Disallowed operations fail before execution.
9. Executor exceptions are durably journalled as `FAILED`.
10. A stored job can be integrity-verified from `job_id` alone.
11. Field execution remains bounded by ADR-003 resource, support, anchor, and projection rules.
12. Queueing, distributed workers, learning, or self-modification may extend the runtime only without weakening these invariants.

## Consequences

### Positive

- Dr Moagi Cloud gains an executable authority boundary rather than a UI-only architecture;
- cloud jobs become reconstructable and tamper-evident;
- the existing field runtime can be invoked through a verified transaction envelope;
- a future queue, worker pool, database, agent planner, or model gateway can plug into stable contracts;
- adaptive proposals remain candidate-first and reversible.

### Negative

- the reference filesystem store is single-node and not horizontally replicated;
- API-key authentication is bootstrap-grade rather than full enterprise identity;
- synchronous dispatch does not yet provide durable distributed scheduling;
- elapsed wall-clock limits detect overruns after executor return and are not a hard process-kill sandbox.

## Validation

The decision may move to **Accepted** when CI demonstrates:

- successful `RECEIVED -> ... -> COMMITTED` execution;
- verifier rejection with no committed result;
- policy rejection before execution;
- resource-ceiling rejection;
- executor exception journaling;
- tamper detection for stored envelopes;
- authenticated API behavior;
- create/fetch/events/verify HTTP round trip;
- successful bounded `dr-moagi-field-step.v1` execution against the canonical field runtime.

The normative operations guide is `docs/DR_MOAGI_CLOUD_RUNTIME_V1.md`.
