# ADR-006: Adaptive Corporation Orchestration Runtime

- Status: Accepted
- Date: 2026-08-15
- Decision scope: Jarvis-X enterprise / industrial orchestration research runtime

## Context

Jarvis-X already separates canonical deterministic execution from research, generative, geometric and adaptive layers. The Adaptive Corporation work extends that architecture into corporate operations: objectives are translated into bounded work graphs spanning software, humans and industrial systems.

A useful architecture must not equate model reasoning with authority. Enterprise execution requires explicit state, identity, policy, idempotency, verification, recovery and audit semantics. The same runtime must also be able to turn inward when security confidence degrades, reducing mutation authority rather than allowing an intelligent planner to continue acting through a suspected compromise.

## Decision

Adopt a reference Adaptive Corporation orchestration boundary with the following constitutional separation:

```text
intelligence plans
governance permits
reliable software executes
reality verifies
memory adapts
```

The normative control loop is:

```text
objective
 -> observe / assemble state
 -> plan / simulate
 -> Pi_Lambda governance + security projection
 -> compile DAG
 -> execute typed capabilities
 -> verify authoritative outcome
 -> compensate / recover on failure
 -> reconcile
 -> persist tamper-evident memory
 -> adapt / replan
```

The reference implementation is `src/jarvisx/adaptive_orchestrator.py`; the operational specification is `docs/ADAPTIVE_CORPORATION_ORCHESTRATION.md`.

## Invariants

1. Planning authority and execution authority are separate.
2. Mutating tasks require explicit scopes and an idempotency key.
3. Security state may contract authority; it may not silently expand it.
4. Typed capabilities are preferred over arbitrary shell/database access.
5. API success is insufficient: declared outcomes are verified.
6. Cross-system failures use explicit compensation or escalation semantics.
7. Event history is causally attributable and tamper-evident.
8. Industrial safety-critical execution remains under deterministic controllers and independently enforced safety systems.
9. The runtime is a reference orchestrator, not a universal transaction manager or security sandbox.
10. Autonomous authority is promoted only after empirical reliability and economic-value evidence.

## Consequences

The architecture can now host enterprise workflows without making an LLM or planner the source of truth. ERP, MES, ledgers, identity systems and physical controllers retain domain authority while Jarvis-X coordinates bounded actions between them.

This gives the Adaptive Corporation recurrence concrete operational terms:

```text
Xi      corporate state
P       planning / scenario generation
Pi_Lambda policy + security projection
U       bounded workflow execution
E       verification / reconciliation error
Omega   operational memory
```

The design adds deliberate engineering work: durable state, distributed idempotency, human approval, connector hardening and production policy systems must be implemented before enterprise deployment.

## Security

The runtime fails closed for insufficient scopes, missing mutation idempotency, low security confidence, or an explicit intrusion signal. This is an inward homeostatic control: observed integrity affects available authority.

It does not replace network segmentation, IAM, secrets management, endpoint security, SIEM/SOAR, OT safety controls, or incident-response procedures.

## Validation

The accompanying test suite covers graph integrity, scope/security gating, verification, compensation, idempotency and hash-chain tamper evidence.
