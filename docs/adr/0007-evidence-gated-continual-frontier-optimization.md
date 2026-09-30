# ADR-007: Evidence-Gated Continual Frontier Optimization

- Status: Accepted
- Date: 2026-08-15

## Context

ADR-006 introduced the Adaptive Corporation orchestration runtime with typed work packets, policy/security projection, deterministic capability execution, verification, compensation, reconciliation, and tamper-evident memory.

The next architectural question is how that runtime may improve itself over time without allowing recursive optimization to become an uncontrolled self-modification or privilege-escalation path.

A useful continual-improvement system must support experimentation and adaptation while preserving four properties:

1. the active production runtime remains identifiable as a fixed champion during evaluation;
2. candidate changes are judged by externalized evidence rather than self-description;
3. safety, security, governance, and other constitutional invariants remain outside candidate control;
4. production promotion is staged and reversible.

Literal promises of infinite or permanently state-of-the-art performance are rejected as untestable. The architecture instead targets open-ended, evidence-gated improvement against versioned external frontiers.

## Decision

Jarvis-X adopts a champion/challenger continual optimization layer with the following mandatory sequence:

```text
active champion
-> candidate generation
-> repeated benchmark evaluation
-> hard invariant / security / risk gates
-> conservative utility comparison
-> SHADOW
-> CANARY
-> PRODUCTION or ROLLBACK
-> new champion
-> next finite generation
```

The reference implementation is `src/jarvisx/continual_optimizer.py`.

## Constitutional invariants

A candidate may not directly:

- promote itself;
- redefine the benchmark used to prove its own superiority;
- remove authorization, audit, provenance, or security gates;
- weaken protected policy floors;
- inject permanent credentials into model context;
- bypass deterministic safety-control boundaries;
- erase failed evaluation evidence.

These restrictions are enforced outside candidate proposal logic.

## Evidence rule

For each metric, the reference evaluator records repeated samples, means, standard deviations, and a digest. The promotion gate evaluates directional improvement, hard metric floors/ceilings, tolerated regressions, risk score, security state, and an aggregate confidence-adjusted utility delta.

A candidate enters shadow mode only when its conservative utility gain clears the configured threshold.

This heuristic is not declared a universal statistical significance test. Production evaluation programs remain responsible for selecting appropriate experimental designs for their domains.

## Frontier rule

State-of-the-art is represented only through a versioned external `FrontierSnapshot`.

A frontier delta is evidence relative to that declared benchmark/version only. Exceeding one frontier metric does not authorize a universal SOTA claim.

## Release rule

Promotion authority is separated from candidate generation.

```text
SHADOW -> CANARY -> PRODUCTION
```

Canary promotion requires a configured minimum observation count, success-rate floor, non-regressing mean value delta, and intact integrity observations. Otherwise the candidate is rejected with rollback semantics.

Only a `PRODUCTION` promotion decision may update the active champion identity.

## Security coupling

Optimization mutation authority is conditioned on the Adaptive Corporation `SecurityState`.

Suspected intrusion or security confidence below the configured mutation threshold denies candidate promotion even when benchmark performance is otherwise favorable.

## Consequences

### Positive

- measurable improvements can accumulate over successive generations;
- benchmark evidence and production authority remain separated;
- regressions can be rejected before wide release;
- safety/security metrics may be configured as non-tradeable hard constraints;
- the system can track a moving external frontier without inventing its own SOTA definition;
- optimization remains compatible with shadow mode, canaries, rollback, and human governance.

### Negative

- repeated evaluation increases compute and operational cost;
- benchmark quality becomes a critical dependency;
- noisy or gamed metrics can still mislead optimization;
- multi-objective performance requires careful metric design;
- external frontier data must be independently refreshed and validated;
- production deployment still requires durable state, sandboxing, artifact provenance, and independent rollback infrastructure beyond the dependency-light reference implementation.

## Rejected alternatives

### Unrestricted recursive self-modification

Rejected because the candidate would control both proposal and authority, creating unacceptable governance, safety, and security failure modes.

### Single-score promotion

Rejected because one measurement can be noisy, brittle, or gamed.

### Weighted utility with no hard invariants

Rejected because large gains in one metric could numerically mask unacceptable safety or integrity regressions.

### Permanent SOTA declaration

Rejected because SOTA is benchmark-, domain-, version-, and time-dependent.

## Canonical interpretation

Jarvis-X may continue improving for an open-ended sequence of finite, authorized generations. It does not claim physically infinite performance or automatic universal superiority.

The governing law is:

```text
NO EVIDENCE -> NO PROMOTION
NO INTEGRITY -> NO MUTATION
NO CANARY SUCCESS -> NO PRODUCTION
NO VERIFIED ADVANTAGE -> NO SOTA CLAIM
```
