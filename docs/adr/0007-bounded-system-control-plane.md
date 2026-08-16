# ADR-007: Bounded System Control Plane

- **Status:** Proposed
- **Date:** 2026-08-16
- **Decision scope:** canonical Python runtime

## Context

Jarvis-X already has a deterministic transactional bytecode VM. Each instruction is checkpointed, executed, policy-checked, journalled and traced, and a failed instruction restores the last authoritative VM state.

The remaining system boundary was task-level rather than instruction-level. Research planners, adaptive layers and future tool adapters need one narrow path into authoritative execution. Without that boundary, the architecture can describe `Plan != Execute` and `Execute != Commit` without enforcing those rules in code.

The production core therefore needs explicit contracts for:

1. side-effect-free candidate planning;
2. capability projection before execution;
3. bounded program, candidate and cycle budgets;
4. isolated tentative execution;
5. verification before system commit;
6. request idempotency and collision rejection;
7. system-level provenance linked to the VM ledger;
8. fail-closed handling when the audit path is unavailable.

## Decision

Introduce `jarvisx.system_runtime` as the task-level control plane around `CodexVM`.

The canonical transition is:

```text
candidate plans
      |
      v
bounded deterministic selector
      |
      v
capability projection (Lambda)
      |
      v
immutable ExecutionRequest
      |
      v
isolated CodexVM execution
      |
      v
VM ledger verification + state digest
      |
      v
system audit append
      |
      v
authoritative committed state + receipt
```

A plan is a proposal only. It cannot mutate canonical state. A VM result is tentative until both the VM ledger and the system audit path verify successfully.

## Core invariants

### 1. Plan is not execution

`PlanCandidate` contains bytecode, utility signals and required capabilities. Candidate scoring is pure and deterministic:

```text
J(P_i) = w_q Q_i - w_c C_i - w_l L_i - w_r R_i
```

Candidate ties are resolved by stable plan identifier ordering.

### 2. Capability monotonicity

For every request:

```text
required_capabilities ⊆ granted_capabilities ⊆ runtime_capabilities
```

The reference runtime currently defines:

- `vm.execute`
- `vm.reflex`

Reflex adaptation remains opt-in and requires `vm.reflex` in addition to `vm.execute`.

### 3. Explicit resource budgets

`ResourceBudget` bounds:

- VM cycles;
- bytecode words;
- candidate plans.

The logical extent of a research model never implies physical allocation or unbounded execution authority.

### 4. Execute is not commit

Every request executes in an isolated `CodexVM`. Its result becomes authoritative only after:

1. request validation;
2. capability projection;
3. bounded VM execution;
4. VM ledger verification;
5. final-state hashing;
6. system audit append;
7. system audit verification.

If any commit-stage operation fails, the system audit append is rolled back and the tentative VM state is discarded.

### 5. Idempotency

A `request_id` is bound to a canonical SHA-256 request fingerprint. Repeating the same request returns the prior receipt without re-executing it. Reusing the identifier for different contents raises `RequestCollisionError` and cannot overwrite the original committed state.

### 6. Provenance

Successful receipts carry:

- request fingerprint;
- selected plan id, when applicable;
- VM cycle count;
- final state digest;
- VM ledger head hash;
- system audit head hash.

The system audit uses reserved control-plane opcodes:

- `0x1000` commit;
- `0x1001` policy rejection;
- `0x1002` execution or verification failure.

These are audit event identifiers, not additions to the canonical VM instruction set.

### 7. Reality boundary

Model output, planner output and tentative VM output are not authoritative reality. The only authoritative task state is the state behind a committed, audit-linked `ExecutionReceipt`.

Formally:

```text
prediction -> plan -> projection -> execution -> verification -> audit -> commit
```

not:

```text
prediction -> state
```

## Consequences

### Positive

- Adaptive and research layers gain a single governed execution path.
- Capability escalation fails closed.
- Task retries become idempotent.
- Planner outputs remain separable from execution authority.
- Failed or unverifiable VM runs cannot leak partial state into the system state map.
- VM and task-level provenance become cryptographically linked.
- Existing `CodexVM` semantics remain unchanged.

### Trade-offs

- The first reference control plane is intentionally in-process and single-node.
- Committed task-state storage is in-memory; durable system-state storage remains a later persistence adapter.
- Capability names are deliberately small until real side-effecting adapters exist.
- Planner utility signals are supplied by callers; the runtime does not claim autonomous objective discovery.

## Non-goals

This ADR does not:

- grant autonomous network, shell, financial, medical or infrastructure authority;
- turn symbolic 3D lattice sizes into physical allocations;
- permit a learning model to rewrite the canonical VM directly;
- make research equations evidence of production performance;
- replace domain-specific authorization, review or regulatory controls.

## Validation

`tests/test_system_runtime.py` covers:

- successful verified commit;
- capability rejection;
- explicit reflex capability;
- cycle-budget failure;
- request idempotency;
- request-id collision rejection;
- deterministic bounded planning;
- selected-plan provenance;
- audit-path failure preventing commit.

## Follow-on boundary

Future network, filesystem, model, market or device adapters must enter through named capabilities and transactional adapters outside the deterministic VM. No adapter may bypass the system runtime to mutate authoritative state.
