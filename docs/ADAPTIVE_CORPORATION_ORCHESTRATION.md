# Adaptive Corporation Orchestration Runtime

## Status

Reference architecture and executable control-loop contract for Jarvis-X corporate and industrial orchestration.

The system turns the Adaptive Corporation model into a bounded software runtime:

```text
OBJECTIVE
  -> OBSERVE
  -> ASSEMBLE STATE
  -> PLAN
  -> SIMULATE
  -> Pi_Lambda POLICY / SECURITY PROJECTION
  -> COMPILE WORK DAG
  -> EXECUTE CAPABILITIES
  -> VERIFY
  -> COMPENSATE / RECOVER
  -> RECONCILE
  -> PERSIST MEMORY
  -> ADAPT / REPLAN
  -> repeat
```

The reference implementation is `src/jarvisx/adaptive_orchestrator.py`.

It is not a replacement for ERP, MES, SCADA, PLC, SIS, payment rails, identity providers, or certified safety controllers. It coordinates typed capabilities around those systems while preserving explicit authority boundaries.

## 1. Constitutional separation

The runtime is split into three parallel planes:

```text
Operational plane   ||  Intelligence plane  ||  Governance/security plane
ERP/MES/APIs            planner/scenario AI      identity/policy/risk
transactions            state interpretation     authorization/audit
physical execution      candidate plans          isolation/recovery
```

The canonical separation is:

```text
intelligence plans
governance permits
reliable software executes
reality verifies
memory adapts
```

A planner may propose a work graph. It does not receive implicit authority to execute arbitrary shell commands, database writes, payment instructions, or machine-control actions.

## 2. Corporate state

A complete deployment can represent corporate state as:

```text
Xi_t = [
  goals,
  operations,
  resources,
  knowledge,
  memory,
  authority,
  constraints,
  human commitments,
  security state
]
```

The reference runtime deliberately accepts state as a caller-owned mapping. Domain systems remain authoritative for their own records.

## 3. Work packet

The unit of executable corporate work is a `WorkPacket`:

```text
task_id
action
payload
depends_on[]
expected{}
required_scopes[]
mutating
idempotency_key
retries
compensate_action
compensate_payload
```

A work packet therefore declares what capability is requested, what must happen first, what result is expected, what authority is required, whether economic state can change, how duplicate execution is prevented, and how completed work can be compensated.

## 4. Workflow DAG

Approved work compiles into a directed acyclic graph:

```text
objective
   |
   +--> reserve inventory ----+
   |                          |
   +--> check capacity -------+--> create production order
                              |
                              +--> quality verification
```

Independent nodes form deterministic wavefronts. A production executor may map a wavefront to concurrent workers, but concurrency never removes dependency, authorization, verification, or idempotency requirements.

The reference runtime rejects duplicate task IDs, unknown dependencies, negative retry counts, and cycles.

## 5. Governance projection Pi_Lambda

Every task passes a fail-closed policy projection:

```text
A_exec = Pi_Lambda(A_candidate)
```

The reference policy engine checks:

1. required scopes are present;
2. mutating actions carry an idempotency key;
3. security confidence is sufficient for mutation;
4. an active intrusion signal prevents mutation.

Production `Pi_Lambda` should additionally encode identity, financial authority, segregation of duties, legal limits, safety envelopes, data classification, cybersecurity policy, human approval requirements, and plant operating boundaries.

## 6. Inward security loop

The orchestration system protects itself by conditioning execution authority on its own integrity state:

```text
Observe self
 -> verify identity / provenance / state
 -> detect deviation
 -> contract authority
 -> isolate
 -> recover
 -> learn
 -> repeat
```

The core rule is:

```text
read broadly
write narrowly
control most narrowly
```

For a mutation, the reference runtime requires security confidence above threshold and no active intrusion signal. A production security plane should additionally use short-lived credentials, network segmentation, independent audit, endpoint/device identity, secrets brokering, provenance, and out-of-band incident controls.

## 7. Typed capabilities

The intelligence layer receives named capabilities rather than unrestricted execution primitives.

Prefer:

```text
reserve_inventory(...)
create_purchase_order(...)
request_maintenance(...)
update_forecast(...)
```

instead of:

```text
execute_shell(...)
run_sql(...)
```

`ToolRegistry` is the reference capability boundary.

## 8. Idempotency

Every mutating work packet requires an idempotency key:

```text
Execute(K) == Execute(K)
```

for replay of the same mutation. This protects economic reality from duplicate invoices, reservations, orders, or other repeated writes after ambiguous failures.

The reference runtime retains successful idempotent receipts in memory. Production implementations require durable transactional storage.

## 9. Verification

A successful API response is not proof that work completed correctly.

For every task:

```text
Expected_i
vs
Actual_i
```

is checked. Production verifiers should use authoritative read-back and cross-system reconciliation, for example:

```text
requested reservation == ERP reservation
shipped quantity == WMS quantity == invoice quantity
approved setpoint == controller setpoint inside operating envelope
payment instruction == settlement confirmation
```

## 10. Recovery and compensation

Corporate workflows span systems that generally cannot share one global database transaction. The runtime therefore supports saga-like compensation:

```text
T1 -> T2 -> T3 -> failure
                 |
                 v
          compensate T2
          compensate T1
```

Compensation is not equivalent to rollback in every physical or economic domain. Irreversible effects must be reconciled or corrected through a new authorized action.

## 11. Operational memory

Every control transition is appended to `HashChainedMemory` with:

```text
sequence
event type
workflow id
task id
payload
previous hash
event hash
```

This provides tamper evidence and causal reconstruction, not absolute immutability.

The target lineage is:

```text
CorporateObjectiveID
 -> WorkflowID
 -> RunID
 -> TaskID
 -> ActionID
 -> authoritative outcome
```

The organization should always be able to reconstruct what happened, why, which state was observed, which authority permitted it, which tool acted, and what reality reported.

## 12. Adaptive recurrence

The runtime maps the Adaptive Corporation recurrence onto concrete components:

```text
Xi_t       current corporate state
P_1:M      planner/scenario candidates
Pi_Lambda  business + security authorization
U_t        bounded work execution
E_t        verification/reconciliation error
Omega_t    operational memory
```

Conceptually:

```text
Xi_(t+1) = Pi_Lambda[
  Xi_t
  + P_1:M(Xi_t)
  - E_t
  + Omega_t
  + U_t
]
```

This is a control-system abstraction, not a claim that heterogeneous enterprise state literally supports vector addition.

## 13. Intrusion response

Authority contracts as confidence deteriorates:

```text
normal      -> ordinary bounded writes
elevated    -> stronger verification / additional approval
high risk   -> read-only or restricted writes
intrusion   -> halt mutation and isolate affected scope
```

Containment should be as granular as safely possible: session, token, agent, workflow, service, host, network zone, or plant zone.

## 14. Human work

Humans are first-class workflow actors. A production graph may mix AI tasks, software tasks, human approvals, and machine tasks. High-impact or ambiguous state transitions should deliberately terminate in human authorization.

## 15. Industrial boundary

For industrial execution:

```text
AI / planner
 -> orchestration runtime
 -> MES / approved adapter
 -> SCADA / DCS policy boundary
 -> deterministic PLC / controller
 -> actuator
 -> physical process
```

Safety-critical control remains under deterministic controllers and independently enforced safety systems. The reference runtime must never be interpreted as authority to connect an LLM directly to a safety-critical actuator.

## 16. Economic measurement

A deployment creates market value only when it improves measurable corporate work. Track at minimum:

```text
cycle time
human touches
failure / exception rate
rework
downtime
working capital
throughput
cost per workflow
verification coverage
recovery time
```

The commercialization path is:

```text
architecture
 -> one high-value workflow
 -> measurable ROI
 -> repeatable deployment
 -> multi-workflow expansion
 -> enterprise control plane
```

## 17. Current reference guarantees

`adaptive_orchestrator.py` currently provides:

- DAG validation;
- deterministic wavefront discovery;
- scope-based authorization;
- security-state mutation gating;
- mandatory idempotency for mutating tasks;
- bounded retries;
- output verification;
- reverse-order compensation;
- hash-chained event memory;
- explicit completion/halt receipts.

It does not yet provide durable external workflow state, distributed locks, real parallel workers, external IAM integration, persistent secrets management, human approval queues, policy-as-code integration, message-bus delivery guarantees, distributed tracing, durable idempotency storage, or production OT adapters.

Those are next-layer implementation tasks, not implicit claims.

## 18. Promotion criteria

Before a workflow moves from advisory/shadow mode to autonomous execution, it should demonstrate:

```text
defined objective and owner
bounded action surface
typed tools
policy coverage
idempotency
authoritative read-back verification
known compensation / recovery path
observable failure modes
security threat model
auditability
measured reliability
measured economic value
```

Authority should lag demonstrated reliability.

## 19. Test contract

`tests/test_adaptive_orchestrator.py` validates successful dependency execution, inward security gating under intrusion, verification-failure compensation, mandatory idempotency for mutations, graph rejection of missing dependencies and cycles, policy threshold validation, and hash-chain tamper detection.
