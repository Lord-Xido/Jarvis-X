# Canonical Kinetic Runtime

## Status

Integration candidate for the Jarvis-X canonical execution plane.

This document operationalizes the engine's own kinetics as one bounded transaction protocol shared by VM, sparse-field, ANN, scheduler, codec and deployment adapters.

The governing law is:

```text
snapshot
-> observe
-> encode
-> propose
-> shadow
-> verify
-> commit | rollback
-> journal
-> re-enter
```

A computation does not become authoritative merely because it produced a candidate.

## 1. State transition

For authoritative state `Xi_t`, observation `U_t`, active mechanics `M_t` and candidate operator `F`:

```text
Xi_candidate = F_Mt(Xi_t, U_t)
```

Admission is transactional:

```text
Xi_(t+1) = Xi_candidate   when V_Lambda(Xi_t, Xi_candidate) = pass
Xi_(t+1) = Xi_t           otherwise
```

The validator is a conjunction, not a single opaque score:

```text
V_Lambda =
    V_schema
  & V_correctness
  & V_numerics
  & V_determinism
  & V_resources
  & V_security
  & V_semantics
  & V_provenance
```

Adapters may add domain-specific gates, but no adapter may bypass the common commit boundary.

## 2. Kinetic stages

### SNAPSHOT

Freeze an isolated copy or immutable reference to the complete authoritative state needed for rollback.

### OBSERVE

Read runtime inputs and telemetry without mutating authority.

### ENCODE

Convert state plus observation into the representation used by the candidate generator. This may be bytecode, sparse geometry, latent tensors, telemetry vectors or a typed adapter state.

### PROPOSE

Generate one bounded candidate. Proposal authority is intentionally weaker than commit authority.

### SHADOW

Evaluate the candidate against the same anchor state or event stream used by the baseline. Shadow execution must not publish irreversible effects.

### VERIFY

Run all declared Lambda validators. Every validator returns a named pass/fail result, metrics and optional reason.

### COMMIT / ROLLBACK

Commit only if every required validator passes. Otherwise restore or retain the snapshot.

### JOURNAL

Emit a deterministic receipt binding parent state, candidate, resulting state, validator evidence and the previous receipt hash.

### RE-ENTER

Feed the committed state and telemetry into the next world-state or mechanics-state cycle.

## 3. Canonical receipt

The v1 receipt records:

```text
schema_version
transaction_id
parent_state_hash
candidate_hash
resulting_state_hash
decision
ordered kinetic stages
validator results
shadow telemetry
previous_receipt_hash
receipt_hash
```

Receipt hashes are integrity evidence. They are not encryption, external witnessing, trusted time or deletion resistance.

## 4. Dual-loop operation

The same protocol applies to both world-state and mechanics-state evolution.

### World-state loop

```text
Xi_t
-> observe U_t
-> encode
-> propose Xi_candidate
-> shadow
-> Lambda
-> commit/rollback
-> receipt
-> Xi_(t+1)
```

### Mechanics-state loop

```text
M_t + telemetry + meta-memory
-> encode mechanics residual
-> propose bounded mechanics candidate
-> shadow baseline versus candidate
-> Lambda
-> canary adapter when required
-> commit/rollback mechanics version
-> receipt
-> M_(t+1)
```

The world-state commit and mechanics-state commit remain separate transactions.

## 5. Backend mapping

```text
Python VM            -> authoritative instruction/state adapter
Dr Moagi Field       -> sparse candidate-state adapter
Inward ANN optimizer -> parameter/model candidate adapter
C++ runtime          -> native execution adapter
CUDA/WebGPU/FPGA     -> accelerator lowering adapter
browser runtime      -> visualization/telemetry unless separately promoted
cloud worker         -> isolated shadow/candidate executor
```

Backends may optimize implementation mechanics but may not alter transaction semantics silently.

## 6. External effects

Network, device, email, cloud and other irreversible operations remain outside the in-memory commit boundary until wrapped in a staged protocol:

```text
PREPARE
-> AUTHORIZE
-> EXECUTE IDEMPOTENTLY
-> RECEIVE EXTERNAL RECEIPT
-> COMMIT LOCAL REFERENCE
```

A failed local candidate must never imply that an already-published external action was rolled back.

## 7. Verification requirement

Bounded assembly equivalence uses distinct symbolic namespaces for each compared program and constrains both programs to the same symbolic initial state. Z3 is an enforced CI dependency for the verification lane.

Formal equivalence remains bounded by the configured step limit and supported ISA.

## 8. Promotion rule

A runtime adapter is canonical only when it provides:

- explicit state identity;
- isolated snapshot semantics;
- bounded candidate generation;
- shadow execution;
- named Lambda validators;
- commit and rollback behavior;
- deterministic receipt generation;
- replay fixtures;
- resource bounds;
- CI coverage;
- explicit external-effect boundary.

This contract is the operational bridge from the canonical VM foundation to the full inward-recursive Jarvis-X stack.