# ADR-010: Unify candidate admission and Omega evidence across Jarvis-X runtimes

**Status:** Proposed  
**Date:** 2026-08-19

## Context

Jarvis-X now contains several executable state-transition systems that share the same conceptual law but expose different local transaction semantics:

- the canonical `CodexVM` checkpoints authoritative state, executes an instruction, writes ledger/trace receipts and rolls back on failure;
- the Dr Moagi sparse field runtime freezes a field snapshot, computes a projected candidate, invokes an optional validator and commits or rejects the candidate;
- the C++ inward processor evaluates bounded genome candidates and admits only a coherent, sufficiently improved champion;
- the Dr Moagi state-space integration candidate computes bounded latent transitions and logical tile emissions;
- Hyperion constructs deterministic evidence bundles and replay commitments over observations.

Without one executable cross-subsystem transaction contract, the symbols `Lambda`, `Omega`, candidate, commit, rollback and state have compatible intent but different schemas. That fragmentation makes end-to-end replay, orchestration and cross-engine provenance unnecessarily difficult.

## Decision

Jarvis-X adopts a common candidate-first control-plane law:

```text
S_t --Transform--> C_t --Lambda--> decision
                                  | commit
                                  v
                                S_t+1 = C_t
                                  |
                                  +-- Omega receipt

                                  | rollback
                                  v
                                S_t+1 = S_t
                                  |
                                  +-- Omega receipt
```

The common transition is:

```text
C_t = T_Theta(S_t, U_t, Omega_t)

d_t = V_Lambda(C_t)

S_(t+1) = C_t  if d_t = commit
S_(t+1) = S_t  if d_t = rollback
```

Every completed candidate decision emits a deterministic transaction receipt.

### State envelope

A `StateEnvelope` does not contain an unrestricted full state object. It binds:

```text
protocol
state_type
state_version
dimensions
payload_digest
authoritative
```

`payload_digest` is SHA-256 over canonical JSON of the subsystem's declared bounded state representation. A subsystem may hash a compact manifest or a digest of large binary memory rather than materializing its entire logical address space.

The authority flags are explicit:

- `before.authoritative == true`
- `candidate.authoritative == false`
- `after.authoritative == true`

### Transaction receipt

A receipt binds:

```text
protocol
sequence
subsystem
operation
transaction_id
decision
reason
before envelope
candidate envelope
after envelope
metrics
previous_hash
receipt_hash
```

The receipt chain is deterministic and intentionally excludes wall-clock timestamps from the canonical hash. Environmental timestamps may be carried separately by subsystem telemetry when needed.

### Commit invariant

For an admitted candidate:

```text
decision = commit
candidate.payload_digest == after.payload_digest
```

### Rollback invariant

For a rejected candidate:

```text
decision = rollback
before.payload_digest == after.payload_digest
```

A rejected transform therefore cannot claim rollback while silently changing authoritative state.

### Omega evidence chain

The first receipt uses a 64-zero genesis hash. Each later receipt binds the prior receipt hash:

```text
H_t = SHA256(receipt_body_t || previous_hash_t)
previous_hash_(t+1) = H_t
```

The current implementation achieves the equivalent binding by including `previous_hash` in the canonical receipt body before SHA-256 hashing.

The chain establishes deterministic transition integrity. It does not:

- prove that an external sensor or user supplied truthful data;
- encrypt state;
- provide hostile-code isolation;
- prove semantic correctness of an admitted candidate;
- make an experimental subsystem authoritative merely because it emits a receipt.

## Initial permeation

This ADR is introduced with a substrate-neutral Python reference implementation in:

```text
src/jarvisx/control_plane.py
```

The first adapters are:

1. **CodexVM** — every successfully completed instruction emits a common commit receipt. The control-plane checkpoint participates in VM rollback, so failed post-execution receipt work cannot leave evidence ahead of authoritative VM state.
2. **Dr Moagi Field Runtime v2** — each completed field candidate emits either a commit receipt or a rollback receipt. A new field `load()` starts a new evidence chain for that run.

The state-space C++ integration branch is the next adoption point. Its linear contraction telemetry remains distinct from a proof of full nonlinear closed-loop stability.

## Required subsystem adapters

A subsystem joining the control plane must provide:

- a stable `state_type` and positive `state_version`;
- deterministic dimensions;
- a deterministic bounded payload representation;
- an explicit candidate boundary;
- an explicit admission decision;
- authoritative before/after envelopes;
- JSON-native deterministic metrics;
- failure behavior that does not advance authoritative state after a failed receipt operation.

## Relationship to Lambda

`Lambda` remains the subsystem-specific admissibility mechanism. The common control plane does not force every subsystem to use the same validator mathematics.

Instead it standardizes the decision boundary:

```text
subsystem-specific V_Lambda
          |
          v
  commit / rollback
          |
          v
common receipt semantics
```

This preserves local numerical and domain expertise while making global orchestration auditable.

## Relationship to Omega

`Omega` is separated into two meanings that must not be conflated:

1. **working correction memory** used inside a model or runtime, such as residual memory in the C++ processor;
2. **provenance evidence** describing authoritative state transitions.

Both may be called Omega in research notation, but the control-plane evidence chain is specifically the second role. Working correction memory must be included in a state digest when it is authoritative for replay.

## Consequences

### Positive

- VM instructions and sparse field steps now share one transaction vocabulary;
- rollback becomes machine-checkable across subsystem boundaries;
- orchestration can consume one receipt schema without understanding every internal runtime;
- evidence scales with active manifests/digests rather than logical virtual extent;
- deterministic replay is not polluted by wall-clock latency or timestamps;
- future state-space, C++, Hyperion and distributed adapters have a precise target contract.

### Negative

- subsystem state adapters must define canonical bounded representations;
- two provenance systems temporarily coexist in the VM: the historical opcode ledger and the unified control-plane chain;
- SHA-256/canonical-JSON parity must be implemented or delegated carefully in native runtimes before cross-language receipt hashes can be identical;
- this contract proves transition integrity, not convergence, safety, intelligence or source truth.

## Validation

Promotion to Accepted requires:

- deterministic unit tests for state envelopes and transaction IDs;
- chain-integrity and tamper tests;
- explicit commit and rollback invariant tests;
- VM rollback tests covering the common evidence checkpoint;
- field commit/rejection receipt tests;
- CI across supported Python versions;
- a documented native-runtime strategy for hash/serialization parity before claiming cross-language receipt equivalence.

## Follow-up permeation

After this contract is validated, the preferred sequence is:

```text
CodexVM + Field Runtime
        -> C++ inward runtime
        -> Dr Moagi state-space loop
        -> Moagi-Helmholtz orchestration
        -> Hyperion evidence envelope
        -> distributed consensus receipts
```

The end state is one system law:

```text
Observe -> Candidate -> Lambda -> Commit/Rollback -> Omega -> Re-enter
```

with subsystem-specific mathematics behind a common authority boundary.
