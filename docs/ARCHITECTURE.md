# Jarvis-X Architecture

## Status

This document defines the canonical architectural boundaries for code merged into the default branch. Experimental pull requests may extend these boundaries, but they do not redefine the canonical system until merged.

## 1. Architectural objective

Jarvis-X is a deterministic bytecode virtual machine surrounded by auditable research layers. The core must remain understandable without requiring the speculative layers.

The architecture follows this dependency direction:

```text
representation → validation → execution → observation → journaling
                                     │
                                     └→ optional bounded adaptation
```

Optional layers may depend on the core. The core must not depend on a specific visualizer, neural model, spatial engine or self-optimization strategy.

## 2. Canonical layers

### Layer 0 — Representation

Responsibilities:

- assembly grammar;
- typed instruction objects;
- fixed-width bytecode encoding;
- deterministic decoding;
- register and memory representation.

Required properties:

- exact field widths;
- explicit endianness where bytes are serialized;
- rejection of malformed inputs;
- round-trip tests for every persistent format.

### Layer 1 — Execution

Responsibilities:

- instruction dispatch;
- authoritative register and memory mutation;
- instruction-pointer management;
- cycle accounting;
- halt and failure behavior.

Required properties:

- ordinary instruction semantics are stable by default;
- side effects outside VM state are explicit;
- execution is bounded by a sandbox limit;
- invalid states fail closed.

### Layer 2 — Policy and transaction control

Responsibilities:

- instruction admissibility;
- precondition checks;
- proposed-state validation;
- commit or rollback where operations span multiple state domains.

The current core provides a minimal policy check. Experimental transactional engines must not be represented as canonical until rollback covers every authoritative state component.

### Layer 3 — Observation and provenance

Responsibilities:

- execution tracing;
- append-only journal entries;
- hash-chain verification;
- persistent state checkpoints;
- deterministic replay fixtures.

Journal entries must contain JSON-native data. Cryptographic digests establish integrity; they are not reversible encodings.

### Layer 4 — Sparse spatial computation

Responsibilities:

- virtual coordinate systems;
- sparse blocks, bricks, tiles or octrees;
- exact logical-to-physical addressing;
- bounded materialization;
- topology and geometry validation.

A large virtual extent never implies dense allocation. Every implementation must state its resident working-set bound.

### Layer 5 — Adaptive research systems

Responsibilities may include:

- residual correction memory;
- parameter candidate generation;
- shadow evaluation;
- bounded schedule search;
- coherence projection;
- commit and rollback decisions.

Adaptive systems must optimize constrained representations, not rewrite arbitrary native instructions. Fitness must exclude uncontrolled nondeterministic signals unless repeated statistical evaluation is explicitly part of the contract.

### Layer 6 — Interfaces and visualization

Responsibilities:

- CLI, API and browser interfaces;
- 2D or 3D visualization;
- telemetry presentation;
- snapshot export.

A visualization is not an authoritative compute substrate unless the architecture and tests demonstrate that role.

## 3. Canonical VM cycle

```text
fetch
  → bounds check
  → decode
  → policy check
  → execute
  → snapshot
  → journal
  → trace
  → optional reflex correction
  → advance IP
  → enforce cycle limit
  → continue or halt
```

Reflex correction is opt-in. It must never silently change normal assembly semantics.

## 4. Core invariants

1. **Determinism:** identical authoritative inputs produce identical VM state, excluding explicitly recorded environmental values such as production timestamps.
2. **Bounded execution:** every run has an enforceable cycle limit.
3. **Explicit persistence:** ordinary VM runs do not write files unless a persistence path is supplied.
4. **Integrity:** a valid journal hash binds its timestamp, opcode, state and previous hash.
5. **Fail-closed validation:** malformed bytecode, invalid program counters and corrupt journals are rejected.
6. **Separation of authority:** visual, predictive and adaptive layers cannot silently mutate the canonical VM state.
7. **Honest scale:** virtual geometry is reported separately from resident memory and measured throughput.
8. **No claim by naming:** terms such as intelligence, cognition, self-evolution or neural do not establish those capabilities without operational tests.

## 5. Data contracts

### Bytecode

Canonical bytecode words are unsigned 64-bit integers. Persistent binary formats must specify:

- magic and version;
- field layout;
- endianness;
- capacity rules;
- integrity checks;
- malformed-input behavior.

### Journal

A canonical journal entry contains exactly:

```json
{
  "timestamp_ns": 0,
  "opcode": 0,
  "state": {},
  "previous_hash": "64 hexadecimal characters",
  "hash": "64 hexadecimal characters"
}
```

The digest is SHA-256 over canonical JSON of every field except `hash`.

### Spatial state

Every spatial subsystem must define:

- coordinate domain;
- linear address equation;
- boundary behavior;
- sparse materialization unit;
- resident-memory estimate;
- serialization order;
- neighborhood topology.

## 6. Integration rule

A research subsystem may enter the canonical package only when it provides:

- a narrow public API;
- executable tests;
- input and state validation;
- deterministic fixtures;
- capability boundaries;
- documentation;
- CI coverage on supported platforms;
- no unresolved conflict with existing authoritative state.

Large pull requests should be decomposed into infrastructure, kernel, integration and demonstration stages.

## 7. Decision records

Material architecture changes should add an Architecture Decision Record under `docs/adr/` using:

```text
# ADR-NNN: Decision title
Status: proposed | accepted | superseded
Context
Decision
Consequences
Validation
```

A newer accepted ADR may supersede an older one, but historical records should remain available.

## 8. Security boundary

The policy layer is an application-level guard, not a complete security sandbox. Untrusted bytecode, native plugins, model files and browser content require dedicated threat models. See [`SECURITY.md`](../SECURITY.md).
