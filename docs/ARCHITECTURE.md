# Jarvis-X Architecture

## Status

This document defines the canonical architectural boundaries for code merged into the default branch. Experimental pull requests may extend these boundaries, but they do not redefine the canonical system until merged.

## 1. Architectural objective

Jarvis-X is a deterministic bytecode virtual machine surrounded by auditable, bounded research layers. The core must remain understandable without requiring the numerical, photonic, visual or adaptive layers.

The dependency direction is:

```text
representation -> validation -> execution -> observation -> journaling
                                      |
                                      +-> sparse/numerical/photonic adapters
                                      +-> optional bounded adaptation
```

Optional layers may depend on the core. The core must not depend on a particular renderer, neural model, spatial engine, GPU backend, cloud provider or self-optimization strategy.

The system-wide pipeline and coordinate conventions are documented in [`SYSTEM_OPERATIONAL_FRAMEWORK.md`](SYSTEM_OPERATIONAL_FRAMEWORK.md).

## 2. Canonical layers

### Layer 0 — Representation

Responsibilities:

- assembly grammar;
- typed instruction objects;
- fixed-width bytecode encoding;
- deterministic decoding;
- register and memory representation;
- versioned persistent envelopes.

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

- instruction and capability admissibility;
- precondition checks;
- whole-cycle state snapshots;
- proposed-state validation;
- commit or rollback across every authoritative state component.

A subsystem that cannot restore all authoritative state after failure must not claim atomic rollback. External API side effects require explicit compensation or idempotency contracts.

### Layer 3 — Observation and provenance

Responsibilities:

- execution tracing;
- append-only journal entries;
- hash-chain verification;
- persistent state checkpoints;
- deterministic replay fixtures;
- machine-readable evidence artifacts.

Journal entries must contain JSON-native data. Cryptographic digests establish integrity; they are not reversible encodings.

### Layer 4 — Sparse spatial computation

Responsibilities:

- virtual coordinate systems;
- sparse blocks, bricks, tiles or octrees;
- exact logical-to-physical addressing;
- bounded materialization;
- topology and geometry validation;
- deterministic serialization and checkpoint recovery.

A large virtual extent never implies dense allocation. Every implementation must state logical capacity, resident working-set bounds and materialization rules separately.

### Layer 5 — Numerical field systems

Responsibilities:

- finite-dimensional field representations;
- discretization and boundary conditions;
- stable solver steps;
- conservation, dissipation or error metrics;
- reference comparisons and tolerance declarations.

A numerical field is an approximation under an explicit model. It must not be represented as a physically calibrated simulation without units, validation data and error bounds.

### Layer 6 — Physical transduction and photonic rendering

Responsibilities:

- physical-to-digital signal contracts;
- wavelength-dependent source and detector models;
- finite-area pixel integration;
- optical/radiometric transport approximations;
- deterministic detector tiling;
- exposure, transfer and quantization;
- optional pixel/depth projection into sparse 3D coordinates.

The reference photonic renderer uses geometric optics and bounded spectral integration. It is not a full-wave Maxwell solver. CPU, GPU and cloud implementations are interchangeable execution backends only when they preserve the declared pixel, digest and transaction semantics.

### Layer 7 — Adaptive and agentic research systems

Responsibilities may include:

- residual correction memory;
- parameter candidate generation;
- task decomposition and routing;
- shadow evaluation;
- bounded schedule search;
- coherence projection;
- commit and rollback recommendations.

Adaptive and agentic systems produce proposals. They must not silently mutate authoritative VM or spatial state. They optimize constrained representations, not arbitrary native instructions.

### Layer 8 — Interfaces and visualization

Responsibilities:

- CLI, API and browser interfaces;
- 2D or 3D visualization;
- telemetry presentation;
- snapshot and artifact export.

A visualization is not an authoritative compute substrate unless its numerical and transaction contracts are documented and tested. Interactive demonstrations must not be described as production renderers solely because they display a plausible animation.

## 3. Canonical operational cycles

### VM instruction cycle

```text
fetch
-> bounds check
-> decode
-> policy check
-> checkpoint authoritative state
-> execute into a proposed state
-> verify complete state and persistence
-> commit or rollback
-> trace and journal the outcome
-> advance IP on commit
-> enforce cycle limit
-> continue or halt
```

Reflex correction is opt-in. It must never silently change ordinary assembly semantics.

### System workflow cycle

```text
event ingestion
-> authenticate/capability
-> validate schema and dimensions
-> route workflow
-> load state
-> plan bounded tasks
-> execute VM instructions, kernels or APIs
-> verify results
-> persist and emit
-> complete, halt or wait
```

### Photonic frame cycle

```text
validate scene, camera and resource bounds
-> partition detector into deterministic tiles
-> transport spectral energy
-> integrate finite pixel samples
-> apply exposure and quantization
-> compute canonical frame digest
-> verify every output
-> commit frame and Omega digest, or rollback
```

## 4. Core invariants

1. **Determinism:** identical authoritative inputs produce identical declared outputs, excluding explicitly recorded environmental values and documented floating-point tolerance profiles.
2. **Bounded execution:** every run has enforceable cycle, memory and work-unit limits.
3. **Explicit persistence:** ordinary runs do not write files unless a persistence path is supplied.
4. **Integrity:** journal and checkpoint digests bind canonical serialized state.
5. **Fail-closed validation:** malformed bytecode, dimensions, coordinates, models and journals are rejected.
6. **Separation of authority:** visual, predictive, numerical and adaptive layers cannot silently mutate canonical VM state.
7. **Honest scale:** virtual geometry is reported separately from resident memory and measured throughput.
8. **Physical-model honesty:** geometric or radiometric approximations are not represented as full electromagnetic solvers.
9. **Backend equivalence:** acceleration may replace execution strategy, not semantic contracts.
10. **No claim by naming:** terms such as intelligence, cognition, self-evolution, photonic or neural do not establish those capabilities without operational evidence.

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

A canonical VM journal entry contains exactly:

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

Deterministic research journals may replace environmental timestamps with explicit sequence numbers, but the chosen schema must be versioned and documented.

### Spatial state

Every spatial subsystem must define:

- coordinate domain;
- linear address equation;
- boundary behavior;
- sparse materialization unit;
- resident-memory estimate;
- serialization order;
- neighborhood topology.

### Numerical and photonic state

Every numerical or photonic subsystem must define:

- coordinate and unit conventions;
- discrete sampling scheme;
- source, detector and boundary assumptions;
- numerical precision profile;
- resource limits;
- output quantization;
- canonical digest inputs;
- validation references and tolerance boundaries.

### Agent proposal

Every agent or scheduler proposal must define:

- role and capability;
- input-state digest;
- requested operation and coordinates;
- resource budget;
- expected output schema;
- evidence and provenance;
- expiry or replay behavior.

A proposal is never equivalent to a committed transition.

## 6. Integration rule

A research subsystem may enter the canonical package only when it provides:

- a narrow public API;
- executable tests;
- input and state validation;
- deterministic fixtures or a documented stochastic protocol;
- capability boundaries;
- transaction or explicit failure semantics;
- complexity and memory bounds;
- documentation;
- CI coverage on supported platforms;
- no unresolved conflict with existing authoritative state.

Large pull requests should be decomposed into infrastructure, reference kernel, integration, acceleration and demonstration stages.

## 7. Decision records

Material architecture changes must add an Architecture Decision Record under `docs/adr/` using:

```text
# ADR-NNN: Decision title
Status: proposed | accepted | superseded
Context
Decision
Consequences
Validation
```

A newer accepted ADR may supersede an older one, but historical records remain available.

## 8. Security boundary

The policy layer is an application-level guard, not a complete security sandbox. Untrusted bytecode, native plugins, scene files, model files, cloud-worker results and browser content require dedicated threat models. Physical emitters, cameras and actuators additionally require device-level permission, rate, power and fail-safe controls. See [`SECURITY.md`](../SECURITY.md).
