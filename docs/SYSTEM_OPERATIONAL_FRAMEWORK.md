# Jarvis-X System Operational Framework

**Status:** canonical operational integration contract  
**Date:** 2026-08-10  
**Core rule:** authority is earned by executable contracts, bounded resources, tests and commit/rollback semantics.

## 1. System objective

Jarvis-X is operated as a deterministic transactional virtual machine with bounded research subsystems around it. Advanced spatial, codec, adaptive, predictive, transport and visualization layers may propose state or derive telemetry, but they do not silently become authoritative compute state.

The constitutional transaction is:

```text
observe / acquire
  -> construct typed candidate state
  -> validate representation and resources
  -> execute bounded transformation
  -> measure evidence
  -> Pi_Lambda admissibility
  -> atomic commit or rollback
  -> Omega provenance / adaptive-memory update
  -> emit receipt and telemetry
```

A symbolic name, virtual scale, visualization, digest or recursion depth does not establish physical capability by itself.

## 2. Authority planes

### Layer 0 — Representation

Authoritative contracts:

- assembly grammar;
- typed instructions;
- fixed-width bytecode;
- versioned persistent binary envelopes;
- exact shape, dtype and endianness where serialized.

### Layer 1 — Execution

Authoritative contracts:

- `CodexVM` instruction dispatch;
- register/memory mutation;
- instruction pointer and cycle state;
- halt/failure semantics;
- enforced cycle ceiling.

### Layer 2 — Policy and transactions

Authoritative contracts:

- instruction policy;
- precondition checks;
- whole-cycle VM checkpoint;
- fail-closed validation;
- commit or rollback.

`Pi_Lambda` is the general admissibility mechanism. Policy/ethical rules may contribute constraints, but the complete gate also includes numerical, version, integrity and resource constraints.

### Layer 3 — Observation and provenance

Authoritative provenance:

- trace records;
- Omega journal;
- hash-chain verification;
- explicit checkpoints;
- reproducible receipts.

A digest establishes integrity of represented bytes; it does not imply reversibility or truth of the represented claim.

### Layer 4 — Sparse spatial and codec computation

Bounded research authority:

- sparse virtual coordinate systems;
- octree and billion-address references;
- 3D codec bitstreams;
- shape/version/integrity contracts;
- bounded materialization.

Virtual extent is always reported separately from resident allocation.

### Layer 5 — Adaptive laboratory

Candidate-only authority until commit:

- residual memory;
- parameter mutation;
- architecture candidates;
- schedulers;
- predictors;
- shadow evaluation.

Candidate state cannot mutate active state before validation. Architecture or entropy-model versions participating in one codec transaction must remain coherent for the entire transaction.

### Layer 6 — Interfaces

Non-authoritative interfaces:

- FastAPI;
- CLI;
- browser dashboard;
- visualization;
- export and telemetry presentation.

Interfaces invoke canonical operations; they do not redefine instruction or codec semantics.

## 3. Canonical operational facade

`src/jarvisx/operational.py` provides the dependency-light execution path shared by interfaces.

```text
source
  -> Parser
  -> Assembler
  -> bytecode
  -> fresh CodexVM
  -> bounded execution
  -> ledger verification
  -> VMExecutionReceipt
```

Every interface-level VM request receives a fresh VM instance so state is not accidentally shared across independent API requests.

The receipt contains:

```text
registers
cycles
ledger_entries
ledger_valid
trace_entries
```

## 4. Dr Moagi 3D codec transaction

The executable alpha reference is:

```text
X
  -> validate
  -> mean-centre E_ref^3D
  -> Q_Delta
  -> signed-64 latent Z
  -> zlib C(Z)
  -> versioned JX3D bitstream B
  -> SHA-256 integrity verification
  -> C^-1(B)
  -> Q_Delta^-1
  -> X_hat
  -> local distortion / anchor distortion / rate
  -> admissibility gate
  -> commit Omega_codec or rollback
```

The reference separates:

```text
Omega_vm     = provenance journal
Omega_codec  = bounded codec statistics
```

These domains must not be conflated.

## 5. Virtual-depth contract

A macro depth such as:

```text
virtual_depth = 1_000_000
```

is a logical parameter unless execution evidence demonstrates equivalent physical work. Every implementation exposing virtual depth must also expose measured execution fields such as:

```text
measured_microsteps_executed
wall_clock_seconds
measured_throughput
resident_memory
```

The current codec reference executes one physical transaction per call and reports that fact explicitly.

## 6. Operational entrypoints

### Python

```python
from jarvisx.operational import execute_source
receipt = execute_source("SET Ψ 10\nHALT")
```

### CLI

```text
jarvisx run program.codex
jarvisx api --host 0.0.0.0 --port 8080
jarvisx web --host 0.0.0.0 --port 8080
jarvisx codec volume.json --bitstream volume.jx3d --reconstructed reconstructed.json
jarvisx codec-decode volume.jx3d decoded.json
jarvisx node --host 127.0.0.1 --port 9000
```

The `node` command remains a legacy research interface and defaults to loopback through the CLI. It is not a production hostile-network execution boundary.

### HTTP

```text
GET  /
GET  /health
POST /run
POST /codec/roundtrip
```

The browser console and JSON API share one FastAPI/uvicorn runtime. Flask is not part of the canonical dependency path.

### Container

The root `Dockerfile` installs the package defined by `pyproject.toml` and serves:

```text
uvicorn jarvisx.api:app
```

CI builds the image, starts it and probes `/health`.

## 7. External and historical subsystem adapters

The following project artifacts are integration targets, not implicit core authority.

### CodexLang

Target integration:

```text
CodexLang source
  -> lexer
  -> typed parser / AST
  -> semantic validation
  -> Jarvis IR
  -> canonical VM or accelerator backend
```

Until that compiler chain exists and is tested, CodexLang-specific syntax remains specification/prototype territory.

### ROM / custom ISA

Target integration:

```text
Jarvis IR
  -> backend lowering
  -> assembler
  -> actual binary ROM bytes
  -> computed digest / Merkle metadata
  -> emulator or hardware test vectors
```

Textual mock-disassembly and placeholder digests are not treated as a verified ROM image.

### CLTP / Lightning Bridge

Target integration:

```text
validated object
  -> canonical codec/serializer
  -> content-addressed envelope
  -> authenticated transport adapter
  -> bounded storage namespace
```

Network exposure requires path canonicalization, request limits, authentication/authorization and integrity verification.

### EM agentic runtime

Target integration:

```text
IQ acquisition
  -> bounded DSP
  -> feature extraction
  -> classified candidate action
  -> policy/resource gate
  -> hardware-control adapter
```

Placeholder hardware classes are not promoted to operational status without device-independent tests and hardware-in-loop evidence.

### FractalGAN / neural generators

Target integration:

```text
versioned model
  -> deterministic/stochastic protocol declaration
  -> bounded inference
  -> benchmark fixtures
  -> provenance
  -> optional candidate contribution to Layer 5
```

Package scaffolding without a model/loss/training/inference implementation remains non-authoritative.

### Predictive / symbolic frameworks

Symbolic forecasting equations may produce candidate features or decision context. Empirical forecasting claims require an observation mapping, calibrated parameters, baselines, held-out evaluation and uncertainty reporting.

### 3D visualization

Visualization consumes telemetry and state snapshots. It does not become authoritative merely because it renders the system in 3D.

## 8. Security boundaries

The canonical runtime currently provides application-level bounds, not process isolation for hostile programs.

Required deployment rules:

- do not expose raw legacy node execution to untrusted networks;
- place public APIs behind authentication, TLS termination, request-size/rate controls and observability;
- treat uploaded bitstreams/model files as untrusted input;
- reject invalid shape/version/digest/resource states before allocation or mutation;
- retain independent backups for provenance journals;
- do not interpret hash integrity as confidentiality;
- isolate native plugins and hardware control paths from the VM process.

## 9. Evidence hierarchy

Every promoted capability should progress through:

```text
specification
  -> executable reference
  -> invariant tests
  -> malformed/adversarial tests
  -> machine-readable evidence
  -> benchmark against named baseline
  -> integration CI
  -> documented capability boundary
```

The current Dr Moagi codec operationalization reaches the executable-reference, focused-invariant, API/container-CI and documented-boundary stages. Consolidation into the package-wide empirical evidence schema is a subsequent promotion step.

## 10. System-wide invariants

1. identical authoritative inputs and declared versions produce reproducible state where deterministic mode is selected;
2. every VM run has an enforceable cycle ceiling;
3. codec bitstreams are versioned and integrity-checked;
4. entropy decoding cannot allocate beyond declared latent/resource ceilings;
5. self-reference preserves an immutable anchor for drift detection;
6. adaptive candidates commit atomically or leave authoritative state unchanged;
7. virtual spatial extent is distinct from resident memory;
8. virtual recursion depth is distinct from measured physical throughput;
9. interfaces and visualizations are non-authoritative;
10. unsupported advanced subsystems remain explicitly classified as specification, scaffold, demonstration or integration target.

## 11. Promotion sequence

The preferred implementation order remains:

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```

### Working

- deterministic VM;
- executable 3D codec reference;
- unified FastAPI/CLI/container paths.

### Robust

- consolidated codec evidence artifact;
- fuzzing of bitstream headers/payloads;
- API authentication and quotas for public deployment;
- durable codec-state persistence and recovery;
- version migration tests.

### Portable

- equivalent C++ codec reference;
- cross-platform bitstream fixtures;
- portable sparse storage;
- Windows/macOS/Linux packaging.

### Elegant

- typed Jarvis IR shared by CodexLang, VM and ROM backends;
- common telemetry/receipt schema;
- unified capability manifest.

### Advanced

- learned 3D transforms and entropy models;
- bounded latent macro-transition acceleration;
- shadow architecture evaluation and rollback;
- GPU/accelerator backends;
- authenticated distributed transport;
- hardware-in-loop EM adapters.

No advanced stage may weaken the deterministic core boundary or replace measured evidence with naming, visualization or virtual scale.
