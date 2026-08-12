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
- topology and geometry validation;
- explicitly signed discrete spatial operators and boundary conditions;
- bounded support closure for local field updates.

A large virtual extent never implies dense allocation. Every implementation must state its resident working-set bound.

The Dr Moagi Field Runtime v2 is the canonical reference for a same-space sparse volumetric transition. Its default logical extent may be `1000^3`, while its physical state remains an explicitly bounded active support with deterministic background semantics.

### Layer 5 — Adaptive and generative research systems

Responsibilities may include:

- residual correction memory;
- autoencoder/decoder closure operators;
- multimodal conditioning;
- conditional geometry generation;
- inward latent or field recurrence;
- bounded geometric refinement;
- rendering orchestration;
- multimedia codec and archive adapters;
- inverse geometric inference;
- rate-distortion measurement;
- parameter candidate generation;
- shadow evaluation;
- bounded schedule search;
- coherence projection;
- commit and rollback decisions.

Adaptive systems must optimize constrained representations, not rewrite arbitrary native instructions. Fitness must exclude uncontrolled nondeterministic signals unless repeated statistical evaluation is explicitly part of the contract.

For volumetric autoencoding systems, additive evolution terms must inhabit the same authoritative field space. Latent tensors are transformed through an explicit decoder before they participate in a field residual.

The Moagi-Helmholtz functional is the canonical Layer 5 orchestration contract for multimodal-conditioned geometry generation, refinement, archival coding and reverse inference. It does not require the canonical VM to depend on a neural, renderer or codec backend.

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

Research runtimes that operate outside the core instruction loop use the analogous candidate-first boundary:

```text
snapshot
  → bounded research transform
  → candidate state
  → projection / policy / resource validation
  → commit or rollback
  → telemetry / provenance
```

The research transform cannot make itself authoritative merely by computing a candidate.

## 4. Core invariants

1. **Determinism:** identical authoritative inputs produce identical VM state, excluding explicitly recorded environmental values such as production timestamps.
2. **Bounded execution:** every run has an enforceable cycle limit.
3. **Explicit persistence:** ordinary VM runs do not write files unless a persistence path is supplied.
4. **Integrity:** a valid journal hash binds its timestamp, opcode, state and previous hash.
5. **Fail-closed validation:** malformed bytecode, invalid program counters and corrupt journals are rejected.
6. **Separation of authority:** visual, predictive and adaptive layers cannot silently mutate the canonical VM state.
7. **Honest scale:** virtual geometry is reported separately from resident memory and measured throughput.
8. **No claim by naming:** terms such as intelligence, cognition, self-evolution or neural do not establish those capabilities without operational tests.
9. **Same-space evolution:** terms combined in one authoritative state equation must share a defined state type and compatible units.
10. **Candidate-first adaptation:** active model, schedule, topology, bytecode, or field state is not replaced until its candidate passes the declared admission gate.
11. **Immutable drift reference:** self-referential research loops that measure generational preservation retain an immutable source anchor for the duration of a run.
12. **Explicit generative boundaries:** conditioning, latent geometry, rendered frames, coded bitstreams and archive containers are distinct state types.
13. **No false inverse:** ordinary lossy rendering or video coding is not treated as globally invertible to original geometry.
14. **Rate-distortion visibility:** codec optimization reports both distortion and coded representation size.
15. **Convergence by evidence:** unique convergence is claimed only under sufficient mathematical assumptions or a validated restricted domain.

## 5. Data contracts

### Bytecode

Canonical bytecode words are unsigned 64-bit integers. Persistent binary formats must specify:

- magic and version;
- field layout;
- endianness;
- capacity rules;
- integrity checks;
- malformed-input behavior.

Experimental tensor instruction formats such as 256-bit, 512-bit, DMEB-32, CUDA, FPGA or native-extension encodings are adapters/research targets until a separate accepted ADR promotes them. They may not silently redefine the canonical core bytecode format.

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

### Dr Moagi field candidate

A Field Runtime v2 candidate must declare:

- logical side length and active-support ceiling;
- codec/version identity;
- timestep and coefficients `alpha`, `lambda`, `eta`;
- reconstruction residual definition;
- Laplacian and glyph sign conventions;
- boundary conditions;
- projection limits;
- immutable anchor identity;
- commit/rejection outcome and telemetry.

The canonical field law is

```text
R(Psi) = Psi - D(E(Psi))

dPsi/dt = -alpha R(Psi)
          + lambda Delta_6 R(Psi)
          + eta (G_moagi * Psi).
```

With the canonical glyph stencil,

```text
G_moagi * Psi = -(1/6) Delta_6 Psi.
```

### Moagi-Helmholtz orchestration candidate

A unified generative candidate declares:

```text
multimodal observation/version
conditioning representation/version
latent representation/version
source, generated and refined geometry
renderer/camera/material contract
coded bitstream and archive/container contract
side-information budget
inverse-inference model/version
cycle-reconstruction metric
anchor-drift metric
rate/distortion telemetry
resource ceilings
validator decision
commit/rollback outcome
```

The forward contract is

```text
M -> Phi -> c -> E -> z -> D -> V0 -> Refine -> V* -> Render -> F -> Codec -> B -> Mux -> A
```

and the reverse contract is

```text
A -> Demux -> B -> Decode -> F_hat -> I_phi -> (z_hat,c_hat) -> D -> V_hat0 -> Refine -> V_hat*.
```

The reverse path is inference unless sufficient side information establishes exact replay.

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

A field or latent runtime must additionally demonstrate sparse support confinement, candidate rollback, explicit timestep/resource limits, and honest information-capacity claims for its latent representation.

A generative/archive runtime must additionally separate renderer, codec and container semantics; account for side information; expose cycle error and archive size; and reject candidates before publication when validation fails.

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

ADR-003 extends ADR-002 by defining the same-space Dr Moagi field evolution contract and its executable sparse transaction semantics.

ADR-004 extends ADR-003 by defining the Moagi-Helmholtz multimodal generative, geometric, rendering, archival and inverse-inference orchestration contract.

## 8. Security boundary

The policy layer is an application-level guard, not a complete security sandbox. Untrusted bytecode, native plugins, model files and browser content require dedicated threat models. See [`SECURITY.md`](../SECURITY.md).

Self-modification is never equivalent to unrestricted writes into authoritative code memory. Adaptive code, model, tile, or schedule changes are represented as candidate patches that require validation and commit; otherwise the prior authoritative state remains active.

Decoded multimedia, model weights, geometry sidecars and archive metadata are also untrusted inputs. Backend adapters must validate sizes, formats, coordinate ranges and version bindings before allocating or publishing authoritative state.

## 9. Dr Moagi Field Runtime v2 integration

The evolved system architecture is:

```text
Canonical authority
  Layer 0  representation / 64-bit bytecode
  Layer 1  deterministic execution
  Layer 2  policy + transaction control
  Layer 3  provenance + replay
       |
       v
Research compute envelope
  Layer 4  sparse 3D support
           -> six-neighbour topology
           -> Delta_6
           -> G_moagi
       |
       v
  Layer 5  encode -> decode -> residual
           -> inward field recurrence
           -> candidate adaptation
           -> Pi_Lambda
       |
       v
Admission boundary
  verify -> COMMIT / ROLLBACK -> provenance
       |
       v
  Layer 6  API / CLI / visualization / hardware adapters
```

The operational cycle for the Dr Moagi field subsystem is:

```text
Psi_n
  -> freeze snapshot
  -> bounded support closure
  -> E_theta(Psi_n)
  -> D_theta(z, support)
  -> R_n = Psi_n - Psi_hat_n
  -> H_n = Delta_6 R_n
  -> P_n = G_moagi * Psi_n
  -> rhs_n = -alpha R_n + lambda H_n + eta P_n
  -> candidate = Psi_n + dt rhs_n
  -> Pi_Lambda
  -> shadow/validator checks
  -> COMMIT or ROLLBACK
  -> reconstruction + anchor + resource telemetry
  -> next inward cycle
```

This architecture is backend-neutral. Pure Python, C++, PyTorch/CUDA, distributed sparse workers, FPGA soft cores, and experimental tensor bytecodes may implement the transform, provided they preserve the same state, resource, and transaction semantics.

## 10. Moagi-Helmholtz system-wide permeation

The unified generative architecture composes the existing field and transaction principles into an end-to-end orchestration plane:

```text
OBSERVE multimodal M
  -> CONDITION c = Phi(M)
  -> ENCODE geometry z = E(G)
  -> GENERATE V0 = D(z,c)
  -> REFINE V* under E_MH and Pi_G
  -> RENDER calibrated frames F
  -> CODE bitstream B under rate-distortion control
  -> ARCHIVE A = Mux(B, side_info)
  -> DECODE / DEMUX
  -> INFER (z_hat,c_hat) from decoded evidence + side info
  -> REGENERATE V_hat*
  -> MEASURE cycle error + anchor drift + rate + resources
  -> PROPOSE model/codec/schedule adaptation
  -> Pi_Lambda / shadow validation
  -> COMMIT or ROLLBACK
  -> JOURNAL
  -> RE-ENTER
```

The Moagi-Helmholtz candidate operator is therefore subordinate to, and protected by, the same canonical transaction boundary as the Field Runtime v2:

```text
Xi_(t+1) = Pi_Lambda(M_MH(Xi_t)).
```

Backend lowering is explicit:

```text
Python reference     -> orchestration conformance
C++                  -> geometric kernels / refinement
PyTorch/CUDA         -> conditioning, latent and tensor acceleration
renderer backend     -> calibrated image/depth production
video codec backend  -> rate-distortion coding and container integration
DMEB/tensor ISA      -> bounded accelerator lowering
FPGA/native ISA      -> verified hardware implementation
browser/3D UI        -> visualization and telemetry unless separately promoted
```

None of these backends may bypass the canonical evidence, resource, policy, provenance, or rollback boundaries.
