# Jarvis-X System Operational Framework

## Status

This document unifies the VM, sparse spatial, numerical, photonic, adaptive and interface layers into one operational model. It describes architecture and contracts; individual subsystem maturity remains authoritative in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## 1. Coordinate system

Jarvis-X uses two complementary coordinate systems.

### Physical or logical coordinates

```text
r = (x, y, z)
```

These identify a spatial location, sparse brick, voxel, detector projection or logical address.

### Operational coordinates

```text
X = workflow progression
Y = time and persisted state
Z = control and abstraction depth
```

A runtime event can therefore be identified by both where it acts and how it is controlled:

```text
Event = (x, y, z, X, Y, Z)
```

The coordinate model is descriptive. It does not imply dense allocation or additional physical hardware.

## 2. Integrated state

The system state is represented by:

```text
Xi_t = (
    input_t,
    authoritative_vm_t,
    sparse_state_t,
    numerical_state_t,
    proposals_t,
    Omega_t,
    Lambda_t,
    journal_t,
    outputs_t
)
```

Where:

- `input_t` is an explicitly acquired external or synthetic input;
- `authoritative_vm_t` is canonical register, memory and instruction state;
- `sparse_state_t` contains bounded materialized coordinates and bricks;
- `numerical_state_t` contains isolated solver or rendering state;
- `proposals_t` contains non-authoritative adaptive or agent outputs;
- `Omega_t` is residual correction and provenance memory;
- `Lambda_t` contains policy, resource and admissibility constraints;
- `journal_t` is the append-only audit trail;
- `outputs_t` contains verified digital or physical-interface outputs.

## 3. Canonical ten-stage pipeline

Every system-level workflow maps to the following order:

```text
1. Event ingestion
2. Authenticate or establish capability
3. Validate schema and dimensions
4. Route workflow
5. Load authoritative and auxiliary state
6. Plan bounded tasks
7. Execute VM instructions, kernels or APIs
8. Verify proposed results
9. Persist state and emit outputs
10. Complete, halt or wait for the next event
```

The compact runtime loop is:

```text
observe -> route -> execute -> verify -> repeat
```

Subsystems may omit stages that do not apply, but they must not bypass validation, authority or verification when authoritative state is mutated.

## 4. Layer mapping

| Stage | VM | Sparse spatial | Photonic rendering | Adaptive/agent layer |
|---|---|---|---|---|
| Ingest | bytecode word | coordinate/range | scene, camera, spectrum | observation/task |
| Authenticate | policy/capability | ASID and address class | trusted local call | bounded role/capability |
| Validate | decode and bounds | coordinate and residency | dimensions and optical bounds | proposal schema |
| Route | opcode dispatch | brick/tile selection | detector tile partition | task graph |
| Load | registers/memory | sparse bricks | scene/runtime state | Omega/context |
| Plan | next instruction | bounded range operations | deterministic work tiles | candidate actions |
| Execute | instruction semantics | bit/vector operation | spectral transport/integration | shadow execution |
| Verify | state invariant | digest/budget | frame and quantization | independent verifier |
| Persist/emit | ledger/checkpoint | sparse checkpoint | frame/Omega digest | approved proposal only |
| Complete/wait | halt/advance | commit/rollback | commit/rollback | reschedule/wait |

## 5. Authoritative transition

A general transition has three distinct states:

```text
current authoritative state
-> proposed next state
-> verified committed state or rollback
```

Formally:

```text
proposal_t = Execute(Plan(Observe(Xi_t)))
verified_t = Verify(proposal_t, Lambda_t)

Xi_(t+1) = Commit(Project_Lambda(proposal_t))  when verified_t
Xi_(t+1) = Rollback(checkpoint_t)              otherwise
```

The compact research notation is:

```text
Xi_(t+1) = Pi_Lambda[
    Xi_t + P(Xi_t) - E_t + Omega_t + U_t
]
```

This notation is valid only when every term is mapped to an explicit data contract and the projection is implemented as ordinary validation and bounded state transition logic.

## 6. Physical-to-digital transduction

The photonic path adds an explicit physical interpretation:

```text
electromagnetic source
-> spectral radiance field
-> material interaction
-> optical projection
-> finite-area pixel integration
-> quantized detector state
-> sparse 3D address projection
-> optional VM or numerical processing
-> verified digital output
```

The pixel measurement is:

```text
p[i,j,c] = Q_b[
    integral over exposure time
    integral over wavelength
    integral over pixel aperture
    S_c(lambda) L(x, y, lambda, t)
]
```

The reference renderer approximates this expression with deterministic finite samples. It does not solve the full electromagnetic field equations.

## 7. GPU and cloud execution boundary

GPU and cloud workers are execution backends, not independent authorities. They may receive bounded tiles or sparse bricks and return proposals plus evidence:

```text
work_unit = (input_digest, coordinates, parameters, budget, version)
result = (output, metrics, output_digest, provenance)
```

The controller must verify version, dimensions, digest, resource use and semantic invariants before commit. Worker count and physical placement must not alter logical coordinates or transaction semantics.

## 8. Agentic orchestration boundary

An agent is a bounded proposal generator:

```text
agent_i = (role, policy, local_state, inbox, outbox, budget, capability)
proposal_i = policy_i(observation_i)
```

Agent consensus selects a candidate; it does not authorize it. Only the canonical verifier and transaction layer may commit authoritative state.

## 9. System-wide invariants

1. **Authority is explicit.** Predictive, visual, numerical and agent layers produce proposals unless a documented adapter grants narrower mutation rights.
2. **Every mutation is bounded.** Cycle, coordinate, memory, tile, pixel, optical-path and task limits are enforceable.
3. **Virtual extent is not residency.** Sparse address spaces report logical capacity separately from materialized payload.
4. **Encoding is not guaranteed compression.** Smaller representations may be lossy or depend on side information.
5. **Integrity is not reversibility.** Digests establish tamper evidence, not decoding.
6. **Simulation is not deployment evidence.** A rendered or deterministic demonstration does not establish physical accuracy, intelligence or production readiness.
7. **Physical claims require calibration.** A physically interpreted model must state approximations, units, tolerances and reference data.
8. **Execution strategy is replaceable.** CPU, GPU and cloud backends must preserve declared semantic and transaction contracts.
9. **Failure is visible.** Rejected inputs and rollbacks are recorded without silently mutating state.
10. **Every capability has an evidence boundary.** Names and diagrams never substitute for tests.

## 10. Promotion sequence

Subsystems progress in this order:

```text
working -> robust -> portable -> elegant -> advanced
```

A promotion requires:

- a narrow public API;
- deterministic fixtures or a documented stochastic protocol;
- malformed-input and resource-bound tests;
- transaction and rollback semantics;
- serialization and versioning where persistent;
- explicit complexity and resident-memory bounds;
- CI evidence on supported platforms;
- benchmark or reference comparison before performance claims;
- security analysis before accepting untrusted inputs.
