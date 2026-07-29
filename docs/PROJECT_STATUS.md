# Jarvis-X Project Status

**Last reviewed:** 2026-07-29  
**Release line:** `0.1.x` alpha

This document is the authoritative implemented-versus-experimental capability matrix. Names, diagrams and specifications do not imply implementation.

## Status definitions

| Status | Meaning |
|---|---|
| Stable reference | deterministic implementation with focused tests; API may still change before `1.0` |
| Alpha | executable on `main`, but incomplete or lightly tested |
| Integration candidate | implemented on a branch or pull request and awaiting canonical integration |
| Demonstration | runnable visualization or example that is not an authoritative runtime |
| Specification | documented mathematical or architectural proposal |
| Future work | issue or roadmap item without a canonical implementation |

## Default-branch capabilities

| Capability | Status | Evidence | Boundary |
|---|---|---|---|
| Assembly parser and assembler | Alpha | `src/jarvisx/parser.py`, `assembler.py`, tests | grammar and diagnostics remain minimal |
| 64-bit bytecode decode | Alpha | `decoder.py`, `instruction.py` | versioned external byte envelope is not yet canonical |
| Register VM | Alpha | `core.py`, `executor.py` | canonical ISA currently has a small instruction surface |
| Lambda instruction policy | Alpha | `ethics.py` | application-level guard, not a security sandbox |
| Cycle sandbox | Stable reference | `sandbox.py` | cycle bound only; no process isolation |
| Trace and Omega journal | Stable reference after PR #47 | `tracer.py`, `ledger.py`, `ledger_store.py` | persistence is opt-in; timestamps are environmental inputs |
| Fractal octree | Stable reference | `fractal_octree.py`, invariant tests | geometric reference, not a general sparse database |
| Hugging Face exporter | Stable reference | `scripts/export_huggingface_model.py` | initialized weights are not trained weights |
| Reality-grounded observer dynamics | Specification | `docs/REALITY_GROUNDED_OBSERVER_DYNAMICS.md` | proposed formal framework |
| 3D swarm bytecode architecture | Specification | `docs/DR_MOAGI_3D_SWARM_BYTECODE_ISA.md` | document does not establish hardware performance |

## Active integration candidates

| Pull request | Capability | Current classification | Required before merge |
|---|---|---|---|
| #45 | dependency-free C++17 sparse inward runtime | Integration candidate | repository-wide CI green, review, canonical API boundary |
| #46 | hierarchical fractional 3D smoothing solver | Integration candidate | review numerical limits, benchmark reference, merge sequencing |
| #47 | VM, CI, documentation and governance foundation | Integration candidate | all required checks green and final review |

Other long-lived draft pull requests are research branches until explicitly classified, rebased and validated.

## Separate visual-computing repository

`Lord-Xido/3D-Virtual-AI-Interactive-Interface` contains browser and native visual demonstrations. Its merged `2048³` voxel-video machine is a bounded sparse renderer over a large virtual address space. It does not allocate or display every virtual voxel.

Open visual-engine pull requests remain demonstrations or integration candidates until merged.

## Claims not made

Jarvis-X does not currently claim:

- consciousness or subjective experience;
- artificial general intelligence;
- unrestricted autonomous source-code mutation;
- production-grade isolation of hostile bytecode;
- lossless compression of arbitrary data into a smaller state without side information;
- physical hardware performance from a virtual address-space description;
- trained model quality from deterministic initialized weights;
- safety certification from the presence of a policy or coherence gate.

## Canonical promotion checklist

A capability moves to `main` only when all applicable items are satisfied:

- [ ] precise public API and data contract;
- [ ] deterministic tests or documented stochastic protocol;
- [ ] malformed-input and boundary tests;
- [ ] persistence and rollback behavior;
- [ ] complexity and memory bounds;
- [ ] implemented-versus-proposed documentation;
- [ ] CI on supported platforms;
- [ ] benchmark or reference comparison for performance claims;
- [ ] security analysis for untrusted inputs;
- [ ] migration or compatibility note when replacing an existing subsystem.

## Release-readiness targets

### `0.2.0` — Canonical VM foundation

- stable execution lifecycle;
- JSON-native verifiable journal;
- versioned bytecode container;
- expanded ISA tests;
- enforced CI and contributor documentation.

### `0.3.0` — Sparse spatial runtime

- one canonical sparse coordinate API;
- durable block storage;
- deterministic serialization;
- transactional concurrency;
- benchmark corpus.

### `0.4.0` — Bounded adaptive laboratory

- shadow evaluation API;
- reproducible candidate generation;
- rollback-complete state transitions;
- metric-hacking tests;
- experiment manifests and replay.
