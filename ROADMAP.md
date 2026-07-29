# Jarvis-X Roadmap

The roadmap prioritizes consolidation over parallel expansion.

## Operating principle

```text
Working → Robust → Portable → Elegant → Advanced
```

A later stage cannot compensate for defects in an earlier stage. New research tracks should not bypass the canonical foundation.

## Phase 1 — Green and authoritative `main`

**Target:** `0.2.0`

- [ ] merge the canonical VM and ledger repairs;
- [ ] keep all required CI checks green;
- [ ] make packaging metadata single-source;
- [ ] version the external bytecode container;
- [ ] document the supported ISA and failure semantics;
- [ ] raise branch coverage from the initial 60% gate toward 80%;
- [ ] remove hidden filesystem and network side effects from default tests;
- [ ] establish changelog and release notes.

### Exit criteria

- every default-branch commit passes supported Python versions;
- ordinary VM arithmetic is deterministic;
- corrupt journals and malformed bytecode fail closed;
- a clean clone can install and run the documented example.

## Phase 2 — Consolidate the research backlog

- [ ] classify every open pull request as merge candidate, superseded, specification, demonstration, future research or abandoned;
- [ ] close superseded pull requests with successor links;
- [ ] extract duplicated infrastructure repairs into small canonical PRs;
- [ ] reduce active draft pull requests to fewer than ten;
- [ ] create ADRs for competing runtime, spatial and bytecode designs;
- [ ] identify one canonical implementation per capability.

### Exit criteria

- no two active PRs claim authority over the same subsystem without an explicit comparison;
- every active PR names its merge dependency and successor relationship;
- the project-status matrix matches the default branch.

## Phase 3 — Canonical sparse spatial runtime

**Target:** `0.3.0`

- [ ] unify coordinate, block and serialization contracts;
- [ ] implement durable backing stores;
- [ ] add revision-aware transactional concurrency;
- [ ] support bounded heterogeneous blocks or octrees;
- [ ] add crash-recovery fixtures;
- [ ] publish resident-memory and throughput benchmarks;
- [ ] compare against straightforward dense and sparse baselines.

### Exit criteria

- exact address round trips;
- deterministic canonical serialization;
- bounded streaming of large sparse fixtures;
- rollback on failed layout or persistence commits.

## Phase 4 — Versioned ROM and replay

- [ ] publish the canonical ISA table;
- [ ] define a versioned ROM envelope with magic, version, endianness and integrity fields;
- [ ] support inspect, verify, decode and replay commands;
- [ ] add cross-language golden vectors;
- [ ] separate reversible encoding from cryptographic integrity;
- [ ] add compatibility policy for format upgrades.

## Phase 5 — Bounded adaptive laboratory

**Target:** `0.4.0`

- [ ] define a shared experiment manifest;
- [ ] use deterministic candidate generation;
- [ ] evaluate candidates in isolated authoritative state;
- [ ] remove uncontrolled wall-clock signals from deterministic selection;
- [ ] add metric-hacking, NaN, drift and rollback tests;
- [ ] persist accepted and rejected decisions;
- [ ] benchmark against fixed-parameter baselines.

### Exit criteria

- identical manifests reproduce identical decisions;
- rejected candidates cannot alter authoritative state;
- every accepted change has a measurable objective improvement and receipt.

## Phase 6 — Performance and portability

- [ ] establish benchmark datasets and hardware disclosure;
- [ ] profile before optimization;
- [ ] add vectorized CPU backends;
- [ ] add optional FFT and GPU backends where justified;
- [ ] test Linux, Windows and macOS where dependencies permit;
- [ ] separate reference correctness kernels from production acceleration kernels;
- [ ] publish latency, throughput, memory and accuracy together.

## Phase 7 — Research communication

- [ ] publish concise technical reports for canonical subsystems;
- [ ] include equations, implementation mappings and limitations;
- [ ] archive reproducible experiment manifests;
- [ ] provide citation metadata and versioned releases;
- [ ] create a project profile README when repository creation is available;
- [ ] recruit external reviewers for VM, numerical methods and security boundaries.

## Explicit non-goals before `1.0`

- unrestricted autonomous self-modification;
- production security certification;
- claims of consciousness or AGI;
- dense realization of enormous virtual spaces;
- performance claims without reproducible benchmarks;
- merging every experimental branch into one monolith.
