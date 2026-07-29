# ADR-001: Separate the canonical VM core from research layers

**Status:** Accepted  
**Date:** 2026-07-29

## Context

Jarvis-X contains a small executable virtual-machine core and a large portfolio of experimental spatial, adaptive, visual and numerical systems. Several long-lived pull requests combine repository repairs with new architectures. This makes it difficult to identify authoritative behavior, review changes independently and keep `main` green.

## Decision

Jarvis-X will maintain a narrow canonical core containing:

- assembly and bytecode representation;
- deterministic VM execution;
- policy and resource bounds;
- tracing and verifiable provenance;
- stable public contracts.

Spatial engines, adaptive optimizers, neural models, visualizations and accelerator kernels will integrate through explicit APIs and remain isolated until they satisfy the promotion checklist in `docs/PROJECT_STATUS.md`.

Infrastructure repairs must be extracted from feature branches whenever practical.

## Consequences

### Positive

- `main` becomes the unambiguous source of truth;
- experimental systems can evolve without destabilizing ordinary VM semantics;
- reviews become smaller and evidence-driven;
- duplicate fixes are reduced;
- releases can communicate stable versus experimental capability honestly.

### Negative

- some feature pull requests require rebasing or decomposition;
- terminology and data structures may need adapters instead of direct coupling;
- experimental code may remain unmerged longer;
- maintainers must actively classify and close superseded work.

## Validation

The decision is successful when:

- default-branch CI remains green;
- every merged subsystem has a narrow public contract;
- open PRs identify their integration dependency;
- the project-status matrix matches the default branch;
- ordinary VM tests do not depend on optional research layers.

## Alternatives considered

### Merge all active branches into one system

Rejected because overlapping state models and duplicated prerequisites would create an unreviewable monolith.

### Keep all work permanently isolated

Rejected because it would prevent the strongest research components from becoming usable, versioned software.

### Treat documentation as the canonical system

Rejected because executable behavior, tests and persistent formats require an authoritative implementation.
