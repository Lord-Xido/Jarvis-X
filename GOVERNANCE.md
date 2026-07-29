# Jarvis-X Governance

Jarvis-X is currently maintained by its founder, Matladi Maxwell Moagi (`@Lord-Xido`). This document defines how technical authority, review and canonical status are determined as the contributor base grows.

## 1. Principles

1. **Evidence outranks terminology.** Names and equations do not establish an implemented capability.
2. **`main` is canonical.** Branches and pull requests are proposals or integration candidates.
3. **Correctness precedes expansion.** Infrastructure repairs should be merged before dependent features.
4. **Reversibility is operational.** State-changing systems must define commit, failure and rollback behavior.
5. **Research boundaries remain visible.** Demonstrations and specifications are not silently promoted to production claims.
6. **History is preserved.** Superseded decisions are documented rather than erased.

## 2. Roles

### Maintainer

A maintainer may:

- merge or close pull requests;
- classify project status;
- approve releases;
- accept or supersede architecture decisions;
- manage security reports and repository settings.

### Contributor

A contributor may propose issues, code, tests, documentation, benchmarks and reviews. Repeated high-quality contributions may lead to delegated ownership of a subsystem.

### Domain reviewer

A domain reviewer provides focused review in areas such as:

- virtual-machine semantics;
- numerical methods;
- cryptography and provenance;
- sparse data structures;
- accelerator kernels;
- browser and graphics runtimes;
- safety and security boundaries.

Domain review is advisory until a maintainer accepts the change.

## 3. Decision process

### Routine changes

Small bug fixes, tests and documentation improvements use normal pull-request review.

### Material architecture changes

Changes to bytecode formats, authoritative state, persistence, security boundaries or public APIs require an Architecture Decision Record under `docs/adr/`.

An ADR must include:

- context;
- considered alternatives;
- decision;
- consequences;
- validation;
- migration or compatibility effect.

### Experimental systems

Experimental work should remain isolated until it satisfies the canonical promotion checklist in `docs/PROJECT_STATUS.md`.

## 4. Pull-request states

| State | Meaning |
|---|---|
| Draft | design or validation is incomplete |
| Ready for review | author believes scope and checks are complete |
| Approved | review found no blocking defect |
| Merged | capability is part of the target branch |
| Closed, superseded | a successor contains the intended work |
| Closed, not planned | the project will not pursue the proposal in its current form |

Draft PRs should not remain indefinitely without a status comment. Long-lived branches must identify dependencies and successors.

## 5. Merge policy

A pull request may merge when:

- required CI checks pass;
- review comments are resolved;
- public behavior is documented;
- tests cover the changed contract;
- experimental claims are bounded;
- the PR is reasonably scoped;
- required ADRs or migration notes exist.

Squash merge is preferred for focused feature branches. Preserve multi-commit history only when the commits form independently meaningful, reviewable stages.

## 6. Release policy

Releases follow semantic versioning as closely as alpha development permits:

- patch: compatible defect and documentation repair;
- minor: new capability or intentional alpha API evolution;
- major: incompatible stable contract change.

Every release should provide:

- a changelog entry;
- tagged source;
- supported Python versions;
- installation and validation commands;
- known limitations;
- format or migration notes.

## 7. Conflict resolution

Technical disagreements should be resolved using:

1. explicit requirements;
2. minimal reproductions;
3. tests or benchmarks;
4. threat models;
5. documented trade-offs.

When evidence does not distinguish alternatives, prefer the simpler and more reversible design.

## 8. Security authority

Potential vulnerabilities are handled privately where possible. Security fixes may bypass normal public discussion until a remediation is available. See `SECURITY.md`.

## 9. Governance evolution

As external participation grows, the project may introduce:

- subsystem maintainers;
- required independent review for security-sensitive changes;
- release managers;
- a formal technical steering group.

Any such transition will be documented through an accepted ADR and an update to this file.
