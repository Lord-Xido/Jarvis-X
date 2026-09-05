# ADR-0013: Shared candidate admission and evidence contract

Status: accepted

## Context

Jarvis-X contains multiple bounded adaptive mechanisms: deterministic planners,
field-runtime validators, residual learning, fixed-point refinement, 3D parameter
search, graph-ANN training, C++ genome/schedule search and multimodal candidate
orchestration. They share the architectural principle that a proposal is not
authoritative until it passes validation, but historically each subsystem encoded
that boundary using its own result type, objective semantics and provenance shape.

That duplication creates two risks:

1. semantic drift, where two optimizers interpret `commit` differently; and
2. metric hacking, where a scalar objective can accidentally substitute for a hard
   safety, integrity or resource constraint.

## Decision

Jarvis-X adopts a backend-neutral candidate admission contract for adaptive research
systems. A candidate transition is represented as:

```text
parent state
  -> candidate proposal
  -> hard constraint checks
  -> resource-envelope checks
  -> objective-improvement gate
  -> COMMIT or ROLLBACK
  -> deterministic receipt
```

The canonical reference implementation is `jarvisx.candidate_contract`.

A `CandidateProposal` binds:

- subsystem, candidate and operator identity;
- parent-state and candidate-state digests;
- objective before and after the proposal;
- named component metrics;
- hard constraint results;
- declared resource envelope and observed deterministic usage.

A `CandidateReceipt` binds the proposal to the final decision, objective improvement,
rejection reasons and a deterministic SHA-256 receipt digest.

Hard constraints are authoritative predicates. They are evaluated independently from
the scalar objective. A candidate that violates a hard constraint or resource bound
must roll back even if its scalar objective is superior.

For the default lower-is-better policy, an admissible proposal commits only when:

```text
J_parent - J_candidate > minimum_improvement + epsilon
```

Subsystems may retain specialized search algorithms and metrics. Grid search,
gradient descent, evolutionary search, symbolic planning and hardware-specific
candidate generation are all compatible with this ADR provided their promotion
boundary can be represented by the shared receipt.

## Consequences

### Positive

- one auditable definition of proposal versus authority;
- deterministic cross-runtime promotion receipts;
- hard policy/resource constraints cannot be hidden inside weighted fitness terms;
- subsystem-local optimizer decisions can be independently checked for semantic drift;
- empirical validation can aggregate heterogeneous adaptive systems using one evidence
  shape;
- future Python, C++, CUDA, FPGA and distributed backends can preserve common
  authority semantics without sharing an optimization algorithm.

### Costs

- existing adaptive runtimes require adapters or native migration;
- objective direction and metric semantics must be declared explicitly;
- state identities used in receipts must be serialized canonically before hashing;
- a receipt proves the declared admission computation, not the truth or completeness of
  the constraints themselves.

## Validation

The initial integration provides:

- deterministic commit and rollback receipt tests;
- a test proving a better objective cannot override a failed hard constraint;
- resource-overrun rollback tests;
- deterministic metric normalization and receipt hashing;
- a virtual-3D optimizer adapter that requires local promotion semantics to agree with
  the shared contract;
- Empirical Validation v2 evidence for the common contract plus selected adaptive
  runtimes.

The migration is intentionally incremental. A subsystem is not considered migrated
merely because this ADR exists; its tests or adapter must demonstrate conformance.

## Boundary

This contract is an application-level authority and evidence mechanism. It is not a
hostile-code sandbox, a proof that an objective is well specified, an AGI or safety
certification, or permission for unrestricted source-code self-modification.
