# ADR-010: Bounded deterministic 3D multiparallel pipeline

- **Status:** Proposed
- **Date:** 2026-08-17
- **Decision scope:** isolated Layer 5 Python research subsystem
- **Tracking:** issue #113

## Context

The supplied JARVIS X multiparallel architecture combines package splitting, process
parallelism, 3D code mapping, branch exploration, compression and genetic topology
adaptation. Several parts need a stricter repository interpretation:

- arbitrary source-line rotation is not generally semantics preserving;
- asynchronous completion must not alter reconciliation order;
- wall-clock duration is not a deterministic fitness input;
- arbitrary stage callables would turn topology data into execution authority;
- compressed packages require framing and bounded decode verification;
- topology candidates must not mutate active state before admission;
- a research pipeline must not become authoritative `CodexVM` state by naming.

Older draft PRs #22 and #43 contain related visual and geometric concepts but predate
the canonical separation and task-control boundaries. Reviving either wholesale would
also import unrelated historical changes.

## Decision

Introduce `jarvisx.multiparallel` and `jarvisx.multiparallel_cli` as an isolated,
dependency-free reference with these rules:

1. accept only text, bytes, immutable vertex batches and complete triangular meshes;
2. use a closed stage enum and a validated v1 linear DAG;
3. derive package and run identities from canonical SHA-256 material;
4. collect work asynchronously but reconcile only by source sequence;
5. frame encoded chunks in a versioned, length-delimited `JXMP` envelope;
6. validate declared and cumulative output bounds before decompression;
7. map source to read-only code geometry and never regenerate source from transformed points;
8. keep branches immutable and return merged results without silent main-state promotion;
9. search a finite type-safe topology family with a seeded generator;
10. use compression and deterministic estimated work for fitness while reporting timing separately;
11. promote a topology only after a final complete verification run;
12. keep the subsystem outside `jarvisx.core` and `jarvisx.system_runtime` authority.

## Governing invariant

```text
parallel proposals
  -> deterministic source-order reconciliation
  -> integrity/resource verification
  -> research-run commit or rollback
```

This is not:

```text
parallel completion -> arbitrary state mutation
```

## Consequences

### Positive

- the operational pipeline has executable and testable kinetics;
- process scheduling cannot change merged bytes or run identity;
- compressed outputs are independently verifiable and bounded;
- topology search is replayable and cannot select on host timing noise;
- code visualization no longer implies source transformation;
- `CodexVM` and system-runtime semantics remain unchanged.

### Trade-offs

- v1 supports a linear DAG rather than arbitrary fan-out/fan-in graphs;
- the standard-library tuple representation is a correctness reference, not a vectorized kernel;
- process startup may dominate small workloads;
- the finite fitness model cannot establish optimality beyond its fixture and objective;
- meshes remain one package so face topology is not split incorrectly.

## Alternatives considered

### Use arbitrary Python processors in each node

Rejected because pickled callables weaken auditability and allow topology data to carry
unbounded execution behavior.

### Rewrite code after rotating its spatial points

Rejected because Python semantics depend on source order, indentation, binding and
control flow. Read-only geometry preserves the visualization without a false
semantics-equivalence claim.

### Select topology using measured wall-clock speed

Rejected for the reference selector because scheduler load and process startup make
single-run timing nondeterministic. Timing remains telemetry for empirical benchmarks.

### Merge old multiparallel draft PRs wholesale

Rejected because they combine superseded core changes and unrelated integration
dependencies. Issue #113 extracts a current, narrow subsystem.

## Validation

Acceptance requires the evidence enumerated in
`docs/JARVIS_X_3D_MULTIPARALLEL_PIPELINE.md`, including process/sequential equivalence,
stable reconciliation, frame tamper rejection, branch isolation, deterministic search,
candidate-first promotion and resource-bound tests.

