# ADR-031: 3D Self-Hosting Inward Runtime Closure

**Status:** Accepted  
**Date:** 2026-09-29  
**Applies to:** Jarvis-X compiler/runtime integration, bytecode VM, 3D Cyber-IDE, sparse memory backends, graphics projections, adaptive loops, future hardware and distributed adapters  
**Extends:** ADR-016, ADR-017, ADR-026, ADR-027, ADR-030  
**Canonical specification:** docs/DR_MOAGI_SELF_HOSTING_INWARD_RUNTIME.md

## Context

Jarvis-X already has a canonical typed-state and transaction boundary (ADR-016), a canonical auto-encoding/decoding processing law (ADR-017), bounded GUI/3D projection surfaces, deterministic bytecode execution, sparse and volumetric memory models, adaptive residual/model/runtime-policy state, and the merged 3D Cyber-IDE pixel engine.

The remaining architectural discontinuity is that compiler state, bytecode state, graphics state, sparse-memory layout and runtime/model state are still mostly treated as separate operational surfaces.

The next structural step is not another larger logical extent. It is self-hosting closure: the whole machine representation becomes an explicit candidate-transform object while authority remains verification-gated.

## Decision

Jarvis-X adopts the 3D Self-Hosting Inward Runtime specification.

The complete recursive state is conceptually

\[
S_t =
[X_t,C_t,A_t,B_t,Z_t,G_t,M_t,\Omega_t,\Theta_t,\Pi_t,R_t,L_t].
\]

The canonical cycle is

~~~text
Observe
 -> Encode
 -> Spatialize
 -> Compile
 -> Execute
 -> Render
 -> Measure
 -> Contrast
 -> Optimize
 -> Verify
 -> COMMIT or ROLLBACK
 -> Re-encode
~~~

The system MAY generate candidates that alter its own source representation, compiler/intermediate form, bytecode, model parameters, runtime policy, memory layout or graphics projection.

It MUST NOT bypass the ADR-016 promotion law.

The architectural invariant is:

~~~text
SELF-HOSTING != SELF-AUTHORIZING
~~~

## Authority rule

Every proposed self-change is provisional.

A multi-domain candidate MUST either commit all touched authority namespaces atomically after verification or restore all touched namespaces.

Compiler/runtime self-reference does not create a privileged path around verification, policy, resource ceilings or audit.

## Measurement rule

Architecture targets, virtual extents and symbolic rates are not measurements.

Self-optimization decisions SHOULD increasingly use measured receipts from actual execution backends.

Performance claims require measured evidence and must name workload, hardware/backend, precision, resident working set, elapsed-time protocol, and uncertainty/repetition where relevant.

## Rendering rule

The 3D Cyber-IDE and future renderers are projections of authoritative state.

Pixel/object picking may map rendered output back to source/runtime identifiers, but graphics state does not become an alternate authority namespace.

User interaction that changes operational state enters the same candidate/verify/commit pipeline.

## Compiler rule

A candidate self-compiler must execute in shadow form until the repository-defined checks for that candidate have passed.

A successful parse or build is insufficient by itself. The relevant transaction may require tests, replay/integrity checks, resource ceilings, policy checks and external evidence.

## Distributed rule

Distributed reconciliation is a future adapter of the same contract.

No implementation may claim consensus, Byzantine tolerance or global agreement without an explicit protocol, fault model and tests.

## Consequences

### Positive

- closes the remaining conceptual boundary between code, bytecode, runtime, geometry, memory and adaptation;
- preserves one authority law across self-hosting and ordinary execution;
- provides a staged route from browser emulation to measured hardware backends;
- turns compiler/runtime adaptation into an auditable transaction rather than unrestricted mutation;
- gives future distributed work a defined reconciliation boundary.

### Costs

- implementations must retain more explicit stage receipts;
- self-hosting candidates require shadow builds/execution rather than direct replacement;
- rollback must cover every touched authority namespace;
- performance optimization becomes evidence-heavy;
- some existing surfaces remain specifications/demonstrations until adapters are built.

## Non-claims

Acceptance of this ADR does not mean the repository currently implements an autonomous self-recompiling compiler, unrestricted self-modification, physical symbolic throughput targets, distributed consensus, AGI, or consciousness.

## Validation target

A future implementation claiming ADR-031 conformance should demonstrate:

1. typed whole-machine candidate state;
2. source/AST/bytecode lineage;
3. shadow compile and execution;
4. bounded resource accounting;
5. deterministic or explicitly stochastic replay semantics;
6. measured execution receipts;
7. applicable correctness/policy/integrity gates;
8. complete rollback across all touched domains;
9. atomic promotion;
10. re-encoding of the newly authoritative state for the next recursive cycle.

## Provenance

This ADR belongs to the Dr Moagi family and inherits ADR-015 and docs/attribution/PERMEATION_MANIFEST.md.
