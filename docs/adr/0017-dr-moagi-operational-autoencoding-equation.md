# ADR-017: Canonical Dr Moagi operational auto-encoding/decoding equation

**Status:** Accepted  
**Date:** 2026-09-13  
**Applies to:** Dr Moagi autoencoders, sparse/volumetric runtimes, multimodal loops, DM-DD, fixed-point engines, runtime/meta-optimizers and future backend adapters  
**Extends:** ADR-003, ADR-004, ADR-015 and ADR-016

## Context

ADR-016 closed the Jarvis-X architecture around one typed system state, one candidate-first transaction law, explicit internal/external verification, named geometry profiles and interchangeable backends. The remaining ambiguity was the end-to-end mathematical law connecting encoding, hierarchical compaction, residual preservation, fixed-point inference, decoding, evidence generation and staged adaptation.

Several repository subsystems already implement pieces of this loop, but they use local notation and different degrees of hierarchy. Without a single operator contract, compatible implementations can appear to be competing architectures.

## Decision

Jarvis-X adopts `docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md` as the canonical research specification for the Dr Moagi auto-encoding/decoding system-of-operations law.

The candidate-generation operator is

\[
S^{\rm cand}_{t+1}
=
\left[
\mathcal U_{\Omega,\Theta,\Pi_{\rm run}}
\circ
\mathcal R_{\rm CTR}
\circ
\mathcal D_{\mathcal R}
\circ
\operatorname{Fix}_{F_\Theta}
\circ
\Phi_{\rm fusion}
\circ
\mathcal C_{\exp}
\circ
\mathcal E
\right](S_t,U_{t+1}).
\]

Authoritative promotion remains exclusively governed by the ADR-016 transaction law:

\[
S_{t+1}
=
V_t\,\Pi_\Lambda(S^{\rm cand}_{t+1})+(1-V_t)S_t,
\]

interpreted structurally.

Therefore the equation does not bypass or replace the candidate/verify/commit boundary.

## Required semantics

Implementations claiming conformance to this equation SHALL preserve the following distinctions:

1. `Omega_mem` is adaptive/temporal memory; audit/journal state is separate.
2. `Theta_model` is model state.
3. `Pi_runtime` is bounded execution policy.
4. `Pi_Lambda` is the admissibility/projection boundary and is not interchangeable with `Pi_runtime`.
5. `A_arch` is slower orchestration/architecture policy.
6. Hierarchical compaction may reduce active representation, but information discarded from a coarse representation must either be explicitly accepted as lossy or retained through residual/side information.
7. A low-dimensional shared latent state is not by itself evidence of lossless multimodal representation.
8. Fixed-point convergence is local inference evidence, not proof of global system equilibrium or external correctness.
9. External/world correspondence remains a separate verification gate when the runtime makes claims about external reality.
10. Adaptive changes are staged until the enclosing transaction commits.

## Canonical processing order

The preferred processing law is

```text
Typed input/event
 -> encode
 -> exponential/multiresolution compaction
 -> residual preservation
 -> shared/modal fusion
 -> bounded fixed-point refinement
 -> selective residual-aware decode
 -> contrast / CTR evidence
 -> stage Omega_mem
 -> stage Theta_model
 -> stage Pi_runtime
 -> verify / Pi_Lambda
 -> atomic commit OR rollback
 -> audit / recur
```

A backend may fuse, vectorize or accelerate adjacent stages, but its receipts must retain the semantic boundaries required for verification and rollback.

## Geometry neutrality

The equation is independent of any single logical extent. It may be instantiated by named ADR-016 profiles such as `BillionField1000`, `InwardCore1M`, `VolumetricROM20`, or future profiles, provided they declare addressing, active-support limits, contraction hierarchy, topology and resource accounting.

Logical extent is not resident memory or measured throughput.

## Adaptation hierarchy

The equation participates in the existing four time scales:

```text
t : authoritative state evolution
u : Omega_mem / Theta_model adaptation
n : Pi_runtime optimization
k : A_arch evolution
```

Slower layers may optimize cadence, budgets and bounded execution choices, but they may not remove mandatory verification or transaction stages.

## Consequences

### Positive

- gives the Dr Moagi family one end-to-end autoencoding/decoding systems equation;
- makes exponential compaction and residual preservation explicit rather than implicit;
- integrates CTR/evidence generation directly into the processing law;
- removes ambiguity between runtime optimization and admissibility projection;
- preserves ADR-016 typed-state and rollback semantics;
- allows sparse Python, C++, GPU and hardware implementations to target one mathematical contract;
- provides a clean target for future standard stage receipts and empirical benchmarking.

### Costs

- existing runtimes are not automatically conformant merely because they implement an encode/decode loop;
- adapters must expose state-type, residual, convergence, resource and verification receipts;
- some local symbol names will need migration as code is touched;
- empirical claims about compaction efficiency, convergence or speedup require separate benchmarks.

## Validation

A conforming implementation should be able to demonstrate, for its declared geometry/backend profile:

1. typed ingest and state mapping;
2. deterministic or explicitly stochastic encoding contract;
3. declared compaction hierarchy and active-support bound;
4. residual/side-information accounting or an explicit lossy boundary;
5. bounded fixed-point termination and residual metric;
6. decode/reconstruction receipt;
7. contrast/evidence receipt;
8. staged memory/model/runtime-policy changes;
9. verification decision;
10. atomic commit or complete rollback across every touched authority namespace.

The specification itself is not evidence that every existing backend already satisfies these requirements.

## Provenance

This ADR belongs to the Dr Moagi family and inherits the canonical attribution and provenance policy defined by ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.