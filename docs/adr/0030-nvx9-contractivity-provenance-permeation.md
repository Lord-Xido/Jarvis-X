# ADR-030: NV-X9 Contractivity and Provenance Permeation

**Status:** Accepted  
**Date:** 2026-09-27  
**Canonical provenance:** `docs/attribution/NVX9_ORIGINALITY_PROVENANCE.md`  
**Marker:** `NVX9-PROVENANCE-LOCK-2026-09-27`

## Context

Jarvis-X now carries a project provenance record for the named **NV-X9 / Dr. Moagi’s Beast**
inward-looping contractive 3D neural virtual-machine architecture. The source technical
report defines a coupled Riemannian observation/latent/execution geometry, inward
gradient flow, time-dependent metric evolution, tangent-space residual correction and
the composite closed-loop operator

\[
S_{t+1}=F\!\left(P\!\left(D\!\left(A_\omega(E(S_t))\right)\right)\right).
\]

The repository needs one unambiguous rule for how that identity propagates through
future specifications and implementations without overstating novelty, implementation
maturity or empirical validation.

## Decision

1. `docs/attribution/NVX9_ORIGINALITY_PROVENANCE.md` is the canonical repository
   provenance record for NV-X9 / Dr. Moagi’s Beast.
2. Any repository artifact that explicitly claims to implement, derive from, benchmark,
   visualize or formalize **NV-X9** or **Dr. Moagi’s Beast** must reference that record
   or this ADR.
3. The canonical architectural sequence is:

```text
observation
  -> geometric encode
  -> contractive fixed-point refine
  -> geometric decode
  -> permeate
  -> inward fold
  -> tangent residual / geodesic correction
  -> recur
```

4. The project-level originality boundary applies to the particular named synthesis,
   operator composition, architectural interpretation and benchmark construction.
5. Established mathematics remains established prior art. In particular, this ADR does
   not claim invention of Riemannian manifolds, Christoffel symbols, gradient flow,
   exponential/logarithmic maps, Ricci flow, contraction mappings, autoencoders,
   residual correction or fixed-point theory.
6. Source-report benchmark values may be cited as report evidence only. They are not
   independent replication and do not establish production throughput, global
   stability, patentability, freedom to operate or scientific novelty.
7. NV-X9 remains a **specification/provenance surface** until an executable bounded
   implementation, tests and reproducible evidence are merged and classified by
   `docs/PROJECT_STATUS.md`.
8. Any future operational NV-X9 adapter must remain subordinate to the canonical
   candidate-first transaction and verification boundaries defined by ADR-016/017:
   proposed state changes are staged, evidenced, verified and then committed or
   rolled back.

## Contractivity boundary

A statement that an NV-X9 refinement is contractive must identify the actual map,
state domain, metric/norm and bound used. For a map (A_\omega), a sufficient
repository-level claim must be tied to a measurable Lipschitz or Jacobian condition,
for example

\[
\sup_z \|J_{A_\omega}(z)\| < 1,
\]

on the declared operating domain. A symbolic parameter inequality alone is not
automatically evidence that an arbitrary learned implementation is contractive.

## Consequences

- README, citation metadata, the permeation manifest and project status reference the
  same provenance boundary.
- Future NV-X9 artifacts inherit one canonical identity instead of introducing
  divergent originality language.
- Mathematical lineage and system-level contribution remain explicitly separable.
- Empirical and legal claims remain evidence-gated.

## Non-goals

This ADR does not establish patent priority, ownership as a matter of law,
non-infringement, publication priority, independent replication, AGI capability or
production readiness.
