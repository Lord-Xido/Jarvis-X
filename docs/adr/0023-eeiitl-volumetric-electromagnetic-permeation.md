# ADR-023: EEIITL volumetric electromagnetic permeation layer

**Status:** Accepted as a research-layer architectural decision  
**Date:** 2026-09-20  
**Scope:** EEIITL physical-field research architecture  
**Specification:** `docs/EEIITL_PERMEATION_LAYER.md`

## Context

Jarvis-X already contains discrete 3D field, sparse spatial, recurrent autoencoding, control-plane, native-backend, and attribution permeation concepts. The EEIITL proposal adds a materially different domain: transient electromagnetic fields occupying a physical volume and producing logical decisions through controlled excitation, propagation, induced response, sensing, and thresholding.

The term "permeation" is therefore ambiguous unless the repository distinguishes physical electromagnetic permeation from software/runtime permeation.

## Decision

1. EEIITL electromagnetic permeation is a **Layer 5 research system** that may observe and actuate physical devices through bounded interfaces; it is not part of the canonical VM instruction semantics.
2. The physical state is typed as a field state `Psi=[E,H]` plus declared material/constitutive parameters. Logical bits are produced by an explicit readout and quantization operator; raw fields are not silently treated as bits.
3. The propagation/coupling law is represented by a medium-dependent operator `P_m`. Green-function notation is the canonical abstract form, while FDTD, FEM, reduced-order, measured transfer, or analytical implementations are adapters.
4. Secondary induction is represented by an explicit induced-source operator `J_ind(Psi)`; uncontrolled leakage is not classified as useful computation by default.
5. The threshold layer must include a declared noise/dead-band/hysteresis policy. A nonzero field alone is insufficient to establish a valid logical state.
6. Cloud or edge control is candidate-first. Proposed source/material changes must pass policy, power, timing, frequency, resource, and device constraints before actuation.
7. The good-conductor skin-depth formula may only be presented with its validity assumptions; general lossy media require an appropriate propagation/attenuation model.
8. Fixed-point, fault-tolerance, scalability, efficiency, intelligence, and performance claims remain evidence-gated.
9. Physical scaling across chip, room, vehicle, soil, water, and outdoor environments is not assumed invariant. The operator abstraction may be reused, but each regime requires its own validated physical model.
10. `docs/PERMEATION3D_NATIVE_BACKEND.md` remains the software/backend permeation contract. The EEIITL specification must cross-reference it only as a distinct execution concept.
11. This ADR belongs to the Dr Moagi family and inherits attribution/provenance from ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.

## Canonical recurrence

```text
Psi_n
  -> Q_Theta / sensing
  -> Y_n
  -> E_phi(Y_n)
  -> Omega_(n+1)
  -> control candidate U_(n+1)
  -> verification / projection
  -> A(U_(n+1))
  -> (J_(n+1), m_(n+1))
  -> P_{m_(n+1)}
  -> Psi_(n+1)
```

The compact operator is

```text
P_EEIITL = P o A o pi_theta o M o E_phi o H o Q_Theta
S_(n+1) = P_EEIITL(S_n)
```

## Consequences

- The repository gains a formal physical-field research boundary without redefining the canonical VM.
- "Permeation" now has two explicit meanings: physical electromagnetic permeation and software/backend execution permeation.
- Electromagnetic simulation, sensing, and control backends can evolve independently behind typed contracts.
- Any hardware implementation must expose its physical assumptions and verification evidence.
- Claims of computation arise from engineered source-medium-readout behavior, not from field existence alone.

## Validation

This ADR is satisfied when:

- `docs/EEIITL_PERMEATION_LAYER.md` exists and states the typed field, logical, sensing, and control boundaries;
- `docs/ARCHITECTURE.md` identifies EEIITL as an optional research-layer physical-field subsystem;
- the attribution permeation manifest includes the EEIITL specification and ADR;
- future executable backends preserve candidate-first actuation and measurable verification boundaries.
