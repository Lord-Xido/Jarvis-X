# ADR-032: Bounded Cognitive Matrix permeation and stochastic active support

**Status:** Accepted  
**Date:** 2026-09-29  
**Applies to:** 3D Cognitive Matrix visualization, cognitive-field geometry, stochastic active-set projections, permeation telemetry, CTR evidence and self-hosting geometry adapters  
**Extends:** ADR-016, ADR-017, ADR-028, ADR-031  
**Canonical reference:** `docs/DR_MOAGI_COGNITIVE_MATRIX_PERMEATION.md`

## Context

Jarvis-X already contains a 3D Cognitive Matrix WebGL demonstration and a non-authoritative cognitive-field geometry verifier. The browser artifact explicitly labels its MSE, latency and cloud indicators as synthetic or visual.

The newer Cognitive Matrix formulation adds four concepts that require an executable boundary: an astronomically large logical 3D domain, a finite sampled active representation, a diffusion-style permeation field coupled to particles, and entropy/boundary-flux receipts.

Without a bounded numerical reference, those concepts would remain visual metaphors or could be misread as physical-memory and physical-field claims.

## Decision

Jarvis-X adopts a bounded Cognitive Matrix permeation reference:

- `src/jarvisx/cognitive_matrix_permeation.py`
- `tests/test_cognitive_matrix_permeation.py`
- `docs/DR_MOAGI_COGNITIVE_MATRIX_PERMEATION.md`

The logical domain is metadata. Only the configured active particle ensemble and radial field are materialized.

## Virtual-domain rule

For logical domain V_N, the site count is N^3. The short-scale septillion example N = 10^24 therefore denotes 10^72 logical sites.

No implementation may infer resident memory or work proportional to that count unless it actually allocates or visits those sites.

## Numerical rule

The reference MUST remain bounded by explicit configuration for active particle count, radial field cells, time step, diffusion/source/decay parameters, geometric boundary radius, entropy-bin count and reconstruction threshold.

Malformed, non-finite or out-of-range inputs fail closed.

## Stochastic rule

Seeded execution MUST replay deterministically for the same configuration and step sequence.

The particle diffusion term MUST scale with sqrt(dt). A stochastic visualization or sampled estimate is not an exact enumeration of the virtual domain.

## Field and flux rule

The reference permeation field is a software-defined reaction-diffusion scalar field. Its current and boundary flux are computational quantities.

They MUST NOT be reported as physical electromagnetic, thermal, biological or other energy flux without an independently calibrated physical model.

A percentage HUD value MUST be normalized against an explicit configured reference magnitude.

## Entropy rule

The reference reports finite histogram Shannon entropy over sampled particle positions. It MUST NOT equate that value with thermodynamic entropy or assume that visual order implies low Shannon entropy.

## Autoencoding rule

The initial executable reference uses a same-width bounded nonlinear encode/decode pair so reconstruction and state flow are testable without adding an untrained compression claim.

A future VAE or dimensional bottleneck adapter must separately declare its training state, prior, KL term, residual information and rate-distortion evidence.

## Authority rule

The Cognitive Matrix reference emits evidence only:

```text
bounded candidate state
  -> Cognitive Matrix receipt
  -> optional cognitive geometry receipt
  -> CTR
  -> Pi_Lambda
  -> COMMIT or ROLLBACK
```

Neither WebGL geometry, particle positions, entropy nor low reconstruction error may directly mutate authoritative state.

## Relation to self-hosting closure

ADR-031 remains authoritative for whole-machine self-application. The Cognitive Matrix becomes one bounded spatial/numerical profile inside that cycle:

`Observe -> Encode -> Spatialize -> Execute/Render -> Measure -> Contrast -> Verify -> COMMIT/ROLLBACK -> Re-encode`

The invariant remains:

`SELF-HOSTING != SELF-AUTHORIZING`

## Validation

Conformance tests cover exact virtual-domain arithmetic without dense allocation, bounded active particle support, encode/decode reconstruction accounting, distinct input/latent/output geometry phases, finite PDE/flux/entropy receipts, seeded deterministic replay, histogram entropy sanity cases, and malformed resource/numeric inputs.

## Consequences

The reference grounds the Cognitive Matrix in a finite executable numerical model, preserves septillion-scale addressing as virtual metadata, supplies real entropy/flux/reconstruction receipts for future UI binding, fits the existing CTR / Pi_Lambda boundary, and bridges the WebGL demo, ADR-028 geometry receipts and ADR-031 self-hosting closure.

The radial PDE remains deliberately low-dimensional and is not a general 3D PDE solver. The initial encoder is same-width and is not a trained compressor. Browser telemetry remains synthetic until explicitly wired to the numerical receipt.

## Provenance

This ADR belongs to the Dr Moagi family and inherits ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.
