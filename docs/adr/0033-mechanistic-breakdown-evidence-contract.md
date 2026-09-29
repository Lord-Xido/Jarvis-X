# ADR-033: Mechanistic Breakdown Evidence Contract

**Status:** Accepted  
**Date:** 2026-09-29  
**Applies to:** Dr Moagi Engine mechanistic descriptions, Cognitive Matrix geometry, inward-fold telemetry, compression/fidelity claims and future self-hosting implementations  
**Extends:** ADR-016, ADR-017, ADR-031, ADR-032  
**Canonical reconciliation:** `docs/DR_MOAGI_ENGINE_MECHANISTIC_RECONCILIATION.md`

## Context

The Dr Moagi Engine mechanistic source describes a very large symbolic input space, a three-axis volumetric projection, golden-angle phase alignment, quantization, an inward rotating collapse, recursive refinement and exact reconstruction.

The source is valuable as design provenance but mixes mathematical definitions, visualization units, target telemetry and empirical claims.

Without an explicit reconciliation contract, later implementations could silently convert symbolic scale or illustrative values into unsupported performance, compression or fidelity claims.

## Decision

Jarvis-X adopts the mechanistic reconciliation document and requires the following categories to remain distinct:

```text
symbolic / virtual extent
resident materialized state
configured target
simulated telemetry
measured telemetry
mathematically proved property
empirically verified property
```

A value may move from one category to another only with an explicit derivation or reproducible receipt.

## Storage and geometry rule

The expression `1GB x 1GB x 1GB` is historical geometric notation unless a file explicitly defines bytes and allocation. If each axis denotes `10^9` logical coordinates, the Cartesian space has `10^27` logical positions and is not a one-gigabyte resident object.

The normalized `1000 m` cube is visualization geometry and does not imply a physical correspondence between metres and bytes.

## Information-preservation rule

Projection into a smaller representation is not automatically lossless. Any exact codec claim must account for latent state, residual/side information and metadata required for deterministic inversion, and must verify exact output equality for byte-level claims.

## Entropy rule

Low Shannon entropy is not equivalent to low information loss. Entropy, reconstruction distortion, residual size and byte fidelity must be reported independently.

The source value `0.00042 bits` remains target/example telemetry until a reproducible experiment establishes it under a declared estimator.

## Inward-fold rule

For the source equation `r(t) = (1 - 0.88 * alpha(t)) * R_rot(t) * r(0)`, with norm-preserving rotation and `alpha in [0,1]`, peak contraction leaves `0.12 * ||r(0)||`.

Therefore the source equation is interpreted as bounded inward contraction to a finite core, not a literal point singularity. A logarithmic-spiral or asymptotic-zero claim requires an additional declared radial/angular law.

## Performance-telemetry rule

The source values `4.6 Hz`, `+14.82% per cycle`, `0.85 tau`, `0.00042 bits` and `100.00000%` are not repository measurements merely because they occur in the source.

Measured promotion requires a named metric, workload, backend, measurement protocol and receipt.

## Fidelity rule

`100% byte fidelity` requires exact decoded-byte equality. A suitable receipt includes lengths, residual/metadata accounting, input and decoded digests and a boolean exact-equality result.

Floating-point closeness is a separate receipt and is not byte equality.

## Authority rule

Mechanistic geometry remains subordinate to the existing transaction boundary:

```text
candidate
  -> mechanistic / Cognitive Matrix receipts
  -> CTR
  -> Pi_Lambda
  -> COMMIT or ROLLBACK
```

No visual collapse, entropy value, precision metric or claimed inverse transform may directly authorize state.

## Consequences

This ADR preserves the original abstraction while preventing unsupported conversion of metaphor, symbolic extent and target telemetry into implementation facts.

Future work may attempt to demonstrate the stronger source claims. They become canonical only when their required evidence exists.

## Provenance

This ADR belongs to the Dr Moagi family and inherits ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.
