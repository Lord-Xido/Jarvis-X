# ADR-006: Adopt M³-ACME as the bounded web-bit manifold adapter

**Status:** Accepted  
**Date:** 2026-08-13

## Context

Jarvis-X already accepts the Dr Moagi 3D adaptive codec-runtime as a Layer 4/5 research architecture under ADR-002. The M³-ACME equation adds a specific data-facing interpretation in which an observed information field is organized along structural, semantic, provenance and temporal coordinates, masked by an explicit compliance/permissions operator before encoding.

The equation is useful only if the implementation preserves the repository's existing authority boundaries. A mathematical compliance term is not evidence of permission to crawl or process a source, and a semantic abstraction stage is not license to invent metadata that the source did not provide.

## Decision

Jarvis-X adopts M³-ACME as a bounded research adapter with the operational identity:

```text
X --C--> X_c --E_Theta--> Z --A--> Z* --D_Phi--> X_hat --L_Moagi--> receipt
```

The reference implementation is `src/jarvisx/m3_acme.py`.

Its deterministic reference encoder is canonical JSON plus zlib compression. This is intentionally not presented as a learned neural encoder. It supplies an executable, exactly reversible contract against which later learned encoders/decoders can be validated.

The four information coordinates are represented as:

```text
x := structure metadata
y := semantic metadata
z := provenance + trust + permission basis
t := timezone-qualified observation time
```

The runtime does **not** perform web crawling. Upstream ingestion must supply explicit authorization assertions. The compliance gate fails closed if authorization, robots/terms assertions, permission basis, provenance URL, timestamp or trust bounds are invalid.

The abstraction operator passes through supported source-provided semantic objects:

```text
entities, relations, topics, sentiment, confidence
```

It does not synthesize missing entities, relations or confidence values in the deterministic reference implementation.

The composite loss is reported as separate components:

```text
L = L_reconstruction
  + lambda_s L_semantic
  + lambda_p L_provenance
  + lambda_t L_temporal
  + lambda_c L_compliance
  + lambda_g L_graph
```

No component may hide failure in another component. In particular, compliance rejection happens before the codec transaction.

## Required invariants

1. **Fail-closed compliance:** denied or incomplete permission assertions never enter the encoder.
2. **Provenance preservation:** URL, observation time, trust and permission basis survive the admitted round trip.
3. **Deterministic reference codec:** identical admitted records produce identical canonical bytes, digests and compressed payloads.
4. **Integrity-bound decode:** corrupt or version-incompatible latent packets are rejected.
5. **No semantic fabrication:** the reference abstraction stage emits only supported source-provided metadata.
6. **Bounded resources:** record size is capped before and after compression/decompression boundaries.
7. **Observable loss:** reconstruction, semantic, provenance, temporal, compliance and graph losses are individually emitted.
8. **No crawler equivalence:** this runtime does not itself establish authorization, robots compliance, terms compliance, privacy compliance or jurisdictional legality for external sources.
9. **ADR-002 compatibility:** learned replacements remain Layer 4/5 research components and do not become authoritative deterministic-core state merely by existing.

## Consequences

M³-ACME now has a runnable reference semantics suitable for fixtures, tests and adapter development. Exact reversible compression provides a baseline where reconstruction error is falsifiable. Production semantic extraction, learned compression, graph construction and authorized ingestion remain separable upstream/downstream concerns and must declare their own evidence and security boundaries.
