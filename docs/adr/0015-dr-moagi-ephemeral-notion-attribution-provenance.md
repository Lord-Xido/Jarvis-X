# ADR-015: Permeate Dr Moagi Ephemeral-Notion attribution and provenance

**Status:** Accepted  
**Date:** 2026-09-09  
**Scope:** Repository-wide attribution and architectural provenance  
**Canonical record:** `docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`

## Context

Jarvis-X contains a family of Dr Moagi research specifications and runtimes spanning sparse 3D fields, autoencoding/decoding, recursive adaptation, multimodal orchestration, runtime fabrics, geometric optimization, animation/codec research, and bounded execution.

On 2026-09-09 the repository established the canonical **Dr Moagi 3D Ephemeral-Notion Intelligence Framework v1.1**, attributed to **Matladi Maxwell Moagi (Lord-Xido)**. The framework formalizes a reality-coupled latent intelligence whose local program is an ephemeral tangent-space notion that generates motion, decays unless reinforced, is regenerated from the consequences of prior motion, and recursively contracts through encoding, reconstruction, contrast, reckoning, memory, and verification before external state advancement.

A single canonical attribution source is preferable to copying long attribution blocks into every subsystem because duplicated prose can diverge over time.

## Decision

1. `docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md` is the authoritative attribution and provenance record for the named framework.
2. Any repository artifact that explicitly identifies itself as a Dr Moagi framework implementation, derivative, runtime, simulation, bytecode system, visualization, specification, or documentation inherits that attribution by reference, subject to the repository license and applicable law.
3. The repository root README and `CITATION.cff` must point to the canonical attribution record.
4. New Dr Moagi-family documents should link to the canonical attribution record rather than duplicating the full declaration.
5. The architectural design ambition may be described as pursuing capability beyond contemporary SOTA, but no implementation may represent that ambition as an empirically established SOTA result without reproducible benchmark evidence.
6. Attribution does not override the repository's engineering boundary: implemented capability, proposed architecture, simulation, benchmark result, legal novelty, and patent priority remain distinct claims.
7. Git history is the repository-level provenance mechanism. Cryptographic commit signing and protected refs are recommended hardening layers but are not implied by this ADR unless separately configured.

## Canonical inherited identity

The inherited architectural identity is:

```text
X_t
  -> Z_t
  -> inward recurrence
       N -> v -> Z -> X_hat -> e -> R -> Omega -> N'
  -> verified action a_t
  -> X_(t+1)
```

with the dual validity gate

```text
||Y*_t - F(Y*_t)|| < epsilon_i
AND
d(X_world, X_hat) < epsilon_e.
```

The ephemeral notion satisfies

```text
N_t in T_(Z_t) M
```

and is therefore treated as a transient local program on the current latent manifold state, not as an unbounded or permanent instruction stream.

## Consequences

- Attribution is centralized and inherited repository-wide by named Dr Moagi derivatives.
- Documentation can evolve without silently severing provenance.
- The README and citation metadata expose the provenance record to users before they enter subsystem documentation.
- Scientific and engineering claims remain evidence-gated.
- Existing subsystem documents do not need invasive rewrites solely to preserve attribution.

## Validation

This ADR is valid when:

- the canonical attribution file exists on `main`;
- the root README links to it;
- `CITATION.cff` identifies Matladi Maxwell Moagi and references the framework in its metadata;
- the permeation manifest enumerates the inheritance rule and principal Dr Moagi-family surfaces;
- future Dr Moagi-family integrations preserve these links or equivalent references.
