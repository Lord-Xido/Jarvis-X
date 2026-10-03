# ADR-024: Active Multimodal Evidence Permeation

Status: **accepted**

Date: 2026-10-03

## Context

Jarvis-X already separates internal convergence from external correspondence and
already requires candidate-first verification before authoritative state
promotion. PR #360 added an active multimodal evidence runtime with temporal,
2D and 3D localization, provenance, confidence, inward latent refinement and
uncertainty-triggered re-retrieval.

The remaining architectural gap was that evidence retrieval could still be run
as a standalone subsystem. A caller could theoretically execute an operational
candidate without making evidence sufficiency part of the authority boundary.

## Decision

When an operation declares externally grounded evidence requirements, Jarvis-X
must treat multimodal evidence verification as a precondition for authoritative
operational promotion.

The canonical evidence-grounded flow is:

```text
query / world-state need
 -> retrieve candidate evidence
 -> localize time / 2D / 3D region
 -> preserve modality + confidence + provenance
 -> encode shared evidence state
 -> contract inward to bounded evidence fixed point
 -> CTR verification
 -> if uncertainty remains: target missing modality and retrieve again
 -> if evidence gate passes: run bounded operational candidate search
 -> operational verification
 -> COMMIT / ROLLBACK
 -> recur
```

The two authority gates are therefore distinct:

1. **Evidence gate** — asks whether the system has sufficient grounded evidence
   for the declared claim or action.
2. **Operational gate** — asks whether the proposed computational/runtime
   candidate is numerically valid, bounded and admissible.

The promotion law is:

```text
commit_allowed =
    evidence_gate.accepted
    AND operational_candidate.verification.valid
```

No amount of internal fixed-point convergence overrides missing required
evidence.

## Typed boundary

An evidence record and a media payload are not interchangeable.

For example:

```text
EvidenceSegment(
  modality="audio",
  time_span=(41.2, 52.8),
  provenance="audio-index",
  confidence=0.93
)
```

is evidence metadata and localized retrieval state. It does not itself contain
the underlying audio bytes unless a native adapter supplied them.

This preserves the invariant:

```text
metadata about media != media bytes
```

Native audio, video, image, point-cloud, code and sensor adapters remain
responsible for materializing the actual data required by downstream execution.

## Mathematical contract

Let E(q) be the active multimodal evidence acquisition operator and let U(E)
be its uncertainty. Let O(C) be the bounded operational candidate operator and
V(C) its verification result.

Evidence is admissible only when:

```text
required_modality_coverage(E) = 1
AND U(E) <= U_max
```

Operational promotion then requires:

```text
A_t =
    V(C_t)
    AND evidence_admissible(E_t)
```

If A_t is false, authoritative state is preserved.

The system therefore extends the canonical CTR recurrence to:

```text
Retrieve
 -> Localize
 -> Encode
 -> Fold Inward
 -> Generate
 -> Contrast
 -> Reckon
 -> Verify
 -> Re-acquire if uncertain
 -> Generate operational candidate
 -> Verify operational candidate
 -> Commit or rollback
 -> Recur
```

## Implementation

Reference implementation:

- `src/jarvisx/multimodal_evidence_runtime.py`
- `src/jarvisx/evidence_grounded_operational.py`
- `tests/test_multimodal_evidence_runtime.py`
- `tests/test_evidence_grounded_operational.py`
- `.github/workflows/multimodal-evidence-runtime.yml`

CLI surfaces:

- `jarvisx-multimodal-evidence`
- `jarvisx-evidence-grounded-operational`

## Consequences

Positive:

- retrieval becomes part of the authority path instead of only context
  augmentation;
- missing modalities remain explicit;
- metadata-only evidence cannot silently masquerade as direct media evidence;
- uncertainty can cause targeted re-acquisition instead of hallucinated
  completion;
- operational commits remain independently verified.

Tradeoffs:

- evidence-gated operations may halt when required media are unavailable;
- production quality still depends on native modality adapters and source
  quality;
- an accepted evidence bundle is not proof that third-party information is
  correct;
- this architecture does not by itself establish AGI.

## Invariant

```text
No authoritative externally grounded operation without sufficient evidence.
No authoritative state transition without operational verification.
```
