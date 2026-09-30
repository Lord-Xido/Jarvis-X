# ADR-0011: Add an epistemic admission gate to the Dr Moagi recursive Codex

**Status:** Proposed  
**Date:** 2026-08-17

## Context

The Dr Moagi 3D Codex is numerically bounded by fixed-point iteration, smoothing, resource ceilings and `Pi_Lambda`. Those properties constrain numerical behavior but do not establish factual correctness. A recursively decoded hypothesis can be internally stable, close to a fixed point, and still disagree with external reality.

The highest-risk failure mode is self-confirmation:

```text
observation
  -> encode
  -> recurse
  -> decode hypothesis
  -> treat hypothesis as next observation
  -> learn from it
  -> permeate it
  -> repeat
```

In that loop, local reconstruction consistency can improve while anchor drift and factual error increase. A stable attractor is therefore not sufficient evidence of truth.

## Decision

Jarvis-X introduces an explicit Layer-5 epistemic transaction boundary around the Dr Moagi Codex.

The constitutional invariant is:

```text
A hypothesis may recurse, but it cannot become an authoritative observation
without independent external verification.
```

The authoritative transition becomes:

```text
external observation O_t
  -> candidate Codex execution
  -> hypothesis H_(t+1)
  -> epistemic verification V(H_(t+1), O_t, A_0, E_1...E_n)
  -> admit or quarantine
```

where:

- `O_t` is a provenance-labelled external observation;
- `A_0` is an immutable run anchor;
- `H_(t+1)` is generated state and remains a hypothesis until admitted;
- `E_i` are independently identified evidence packets;
- `V` is a fail-closed evidence gate.

### Promotion rule

A candidate may be committed only if all configured tests pass:

```text
NRMSE(H, O_t) <= tau_observation
NRMSE(H, A_0) <= tau_anchor       # when anchor checking is enabled
NRMSE(H, E_i) <= tau_evidence     # for every admitted evidence source
independent_evidence_count >= K
source_confidence >= tau_confidence
source kind is externally admissible
source is not derived from the current hypothesis
support contract is satisfied
```

The reference implementation is:

`src/jarvisx/dr_moagi_epistemic.py`

### State separation

The system distinguishes:

```text
O = observations from admissible external provenance
H = generated hypotheses
B = admitted beliefs / committed research state
```

There is no implicit transition:

```text
H -> O
```

A model-generated or hypothesis-derived packet is rejected at the observation boundary by default.

### Verified-only learning

Parameter updates are candidate data until epistemic admission.

```text
if admitted:
    Theta_(t+1) = Theta_t - eta_Theta * grad_Theta L
else:
    Theta_(t+1) = Theta_t
```

Thus self-generated output cannot directly train the active parameter state through this boundary.

### Verified-only permeation release

The base Codex may compute a candidate permeation field as a pure research calculation, but the epistemic wrapper exposes authoritative `source_charge` and `permeation_field` only after admission.

```text
if admitted:
    release Q[H]
    release Phi[H]
else:
    quarantine H
    release no authoritative Q or Phi
```

### Evidence independence

Multiple records from the same underlying source do not count as multiple independent confirmations. `independence_key` groups correlated evidence. Duplicate `source_id` values fail closed.

Model-generated evidence is never counted as independent external evidence in the default policy.

## Security and trust boundary

The gate validates provenance metadata but cannot prove that an untrusted caller labelled it truthfully. Production deployments that depend on strong provenance must bind evidence packets to authenticated adapters, device identities, signed records, trusted retrieval systems, or another auditable source-of-origin mechanism.

The epistemic gate is therefore necessary but not sufficient for cryptographic provenance.

## Consequences

### Positive

- stable hallucinations cannot become committed state merely because recursion converged;
- generated outputs cannot silently become new observations;
- missing or correlated evidence fails closed;
- rejected hypotheses cannot commit parameter learning;
- rejected hypotheses do not release authoritative permeation;
- immutable anchor drift remains visible;
- evidence provenance and error metrics become auditable telemetry.

### Negative

- stricter admission increases false rejections when external evidence is sparse or noisy;
- dynamic scenes may require disabling or adapting the immutable-anchor drift threshold;
- independent evidence sources add latency and integration cost;
- scalar-field NRMSE is a reference metric, not a universal semantic-truth metric;
- provenance integrity still depends on trusted adapters in production.

## Required invariants

1. **Observation provenance:** authoritative observations use admissible external provenance.
2. **No hypothesis promotion by relabelling:** model/hypothesis-derived packets are rejected by default.
3. **Immutable run anchor:** the anchor cannot change until an explicit run reset.
4. **Independent evidence:** correlated records share one independence key and count once.
5. **Fail closed:** missing evidence, support mismatch, excess error, duplicate source IDs or inadmissible provenance prevent commit.
6. **Verified-only learning:** rejected candidates leave active parameters unchanged.
7. **Verified-only permeation:** rejected candidates expose no authoritative source or permeation field.
8. **Candidate visibility:** rejected outputs may remain inspectable as hypotheses for debugging but are not authoritative state.
9. **Separation from VM authority:** this Layer-5 gate does not mutate the deterministic Jarvis-X VM.
10. **Auditability:** admission reasons and evidence digests are retained in the result.

## Validation

The focused conformance suite must cover:

- rejection of model-generated observations before candidate execution;
- quarantine when independent evidence is missing;
- rejection of observation/evidence disagreement;
- successful commit when independent evidence agrees;
- verified-only parameter learning;
- verified-only permeation release;
- correlated-evidence deduplication;
- rejection of hypothesis-derived evidence;
- immutable anchor behavior;
- deterministic scene digests and fail-closed support mismatch.

## Research specification

See:

`docs/research/DR_MOAGI_EPISTEMIC_VERIFICATION.md`
