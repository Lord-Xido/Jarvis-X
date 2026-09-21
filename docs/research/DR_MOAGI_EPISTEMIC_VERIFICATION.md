# Dr Moagi 3D Codex — Epistemic Verification Plane

**Status:** Proposed research specification  
**Architecture decision:** `docs/adr/0011-dr-moagi-epistemic-admission-gate.md`

## 1. Objective

The Dr Moagi 3D Codex already constrains numerical state through bounded recursion, smoothing, resource ceilings and `Pi_Lambda`. The epistemic verification plane adds a separate question:

```text
Is the candidate state supported by evidence strongly enough to become authoritative?
```

Numerical admissibility and epistemic admissibility are distinct:

```text
bounded != true
converged != true
self-consistent != externally verified
```

The reference implementation is `src/jarvisx/dr_moagi_epistemic.py`.

---

## 2. State classes

The system carries three logical classes of state:

```text
O_t : externally sourced observation
H_t : generated hypothesis
B_t : admitted / committed research state
```

The prohibited shortcut is:

```text
H_t -> O_(t+1)
```

unless a separate trusted observation adapter supplies independent provenance. A caller cannot legitimately make generated output authoritative merely by feeding it back into the encoder.

---

## 3. Epistemic transaction

One guarded cycle is:

```text
1. Receive ObservationPacket O_t
2. Validate provenance class and confidence
3. Freeze immutable run anchor A_0 if this is the first cycle
4. Execute bounded Dr Moagi Codex as a candidate transform
5. Label decoded result H_(t+1), not observation
6. Compare H_(t+1) with O_t
7. Compare H_(t+1) with A_0 when anchor drift checking is enabled
8. Validate independent EvidencePacket records E_1...E_n
9. Measure hypothesis/evidence disagreement
10. Count independent evidence groups
11. Produce EpistemicVerdict
12a. If admitted: commit scene, parameter update and permeation release
12b. If rejected: quarantine hypothesis; preserve parameters; release no authoritative field
```

The outer transition is therefore:

```text
H_(t+1) = G_DrMoagi(O_t)

V_t = Verify(H_(t+1), O_t, A_0, E_1...E_n)

B_(t+1) = H_(t+1)   if V_t = admit
B_(t+1) = B_t       otherwise
```

---

## 4. Provenance classes

Reference provenance classes are:

```text
sensor
instrument
retrieval
user
model
simulation
```

Default admissible external classes are:

```text
sensor
instrument
retrieval
user
```

`model` is explicitly excluded from the external evidence set. `simulation` is also excluded by default because a simulation is generated evidence unless a deployment deliberately establishes a separate validated simulation contract.

Every observation and evidence packet carries:

```text
source_id
kind
confidence
derived_from_hypothesis
```

Evidence additionally carries:

```text
independence_key
```

which identifies the underlying dependency group.

---

## 5. Immutable anchor

At the beginning of a run:

```text
A_0 = O_0
```

The reference layer stores both the normalized sparse field and a deterministic SHA-256 digest. The anchor cannot be replaced until an explicit `reset_run()`.

The anchor is a drift detector, not an assumption that the world is static. Dynamic applications may configure the anchor threshold separately or disable it while still requiring fresh observation/evidence agreement.

---

## 6. Reference disagreement metric

For scalar same-support fields, the reference implementation uses normalized root-mean-square error:

```text
NRMSE(H, R)
  = sqrt(mean((H-R)^2))
    / max(sqrt(mean(R^2)), epsilon)
```

The default gate checks:

```text
NRMSE(H, O_t) <= tau_observation
NRMSE(H, A_0) <= tau_anchor
NRMSE(H, E_i) <= tau_evidence
```

Exact support is required by default. A support mismatch therefore fails closed rather than being silently filled or cropped.

NRMSE is a deterministic reference metric. It does not claim to solve semantic truth verification for text, images, language, medicine, finance or other domains. Domain-specific verifiers should map into the same admission decision without weakening provenance requirements.

---

## 7. Evidence independence

A system can appear highly corroborated while repeatedly reading one underlying source. The gate therefore separates record count from independence count.

For evidence packets:

```text
independence_group(E_i) = E_i.independence_key or E_i.source_id
```

Then:

```text
K_independent = cardinality(unique independence groups)
```

Admission requires:

```text
K_independent >= K_min
```

Two API endpoints, cached copies or transformed records that depend on one source should use the same independence key.

---

## 8. Anti-self-confirmation rules

The following packets are inadmissible by default:

```text
kind == model
derived_from_hypothesis == true
confidence < confidence_min
source_id duplicated in the same evidence set
kind not in allowed_external_kinds
```

This prevents the common recursive failure:

```text
model generates H
H is stored as evidence
model reads stored H
stored H appears to confirm H
H becomes authoritative
```

The epistemic layer instead preserves the distinction between generation and observation.

---

## 9. Verified-only learning

The base Codex parameter update remains:

```text
Theta_candidate = Theta_t - eta_Theta * grad_Theta L
```

The epistemic wrapper changes commit semantics:

```text
Theta_(t+1) = Theta_candidate   if admitted
Theta_(t+1) = Theta_t           otherwise
```

The wrapper deliberately invokes the base Codex candidate execution with parameter updates disabled, then computes the parameter update only after admission.

Thus a rejected hypothesis cannot become a self-training sample through this interface.

---

## 10. Verified-only permeation

The candidate Codex may compute a pure source map and Green/Helmholtz field for research inspection, but the wrapper treats them as speculative until admission.

External consumers receive:

```text
released_source_charge = Q[H]   if admitted else {}
released_permeation     = Phi[H] if admitted else {}
```

This creates the ordering:

```text
generate -> verify -> commit -> permeate
```

rather than:

```text
generate -> permeate -> discover later that the premise was false
```

---

## 11. Revised Dr Moagi guarded equation

Let the bounded candidate operator be:

```text
H_(t+1)
  = D(
      Pi_Lambda[
        S_dt(
          R_inward(E(O_t))
          + P_t
          - K_epsilon * epsilon_theta
          - eta_Z * grad_Z L
        )
      ]
    )
```

Define the epistemic admission operator:

```text
A_epi(H; O, A_0, E_1...E_n)
  in {ADMIT, QUARANTINE}
```

Then the authoritative state is:

```text
Xi_(t+1) = H_(t+1)  if A_epi = ADMIT
Xi_(t+1) = Xi_t      if A_epi = QUARANTINE
```

Learning becomes:

```text
Theta_(t+1)
  = Theta_t - eta_Theta * grad_Theta L   if ADMIT
  = Theta_t                              if QUARANTINE
```

Permeation becomes:

```text
Phi_(t+1) = G_k * Q[Xi_(t+1)]  only after ADMIT
```

This makes epistemic verification a commit gate rather than another additive latent term.

---

## 12. Failure behavior

The gate fails closed on:

```text
model-generated observation
hypothesis-derived observation/evidence
missing independent evidence
low-confidence provenance
observation disagreement
anchor drift beyond threshold
evidence disagreement
support mismatch
duplicate evidence source_id
malformed or non-finite field data
```

A rejected hypothesis remains inspectable for debugging and research telemetry but is not committed.

---

## 13. Trust limitation

Software cannot determine from a string alone whether `source_id="camera-1"` actually came from a camera. The reference gate enforces a provenance contract but does not cryptographically authenticate origin.

Production hardening should bind packets to one or more of:

```text
signed device identity
mTLS-authenticated sensor adapter
attested execution environment
signed retrieval result / source manifest
append-only provenance ledger
content digest plus source signature
trusted timestamp
```

The epistemic gate should reject unsigned or unverifiable provenance when the deployment's threat model requires it.

---

## 14. Recommended production extension

For multimodal or semantic systems, replace scalar NRMSE-only verification with a verifier ensemble while preserving the same commit rule:

```text
V = {
  geometric residual,
  sensor consistency,
  retrieval/source agreement,
  temporal consistency,
  calibrated uncertainty,
  domain-specific constraints,
  provenance integrity
}
```

No single learned verifier should be allowed to mark its own generator output as independently verified.

---

## 15. Conformance tests

Reference tests live at:

`tests/test_dr_moagi_epistemic.py`

They cover:

```text
model observation rejection
missing-evidence quarantine
observation/evidence conflict rejection
successful verified commit
verified-only learning
verified-only permeation release
independence-group counting
hypothesis-derived evidence rejection
immutable anchor behavior
deterministic digest behavior
support mismatch failure
```

---

## 16. Operational summary

The hardened Codex loop is:

```text
SENSE (external)
  -> ENCODE
  -> RECURSE
  -> OPTIMIZE CANDIDATE
  -> PROJECT
  -> DECODE HYPOTHESIS
  -> VERIFY AGAINST OBSERVATION + ANCHOR + INDEPENDENT EVIDENCE
     -> reject: quarantine / no learning / no permeation
     -> admit: commit / learn / permeate
  -> next externally observed cycle
```

The central invariant is:

```text
CONVERGENCE IS NOT TRUTH.
VERIFICATION PRECEDES AUTHORITY.
```
