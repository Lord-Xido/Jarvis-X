# ADR-0012: Dr Moagi master transaction runtime

**Status:** Proposed  
**Date:** 2026-08-17

## Context

Jarvis-X now has two mature but previously adjacent trust boundaries:

1. the Dr Moagi epistemic plane decides whether a generated hypothesis is sufficiently supported by external observation, immutable anchor and independent evidence;
2. `jarvisx.system_runtime` decides whether a bounded execution plan is capability-admissible, verifiable, audit-linked and committed as authoritative task/VM state.

Treating those as unrelated commit concepts leaves an integration ambiguity. A hypothesis may be epistemically admitted yet still fail capability projection, bytecode validation, VM verification or audit commit. Conversely, a deterministic VM can faithfully execute a program whose upstream premise was never epistemically verified.

The system therefore requires one outer transaction membrane with the law:

```text
observe
  -> generate hypothesis
  -> epistemic verify
  -> compile/build bounded authority plans
  -> capability projection
  -> isolated deterministic execution
  -> ledger verification
  -> system audit
  -> COMMIT
```

Neither inner gate may substitute for the other.

## Decision

Jarvis-X adds `src/jarvisx/dr_moagi_master_runtime.py` as a Layer-2/5 integration boundary. It does not replace the canonical VM, the epistemic Codex, or `SystemRuntime`; it composes them.

The outer transition is:

\[
H_{t+1}=G_{DM}(O_t),
\]

\[
V_t^{epi}=A_{epi}(H_{t+1};O_t,A_0,E_{1:n}),
\]

and only if `V_t^epi = ADMIT` may an explicit adapter construct bounded authority plans:

\[
P_{1:m}=C_{H\rightarrow VM}(H_{t+1}).
\]

The control plane then performs:

\[
\rho_t=
A_{sys}(P_{1:m};\Lambda_{cap},\Lambda_{res}),
\]

where `rho_t` is an audit-linked `ExecutionReceipt`.

The master commit rule is:

\[
Commit_{DM}
=
[V_t^{epi}=ADMIT]
\land
[\rho_t.committed=true].
\]

Authoritative research output is exposed only under this conjunction.

## Plan-builder boundary

There is intentionally no implicit conversion from a decoded 3D scene to canonical 64-bit Jarvis-X bytecode. A caller must provide an explicit `PlanBuilder`:

```text
EpistemicExecutionResult
    -> Iterable[PlanCandidate]
```

This adapter is the compiler/lowering boundary. It must declare how verified research state becomes a bounded executable plan and remains subordinate to capability, word-count, cycle and candidate budgets.

This prevents a symbolic 3D state from acquiring execution authority merely because it is representable.

## Rollback semantics

### Epistemic rejection

If the hypothesis fails epistemic admission:

```text
no plan construction
no VM execution
Theta_authoritative remains Theta_t
Q_authoritative = unreleased
Phi_authoritative = unreleased
scene_authoritative = unchanged / absent
```

### Authority rejection or failure

If epistemic admission succeeds but the downstream authority path fails:

```text
verified research candidate remains diagnostic only
Theta_authoritative remains Theta_t
Q_authoritative = unreleased
Phi_authoritative = unreleased
scene_authoritative = unchanged / absent
```

The inner epistemic result may contain speculative candidate data for diagnostics, but the master result exposes separate authoritative fields. Consumers must use the outer authoritative fields for state transition decisions.

### Dual admission

Only when both gates succeed:

```text
scene_authoritative = admitted scene
Theta_authoritative = verified candidate Theta
Q_authoritative = admitted source map
Phi_authoritative = admitted permeation field
execution_receipt = committed audit-linked receipt
```

## Core invariants

1. **Intelligence is not authority:** a generated or even epistemically admitted candidate cannot directly mutate canonical VM/task state.
2. **Execution is not truth:** deterministic execution cannot make an unverified premise factual.
3. **Dual admission:** authoritative Dr Moagi publication requires both epistemic admission and system commit.
4. **No implicit compiler:** research state reaches bytecode only through a named plan-builder/lowering adapter.
5. **Fail closed:** missing plans, compiler exceptions, capability failures, VM failures, ledger failures and audit failures suppress authoritative learning/output release.
6. **Parameter rollback:** an epistemically eligible `Theta` update remains non-authoritative if system execution does not commit.
7. **Permeation rollback:** speculative `Q/Phi` data remain non-authoritative until the outer commit succeeds.
8. **Bounded execution:** plan count, program words and VM cycles remain controlled by `ResourceBudget`.
9. **Receipt linkage:** authoritative execution is accompanied by the canonical `ExecutionReceipt`.
10. **No authority inversion:** Layer-5 research modules cannot bypass Layer-2/3 policy, execution verification or provenance.

## Relationship to existing ADRs

- **ADR-001:** preserves the deterministic core/research-layer separation.
- **ADR-002/003/004/006:** provide codec, field, generative and kinetic candidate semantics.
- **ADR-007:** remains the canonical task-level authority and audit control plane.
- **ADR-009:** native/swarm VRAM remains a research execution substrate and cannot bypass this boundary.
- **ADR-0010:** defines the typed same-space candidate recurrence.
- **ADR-0011:** defines epistemic admission and anti-self-confirmation rules.

ADR-0012 composes ADR-0011 and ADR-007; it does not weaken either.

## Trust limitations

This runtime does not solve provenance authenticity by itself. `ObservationPacket` and `EvidencePacket` still require authenticated/signed adapters for deployments where source spoofing is in the threat model.

The current canonical `SystemRuntime` is also in-process and its committed state is not yet a durable distributed transaction store. Production deployment requires process isolation, durable state, recovery semantics and explicit external side-effect adapters.

## Validation

`tests/test_dr_moagi_master_runtime.py` must demonstrate:

- epistemic rejection never reaches the authority plane;
- authority rejection after epistemic admission suppresses authoritative learning and permeation;
- dual admission exposes authoritative scene, parameter update and permeation;
- empty plan sets fail closed;
- compiler/plan-builder exceptions fail closed;
- the outer result retains an audit-linked execution receipt when committed.

## Canonical system law

The resulting Jarvis-X master loop is:

```text
REALITY / OBSERVATION
  -> bounded research transform
  -> HYPOTHESIS
  -> EPISTEMIC VERIFY
       -> reject: quarantine
       -> admit: verified candidate
  -> explicit lowering / PlanBuilder
  -> CAPABILITY + RESOURCE PROJECTION
  -> ISOLATED VM EXECUTION
  -> LEDGER VERIFY
  -> SYSTEM AUDIT
       -> reject/fail: rollback authority
       -> commit: authoritative state
  -> LEARN / PERMEATE / RENDER AS AUTHORITATIVE
  -> RE-OBSERVE
```

In compact form:

\[
\boxed{
\text{Observe}
\rightarrow
\text{Hypothesize}
\rightarrow
\text{Verify}
\rightarrow
\text{Authorize}
\rightarrow
\text{Execute}
\rightarrow
\text{Audit}
\rightarrow
\text{Commit}
}
\]
